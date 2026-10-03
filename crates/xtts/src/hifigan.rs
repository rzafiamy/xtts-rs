//! HiFi-GAN decoder: GPT latents (one per 1024 samples at 22.05 kHz) →
//! 24 kHz waveform, conditioned on a 512-d speaker embedding added after the
//! input convolution and after every upsampling layer.

use crate::weights::Weights;
use anyhow::Result;
use candle_core::{DType, Device, Tensor};
use candle_nn::ops::leaky_relu;

const LRELU_SLOPE: f64 = 0.1;
const UPSAMPLE_RATES: [usize; 4] = [8, 8, 2, 2];
const UPSAMPLE_KERNELS: [usize; 4] = [16, 16, 4, 4];
const RES_KERNELS: [usize; 3] = [3, 7, 11];
const RES_DILATIONS: [usize; 3] = [1, 3, 5];

struct Conv {
    weight: Tensor,
    bias: Option<Tensor>,
    padding: usize,
    dilation: usize,
}

impl Conv {
    fn load(
        w: &Weights,
        name: &str,
        bias: bool,
        padding: usize,
        dilation: usize,
        dtype: DType,
    ) -> Result<Self> {
        Ok(Self {
            weight: w.get(&format!("{name}.weight"), dtype)?,
            bias: if bias {
                Some(w.get(&format!("{name}.bias"), dtype)?.reshape(((), 1))?)
            } else {
                None
            },
            padding,
            dilation,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let y = x.conv1d(&self.weight, self.padding, 1, self.dilation, 1)?;
        Ok(match &self.bias {
            Some(b) => y.broadcast_add(b)?,
            None => y,
        })
    }
}

struct Up {
    weight: Tensor,
    bias: Tensor,
    stride: usize,
    padding: usize,
}

impl Up {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let y = x.conv_transpose1d(&self.weight, self.padding, 0, self.stride, 1, 1)?;
        Ok(y.broadcast_add(&self.bias)?)
    }
}

struct ResBlock {
    convs1: Vec<Conv>,
    convs2: Vec<Conv>,
}

impl ResBlock {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let mut x = x.clone();
        for (c1, c2) in self.convs1.iter().zip(&self.convs2) {
            let xt = c1.forward(&leaky_relu(&x, LRELU_SLOPE)?)?;
            let xt = c2.forward(&leaky_relu(&xt, LRELU_SLOPE)?)?;
            x = (xt + x)?;
        }
        Ok(x)
    }
}

pub struct HifiGan {
    conv_pre: Conv,
    cond_layer: Conv,
    ups: Vec<Up>,
    conds: Vec<Conv>,
    resblocks: Vec<ResBlock>,
    conv_post: Conv,
    /// GPT latent frames → 22.05 kHz mel frames.
    scale_hop: f64,
    /// 22.05 kHz → 24 kHz.
    scale_rate: f64,
    pub dtype: DType,
    device: Device,
}

impl HifiGan {
    pub fn load(w: &Weights, cfg: &crate::config::ModelArgs, dtype: DType) -> Result<Self> {
        let p = "hifigan_decoder.waveform_decoder";
        let mut ups = Vec::new();
        let mut conds = Vec::new();
        let mut resblocks = Vec::new();
        for (i, (&u, &k)) in UPSAMPLE_RATES.iter().zip(&UPSAMPLE_KERNELS).enumerate() {
            ups.push(Up {
                weight: w.get(&format!("{p}.ups.{i}.weight"), dtype)?,
                bias: w
                    .get(&format!("{p}.ups.{i}.bias"), dtype)?
                    .reshape(((), 1))?,
                stride: u,
                padding: (k - u) / 2,
            });
            conds.push(Conv::load(w, &format!("{p}.conds.{i}"), true, 0, 1, dtype)?);
            for (j, &rk) in RES_KERNELS.iter().enumerate() {
                let r = format!("{p}.resblocks.{}", i * RES_KERNELS.len() + j);
                let mut convs1 = Vec::new();
                let mut convs2 = Vec::new();
                for (n, &d) in RES_DILATIONS.iter().enumerate() {
                    convs1.push(Conv::load(
                        w,
                        &format!("{r}.convs1.{n}"),
                        true,
                        (rk * d - d) / 2,
                        d,
                        dtype,
                    )?);
                    convs2.push(Conv::load(
                        w,
                        &format!("{r}.convs2.{n}"),
                        true,
                        (rk - 1) / 2,
                        1,
                        dtype,
                    )?);
                }
                resblocks.push(ResBlock { convs1, convs2 });
            }
        }
        Ok(Self {
            conv_pre: Conv::load(w, &format!("{p}.conv_pre"), true, 3, 1, dtype)?,
            cond_layer: Conv::load(w, &format!("{p}.cond_layer"), true, 0, 1, dtype)?,
            ups,
            conds,
            resblocks,
            conv_post: Conv::load(w, &format!("{p}.conv_post"), false, 3, 1, dtype)?,
            scale_hop: cfg.gpt_code_stride_len as f64 / cfg.output_hop_length as f64,
            scale_rate: cfg.output_sample_rate as f64 / cfg.input_sample_rate as f64,
            dtype,
            device: w.device.clone(),
        })
    }

    /// `latents` `[1, N, 1024]`, `spk` `[512]` → samples at 24 kHz, in one
    /// pass (memory grows with the length; [`StreamDecoder`] bounds it).
    pub fn forward(&self, latents: &Tensor, spk: &Tensor) -> Result<Vec<f32>> {
        let frames = self.frames(latents, 1.0)?;
        let o = self.generate(&frames, spk)?;
        Ok(o.to_dtype(DType::F32)?.flatten_all()?.to_vec1()?)
    }

    /// Latents `[1, N, 1024]` → decoder input frames `[1, 1024, F]` (one per
    /// 256 output samples): Coqui's two linear interpolations (x4, then
    /// 22.05 → 24 kHz). `speed` != 1 stretches the first one.
    pub fn frames(&self, latents: &Tensor, speed: f64) -> Result<Tensor> {
        let x = latents
            .to_device(&self.device)?
            .to_dtype(self.dtype)?
            .transpose(1, 2)?
            .contiguous()?;
        let x = interpolate_linear(&x, self.scale_hop / speed)?;
        interpolate_linear(&x, self.scale_rate)
    }

    /// Frames whose value no longer changes when latents are appended to
    /// `n` latents (the interpolation clamps at the last one).
    pub fn settled_frames(&self, n: usize, speed: f64) -> usize {
        let settled = |t_in: usize, scale: f64| -> usize {
            let t_out = (t_in as f64 * scale).floor() as usize;
            (0..t_out)
                .take_while(|&i| {
                    let src = ((i as f64 + 0.5) / scale - 0.5).max(0.0);
                    (src.floor() as usize) + 1 < t_in
                })
                .count()
        };
        let hop = settled(n, self.scale_hop / speed);
        settled(hop, self.scale_rate)
    }

    /// Frames `[1, 1024, F]` → waveform `[1, 1, F * 256]` (decoder dtype).
    pub fn generate(&self, frames: &Tensor, spk: &Tensor) -> Result<Tensor> {
        let g = spk
            .to_device(&self.device)?
            .to_dtype(self.dtype)?
            .reshape((1, (), 1))?;
        let mut o = self
            .conv_pre
            .forward(frames)?
            .broadcast_add(&self.cond_layer.forward(&g)?)?;
        let nk = RES_KERNELS.len();
        for (i, up) in self.ups.iter().enumerate() {
            o = up.forward(&leaky_relu(&o, LRELU_SLOPE)?)?;
            o = o.broadcast_add(&self.conds[i].forward(&g)?)?;
            let mut sum = self.resblocks[i * nk].forward(&o)?;
            for j in 1..nk {
                sum = (sum + self.resblocks[i * nk + j].forward(&o)?)?;
            }
            o = (sum / nk as f64)?;
        }
        // The last activation uses PyTorch's default slope.
        Ok(self.conv_post.forward(&leaky_relu(&o, 0.01)?)?.tanh()?)
    }
}

/// Samples per decoder frame.
pub const HOP: usize = 256;

/// Incremental decoding: latents arrive one by one, audio leaves in
/// windows of frames with `context` frames of left context and `lookahead`
/// frames held back until the next latents settle them. The frame grid is
/// the one of a single full pass, so windows join without a crossfade; the
/// only difference with one pass is the HiFi-GAN receptive field cut at
/// `context` / `lookahead` frames.
pub struct StreamDecoder<'a> {
    dec: &'a HifiGan,
    spk: Tensor,
    speed: f64,
    latents: Vec<Tensor>,
    emitted: usize,
    pub context: usize,
    pub lookahead: usize,
}

impl<'a> StreamDecoder<'a> {
    pub fn new(dec: &'a HifiGan, spk: &Tensor, speed: f64) -> Self {
        Self {
            dec,
            spk: spk.clone(),
            speed,
            latents: Vec::new(),
            emitted: 0,
            // The decoder's receptive field is ~12 frames each side:
            // 16/12 match one pass (89 dB SNR), 16/10 already drops to 36 dB.
            context: 16,
            lookahead: 12,
        }
    }

    /// Appends latents `[n, 1024]`.
    pub fn push(&mut self, latents: &Tensor) -> Result<()> {
        self.latents.push(latents.to_dtype(self.dec.dtype)?);
        Ok(())
    }

    pub fn n_latents(&self) -> usize {
        self.latents.iter().map(|t| t.dim(0).unwrap_or(0)).sum()
    }

    /// Samples ready so far; `last`: no more latents will come.
    pub fn decode(&mut self, last: bool) -> Result<Vec<f32>> {
        let n = self.n_latents();
        if n == 0 {
            return Ok(Vec::new());
        }
        let all = Tensor::cat(&self.latents, 0)?.unsqueeze(0)?;
        self.latents = vec![all.squeeze(0)?];
        let frames = self.dec.frames(&all, self.speed)?;
        let total = frames.dim(2)?;
        let (end, w_end) = if last {
            (total, total)
        } else {
            let ready = self.dec.settled_frames(n, self.speed).min(total);
            (ready.saturating_sub(self.lookahead), ready)
        };
        if end <= self.emitted {
            return Ok(Vec::new());
        }
        let w0 = self.emitted.saturating_sub(self.context);
        let window = frames.narrow(2, w0, w_end - w0)?.contiguous()?;
        let wav = self.dec.generate(&window, &self.spk)?;
        let wav = wav.narrow(2, (self.emitted - w0) * HOP, (end - self.emitted) * HOP)?;
        self.emitted = end;
        Ok(wav.to_dtype(DType::F32)?.flatten_all()?.to_vec1()?)
    }
}

/// `F.interpolate(x, scale_factor=scale, mode="linear")` on `[B, C, T]`:
/// `floor(T * scale)` outputs, source index `(i + 0.5) / scale - 0.5`
/// clamped at 0 (align_corners=False, scale used as given).
pub fn interpolate_linear(x: &Tensor, scale: f64) -> Result<Tensor> {
    let t_in = x.dim(2)?;
    let t_out = (t_in as f64 * scale).floor() as usize;
    let mut i0 = Vec::with_capacity(t_out);
    let mut i1 = Vec::with_capacity(t_out);
    let mut l1 = Vec::with_capacity(t_out);
    for i in 0..t_out {
        let src = ((i as f64 + 0.5) / scale - 0.5).max(0.0);
        let a = (src.floor() as usize).min(t_in - 1);
        let b = if a < t_in - 1 { a + 1 } else { a };
        i0.push(a as u32);
        i1.push(b as u32);
        l1.push((src - a as f64) as f32);
    }
    let dev = x.device();
    let x0 = x.index_select(&Tensor::new(i0, dev)?, 2)?;
    let x1 = x.index_select(&Tensor::new(i1, dev)?, 2)?;
    let l1 = Tensor::from_vec(l1, (1, 1, t_out), dev)?.to_dtype(x.dtype())?;
    let l0 = (1.0 - &l1)?;
    Ok((x0.broadcast_mul(&l0)? + x1.broadcast_mul(&l1)?)?)
}
