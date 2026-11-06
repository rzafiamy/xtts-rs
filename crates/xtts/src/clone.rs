//! Voice cloning: reference audio → GPT conditioning latents and HiFi-GAN
//! speaker embedding, the same computation as Coqui's
//! `Xtts.get_conditioning_latents` (`TTS/tts/models/xtts.py`,
//! `layers/tortoise/arch_utils.py`, `layers/xtts/perceiver_encoder.py`,
//! `encoder/models/resnet.py`).
//!
//! The audio front end runs on the CPU, the encoders on the model's device.

use crate::audio;
use crate::model::Voice;
use crate::weights::{Linear, Weights};
use anyhow::{Context, Result, bail};
use candle_core::{D, DType, Device, Module, Tensor};
use candle_nn::ops::{sigmoid, softmax};
use rayon::prelude::*;
use std::path::{Path, PathBuf};

const GPT_SR: u32 = 22050;
const SPK_SR: u32 = 16000;

/// `get_conditioning_latents` parameters; the defaults are XTTS v2's official
/// config.
#[derive(Debug, Clone)]
pub struct CloneOptions {
    /// Seconds kept of each reference clip.
    pub max_ref_length: usize,
    /// Seconds of the concatenated clips the GPT latents see.
    pub gpt_cond_len: usize,
    /// Seconds per chunk; the latents are averaged over chunks.
    pub gpt_cond_chunk_len: usize,
}

impl Default for CloneOptions {
    fn default() -> Self {
        Self {
            max_ref_length: 30,
            gpt_cond_len: 30,
            gpt_cond_chunk_len: 4,
        }
    }
}

/// Seconds spent in each stage of [`CloneEncoder::clone_files`].
#[derive(Debug, Clone, Default)]
pub struct CloneTimes {
    pub audio: f64,
    pub cond: f64,
    pub speaker: f64,
}

fn to_tensor(rows: &[Vec<f32>], device: &Device) -> Result<Tensor> {
    let (r, c) = (rows.len(), rows[0].len());
    let flat: Vec<f32> = rows.iter().flatten().copied().collect();
    Ok(Tensor::from_vec(flat, (r, c), device)?)
}

/// Waits for queued device work (a no-op on the CPU), so timings mean something.
fn sync(t: &Tensor) -> Result<()> {
    let _ = t.to_device(&Device::Cpu)?;
    Ok(())
}

/// `exp` for x <= 0 (Cephes' `expf`, 1-2 ulp), in a loop the compiler vectorizes.
#[inline(always)]
fn exp_neg(x: f32) -> f32 {
    let x = x.max(-87.3);
    let z = (x * std::f32::consts::LOG2_E + 0.5).floor();
    let r = x - z * 0.693_359_4 + z * 2.121_944_4e-4;
    let p = ((((((1.987_569_1e-4 * r + 1.398_199_9e-3) * r + 8.333_452e-3) * r + 4.166_579_6e-2)
        * r
        + 0.166_666_65)
        * r
        + 0.5)
        * r)
        * r
        + r
        + 1.0;
    p * f32::from_bits(((z as i32 + 127) as u32) << 23)
}

/// Softmax over the rows of `[R, C]`. On the CPU this is the hottest loop of
/// the conditioning encoder (R x C x heads x blocks values, each with an
/// `exp` that Candle does not vectorize).
fn softmax_rows(x: &Tensor) -> Result<Tensor> {
    if !x.device().is_cpu() {
        return Ok(softmax(x, D::Minus1)?);
    }
    let (r, c) = x.dims2()?;
    let mut v: Vec<f32> = x.flatten_all()?.to_vec1()?;
    for row in v.chunks_exact_mut(c) {
        let max = row.iter().fold(f32::NEG_INFINITY, |m, a| m.max(*a));
        for a in row.iter_mut() {
            *a = exp_neg(*a - max);
        }
        let inv = 1.0 / row.iter().sum::<f32>();
        for a in row.iter_mut() {
            *a *= inv;
        }
    }
    Ok(Tensor::from_vec(v, (r, c), &Device::Cpu)?)
}

// ---------------------------------------------------------------- GPT side

/// `GroupNorm` over `[T, C]`.
struct GroupNorm {
    groups: usize,
    weight: Tensor,
    bias: Tensor,
}

impl GroupNorm {
    fn load(w: &Weights, name: &str, groups: usize) -> Result<Self> {
        Ok(Self {
            groups,
            weight: w.get(&format!("{name}.weight"), DType::F32)?,
            bias: w.get(&format!("{name}.bias"), DType::F32)?,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let (t, c) = x.dims2()?;
        let g = x.reshape((t, self.groups, c / self.groups))?;
        let mean = g.mean_keepdim(0)?.mean_keepdim(2)?;
        let d = g.broadcast_sub(&mean)?;
        let var = d.sqr()?.mean_keepdim(0)?.mean_keepdim(2)?;
        let y = d.broadcast_div(&(var + 1e-5)?.sqrt()?)?.reshape((t, c))?;
        Ok(y.broadcast_mul(&self.weight)?.broadcast_add(&self.bias)?)
    }
}

/// `AttentionBlock`: QKV self-attention over time.
struct AttentionBlock {
    norm: GroupNorm,
    qkv: Linear,
    proj_out: Linear,
    heads: usize,
}

impl AttentionBlock {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let xn = self.norm.forward(x)?;
        let qkv = self.qkv.forward(&xn)?;
        let ch = qkv.dim(1)? / (3 * self.heads);
        let scale = 1.0 / (ch as f64).sqrt().sqrt();
        let mut outs = Vec::with_capacity(self.heads);
        for h in 0..self.heads {
            let base = h * 3 * ch;
            let q = (qkv.narrow(1, base, ch)? * scale)?;
            let k = (qkv.narrow(1, base + ch, ch)? * scale)?;
            let v = qkv.narrow(1, base + 2 * ch, ch)?.contiguous()?;
            let w = softmax_rows(&q.matmul(&k.t()?.contiguous()?)?)?;
            outs.push(w.matmul(&v)?);
        }
        let h = self.proj_out.forward(&Tensor::cat(&outs, 1)?)?;
        Ok((xn + h)?)
    }
}

/// Multi-head cross-attention of the perceiver; the context includes the
/// queries (`cross_attn_include_queries`).
struct PerceiverAttention {
    to_q: Linear,
    to_kv: Linear,
    to_out: Linear,
    heads: usize,
}

impl PerceiverAttention {
    fn forward(&self, latents: &Tensor, context: &Tensor) -> Result<Tensor> {
        let ctx = Tensor::cat(&[latents, context], 0)?;
        let q = self.to_q.forward(latents)?;
        let kv = self.to_kv.forward(&ctx)?;
        let d = q.dim(1)? / self.heads;
        let scale = (d as f64).powf(-0.5);
        let half = kv.dim(1)? / 2;
        let mut outs = Vec::with_capacity(self.heads);
        for h in 0..self.heads {
            let qh = q.narrow(1, h * d, d)?.contiguous()?;
            let kh = kv.narrow(1, h * d, d)?.t()?.contiguous()?;
            let vh = kv.narrow(1, half + h * d, d)?.contiguous()?;
            let sim = (qh.matmul(&kh)? * scale)?;
            outs.push(softmax(&sim, D::Minus1)?.matmul(&vh)?);
        }
        Ok(self.to_out.forward(&Tensor::cat(&outs, 1)?)?)
    }
}

struct PerceiverFeedForward {
    up: Linear,
    down: Linear,
}

impl PerceiverFeedForward {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let y = self.up.forward(x)?;
        let half = y.dim(1)? / 2;
        let a = y.narrow(1, 0, half)?;
        // Exact GELU (F.gelu's default), not the tanh approximation.
        let gate = y.narrow(1, half, half)?.gelu_erf()?;
        Ok(self.down.forward(&(a * gate)?)?)
    }
}

/// `PerceiverResampler`: any number of frames → 32 latents.
struct Perceiver {
    latents: Tensor,
    layers: Vec<(PerceiverAttention, PerceiverFeedForward)>,
    gamma: Tensor,
    scale: f64,
}

impl Perceiver {
    fn load(w: &Weights) -> Result<Self> {
        let p = "gpt.conditioning_perceiver";
        let mut layers = Vec::new();
        for i in 0..2 {
            let l = |s: &str| format!("{p}.layers.{i}.{s}");
            layers.push((
                PerceiverAttention {
                    to_q: w.linear(&l("0.to_q"), false, DType::F32)?,
                    to_kv: w.linear(&l("0.to_kv"), false, DType::F32)?,
                    to_out: w.linear(&l("0.to_out"), false, DType::F32)?,
                    heads: 8,
                },
                PerceiverFeedForward {
                    up: w.linear(&l("1.0"), true, DType::F32)?,
                    down: w.linear(&l("1.2"), true, DType::F32)?,
                },
            ));
        }
        let latents = w.get(&format!("{p}.latents"), DType::F32)?;
        let dim = latents.dim(1)? as f64;
        Ok(Self {
            latents,
            layers,
            gamma: w.get(&format!("{p}.norm.gamma"), DType::F32)?,
            scale: dim.sqrt(),
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let mut l = self.latents.clone();
        for (attn, ff) in &self.layers {
            l = (attn.forward(&l, x)? + &l)?;
            l = (ff.forward(&l)? + &l)?;
        }
        // RMSNorm: F.normalize(dim=-1) * sqrt(dim) * gamma.
        let norm = l
            .sqr()?
            .sum_keepdim(1)?
            .sqrt()?
            .clamp(1e-12, f64::INFINITY)?;
        Ok((l.broadcast_div(&norm)? * self.scale)?.broadcast_mul(&self.gamma)?)
    }
}

// ------------------------------------------------------------- speaker side

struct Conv2d {
    weight: Tensor,
    bias: Option<Tensor>,
    stride: usize,
    padding: usize,
    /// For the CPU: `[C_out, K * K * C_in]`, ordered (kh, kw, c_in).
    flat: Tensor,
    kernel: usize,
}

impl Conv2d {
    fn load(w: &Weights, name: &str, stride: usize, padding: usize, bias: bool) -> Result<Self> {
        let bias = if bias {
            let b = w.get(&format!("{name}.bias"), DType::F32)?;
            let n = b.dim(0)?;
            Some(b.reshape((1, n, 1, 1))?)
        } else {
            None
        };
        let weight = w.get(&format!("{name}.weight"), DType::F32)?;
        let (co, ci, k, _) = weight.dims4()?;
        let flat = weight
            .permute((0, 2, 3, 1))?
            .reshape((co, k * k * ci))?
            .contiguous()?;
        Ok(Self {
            weight,
            bias,
            stride,
            padding,
            flat,
            kernel: k,
        })
    }

    /// Full im2col and one big matmul. Candle's tiled conv2d takes twice as
    /// long on the CPU: its tiles are small and each runs its own matmul.
    fn forward_cpu(&self, x: &Tensor) -> Result<Tensor> {
        let (_, c, h, w) = x.dims4()?;
        let v: Vec<f32> = x.flatten_all()?.to_vec1()?;
        let (k, s, p) = (self.kernel, self.stride, self.padding);
        let (oh, ow) = ((h + 2 * p - k) / s + 1, (w + 2 * p - k) / s + 1);
        let n = oh * ow;
        let ks = k * k * c;
        // [K*K*C, oh*ow]: one row per kernel position and input channel.
        let mut a = vec![0f32; ks * n];
        a.par_chunks_mut(n).enumerate().for_each(|(i, row)| {
            let (pos, ch) = (i / c, i % c);
            let (kh, kw) = (pos / k, pos % k);
            let src = &v[ch * h * w..(ch + 1) * h * w];
            // Output columns whose input column is inside the image.
            let x0 = p.saturating_sub(kw).div_ceil(s);
            let x1 = ((w + p - kw - 1) / s + 1).min(ow);
            let start = x0 * s + kw - p;
            for (oy, line) in row.chunks_exact_mut(ow).enumerate() {
                let iy = (oy * s + kh) as isize - p as isize;
                if iy < 0 || iy >= h as isize {
                    continue;
                }
                let from = &src[iy as usize * w + start..(iy as usize + 1) * w];
                if s == 1 {
                    line[x0..x1].copy_from_slice(&from[..x1 - x0]);
                } else {
                    for (o, a) in line[x0..x1].iter_mut().zip(from.iter().step_by(s)) {
                        *o = *a;
                    }
                }
            }
        });
        let a = Tensor::from_vec(a, (ks, n), &Device::Cpu)?;
        let co = self.flat.dim(0)?;
        Ok(self.flat.matmul(&a)?.reshape((1, co, oh, ow))?)
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let y = if x.device().is_cpu() {
            self.forward_cpu(x)?
        } else {
            x.conv2d(&self.weight, self.padding, self.stride, 1, 1)?
        };
        Ok(match &self.bias {
            Some(b) => y.broadcast_add(b)?,
            None => y,
        })
    }
}

/// BatchNorm in eval mode; `shape` places the channel axis for broadcasting.
struct BatchNorm {
    mean: Tensor,
    std: Tensor,
    weight: Tensor,
    bias: Tensor,
}

impl BatchNorm {
    fn load(w: &Weights, name: &str, shape: &[usize]) -> Result<Self> {
        let get = |s: &str| -> Result<Tensor> {
            Ok(w.get(&format!("{name}.{s}"), DType::F32)?.reshape(shape)?)
        };
        Ok(Self {
            mean: get("running_mean")?,
            std: (get("running_var")? + 1e-5)?.sqrt()?,
            weight: get("weight")?,
            bias: get("bias")?,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        Ok(x.broadcast_sub(&self.mean)?
            .broadcast_div(&self.std)?
            .broadcast_mul(&self.weight)?
            .broadcast_add(&self.bias)?)
    }
}

struct SeBlock {
    conv1: Conv2d,
    bn1: BatchNorm,
    conv2: Conv2d,
    bn2: BatchNorm,
    fc1: Linear,
    fc2: Linear,
    down: Option<(Conv2d, BatchNorm)>,
}

impl SeBlock {
    fn load(w: &Weights, p: &str, stride: usize, down: bool) -> Result<Self> {
        let n = |s: &str| format!("{p}.{s}");
        const C: [usize; 4] = [1, usize::MAX, 1, 1];
        let chan = |name: &str| -> Result<Vec<usize>> {
            let c = w.get(&format!("{name}.weight"), DType::F32)?.dim(0)?;
            Ok(C.map(|v| if v == usize::MAX { c } else { v }).to_vec())
        };
        Ok(Self {
            conv1: Conv2d::load(w, &n("conv1"), stride, 1, false)?,
            bn1: BatchNorm::load(w, &n("bn1"), &chan(&n("bn1"))?)?,
            conv2: Conv2d::load(w, &n("conv2"), 1, 1, false)?,
            bn2: BatchNorm::load(w, &n("bn2"), &chan(&n("bn2"))?)?,
            fc1: w.linear(&n("se.fc.0"), true, DType::F32)?,
            fc2: w.linear(&n("se.fc.2"), true, DType::F32)?,
            down: if down {
                Some((
                    Conv2d::load(w, &n("downsample.0"), stride, 0, false)?,
                    BatchNorm::load(w, &n("downsample.1"), &chan(&n("downsample.1"))?)?,
                ))
            } else {
                None
            },
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let mut y = self.bn1.forward(&self.conv1.forward(x)?.relu()?)?;
        y = self.bn2.forward(&self.conv2.forward(&y)?)?;
        // Squeeze and excitation.
        let c = y.dim(1)?;
        let pooled = y.mean_keepdim(2)?.mean_keepdim(3)?.reshape((1, c))?;
        let s = sigmoid(&self.fc2.forward(&self.fc1.forward(&pooled)?.relu()?)?)?;
        y = y.broadcast_mul(&s.reshape((1, c, 1, 1))?)?;
        let res = match &self.down {
            Some((conv, bn)) => bn.forward(&conv.forward(x)?)?,
            None => x.clone(),
        };
        Ok((y + res)?.relu()?)
    }
}

/// `ResNetSpeakerEncoder` (H/ASP) with its own torch_spec front end.
struct SpeakerEncoder {
    conv1: Conv2d,
    bn1: BatchNorm,
    blocks: Vec<SeBlock>,
    att1: Linear,
    att_bn: BatchNorm,
    att2: Linear,
    fc: Linear,
    window: Vec<f32>,
    fb: Vec<Vec<f32>>,
    preemph: f32,
}

/// A 1x1 convolution stored `[out, in, 1]`, as a linear layer over `[T, C]`.
fn conv1x1(w: &Weights, name: &str, device: &Device) -> Result<Linear> {
    let weight = w.get_on(&format!("{name}.weight"), DType::F32, &Device::Cpu)?;
    let weight = weight.squeeze(2)?.to_device(device)?;
    let bias = w.get(&format!("{name}.bias"), DType::F32)?;
    Ok(Linear::from_dense(weight, Some(bias)))
}

impl SpeakerEncoder {
    fn load(w: &Weights) -> Result<Self> {
        let p = "hifigan_decoder.speaker_encoder";
        let n = |s: &str| format!("{p}.{s}");
        let mut blocks = Vec::new();
        for (layer, count, stride) in [(1, 3, 1), (2, 4, 2), (3, 6, 2), (4, 3, 2)] {
            for i in 0..count {
                let first = i == 0;
                blocks.push(SeBlock::load(
                    w,
                    &n(&format!("layer{layer}.{i}")),
                    if first { stride } else { 1 },
                    first && layer != 1,
                )?);
            }
        }
        let cpu = &Device::Cpu;
        let window: Vec<f32> = w
            .get_on(&n("torch_spec.1.spectrogram.window"), DType::F32, cpu)?
            .to_vec1()?;
        let fb: Vec<Vec<f32>> = w
            .get_on(&n("torch_spec.1.mel_scale.fb"), DType::F32, cpu)?
            .to_vec2()?;
        let filter: Vec<f32> = w
            .get_on(&n("torch_spec.0.filter"), DType::F32, cpu)?
            .flatten_all()?
            .to_vec1()?;
        Ok(Self {
            conv1: Conv2d::load(w, &n("conv1"), 1, 1, true)?,
            bn1: BatchNorm::load(w, &n("bn1"), &[1, 32, 1, 1])?,
            blocks,
            att1: conv1x1(w, &n("attention.0"), &w.device)?,
            att_bn: BatchNorm::load(w, &n("attention.2"), &[128])?,
            att2: conv1x1(w, &n("attention.3"), &w.device)?,
            fc: w.linear(&n("fc"), true, DType::F32)?,
            window,
            fb,
            preemph: -filter[0],
        })
    }

    /// `torch_spec`, the log and the instance norm: `[frames][64]`.
    fn features(&self, wav16k: &[f32]) -> Result<Vec<Vec<f32>>> {
        let x = audio::pre_emphasis(wav16k, self.preemph);
        let spec = audio::power_spectrogram(&x, 512, 160, &self.window)?;
        let mut mel = audio::apply_filterbank(&spec, &self.fb);
        let (t, m) = (mel.len(), mel[0].len());
        for c in 0..m {
            let log: Vec<f64> = mel.iter().map(|r| ((r[c] + 1e-6).ln()) as f64).collect();
            let mean = log.iter().sum::<f64>() / t as f64;
            let var = log.iter().map(|v| (v - mean) * (v - mean)).sum::<f64>() / t as f64;
            let sd = (var + 1e-5).sqrt();
            for (r, v) in mel.iter_mut().zip(&log) {
                r[c] = ((v - mean) / sd) as f32;
            }
        }
        Ok(mel)
    }

    /// `[frames][64]` → L2-normalized `[512]`.
    fn forward(&self, feats: &Tensor) -> Result<Tensor> {
        let (t, m) = feats.dims2()?;
        let x = feats.t()?.reshape((1, 1, m, t))?.contiguous()?;
        let mut x = self.bn1.forward(&self.conv1.forward(&x)?.relu()?)?;
        for b in &self.blocks {
            x = b.forward(&x)?;
        }
        let (_, c, f, tt) = x.dims4()?;
        // [1, C*F, T'] as [T', C*F].
        let x = x.reshape((c * f, tt))?.t()?.contiguous()?;
        let a = self.att1.forward(&x)?.relu()?;
        let a = self.att_bn.forward(&a)?;
        let w = softmax(&self.att2.forward(&a)?, 0)?;
        let mu = (&x * &w)?.sum_keepdim(0)?;
        let m2 = (x.sqr()? * &w)?.sum_keepdim(0)?;
        let sg = (m2 - mu.sqr()?)?.clamp(1e-5, f64::INFINITY)?.sqrt()?;
        let y = self.fc.forward(&Tensor::cat(&[mu, sg], 1)?)?;
        let norm = y
            .sqr()?
            .sum_keepdim(1)?
            .sqrt()?
            .clamp(1e-12, f64::INFINITY)?;
        Ok(y.broadcast_div(&norm)?.squeeze(0)?)
    }
}

// -------------------------------------------------------------- the encoder

/// The three voice-cloning networks, without the GPT or the decoder.
pub struct CloneEncoder {
    mel_stats: Vec<f32>,
    init: Linear,
    attn: Vec<AttentionBlock>,
    perceiver: Perceiver,
    speaker: SpeakerEncoder,
    hann: Vec<f32>,
    mel_fb: Vec<Vec<f32>>,
    pub device: Device,
}

impl CloneEncoder {
    /// Loads from a GGUF that has the encoders (a full one or one written by
    /// `xtts convert --only-cloning`) or a Coqui checkpoint directory.
    pub fn load(path: &Path, device: &Device) -> Result<Self> {
        let w = crate::gguf::load_cloning(path, device)?;
        Self::from_weights(&w)
    }

    fn from_weights(w: &Weights) -> Result<Self> {
        let p = "gpt.conditioning_encoder";
        if !w.contains(&format!("{p}.init.weight")) {
            bail!(
                "this model has no cloning encoders (converted with --no-cloning); \
                 use one from `xtts convert --only-cloning`"
            );
        }
        let mut attn = Vec::new();
        for i in 0..6 {
            let n = format!("{p}.attn.{i}");
            attn.push(AttentionBlock {
                norm: GroupNorm::load(w, &format!("{n}.norm"), 32)?,
                qkv: w.linear(&format!("{n}.qkv"), true, DType::F32)?,
                proj_out: w.linear(&format!("{n}.proj_out"), true, DType::F32)?,
                // ConditioningEncoder(..., num_attn_heads=gpt_n_heads)
                heads: 16,
            });
        }
        Ok(Self {
            mel_stats: w.get_on("mel_stats", DType::F32, &Device::Cpu)?.to_vec1()?,
            init: w.linear(&format!("{p}.init"), true, DType::F32)?,
            attn,
            perceiver: Perceiver::load(w)?,
            speaker: SpeakerEncoder::load(w)?,
            hann: audio::hann_window(1024),
            mel_fb: audio::mel_filterbank(1025, 0.0, 8000.0, 80, GPT_SR, true),
            device: w.device.clone(),
        })
    }

    /// `wav_to_mel_cloning` at 22.05 kHz: `[frames][80]`.
    pub fn gpt_mel(&self, chunk: &[f32]) -> Result<Vec<Vec<f32>>> {
        let spec = audio::power_spectrogram(chunk, 2048, 256, &self.hann)?;
        let mut mel = audio::apply_filterbank(&spec, &self.mel_fb);
        for r in &mut mel {
            for (v, s) in r.iter_mut().zip(&self.mel_stats) {
                *v = v.max(1e-5).ln() / s;
            }
        }
        Ok(mel)
    }

    /// `ConditioningEncoder`: `[T, 80]` → `[T, 1024]`.
    pub fn cond_encoder(&self, mel: &Tensor) -> Result<Tensor> {
        let mut x = self.init.forward(mel)?;
        for b in &self.attn {
            x = b.forward(&x)?;
        }
        Ok(x)
    }

    /// `[T, 1024]` → `[32, 1024]`.
    pub fn perceiver(&self, x: &Tensor) -> Result<Tensor> {
        self.perceiver.forward(x)
    }

    /// `get_style_emb` for one chunk of audio.
    pub fn style_emb(&self, chunk: &[f32]) -> Result<Tensor> {
        let mel = to_tensor(&self.gpt_mel(chunk)?, &self.device)?;
        self.perceiver(&self.cond_encoder(&mel)?)
    }

    pub fn speaker_features(&self, wav16k: &[f32]) -> Result<Vec<Vec<f32>>> {
        self.speaker.features(wav16k)
    }

    /// `[frames, 64]` features → `[512]`.
    pub fn speaker_forward(&self, feats: &Tensor) -> Result<Tensor> {
        self.speaker.forward(feats)
    }

    /// One clip at 22.05 kHz → `[512]`.
    pub fn speaker_embedding(&self, wav: &[f32]) -> Result<Tensor> {
        let w16 = audio::resample(wav, GPT_SR, SPK_SR);
        let feats = to_tensor(&self.speaker.features(&w16)?, &self.device)?;
        self.speaker.forward(&feats)
    }

    /// GPT latents `[32, 1024]` of the concatenated clips at 22.05 kHz.
    pub fn gpt_latents(&self, audio22k: &[f32], o: &CloneOptions) -> Result<Tensor> {
        let total = audio22k.len().min(GPT_SR as usize * o.gpt_cond_len);
        let step = GPT_SR as usize * o.gpt_cond_chunk_len;
        let mut sum: Option<Tensor> = None;
        let mut n = 0usize;
        for start in (0..total).step_by(step) {
            let chunk = &audio22k[start..(start + step).min(total)];
            // Coqui's MIN_AUDIO_SECONDS.
            if (chunk.len() as f64) < GPT_SR as f64 * 0.33 {
                continue;
            }
            let emb = self.style_emb(chunk)?;
            sum = Some(match sum {
                Some(s) => (s + emb)?,
                None => emb,
            });
            n += 1;
        }
        let sum = sum.context("reference audio too short (minimum 0.33 s)")?;
        Ok((sum / n as f64)?)
    }

    /// Reads one clip the way Coqui's `load_audio` does: mono, 22.05 kHz,
    /// clipped to [-1, 1], at most `max_ref_length` seconds.
    pub fn load_clip(&self, path: &Path, o: &CloneOptions) -> Result<Vec<f32>> {
        let (x, sr) = audio::read_wav(path)?;
        let mut y = audio::resample(&x, sr, GPT_SR);
        for v in &mut y {
            *v = v.clamp(-1.0, 1.0);
        }
        y.truncate(GPT_SR as usize * o.max_ref_length);
        Ok(y)
    }

    /// Reference clips → voice on the CPU, as `get_conditioning_latents`.
    pub fn clone_files(&self, paths: &[PathBuf], o: &CloneOptions) -> Result<(Voice, CloneTimes)> {
        if paths.is_empty() {
            bail!("no reference audio");
        }
        let mut times = CloneTimes::default();
        let t0 = std::time::Instant::now();
        let mut clips = Vec::with_capacity(paths.len());
        for p in paths {
            clips.push(self.load_clip(p, o)?);
        }
        times.audio = t0.elapsed().as_secs_f64();

        let t0 = std::time::Instant::now();
        let mut sum: Option<Tensor> = None;
        for c in &clips {
            let e = self.speaker_embedding(c)?;
            sum = Some(match sum {
                Some(s) => (s + e)?,
                None => e,
            });
        }
        let spk = (sum.unwrap() / clips.len() as f64)?;
        let spk = spk.to_device(&Device::Cpu)?;
        times.speaker = t0.elapsed().as_secs_f64();

        let t0 = std::time::Instant::now();
        let full: Vec<f32> = clips.into_iter().flatten().collect();
        let cond = self.gpt_latents(&full, o)?;
        sync(&cond)?;
        let cond = cond.to_device(&Device::Cpu)?.unsqueeze(0)?;
        times.cond = t0.elapsed().as_secs_f64();
        Ok((Voice { cond, spk }, times))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exp_neg_is_accurate() {
        let mut i = 0.0f32;
        while i > -80.0 {
            let (a, b) = (exp_neg(i), i.exp());
            assert!(((a - b) / b).abs() < 4e-7, "{i}: {a} vs {b}");
            i -= 0.0137;
        }
        assert!(exp_neg(-1000.0) < 1e-37);
    }
}
