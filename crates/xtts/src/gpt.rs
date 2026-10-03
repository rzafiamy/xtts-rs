//! The XTTS autoregressive model: a GPT-2 (no built-in position embedding)
//! over `[conditioning latents | text | audio codes]`, with learned position
//! embeddings per segment, `ln_f` then a second `final_norm`, and the mel
//! head. The output of `final_norm` at each audio position is the latent the
//! HiFi-GAN decoder turns into audio.

use crate::weights::{LayerNorm, Linear, Weights};
use anyhow::Result;
use candle_core::{D, DType, Device, Module, Tensor};
use candle_nn::kv_cache::KvCache;

struct Block {
    ln_1: LayerNorm,
    c_attn: Linear,
    c_proj: Linear,
    ln_2: LayerNorm,
    c_fc: Linear,
    mlp_proj: Linear,
}

pub struct Gpt {
    text_emb: Tensor,
    text_pos: Tensor,
    mel_emb: Tensor,
    mel_pos: Tensor,
    blocks: Vec<Block>,
    ln_f: LayerNorm,
    final_norm: LayerNorm,
    mel_head: Linear,
    device: Device,
    heads: usize,
    dim: usize,
    pub dtype: DType,
    max_seq: usize,
}

/// Key/value cache of one generation.
pub struct Cache {
    kv: Vec<KvCache>,
}

impl Cache {
    pub fn seq_len(&self) -> usize {
        self.kv[0].current_seq_len()
    }
}

impl Gpt {
    pub fn load(w: &Weights, layers: usize, heads: usize, dtype: DType) -> Result<Self> {
        let p = "gpt.gpt";
        let mut blocks = Vec::with_capacity(layers);
        for i in 0..layers {
            let b = format!("{p}.h.{i}");
            blocks.push(Block {
                ln_1: LayerNorm::load(w, &format!("{b}.ln_1"), 1e-5)?,
                c_attn: w.linear(&format!("{b}.attn.c_attn"), true, dtype)?,
                c_proj: w.linear(&format!("{b}.attn.c_proj"), true, dtype)?,
                ln_2: LayerNorm::load(w, &format!("{b}.ln_2"), 1e-5)?,
                c_fc: w.linear(&format!("{b}.mlp.c_fc"), true, dtype)?,
                mlp_proj: w.linear(&format!("{b}.mlp.c_proj"), true, dtype)?,
            });
        }
        // Embedding tables stay on the CPU (36 MB): only one row per step
        // is read, then copied to the device.
        let cpu = |name: &str| -> Result<Tensor> { w.get_on(name, DType::F32, &Device::Cpu) };
        let text_pos = cpu("gpt.text_pos_embedding.emb.weight")?;
        let mel_pos = cpu("gpt.mel_pos_embedding.emb.weight")?;
        let dim = text_pos.dim(1)?;
        // Conditioning latents + every text and audio position.
        let max_seq = 32 + text_pos.dim(0)? + mel_pos.dim(0)?;
        Ok(Self {
            text_emb: cpu("gpt.text_embedding.weight")?,
            text_pos,
            mel_emb: cpu("gpt.mel_embedding.weight")?,
            mel_pos,
            blocks,
            ln_f: LayerNorm::load(w, &format!("{p}.ln_f"), 1e-5)?,
            final_norm: LayerNorm::load(w, "gpt.final_norm", 1e-5)?,
            mel_head: w.linear("gpt.mel_head", true, dtype)?,
            device: w.device.clone(),
            heads,
            dim,
            dtype,
            max_seq,
        })
    }

    pub fn device(&self) -> &Device {
        &self.device
    }

    /// A cache that holds `capacity` positions before growing (by the
    /// same amount); size it to the expected sequence to save memory.
    pub fn new_cache(&self, capacity: usize) -> Cache {
        let capacity = capacity.clamp(1, self.max_seq);
        Cache {
            kv: (0..self.blocks.len())
                .map(|_| KvCache::new(2, capacity))
                .collect(),
        }
    }

    /// Longest text (with start/stop) the text position table covers.
    pub fn max_text_tokens(&self) -> usize {
        self.text_pos.dim(0).unwrap_or(0)
    }

    /// Longest audio sequence (with the start code) the position table covers.
    pub fn max_audio_positions(&self) -> usize {
        self.mel_pos.dim(0).unwrap_or(0)
    }

    /// Embeddings of `[cond | start, text..., stop | start_audio]`.
    /// `cond`: `[1, 32, dim]`; `text`: token ids already wrapped in
    /// start/stop.
    pub fn prefix_embeddings(
        &self,
        cond: &Tensor,
        text: &[u32],
        start_audio: u32,
    ) -> Result<Tensor> {
        let ids = Tensor::new(text, &Device::Cpu)?;
        let t = (self.text_emb.index_select(&ids, 0)? + self.text_pos.narrow(0, 0, text.len())?)?;
        let a = self.audio_embedding_cpu(start_audio, 0)?;
        let cond = cond
            .to_device(&Device::Cpu)?
            .to_dtype(DType::F32)?
            .squeeze(0)?;
        let x = Tensor::cat(&[&cond, &t, &a], 0)?.unsqueeze(0)?;
        Ok(x.to_device(&self.device)?.to_dtype(self.dtype)?)
    }

    /// Embedding `[1, dim]` of audio code `code` at audio position `pos`
    /// (0 is the start code).
    pub fn audio_embedding(&self, code: u32, pos: usize) -> Result<Tensor> {
        let x = self.audio_embedding_cpu(code, pos)?;
        Ok(x.to_device(&self.device)?.to_dtype(self.dtype)?)
    }

    fn audio_embedding_cpu(&self, code: u32, pos: usize) -> Result<Tensor> {
        let id = Tensor::new(&[code], &Device::Cpu)?;
        Ok((self.mel_emb.index_select(&id, 0)? + self.mel_pos.narrow(0, pos, 1)?)?)
    }

    /// Runs the transformer over `emb` `[1, T, dim]`, appending to `cache`.
    /// Returns the latents `[1, T, dim]` (after `ln_f` and `final_norm`).
    pub fn forward(&self, emb: &Tensor, cache: &mut Cache) -> Result<Tensor> {
        let (_, t, _) = emb.dims3()?;
        let offset = cache.seq_len();
        let mask = if t > 1 {
            Some(causal_mask(t, offset, self.device())?.to_dtype(self.dtype)?)
        } else {
            None
        };
        let mut x = emb.clone();
        for (block, kv) in self.blocks.iter().zip(cache.kv.iter_mut()) {
            let h = block.ln_1.forward(&x)?;
            let h = self.attention(block, &h, kv, mask.as_ref())?;
            x = (x + h)?;
            let h = block.ln_2.forward(&x)?;
            let h = block.c_fc.forward(&h)?.gelu()?;
            let h = block.mlp_proj.forward(&h)?;
            x = (x + h)?;
        }
        let x = self.ln_f.forward(&x)?;
        Ok(self.final_norm.forward(&x)?)
    }

    /// Mel-code logits (F32) of a latent `[1, dim]`.
    pub fn logits(&self, latent: &Tensor) -> Result<Tensor> {
        Ok(self.mel_head.forward(latent)?.to_dtype(DType::F32)?)
    }

    fn attention(
        &self,
        block: &Block,
        x: &Tensor,
        kv: &mut KvCache,
        mask: Option<&Tensor>,
    ) -> Result<Tensor> {
        let (b, t, _) = x.dims3()?;
        let hd = self.dim / self.heads;
        let qkv = block.c_attn.forward(x)?;
        let split = |i: usize| -> Result<Tensor> {
            Ok(qkv
                .narrow(D::Minus1, i * self.dim, self.dim)?
                .reshape((b, t, self.heads, hd))?
                .transpose(1, 2)?
                .contiguous()?)
        };
        let (q, k, v) = (split(0)?, split(1)?, split(2)?);
        let (k, v) = kv.append(&k, &v)?;
        let scale = 1.0 / (hd as f64).sqrt();
        let q = q.reshape((b * self.heads, t, hd))?;
        let s = k.dim(2)?;
        let k = k.reshape((b * self.heads, s, hd))?;
        let v = v.reshape((b * self.heads, s, hd))?;
        let mut att = (q.matmul(&k.t()?)? * scale)?;
        if let Some(m) = mask {
            att = att.broadcast_add(m)?;
        }
        let att =
            candle_nn::ops::softmax_last_dim(&att.to_dtype(DType::F32)?)?.to_dtype(self.dtype)?;
        let y = att.matmul(&v.contiguous()?)?;
        let y = y
            .reshape((b, self.heads, t, hd))?
            .transpose(1, 2)?
            .reshape((b, t, self.dim))?;
        Ok(block.c_proj.forward(&y)?)
    }
}

/// `[t, offset + t]` additive mask: query i sees keys `..= offset + i`.
fn causal_mask(t: usize, offset: usize, dev: &Device) -> Result<Tensor> {
    let s = offset + t;
    let data: Vec<f32> = (0..t)
        .flat_map(|i| {
            (0..s).map(move |j| {
                if j <= offset + i {
                    0.0
                } else {
                    f32::NEG_INFINITY
                }
            })
        })
        .collect();
    Ok(Tensor::from_vec(data, (t, s), dev)?)
}
