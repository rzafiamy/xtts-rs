//! XTTS-v2 end to end: text → codes (GPT, autoregressive) → latents →
//! HiFi-GAN waveform at 24 kHz.

use crate::config::Config;
use crate::gpt::Gpt;
use crate::hifigan::{HifiGan, StreamDecoder};
use crate::sampling::{Sampler, SamplingOptions};
use crate::text::{self, TextTokenizer};
use anyhow::{Context, Result, bail};
use candle_core::{DType, Device, Tensor};
use std::path::Path;

/// Conditioning of one voice: GPT latents `[1, 32, 1024]` and the
/// HiFi-GAN speaker embedding `[512]`.
#[derive(Clone)]
pub struct Voice {
    pub cond: Tensor,
    pub spk: Tensor,
}

#[derive(Debug, Clone)]
pub struct SynthOptions {
    pub sampling: SamplingOptions,
    /// Spell out numbers and symbols (`tn`) before XTTS' cleaners.
    pub normalize: bool,
    /// Split text longer than the language's character limit at sentences.
    pub split: bool,
    /// Speech rate (latents are resampled before the decoder).
    pub speed: f32,
    /// Upper bound on codes per chunk (XTTS: 602, ~28 s).
    pub max_codes: usize,
    /// Codes decoded per decoder window by [`Xtts::synthesize`]; bounds the
    /// decoder's memory (its activations grow with the window).
    pub decode_chunk: usize,
}

impl Default for SynthOptions {
    fn default() -> Self {
        Self {
            sampling: SamplingOptions::default(),
            normalize: true,
            split: true,
            speed: 1.0,
            max_codes: 602,
            decode_chunk: 48,
        }
    }
}

/// Output of [`Xtts::generate_codes`].
pub struct Codes {
    /// Audio codes without the stop code.
    pub codes: Vec<u32>,
    /// The latent of each code (the GPT state that produced it), plus the
    /// one that produced the stop code: `[1, N (+1), 1024]`.
    pub latents: Tensor,
    /// Whether the stop code was produced (false: `max_codes` reached).
    pub stopped: bool,
}

pub struct Xtts {
    pub config: Config,
    pub gpt: Gpt,
    pub decoder: HifiGan,
    pub tokenizer: TextTokenizer,
    voice_names: Vec<String>,
    voice_cond: Option<Tensor>,
    voice_spk: Option<Tensor>,
    pub device: Device,
}

impl Xtts {
    /// Loads a GGUF file or a Coqui checkpoint directory. `gpt_dtype` is the
    /// compute dtype of the GPT's dense layers; the decoder runs in F32.
    pub fn load(path: &Path, device: &Device, gpt_dtype: DType) -> Result<Self> {
        Self::load_with(path, device, gpt_dtype, DType::F32)
    }

    /// [`Self::load`] with the decoder's compute dtype too. F16 is not worth
    /// it: 27 dB SNR against PyTorch (F32: 79 dB) and slower on an RTX 4090.
    pub fn load_with(
        path: &Path,
        device: &Device,
        gpt_dtype: DType,
        decoder_dtype: DType,
    ) -> Result<Self> {
        let (weights, assets) = crate::gguf::load(path, device)?;
        let config = Config::from_json(&assets.config).context("config.json")?;
        let a = &config.model_args;
        let gpt = Gpt::load(&weights, a.gpt_layers, a.gpt_n_heads, gpt_dtype)?;
        let decoder = HifiGan::load(&weights, a, decoder_dtype)?;
        let tokenizer = TextTokenizer::from_json(&assets.tokenizer)?;
        let (voice_cond, voice_spk) = if weights.contains("voices.cond") {
            (
                // Built-in voices stay on the CPU; one is copied per request.
                Some(weights.get_on("voices.cond", DType::F32, &Device::Cpu)?),
                Some(weights.get_on("voices.spk", DType::F32, &Device::Cpu)?),
            )
        } else {
            (None, None)
        };
        Ok(Self {
            config,
            gpt,
            decoder,
            tokenizer,
            voice_names: assets.voices,
            voice_cond,
            voice_spk,
            device: device.clone(),
        })
    }

    pub fn sample_rate(&self) -> u32 {
        self.config.model_args.output_sample_rate as u32
    }

    /// Built-in voice names (Coqui's 58 studio speakers).
    pub fn voices(&self) -> &[String] {
        &self.voice_names
    }

    /// A built-in voice, by name (case-insensitive; `_` or `-` may stand
    /// for spaces).
    pub fn voice(&self, name: &str) -> Result<Voice> {
        let norm = |s: &str| s.to_lowercase().replace(['_', '-'], " ");
        let want = norm(name);
        let i = self
            .voice_names
            .iter()
            .position(|n| norm(n) == want)
            .with_context(|| format!("unknown voice '{name}'"))?;
        let (cond, spk) = match (&self.voice_cond, &self.voice_spk) {
            (Some(c), Some(s)) => (c, s),
            _ => bail!("this model has no built-in voices"),
        };
        Ok(Voice {
            cond: cond.get(i)?.unsqueeze(0)?.to_device(&self.device)?,
            spk: spk.get(i)?.to_device(&self.device)?,
        })
    }

    /// Text chunks (normalized, split) that [`Self::synthesize`] speaks.
    pub fn prepare_text(&self, text: &str, lang: &str, opts: &SynthOptions) -> Vec<String> {
        let lang = text::base_lang(lang);
        let t = if opts.normalize {
            text::normalize(text, &lang)
        } else {
            text.to_string()
        };
        let t = t.replace('\n', " ");
        let chunks = if opts.split {
            text::split_sentences(&t, text::char_limit(&lang))
        } else {
            vec![t.trim().to_string()]
        };
        chunks.iter().map(|c| text::drop_final_period(c)).collect()
    }

    /// Runs the GPT on one chunk of text; `on_code` sees each new code with
    /// its latent and may return false to stop early.
    pub fn generate_codes(
        &self,
        text_ids: &[u32],
        voice: &Voice,
        sampling: &SamplingOptions,
        max_codes: usize,
        mut on_code: impl FnMut(u32, &Tensor) -> bool,
    ) -> Result<Codes> {
        let a = &self.config.model_args;
        let mut ids = Vec::with_capacity(text_ids.len() + 2);
        ids.push(self.tokenizer.start);
        ids.extend_from_slice(text_ids);
        ids.push(self.tokenizer.stop);
        if ids.len() > self.gpt.max_text_tokens() {
            bail!(
                "text too long: {} tokens (max {})",
                ids.len(),
                self.gpt.max_text_tokens()
            );
        }
        let max_codes = max_codes.min(self.gpt.max_audio_positions().saturating_sub(3));

        let prefix = self
            .gpt
            .prefix_embeddings(&voice.cond, &ids, a.gpt_start_audio_token)?;
        let prefix_len = prefix.dim(1)?;
        // Speech runs at ~1 code per character or less; grows if needed.
        let mut cache = self
            .gpt
            .new_cache(prefix_len + (ids.len() * 2).clamp(64, max_codes + 1));
        // `generate` sees `[1] * prefix | start_audio` as its input ids, so
        // the repetition penalty starts with codes 1 and start.
        let mut sampler = Sampler::new(
            sampling.clone(),
            a.gpt_num_audio_tokens,
            &[1, a.gpt_start_audio_token],
        )
        .with_stop(a.gpt_stop_audio_token);
        let h = self.gpt.forward(&prefix, &mut cache)?;
        let mut latent = h.narrow(1, prefix_len - 1, 1)?.squeeze(1)?;
        let mut codes = Vec::new();
        let mut latents = Vec::new();
        let mut stopped = false;
        while codes.len() < max_codes {
            let logits: Vec<f32> = self.gpt.logits(&latent)?.flatten_all()?.to_vec1()?;
            let code = sampler.next(&logits);
            if code == a.gpt_stop_audio_token {
                // Coqui decodes the latent that produced the stop code too
                // (`gpt(..., return_latent=True)[:, :-5]` keeps it).
                latents.push(latent.clone());
                stopped = true;
                break;
            }
            codes.push(code);
            latents.push(latent.clone());
            if !on_code(code, &latent) {
                break;
            }
            let emb = self.gpt.audio_embedding(code, codes.len())?.unsqueeze(0)?;
            latent = self.gpt.forward(&emb, &mut cache)?.squeeze(1)?;
        }
        let latents = if latents.is_empty() {
            Tensor::zeros((1, 0, self.gpt_dim()), DType::F32, &self.device)?
        } else {
            Tensor::cat(&latents, 0)?
                .unsqueeze(0)?
                .to_dtype(DType::F32)?
        };
        Ok(Codes {
            codes,
            latents,
            stopped,
        })
    }

    fn gpt_dim(&self) -> usize {
        self.config.model_args.gpt_n_model_channels
    }

    /// Latents → waveform, with the speed change applied first.
    pub fn decode(&self, latents: &Tensor, voice: &Voice, speed: f32) -> Result<Vec<f32>> {
        if latents.dim(1)? == 0 {
            return Ok(Vec::new());
        }
        let latents = if (speed - 1.0).abs() > 1e-3 {
            let scale = 1.0 / speed.max(0.05) as f64;
            crate::hifigan::interpolate_linear(&latents.transpose(1, 2)?.contiguous()?, scale)?
                .transpose(1, 2)?
                .contiguous()?
        } else {
            latents.clone()
        };
        self.decoder.forward(&latents, &voice.spk)
    }

    /// Text → 24 kHz samples.
    pub fn synthesize(
        &self,
        text: &str,
        lang: &str,
        voice: &Voice,
        opts: &SynthOptions,
    ) -> Result<Vec<f32>> {
        let mut out = Vec::new();
        let stream = StreamOptions {
            first_chunk: opts.decode_chunk,
            chunk: opts.decode_chunk,
        };
        self.synthesize_stream(text, lang, voice, opts, &stream, |a| {
            out.extend_from_slice(a);
            true
        })?;
        Ok(out)
    }

    /// Text → 24 kHz samples, handed to `on_audio` as soon as they are
    /// decoded (return false to stop).
    pub fn synthesize_stream(
        &self,
        text: &str,
        lang: &str,
        voice: &Voice,
        opts: &SynthOptions,
        stream: &StreamOptions,
        mut on_audio: impl FnMut(&[f32]) -> bool,
    ) -> Result<()> {
        for chunk in self.prepare_text(text, lang, opts) {
            let ids = self.tokenizer.encode(&chunk, lang)?;
            let mut dec =
                StreamDecoder::new(&self.decoder, &voice.spk, opts.speed.max(0.05) as f64);
            let mut pending = 0;
            let mut first = true;
            let mut stopped = false;
            let mut error = None;
            let out =
                self.generate_codes(&ids, voice, &opts.sampling, opts.max_codes, |_, latent| {
                    if let Err(e) = dec.push(latent) {
                        error = Some(e);
                        return false;
                    }
                    pending += 1;
                    let due = if first {
                        stream.first_chunk
                    } else {
                        stream.chunk
                    };
                    if pending < due {
                        return true;
                    }
                    pending = 0;
                    match dec.decode(false) {
                        Ok(a) if a.is_empty() => true,
                        Ok(a) => {
                            first = false;
                            stopped = !on_audio(&a);
                            !stopped
                        }
                        Err(e) => {
                            error = Some(e);
                            false
                        }
                    }
                })?;
            if let Some(e) = error {
                return Err(e);
            }
            if stopped {
                return Ok(());
            }
            // The latent of the stop code, which the callback did not see.
            let seen = dec.n_latents();
            let n = out.latents.dim(1)?;
            if n > seen {
                dec.push(&out.latents.squeeze(0)?.narrow(0, seen, n - seen)?)?;
            }
            let a = dec.decode(true)?;
            if !a.is_empty() && !on_audio(&a) {
                return Ok(());
            }
        }
        Ok(())
    }
}

/// When streamed audio is decoded, in codes (~46 ms of audio each).
#[derive(Debug, Clone)]
pub struct StreamOptions {
    /// Codes before the first audio (latency vs. a first-chunk hiccup).
    pub first_chunk: usize,
    /// Codes per later chunk.
    pub chunk: usize,
}

impl Default for StreamOptions {
    fn default() -> Self {
        Self {
            first_chunk: 8,
            chunk: 16,
        }
    }
}
