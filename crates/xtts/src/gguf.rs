//! Single-file GGUF models and loading from the original Coqui checkpoint
//! directory (`model.pth`, `config.json`, `vocab.json`, `speakers_xtts.pth`).
//!
//! GGUF layout:
//! - metadata `general.architecture = "xtts"`, `xtts.config` (config.json),
//!   `xtts.tokenizer` (vocab.json), `xtts.voices` (JSON array of the
//!   built-in speaker names, in the order of the `voices.*` tensors);
//! - the checkpoint tensors under their original names, with three changes:
//!   weight norm is folded (`*.parametrizations.weight.original0/1` →
//!   `*.weight`), the GPT-2 `Conv1D` weights are stored `[out, in]` like a
//!   linear layer, and 1x1 convolutions of the conditioning encoder are
//!   stored 2-D. Unused tensors (text head, DVAE, training buffers) are
//!   dropped, and with `--no-cloning` the voice-cloning encoders too;
//! - `voices.cond` `[n, 32, 1024]` (GPT conditioning latents) and
//!   `voices.spk` `[n, 512]` (speaker embeddings) for the built-in voices.
//!
//! The GPT-2 linear layers and the mel head use `gpt_dtype`; the HiFi-GAN
//! decoder uses `decoder_dtype`; everything else is F16 (embeddings,
//! conditioning encoder, perceiver) or F32 (norms, biases, speaker encoder).

use crate::weights::Weights;
use anyhow::{Context, Result, bail};
use candle_core::pickle::{Object, Stack};
use candle_core::quantized::{GgmlDType, QTensor, gguf_file};
use candle_core::{D, DType, Device, Tensor};
use std::collections::HashMap;
use std::io::Read;
use std::path::Path;
use std::sync::Arc;

pub const ARCH: &str = "xtts";

/// Files a model needs besides its weights.
pub struct Assets {
    pub config: String,
    pub tokenizer: String,
    pub voices: Vec<String>,
}

/// Parses a dtype name: f32, f16, bf16, q8_0, q6k, q5k, q4k, q4_0.
pub fn parse_dtype(name: &str) -> Result<GgmlDType> {
    Ok(match name.to_ascii_lowercase().as_str() {
        "f32" => GgmlDType::F32,
        "f16" => GgmlDType::F16,
        "bf16" => GgmlDType::BF16,
        "q8_0" | "q8" => GgmlDType::Q8_0,
        "q6k" | "q6_k" => GgmlDType::Q6K,
        "q5k" | "q5_k" => GgmlDType::Q5K,
        "q4k" | "q4_k" => GgmlDType::Q4K,
        "q4_0" => GgmlDType::Q4_0,
        other => bail!("unknown dtype '{other}' (f32, f16, bf16, q8_0, q6k, q5k, q4k, q4_0)"),
    })
}

/// `wanted` if the row length fits its block, else the closest that does.
fn fit_dtype(wanted: GgmlDType, row: usize) -> GgmlDType {
    for dt in [wanted, GgmlDType::Q8_0, GgmlDType::F16] {
        if row.is_multiple_of(dt.block_size()) {
            return dt;
        }
    }
    GgmlDType::F32
}

pub struct ConvertOptions {
    pub gpt_dtype: GgmlDType,
    pub decoder_dtype: GgmlDType,
    /// Leave out the voice-cloning encoders (conditioning encoder,
    /// perceiver, speaker encoder: 125 MB) — built-in voices only.
    pub no_cloning: bool,
}

impl Default for ConvertOptions {
    fn default() -> Self {
        Self {
            gpt_dtype: GgmlDType::Q8_0,
            decoder_dtype: GgmlDType::F16,
            no_cloning: false,
        }
    }
}

/// Names of the top-level keys of a pickled dict (`speakers_xtts.pth`).
fn pth_dict_keys(path: &Path) -> Result<Vec<String>> {
    let file = std::fs::File::open(path).with_context(|| format!("opening {path:?}"))?;
    let mut zip = zip::ZipArchive::new(std::io::BufReader::new(file))?;
    let name = zip
        .file_names()
        .find(|f| f.ends_with("data.pkl"))
        .map(str::to_string)
        .with_context(|| format!("{path:?}: no data.pkl"))?;
    let mut data = Vec::new();
    zip.by_name(&name)?.read_to_end(&mut data)?;
    let mut stack = Stack::empty();
    stack.read_loop(&mut std::io::Cursor::new(data))?;
    match stack.finalize()? {
        Object::Dict(kv) => Ok(kv
            .into_iter()
            .filter_map(|(k, _)| match k {
                Object::Unicode(s) => Some(s),
                _ => None,
            })
            .collect()),
        _ => bail!("{path:?}: not a dict"),
    }
}

/// Built-in voices: names, `[n, 32, 1024]` latents, `[n, 512]` embeddings.
fn read_voices(path: &Path) -> Result<(Vec<String>, Tensor, Tensor)> {
    let names = pth_dict_keys(path)?;
    let (mut conds, mut spks) = (Vec::new(), Vec::new());
    for name in &names {
        let t: HashMap<String, Tensor> = candle_core::pickle::read_all_with_key(path, Some(name))?
            .into_iter()
            .collect();
        let cond = t
            .get("gpt_cond_latent")
            .with_context(|| format!("voice {name}: no gpt_cond_latent"))?;
        let spk = t
            .get("speaker_embedding")
            .with_context(|| format!("voice {name}: no speaker_embedding"))?;
        conds.push(cond.reshape((32, 1024))?.to_dtype(DType::F32)?);
        spks.push(spk.reshape(512)?.to_dtype(DType::F32)?);
    }
    Ok((names, Tensor::stack(&conds, 0)?, Tensor::stack(&spks, 0)?))
}

/// Checkpoint tensors after the renames, folds and drops listed above.
fn read_checkpoint(path: &Path) -> Result<HashMap<String, Tensor>> {
    let raw = candle_core::pickle::read_all_with_key(path, Some("model"))
        .with_context(|| format!("reading {path:?}"))?;
    let mut raw: HashMap<String, Tensor> = raw.into_iter().collect();
    let mut out = HashMap::new();

    // Weight norm: w = g * v / ||v||, the norm taken over all dims but 0.
    let wn: Vec<String> = raw
        .keys()
        .filter_map(|k| k.strip_suffix(".parametrizations.weight.original0"))
        .map(str::to_string)
        .collect();
    for base in wn {
        let g = raw
            .remove(&format!("{base}.parametrizations.weight.original0"))
            .unwrap()
            .to_dtype(DType::F32)?;
        let v = raw
            .remove(&format!("{base}.parametrizations.weight.original1"))
            .with_context(|| format!("{base}: weight norm without original1"))?
            .to_dtype(DType::F32)?;
        let n = v.dim(0)?;
        let norm = v.reshape((n, ()))?.sqr()?.sum_keepdim(D::Minus1)?.sqrt()?;
        let mut shape = vec![n];
        shape.extend(std::iter::repeat_n(1, v.rank() - 1));
        let scale = (g.reshape(shape.as_slice())? / norm.reshape(shape.as_slice())?)?;
        out.insert(format!("{base}.weight"), v.broadcast_mul(&scale)?);
    }

    for (name, t) in raw {
        let drop = name.starts_with("gpt.text_head.")
            || name.starts_with("dvae.")
            || name.ends_with("num_batches_tracked")
            || name.starts_with("torch_mel_spectrogram");
        if drop {
            continue;
        }
        let t = t.to_dtype(DType::F32)?;
        let t = if name.starts_with("gpt.gpt.h.") && name.ends_with(".weight") && t.rank() == 2 {
            // GPT-2 Conv1D stores [in, out].
            t.t()?.contiguous()?
        } else if name.starts_with("gpt.conditioning_encoder.") && t.rank() == 3 && t.dim(2)? == 1 {
            t.squeeze(2)?
        } else {
            t
        };
        out.insert(name, t);
    }
    Ok(out)
}

fn read_assets_and_tensors(dir: &Path) -> Result<(Assets, HashMap<String, Tensor>)> {
    let read = |f: &str| {
        std::fs::read_to_string(dir.join(f)).with_context(|| format!("reading {:?}", dir.join(f)))
    };
    let mut tensors = read_checkpoint(&dir.join("model.pth"))?;
    let speakers = dir.join("speakers_xtts.pth");
    let voices = if speakers.exists() {
        let (names, cond, spk) = read_voices(&speakers)?;
        tensors.insert("voices.cond".into(), cond);
        tensors.insert("voices.spk".into(), spk);
        names
    } else {
        Vec::new()
    };
    Ok((
        Assets {
            config: read("config.json")?,
            tokenizer: read("vocab.json")?,
            voices,
        },
        tensors,
    ))
}

fn dtype_for(name: &str, t: &Tensor, opts: &ConvertOptions) -> Result<GgmlDType> {
    let rank = t.rank();
    let last = *t.dims().last().unwrap_or(&1);
    Ok(
        if name.starts_with("hifigan_decoder.speaker_encoder.")
            || name.starts_with("voices.")
            || rank == 1
            || name == "mel_stats"
        {
            GgmlDType::F32
        } else if name.starts_with("gpt.gpt.h.") || name.starts_with("gpt.mel_head.") {
            fit_dtype(opts.gpt_dtype, last)
        } else if name.starts_with("hifigan_decoder.waveform_decoder.") {
            match opts.decoder_dtype {
                dt @ (GgmlDType::F32 | GgmlDType::F16 | GgmlDType::BF16) => dt,
                _ => GgmlDType::F16,
            }
        } else {
            GgmlDType::F16
        },
    )
}

/// Tensors only voice cloning uses (computing a voice from reference audio).
fn is_cloning(name: &str) -> bool {
    name.starts_with("gpt.conditioning_encoder.")
        || name.starts_with("gpt.conditioning_perceiver.")
        || name.starts_with("hifigan_decoder.speaker_encoder.")
}

/// Converts a Coqui checkpoint directory to one GGUF file; returns per-dtype
/// tensor counts.
pub fn convert(dir: &Path, out: &Path, opts: &ConvertOptions) -> Result<Vec<(String, usize)>> {
    let (assets, tensors) = read_assets_and_tensors(dir)?;
    let mut names: Vec<&String> = tensors
        .keys()
        .filter(|n| !(opts.no_cloning && is_cloning(n)))
        .collect();
    names.sort();

    let mut qtensors: Vec<(String, QTensor)> = Vec::new();
    let mut counts: HashMap<String, usize> = HashMap::new();
    for name in names {
        let t = &tensors[name];
        let dt = dtype_for(name, t, opts)?;
        *counts.entry(format!("{dt:?}")).or_default() += 1;
        qtensors.push((name.clone(), QTensor::quantize(t, dt)?));
    }

    let metadata = [
        (
            "general.architecture",
            gguf_file::Value::String(ARCH.into()),
        ),
        ("xtts.config", gguf_file::Value::String(assets.config)),
        ("xtts.tokenizer", gguf_file::Value::String(assets.tokenizer)),
        (
            "xtts.voices",
            gguf_file::Value::String(serde_json::to_string(&assets.voices)?),
        ),
    ];
    let metadata: Vec<(&str, &gguf_file::Value)> = metadata.iter().map(|(k, v)| (*k, v)).collect();
    let refs: Vec<(&str, &QTensor)> = qtensors.iter().map(|(n, t)| (n.as_str(), t)).collect();
    let mut file = std::io::BufWriter::new(std::fs::File::create(out)?);
    gguf_file::write(&mut file, &metadata, &refs)?;

    let mut counts: Vec<_> = counts.into_iter().collect();
    counts.sort();
    Ok(counts)
}

/// Loads a GGUF file or a Coqui checkpoint directory (F32).
pub fn load(path: &Path, device: &Device) -> Result<(Weights, Assets)> {
    if path.is_dir() {
        let (assets, tensors) = read_assets_and_tensors(path)?;
        let mut map = HashMap::new();
        for (name, t) in tensors {
            map.insert(name, Arc::new(QTensor::quantize(&t, GgmlDType::F32)?));
        }
        // Dense tensors are moved to the device when the layers are built.
        return Ok((Weights::new(map, device.clone()), assets));
    }

    let mut file = std::fs::File::open(path).with_context(|| format!("opening {path:?}"))?;
    let content =
        gguf_file::Content::read(&mut file).with_context(|| format!("reading {path:?}"))?;
    let meta = |k: &str| -> Result<String> {
        match content.metadata.get(k) {
            Some(gguf_file::Value::String(s)) => Ok(s.clone()),
            _ => bail!("{path:?}: missing metadata '{k}'"),
        }
    };
    if meta("general.architecture")? != ARCH {
        bail!("{path:?} is not an {ARCH} GGUF");
    }
    let assets = Assets {
        config: meta("xtts.config")?,
        tokenizer: meta("xtts.tokenizer")?,
        voices: serde_json::from_str(&meta("xtts.voices").unwrap_or_else(|_| "[]".into()))?,
    };
    let mut map = HashMap::new();
    let names: Vec<String> = content.tensor_infos.keys().cloned().collect();
    for name in names {
        // Block-quantized tensors go straight to the device (they are used
        // as is); dense ones stay on the CPU until a layer converts them to
        // its compute dtype, so the device never holds two copies.
        let quantized = !matches!(
            content.tensor_infos[&name].ggml_dtype,
            GgmlDType::F32 | GgmlDType::F16 | GgmlDType::BF16
        );
        let dev = if quantized { device } else { &Device::Cpu };
        let t = content.tensor(&mut file, &name, dev)?;
        map.insert(name, Arc::new(t));
    }
    Ok((Weights::new(map, device.clone()), assets))
}
