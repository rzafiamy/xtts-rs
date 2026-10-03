//! Compares the port with a dump of scripts/ref.py (greedy decoding).
//!
//! cargo run --release --example parity -- <model> <ref.safetensors> <lang> "<text>" [--cpu] [--f16]
//!
//! Checks, in order: text ids, GPT codes (must match exactly), latents of
//! the reference codes (teacher-forced: no divergence after a mismatch), and
//! the waveform of the reference latents through the HiFi-GAN decoder.

use anyhow::{Result, bail};
use candle_core::{DType, Device, IndexOp, Tensor};
use std::path::Path;
use xtts::{SamplingOptions, Voice, Xtts};

fn max_abs(a: &Tensor, b: &Tensor) -> Result<f32> {
    Ok((a - b)?.abs()?.flatten_all()?.max(0)?.to_scalar::<f32>()?)
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 5 {
        bail!("usage: parity <model> <ref.safetensors> <lang> <text> [--cpu] [--f16]");
    }
    let cpu = args.iter().any(|a| a == "--cpu");
    let f16 = args.iter().any(|a| a == "--f16");
    let device = if cpu {
        Device::Cpu
    } else {
        Device::cuda_if_available(0)?
    };
    let dtype = if f16 { DType::F16 } else { DType::F32 };
    let dec_dtype = if args.iter().any(|a| a == "--dec-f16") {
        DType::F16
    } else {
        DType::F32
    };
    let model = Xtts::load_with(Path::new(&args[1]), &device, dtype, dec_dtype)?;
    let r = candle_core::safetensors::load(&args[2], &device)?;
    let (lang, text) = (&args[3], &args[4]);

    // 1. Text ids (Coqui lowercases before encoding; no tn here).
    let ref_ids: Vec<u32> = r["text_ids"].to_dtype(DType::U32)?.to_vec1()?;
    let ids = model.tokenizer.encode(&text.to_lowercase(), lang)?;
    println!(
        "text ids: {}",
        if ids == ref_ids {
            "match".into()
        } else {
            format!("DIFFER\n  ref  {ref_ids:?}\n  rust {ids:?}")
        }
    );

    let voice = Voice {
        cond: r["cond"].clone(),
        spk: r["spk"].clone(),
    };

    // 2. Greedy codes.
    let opts = SamplingOptions {
        do_sample: false,
        repetition_penalty: model.config.repetition_penalty,
        ..Default::default()
    };
    let t0 = std::time::Instant::now();
    let out = model.generate_codes(&ref_ids, &voice, &opts, 602, |_, _| true)?;
    let gen_s = t0.elapsed().as_secs_f64();
    let ref_codes: Vec<u32> = r["codes"].to_dtype(DType::U32)?.to_vec1()?;
    let mut codes = out.codes.clone();
    if out.stopped {
        codes.push(model.config.model_args.gpt_stop_audio_token);
    }
    let first_diff = codes.iter().zip(&ref_codes).position(|(a, b)| a != b);
    let same = codes == ref_codes;
    println!(
        "codes: {} ({} rust / {} ref{}) in {:.2} s, {:.1} codes/s",
        if same { "match" } else { "DIFFER" },
        codes.len(),
        ref_codes.len(),
        first_diff
            .map(|i| format!(", first diff at {i}"))
            .unwrap_or_default(),
        gen_s,
        codes.len() as f64 / gen_s
    );

    // 3. Latents: when codes match, compare directly.
    let ref_lat = &r["latents"];
    if same {
        let d = max_abs(&out.latents, ref_lat)?;
        let scale = ref_lat.abs()?.flatten_all()?.max(0)?.to_scalar::<f32>()?;
        println!("latents: max |diff| {d:.2e} (max |ref| {scale:.2})");
    }

    // 4. Decoder on the reference latents.
    let t0 = std::time::Instant::now();
    let wav = model.decode(ref_lat, &voice, 1.0)?;
    let dec_s = t0.elapsed().as_secs_f64();
    let ref_wav: Vec<f32> = r["wav"].to_vec1()?;
    let n = wav.len().min(ref_wav.len());
    let maxd = (0..n)
        .map(|i| (wav[i] - ref_wav[i]).abs())
        .fold(0f32, f32::max);
    let err: f64 = (0..n).map(|i| ((wav[i] - ref_wav[i]) as f64).powi(2)).sum();
    let sig: f64 = ref_wav[..n].iter().map(|x| (*x as f64).powi(2)).sum();
    println!(
        "decoder: {} samples (ref {}), max |diff| {maxd:.2e}, SNR {:.1} dB, {:.2} s",
        wav.len(),
        ref_wav.len(),
        10.0 * (sig / err.max(1e-20)).log10(),
        dec_s
    );
    let _ = ref_lat.i((0, 0))?;

    // Full Rust output for listening.
    let wav = model.decode(&out.latents, &voice, 1.0)?;
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate: model.sample_rate(),
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    let mut w = hound::WavWriter::create("parity.wav", spec)?;
    for s in wav {
        w.write_sample((s.clamp(-1.0, 1.0) * 32767.0) as i16)?;
    }
    w.finalize()?;
    Ok(())
}
