//! Compares each stage of `xtts clone` with a dump of scripts/clone_ref.py.
//!
//! clone_parity <model> <ref.safetensors> <cpu|cuda> a.wav b.wav ...

use anyhow::Result;
use candle_core::{Device, Tensor};
use std::path::PathBuf;
use xtts::{CloneEncoder, CloneOptions, audio};

fn report(name: &str, a: &Tensor, b: &Tensor) -> Result<()> {
    let a = a.flatten_all()?.to_device(&Device::Cpu)?.to_vec1::<f32>()?;
    let b = b.flatten_all()?.to_device(&Device::Cpu)?.to_vec1::<f32>()?;
    if a.len() != b.len() {
        println!("{name:10} SIZES DIFFER: {} vs {}", a.len(), b.len());
        return Ok(());
    }
    let (mut ab, mut aa, mut bb, mut max) = (0f64, 0f64, 0f64, 0f64);
    for (x, y) in a.iter().zip(&b) {
        ab += (*x as f64) * (*y as f64);
        aa += (*x as f64) * (*x as f64);
        bb += (*y as f64) * (*y as f64);
        max = max.max((*x as f64 - *y as f64).abs());
    }
    let peak = b.iter().fold(0f32, |m, v| m.max(v.abs()));
    println!(
        "{name:10} cos-1 {:+.3e}  max|diff| {max:.3e}  (peak {peak:.3})",
        ab / (aa * bb).sqrt() - 1.0
    );
    Ok(())
}

fn tensor1(v: Vec<f32>) -> Result<Tensor> {
    let n = v.len();
    Ok(Tensor::from_vec(v, n, &Device::Cpu)?)
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let device = if args[3] == "cpu" {
        Device::Cpu
    } else {
        Device::new_cuda(0)?
    };
    let enc = CloneEncoder::load(args[1].as_ref(), &device)?;
    let r = candle_core::safetensors::load(&args[2], &device)?;
    let wavs: Vec<PathBuf> = args[4..].iter().map(PathBuf::from).collect();
    let o = CloneOptions::default();

    // Audio front end, compared with PyTorch's.
    let clips: Vec<Vec<f32>> = wavs
        .iter()
        .map(|p| enc.load_clip(p, &o))
        .collect::<Result<_>>()?;
    let first = clips[0].clone();
    let full: Vec<f32> = clips.iter().flatten().copied().collect();
    report("wav22k", &tensor1(full.clone())?, &r["wav22k"])?;
    let w16 = audio::resample(&first, 22050, 16000);
    report("wav16k0", &tensor1(w16.clone())?, &r["wav16k0"])?;
    let chunk = &full[..(22050 * 4).min(full.len())];
    let mel = enc.gpt_mel(chunk)?;
    let (t, m) = (mel.len(), mel[0].len());
    let mel_t = Tensor::from_vec(mel.concat(), (t, m), &device)?;
    report("mel0", &mel_t.t()?, &r["mel0"])?;
    let feats = enc.speaker_features(&w16)?;
    let (t, m) = (feats.len(), feats[0].len());
    let feats_t = Tensor::from_vec(feats.concat(), (t, m), &device)?;
    report("spkfeat0", &feats_t.t()?, &r["spkfeat0"])?;

    // Encoders on PyTorch's inputs, so their errors are only their own.
    let ref_mel = r["mel0"].t()?.contiguous()?;
    let c = enc.cond_encoder(&ref_mel)?;
    report("cond0", &c.t()?, &r["cond0"])?;
    let ref_c = r["cond0"].t()?.contiguous()?;
    report("perc0", &enc.perceiver(&ref_c)?, &r["perc0"])?;
    let ref_f = r["spkfeat0"].t()?.contiguous()?;
    report("spk0", &enc.speaker_forward(&ref_f)?, &r["spk0"])?;

    // End to end.
    let (v, _) = enc.clone_files(&wavs, &o)?;
    report("cond", &v.cond, &r["cond"])?;
    report("spk", &v.spk, &r["spk"])?;
    Ok(())
}
