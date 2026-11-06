//! Times each stage of voice cloning on one clip.
//!
//! clone_prof <model> <cpu|cuda> <threads> a.wav

use anyhow::Result;
use candle_core::{Device, Tensor};
use std::time::Instant;
use xtts::{CloneEncoder, CloneOptions, audio};

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    unsafe { std::env::set_var("RAYON_NUM_THREADS", &args[3]) };
    let device = if args[2] == "cpu" {
        Device::Cpu
    } else {
        Device::new_cuda(0)?
    };
    let t = Instant::now();
    let enc = CloneEncoder::load(args[1].as_ref(), &device)?;
    println!("load        {:7.3}", t.elapsed().as_secs_f64());
    let o = CloneOptions::default();
    for round in 0..3 {
        let t = Instant::now();
        let clip = enc.load_clip(args[4].as_ref(), &o)?;
        let a = t.elapsed().as_secs_f64();
        let chunk = &clip[..(22050 * 4).min(clip.len())];
        let t = Instant::now();
        let mel = enc.gpt_mel(chunk)?;
        let b = t.elapsed().as_secs_f64();
        let (n, m) = (mel.len(), mel[0].len());
        let mel = Tensor::from_vec(mel.concat(), (n, m), &device)?;
        let t = Instant::now();
        let c = enc.cond_encoder(&mel)?;
        let _ = c.to_device(&Device::Cpu)?;
        let d = t.elapsed().as_secs_f64();
        let t = Instant::now();
        let p = enc.perceiver(&c)?;
        let _ = p.to_device(&Device::Cpu)?;
        let e = t.elapsed().as_secs_f64();
        let t = Instant::now();
        let w16 = audio::resample(&clip, 22050, 16000);
        let f = t.elapsed().as_secs_f64();
        let t = Instant::now();
        let feats = enc.speaker_features(&w16)?;
        let g = t.elapsed().as_secs_f64();
        let (n, m) = (feats.len(), feats[0].len());
        let feats = Tensor::from_vec(feats.concat(), (n, m), &device)?;
        let t = Instant::now();
        let s = enc.speaker_forward(&feats)?;
        let _ = s.to_device(&Device::Cpu)?;
        let h = t.elapsed().as_secs_f64();
        println!(
            "round {round}: clip {:.3} s, read+resample {a:.3}, mel {b:.3}, cond_encoder {d:.3}, perceiver {e:.3}, \
             resample16k {f:.3}, spk features {g:.3}, spk net ({n} frames) {h:.3}",
            clip.len() as f64 / 22050.0
        );
    }
    Ok(())
}
