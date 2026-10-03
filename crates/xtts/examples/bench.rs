//! Latency and speed: time to first audio (streaming), real-time factor,
//! and how close windowed decoding is to one decoder pass.
//!
//! cargo run --release --example bench -- <model> <lang> "<text>" [--cpu] [--f16] [--runs N]
//!     [--first N] [--chunk N] [--context N] [--lookahead N]

use anyhow::Result;
use candle_core::{DType, Device};
use std::path::Path;
use std::time::Instant;
use xtts::hifigan::StreamDecoder;
use xtts::{SamplingOptions, StreamOptions, SynthOptions, Xtts};

fn arg<T: std::str::FromStr>(args: &[String], name: &str, default: T) -> T {
    args.iter()
        .position(|a| a == name)
        .and_then(|i| args.get(i + 1))
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

fn snr(a: &[f32], b: &[f32]) -> f64 {
    let n = a.len().min(b.len());
    let err: f64 = (0..n).map(|i| ((a[i] - b[i]) as f64).powi(2)).sum();
    let sig: f64 = b[..n].iter().map(|x| (*x as f64).powi(2)).sum();
    10.0 * (sig / err.max(1e-20)).log10()
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let cpu = args.iter().any(|a| a == "--cpu");
    let device = if cpu {
        Device::Cpu
    } else {
        Device::cuda_if_available(0)?
    };
    let dtype = if args.iter().any(|a| a == "--f16") {
        DType::F16
    } else {
        DType::F32
    };
    let t0 = Instant::now();
    let model = Xtts::load(Path::new(&args[1]), &device, dtype)?;
    println!("load: {:.2} s", t0.elapsed().as_secs_f64());
    let (lang, text) = (&args[2], &args[3]);
    let runs: usize = arg(&args, "--runs", 3);
    let stream = StreamOptions {
        first_chunk: arg(&args, "--first", 8),
        chunk: arg(&args, "--chunk", 16),
    };
    let voice = model.voice(&arg(&args, "--voice", "Claribel Dervla".to_string()))?;
    let rate = model.sample_rate() as f64;

    let mut opts = SynthOptions::default();
    opts.sampling.seed = Some(42);
    model.synthesize("Warm up.", "en", &voice, &opts)?;

    for run in 0..runs {
        let t0 = Instant::now();
        let mut first = None;
        let mut samples = 0usize;
        let mut chunks = 0;
        model.synthesize_stream(text, lang, &voice, &opts, &stream, |a| {
            first.get_or_insert_with(|| t0.elapsed().as_secs_f64());
            samples += a.len();
            chunks += 1;
            true
        })?;
        let total = t0.elapsed().as_secs_f64();
        let dur = samples as f64 / rate;
        println!(
            "run {run}: first audio {:.0} ms, {dur:.2} s audio in {total:.2} s ({:.1}x RT), {chunks} chunks",
            first.unwrap_or(0.0) * 1e3,
            dur / total
        );
    }

    // Windowed vs. one-pass decoding of the same latents.
    let greedy = SamplingOptions {
        do_sample: false,
        ..Default::default()
    };
    let chunks = model.prepare_text(text, lang, &opts);
    let ids = model.tokenizer.encode(&chunks[0], lang)?;
    let codes = model.generate_codes(&ids, &voice, &greedy, 602, |_, _| true)?;
    let full = model.decoder.forward(&codes.latents, &voice.spk)?;
    let mut dec = StreamDecoder::new(&model.decoder, &voice.spk, 1.0);
    dec.context = arg(&args, "--context", dec.context);
    dec.lookahead = arg(&args, "--lookahead", dec.lookahead);
    let lat = codes.latents.squeeze(0)?;
    let mut out = Vec::new();
    let n = lat.dim(0)?;
    let mut i = 0;
    while i < n {
        let k = if i == 0 {
            stream.first_chunk
        } else {
            stream.chunk
        }
        .min(n - i);
        dec.push(&lat.narrow(0, i, k)?)?;
        i += k;
        out.extend(dec.decode(i == n)?);
    }
    println!(
        "windowed decoding (context {}, lookahead {}): {} vs {} samples, SNR {:.1} dB vs one pass",
        dec.context,
        dec.lookahead,
        out.len(),
        full.len(),
        snr(&out, &full)
    );
    Ok(())
}
