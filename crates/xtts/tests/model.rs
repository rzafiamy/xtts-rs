//! Tests on a real model (CPU). They need a GGUF written by `xtts convert`:
//!
//!     XTTS_MODEL=models/xtts-v2-q4k.gguf cargo test --release -p xtts --test model
//!
//! Without `XTTS_MODEL` each test prints a note and passes (CI has no weights).

use candle_core::{DType, Device};
use std::sync::OnceLock;
use xtts::{SamplingOptions, StreamOptions, SynthOptions, Xtts};

fn model() -> Option<&'static Xtts> {
    static MODEL: OnceLock<Option<Xtts>> = OnceLock::new();
    MODEL
        .get_or_init(|| {
            let path = std::env::var("XTTS_MODEL").ok()?;
            Some(Xtts::load(path.as_ref(), &Device::Cpu, DType::F32).expect("loading XTTS_MODEL"))
        })
        .as_ref()
}

macro_rules! need_model {
    () => {
        match model() {
            Some(m) => m,
            None => {
                eprintln!("XTTS_MODEL not set: skipped");
                return;
            }
        }
    };
}

fn greedy() -> SynthOptions {
    SynthOptions {
        sampling: SamplingOptions {
            do_sample: false,
            seed: Some(0),
            ..Default::default()
        },
        ..Default::default()
    }
}

/// covers: REQ-GGF-001, REQ-VOI-001
#[test]
fn gguf_has_builtin_voices() {
    let m = need_model!();
    assert_eq!(m.voices().len(), 58);
    assert_eq!(m.sample_rate(), 24000);
    // Names match ignoring case, `_` and `-`.
    let a = m.voice("Ana Florence").unwrap();
    let b = m.voice("ana_florence").unwrap();
    assert_eq!(
        a.spk.to_vec1::<f32>().unwrap(),
        b.spk.to_vec1::<f32>().unwrap()
    );
    assert!(m.voice("alloy").is_err());
}

/// covers: REQ-INF-002, REQ-TXT-004
#[test]
fn short_sentence_stops_after_the_last_word() {
    let m = need_model!();
    let voice = m.voice("Ana Florence").unwrap();
    for (text, lang) in [("Bonjour à tous.", "fr"), ("Hello, how are you?", "en")] {
        let wav = m.synthesize(text, lang, &voice, &greedy()).unwrap();
        let secs = wav.len() as f32 / 24000.0;
        // ~1 s of speech; tail babble would push it past 2.5 s.
        assert!((0.5..2.5).contains(&secs), "{text}: {secs:.2} s");
        let peak = wav.iter().fold(0f32, |p, s| p.max(s.abs()));
        assert!(peak > 0.05, "{text}: silent output");
    }
}

/// covers: REQ-STR-001
#[test]
fn streamed_audio_matches_one_pass() {
    let m = need_model!();
    let voice = m.voice("Claribel Dervla").unwrap();
    let text = "Le streaming produit le même signal.";
    let full = m.synthesize(text, "fr", &voice, &greedy()).unwrap();
    let mut streamed = Vec::new();
    let mut chunks = 0;
    m.synthesize_stream(
        text,
        "fr",
        &voice,
        &greedy(),
        &StreamOptions::default(),
        |a| {
            streamed.extend_from_slice(a);
            chunks += 1;
            true
        },
    )
    .unwrap();
    assert!(chunks > 1, "audio came in one piece");
    assert_eq!(streamed.len(), full.len());
    let (mut sig, mut err) = (0f64, 0f64);
    for (a, b) in full.iter().zip(&streamed) {
        sig += (*a as f64).powi(2);
        err += (*a as f64 - *b as f64).powi(2);
    }
    let snr = 10.0 * (sig / err.max(1e-12)).log10();
    assert!(snr > 40.0, "stream vs one pass: {snr:.1} dB");
}

/// covers: REQ-TXT-001
#[test]
fn unsupported_language_is_an_error() {
    let m = need_model!();
    let voice = m.voice("Ana Florence").unwrap();
    let err = m
        .synthesize("你好", "zh-cn", &voice, &greedy())
        .unwrap_err();
    assert!(err.to_string().contains("not supported"), "{err}");
}

/// covers: REQ-INF-003
#[test]
fn speed_changes_duration() {
    let m = need_model!();
    let voice = m.voice("Ana Florence").unwrap();
    let text = "Une phrase un peu plus longue pour mesurer la vitesse.";
    let normal = m.synthesize(text, "fr", &voice, &greedy()).unwrap().len() as f32;
    let fast = SynthOptions {
        speed: 1.5,
        ..greedy()
    };
    let quick = m.synthesize(text, "fr", &voice, &fast).unwrap().len() as f32;
    let ratio = normal / quick;
    assert!((1.3..1.7).contains(&ratio), "speed 1.5 → ratio {ratio:.2}");
}
