use anyhow::{Context, Result};
use candle_core::{DType, Device};
use clap::{Parser, Subcommand};
use std::path::{Path, PathBuf};
use xtts::{SamplingOptions, SynthOptions, Xtts};

mod server;

#[derive(Parser)]
#[command(
    name = "xtts",
    version,
    about = "Coqui XTTS-v2 text-to-speech (Rust/Candle, GGUF)"
)]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(clap::Args, Clone)]
pub struct ModelArgs {
    /// Model: a .gguf file or a Coqui checkpoint directory.
    #[arg(short, long, env = "XTTS_MODEL")]
    model: PathBuf,
    /// Device: cpu, cuda, cuda:N, metal (default: cuda if available).
    #[arg(long, default_value = "auto", env = "XTTS_DEVICE")]
    device: String,
    /// CPU threads (default: all cores).
    #[arg(long, env = "XTTS_THREADS")]
    threads: Option<usize>,
    /// Compute dtype of the GPT's dense layers: f32, f16 or bf16 (default
    /// f32: with quantized weights the activations are F32 anyway, and F16
    /// only adds conversions — 170 vs 307 codes/s on an RTX 4090).
    #[arg(long)]
    dtype: Option<String>,
    /// Folder of `.voice` files to serve besides the built-in voices.
    #[arg(long, env = "XTTS_VOICES_DIR")]
    voices_dir: Option<PathBuf>,
}

#[derive(clap::Args, Clone)]
pub struct GenArgs {
    /// Sampling temperature (XTTS default 0.75).
    #[arg(long)]
    temperature: Option<f32>,
    #[arg(long)]
    top_k: Option<usize>,
    #[arg(long)]
    top_p: Option<f32>,
    /// Repetition penalty (XTTS default 5.0).
    #[arg(long)]
    repetition_penalty: Option<f32>,
    /// Greedy decoding (no sampling).
    #[arg(long)]
    greedy: bool,
    /// Random seed (default: random).
    #[arg(long)]
    seed: Option<u64>,
    /// Stop as soon as the stop code reaches this probability (0 = off).
    #[arg(long, default_value_t = 0.0)]
    stop_prob: f32,
    /// Do not spell out numbers and symbols before synthesis.
    #[arg(long)]
    no_normalize: bool,
    /// Codes per decoder window (~46 ms of audio each): smaller windows
    /// use less GPU memory, larger ones are slightly faster.
    #[arg(long, default_value_t = 48)]
    decode_chunk: usize,
}

impl GenArgs {
    fn sampling(&self, model: &Xtts) -> SamplingOptions {
        let c = &model.config;
        SamplingOptions {
            temperature: self.temperature.unwrap_or(c.temperature),
            top_k: self.top_k.unwrap_or(c.top_k),
            top_p: self.top_p.unwrap_or(c.top_p),
            repetition_penalty: self.repetition_penalty.unwrap_or(c.repetition_penalty),
            do_sample: !self.greedy,
            seed: self.seed,
            stop_prob: self.stop_prob,
        }
    }
}

/// One voice of `xtts clone --manifest`.
#[derive(serde::Deserialize)]
struct CloneJob {
    name: String,
    out: PathBuf,
    files: Vec<PathBuf>,
}

#[derive(Subcommand)]
enum Cmd {
    /// Convert a Coqui checkpoint directory (model.pth, config.json,
    /// vocab.json, speakers_xtts.pth) to GGUF.
    Convert {
        dir: PathBuf,
        #[arg(short, long)]
        out: PathBuf,
        /// Dtype of the GPT's linear weights and mel head.
        #[arg(long, default_value = "q8_0")]
        gpt_dtype: String,
        /// Dtype of the HiFi-GAN decoder weights (f32, f16, bf16).
        #[arg(long, default_value = "f16")]
        decoder_dtype: String,
        /// Leave out the voice-cloning encoders (125 MB): built-in voices only.
        #[arg(long)]
        no_cloning: bool,
        /// Write only the voice-cloning encoders, for `xtts clone`.
        #[arg(long)]
        only_cloning: bool,
        /// Leave out the built-in voices (they live in `.voice` files).
        #[arg(long)]
        no_voices: bool,
        /// Dtype of the conditioning encoder and perceiver.
        #[arg(long, default_value = "f16")]
        cloning_dtype: String,
        /// Dtype of the speaker encoder (f32, f16, bf16).
        #[arg(long, default_value = "f32")]
        speaker_dtype: String,
    },
    /// Compute a voice from reference audio and write it as a `.voice` file.
    Clone {
        /// Model with the cloning encoders: GGUF or Coqui checkpoint directory.
        #[arg(short, long, env = "XTTS_CLONE_MODEL")]
        model: PathBuf,
        /// Device: cpu, cuda, cuda:N, metal (default: cuda if available).
        #[arg(long, default_value = "auto", env = "XTTS_DEVICE")]
        device: String,
        /// CPU threads (default: all cores).
        #[arg(long, env = "XTTS_THREADS")]
        threads: Option<usize>,
        /// Output file (single voice).
        #[arg(short, long)]
        out: Option<PathBuf>,
        /// Voice name (default: the output file's stem).
        #[arg(short, long)]
        name: Option<String>,
        /// Many voices with one model load: a JSON array of
        /// `{"name": ..., "out": ..., "files": [...]}`.
        #[arg(long, conflicts_with_all = ["out", "name", "wavs"])]
        manifest: Option<PathBuf>,
        /// Seconds kept of each clip.
        #[arg(long, default_value_t = 30)]
        max_ref_length: usize,
        /// Seconds of the concatenated clips the GPT latents see.
        #[arg(long, default_value_t = 30)]
        gpt_cond_len: usize,
        /// Seconds per chunk of the GPT latents.
        #[arg(long, default_value_t = 4)]
        gpt_cond_chunk_len: usize,
        /// Reference clips (WAV), used in this order.
        wavs: Vec<PathBuf>,
    },
    /// Synthesize speech to a WAV file.
    Speak {
        #[command(flatten)]
        model: ModelArgs,
        #[command(flatten)]
        gen_args: GenArgs,
        /// Text to speak ("-" reads stdin).
        text: String,
        /// Language: en, fr, es, de, it, pt, pl, tr, ru, nl, cs, ar, hu, hi.
        #[arg(short, long, default_value = "en")]
        lang: String,
        /// Built-in voice.
        #[arg(short, long, default_value = "Claribel Dervla")]
        voice: String,
        #[arg(long, default_value_t = 1.0)]
        speed: f32,
        #[arg(short, long, default_value = "out.wav")]
        out: PathBuf,
    },
    /// List the built-in voices.
    Voices {
        #[command(flatten)]
        model: ModelArgs,
    },
    /// Serve the HTTP API (/v1/audio/speech, /health).
    Serve {
        #[command(flatten)]
        model: ModelArgs,
        #[command(flatten)]
        gen_args: GenArgs,
        #[arg(long, default_value = "127.0.0.1", env = "XTTS_HOST")]
        host: String,
        #[arg(long, default_value_t = 8091, env = "XTTS_PORT")]
        port: u16,
        /// Default voice.
        #[arg(long, default_value = "Claribel Dervla", env = "XTTS_VOICE")]
        voice: String,
        /// Default language.
        #[arg(long, default_value = "en", env = "XTTS_LANG")]
        lang: String,
        /// Model id reported by /v1/models.
        #[arg(long, default_value = "xtts-v2")]
        model_id: String,
        /// Skip the warmup sentence (the first request is then slower).
        #[arg(long)]
        no_warmup: bool,
    },
}

fn set_device(device: &str, threads: Option<usize>) -> Result<Device> {
    if let Some(n) = threads {
        // Candle's CPU kernels use rayon's global pool.
        unsafe { std::env::set_var("RAYON_NUM_THREADS", n.to_string()) };
    }
    Ok(match device {
        "auto" => Device::cuda_if_available(0)?,
        "cpu" => Device::Cpu,
        "cuda" => Device::new_cuda(0)?,
        "metal" => Device::new_metal(0)?,
        d => match d.strip_prefix("cuda:") {
            Some(n) => Device::new_cuda(n.parse().context("cuda:N")?)?,
            None => anyhow::bail!("unknown device '{d}'"),
        },
    })
}

pub fn load_model(args: &ModelArgs) -> Result<Xtts> {
    let device = set_device(&args.device, args.threads)?;
    let dtype = match args.dtype.as_deref() {
        Some("f32") => DType::F32,
        Some("f16") => DType::F16,
        Some("bf16") => DType::BF16,
        Some(d) => anyhow::bail!("unknown dtype '{d}'"),
        None => DType::F32,
    };
    let t0 = std::time::Instant::now();
    let mut model = Xtts::load(&args.model, &device, dtype)?;
    if let Some(dir) = &args.voices_dir {
        let t0 = std::time::Instant::now();
        let d = xtts::VoiceDir::open(dir)?;
        tracing::info!(
            "{} voices in {dir:?}, scanned in {:.0} ms",
            d.names().len(),
            t0.elapsed().as_secs_f64() * 1e3
        );
        model.attach_voice_dir(std::sync::Arc::new(d));
    }
    tracing::info!(
        "loaded {:?} on {:?} ({dtype:?}) in {:.1} s",
        args.model,
        device,
        t0.elapsed().as_secs_f64()
    );
    Ok(model)
}

pub fn wav_bytes(samples: &[f32], rate: u32) -> Result<Vec<u8>> {
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate: rate,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    let mut buf = std::io::Cursor::new(Vec::new());
    let mut w = hound::WavWriter::new(&mut buf, spec)?;
    for s in samples {
        w.write_sample((s.clamp(-1.0, 1.0) * 32767.0) as i16)?;
    }
    w.finalize()?;
    Ok(buf.into_inner())
}

fn read_text(text: &str) -> Result<String> {
    if text == "-" {
        let mut s = String::new();
        std::io::Read::read_to_string(&mut std::io::stdin(), &mut s)?;
        Ok(s)
    } else {
        Ok(text.to_string())
    }
}

fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_| "info".into()),
        )
        .with_writer(std::io::stderr)
        .init();
    match Cli::parse().cmd {
        Cmd::Convert {
            dir,
            out,
            gpt_dtype,
            decoder_dtype,
            no_cloning,
            only_cloning,
            no_voices,
            cloning_dtype,
            speaker_dtype,
        } => {
            let opts = xtts::gguf::ConvertOptions {
                gpt_dtype: xtts::gguf::parse_dtype(&gpt_dtype)?,
                decoder_dtype: xtts::gguf::parse_dtype(&decoder_dtype)?,
                no_cloning,
                only_cloning,
                no_voices,
                cloning_dtype: xtts::gguf::parse_dtype(&cloning_dtype)?,
                speaker_dtype: xtts::gguf::parse_dtype(&speaker_dtype)?,
            };
            let counts = xtts::gguf::convert(&dir, &out, &opts)?;
            for (dt, n) in counts {
                println!("{dt}: {n} tensors");
            }
            let size = std::fs::metadata(&out)?.len();
            println!("wrote {} ({:.0} MB)", out.display(), size as f64 / 1e6);
        }
        Cmd::Clone {
            model,
            device,
            threads,
            out,
            name,
            manifest,
            max_ref_length,
            gpt_cond_len,
            gpt_cond_chunk_len,
            wavs,
        } => {
            let device = set_device(&device, threads)?;
            let t0 = std::time::Instant::now();
            let enc = xtts::CloneEncoder::load(&model, &device)?;
            tracing::info!(
                "cloning encoders loaded on {device:?} in {:.2} s",
                t0.elapsed().as_secs_f64()
            );
            let opts = xtts::CloneOptions {
                max_ref_length,
                gpt_cond_len,
                gpt_cond_chunk_len,
            };
            let jobs: Vec<CloneJob> = match manifest {
                Some(m) => {
                    let s =
                        std::fs::read_to_string(&m).with_context(|| format!("reading {m:?}"))?;
                    serde_json::from_str(&s).with_context(|| format!("parsing {m:?}"))?
                }
                None => {
                    let out = out.context("--out or --manifest is required")?;
                    let name = match name {
                        Some(n) => n,
                        None => out
                            .file_stem()
                            .context("--out has no file name")?
                            .to_string_lossy()
                            .into_owned(),
                    };
                    vec![CloneJob {
                        name,
                        out,
                        files: wavs,
                    }]
                }
            };
            for job in &jobs {
                let t0 = std::time::Instant::now();
                let (voice, times) = enc.clone_files(&job.files, &opts)?;
                let secs = t0.elapsed().as_secs_f64();
                xtts::voicefile::write_voice(&job.out, &job.name, &voice)?;
                // One JSON line per voice on stdout.
                println!(
                    "{}",
                    serde_json::json!({
                        "name": job.name,
                        "out": job.out,
                        "clips": job.files.len(),
                        "seconds": secs,
                        "audio_s": times.audio,
                        "speaker_s": times.speaker,
                        "cond_s": times.cond,
                    })
                );
            }
        }
        Cmd::Speak {
            model,
            gen_args,
            text,
            lang,
            voice,
            speed,
            out,
        } => {
            let m = load_model(&model)?;
            let v = m.voice(&voice)?;
            let opts = SynthOptions {
                sampling: gen_args.sampling(&m),
                normalize: !gen_args.no_normalize,
                decode_chunk: gen_args.decode_chunk,
                speed,
                ..Default::default()
            };
            let text = read_text(&text)?;
            let t0 = std::time::Instant::now();
            let wav = m.synthesize(&text, &lang, &v, &opts)?;
            let secs = t0.elapsed().as_secs_f64();
            let dur = wav.len() as f64 / m.sample_rate() as f64;
            write_file(&out, &wav_bytes(&wav, m.sample_rate())?)?;
            eprintln!(
                "{}: {dur:.2} s of audio in {secs:.2} s ({:.1}x real time)",
                out.display(),
                dur / secs
            );
        }
        Cmd::Voices { model } => {
            let m = load_model(&model)?;
            for v in m.voices() {
                println!("{v}");
            }
        }
        Cmd::Serve {
            model,
            gen_args,
            host,
            port,
            voice,
            lang,
            model_id,
            no_warmup,
        } => {
            let m = load_model(&model)?;
            // A model without voices relies on the folder, which may even be
            // empty on a first run: the default voice is then whatever is there.
            let voice = match m.voice(&voice) {
                Ok(_) => Some(voice),
                Err(e) if model.voices_dir.is_none() => return Err(e),
                Err(_) => {
                    let first = m.voices().into_iter().next();
                    tracing::warn!("voice '{voice}' not found, default voice: {first:?}");
                    first
                }
            };
            let sampling = gen_args.sampling(&m);
            server::serve(server::Settings {
                model: m,
                host,
                port,
                voice,
                lang,
                model_id,
                sampling,
                normalize: !gen_args.no_normalize,
                decode_chunk: gen_args.decode_chunk,
                warmup: !no_warmup,
            })?;
        }
    }
    Ok(())
}

fn write_file(path: &Path, bytes: &[u8]) -> Result<()> {
    std::fs::write(path, bytes).with_context(|| format!("writing {path:?}"))
}
