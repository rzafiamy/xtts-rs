//! HTTP API: `POST /v1/audio/speech` (OpenAI shape plus `language` and
//! `stream`), `POST /stream` (pocket-tts-server's streaming contract:
//! `{text, voice}` in, raw 24 kHz PCM16 out as it is generated),
//! `GET /v1/voices`, `GET /v1/models`, `GET /health`. One synthesis at a
//! time (the model is behind a mutex); the listener binds only after the
//! model is loaded and (unless disabled) a warmup sentence was spoken.

use anyhow::Result;
use axum::body::{Body, Bytes};
use axum::extract::State;
use axum::http::{StatusCode, header};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use serde::Deserialize;
use serde_json::json;
use std::sync::Arc;
use tokio::sync::Mutex;
use xtts::{SamplingOptions, StreamOptions, SynthOptions, Xtts};

pub struct Settings {
    pub model: Xtts,
    pub host: String,
    pub port: u16,
    pub voice: String,
    pub lang: String,
    pub model_id: String,
    pub sampling: SamplingOptions,
    pub normalize: bool,
    pub decode_chunk: usize,
    pub warmup: bool,
}

struct AppState {
    model: Mutex<Xtts>,
    voice: String,
    lang: String,
    model_id: String,
    sampling: SamplingOptions,
    normalize: bool,
    decode_chunk: usize,
    voices: Vec<String>,
}

#[derive(Deserialize)]
struct SpeechRequest {
    #[serde(alias = "text")]
    input: String,
    voice: Option<String>,
    language: Option<String>,
    speed: Option<f32>,
    /// wav (default) or pcm (16-bit little-endian mono, 24 kHz).
    response_format: Option<String>,
    temperature: Option<f32>,
    top_k: Option<usize>,
    top_p: Option<f32>,
    repetition_penalty: Option<f32>,
    seed: Option<u64>,
    stop_prob: Option<f32>,
    /// Send audio as it is generated (chunked transfer).
    stream: Option<bool>,
}

impl AppState {
    fn options(&self, req: &SpeechRequest) -> SynthOptions {
        let s = &self.sampling;
        SynthOptions {
            sampling: SamplingOptions {
                temperature: req.temperature.unwrap_or(s.temperature),
                top_k: req.top_k.unwrap_or(s.top_k),
                top_p: req.top_p.unwrap_or(s.top_p),
                repetition_penalty: req.repetition_penalty.unwrap_or(s.repetition_penalty),
                do_sample: s.do_sample,
                seed: req.seed.or(s.seed),
                stop_prob: req.stop_prob.unwrap_or(s.stop_prob),
            },
            normalize: self.normalize,
            decode_chunk: self.decode_chunk,
            speed: req.speed.unwrap_or(1.0).clamp(0.25, 4.0),
            ..Default::default()
        }
    }

    /// Checks the voice before a stream starts (errors cannot be reported
    /// once audio is flowing).
    fn check_voice(&self, req: &SpeechRequest) -> Result<(), Box<Response>> {
        let name = req.voice.as_deref().unwrap_or(&self.voice);
        let norm = |s: &str| s.to_lowercase().replace(['_', '-'], " ");
        if self.voices.iter().any(|v| norm(v) == norm(name)) {
            Ok(())
        } else {
            Err(Box::new(error(
                StatusCode::BAD_REQUEST,
                format!("unknown voice '{name}'"),
            )))
        }
    }
}

fn pcm16(samples: &[f32]) -> Vec<u8> {
    samples
        .iter()
        .flat_map(|s| ((s.clamp(-1.0, 1.0) * 32767.0) as i16).to_le_bytes())
        .collect()
}

/// WAV header for a stream of unknown length (sizes set to the maximum,
/// which players read as "until the end").
fn streaming_wav_header(rate: u32) -> Vec<u8> {
    let mut h = Vec::with_capacity(44);
    h.extend_from_slice(b"RIFF");
    h.extend_from_slice(&u32::MAX.to_le_bytes());
    h.extend_from_slice(b"WAVEfmt ");
    h.extend_from_slice(&16u32.to_le_bytes());
    h.extend_from_slice(&1u16.to_le_bytes()); // PCM
    h.extend_from_slice(&1u16.to_le_bytes()); // mono
    h.extend_from_slice(&rate.to_le_bytes());
    h.extend_from_slice(&(rate * 2).to_le_bytes());
    h.extend_from_slice(&2u16.to_le_bytes());
    h.extend_from_slice(&16u16.to_le_bytes());
    h.extend_from_slice(b"data");
    h.extend_from_slice(&u32::MAX.to_le_bytes());
    h
}

/// Streams PCM16 (after a WAV header when `wav`) as chunks are decoded;
/// generation stops when the client goes away.
fn stream_response(st: Arc<AppState>, req: SpeechRequest, wav: bool) -> Response {
    if let Err(r) = st.check_voice(&req) {
        return *r;
    }
    let (tx, rx) = tokio::sync::mpsc::channel::<Result<Bytes, std::io::Error>>(32);
    tokio::task::spawn_blocking(move || {
        let model = st.model.blocking_lock();
        let rate = model.sample_rate();
        if wav
            && tx
                .blocking_send(Ok(Bytes::from(streaming_wav_header(rate))))
                .is_err()
        {
            return;
        }
        let res = (|| -> anyhow::Result<()> {
            let voice = model.voice(req.voice.as_deref().unwrap_or(&st.voice))?;
            let lang = req.language.as_deref().unwrap_or(&st.lang);
            let opts = st.options(&req);
            let t0 = std::time::Instant::now();
            let mut first = None;
            let mut samples = 0usize;
            model.synthesize_stream(
                &req.input,
                lang,
                &voice,
                &opts,
                &StreamOptions::default(),
                |a| {
                    first.get_or_insert_with(|| t0.elapsed().as_secs_f64());
                    samples += a.len();
                    tx.blocking_send(Ok(Bytes::from(pcm16(a)))).is_ok()
                },
            )?;
            tracing::info!(
                "stream: {} chars, first audio {:.0} ms, {:.2} s of audio in {:.2} s",
                req.input.chars().count(),
                first.unwrap_or(0.0) * 1e3,
                samples as f64 / rate as f64,
                t0.elapsed().as_secs_f64()
            );
            Ok(())
        })();
        if let Err(e) = res {
            tracing::warn!("stream: {e}");
            let _ = tx.blocking_send(Err(std::io::Error::other(e.to_string())));
        }
        if let Err(e) = xtts::release_cached_memory(&model.device) {
            tracing::warn!("releasing cached GPU memory: {e}");
        }
    });
    let ctype = if wav {
        "audio/wav"
    } else {
        "application/octet-stream"
    };
    let body = Body::from_stream(tokio_stream::wrappers::ReceiverStream::new(rx));
    ([(header::CONTENT_TYPE, ctype)], body).into_response()
}

/// pocket-tts-server's `/stream`: raw PCM16 at 24 kHz.
async fn stream(State(st): State<Arc<AppState>>, Json(req): Json<SpeechRequest>) -> Response {
    if req.input.trim().is_empty() {
        return error(StatusCode::BAD_REQUEST, "empty text");
    }
    stream_response(st, req, false)
}

fn error(status: StatusCode, msg: impl ToString) -> Response {
    (
        status,
        Json(json!({ "error": { "message": msg.to_string() } })),
    )
        .into_response()
}

async fn speech(State(st): State<Arc<AppState>>, Json(req): Json<SpeechRequest>) -> Response {
    if req.input.trim().is_empty() {
        return error(StatusCode::BAD_REQUEST, "empty input");
    }
    let format = req.response_format.clone().unwrap_or_else(|| "wav".into());
    if format != "wav" && format != "pcm" {
        return error(
            StatusCode::BAD_REQUEST,
            format!("unsupported response_format '{format}' (wav, pcm)"),
        );
    }
    if req.stream == Some(true) {
        return stream_response(st, req, format == "wav");
    }
    let st2 = st.clone();
    let res =
        tokio::task::spawn_blocking(move || -> Result<(Vec<f32>, u32), (StatusCode, String)> {
            let model = st2.model.blocking_lock();
            let name = req.voice.as_deref().unwrap_or(&st2.voice);
            let voice = model
                .voice(name)
                .map_err(|e| (StatusCode::BAD_REQUEST, e.to_string()))?;
            let lang = req.language.as_deref().unwrap_or(&st2.lang);
            let opts = st2.options(&req);
            let t0 = std::time::Instant::now();
            let wav = model
                .synthesize(&req.input, lang, &voice, &opts)
                .map_err(|e| (StatusCode::BAD_REQUEST, e.to_string()))?;
            let rate = model.sample_rate();
            tracing::info!(
                "{} chars, {:.2} s of audio in {:.2} s",
                req.input.chars().count(),
                wav.len() as f64 / rate as f64,
                t0.elapsed().as_secs_f64()
            );
            if let Err(e) = xtts::release_cached_memory(&model.device) {
                tracing::warn!("releasing cached GPU memory: {e}");
            }
            Ok((wav, rate))
        })
        .await;
    let (wav, rate) = match res {
        Ok(Ok(v)) => v,
        Ok(Err((code, msg))) => return error(code, msg),
        Err(e) => return error(StatusCode::INTERNAL_SERVER_ERROR, e),
    };
    if format == "pcm" {
        return ([(header::CONTENT_TYPE, "audio/pcm")], pcm16(&wav)).into_response();
    }
    match crate::wav_bytes(&wav, rate) {
        Ok(b) => ([(header::CONTENT_TYPE, "audio/wav")], b).into_response(),
        Err(e) => error(StatusCode::INTERNAL_SERVER_ERROR, e),
    }
}

async fn voices(State(st): State<Arc<AppState>>) -> Response {
    let model = st.model.lock().await;
    Json(json!({ "voices": model.voices(), "default": st.voice })).into_response()
}

async fn models(State(st): State<Arc<AppState>>) -> Response {
    Json(json!({ "object": "list", "data": [{ "id": st.model_id, "object": "model" }] }))
        .into_response()
}

async fn health() -> Response {
    Json(json!({ "status": "ok" })).into_response()
}

pub fn serve(s: Settings) -> Result<()> {
    // Warmup: CUDA kernels and allocator before the first request.
    let voice = s.model.voice(&s.voice)?;
    if s.warmup {
        let warm = SynthOptions {
            sampling: SamplingOptions {
                seed: Some(0),
                ..s.sampling.clone()
            },
            ..Default::default()
        };
        let t0 = std::time::Instant::now();
        s.model.synthesize("Hello.", "en", &voice, &warm)?;
        tracing::info!("warmup in {:.2} s", t0.elapsed().as_secs_f64());
        xtts::release_cached_memory(&s.model.device)?;
    }

    let addr = format!("{}:{}", s.host, s.port);
    let state = Arc::new(AppState {
        voices: s.model.voices().to_vec(),
        model: Mutex::new(s.model),
        voice: s.voice,
        lang: s.lang,
        model_id: s.model_id,
        sampling: s.sampling,
        normalize: s.normalize,
        decode_chunk: s.decode_chunk,
    });
    let app = Router::new()
        .route("/v1/audio/speech", post(speech))
        .route("/stream", post(stream))
        .route("/v1/voices", get(voices))
        .route("/v1/models", get(models))
        .route("/health", get(health))
        .with_state(state);
    tokio::runtime::Runtime::new()?.block_on(async move {
        let listener = tokio::net::TcpListener::bind(&addr).await?;
        tracing::info!("listening on http://{addr}");
        axum::serve(listener, app).await?;
        Ok(())
    })
}
