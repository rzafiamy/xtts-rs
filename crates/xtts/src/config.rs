//! The parts of Coqui's `config.json` the model needs.

use anyhow::Result;
use serde::Deserialize;

#[derive(Debug, Clone, Deserialize)]
pub struct ModelArgs {
    pub gpt_max_audio_tokens: usize,
    pub gpt_max_text_tokens: usize,
    pub gpt_layers: usize,
    pub gpt_n_model_channels: usize,
    pub gpt_n_heads: usize,
    pub gpt_num_audio_tokens: usize,
    pub gpt_start_audio_token: u32,
    pub gpt_stop_audio_token: u32,
    pub gpt_code_stride_len: usize,
    pub input_sample_rate: usize,
    pub output_sample_rate: usize,
    pub output_hop_length: usize,
    pub d_vector_dim: usize,
}

#[derive(Debug, Clone, Deserialize)]
pub struct Config {
    pub model_args: ModelArgs,
    pub languages: Vec<String>,
    pub temperature: f32,
    pub repetition_penalty: f32,
    pub top_k: usize,
    pub top_p: f32,
    pub gpt_cond_len: usize,
    pub gpt_cond_chunk_len: usize,
}

impl Config {
    pub fn from_json(s: &str) -> Result<Self> {
        // Coqui writes `Infinity`, which is not JSON.
        let s = s.replace("Infinity", "null");
        Ok(serde_json::from_str(&s)?)
    }
}
