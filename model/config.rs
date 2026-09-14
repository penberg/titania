use std::{fs, path::Path};

use serde::Deserialize;

use crate::Result;

/// Architecture of a decoder-only transformer, read from a Hugging Face
/// `config.json`.
#[derive(Debug, Deserialize)]
pub struct Config {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    /// Width of each attention head. Qwen3 sets it explicitly; Llama models
    /// leave it to be derived from the hidden size.
    head_dim: Option<usize>,
    pub rms_norm_eps: f32,
    pub rope_theta: f32,
    pub vocab_size: usize,
    /// Whether the output projection shares the token embedding table.
    #[serde(default)]
    pub tie_word_embeddings: bool,
}

impl Config {
    pub fn load(path: &Path) -> Result<Self> {
        Ok(serde_json::from_str(&fs::read_to_string(path)?)?)
    }

    pub fn head_dim(&self) -> usize {
        self.head_dim
            .unwrap_or(self.hidden_size / self.num_attention_heads)
    }
}
