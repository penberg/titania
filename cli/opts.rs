use argh::FromArgs;

use crate::models;

/// A complete LLM system, from transformer to transistor
#[derive(FromArgs)]
pub struct Opts {
    #[argh(subcommand)]
    pub command: Cmd,
}

#[derive(FromArgs)]
#[argh(subcommand)]
pub enum Cmd {
    Fetch(FetchCmd),
}

/// Download a model's weights
#[derive(FromArgs)]
#[argh(subcommand, name = "fetch")]
pub struct FetchCmd {
    /// model to fetch (default: qwen3-0.6b)
    #[argh(positional, default = "models::DEFAULT.to_string()")]
    pub model: String,
}
