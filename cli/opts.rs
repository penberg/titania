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
    Run(RunCmd),
}

/// Chat with a model, fetching it first if needed
#[derive(FromArgs)]
#[argh(subcommand, name = "run")]
pub struct RunCmd {
    /// model to run (default: qwen3-0.6b)
    #[argh(positional, default = "models::DEFAULT.to_string()")]
    pub model: String,

    /// device to run the model on: cpu (default), or sim for the Titania ISA
    /// simulator
    #[argh(option, default = "Device::Cpu")]
    pub device: Device,
}

/// Where to run the model.
#[derive(Clone, Copy)]
pub enum Device {
    Cpu,
    Sim,
}

impl argh::FromArgValue for Device {
    fn from_arg_value(value: &str) -> Result<Self, String> {
        match value {
            "cpu" => Ok(Device::Cpu),
            "sim" => Ok(Device::Sim),
            _ => Err(format!("unknown device '{value}' (expected cpu or sim)")),
        }
    }
}
