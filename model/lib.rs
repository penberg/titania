//! A decoder-only transformer language model, written as a sequence of
//! operations on a [`Device`].

mod chat;
mod config;
mod cpu;
mod device;
mod sampler;
mod tokenizer;
mod transformer;
mod weights;

pub use chat::Chat;
pub use config::Config;
pub use cpu::Cpu;
pub use device::Device;
pub use sampler::Sampler;
pub use tokenizer::Tokenizer;
pub use transformer::{Model, State};
pub use weights::{Tensor, Weights};

pub type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;
