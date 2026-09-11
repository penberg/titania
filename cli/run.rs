use std::error::Error;
use std::io::{self, BufRead, Write};
use std::path::Path;
use std::time::{Instant, SystemTime};

use titania_model::{Chat, Cpu, Model, Sampler, Tokenizer};
use titania_runtime::Titania;

use crate::fetch;
use crate::opts::Device;

/// Longest conversation, in tokens, the key/value cache has room for.
const MAX_LEN: usize = 4096;

/// Chats with a model on the command line, fetching it first if needed.
pub fn run(name: &str, device: Device) -> Result<(), Box<dyn Error>> {
    let dir = fetch::fetch(name)?;
    match device {
        Device::Cpu => chat(name, &dir, Cpu),
        Device::Sim => chat(name, &dir, Titania::new()),
    }
}

fn chat<D: titania_model::Device>(name: &str, dir: &Path, device: D) -> Result<(), Box<dyn Error>> {
    eprintln!("Loading {name}...");
    let start = Instant::now();
    let model = Model::load(dir, device)?;
    let tokenizer = Tokenizer::load(&dir.join("tokenizer.json"))?;
    eprintln!("Loaded in {:.1}s.", start.elapsed().as_secs_f32());

    // Qwen3's recommended sampling settings for replies without thinking.
    let sampler = Sampler::new(0.7, 20, 0.8, seed());
    let mut chat = Chat::new(model, tokenizer, sampler, MAX_LEN)?;

    let mut lines = io::stdin().lock().lines();
    loop {
        print!("\n> ");
        io::stdout().flush()?;
        let Some(line) = lines.next() else {
            println!();
            return Ok(());
        };
        let line = line?;
        let message = line.trim();
        if message.is_empty() {
            continue;
        }
        println!();
        chat.send(message, |text| {
            print!("{text}");
            let _ = io::stdout().flush();
        })?;
        println!();
    }
}

/// Seed for sampling, different on every run.
fn seed() -> u64 {
    SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .map(|elapsed| elapsed.as_nanos() as u64)
        .unwrap_or(1)
}
