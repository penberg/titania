use std::error::Error;
use std::fs::{self, File};
use std::io::{self, Read, Write};
use std::path::Path;

use crate::models::{MODELS, Model};

/// Downloads a model's files into its local directory, skipping any that are
/// already there, and prints the directory on stdout.
pub fn fetch(name: &str) -> Result<(), Box<dyn Error>> {
    let model = Model::find(name).ok_or_else(|| {
        let known: Vec<_> = MODELS.iter().map(|model| model.name).collect();
        format!("unknown model '{name}' (known: {})", known.join(", "))
    })?;
    let dir = model.dir().ok_or("cannot determine the data directory")?;
    fs::create_dir_all(&dir)?;

    for file in model.files {
        let path = dir.join(file);
        if path.exists() {
            continue;
        }
        download(model, file, &path)?;
    }

    println!("{}", dir.display());
    Ok(())
}

/// Downloads one of the model's files to `path`, reporting progress on stderr.
///
/// The file is written under a temporary name and renamed into place once
/// complete, so an interrupted download never looks like a finished one.
fn download(model: &Model, file: &str, path: &Path) -> Result<(), Box<dyn Error>> {
    let response = ureq::get(&model.url(file)).call()?;
    let total = response
        .headers()
        .get("content-length")
        .and_then(|value| value.to_str().ok())
        .and_then(|value| value.parse::<u64>().ok());

    let partial = path.with_file_name(format!("{file}.part"));
    let mut reader = response.into_body().into_reader();
    let mut writer = File::create(&partial)?;
    let mut buf = vec![0; 1 << 20];
    let mut done = 0;
    loop {
        let n = reader.read(&mut buf)?;
        if n == 0 {
            break;
        }
        writer.write_all(&buf[..n])?;
        done += n as u64;
        progress(model.name, file, done, total);
    }
    eprintln!();
    fs::rename(&partial, path)?;
    Ok(())
}

fn progress(model: &str, file: &str, done: u64, total: Option<u64>) {
    let line = match total {
        Some(total) => format!(
            "{model}: {file} {} / {} ({}%)",
            size(done),
            size(total),
            done * 100 / total.max(1)
        ),
        None => format!("{model}: {file} {}", size(done)),
    };
    eprint!("\r\x1b[2K{line}");
    let _ = io::stderr().flush();
}

/// Formats a byte count for humans, in decimal units.
fn size(bytes: u64) -> String {
    const UNITS: [&str; 4] = ["KB", "MB", "GB", "TB"];
    if bytes < 1000 {
        return format!("{bytes} B");
    }
    let mut value = bytes as f64 / 1000.0;
    let mut unit = 0;
    while value >= 1000.0 && unit < UNITS.len() - 1 {
        value /= 1000.0;
        unit += 1;
    }
    format!("{value:.1} {}", UNITS[unit])
}
