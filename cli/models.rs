use std::path::PathBuf;

/// Model used when none is named on the command line.
pub const DEFAULT: &str = "qwen3-0.6b";

/// Models Titania knows how to fetch.
pub const MODELS: &[Model] = &[Model {
    name: "qwen3-0.6b",
    repo: "Qwen/Qwen3-0.6B",
    revision: "c1899de289a04d12100db370d81485cdf75e47ca",
    files: &[
        "LICENSE",
        "config.json",
        "generation_config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "model.safetensors",
    ],
}];

/// A model hosted on Hugging Face.
///
/// The revision pins a commit, so the weights can't change underneath us and
/// outputs stay reproducible.
pub struct Model {
    pub name: &'static str,
    pub repo: &'static str,
    pub revision: &'static str,
    pub files: &'static [&'static str],
}

impl Model {
    /// Looks up a model by name.
    pub fn find(name: &str) -> Option<&'static Model> {
        MODELS.iter().find(|model| model.name == name)
    }

    /// URL to download one of the model's files from.
    pub fn url(&self, file: &str) -> String {
        format!(
            "https://huggingface.co/{}/resolve/{}/{}",
            self.repo, self.revision, file
        )
    }

    /// Local directory the model's files are stored in, under the platform's
    /// data directory (`~/.local/share` on Linux, `~/Library/Application
    /// Support` on macOS).
    pub fn dir(&self) -> Option<PathBuf> {
        dirs::data_dir().map(|dir| dir.join("titania").join("models").join(self.name))
    }

    /// How many of the model's files are present locally.
    pub fn status(&self) -> Status {
        let Some(dir) = self.dir() else {
            return Status::Missing;
        };
        let present = self
            .files
            .iter()
            .filter(|file| dir.join(file).exists())
            .count();
        match present {
            0 => Status::Missing,
            n if n == self.files.len() => Status::Downloaded,
            _ => Status::Incomplete,
        }
    }
}

/// Whether a model's files are present locally.
pub enum Status {
    Downloaded,
    Incomplete,
    Missing,
}

impl std::fmt::Display for Status {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let status = match self {
            Status::Downloaded => "downloaded",
            Status::Incomplete => "incomplete",
            Status::Missing => "not downloaded",
        };
        f.pad(status)
    }
}
