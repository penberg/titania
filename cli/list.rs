use std::error::Error;

use crate::models::MODELS;

/// Prints the models Titania knows about, whether each one is downloaded, and
/// where its files live.
pub fn list() -> Result<(), Box<dyn Error>> {
    let width = MODELS
        .iter()
        .map(|model| model.name.len())
        .max()
        .unwrap_or(0)
        .max("MODEL".len());

    println!("{:<width$}  {:<14}  PATH", "MODEL", "STATUS");
    for model in MODELS {
        let path = model
            .dir()
            .map(|dir| dir.display().to_string())
            .unwrap_or_default();
        println!("{:<width$}  {:<14}  {path}", model.name, model.status());
    }
    Ok(())
}
