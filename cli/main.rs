mod fetch;
mod models;
mod opts;

use opts::{Cmd, Opts};

fn main() {
    let opts: Opts = argh::from_env();

    let result = match opts.command {
        Cmd::Fetch(cmd) => fetch::fetch(&cmd.model),
    };

    if let Err(e) = result {
        eprintln!("error: {e}");
        std::process::exit(1);
    }
}
