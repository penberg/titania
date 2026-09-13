mod fetch;
mod harness;
mod logo;
mod models;
mod monitor;
mod opts;
mod run;
mod tui;

use opts::{Cmd, Opts};

fn main() {
    let opts: Opts = argh::from_env();

    let result = match opts.command {
        Cmd::Run(cmd) => run::run(&cmd.model, cmd.device),
    };

    if let Err(e) = result {
        eprintln!("error: {e}");
        std::process::exit(1);
    }
}
