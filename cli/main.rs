mod fetch;
mod list;
mod models;
mod opts;
mod run;

use std::path::Path;

use argh::FromArgs;
use opts::{Cmd, Opts};

fn main() {
    let opts = parse_args();

    let result = match opts.command {
        Cmd::Fetch(cmd) => fetch::fetch(&cmd.model).map(|dir| println!("{}", dir.display())),
        Cmd::Models(_) => list::list(),
        Cmd::Run(cmd) => run::run(&cmd.model, cmd.device),
    };

    if let Err(e) = result {
        eprintln!("error: {e}");
        std::process::exit(1);
    }
}

/// Parses the command line like `argh::from_env`, but also accepts options
/// written as `--option=value`, which argh doesn't.
fn parse_args() -> Opts {
    let args: Vec<String> = std::env::args()
        .flat_map(|arg| match arg.split_once('=') {
            Some((option, value)) if option.starts_with("--") => vec![option.to_string(), value.to_string()],
            _ => vec![arg],
        })
        .collect();
    let cmd = Path::new(&args[0])
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("titania");
    let rest: Vec<&str> = args[1..].iter().map(String::as_str).collect();
    Opts::from_args(&[cmd], &rest).unwrap_or_else(|exit| match exit.status {
        Ok(()) => {
            println!("{}", exit.output);
            std::process::exit(0);
        }
        Err(()) => {
            eprintln!("{}\nRun {cmd} --help for more information.", exit.output);
            std::process::exit(1);
        }
    })
}
