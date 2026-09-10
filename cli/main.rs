mod fetch;
mod list;
mod models;
mod opts;
mod run;

use opts::{Cmd, Opts};

fn main() {
    let opts: Opts = argh::from_env();

    let result = match opts.command {
        Cmd::Fetch(cmd) => fetch::fetch(&cmd.model).map(|dir| println!("{}", dir.display())),
        Cmd::Models(_) => list::list(),
        Cmd::Run(cmd) => run::run(&cmd.model),
    };

    if let Err(e) = result {
        eprintln!("error: {e}");
        std::process::exit(1);
    }
}
