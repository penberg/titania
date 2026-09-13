//! Compiles the Titania RTL in `rtl/` into a C++ model with Verilator, and
//! links it, with the harness, into this crate.
//!
//! Verilator is found through `VERILATOR_ROOT`, or on `PATH`. Without it the
//! crate still builds, but [`Rtlsim::new`](crate::Rtlsim::new) reports that
//! the simulator is unavailable.

use std::env;
use std::path::{Path, PathBuf};
use std::process::Command;

fn main() {
    let workspace = Path::new(env!("CARGO_MANIFEST_DIR")).parent().unwrap();
    let rtl = workspace.join("rtl");
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=harness.cpp");
    println!("cargo:rerun-if-changed={}", rtl.display());
    println!("cargo:rerun-if-env-changed=VERILATOR_ROOT");
    println!("cargo:rerun-if-env-changed=PATH");
    println!("cargo:rerun-if-env-changed=TITANIA_SMS");
    println!("cargo:rerun-if-env-changed=TITANIA_THREADS");
    println!("cargo:rustc-check-cfg=cfg(verilated)");

    let Some(verilator) = find_verilator() else {
        println!("cargo:warning=Verilator not found: `titania run --device rtlsim` will be unavailable");
        return;
    };

    let out = PathBuf::from(env::var("OUT_DIR").unwrap());
    let obj = out.join("verilated");
    let mut sources: Vec<PathBuf> = std::fs::read_dir(&rtl)
        .unwrap()
        .filter_map(|entry| entry.ok().map(|entry| entry.path()))
        .filter(|path| path.extension().is_some_and(|ext| ext == "sv"))
        .collect();
    // Packages must come before the modules that import them.
    sources.sort_by_key(|path| {
        let name = path.file_name().unwrap().to_str().unwrap();
        (!matches!(name, "fpu.sv" | "isa.sv"), name.to_string())
    });
    let cores = std::thread::available_parallelism().map_or(1, |n| n.get());
    // The number of SMs, and the threads that simulate them.
    let sms: usize = env::var("TITANIA_SMS").ok().and_then(|n| n.parse().ok()).unwrap_or(4);
    let threads: usize = env::var("TITANIA_THREADS").ok().and_then(|n| n.parse().ok()).unwrap_or(1);
    let status = Command::new(&verilator)
        .args(["--cc", "--build", "-O3", "-Wall", "-Wno-UNUSEDSIGNAL", "-Wno-fatal"])
        .args(["--top-module", "titania", "--prefix", "Vtitania"])
        .arg(format!("-GNUM_SMS={sms}"))
        .args(["--threads", &threads.to_string()])
        .args(["-CFLAGS", "-O2 -fPIC", "-j", &cores.to_string()])
        .arg("--Mdir")
        .arg(&obj)
        .args(&sources)
        .status()
        .expect("failed to run verilator");
    assert!(status.success(), "verilator failed");

    let root = Command::new(&verilator)
        .args(["--getenv", "VERILATOR_ROOT"])
        .output()
        .expect("failed to run verilator");
    let root = PathBuf::from(String::from_utf8(root.stdout).unwrap().trim());
    let include = root.join("include");

    cc::Build::new()
        .cpp(true)
        .file("harness.cpp")
        .define("TITANIA_SMS", sms.to_string().as_str())
        .include(&obj)
        .include(&include)
        .include(include.join("vltstd"))
        .flag("-std=c++20")
        .flag("-O2")
        .warnings(false)
        .compile("titania_harness");

    println!("cargo:rustc-link-search=native={}", obj.display());
    println!("cargo:rustc-link-lib=static=Vtitania");
    println!("cargo:rustc-link-lib=static=verilated");
    println!("cargo:rustc-link-lib=stdc++");
    println!("cargo:rustc-cfg=verilated");
}

/// The Verilator executable, if there is one.
fn find_verilator() -> Option<PathBuf> {
    if let Ok(root) = env::var("VERILATOR_ROOT") {
        let path = Path::new(&root).join("bin").join("verilator");
        if path.is_file() {
            return Some(path);
        }
    }
    env::var_os("PATH").and_then(|path| {
        env::split_paths(&path)
            .map(|dir| dir.join("verilator"))
            .find(|path| path.is_file())
    })
}
