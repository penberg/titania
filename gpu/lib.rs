//! A Titania GPU, as the host sees it.
//!
//! Whatever executes Titania programs, the ISA simulator, an RTL simulator
//! of the GPU design, or the GPU itself, is a [`Gpu`] to the host: global
//! memory to allocate, write, and read, and kernels to launch (§5 of the
//! architecture manual). The runtime runs the model on any of them.

use std::{
    fmt,
    sync::{
        Arc,
        atomic::{AtomicU64, Ordering::Relaxed},
    },
};

/// A GPU that runs Titania programs: what the host sees of it.
pub trait Gpu {
    /// What the GPU is doing, to watch it from another thread.
    fn activity(&self) -> Arc<Activity>;

    /// Allocates `bytes` of zeroed global memory, returning its address.
    fn alloc(&mut self, bytes: usize) -> u32;

    /// Writes words to global memory at `addr`.
    fn write(&mut self, addr: u32, words: &[u32]);

    /// Reads `len` words from global memory at `addr`.
    fn read(&self, addr: u32, len: usize) -> Vec<u32>;

    /// Runs a kernel to completion.
    fn launch(&mut self, launch: &Launch) -> Result<(), Error>;
}

/// A kernel launch (§5).
pub struct Launch<'a> {
    /// The kernel's encoded instructions.
    pub program: &'a [u64],
    /// Blocks along x and y.
    pub grid: [u32; 2],
    pub block: u32,
    /// Bytes of shared memory per block.
    pub shared: u32,
    pub params: &'a [u32],
}

/// An error that stops a launch (§6).
#[derive(Debug)]
pub enum Error {
    InvalidLaunch(&'static str),
    InvalidInstruction { pc: usize, word: u64 },
    PcOutOfRange { pc: usize },
    DivergentBranch { pc: usize },
    DivergentBarrier { pc: usize },
    Memory { pc: usize, space: &'static str, addr: u32 },
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::InvalidLaunch(reason) => write!(f, "invalid launch: {reason}"),
            Error::InvalidInstruction { pc, word } => write!(f, "instruction {pc}: invalid instruction {word:#018x}"),
            Error::PcOutOfRange { pc } => write!(f, "PC {pc} is outside the program"),
            Error::DivergentBranch { pc } => write!(f, "instruction {pc}: divergent branch"),
            Error::DivergentBarrier { pc } => write!(f, "instruction {pc}: divergent barrier"),
            Error::Memory { pc, space, addr } => {
                write!(f, "instruction {pc}: invalid {space} memory access at {addr:#x}")
            }
        }
    }
}

impl std::error::Error for Error {}

/// What a GPU is doing, updated as it runs so that another thread can watch
/// it.
///
/// Keeping it up to date costs next to nothing: a GPU adds up the
/// instructions it executes and publishes them when a block finishes, and
/// records where a warp is only now and then.
#[derive(Default)]
pub struct Activity {
    launches: AtomicU64,
    instructions: AtomicU64,
    /// A [`Sample`], packed into one word so that it is read whole.
    sample: AtomicU64,
}

/// Where a warp was when sampled.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Sample {
    pub block: u32,
    pub warp: u32,
    pub pc: usize,
}

impl Activity {
    /// Kernels launched so far.
    pub fn launches(&self) -> u64 {
        self.launches.load(Relaxed)
    }

    /// Instructions executed so far by blocks that have finished, counting
    /// each instruction once per warp that executes it.
    pub fn instructions(&self) -> u64 {
        self.instructions.load(Relaxed)
    }

    /// Where a warp of the kernel launched last was recently.
    pub fn sample(&self) -> Sample {
        let packed = self.sample.load(Relaxed);
        Sample {
            block: (packed >> 32) as u32,
            warp: (packed >> 24 & 0xff) as u32,
            pc: (packed & 0xff_ffff) as usize,
        }
    }

    /// Records that a kernel was launched.
    pub fn launched(&self) {
        self.launches.fetch_add(1, Relaxed);
        self.record(Sample { block: 0, warp: 0, pc: 0 });
    }

    /// Records that `instructions` more instructions have executed.
    pub fn executed(&self, instructions: u64) {
        self.instructions.fetch_add(instructions, Relaxed);
    }

    /// Records where a warp is.
    pub fn record(&self, sample: Sample) {
        let packed = (sample.block as u64) << 32 | (sample.warp as u64) << 24 | (sample.pc as u64 & 0xff_ffff);
        self.sample.store(packed, Relaxed);
    }
}
