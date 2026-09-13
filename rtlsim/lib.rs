//! The Titania RTL simulator: runs programs on the GPU designed in `rtl/`,
//! simulated cycle by cycle with Verilator.
//!
//! It has the same interface as the ISA simulator, so the runtime runs the
//! model on either, and the two must agree bit for bit on every program
//! whose behavior the architecture manual defines.
//!
//! Global memory lives here, not in the design: the harness in `harness.cpp`
//! serves the GPU's memory ports from it, as a memory controller would.

use std::fmt;
use std::sync::Arc;

pub use titania_simulator::{Activity, Error, Launch, Sample};

/// Cycles to simulate between updates of the activity monitor.
const SLICE: u64 = 4096;

/// A simulated Titania GPU with its global memory.
pub struct Rtlsim {
    chip: chip::Chip,
    /// Global memory, one word per element.
    memory: Vec<u32>,
    activity: Arc<Activity>,
}

/// The simulator was built without Verilator, so there is no design to run.
#[derive(Debug)]
pub struct Unavailable;

impl fmt::Display for Unavailable {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "the RTL simulator was not built: Verilator was not found when compiling titania-rtlsim; \
             install Verilator (https://verilator.org), then rebuild with \
             `cargo clean -p titania-rtlsim && cargo build`"
        )
    }
}

impl std::error::Error for Unavailable {}

/// How a launch went: what the hardware reports when it stops.
#[derive(Clone, Copy, Debug, Default)]
#[repr(C)]
pub struct Status {
    /// Cycles since the launch started.
    pub cycles: u64,
    /// Instructions completed since the launch started, counting each once
    /// per warp that executes it.
    pub retired: u64,
    sample_block: u32,
    sample_warp: u32,
    sample_pc: u32,
    error_code: u32,
    error_pc: u32,
    error_addr: u32,
}

impl Rtlsim {
    pub fn new() -> Result<Self, Unavailable> {
        Ok(Self {
            chip: chip::Chip::new().ok_or(Unavailable)?,
            memory: Vec::new(),
            activity: Arc::default(),
        })
    }

    /// What the simulator is doing, to watch it from another thread.
    pub fn activity(&self) -> Arc<Activity> {
        self.activity.clone()
    }

    /// Allocates `bytes` of zeroed global memory, returning its address.
    pub fn alloc(&mut self, bytes: usize) -> u32 {
        let start = (self.memory.len() * 4).next_multiple_of(256);
        let end = start + bytes.next_multiple_of(4);
        assert!(end <= 1 << 32, "global memory is limited to 4 GiB");
        self.memory.resize(end / 4, 0);
        start as u32
    }

    /// Writes words to global memory at `addr`.
    pub fn write(&mut self, addr: u32, words: &[u32]) {
        let start = addr as usize / 4;
        self.memory[start..start + words.len()].copy_from_slice(words);
    }

    /// Reads `len` words from global memory at `addr`.
    pub fn read(&self, addr: u32, len: usize) -> Vec<u32> {
        let start = addr as usize / 4;
        self.memory[start..start + len].to_vec()
    }

    /// Runs a kernel to completion, returning how long it took.
    pub fn launch(&mut self, launch: &Launch) -> Result<Status, Error> {
        let [width, height] = launch.grid;
        if width == 0 || height == 0 {
            return Err(Error::InvalidLaunch("the grid is empty"));
        }
        if !(1..=1024).contains(&launch.block) {
            return Err(Error::InvalidLaunch("block size must be from 1 to 1024"));
        }
        if !launch.shared.is_multiple_of(4) || launch.shared > 65536 {
            return Err(Error::InvalidLaunch("invalid shared memory size"));
        }
        if launch.params.len() > 64 {
            return Err(Error::InvalidLaunch("parameters are limited to 256 bytes"));
        }
        if launch.program.len() > self.chip.imem_words() {
            return Err(Error::InvalidLaunch("the program does not fit in instruction memory"));
        }

        self.activity.launched();
        self.chip.start(launch, &mut self.memory);
        let mut last = Status::default();
        loop {
            let (state, status) = self.chip.run(SLICE);
            self.activity.executed(status.retired - last.retired);
            self.activity.clocked(status.cycles - last.cycles);
            self.activity.record(Sample {
                block: status.sample_block,
                warp: status.sample_warp,
                pc: status.sample_pc as usize,
            });
            last = status;
            match state {
                chip::State::Running => {}
                chip::State::Done => return Ok(status),
                chip::State::Failed => return Err(error(launch, status)),
            }
        }
    }
}

/// The error a stopped launch reports, in the terms of the ISA simulator.
fn error(launch: &Launch, status: Status) -> Error {
    let pc = status.error_pc as usize;
    let addr = status.error_addr;
    match status.error_code {
        1 => Error::InvalidInstruction {
            pc,
            word: launch.program.get(pc).copied().unwrap_or(0),
        },
        2 => Error::PcOutOfRange { pc },
        3 => Error::DivergentBranch { pc },
        4 => Error::DivergentBarrier { pc },
        5 => Error::Memory { pc, space: "global", addr },
        6 => Error::Memory { pc, space: "shared", addr },
        _ => Error::Memory { pc, space: "param", addr },
    }
}

/// The Verilated design, behind the C interface of `harness.cpp`.
#[cfg(verilated)]
mod chip {
    use std::ffi::c_void;

    use super::{Launch, Status};

    unsafe extern "C" {
        fn titania_new() -> *mut c_void;
        fn titania_free(chip: *mut c_void);
        fn titania_imem_words() -> u32;
        fn titania_start(
            chip: *mut c_void,
            program: *const u64,
            prog_len: u32,
            params: *const u32,
            nparams: u32,
            grid_width: u32,
            grid_height: u32,
            block: u32,
            shared: u32,
            mem: *mut u32,
            mem_words: usize,
        );
        fn titania_run(chip: *mut c_void, max_cycles: u64, status: *mut Status) -> i32;
    }

    pub enum State {
        Running,
        Done,
        Failed,
    }

    pub struct Chip {
        ptr: *mut c_void,
    }

    // The design is only ever used from the thread that owns it.
    unsafe impl Send for Chip {}

    impl Chip {
        pub fn new() -> Option<Self> {
            let ptr = unsafe { titania_new() };
            (!ptr.is_null()).then_some(Self { ptr })
        }

        pub fn imem_words(&self) -> usize {
            unsafe { titania_imem_words() as usize }
        }

        pub fn start(&mut self, launch: &Launch, memory: &mut [u32]) {
            unsafe {
                titania_start(
                    self.ptr,
                    launch.program.as_ptr(),
                    launch.program.len() as u32,
                    launch.params.as_ptr(),
                    launch.params.len() as u32,
                    launch.grid[0],
                    launch.grid[1],
                    launch.block,
                    launch.shared,
                    memory.as_mut_ptr(),
                    memory.len(),
                );
            }
        }

        /// Simulates up to `cycles` cycles, or until the launch stops.
        pub fn run(&mut self, cycles: u64) -> (State, Status) {
            let mut status = Status::default();
            let state = unsafe { titania_run(self.ptr, cycles, &mut status) };
            let state = match state {
                0 => State::Running,
                1 => State::Done,
                _ => State::Failed,
            };
            (state, status)
        }
    }

    impl Drop for Chip {
        fn drop(&mut self) {
            unsafe { titania_free(self.ptr) }
        }
    }
}

/// Stands in for the design when the crate was built without Verilator.
#[cfg(not(verilated))]
mod chip {
    use super::{Launch, Status};

    #[allow(dead_code)]
    pub enum State {
        Running,
        Done,
        Failed,
    }

    pub struct Chip;

    impl Chip {
        pub fn new() -> Option<Self> {
            None
        }

        pub fn imem_words(&self) -> usize {
            unreachable!("the RTL simulator was built without Verilator")
        }

        pub fn start(&mut self, _launch: &Launch, _memory: &mut [u32]) {
            unreachable!("the RTL simulator was built without Verilator")
        }

        pub fn run(&mut self, _cycles: u64) -> (State, Status) {
            unreachable!("the RTL simulator was built without Verilator")
        }
    }
}

/// Checks the hardware against the ISA simulator, the reference: every
/// program must produce the same memory, or the same error, on both.
#[cfg(test)]
mod tests {
    use super::*;
    use titania_simulator::Simulator;

    /// What the tests need of a simulator.
    trait Gpu {
        fn alloc(&mut self, bytes: usize) -> u32;
        fn write(&mut self, addr: u32, words: &[u32]);
        fn read(&self, addr: u32, len: usize) -> Vec<u32>;
        fn launch(&mut self, launch: &Launch) -> Result<(), Error>;
    }

    impl Gpu for Simulator {
        fn alloc(&mut self, bytes: usize) -> u32 {
            Simulator::alloc(self, bytes)
        }

        fn write(&mut self, addr: u32, words: &[u32]) {
            Simulator::write(self, addr, words)
        }

        fn read(&self, addr: u32, len: usize) -> Vec<u32> {
            Simulator::read(self, addr, len)
        }

        fn launch(&mut self, launch: &Launch) -> Result<(), Error> {
            Simulator::launch(self, launch)
        }
    }

    impl Gpu for Rtlsim {
        fn alloc(&mut self, bytes: usize) -> u32 {
            Rtlsim::alloc(self, bytes)
        }

        fn write(&mut self, addr: u32, words: &[u32]) {
            Rtlsim::write(self, addr, words)
        }

        fn read(&self, addr: u32, len: usize) -> Vec<u32> {
            Rtlsim::read(self, addr, len)
        }

        fn launch(&mut self, launch: &Launch) -> Result<(), Error> {
            Rtlsim::launch(self, launch).map(|_| ())
        }
    }

    // Opcodes (§8).
    const IADD: u8 = 0x01;
    const ISUB: u8 = 0x02;
    const IMUL: u8 = 0x03;
    const IMAD: u8 = 0x04;
    const AND: u8 = 0x05;
    const OR: u8 = 0x06;
    const XOR: u8 = 0x07;
    const SHL: u8 = 0x08;
    const SHR: u8 = 0x09;
    const SRA: u8 = 0x0A;
    const FADD: u8 = 0x10;
    const FSUB: u8 = 0x11;
    const FMUL: u8 = 0x12;
    const FFMA: u8 = 0x13;
    const FMIN: u8 = 0x14;
    const FMAX: u8 = 0x15;
    const FDIV: u8 = 0x16;
    const FSQRT: u8 = 0x17;
    const I2F: u8 = 0x18;
    const F2I: u8 = 0x19;
    const MOV: u8 = 0x20;
    const S2R: u8 = 0x21;
    const ISETP_EQ: u8 = 0x28;
    const ISETP_GE: u8 = 0x2D;
    const FSETP_EQ: u8 = 0x30;
    const FSETP_GE: u8 = 0x35;
    const LDG: u8 = 0x40;
    const STG: u8 = 0x41;
    const LDS: u8 = 0x42;
    const STS: u8 = 0x43;
    const LDP: u8 = 0x44;
    const SHFL_IDX: u8 = 0x48;
    const SHFL_BFLY: u8 = 0x49;
    const BRA: u8 = 0x50;
    const BAR: u8 = 0x51;
    const EXIT: u8 = 0x52;

    const PT: u8 = 7;

    /// Encodes an instruction (§7).
    #[allow(clippy::too_many_arguments)]
    fn encode(op: u8, guard: u8, neg: bool, rd: u8, ra: u8, rb: u8, rc: u8, imm: Option<u32>) -> u64 {
        let low = op as u64
            | (guard as u64) << 8
            | (neg as u64) << 11
            | (rd as u64) << 12
            | (ra as u64) << 18
            | (rb as u64) << 24
            | (imm.is_some() as u64) << 30;
        low | imm.map_or(rc as u64, |imm| imm as u64) << 32
    }

    /// An unguarded instruction.
    fn inst(op: u8, rd: u8, ra: u8, rb: u8, rc: u8, imm: Option<u32>) -> u64 {
        encode(op, PT, false, rd, ra, rb, rc, imm)
    }

    /// An instruction guarded by `p0`, or `!p0`.
    fn guarded(neg: bool, op: u8, rd: u8, ra: u8, rb: u8, imm: Option<u32>) -> u64 {
        encode(op, 0, neg, rd, ra, rb, 0, imm)
    }

    /// Pseudorandom words: mostly arbitrary bit patterns, with a share of
    /// floating-point edge cases and small integers.
    fn words(n: usize, seed: u64) -> Vec<u32> {
        const SPECIALS: [u32; 20] = [
            0, 0x8000_0000, 0x3f80_0000, 0xbf80_0000, 0x7f80_0000, 0xff80_0000, 0x7fc0_0000, 0x7f80_0001,
            0x0000_0001, 0x8000_0001, 0x007f_ffff, 0x0080_0000, 0x7f7f_ffff, 0x3f00_0000, 0x4f00_0000,
            0xcf00_0000, 0x4eff_ffff, 0x3eff_ffff, 0x3f7f_ffff, 0x4b00_0000,
        ];
        let mut state = seed.wrapping_mul(0x9e37_79b9_7f4a_7c15) | 1;
        let mut next = move || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };
        (0..n)
            .map(|_| match next() % 5 {
                0 => SPECIALS[(next() % SPECIALS.len() as u64) as usize],
                1 => (next() % 64) as u32,
                2 => {
                    let e = 120 + next() % 16;
                    ((next() & 1) << 31 | e << 23 | next() & 0x7f_ffff) as u32
                }
                _ => next() as u32,
            })
            .collect()
    }

    /// The memory of a GPU after a launch, or its error.
    struct Run {
        memory: Vec<u32>,
        result: Result<(), String>,
    }

    /// Runs a launch of a `grid` of blocks on both simulators, with the same
    /// buffers uploaded to each; the parameters are the buffers' addresses,
    /// then `extra`.
    ///
    /// Without Verilator there is no design to test: the simulators are then
    /// both the ISA simulator, and the test passes trivially, with a note.
    fn both(program: &[u64], grid: [u32; 2], block: u32, shared: u32, buffers: &[Vec<u32>], extra: &[u32]) -> (Run, Run) {
        let mut sim = Simulator::new();
        let mut rtl: Box<dyn Gpu> = match Rtlsim::new() {
            Ok(rtl) => Box::new(rtl),
            Err(e) => {
                eprintln!("skipping the RTL simulator: {e}");
                Box::new(Simulator::new())
            }
        };
        let mut params = Vec::new();
        for buffer in buffers {
            let addr = sim.alloc(buffer.len() * 4);
            assert_eq!(addr, rtl.alloc(buffer.len() * 4));
            sim.write(addr, buffer);
            rtl.write(addr, buffer);
            params.push(addr);
        }
        params.extend(extra);
        let launch = Launch { program, grid, block, shared, params: &params };
        let sim_result = sim.launch(&launch).map_err(|e| e.to_string());
        let rtl_result = rtl.launch(&launch).map_err(|e| e.to_string());
        let last = params[buffers.len() - 1] as usize / 4 + buffers.last().unwrap().len();
        (
            Run { memory: sim.read(0, last), result: sim_result },
            Run { memory: rtl.read(0, last), result: rtl_result },
        )
    }

    fn assert_same(sim: &Run, rtl: &Run) {
        assert_eq!(sim.result, rtl.result);
        if let Some(i) = (0..sim.memory.len()).find(|&i| sim.memory[i] != rtl.memory[i]) {
            panic!(
                "word {i} differs: the ISA simulator has {:#010x}, the RTL simulator {:#010x}",
                sim.memory[i], rtl.memory[i]
            );
        }
    }

    /// A kernel with one thread per element: loads `a`, `b`, and `c` into
    /// `r11`, `r13`, and `r15`, then runs `body`, and stores `r16`.
    fn elementwise(body: &[u64]) -> Vec<u64> {
        let mut program = vec![
            inst(S2R, 1, 0, 0, 0, Some(0)),
            inst(S2R, 2, 0, 0, 0, Some(1)),
            inst(S2R, 3, 0, 0, 0, Some(2)),
            inst(IMAD, 4, 3, 2, 1, None),
            inst(SHL, 5, 4, 0, 0, Some(2)),
            inst(LDP, 6, 0, 0, 0, Some(0)),
            inst(LDP, 7, 0, 0, 0, Some(4)),
            inst(LDP, 8, 0, 0, 0, Some(8)),
            inst(LDP, 9, 0, 0, 0, Some(12)),
            inst(IADD, 10, 6, 5, 0, None),
            inst(LDG, 11, 10, 0, 0, Some(0)),
            inst(IADD, 12, 7, 5, 0, None),
            inst(LDG, 13, 12, 0, 0, Some(0)),
            inst(IADD, 14, 8, 5, 0, None),
            inst(LDG, 15, 14, 0, 0, Some(0)),
        ];
        program.extend_from_slice(body);
        program.extend([
            inst(IADD, 17, 9, 5, 0, None),
            inst(STG, 0, 17, 16, 0, Some(0)),
            inst(EXIT, 0, 0, 0, 0, None),
        ]);
        program
    }

    /// Runs an elementwise kernel over random inputs on both simulators.
    fn check_elementwise(body: &[u64], seed: u64) {
        let n = 2048;
        let inputs = [words(n, seed), words(n, seed + 1), words(n, seed + 2), vec![0; n]];
        let (sim, rtl) = both(&elementwise(body), [(n / 256) as u32, 1], 256, 0, &inputs, &[]);
        assert!(sim.result.is_ok(), "{:?}", sim.result);
        assert_same(&sim, &rtl);
    }

    #[test]
    fn arithmetic() {
        for op in [IADD, ISUB, IMUL, AND, OR, XOR, SHL, SHR, SRA, FADD, FSUB, FMUL, FMIN, FMAX, FDIV] {
            check_elementwise(&[inst(op, 16, 11, 13, 0, None)], op as u64);
            check_elementwise(&[inst(op, 16, 11, 0, 0, Some(0x4048_f5c3))], op as u64 + 100);
            check_elementwise(&[inst(op, 16, 11, 0, 0, Some(0x8000_0007))], op as u64 + 200);
        }
        for op in [IMAD, FFMA] {
            check_elementwise(&[inst(op, 16, 11, 13, 15, None)], op as u64);
            check_elementwise(&[inst(op, 16, 11, 13, 0, Some(0xbf80_0000))], op as u64 + 100);
        }
        for op in [FSQRT, I2F, F2I, MOV] {
            check_elementwise(&[inst(op, 16, 11, 0, 0, None)], op as u64);
            check_elementwise(&[inst(op, 16, 0, 0, 0, Some(0x4110_0000))], op as u64 + 100);
        }
    }

    #[test]
    fn fma_cancellation() {
        // c = -(a × b), so that the sum cancels down to the rounding error.
        let n = 2048;
        let a = words(n, 77);
        let b = words(n, 78);
        let c: Vec<u32> = a
            .iter()
            .zip(&b)
            .map(|(&x, &y)| (-(f32::from_bits(x) * f32::from_bits(y))).to_bits())
            .collect();
        let inputs = [a, b, c, vec![0; n]];
        let body = [inst(FFMA, 16, 11, 13, 15, None)];
        let (sim, rtl) = both(&elementwise(&body), [8, 1], 256, 0, &inputs, &[]);
        assert_same(&sim, &rtl);
    }

    #[test]
    fn comparisons() {
        for op in (ISETP_EQ..=ISETP_GE).chain(FSETP_EQ..=FSETP_GE) {
            // r16 = a op b ? 1 : 0, and r16 = 2 where the guard is false.
            let body = [
                inst(op, 0, 11, 13, 0, None),
                inst(MOV, 16, 0, 0, 0, Some(2)),
                guarded(false, MOV, 16, 0, 0, Some(1)),
                guarded(true, MOV, 16, 0, 0, Some(0)),
            ];
            check_elementwise(&body, op as u64);
            let body = [
                inst(op, 0, 11, 0, 0, Some(0x4000_0000)),
                inst(MOV, 16, 0, 0, 0, Some(2)),
                guarded(false, MOV, 16, 0, 0, Some(1)),
            ];
            check_elementwise(&body, op as u64 + 100);
        }
    }

    #[test]
    fn shuffles() {
        // Every lane reads its neighbors, in a block whose last warp is
        // partial so that some source lanes do not exist.
        for op in [SHFL_IDX, SHFL_BFLY] {
            let n = 320;
            let inputs = [words(n, 5), words(n, 6), words(n, 7), vec![0; n]];
            let body = [inst(op, 16, 11, 13, 0, None)];
            let (sim, rtl) = both(&elementwise(&body), [1, 1], 200, 0, &inputs, &[]);
            assert_same(&sim, &rtl);
            let body = [inst(op, 16, 11, 0, 0, Some(5))];
            let (sim, rtl) = both(&elementwise(&body), [1, 1], 200, 0, &inputs, &[]);
            assert_same(&sim, &rtl);
        }
    }

    #[test]
    fn warp_sum_with_early_exits() {
        // Lanes above 20 exit before the reduction, so the shuffles read
        // dead lanes; then the warp sums thread numbers with butterflies.
        let mut program = vec![
            inst(S2R, 1, 0, 0, 0, Some(0)),
            inst(AND, 20, 1, 0, 0, Some(31)),
            inst(ISETP_GE, 0, 20, 0, 0, Some(21)),
            guarded(false, EXIT, 0, 0, 0, None),
            inst(MOV, 2, 1, 0, 0, None),
        ];
        for offset in [16, 8, 4, 2, 1] {
            program.push(inst(SHFL_BFLY, 3, 2, 0, 0, Some(offset)));
            program.push(inst(IADD, 2, 2, 3, 0, None));
        }
        program.extend([
            inst(LDP, 4, 0, 0, 0, Some(0)),
            inst(SHL, 5, 1, 0, 0, Some(2)),
            inst(IADD, 4, 4, 5, 0, None),
            inst(STG, 0, 4, 2, 0, Some(0)),
            inst(EXIT, 0, 0, 0, 0, None),
        ]);
        let (sim, rtl) = both(&program, [3, 1], 100, 0, &[vec![0; 300]], &[]);
        assert!(sim.result.is_ok());
        assert_same(&sim, &rtl);
    }

    #[test]
    fn shared_memory_and_barriers() {
        // Each thread stores to shared memory, the block synchronizes, and
        // each thread reads another's word: first a permutation, then a
        // pattern where every lane of a warp hits the same bank.
        let block = 256;
        let program = vec![
            inst(S2R, 1, 0, 0, 0, Some(0)),
            inst(S2R, 3, 0, 0, 0, Some(2)),
            inst(SHL, 2, 1, 0, 0, Some(2)),
            inst(IMUL, 4, 1, 0, 0, Some(3)),
            inst(IADD, 4, 4, 3, 0, None),
            inst(STS, 0, 2, 4, 0, Some(0)),
            inst(BAR, 0, 0, 0, 0, None),
            // Reversed: word 255 - tid.
            inst(ISUB, 5, 0, 1, 0, None),
            inst(IADD, 5, 5, 0, 0, Some(255)),
            inst(SHL, 5, 5, 0, 0, Some(2)),
            inst(LDS, 6, 5, 0, 0, Some(0)),
            // Conflicting: word (tid × 32) mod 256.
            inst(SHL, 7, 1, 0, 0, Some(5)),
            inst(AND, 7, 7, 0, 0, Some(255)),
            inst(SHL, 7, 7, 0, 0, Some(2)),
            inst(LDS, 8, 7, 0, 0, Some(0)),
            inst(BAR, 0, 0, 0, 0, None),
            inst(STS, 0, 2, 8, 0, Some(0)),
            inst(BAR, 0, 0, 0, 0, None),
            inst(LDS, 9, 5, 0, 0, Some(0)),
            inst(IADD, 6, 6, 9, 0, None),
            inst(IADD, 6, 6, 8, 0, None),
            // out[block × 256 + tid] = r6
            inst(LDP, 10, 0, 0, 0, Some(0)),
            inst(SHL, 11, 3, 0, 0, Some(10)),
            inst(IADD, 10, 10, 11, 0, None),
            inst(IADD, 10, 10, 2, 0, None),
            inst(STG, 0, 10, 6, 0, Some(0)),
            inst(EXIT, 0, 0, 0, 0, None),
        ];
        let (sim, rtl) = both(&program, [5, 1], block, 1024, &[vec![0; 5 * 256]], &[]);
        assert!(sim.result.is_ok(), "{:?}", sim.result);
        assert_same(&sim, &rtl);
    }

    #[test]
    fn loops_and_blocks() {
        // Each block's threads count to the block number in a loop, so that
        // blocks take different times, and there are more blocks than SMs.
        let program = vec![
            inst(S2R, 1, 0, 0, 0, Some(0)),
            inst(S2R, 2, 0, 0, 0, Some(2)),
            inst(S2R, 3, 0, 0, 0, Some(3)),
            inst(S2R, 4, 0, 0, 0, Some(1)),
            inst(MOV, 5, 0, 0, 0, Some(0)),
            inst(MOV, 6, 0, 0, 0, Some(0)),
            // loop:
            inst(ISETP_GE, 0, 5, 2, 0, None),
            guarded(false, BRA, 0, 0, 0, Some(11)),
            inst(IADD, 6, 6, 5, 0, None),
            inst(IADD, 5, 5, 0, 0, Some(1)),
            inst(BRA, 0, 0, 0, 0, Some(6)),
            // done: out[block × 40 + tid] = sum × grid + block size
            inst(IMAD, 7, 6, 3, 4, None),
            inst(IMAD, 8, 2, 4, 1, None),
            inst(SHL, 8, 8, 0, 0, Some(2)),
            inst(LDP, 9, 0, 0, 0, Some(0)),
            inst(IADD, 9, 9, 8, 0, None),
            inst(STG, 0, 9, 7, 0, Some(0)),
            inst(EXIT, 0, 0, 0, 0, None),
        ];
        let (sim, rtl) = both(&program, [37, 1], 40, 0, &[vec![0; 37 * 40]], &[]);
        assert!(sim.result.is_ok(), "{:?}", sim.result);
        assert_same(&sim, &rtl);
    }

    #[test]
    fn two_dimensional_grid() {
        // Every thread of a 7×5 grid writes its block's coordinates and the
        // grid's size to out[(y × width + x) × block size + tid], so that
        // every block must run, exactly once, and see all four special
        // registers.
        let (width, height, block) = (7u32, 5u32, 40u32);
        let program = vec![
            inst(S2R, 1, 0, 0, 0, Some(0)),
            inst(S2R, 2, 0, 0, 0, Some(2)),
            inst(S2R, 3, 0, 0, 0, Some(3)),
            inst(S2R, 4, 0, 0, 0, Some(4)),
            inst(S2R, 5, 0, 0, 0, Some(5)),
            inst(S2R, 6, 0, 0, 0, Some(1)),
            // r8 = out + ((y × width + x) × block size + tid) × 4
            inst(IMAD, 7, 4, 3, 2, None),
            inst(IMAD, 7, 7, 6, 1, None),
            inst(SHL, 7, 7, 0, 0, Some(2)),
            inst(LDP, 8, 0, 0, 0, Some(0)),
            inst(IADD, 8, 8, 7, 0, None),
            // r9 = x + 100y + 10000 × width + 1000000 × height
            inst(MOV, 10, 0, 0, 0, Some(100)),
            inst(IMAD, 9, 4, 10, 2, None),
            inst(MOV, 10, 0, 0, 0, Some(10_000)),
            inst(IMAD, 9, 3, 10, 9, None),
            inst(MOV, 10, 0, 0, 0, Some(1_000_000)),
            inst(IMAD, 9, 5, 10, 9, None),
            inst(STG, 0, 8, 9, 0, Some(0)),
            inst(EXIT, 0, 0, 0, 0, None),
        ];
        let out = vec![0; (width * height * block) as usize];
        let (sim, rtl) = both(&program, [width, height], block, 0, &[out], &[]);
        assert!(sim.result.is_ok(), "{:?}", sim.result);
        assert_same(&sim, &rtl);
        let start = rtl.memory.len() - (width * height * block) as usize;
        for y in 0..height {
            for x in 0..width {
                let expected = x + 100 * y + 10_000 * width + 1_000_000 * height;
                let block_start = start + ((y * width + x) * block) as usize;
                assert!(rtl.memory[block_start..block_start + block as usize].iter().all(|&w| w == expected));
            }
        }
    }

    #[test]
    fn errors() {
        let out = vec![0u32; 64];
        let prologue = |rest: &[u64]| {
            let mut program = vec![inst(S2R, 1, 0, 0, 0, Some(0)), inst(LDP, 2, 0, 0, 0, Some(0))];
            program.extend_from_slice(rest);
            program
        };
        let cases: Vec<(&str, Vec<u64>)> = vec![
            (
                "divergent branch",
                prologue(&[
                    inst(ISETP_GE, 0, 1, 0, 0, Some(16)),
                    guarded(false, BRA, 0, 0, 0, Some(4)),
                    inst(EXIT, 0, 0, 0, 0, None),
                ]),
            ),
            (
                "divergent barrier",
                prologue(&[
                    inst(ISETP_GE, 0, 1, 0, 0, Some(16)),
                    guarded(false, BAR, 0, 0, 0, None),
                    inst(EXIT, 0, 0, 0, 0, None),
                ]),
            ),
            ("invalid instruction", prologue(&[inst(0x7f, 0, 0, 0, 0, None), inst(EXIT, 0, 0, 0, 0, None)])),
            (
                "unknown special register",
                prologue(&[inst(S2R, 3, 0, 0, 0, Some(6)), inst(EXIT, 0, 0, 0, 0, None)]),
            ),
            ("pc out of range", prologue(&[inst(IADD, 3, 1, 2, 0, None)])),
            (
                "global misaligned",
                prologue(&[inst(LDG, 3, 2, 0, 0, Some(2)), inst(EXIT, 0, 0, 0, 0, None)]),
            ),
            (
                "global out of range",
                prologue(&[inst(LDG, 3, 2, 0, 0, Some(1 << 20)), inst(EXIT, 0, 0, 0, 0, None)]),
            ),
            (
                "shared out of range",
                prologue(&[inst(STS, 0, 1, 2, 0, Some(1024)), inst(EXIT, 0, 0, 0, 0, None)]),
            ),
            (
                "param out of range",
                prologue(&[inst(LDP, 3, 0, 0, 0, Some(8)), inst(EXIT, 0, 0, 0, 0, None)]),
            ),
        ];
        for (name, program) in cases {
            let (sim, rtl) = both(&program, [1, 1], 64, 256, std::slice::from_ref(&out), &[]);
            assert!(sim.result.is_err(), "{name}: the ISA simulator accepted the program");
            assert_eq!(sim.result, rtl.result, "{name}");
        }
    }
}
