//! The Titania ISA simulator: the reference implementation of the Titania GPU
//! Architecture Reference Manual (`docs/architecture-reference.md`).
//!
//! It models what each instruction does, not how long it takes. Each warp
//! executes an instruction for all of its threads at once, as the hardware
//! does, and blocks run in parallel on the host's cores.

use std::sync::{
    Arc,
    atomic::{AtomicU32, Ordering::Relaxed},
};

use rayon::prelude::*;
use titania_gpu::{Activity, Error, Gpu, Launch, Sample};

/// Number of threads in a warp.
const WARP_SIZE: u32 = 32;

/// Number of general-purpose registers.
const NUM_REGS: usize = 64;

/// Instructions a block executes between samples of where one of its warps
/// is, for [`Activity`].
const SAMPLE_INTERVAL: u64 = 1024;

/// The guard field value for `pt`, the always-true predicate.
const PT: u8 = 7;

/// Special registers, read with `S2R` (§2.3).
const SR_TID: u32 = 0;
const SR_NTID: u32 = 1;
const SR_CTAID_X: u32 = 2;
const SR_NCTAID_X: u32 = 3;
const SR_CTAID_Y: u32 = 4;
const SR_NCTAID_Y: u32 = 5;

/// The canonical NaN every floating-point operation produces (§4).
const CANONICAL_NAN: u32 = 0x7fc0_0000;

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
const ISETP_NE: u8 = 0x29;
const ISETP_LT: u8 = 0x2A;
const ISETP_LE: u8 = 0x2B;
const ISETP_GT: u8 = 0x2C;
const ISETP_GE: u8 = 0x2D;
const FSETP_EQ: u8 = 0x30;
const FSETP_NE: u8 = 0x31;
const FSETP_LT: u8 = 0x32;
const FSETP_LE: u8 = 0x33;
const FSETP_GT: u8 = 0x34;
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

/// The operands an instruction takes (§7 and §8).
#[derive(Clone, Copy, PartialEq, Eq)]
enum Format {
    /// `rd, ra, b`
    R2,
    /// `rd, ra, rb, c`
    R3,
    /// `rd, a`
    R1,
    /// `pd, ra, b`
    Setp,
    /// `rd, [ra + imm]`
    Load,
    /// `[ra + imm], rb`
    Store,
    /// `rd, imm`
    Special,
    /// `imm`
    Branch,
    /// No operands.
    None,
}

/// The operands an opcode takes, or `None` if the opcode is unknown.
fn format(op: u8) -> Option<Format> {
    Some(match op {
        IADD | ISUB | IMUL | AND | OR | XOR | SHL | SHR | SRA => Format::R2,
        FADD | FSUB | FMUL | FMIN | FMAX | FDIV | SHFL_IDX | SHFL_BFLY => Format::R2,
        IMAD | FFMA => Format::R3,
        FSQRT | I2F | F2I | MOV => Format::R1,
        ISETP_EQ..=ISETP_GE | FSETP_EQ..=FSETP_GE => Format::Setp,
        LDG | LDS | LDP => Format::Load,
        STG | STS => Format::Store,
        S2R => Format::Special,
        BRA => Format::Branch,
        BAR | EXIT => Format::None,
        _ => return None,
    })
}

/// A decoded instruction: the fields of §7.
#[derive(Clone, Copy)]
struct Inst {
    op: u8,
    format: Format,
    guard: u8,
    neg: bool,
    rd: u8,
    ra: u8,
    rb: u8,
    rc: u8,
    /// The immediate, if the `i` bit is set.
    imm: Option<u32>,
}

/// Decodes an instruction word, or returns `None` if its opcode is unknown or
/// it sets a field its format doesn't use.
fn decode(word: u64) -> Option<Inst> {
    let op = word as u8;
    let format = format(op)?;
    let field = |shift: u32| (word >> shift & 0x3f) as u8;
    let has_imm = word >> 30 & 1 == 1;
    let high = (word >> 32) as u32;
    let inst = Inst {
        op,
        format,
        guard: (word >> 8 & 7) as u8,
        neg: word >> 11 & 1 == 1,
        rd: field(12),
        ra: field(18),
        rb: field(24),
        rc: if has_imm { 0 } else { (high & 0x3f) as u8 },
        imm: has_imm.then_some(high),
    };
    let (rd, ra, rb, rc) = (inst.rd, inst.ra, inst.rb, inst.rc);
    let fields_used = match format {
        Format::R2 => rc == 0 && (!has_imm || rb == 0),
        Format::R3 => true,
        Format::R1 => rb == 0 && rc == 0 && (!has_imm || ra == 0),
        Format::Setp => rd < 8 && rc == 0 && (!has_imm || rb == 0),
        Format::Load => has_imm && rb == 0,
        Format::Store => has_imm && rd == 0,
        Format::Special => has_imm && ra == 0 && rb == 0 && high <= SR_NCTAID_Y,
        Format::Branch => has_imm && rd == 0 && ra == 0 && rb == 0,
        Format::None => !has_imm && rd == 0 && ra == 0 && rb == 0 && rc == 0,
    };
    let reserved_clear = word >> 31 & 1 == 0 && (has_imm || high >> 6 == 0);
    (fields_used && reserved_clear).then_some(inst)
}

/// A simulated Titania GPU with its global memory.
pub struct Simulator {
    /// Global memory, one word per element: every access is a whole,
    /// aligned word. Atomics let blocks run in parallel; relaxed ordering is
    /// enough because blocks never access the same location when one of them
    /// writes it (§3).
    memory: Vec<AtomicU32>,
    activity: Arc<Activity>,
    /// Whether the host has the vector extensions [`map_simd`] uses.
    simd: bool,
}

impl Default for Simulator {
    fn default() -> Self {
        Self::new()
    }
}

impl Simulator {
    pub fn new() -> Self {
        Self {
            memory: Vec::new(),
            activity: Arc::default(),
            simd: host_has_simd(),
        }
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
        self.memory.resize_with(end / 4, || AtomicU32::new(0));
        start as u32
    }

    /// Writes words to global memory at `addr`.
    pub fn write(&self, addr: u32, words: &[u32]) {
        let start = addr as usize / 4;
        for (cell, &word) in self.memory[start..start + words.len()].iter().zip(words) {
            cell.store(word, Relaxed);
        }
    }

    /// Reads `len` words from global memory at `addr`.
    pub fn read(&self, addr: u32, len: usize) -> Vec<u32> {
        let start = addr as usize / 4;
        self.memory[start..start + len].iter().map(|cell| cell.load(Relaxed)).collect()
    }

    /// Runs a kernel to completion.
    pub fn launch(&self, launch: &Launch) -> Result<(), Error> {
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
        let program = launch
            .program
            .iter()
            .enumerate()
            .map(|(pc, &word)| decode(word).ok_or(Error::InvalidInstruction { pc, word }))
            .collect::<Result<Vec<_>, _>>()?;

        self.activity.launched();
        (0..width * height).into_par_iter().try_for_each(|id| {
            Block {
                memory: &self.memory,
                activity: &self.activity,
                program: &program,
                launch,
                id,
                x: id % width,
                y: id / width,
                shared: (0..launch.shared / 4).map(|_| AtomicU32::new(0)).collect(),
                simd: self.simd,
            }
            .run()
        })
    }
}

/// A block being executed.
struct Block<'a> {
    memory: &'a [AtomicU32],
    activity: &'a Activity,
    program: &'a [Inst],
    launch: &'a Launch<'a>,
    /// The block's number in row-major order, for the monitor.
    id: u32,
    x: u32,
    y: u32,
    /// Shared memory. Only one warp runs at a time, so relaxed atomics
    /// cost nothing over plain words and let it share global memory's code.
    shared: Vec<AtomicU32>,
    /// Whether to run whole rows of lanes through [`map_simd`].
    simd: bool,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum State {
    Running,
    /// Waiting at a barrier.
    Waiting,
    Finished,
}

struct Warp {
    id: u32,
    pc: usize,
    /// Threads that exist and have not exited, one bit per lane.
    live: u32,
    /// Register `r` of lane `l` is `regs[r][l]`.
    regs: Box<[Row; NUM_REGS]>,
    /// Predicate registers, one bit per lane; the last is `pt`.
    preds: [u32; 8],
    state: State,
}

/// A source operand: a register, or a row every lane reads, for an
/// immediate.
#[derive(Clone, Copy)]
enum Src<'a> {
    Reg(u8),
    Imm(&'a Row),
}

/// The index of register `reg` in a warp's register file. Register fields
/// are 6 bits wide, so every value fits; the mask lets the compiler drop the
/// bounds check.
fn reg(reg: u8) -> usize {
    reg as usize & (NUM_REGS - 1)
}

/// The ISA simulator is the reference [`Gpu`]: the one the hardware is
/// checked against.
impl Gpu for Simulator {
    fn activity(&self) -> Arc<Activity> {
        Simulator::activity(self)
    }

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

impl Warp {
    fn new(id: u32, block: u32) -> Self {
        let threads = (block - id * WARP_SIZE).min(WARP_SIZE);
        Self {
            id,
            pc: 0,
            live: if threads == WARP_SIZE { !0 } else { (1 << threads) - 1 },
            regs: Box::new([[0; WARP_SIZE as usize]; NUM_REGS]),
            preds: [0, 0, 0, 0, 0, 0, 0, !0],
            state: State::Running,
        }
    }

    fn get(&self, src: Src, lane: usize) -> u32 {
        self.row(src)[lane]
    }

    /// Reads a source operand in every lane.
    fn row<'s>(&'s self, src: Src<'s>) -> &'s Row {
        match src {
            Src::Reg(r) => &self.regs[reg(r)],
            Src::Imm(row) => row,
        }
    }

    /// Writes `value` to register `rd` of `lane`, discarding writes to `r0`.
    fn set(&mut self, rd: u8, lane: usize, value: u32) {
        if rd != 0 {
            self.regs[reg(rd)][lane] = value;
        }
    }

    /// Writes `values` to register `rd` in the lanes of `mask`, discarding
    /// writes to `r0`.
    fn set_lanes(&mut self, rd: u8, mask: u32, values: &Row) {
        if rd == 0 {
            return;
        }
        let dst = &mut self.regs[reg(rd)];
        if mask == FULL {
            *dst = *values;
        } else {
            for lane in lanes(mask) {
                dst[lane] = values[lane];
            }
        }
    }
}

/// Reads the word at `addrs[lane] + imm` of `space` for each `lane`, or
/// returns the first address that isn't a word in `space`.
fn gather<T>(
    space: &[T],
    addrs: &Row,
    imm: u32,
    lanes: impl Iterator<Item = usize>,
    read: impl Fn(&T) -> u32,
) -> Result<Row, u32> {
    let mut out = [0; WARP_SIZE as usize];
    for lane in lanes {
        let addr = addrs[lane].wrapping_add(imm);
        out[lane] = read(word(space, addr).ok_or(addr)?);
    }
    Ok(out)
}

/// Writes `values[lane]` to the word at `addrs[lane] + imm` of `space` for
/// each `lane`, or returns the first address that isn't a word in `space`.
fn scatter<T>(
    space: &[T],
    addrs: &Row,
    values: &Row,
    imm: u32,
    lanes: impl Iterator<Item = usize>,
    write: impl Fn(&T, u32),
) -> Result<(), u32> {
    for lane in lanes {
        let addr = addrs[lane].wrapping_add(imm);
        write(word(space, addr).ok_or(addr)?, values[lane]);
    }
    Ok(())
}

/// A two-operand integer operation as the ALU takes it. Generic, rather than
/// taking a function pointer, so that the operation inlines into the loop
/// over the lanes.
#[inline(always)]
fn int(f: impl Fn(u32, u32) -> u32) -> impl Fn(u32, u32, u32) -> u32 {
    move |x, y, _| f(x, y)
}

/// A two-operand floating-point operation as the ALU takes it.
#[inline(always)]
fn float(f: impl Fn(f32, f32) -> f32) -> impl Fn(u32, u32, u32) -> u32 {
    move |x, y, _| canonical(f(f32::from_bits(x), f32::from_bits(y)))
}

/// A register's value in every lane of a warp.
type Row = [u32; WARP_SIZE as usize];

/// `out[l] = f(a[l], b[l], c[l])` for every lane. The loop has a fixed trip
/// count and no branches of its own, so the compiler vectorizes it for
/// whatever `f` allows.
#[inline(always)]
fn map(a: &Row, b: &Row, c: &Row, f: impl Fn(u32, u32, u32) -> u32) -> Row {
    let mut out = [0; WARP_SIZE as usize];
    for lane in 0..WARP_SIZE as usize {
        out[lane] = f(a[lane], b[lane], c[lane]);
    }
    out
}

/// [`map`], compiled for the vector extensions the host was found to have at
/// run time (`Simulator::new` checks), so that it uses them even when the
/// build targets a baseline CPU: in particular, fused multiply-add stays a
/// vector instruction rather than a call into the math library.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
fn map_simd(a: &Row, b: &Row, c: &Row, f: impl Fn(u32, u32, u32) -> u32) -> Row {
    map(a, b, c, f)
}

#[cfg(not(target_arch = "x86_64"))]
fn map_simd(a: &Row, b: &Row, c: &Row, f: impl Fn(u32, u32, u32) -> u32) -> Row {
    map(a, b, c, f)
}

/// Whether the host has the vector extensions [`map_simd`] is compiled for.
fn host_has_simd() -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        std::arch::is_x86_feature_detected!("avx2") && std::arch::is_x86_feature_detected!("fma")
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        false
    }
}

/// The mask with every lane of a warp set.
const FULL: u32 = u32::MAX;

/// Zero in every lane: what an instruction's unused operands read.
static ZERO: Row = [0; WARP_SIZE as usize];

/// The lanes whose bits are set in `mask`.
fn lanes(mut mask: u32) -> impl Iterator<Item = usize> {
    std::iter::from_fn(move || {
        (mask != 0).then(|| {
            let lane = mask.trailing_zeros() as usize;
            mask &= mask - 1;
            lane
        })
    })
}

impl Block<'_> {
    fn run(mut self) -> Result<(), Error> {
        let mut warps: Vec<Warp> = (0..self.launch.block.div_ceil(WARP_SIZE))
            .map(|id| Warp::new(id, self.launch.block))
            .collect();
        let mut executed = 0;
        loop {
            // Run each warp until it finishes or waits at a barrier. When every
            // unfinished warp is waiting, the barrier releases them all.
            for warp in &mut warps {
                while warp.state == State::Running {
                    let pc = warp.pc;
                    self.step(warp)?;
                    executed += 1;
                    if executed % SAMPLE_INTERVAL == 0 {
                        self.activity.record(Sample {
                            block: self.id,
                            warp: warp.id,
                            pc,
                        });
                    }
                }
            }
            if warps.iter().all(|warp| warp.state == State::Finished) {
                self.activity.executed(executed);
                return Ok(());
            }
            for warp in &mut warps {
                if warp.state == State::Waiting {
                    warp.state = State::Running;
                }
            }
        }
    }

    /// Executes the instruction at the warp's PC.
    fn step(&mut self, w: &mut Warp) -> Result<(), Error> {
        let pc = w.pc;
        let inst = *self.program.get(pc).ok_or(Error::PcOutOfRange { pc })?;
        let guard = w.preds[inst.guard as usize];
        let mask = if inst.neg { !guard } else { guard } & w.live;
        w.pc += 1;

        let imm = inst.imm.unwrap_or(0);
        let imm_row = [imm; WARP_SIZE as usize];
        let zero = Src::Imm(&ZERO);
        let last = |reg: u8| if inst.imm.is_some() { Src::Imm(&imm_row) } else { Src::Reg(reg) };
        let (a, b, c) = match inst.format {
            Format::R1 => (last(inst.ra), zero, zero),
            Format::R2 | Format::Setp => (Src::Reg(inst.ra), last(inst.rb), zero),
            Format::R3 => (Src::Reg(inst.ra), Src::Reg(inst.rb), last(inst.rc)),
            _ => (Src::Reg(inst.ra), Src::Reg(inst.rb), zero),
        };

        match inst.op {
            IADD => self.alu(w, &inst, mask, a, b, c, int(u32::wrapping_add)),
            ISUB => self.alu(w, &inst, mask, a, b, c, int(u32::wrapping_sub)),
            IMUL => self.alu(w, &inst, mask, a, b, c, int(u32::wrapping_mul)),
            IMAD => self.alu(w, &inst, mask, a, b, c, |x, y, z| x.wrapping_mul(y).wrapping_add(z)),
            AND => self.alu(w, &inst, mask, a, b, c, int(|x, y| x & y)),
            OR => self.alu(w, &inst, mask, a, b, c, int(|x, y| x | y)),
            XOR => self.alu(w, &inst, mask, a, b, c, int(|x, y| x ^ y)),
            SHL => self.alu(w, &inst, mask, a, b, c, int(|x, y| x << (y & 31))),
            SHR => self.alu(w, &inst, mask, a, b, c, int(|x, y| x >> (y & 31))),
            SRA => self.alu(w, &inst, mask, a, b, c, int(|x, y| ((x as i32) >> (y & 31)) as u32)),
            FADD => self.alu(w, &inst, mask, a, b, c, float(|x, y| x + y)),
            FSUB => self.alu(w, &inst, mask, a, b, c, float(|x, y| x - y)),
            FMUL => self.alu(w, &inst, mask, a, b, c, float(|x, y| x * y)),
            FDIV => self.alu(w, &inst, mask, a, b, c, float(|x, y| x / y)),
            FFMA => self.alu(w, &inst, mask, a, b, c, |x, y, z| {
                canonical(f32::from_bits(x).mul_add(f32::from_bits(y), f32::from_bits(z)))
            }),
            FMIN => self.alu(w, &inst, mask, a, b, c, |x, y, _| fmin(x, y)),
            FMAX => self.alu(w, &inst, mask, a, b, c, |x, y, _| fmax(x, y)),
            FSQRT => self.alu(w, &inst, mask, a, b, c, |x, _, _| canonical(f32::from_bits(x).sqrt())),
            I2F => self.alu(w, &inst, mask, a, b, c, |x, _, _| (x as i32 as f32).to_bits()),
            // Float-to-integer casts saturate, and convert NaN to zero.
            F2I => self.alu(w, &inst, mask, a, b, c, |x, _, _| {
                f32::from_bits(x).round_ties_even() as i32 as u32
            }),
            MOV => self.alu(w, &inst, mask, a, b, c, |x, _, _| x),
            S2R => {
                for lane in lanes(mask) {
                    let value = match imm {
                        SR_TID => w.id * WARP_SIZE + lane as u32,
                        SR_NTID => self.launch.block,
                        SR_CTAID_X => self.x,
                        SR_NCTAID_X => self.launch.grid[0],
                        SR_CTAID_Y => self.y,
                        SR_NCTAID_Y => self.launch.grid[1],
                        _ => unreachable!("decode checks the special register"),
                    };
                    w.set(inst.rd, lane, value);
                }
            }
            ISETP_EQ => self.setp(w, &inst, mask, a, b, |x, y| x == y),
            ISETP_NE => self.setp(w, &inst, mask, a, b, |x, y| x != y),
            ISETP_LT => self.setp(w, &inst, mask, a, b, |x, y| (x as i32) < (y as i32)),
            ISETP_LE => self.setp(w, &inst, mask, a, b, |x, y| (x as i32) <= (y as i32)),
            ISETP_GT => self.setp(w, &inst, mask, a, b, |x, y| (x as i32) > (y as i32)),
            ISETP_GE => self.setp(w, &inst, mask, a, b, |x, y| (x as i32) >= (y as i32)),
            FSETP_EQ => self.setp(w, &inst, mask, a, b, |x, y| f32::from_bits(x) == f32::from_bits(y)),
            FSETP_NE => self.setp(w, &inst, mask, a, b, |x, y| f32::from_bits(x) != f32::from_bits(y)),
            FSETP_LT => self.setp(w, &inst, mask, a, b, |x, y| f32::from_bits(x) < f32::from_bits(y)),
            FSETP_LE => self.setp(w, &inst, mask, a, b, |x, y| f32::from_bits(x) <= f32::from_bits(y)),
            FSETP_GT => self.setp(w, &inst, mask, a, b, |x, y| f32::from_bits(x) > f32::from_bits(y)),
            FSETP_GE => self.setp(w, &inst, mask, a, b, |x, y| f32::from_bits(x) >= f32::from_bits(y)),
            LDG | LDS | LDP => {
                let addrs = &w.regs[reg(inst.ra)];
                let load = |cell: &AtomicU32| cell.load(Relaxed);
                // Whole warps take a loop with a fixed trip count.
                let (space, values) = if mask == FULL {
                    let all = 0..WARP_SIZE as usize;
                    match inst.op {
                        LDG => ("global", gather(self.memory, addrs, imm, all, load)),
                        LDS => ("shared", gather(&self.shared, addrs, imm, all, load)),
                        _ => ("param", gather(self.launch.params, addrs, imm, all, |x| *x)),
                    }
                } else {
                    let some = lanes(mask);
                    match inst.op {
                        LDG => ("global", gather(self.memory, addrs, imm, some, load)),
                        LDS => ("shared", gather(&self.shared, addrs, imm, some, load)),
                        _ => ("param", gather(self.launch.params, addrs, imm, some, |x| *x)),
                    }
                };
                let values = values.map_err(|addr| Error::Memory { pc, space, addr })?;
                w.set_lanes(inst.rd, mask, &values);
            }
            STG | STS => {
                let addrs = &w.regs[reg(inst.ra)];
                let values = &w.regs[reg(inst.rb)];
                let store = |cell: &AtomicU32, value| cell.store(value, Relaxed);
                let (space, result) = if mask == FULL {
                    let all = 0..WARP_SIZE as usize;
                    match inst.op {
                        STG => ("global", scatter(self.memory, addrs, values, imm, all, store)),
                        _ => ("shared", scatter(&self.shared, addrs, values, imm, all, store)),
                    }
                } else {
                    let some = lanes(mask);
                    match inst.op {
                        STG => ("global", scatter(self.memory, addrs, values, imm, some, store)),
                        _ => ("shared", scatter(&self.shared, addrs, values, imm, some, store)),
                    }
                };
                result.map_err(|addr| Error::Memory { pc, space, addr })?;
            }
            SHFL_IDX | SHFL_BFLY => {
                // Every lane reads its source before any lane writes.
                let src = w.regs[reg(inst.ra)];
                for lane in lanes(mask) {
                    let offset = w.get(b, lane) & 31;
                    let from = match inst.op {
                        SHFL_IDX => offset as usize,
                        _ => lane ^ offset as usize,
                    };
                    let from = if w.live >> from & 1 == 1 { from } else { lane };
                    w.set(inst.rd, lane, src[from]);
                }
            }
            BRA => {
                if mask == w.live {
                    w.pc = imm as usize;
                } else if mask != 0 {
                    return Err(Error::DivergentBranch { pc });
                }
            }
            BAR => {
                if mask == w.live {
                    w.state = State::Waiting;
                } else if mask != 0 {
                    return Err(Error::DivergentBarrier { pc });
                }
            }
            EXIT => {
                w.live &= !mask;
                if w.live == 0 {
                    w.state = State::Finished;
                }
            }
            _ => unreachable!("decode rejects unknown opcodes"),
        }
        Ok(())
    }

    /// Executes an arithmetic instruction: `rd = f(a, b, c)` in every lane.
    #[allow(clippy::too_many_arguments)]
    fn alu(
        &self,
        w: &mut Warp,
        inst: &Inst,
        mask: u32,
        a: Src,
        b: Src,
        c: Src,
        f: impl Fn(u32, u32, u32) -> u32,
    ) {
        if mask == FULL {
            // Every lane is active: compute the whole row at once, which the
            // host can do with its own vector instructions.
            let (a, b, c) = (w.row(a), w.row(b), w.row(c));
            let out = if self.simd {
                // SAFETY: `simd` is set only when the host has every
                // extension `map_simd` is compiled for.
                unsafe { map_simd(a, b, c, f) }
            } else {
                map(a, b, c, f)
            };
            w.set_lanes(inst.rd, FULL, &out);
            return;
        }
        for lane in lanes(mask) {
            let value = f(w.get(a, lane), w.get(b, lane), w.get(c, lane));
            w.set(inst.rd, lane, value);
        }
    }

    /// Executes a comparison: `pd = f(a, b)` in every lane.
    fn setp(&self, w: &mut Warp, inst: &Inst, mask: u32, a: Src, b: Src, f: impl Fn(u32, u32) -> bool) {
        let mut bits = 0;
        if mask == FULL {
            let (a, b) = (w.row(a), w.row(b));
            for lane in 0..WARP_SIZE as usize {
                bits |= (f(a[lane], b[lane]) as u32) << lane;
            }
        } else {
            for lane in lanes(mask) {
                bits |= (f(w.get(a, lane), w.get(b, lane)) as u32) << lane;
            }
        }
        if inst.rd != PT {
            let pred = &mut w.preds[inst.rd as usize];
            *pred = (*pred & !mask) | bits;
        }
    }

}

/// The word at byte address `addr` of a memory space, if it's aligned and in
/// range.
fn word<T>(space: &[T], addr: u32) -> Option<&T> {
    if !addr.is_multiple_of(4) {
        return None;
    }
    space.get(addr as usize / 4)
}

fn canonical(x: f32) -> u32 {
    if x.is_nan() { CANONICAL_NAN } else { x.to_bits() }
}

/// `FMIN`: `-0.0` is less than `+0.0`, and a NaN operand yields the other.
fn fmin(a: u32, b: u32) -> u32 {
    let (x, y) = (f32::from_bits(a), f32::from_bits(b));
    match (x.is_nan(), y.is_nan()) {
        (true, true) => CANONICAL_NAN,
        (true, false) => b,
        (false, true) => a,
        _ if x < y => a,
        _ if y < x => b,
        // Equal: either the same bits, or zeros of opposite signs.
        _ => a | b,
    }
}

/// `FMAX`: `+0.0` is greater than `-0.0`, and a NaN operand yields the other.
fn fmax(a: u32, b: u32) -> u32 {
    let (x, y) = (f32::from_bits(a), f32::from_bits(b));
    match (x.is_nan(), y.is_nan()) {
        (true, true) => CANONICAL_NAN,
        (true, false) => b,
        (false, true) => a,
        _ if x > y => a,
        _ if y > x => b,
        _ => a & b,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Encodes an unguarded instruction.
    fn inst(op: u8, rd: u8, ra: u8, rb: u8, imm: Option<u32>) -> u64 {
        let low = op as u64
            | (PT as u64) << 8
            | (rd as u64) << 12
            | (ra as u64) << 18
            | (rb as u64) << 24
            | (imm.is_some() as u64) << 30;
        low | (imm.unwrap_or(0) as u64) << 32
    }

    /// Each of 64 threads writes the sum of its thread number across its
    /// warp, reduced with butterfly shuffles, to `out[tid]`.
    #[test]
    fn warp_sum() {
        let mut sim = Simulator::new();
        let out = sim.alloc(64 * 4);
        let mut program = vec![inst(S2R, 1, 0, 0, Some(SR_TID)), inst(MOV, 2, 1, 0, None)];
        for offset in [16, 8, 4, 2, 1] {
            program.push(inst(SHFL_BFLY, 3, 2, 0, Some(offset)));
            program.push(inst(IADD, 2, 2, 3, None));
        }
        program.extend([
            inst(LDP, 4, 0, 0, Some(0)),
            inst(SHL, 5, 1, 0, Some(2)),
            inst(IADD, 4, 4, 5, None),
            inst(STG, 0, 4, 2, Some(0)),
            inst(EXIT, 0, 0, 0, None),
        ]);
        sim.launch(&Launch { program: &program, grid: [1, 1], block: 64, shared: 0, params: &[out] })
            .unwrap();
        let expected: Vec<u32> = (0..64).map(|tid| if tid < 32 { 496 } else { 1520 }).collect();
        assert_eq!(sim.read(out, 64), expected);
    }

    /// One thread per block writes `x + 10y` to `out[x + 4y]` in a 4×3 grid.
    #[test]
    fn grid_is_two_dimensional() {
        let mut sim = Simulator::new();
        let out = sim.alloc(12 * 4);
        let program = [
            inst(S2R, 1, 0, 0, Some(SR_CTAID_X)),
            inst(S2R, 2, 0, 0, Some(SR_CTAID_Y)),
            inst(S2R, 3, 0, 0, Some(SR_NCTAID_X)),
            // r4 = r2 * r3 + r1, with `rc` in the high word.
            inst(IMAD, 4, 2, 3, None) | (1 << 32),
            inst(SHL, 5, 4, 0, Some(2)),
            inst(LDP, 6, 0, 0, Some(0)),
            inst(IADD, 6, 6, 5, None),
            inst(IMUL, 7, 2, 0, Some(10)),
            inst(IADD, 7, 7, 1, None),
            inst(STG, 0, 6, 7, Some(0)),
            inst(EXIT, 0, 0, 0, None),
        ];
        sim.launch(&Launch { program: &program, grid: [4, 3], block: 1, shared: 0, params: &[out] })
            .unwrap();
        let expected: Vec<u32> = (0..3).flat_map(|y| (0..4).map(move |x| x + 10 * y)).collect();
        assert_eq!(sim.read(out, 12), expected);
    }

    #[test]
    fn divergent_branch_is_an_error() {
        let sim = Simulator::new();
        // Guarded by p0 rather than pt.
        let branch = inst(BRA, 0, 0, 0, Some(3)) & !(7 << 8);
        let program = [
            inst(S2R, 1, 0, 0, Some(SR_TID)),
            inst(ISETP_LT, 0, 1, 0, Some(16)),
            branch,
            inst(EXIT, 0, 0, 0, None),
        ];
        let result = sim.launch(&Launch { program: &program, grid: [1, 1], block: 32, shared: 0, params: &[] });
        assert!(matches!(result, Err(Error::DivergentBranch { pc: 2 })));
    }

    #[test]
    fn decode_rejects_invalid_instructions() {
        assert!(decode(0).is_none(), "unknown opcode");
        assert!(decode(inst(EXIT, 1, 0, 0, None)).is_none(), "EXIT with a destination");
        assert!(decode(inst(LDG, 1, 2, 0, None)).is_none(), "load without an offset");
        assert!(decode(inst(S2R, 1, 0, 0, Some(6))).is_none(), "unknown special register");
        assert!(decode(inst(FFMA, 1, 2, 3, Some(0))).is_some());
    }

    #[test]
    fn min_max_zeros_and_nans() {
        let (pos, neg) = (0.0f32.to_bits(), (-0.0f32).to_bits());
        assert_eq!(fmin(pos, neg), neg);
        assert_eq!(fmax(neg, pos), pos);
        assert_eq!(fmin(f32::NAN.to_bits(), 1.0f32.to_bits()), 1.0f32.to_bits());
        assert_eq!(fmax(f32::NAN.to_bits(), f32::NAN.to_bits()), CANONICAL_NAN);
    }
}
