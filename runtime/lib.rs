//! Runs the model on a Titania GPU, here simulated by the ISA simulator.
//!
//! [`Titania`] implements the model's [`Device`] operations by compiling each
//! one, the first time it is used with a given shape, into a Titania kernel,
//! and launching it. A [`Monitor`] watches it do so from another thread.

use std::cell::RefCell;
use std::collections::HashMap;
use std::fmt;
use std::rc::Rc;
use std::sync::{Arc, Mutex};

use titania_compiler::{Kernel, kernels};
use titania_model::{Device, Tensor};
use titania_simulator::{Activity, Launch, Simulator};

/// A Titania GPU, simulated.
pub struct Titania {
    sim: RefCell<Simulator>,
    /// Compiled kernels, by operation and shape.
    kernels: RefCell<HashMap<Op, Rc<Compiled>>>,
    /// Rotary position embedding tables, by head size and base frequency.
    rope_tables: RefCell<HashMap<(usize, u32), RopeTable>>,
    monitor: Arc<Monitor>,
}

/// What a [`Titania`] is running, for another thread to watch.
pub struct Monitor {
    kernel: Mutex<Option<Arc<KernelInfo>>>,
    activity: Arc<Activity>,
}

impl Monitor {
    /// The kernel launched last: the one running, unless the GPU is idle.
    pub fn kernel(&self) -> Option<Arc<KernelInfo>> {
        self.kernel.lock().unwrap().clone()
    }

    /// What the simulator is doing.
    pub fn activity(&self) -> &Activity {
        &self.activity
    }
}

/// A compiled kernel, as a [`Monitor`] shows it.
pub struct KernelInfo {
    /// The operation the kernel computes, with its shape.
    pub name: String,
    /// Blocks along x and y.
    pub grid: [u32; 2],
    pub block: u32,
    /// Bytes of shared memory per block.
    pub shared: u32,
    /// The kernel's instructions in assembly syntax, by PC.
    pub listing: Vec<String>,
}

/// A kernel compiled for an operation.
struct Compiled {
    kernel: Kernel,
    info: Arc<KernelInfo>,
}

impl Compiled {
    fn new(op: Op) -> Self {
        let kernel = op.compile();
        let info = Arc::new(KernelInfo {
            name: op.to_string(),
            grid: kernel.grid,
            block: kernel.block,
            shared: kernel.shared,
            listing: kernel.instructions.iter().map(ToString::to_string).collect(),
        });
        Self { kernel, info }
    }
}

/// An operation, specialized to its shapes: what a kernel is compiled for.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
enum Op {
    Copy(usize),
    Embed(usize),
    Matmul { rows: usize, cols: usize, n: usize },
    Add(usize),
    Rmsnorm { rows: usize, dim: usize, eps: u32 },
    Rope { n_heads: usize, head_dim: usize },
    Attention { n_heads: usize, head_dim: usize, n_kv_heads: usize, max_len: usize },
    SiluMul(usize),
}

impl Op {
    fn compile(self) -> Kernel {
        match self {
            Op::Copy(n) => kernels::copy(n),
            Op::Embed(dim) => kernels::embed(dim),
            Op::Matmul { rows, cols, n } => kernels::matmul(rows, cols, n),
            Op::Add(n) => kernels::add(n),
            Op::Rmsnorm { rows, dim, eps } => kernels::rmsnorm(rows, dim, f32::from_bits(eps)),
            Op::Rope { n_heads, head_dim } => kernels::rope(n_heads, head_dim),
            Op::Attention { n_heads, head_dim, n_kv_heads, max_len } => {
                kernels::attention(n_heads, head_dim, n_kv_heads, max_len)
            }
            Op::SiluMul(n) => kernels::silu_mul(n),
        }
    }
}

impl fmt::Display for Op {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Op::Copy(n) => write!(f, "copy {n}"),
            Op::Embed(dim) => write!(f, "embed {dim}"),
            Op::Matmul { rows, cols, n } => write!(f, "matmul {rows}×{cols}{}", tokens(*n)),
            Op::Add(n) => write!(f, "add {n}"),
            Op::Rmsnorm { rows, dim, .. } => write!(f, "rmsnorm {rows}×{dim}"),
            Op::Rope { n_heads, head_dim } => write!(f, "rope {n_heads}×{head_dim}"),
            Op::Attention { n_heads, head_dim, .. } => write!(f, "attention {n_heads}×{head_dim}"),
            Op::SiluMul(n) => write!(f, "silu_mul {n}"),
        }
    }
}

/// How many tokens an operation runs at once, when more than one.
fn tokens(n: usize) -> String {
    if n > 1 { format!(" · {n} tokens") } else { String::new() }
}

/// Cosines and sines of every rotation angle, for positions up to
/// `positions`.
struct RopeTable {
    addr: u32,
    positions: usize,
}

/// A buffer of `f32` activations in global memory.
pub struct Buffer {
    addr: u32,
    len: usize,
}

/// A bf16 weight tensor in global memory.
pub struct Weight {
    addr: u32,
    shape: Vec<usize>,
}

impl Titania {
    pub fn new() -> Self {
        let sim = Simulator::new();
        let monitor = Arc::new(Monitor {
            kernel: Mutex::new(None),
            activity: sim.activity(),
        });
        Self {
            sim: RefCell::new(sim),
            kernels: RefCell::default(),
            rope_tables: RefCell::default(),
            monitor,
        }
    }

    /// A monitor, to watch the GPU from another thread.
    pub fn monitor(&self) -> Arc<Monitor> {
        self.monitor.clone()
    }

    /// Runs an operation, compiling it first if this is its first use.
    fn run(&self, op: Op, params: &[u32]) {
        let compiled = self
            .kernels
            .borrow_mut()
            .entry(op)
            .or_insert_with(|| Rc::new(Compiled::new(op)))
            .clone();
        *self.monitor.kernel.lock().unwrap() = Some(compiled.info.clone());
        let kernel = &compiled.kernel;
        let launch = Launch {
            program: &kernel.program,
            grid: kernel.grid,
            block: kernel.block,
            shared: kernel.shared,
            params,
        };
        if let Err(e) = self.sim.borrow().launch(&launch) {
            panic!("{op:?} failed: {e}\n{}", kernel.disassemble());
        }
    }

    /// The address of a table with the rotation angles for positions up to at
    /// least `pos`, computed as the CPU computes them.
    fn rope_table(&self, head_dim: usize, theta: f32, pos: usize) -> u32 {
        let key = (head_dim, theta.to_bits());
        let mut tables = self.rope_tables.borrow_mut();
        if let Some(table) = tables.get(&key)
            && pos < table.positions
        {
            return table.addr;
        }
        let positions = (pos + 1).next_power_of_two().max(4096);
        let half = head_dim / 2;
        let mut words = Vec::with_capacity(positions * half * 2);
        for p in 0..positions {
            for i in 0..half {
                let freq = 1.0 / theta.powf((2 * i) as f32 / head_dim as f32);
                let (sin, cos) = (p as f32 * freq).sin_cos();
                words.extend([cos.to_bits(), sin.to_bits()]);
            }
        }
        let addr = self.upload_words(&words);
        tables.insert(key, RopeTable { addr, positions });
        addr
    }

    fn upload_words(&self, words: &[u32]) -> u32 {
        let mut sim = self.sim.borrow_mut();
        let addr = sim.alloc(words.len() * 4);
        sim.write(addr, words);
        addr
    }
}

impl Default for Titania {
    fn default() -> Self {
        Self::new()
    }
}

impl Device for Titania {
    type Buffer = Buffer;
    type Weight = Weight;

    fn upload(&self, tensor: Tensor) -> Weight {
        // Two bf16 numbers per word, the first in the lower half.
        let words: Vec<u32> = tensor
            .data
            .chunks(2)
            .map(|pair| pair[0] as u32 | (pair.get(1).copied().unwrap_or(0) as u32) << 16)
            .collect();
        Weight {
            addr: self.upload_words(&words),
            shape: tensor.shape,
        }
    }

    fn alloc(&self, len: usize) -> Buffer {
        Buffer {
            addr: self.sim.borrow_mut().alloc(len * 4),
            len,
        }
    }

    fn read(&self, buf: &Buffer) -> Vec<f32> {
        self.sim.borrow().read(buf.addr, buf.len).into_iter().map(f32::from_bits).collect()
    }

    fn copy(&self, dst: &mut Buffer, offset: usize, src: &Buffer) {
        assert!(offset + src.len <= dst.len);
        self.run(Op::Copy(src.len), &[dst.addr + (offset * 4) as u32, src.addr]);
    }

    fn embed(&self, out: &mut Buffer, table: &Weight, token: usize) {
        assert_eq!(table.shape[1], out.len);
        self.run(Op::Embed(out.len), &[out.addr, table.addr, token as u32]);
    }

    fn matvec(&self, out: &mut Buffer, w: &Weight, x: &Buffer) {
        assert_eq!(w.shape, [out.len, x.len]);
        self.run(Op::Matmul { rows: out.len, cols: x.len, n: 1 }, &[out.addr, w.addr, x.addr]);
    }

    fn add(&self, x: &mut Buffer, y: &Buffer) {
        self.run(Op::Add(x.len), &[x.addr, y.addr]);
    }

    fn rmsnorm(&self, x: &mut Buffer, weight: &Weight, eps: f32) {
        let dim = weight.shape[0];
        let op = Op::Rmsnorm {
            rows: x.len / dim,
            dim,
            eps: eps.to_bits(),
        };
        self.run(op, &[x.addr, weight.addr]);
    }

    fn rope(&self, x: &mut Buffer, pos: usize, head_dim: usize, theta: f32) {
        let table = self.rope_table(head_dim, theta, pos);
        let op = Op::Rope {
            n_heads: x.len / head_dim,
            head_dim,
        };
        self.run(op, &[x.addr, table, pos as u32]);
    }

    fn attention(
        &self,
        out: &mut Buffer,
        q: &Buffer,
        k_cache: &Buffer,
        v_cache: &Buffer,
        len: usize,
        head_dim: usize,
        n_kv_heads: usize,
    ) {
        let op = Op::Attention {
            n_heads: q.len / head_dim,
            head_dim,
            n_kv_heads,
            max_len: k_cache.len / (n_kv_heads * head_dim),
        };
        let params = [out.addr, q.addr, k_cache.addr, v_cache.addr, len as u32];
        self.run(op, &params);
    }

    fn silu_mul(&self, gate: &mut Buffer, up: &Buffer) {
        self.run(Op::SiluMul(gate.len), &[gate.addr, up.addr]);
    }
}

/// Checks every operation against the CPU, the reference implementation.
#[cfg(test)]
mod tests {
    use super::*;
    use titania_model::Cpu;

    /// Pseudorandom numbers in `[-1, 1)`.
    fn numbers(n: usize, seed: u32) -> Vec<f32> {
        let mut state = seed.wrapping_mul(2_654_435_761).max(1);
        (0..n)
            .map(|_| {
                state ^= state << 13;
                state ^= state >> 17;
                state ^= state << 5;
                (state >> 8) as f32 / (1 << 23) as f32 - 1.0
            })
            .collect()
    }

    fn tensor(shape: &[usize], seed: u32) -> Tensor {
        let n = shape.iter().product();
        Tensor {
            shape: shape.to_vec(),
            data: numbers(n, seed).iter().map(|x| (x.to_bits() >> 16) as u16).collect(),
        }
    }

    /// A buffer on each device, holding the same numbers.
    fn buffers(gpu: &Titania, n: usize, seed: u32) -> (Vec<f32>, Buffer) {
        let values = numbers(n, seed);
        let words: Vec<u32> = values.iter().map(|x| x.to_bits()).collect();
        let buf = Buffer { addr: gpu.upload_words(&words), len: n };
        (values, buf)
    }

    fn assert_close(gpu: &Titania, actual: &Buffer, expected: &[f32]) {
        let actual = gpu.read(actual);
        assert_eq!(actual.len(), expected.len());
        for (i, (a, e)) in actual.iter().zip(expected).enumerate() {
            let tolerance = 1e-5 + 1e-4 * e.abs();
            assert!((a - e).abs() <= tolerance, "element {i}: got {a}, expected {e}");
        }
    }

    #[test]
    fn elementwise() {
        let gpu = Titania::new();
        let n = 300;
        let (mut cpu_x, mut x) = buffers(&gpu, n, 1);
        let (cpu_y, y) = buffers(&gpu, n, 2);
        Cpu.add(&mut cpu_x, &cpu_y);
        gpu.add(&mut x, &y);
        assert_close(&gpu, &x, &cpu_x);

        Cpu.silu_mul(&mut cpu_x, &cpu_y);
        gpu.silu_mul(&mut x, &y);
        assert_close(&gpu, &x, &cpu_x);

        let (mut cpu_dst, mut dst) = buffers(&gpu, n + 10, 3);
        Cpu.copy(&mut cpu_dst, 7, &cpu_x);
        gpu.copy(&mut dst, 7, &x);
        assert_close(&gpu, &dst, &cpu_dst);
    }

    #[test]
    fn embed() {
        let gpu = Titania::new();
        let table = tensor(&[10, 128], 4);
        let (mut cpu_out, mut out) = buffers(&gpu, 128, 5);
        Cpu.embed(&mut cpu_out, &table, 3);
        let table = gpu.upload(table);
        gpu.embed(&mut out, &table, 3);
        assert_close(&gpu, &out, &cpu_out);
    }

    /// One token, a partial tile, and whole tiles.
    #[test]
    fn matmul() {
        for (rows, cols, n) in [(24, 256, 1), (5, 64, 3), (9, 384, 16)] {
            let gpu = Titania::new();
            let w = tensor(&[rows, cols], 6);
            let (cpu_x, x) = buffers(&gpu, n * cols, 7);
            let (mut cpu_out, mut out) = buffers(&gpu, n * rows, 8);
            // The CPU multiplies one token at a time.
            for (out, x) in cpu_out.chunks_exact_mut(rows).zip(cpu_x.chunks_exact(cols)) {
                let mut result = vec![0.0; rows];
                Cpu.matvec(&mut result, &w, &x.to_vec());
                out.copy_from_slice(&result);
            }
            let w = gpu.upload(w);
            // The device runs the kernel for one token; only the kernel
            // takes a tile so far.
            match n {
                1 => gpu.matvec(&mut out, &w, &x),
                _ => gpu.run(Op::Matmul { rows, cols, n }, &[out.addr, w.addr, x.addr]),
            }
            assert_close(&gpu, &out, &cpu_out);
        }
    }

    #[test]
    fn rmsnorm() {
        for (rows, dim) in [(3, 128), (1, 1024), (9, 64)] {
            let gpu = Titania::new();
            let weight = tensor(&[dim], 9);
            let (mut cpu_x, mut x) = buffers(&gpu, rows * dim, 10);
            Cpu.rmsnorm(&mut cpu_x, &weight, 1e-6);
            let weight = gpu.upload(weight);
            gpu.rmsnorm(&mut x, &weight, 1e-6);
            assert_close(&gpu, &x, &cpu_x);
        }
    }

    #[test]
    fn rope() {
        let gpu = Titania::new();
        let (mut cpu_x, mut x) = buffers(&gpu, 4 * 128, 11);
        Cpu.rope(&mut cpu_x, 37, 128, 1_000_000.0);
        gpu.rope(&mut x, 37, 128, 1_000_000.0);
        assert_close(&gpu, &x, &cpu_x);
    }

    #[test]
    fn attention() {
        let (n_heads, head_dim, n_kv_heads, max_len) = (4, 64, 2, 40);
        for len in [1, 7, 33, 40] {
            let gpu = Titania::new();
            let kv_len = max_len * n_kv_heads * head_dim;
            let (cpu_q, q) = buffers(&gpu, n_heads * head_dim, 12);
            let (cpu_k, k) = buffers(&gpu, kv_len, 13);
            let (cpu_v, v) = buffers(&gpu, kv_len, 14);
            let (mut cpu_out, mut out) = buffers(&gpu, n_heads * head_dim, 15);
            Cpu.attention(&mut cpu_out, &cpu_q, &cpu_k, &cpu_v, len, head_dim, n_kv_heads);
            gpu.attention(&mut out, &q, &k, &v, len, head_dim, n_kv_heads);
            assert_close(&gpu, &out, &cpu_out);
        }
    }
}
