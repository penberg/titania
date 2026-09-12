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

use titania_compiler::kernels::TILE;
use titania_compiler::{Kernel, kernels};
use titania_model::{Device, Tensor};
use titania_simulator::{Activity, Launch, Simulator};

/// A Titania GPU, simulated.
pub struct Titania {
    sim: RefCell<Simulator>,
    /// Compiled kernels, by operation and shape.
    kernels: RefCell<HashMap<Op, Rc<Compiled>>>,
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
    Rope { n_heads: usize, head_dim: usize, n: usize },
    Attention { n_heads: usize, head_dim: usize, n_kv_heads: usize, max_len: usize, n: usize },
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
            Op::Rope { n_heads, head_dim, n } => kernels::rope(n_heads, head_dim, n),
            Op::Attention { n_heads, head_dim, n_kv_heads, max_len, n } => {
                kernels::attention(n_heads, head_dim, n_kv_heads, max_len, n)
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
            Op::Rope { n_heads, head_dim, n } => write!(f, "rope {n_heads}×{head_dim}{}", tokens(*n)),
            Op::Attention { n_heads, head_dim, n, .. } => {
                write!(f, "attention {n_heads}×{head_dim}{}", tokens(*n))
            }
            Op::SiluMul(n) => write!(f, "silu_mul {n}"),
        }
    }
}

/// How many tokens an operation runs at once, when more than one.
fn tokens(n: usize) -> String {
    if n > 1 { format!(" · {n} tokens") } else { String::new() }
}

/// A buffer of `f32` activations in global memory.
pub struct Buffer {
    addr: u32,
    len: usize,
    /// The length it was allocated with.
    capacity: usize,
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
            capacity: len,
        }
    }

    fn resize(&self, buf: &mut Buffer, len: usize) {
        assert!(len <= buf.capacity, "a buffer of {} activations can't hold {len}", buf.capacity);
        buf.len = len;
    }

    fn read(&self, buf: &Buffer) -> Vec<f32> {
        self.sim.borrow().read(buf.addr, buf.len).into_iter().map(f32::from_bits).collect()
    }

    fn write(&self, buf: &mut Buffer, data: &[f32]) {
        assert_eq!(data.len(), buf.len);
        let words: Vec<u32> = data.iter().map(|x| x.to_bits()).collect();
        self.sim.borrow().write(buf.addr, &words);
    }

    fn copy(&self, dst: &mut Buffer, dst_offset: usize, src: &Buffer, src_offset: usize, len: usize) {
        assert!(dst_offset + len <= dst.len && src_offset + len <= src.len);
        let params = [dst.addr + (dst_offset * 4) as u32, src.addr + (src_offset * 4) as u32];
        self.run(Op::Copy(len), &params);
    }

    fn embed(&self, out: &mut Buffer, table: &Weight, tokens: &[u32]) {
        let dim = table.shape[1];
        assert_eq!(out.len, tokens.len() * dim);
        for (t, &token) in tokens.iter().enumerate() {
            self.run(Op::Embed(dim), &[out.addr + (t * dim * 4) as u32, table.addr, token]);
        }
    }

    fn matmul(&self, out: &mut Buffer, w: &Weight, x: &Buffer) {
        let (rows, cols) = (w.shape[0], w.shape[1]);
        let n = x.len / cols;
        assert_eq!(x.len, n * cols);
        assert_eq!(out.len, n * rows);
        // The kernel takes whole tiles of tokens, or a batch smaller than a
        // tile: the full tiles go in one launch and the rest in another.
        let full = n / TILE * TILE;
        if full > 0 {
            self.run(Op::Matmul { rows, cols, n: full }, &[out.addr, w.addr, x.addr]);
        }
        if n > full {
            let params = [out.addr + (full * rows * 4) as u32, w.addr, x.addr + (full * cols * 4) as u32];
            self.run(Op::Matmul { rows, cols, n: n - full }, &params);
        }
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

    fn rope(&self, x: &mut Buffer, table: &Buffer, pos: usize, n_heads: usize, head_dim: usize) {
        let n = x.len / (n_heads * head_dim);
        assert_eq!(x.len, n * n_heads * head_dim);
        let last = pos + n - 1;
        assert!((last + 1) * head_dim <= table.len, "position {last} is past the end of the table");
        self.run(Op::Rope { n_heads, head_dim, n }, &[x.addr, table.addr, pos as u32]);
    }

    fn attention(
        &self,
        out: &mut Buffer,
        q: &Buffer,
        k_cache: &Buffer,
        v_cache: &Buffer,
        pos: usize,
        n_heads: usize,
        head_dim: usize,
        n_kv_heads: usize,
    ) {
        let n = q.len / (n_heads * head_dim);
        assert_eq!(q.len, n * n_heads * head_dim);
        assert_eq!(out.len, q.len);
        let op = Op::Attention {
            n_heads,
            head_dim,
            n_kv_heads,
            max_len: k_cache.len / (n_kv_heads * head_dim),
            n,
        };
        let params = [out.addr, q.addr, k_cache.addr, v_cache.addr, pos as u32];
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
        let mut buf = gpu.alloc(n);
        gpu.write(&mut buf, &values);
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
        Cpu.copy(&mut cpu_dst, 7, &cpu_x, 5, n - 5);
        gpu.copy(&mut dst, 7, &x, 5, n - 5);
        assert_close(&gpu, &dst, &cpu_dst);
    }

    #[test]
    fn resize() {
        let gpu = Titania::new();
        let (mut cpu_x, mut x) = buffers(&gpu, 300, 1);
        let (cpu_y, y) = buffers(&gpu, 100, 2);
        Cpu.resize(&mut cpu_x, 100);
        gpu.resize(&mut x, 100);
        Cpu.add(&mut cpu_x, &cpu_y);
        gpu.add(&mut x, &y);
        assert_close(&gpu, &x, &cpu_x);
        Cpu.resize(&mut cpu_x, 300);
        gpu.resize(&mut x, 300);
        assert_eq!(gpu.read(&x).len(), 300);
    }

    #[test]
    fn embed() {
        let gpu = Titania::new();
        let table = tensor(&[10, 128], 4);
        let tokens = [3, 9, 0];
        let (mut cpu_out, mut out) = buffers(&gpu, 3 * 128, 5);
        Cpu.embed(&mut cpu_out, &table, &tokens);
        let table = gpu.upload(table);
        gpu.embed(&mut out, &table, &tokens);
        assert_close(&gpu, &out, &cpu_out);
    }

    /// Batches of one token, a partial tile, whole tiles, and whole tiles
    /// with a remainder.
    #[test]
    fn matmul() {
        for (rows, cols, n) in [(24, 256, 1), (5, 64, 3), (9, 384, 16), (24, 128, 19)] {
            let gpu = Titania::new();
            let w = tensor(&[rows, cols], 6);
            let (cpu_x, x) = buffers(&gpu, n * cols, 7);
            let (mut cpu_out, mut out) = buffers(&gpu, n * rows, 8);
            Cpu.matmul(&mut cpu_out, &w, &cpu_x);
            let w = gpu.upload(w);
            gpu.matmul(&mut out, &w, &x);
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
        for n in [1, 3] {
            let gpu = Titania::new();
            let (mut cpu_x, mut x) = buffers(&gpu, n * 4 * 128, 11);
            let (cpu_table, table) = buffers(&gpu, 40 * 128, 16);
            Cpu.rope(&mut cpu_x, &cpu_table, 37, 4, 128);
            gpu.rope(&mut x, &table, 37, 4, 128);
            assert_close(&gpu, &x, &cpu_x);
        }
    }

    #[test]
    fn attention() {
        let (n_heads, head_dim, n_kv_heads, max_len) = (4, 64, 2, 40);
        for (pos, n) in [(0, 1), (6, 1), (32, 1), (39, 1), (0, 5), (30, 10)] {
            let gpu = Titania::new();
            let kv_len = max_len * n_kv_heads * head_dim;
            let (cpu_q, q) = buffers(&gpu, n * n_heads * head_dim, 12);
            let (cpu_k, k) = buffers(&gpu, kv_len, 13);
            let (cpu_v, v) = buffers(&gpu, kv_len, 14);
            let (mut cpu_out, mut out) = buffers(&gpu, n * n_heads * head_dim, 15);
            Cpu.attention(&mut cpu_out, &cpu_q, &cpu_k, &cpu_v, pos, n_heads, head_dim, n_kv_heads);
            gpu.attention(&mut out, &q, &k, &v, pos, n_heads, head_dim, n_kv_heads);
            assert_close(&gpu, &out, &cpu_out);
        }
    }
}
