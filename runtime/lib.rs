//! Runs the model on a Titania GPU, here simulated by the ISA simulator.
//!
//! [`Titania`] implements the model's [`Device`] operations by compiling each
//! one, the first time it is used with a given shape, into a Titania kernel,
//! and launching it.

use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;

use titania_compiler::{Kernel, kernels};
use titania_model::{Device, Tensor};
use titania_simulator::{Launch, Simulator};

/// A Titania GPU, simulated.
#[derive(Default)]
pub struct Titania {
    sim: RefCell<Simulator>,
    /// Compiled kernels, by operation and shape.
    kernels: RefCell<HashMap<Op, Rc<Kernel>>>,
    /// Rotary position embedding tables, by head size and base frequency.
    rope_tables: RefCell<HashMap<(usize, u32), RopeTable>>,
}

/// An operation, specialized to its shapes: what a kernel is compiled for.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
enum Op {
    Copy(usize),
    Embed(usize),
    Matvec(usize, usize),
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
            Op::Matvec(rows, cols) => kernels::matvec(rows, cols),
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
        Self::default()
    }

    /// Runs an operation, compiling it first if this is its first use.
    fn run(&self, op: Op, params: &[u32]) {
        let kernel = self
            .kernels
            .borrow_mut()
            .entry(op)
            .or_insert_with(|| Rc::new(op.compile()))
            .clone();
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
        self.run(Op::Matvec(out.len, x.len), &[out.addr, w.addr, x.addr]);
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

    #[test]
    fn matvec() {
        for (rows, cols) in [(24, 256), (5, 64), (9, 384)] {
            let gpu = Titania::new();
            let w = tensor(&[rows, cols], 6);
            let (cpu_x, x) = buffers(&gpu, cols, 7);
            let (mut cpu_out, mut out) = buffers(&gpu, rows, 8);
            Cpu.matvec(&mut cpu_out, &w, &cpu_x);
            let w = gpu.upload(w);
            gpu.matvec(&mut out, &w, &x);
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
