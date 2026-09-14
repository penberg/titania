//! Runs the model on a Titania GPU.
//!
//! [`Titania`] implements the model's [`Device`] operations by compiling each
//! one, the first time it is used with a given shape, into a Titania kernel,
//! and launching it on a [`Gpu`]: the ISA simulator, or anything else that
//! runs Titania programs. A [`Monitor`] watches it do so from another thread.

use std::{
    cell::RefCell,
    sync::{Arc, Mutex},
};

use titania_compiler::{Kernel, Kernels, Op, kernels::TILE};
use titania_gpu::{Activity, Gpu, Launch};
use titania_model::{Device, Tensor};

/// A Titania GPU running the model.
pub struct Titania<G: Gpu> {
    gpu: RefCell<G>,
    kernels: Kernels,
    monitor: Arc<Monitor>,
}

/// What a [`Titania`] is running, for another thread to watch.
pub struct Monitor {
    kernel: Mutex<Option<Arc<Kernel>>>,
    activity: Arc<Activity>,
}

impl Monitor {
    /// The kernel launched last: the one running, unless the GPU is idle.
    pub fn kernel(&self) -> Option<Arc<Kernel>> {
        self.kernel.lock().unwrap().clone()
    }

    /// What the GPU is doing.
    pub fn activity(&self) -> &Activity {
        &self.activity
    }
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

impl<G: Gpu> Titania<G> {
    pub fn new(gpu: G) -> Self {
        let monitor = Arc::new(Monitor {
            kernel: Mutex::new(None),
            activity: gpu.activity(),
        });
        Self {
            gpu: RefCell::new(gpu),
            kernels: Kernels::new(),
            monitor,
        }
    }

    /// A monitor, to watch the GPU from another thread.
    pub fn monitor(&self) -> Arc<Monitor> {
        self.monitor.clone()
    }

    /// Runs an operation, compiling it first if this is its first use.
    fn run(&self, op: Op, params: &[u32]) {
        let kernel = self.kernels.get(op);
        *self.monitor.kernel.lock().unwrap() = Some(kernel.clone());
        let launch = Launch {
            program: &kernel.program,
            grid: kernel.grid,
            block: kernel.block,
            shared: kernel.shared,
            params,
        };
        if let Err(e) = self.gpu.borrow_mut().launch(&launch) {
            panic!("{} failed: {e}\n{}", kernel.name, kernel.disassemble());
        }
    }

    fn upload_words(&self, words: &[u32]) -> u32 {
        let mut gpu = self.gpu.borrow_mut();
        let addr = gpu.alloc(words.len() * 4);
        gpu.write(addr, words);
        addr
    }
}

impl<G: Gpu> Device for Titania<G> {
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
            addr: self.gpu.borrow_mut().alloc(len * 4),
            len,
            capacity: len,
        }
    }

    fn resize(&self, buf: &mut Buffer, len: usize) {
        assert!(len <= buf.capacity, "a buffer of {} activations can't hold {len}", buf.capacity);
        buf.len = len;
    }

    fn read(&self, buf: &Buffer) -> Vec<f32> {
        self.gpu.borrow().read(buf.addr, buf.len).into_iter().map(f32::from_bits).collect()
    }

    fn write(&self, buf: &mut Buffer, data: &[f32]) {
        assert_eq!(data.len(), buf.len);
        let words: Vec<u32> = data.iter().map(|x| x.to_bits()).collect();
        self.gpu.borrow_mut().write(buf.addr, &words);
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
    use titania_model::Cpu;
    use titania_simulator::Simulator;

    use super::*;

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
    fn buffers(gpu: &Titania<Simulator>, n: usize, seed: u32) -> (Vec<f32>, Buffer) {
        let values = numbers(n, seed);
        let mut buf = gpu.alloc(n);
        gpu.write(&mut buf, &values);
        (values, buf)
    }

    fn assert_close(gpu: &Titania<Simulator>, actual: &Buffer, expected: &[f32]) {
        let actual = gpu.read(actual);
        assert_eq!(actual.len(), expected.len());
        for (i, (a, e)) in actual.iter().zip(expected).enumerate() {
            let tolerance = 1e-5 + 1e-4 * e.abs();
            assert!((a - e).abs() <= tolerance, "element {i}: got {a}, expected {e}");
        }
    }

    #[test]
    fn elementwise() {
        let gpu = Titania::new(Simulator::new());
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
        let gpu = Titania::new(Simulator::new());
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
        let gpu = Titania::new(Simulator::new());
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
            let gpu = Titania::new(Simulator::new());
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
            let gpu = Titania::new(Simulator::new());
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
            let gpu = Titania::new(Simulator::new());
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
            let gpu = Titania::new(Simulator::new());
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
