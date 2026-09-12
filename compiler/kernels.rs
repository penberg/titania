//! Kernels for each of a transformer's operations, specialized to their
//! shapes.
//!
//! Every kernel documents its parameters: the 32-bit words passed at launch,
//! in order. Buffers are passed by their global memory address; activations
//! are `f32`, and weights are bf16.

use crate::insn::Instruction;

use crate::{Builder, Cond};

/// A compiled kernel, with the launch geometry it was compiled for.
#[derive(Debug)]
pub struct Kernel {
    pub instructions: Vec<Instruction>,
    /// The instructions, encoded.
    pub program: Vec<u64>,
    /// Blocks along x and y.
    pub grid: [u32; 2],
    pub block: u32,
    /// Bytes of shared memory per block.
    pub shared: u32,
}

impl Kernel {
    /// The kernel's instructions in assembly syntax, one per line.
    pub fn disassemble(&self) -> String {
        self.instructions
            .iter()
            .enumerate()
            .map(|(pc, inst)| format!("{pc:4}  {inst}\n"))
            .collect()
    }
}

fn kernel(b: Builder, grid: [usize; 2], block: usize, shared: usize) -> Kernel {
    let instructions = b.finish().unwrap_or_else(|e| panic!("failed to compile kernel: {e}"));
    Kernel {
        program: instructions.iter().map(Instruction::encode).collect(),
        instructions,
        grid: grid.map(|n| n as u32),
        block: block as u32,
        shared: shared as u32,
    }
}

/// Threads per block for kernels with one thread per element.
const BLOCK: usize = 256;

/// Warps per block for kernels with one warp per row.
const WARPS: usize = 8;

/// Upper half of a word: the second of two bf16 numbers.
const HIGH_HALF: u32 = 0xffff_0000;

/// Starts a kernel with one thread per element for `n` elements, returning
/// the thread's element.
fn elementwise(b: &mut Builder, n: usize) -> crate::Value {
    let i = b.global_id();
    let past = b.isetp(Cond::Ge, i, n as u32);
    b.exit_if(past);
    i
}

/// Starts a kernel with one warp per row for `rows` rows, returning the lane
/// and the warp's row.
fn warp_per_row(b: &mut Builder, rows: usize) -> (crate::Value, crate::Value) {
    let tid = b.tid();
    let lane = b.and(tid, 31);
    let warp = b.shr(tid, 5);
    let block = b.ctaid_x();
    let warps = b.mov(WARPS as u32);
    let row = b.imad(block, warps, warp);
    let past = b.isetp(Cond::Ge, row, rows as u32);
    b.exit_if(past);
    (lane, row)
}

/// `dst[i] = src[i]` for `i < n`.
///
/// Parameters: `dst`, `src`.
pub fn copy(n: usize) -> Kernel {
    let mut b = Builder::new();
    let i = elementwise(&mut b, n);
    let (dst, src) = (b.param(0), b.param(1));
    let offset = b.shl(i, 2);
    let from = b.iadd(src, offset);
    let value = b.ldg(from, 0);
    let to = b.iadd(dst, offset);
    b.stg(to, 0, value);
    kernel(b, [n.div_ceil(BLOCK), 1], BLOCK, 0)
}

/// `x[i] += y[i]` for `i < n`.
///
/// Parameters: `x`, `y`.
pub fn add(n: usize) -> Kernel {
    let mut b = Builder::new();
    let i = elementwise(&mut b, n);
    let (x, y) = (b.param(0), b.param(1));
    let offset = b.shl(i, 2);
    let x_addr = b.iadd(x, offset);
    let y_addr = b.iadd(y, offset);
    let a = b.ldg(x_addr, 0);
    let c = b.ldg(y_addr, 0);
    let sum = b.fadd(a, c);
    b.stg(x_addr, 0, sum);
    kernel(b, [n.div_ceil(BLOCK), 1], BLOCK, 0)
}

/// `gate[i] = silu(gate[i]) * up[i]` for `i < n`, where
/// `silu(g) = g / (1 + e⁻ᵍ)`.
///
/// Parameters: `gate`, `up`.
pub fn silu_mul(n: usize) -> Kernel {
    let mut b = Builder::new();
    let i = elementwise(&mut b, n);
    let (gate, up) = (b.param(0), b.param(1));
    let offset = b.shl(i, 2);
    let gate_addr = b.iadd(gate, offset);
    let up_addr = b.iadd(up, offset);
    let g = b.ldg(gate_addr, 0);
    let u = b.ldg(up_addr, 0);
    let neg = b.xor(g, 0x8000_0000u32);
    let e = b.exp(neg);
    let denom = b.fadd(e, 1.0f32);
    let silu = b.fdiv(g, denom);
    let result = b.fmul(silu, u);
    b.stg(gate_addr, 0, result);
    kernel(b, [n.div_ceil(BLOCK), 1], BLOCK, 0)
}

/// `out = table[token]`, for a bf16 table with rows of `dim` elements. Each
/// thread converts one word: two elements.
///
/// Parameters: `out`, `table`, `token`.
pub fn embed(dim: usize) -> Kernel {
    assert!(dim.is_multiple_of(2), "embedding width must be even, got {dim}");
    let pairs = dim / 2;
    let mut b = Builder::new();
    let k = elementwise(&mut b, pairs);
    let (out, table, token) = (b.param(0), b.param(1), b.param(2));
    let row_words = b.mov(pairs as u32);
    let index = b.imad(token, row_words, k);
    let offset = b.shl(index, 2);
    let from = b.iadd(table, offset);
    let word = b.ldg(from, 0);
    let first = b.shl(word, 16);
    let second = b.and(word, HIGH_HALF);
    let out_offset = b.shl(k, 3);
    let to = b.iadd(out, out_offset);
    b.stg(to, 0, first);
    b.stg(to, 4, second);
    kernel(b, [pairs.div_ceil(BLOCK), 1], BLOCK, 0)
}

/// `out = w · x` for a bf16 matrix `w` of shape `[rows, cols]`.
///
/// Each warp computes one row: lane `l` multiplies words `l`, `l + 32`, ...
/// of the row, two elements each, and the warp sums the lanes' results.
///
/// Parameters: `out`, `w`, `x`.
pub fn matvec(rows: usize, cols: usize) -> Kernel {
    assert!(cols.is_multiple_of(64), "matvec needs a multiple of 64 columns, got {cols}");
    let steps = cols / 64;
    let unroll = [4, 2, 1].into_iter().find(|u| steps.is_multiple_of(*u)).unwrap();
    let mut b = Builder::new();
    let (lane, row) = warp_per_row(&mut b, rows);
    let (out, w, x) = (b.param(0), b.param(1), b.param(2));
    let row_bytes = b.mov((cols * 2) as u32);
    let w_row = b.imad(row, row_bytes, w);
    let lane_word = b.shl(lane, 2);
    let w_addr = b.iadd(w_row, lane_word);
    let lane_pair = b.shl(lane, 3);
    let x_addr = b.iadd(x, lane_pair);
    let even = b.mov(0.0f32);
    let odd = b.mov(0.0f32);
    b.for_range(0u32, steps as u32, unroll as u32, |b, _| {
        for u in 0..unroll as u32 {
            let word = b.ldg(w_addr, u * 128);
            let x0 = b.ldg(x_addr, u * 256);
            let x1 = b.ldg(x_addr, u * 256 + 4);
            let w0 = b.shl(word, 16);
            let w1 = b.and(word, HIGH_HALF);
            let sum = b.ffma(w0, x0, even);
            b.assign(even, sum);
            let sum = b.ffma(w1, x1, odd);
            b.assign(odd, sum);
        }
        let next = b.iadd(w_addr, 128 * unroll as u32);
        b.assign(w_addr, next);
        let next = b.iadd(x_addr, 256 * unroll as u32);
        b.assign(x_addr, next);
    });
    let partial = b.fadd(even, odd);
    let sum = b.warp_sum(partial);
    let row_offset = b.shl(row, 2);
    let to = b.iadd(out, row_offset);
    let first = b.isetp(Cond::Eq, lane, 0);
    b.when(first, |b| b.stg(to, 0, sum));
    kernel(b, [rows.div_ceil(WARPS), 1], WARPS * 32, 0)
}

/// Normalizes each of `rows` rows of `x`, each `dim` elements long, by its
/// root mean square, and scales it by the bf16 `weight`.
///
/// Each warp normalizes one row, with each lane taking pairs of elements.
///
/// Parameters: `x`, `weight`.
pub fn rmsnorm(rows: usize, dim: usize, eps: f32) -> Kernel {
    assert!(dim.is_multiple_of(64), "rmsnorm needs a multiple of 64 elements, got {dim}");
    let steps = dim / 64;
    let mut b = Builder::new();
    let (lane, row) = warp_per_row(&mut b, rows);
    let (x, weight) = (b.param(0), b.param(1));
    let row_bytes = b.mov((dim * 4) as u32);
    let x_row = b.imad(row, row_bytes, x);
    let lane_pair = b.shl(lane, 3);
    let x_addr = b.iadd(x_row, lane_pair);

    let mut squares = b.mov(0.0f32);
    for step in 0..steps as u32 {
        for half in [0, 4] {
            let v = b.ldg(x_addr, step * 256 + half);
            squares = b.ffma(v, v, squares);
        }
    }
    let sum = b.warp_sum(squares);
    let mean = b.fdiv(sum, dim as f32);
    let mean = b.fadd(mean, eps);
    let root = b.fsqrt(mean);
    let one = b.mov(1.0f32);
    let scale = b.fdiv(one, root);

    let lane_word = b.shl(lane, 2);
    let w_addr = b.iadd(weight, lane_word);
    for step in 0..steps as u32 {
        let word = b.ldg(w_addr, step * 128);
        let w0 = b.shl(word, 16);
        let w1 = b.and(word, HIGH_HALF);
        for (half, w) in [(0, w0), (4, w1)] {
            let v = b.ldg(x_addr, step * 256 + half);
            let factor = b.fmul(scale, w);
            let y = b.fmul(v, factor);
            b.stg(x_addr, step * 256 + half, y);
        }
    }
    kernel(b, [rows.div_ceil(WARPS), 1], WARPS * 32, 0)
}

/// Rotates each of `n_heads` heads of `x`, each `head_dim` elements long, to
/// encode position `pos`. Element `i` of a head pairs with element
/// `i + head_dim / 2`, rotated by the angle in a table of cosines and sines:
/// `table[pos][i] = (cos, sin)`.
///
/// Each block rotates one head, with one thread per pair.
///
/// Parameters: `x`, `table`, `pos`.
pub fn rope(n_heads: usize, head_dim: usize) -> Kernel {
    let half = head_dim / 2;
    let mut b = Builder::new();
    let (head, i) = (b.ctaid_x(), b.tid());
    let (x, table, pos) = (b.param(0), b.param(1), b.param(2));
    let head_bytes = b.mov((head_dim * 4) as u32);
    let x_head = b.imad(head, head_bytes, x);
    let offset = b.shl(i, 2);
    let x_addr = b.iadd(x_head, offset);
    let a = b.ldg(x_addr, 0);
    let c = b.ldg(x_addr, (half * 4) as u32);
    let row = b.mov(half as u32);
    let entry = b.imad(pos, row, i);
    let entry_offset = b.shl(entry, 3);
    let t_addr = b.iadd(table, entry_offset);
    let cos = b.ldg(t_addr, 0);
    let sin = b.ldg(t_addr, 4);
    let a_cos = b.fmul(a, cos);
    let c_sin = b.fmul(c, sin);
    let first = b.fsub(a_cos, c_sin);
    let c_cos = b.fmul(c, cos);
    let a_sin = b.fmul(a, sin);
    let second = b.fadd(c_cos, a_sin);
    b.stg(x_addr, 0, first);
    b.stg(x_addr, (half * 4) as u32, second);
    kernel(b, [n_heads, 1], half, 0)
}

/// Causal self-attention: each of `n_heads` query heads in `q` attends over
/// the first `len` positions of the key and value caches, which hold
/// `n_kv_heads` heads per position and room for `max_len` positions.
///
/// Each block handles one head, with one thread per element of the head:
///
/// 1. Each warp scores a share of the positions: the dot product of the
///    query with the position's key.
/// 2. The block takes the scores' maximum, and replaces each score `s` by
///    `e^(s - max)`, summing them.
/// 3. Each thread sums its element of the values, weighted by the scores,
///    and divides by the scores' sum.
///
/// Shared memory holds one partial result per warp, then the scores.
///
/// Parameters: `out`, `q`, `k_cache`, `v_cache`, `len`.
pub fn attention(n_heads: usize, head_dim: usize, n_kv_heads: usize, max_len: usize) -> Kernel {
    assert!(head_dim.is_multiple_of(32) && head_dim <= 1024, "unsupported head size {head_dim}");
    let group = n_heads / n_kv_heads;
    assert!(group.is_power_of_two(), "query heads per key/value head must be a power of two");
    const SCORES: u32 = 32 * 4;
    let shared = SCORES as usize + max_len * 4;
    assert!(shared <= 65536, "the key/value cache is too long for shared memory");
    let warps = (head_dim / 32) as u32;
    let head_bytes = (head_dim * 4) as u32;
    let stride = (n_kv_heads * head_dim * 4) as u32;
    let scale = 1.0 / (head_dim as f32).sqrt();

    let mut b = Builder::new();
    let (head, tid) = (b.ctaid_x(), b.tid());
    let lane = b.and(tid, 31);
    let warp = b.shr(tid, 5);
    let first = b.isetp(Cond::Eq, lane, 0);
    let tid_offset = b.shl(tid, 2);
    let warp_offset = b.shl(warp, 2);
    let (out, q, k, v, len) = (b.param(0), b.param(1), b.param(2), b.param(3), b.param(4));
    let kv_head = b.shr(head, group.trailing_zeros());
    let kv_offset = b.imul(kv_head, head_bytes);
    let head_offset = b.imul(head, head_bytes);

    // 1. Scores. Each lane holds every 32nd element of the query.
    let lane_offset = b.shl(lane, 2);
    let q_head = b.iadd(q, head_offset);
    let q_addr = b.iadd(q_head, lane_offset);
    let query: Vec<_> = (0..warps).map(|m| b.ldg(q_addr, m * 128)).collect();
    let k_head = b.iadd(k, kv_offset);
    let k_lane = b.iadd(k_head, lane_offset);
    b.for_range(warp, len, warps, |b, t| {
        let position = b.imul(t, stride);
        let k_addr = b.iadd(k_lane, position);
        let key = b.ldg(k_addr, 0);
        let mut dot = b.fmul(query[0], key);
        for (m, &qm) in query.iter().enumerate().skip(1) {
            let key = b.ldg(k_addr, m as u32 * 128);
            dot = b.ffma(qm, key, dot);
        }
        let dot = b.warp_sum(dot);
        let score = b.fmul(dot, scale);
        let s_addr = b.shl(t, 2);
        b.when(first, |b| b.sts(s_addr, SCORES, score));
    });
    b.barrier();

    // 2. Softmax, without the final division: first the maximum...
    let max = b.mov(f32::NEG_INFINITY);
    b.for_range(0u32, len, head_dim as u32, |b, j| {
        let t = b.iadd(j, tid);
        let inside = b.isetp(Cond::Lt, t, len);
        let s_addr = b.shl(t, 2);
        b.when(inside, |b| {
            let score = b.lds(s_addr, SCORES);
            let larger = b.fmax(max, score);
            b.assign(max, larger);
        });
    });
    let warp_max = b.warp_max(max);
    b.when(first, |b| b.sts(warp_offset, 0, warp_max));
    b.barrier();
    let mut max = b.lds(crate::Value::ZERO, 0);
    for w in 1..warps {
        let other = b.lds(crate::Value::ZERO, w * 4);
        max = b.fmax(max, other);
    }
    // Every warp reads the partial maximums before they are overwritten.
    b.barrier();

    // ...then the exponentials and their sum.
    let sum = b.mov(0.0f32);
    b.for_range(0u32, len, head_dim as u32, |b, j| {
        let t = b.iadd(j, tid);
        let inside = b.isetp(Cond::Lt, t, len);
        let s_addr = b.shl(t, 2);
        b.when(inside, |b| {
            let score = b.lds(s_addr, SCORES);
            let shifted = b.fsub(score, max);
            let weight = b.exp(shifted);
            b.sts(s_addr, SCORES, weight);
            let total = b.fadd(sum, weight);
            b.assign(sum, total);
        });
    });
    let warp_sum = b.warp_sum(sum);
    b.when(first, |b| b.sts(warp_offset, 0, warp_sum));
    b.barrier();
    let mut total = b.lds(crate::Value::ZERO, 0);
    for w in 1..warps {
        let other = b.lds(crate::Value::ZERO, w * 4);
        total = b.fadd(total, other);
    }

    // 3. The weighted sum of the values, one element per thread.
    let v_head = b.iadd(v, kv_offset);
    let v_addr = b.iadd(v_head, tid_offset);
    let s_addr = b.mov(SCORES);
    let acc = b.mov(0.0f32);
    b.for_range(0u32, len, 1, |b, _| {
        let weight = b.lds(s_addr, 0);
        let value = b.ldg(v_addr, 0);
        let sum = b.ffma(weight, value, acc);
        b.assign(acc, sum);
        let next = b.iadd(v_addr, stride);
        b.assign(v_addr, next);
        let next = b.iadd(s_addr, 4);
        b.assign(s_addr, next);
    });
    let result = b.fdiv(acc, total);
    let out_head = b.iadd(out, head_offset);
    let out_addr = b.iadd(out_head, tid_offset);
    b.stg(out_addr, 0, result);
    kernel(b, [n_heads, 1], head_dim, shared)
}
