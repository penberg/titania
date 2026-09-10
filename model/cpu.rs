use rayon::prelude::*;

use crate::{Device, Tensor};

/// Runs the model on the CPU: the reference implementation of every
/// operation.
pub struct Cpu;

impl Device for Cpu {
    type Buffer = Vec<f32>;
    type Weight = Tensor;

    fn upload(&self, tensor: Tensor) -> Tensor {
        tensor
    }

    fn alloc(&self, len: usize) -> Vec<f32> {
        vec![0.0; len]
    }

    fn read(&self, buf: &Vec<f32>) -> Vec<f32> {
        buf.clone()
    }

    fn copy(&self, dst: &mut Vec<f32>, offset: usize, src: &Vec<f32>) {
        dst[offset..offset + src.len()].copy_from_slice(src);
    }

    fn embed(&self, out: &mut Vec<f32>, table: &Tensor, token: usize) {
        let row = &table.data[token * out.len()..][..out.len()];
        for (o, &w) in out.iter_mut().zip(row) {
            *o = bf16(w);
        }
    }

    fn matvec(&self, out: &mut Vec<f32>, w: &Tensor, x: &Vec<f32>) {
        assert_eq!(w.shape, [out.len(), x.len()]);
        out.par_iter_mut()
            .zip(w.data.par_chunks_exact(x.len()))
            .for_each(|(o, row)| *o = dot(row, x));
    }

    fn add(&self, x: &mut Vec<f32>, y: &Vec<f32>) {
        for (a, b) in x.iter_mut().zip(y) {
            *a += b;
        }
    }

    fn rmsnorm(&self, x: &mut Vec<f32>, weight: &Tensor, eps: f32) {
        let dim = weight.data.len();
        for row in x.chunks_exact_mut(dim) {
            let mean_square = row.iter().map(|v| v * v).sum::<f32>() / dim as f32;
            let scale = 1.0 / (mean_square + eps).sqrt();
            for (v, &w) in row.iter_mut().zip(&weight.data) {
                *v *= scale * bf16(w);
            }
        }
    }

    fn rope(&self, x: &mut Vec<f32>, pos: usize, head_dim: usize, theta: f32) {
        // Each head is rotated as pairs of elements half a head apart, each
        // pair at its own frequency.
        let half = head_dim / 2;
        for head in x.chunks_exact_mut(head_dim) {
            for i in 0..half {
                let freq = 1.0 / theta.powf((2 * i) as f32 / head_dim as f32);
                let (sin, cos) = (pos as f32 * freq).sin_cos();
                let (a, b) = (head[i], head[i + half]);
                head[i] = a * cos - b * sin;
                head[i + half] = b * cos + a * sin;
            }
        }
    }

    fn attention(
        &self,
        out: &mut Vec<f32>,
        q: &Vec<f32>,
        k_cache: &Vec<f32>,
        v_cache: &Vec<f32>,
        len: usize,
        head_dim: usize,
        n_kv_heads: usize,
    ) {
        // With grouped-query attention, consecutive query heads share a key
        // and value head.
        let group = q.len() / head_dim / n_kv_heads;
        let kv_dim = n_kv_heads * head_dim;
        let scale = 1.0 / (head_dim as f32).sqrt();
        out.par_chunks_exact_mut(head_dim)
            .zip(q.par_chunks_exact(head_dim))
            .enumerate()
            .for_each(|(head, (out, q))| {
                let kv = head / group * head_dim;
                let mut scores: Vec<f32> = (0..len)
                    .map(|t| scale * dot_f32(q, &k_cache[t * kv_dim + kv..][..head_dim]))
                    .collect();
                softmax(&mut scores);
                out.fill(0.0);
                for (t, &score) in scores.iter().enumerate() {
                    let v = &v_cache[t * kv_dim + kv..][..head_dim];
                    for (o, &v) in out.iter_mut().zip(v) {
                        *o += score * v;
                    }
                }
            });
    }

    fn silu_mul(&self, gate: &mut Vec<f32>, up: &Vec<f32>) {
        for (g, &u) in gate.iter_mut().zip(up) {
            *g = *g / (1.0 + (-*g).exp()) * u;
        }
    }
}

/// Converts bf16 bits to f32: bf16 is the top half of an f32.
fn bf16(bits: u16) -> f32 {
    f32::from_bits((bits as u32) << 16)
}

/// Dot product of a row of bf16 weights with f32 activations.
///
/// Accumulating into eight independent sums lets the compiler vectorize the
/// loop, which a single running sum would not allow.
fn dot(w: &[u16], x: &[f32]) -> f32 {
    let mut acc = [0.0f32; 8];
    for (w, x) in w.chunks_exact(8).zip(x.chunks_exact(8)) {
        for i in 0..8 {
            acc[i] += bf16(w[i]) * x[i];
        }
    }
    let done = w.len() / 8 * 8;
    let rest: f32 = w[done..].iter().zip(&x[done..]).map(|(&w, x)| bf16(w) * x).sum();
    acc.iter().sum::<f32>() + rest
}

fn dot_f32(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(a, b)| a * b).sum()
}

fn softmax(x: &mut [f32]) {
    let max = x.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0;
    for v in x.iter_mut() {
        *v = (*v - max).exp();
        sum += *v;
    }
    for v in x.iter_mut() {
        *v /= sum;
    }
}
