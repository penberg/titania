use crate::Tensor;

/// The operations a transformer's forward pass is built from.
///
/// A device owns the memory the model computes in: weights are uploaded to it
/// once, activations live in buffers allocated on it, and every operation runs
/// on it. [`Cpu`](crate::Cpu) is the reference implementation; other devices
/// run the same operations on Titania.
pub trait Device {
    /// A vector of f32 activations in device memory.
    type Buffer;
    /// A bf16 weight tensor in device memory.
    type Weight;

    /// Copies a weight tensor into device memory.
    fn upload(&self, tensor: Tensor) -> Self::Weight;

    /// Allocates a zeroed buffer of `len` activations.
    fn alloc(&self, len: usize) -> Self::Buffer;

    /// Copies a buffer out of device memory.
    fn read(&self, buf: &Self::Buffer) -> Vec<f32>;

    /// `dst[offset..offset + src.len()] = src`
    fn copy(&self, dst: &mut Self::Buffer, offset: usize, src: &Self::Buffer);

    /// `out = table[token]`: looks up a row of an embedding table.
    fn embed(&self, out: &mut Self::Buffer, table: &Self::Weight, token: usize);

    /// `out = w · x`, for a weight matrix `w` of shape `[out.len(), x.len()]`.
    fn matvec(&self, out: &mut Self::Buffer, w: &Self::Weight, x: &Self::Buffer);

    /// `x += y`
    fn add(&self, x: &mut Self::Buffer, y: &Self::Buffer);

    /// Normalizes each `weight.len()`-long row of `x` by its root mean square,
    /// then scales it by `weight`.
    fn rmsnorm(&self, x: &mut Self::Buffer, weight: &Self::Weight, eps: f32);

    /// Rotates each `head_dim`-long head of `x` to encode position `pos`
    /// (rotary position embeddings).
    fn rope(&self, x: &mut Self::Buffer, pos: usize, head_dim: usize, theta: f32);

    /// Causal self-attention: each `head_dim`-long head of `q` attends over
    /// the first `len` positions of the key and value caches, which hold
    /// `n_kv_heads` heads per position, and writes its result to `out`.
    #[allow(clippy::too_many_arguments)]
    fn attention(
        &self,
        out: &mut Self::Buffer,
        q: &Self::Buffer,
        k_cache: &Self::Buffer,
        v_cache: &Self::Buffer,
        len: usize,
        head_dim: usize,
        n_kv_heads: usize,
    );

    /// `gate = silu(gate) * up`: the SwiGLU activation.
    fn silu_mul(&self, gate: &mut Self::Buffer, up: &Self::Buffer);
}
