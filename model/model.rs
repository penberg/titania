use std::path::Path;

use crate::{Config, Device, Result, Weights};

/// Most tokens a forward pass runs through the model at once. Running a
/// batch of tokens together reads each weight once for the whole batch,
/// rather than once per token; the activations are allocated for this many.
pub const BATCH: usize = 64;

/// A decoder-only transformer, with its weights on a device.
pub struct Model<D: Device> {
    pub config: Config,
    pub device: D,
    embed: D::Weight,
    layers: Vec<Layer<D>>,
    norm: D::Weight,
    /// Output projection, or `None` when it shares the embedding table.
    lm_head: Option<D::Weight>,
}

struct Layer<D: Device> {
    attn_norm: D::Weight,
    q: D::Weight,
    k: D::Weight,
    v: D::Weight,
    o: D::Weight,
    /// Per-head normalization of queries and keys (QK-norm), in models that
    /// have it.
    q_norm: Option<D::Weight>,
    k_norm: Option<D::Weight>,
    mlp_norm: D::Weight,
    gate: D::Weight,
    up: D::Weight,
    down: D::Weight,
}

impl<D: Device> Model<D> {
    /// Loads a model from a directory holding a Hugging Face `config.json` and
    /// `model.safetensors`, uploading its weights to `device`.
    pub fn load(dir: &Path, device: D) -> Result<Self> {
        let config = Config::load(&dir.join("config.json"))?;
        let mut weights = Weights::open(&dir.join("model.safetensors"))?;
        let w = &mut weights;
        let d = &device;

        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        for i in 0..config.num_hidden_layers {
            let prefix = format!("model.layers.{i}");
            layers.push(Layer {
                attn_norm: load(d, w, &format!("{prefix}.input_layernorm.weight"))?,
                q: load(d, w, &format!("{prefix}.self_attn.q_proj.weight"))?,
                k: load(d, w, &format!("{prefix}.self_attn.k_proj.weight"))?,
                v: load(d, w, &format!("{prefix}.self_attn.v_proj.weight"))?,
                o: load(d, w, &format!("{prefix}.self_attn.o_proj.weight"))?,
                q_norm: load_optional(d, w, &format!("{prefix}.self_attn.q_norm.weight"))?,
                k_norm: load_optional(d, w, &format!("{prefix}.self_attn.k_norm.weight"))?,
                mlp_norm: load(d, w, &format!("{prefix}.post_attention_layernorm.weight"))?,
                gate: load(d, w, &format!("{prefix}.mlp.gate_proj.weight"))?,
                up: load(d, w, &format!("{prefix}.mlp.up_proj.weight"))?,
                down: load(d, w, &format!("{prefix}.mlp.down_proj.weight"))?,
            });
        }
        let lm_head = if config.tie_word_embeddings {
            None
        } else {
            Some(load(d, w, "lm_head.weight")?)
        };

        Ok(Self {
            embed: load(d, w, "model.embed_tokens.weight")?,
            layers,
            norm: load(d, w, "model.norm.weight")?,
            lm_head,
            config,
            device,
        })
    }

    /// Runs `tokens`, the first at position `pos`, through the model, adding
    /// their keys and values to the cache in `state`, and returns the logits
    /// for the token that follows the last of them. Tokens run through the
    /// model in batches of up to [`BATCH`].
    pub fn forward(&self, state: &mut State<D>, tokens: &[u32], pos: usize) -> Vec<f32> {
        assert!(!tokens.is_empty(), "no tokens to run");
        assert!(pos + tokens.len() <= state.max_len, "tokens past the end of the cache");
        let mut logits = Vec::new();
        for (i, batch) in tokens.chunks(BATCH).enumerate() {
            logits = self.forward_batch(state, batch, pos + i * BATCH);
        }
        logits
    }

    fn forward_batch(&self, state: &mut State<D>, tokens: &[u32], pos: usize) -> Vec<f32> {
        let c = &self.config;
        let d = &self.device;
        let s = state;
        let n = tokens.len();
        let eps = c.rms_norm_eps;
        let head_dim = c.head_dim();
        let kv_dim = c.num_key_value_heads * head_dim;
        s.batch(d, n);

        d.embed(&mut s.x, &self.embed, tokens);
        for (i, layer) in self.layers.iter().enumerate() {
            // Attention, with the result added back into the residual stream.
            d.copy(&mut s.xb, 0, &s.x, 0, n * c.hidden_size);
            d.rmsnorm(&mut s.xb, &layer.attn_norm, eps);
            d.matmul(&mut s.q, &layer.q, &s.xb);
            d.matmul(&mut s.k, &layer.k, &s.xb);
            d.matmul(&mut s.v, &layer.v, &s.xb);
            if let Some(q_norm) = &layer.q_norm {
                d.rmsnorm(&mut s.q, q_norm, eps);
            }
            if let Some(k_norm) = &layer.k_norm {
                d.rmsnorm(&mut s.k, k_norm, eps);
            }
            d.rope(&mut s.q, pos, c.num_attention_heads, head_dim, c.rope_theta);
            d.rope(&mut s.k, pos, c.num_key_value_heads, head_dim, c.rope_theta);
            d.copy(&mut s.k_cache[i], pos * kv_dim, &s.k, 0, n * kv_dim);
            d.copy(&mut s.v_cache[i], pos * kv_dim, &s.v, 0, n * kv_dim);
            d.attention(
                &mut s.att,
                &s.q,
                &s.k_cache[i],
                &s.v_cache[i],
                pos,
                c.num_attention_heads,
                head_dim,
                c.num_key_value_heads,
            );
            d.matmul(&mut s.xb, &layer.o, &s.att);
            d.add(&mut s.x, &s.xb);

            // Feed-forward network, likewise added back.
            d.copy(&mut s.xb, 0, &s.x, 0, n * c.hidden_size);
            d.rmsnorm(&mut s.xb, &layer.mlp_norm, eps);
            d.matmul(&mut s.gate, &layer.gate, &s.xb);
            d.matmul(&mut s.up, &layer.up, &s.xb);
            d.silu_mul(&mut s.gate, &s.up);
            d.matmul(&mut s.xb, &layer.down, &s.gate);
            d.add(&mut s.x, &s.xb);
        }
        // Only the last token's logits are wanted.
        d.resize(&mut s.xb, c.hidden_size);
        d.copy(&mut s.xb, 0, &s.x, (n - 1) * c.hidden_size, c.hidden_size);
        d.rmsnorm(&mut s.xb, &self.norm, eps);
        d.matmul(&mut s.logits, self.lm_head.as_ref().unwrap_or(&self.embed), &s.xb);
        d.read(&s.logits)
    }
}

/// Buffers the forward pass computes in, with room for a batch of
/// [`BATCH`] tokens, and the key/value cache holding every position seen so
/// far.
pub struct State<D: Device> {
    /// Activations per token of each buffer, in the order of the fields.
    widths: [usize; 8],
    x: D::Buffer,
    xb: D::Buffer,
    q: D::Buffer,
    k: D::Buffer,
    v: D::Buffer,
    att: D::Buffer,
    gate: D::Buffer,
    up: D::Buffer,
    logits: D::Buffer,
    k_cache: Vec<D::Buffer>,
    v_cache: Vec<D::Buffer>,
    max_len: usize,
}

impl<D: Device> State<D> {
    /// Allocates state for sequences of up to `max_len` tokens.
    pub fn new(model: &Model<D>, max_len: usize) -> Self {
        let c = &model.config;
        let d = &model.device;
        let q_dim = c.num_attention_heads * c.head_dim();
        let kv_dim = c.num_key_value_heads * c.head_dim();
        let widths = [
            c.hidden_size,
            c.hidden_size,
            q_dim,
            kv_dim,
            kv_dim,
            q_dim,
            c.intermediate_size,
            c.intermediate_size,
        ];
        let [x, xb, q, k, v, att, gate, up] = widths.map(|width| d.alloc(BATCH * width));
        Self {
            widths,
            x,
            xb,
            q,
            k,
            v,
            att,
            gate,
            up,
            logits: d.alloc(c.vocab_size),
            k_cache: (0..c.num_hidden_layers).map(|_| d.alloc(max_len * kv_dim)).collect(),
            v_cache: (0..c.num_hidden_layers).map(|_| d.alloc(max_len * kv_dim)).collect(),
            max_len,
        }
    }

    pub fn max_len(&self) -> usize {
        self.max_len
    }

    /// Sizes the buffers for a batch of `n` tokens.
    fn batch(&mut self, device: &D, n: usize) {
        assert!(n <= BATCH, "a batch of {n} tokens is more than {BATCH}");
        let buffers = [
            &mut self.x,
            &mut self.xb,
            &mut self.q,
            &mut self.k,
            &mut self.v,
            &mut self.att,
            &mut self.gate,
            &mut self.up,
        ];
        for (buffer, width) in buffers.into_iter().zip(self.widths) {
            device.resize(buffer, n * width);
        }
    }
}

fn load<D: Device>(device: &D, weights: &mut Weights, name: &str) -> Result<D::Weight> {
    Ok(device.upload(weights.read(name)?))
}

fn load_optional<D: Device>(
    device: &D,
    weights: &mut Weights,
    name: &str,
) -> Result<Option<D::Weight>> {
    if weights.contains(name) {
        load(device, weights, name).map(Some)
    } else {
        Ok(None)
    }
}
