use std::path::Path;

use crate::{Config, Device, Result, Weights};

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

    /// Runs `token` at position `pos` through the model, adding its keys and
    /// values to the cache in `state`, and returns the logits for the token
    /// that follows it.
    pub fn forward(&self, state: &mut State<D>, token: u32, pos: usize) -> Vec<f32> {
        assert!(pos < state.max_len, "position {pos} is past the end of the cache");
        let c = &self.config;
        let d = &self.device;
        let s = state;
        let eps = c.rms_norm_eps;
        let head_dim = c.head_dim();
        let kv_dim = c.num_key_value_heads * head_dim;

        d.embed(&mut s.x, &self.embed, token as usize);
        for (i, layer) in self.layers.iter().enumerate() {
            // Attention, with the result added back into the residual stream.
            d.copy(&mut s.xb, 0, &s.x);
            d.rmsnorm(&mut s.xb, &layer.attn_norm, eps);
            d.matvec(&mut s.q, &layer.q, &s.xb);
            d.matvec(&mut s.k, &layer.k, &s.xb);
            d.matvec(&mut s.v, &layer.v, &s.xb);
            if let Some(q_norm) = &layer.q_norm {
                d.rmsnorm(&mut s.q, q_norm, eps);
            }
            if let Some(k_norm) = &layer.k_norm {
                d.rmsnorm(&mut s.k, k_norm, eps);
            }
            d.rope(&mut s.q, pos, head_dim, c.rope_theta);
            d.rope(&mut s.k, pos, head_dim, c.rope_theta);
            d.copy(&mut s.k_cache[i], pos * kv_dim, &s.k);
            d.copy(&mut s.v_cache[i], pos * kv_dim, &s.v);
            d.attention(
                &mut s.att,
                &s.q,
                &s.k_cache[i],
                &s.v_cache[i],
                pos + 1,
                head_dim,
                c.num_key_value_heads,
            );
            d.matvec(&mut s.xb, &layer.o, &s.att);
            d.add(&mut s.x, &s.xb);

            // Feed-forward network, likewise added back.
            d.copy(&mut s.xb, 0, &s.x);
            d.rmsnorm(&mut s.xb, &layer.mlp_norm, eps);
            d.matvec(&mut s.gate, &layer.gate, &s.xb);
            d.matvec(&mut s.up, &layer.up, &s.xb);
            d.silu_mul(&mut s.gate, &s.up);
            d.matvec(&mut s.xb, &layer.down, &s.gate);
            d.add(&mut s.x, &s.xb);
        }
        d.rmsnorm(&mut s.x, &self.norm, eps);
        d.matvec(&mut s.logits, self.lm_head.as_ref().unwrap_or(&self.embed), &s.x);
        d.read(&s.logits)
    }
}

/// Buffers the forward pass computes in, and the key/value cache holding
/// every position seen so far.
pub struct State<D: Device> {
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
        Self {
            x: d.alloc(c.hidden_size),
            xb: d.alloc(c.hidden_size),
            q: d.alloc(q_dim),
            k: d.alloc(kv_dim),
            v: d.alloc(kv_dim),
            att: d.alloc(q_dim),
            gate: d.alloc(c.intermediate_size),
            up: d.alloc(c.intermediate_size),
            logits: d.alloc(c.vocab_size),
            k_cache: (0..c.num_hidden_layers).map(|_| d.alloc(max_len * kv_dim)).collect(),
            v_cache: (0..c.num_hidden_layers).map(|_| d.alloc(max_len * kv_dim)).collect(),
            max_len,
        }
    }

    pub fn max_len(&self) -> usize {
        self.max_len
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
