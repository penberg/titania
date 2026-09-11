use std::ops::ControlFlow;

use crate::{Device, Model, Result, Sampler, State, Tokenizer};

/// A conversation with an instruction-tuned model, in the ChatML format that
/// Qwen models are trained on:
///
/// ```text
/// <|im_start|>user
/// Hello!<|im_end|>
/// <|im_start|>assistant
/// Hi there!<|im_end|>
/// ```
///
/// The whole conversation stays in the key/value cache, so each turn only
/// runs the model over its new tokens.
pub struct Chat<D: Device> {
    model: Model<D>,
    state: State<D>,
    tokenizer: Tokenizer,
    sampler: Sampler,
    /// Number of tokens in the conversation so far.
    len: usize,
    im_start: u32,
    im_end: u32,
    end_of_text: u32,
    think: u32,
    think_end: u32,
}

impl<D: Device> Chat<D> {
    /// Starts a conversation with room for `max_len` tokens.
    pub fn new(model: Model<D>, tokenizer: Tokenizer, sampler: Sampler, max_len: usize) -> Result<Self> {
        Ok(Self {
            state: State::new(&model, max_len),
            model,
            im_start: tokenizer.special("<|im_start|>")?,
            im_end: tokenizer.special("<|im_end|>")?,
            end_of_text: tokenizer.special("<|endoftext|>")?,
            think: tokenizer.special("<think>")?,
            think_end: tokenizer.special("</think>")?,
            tokenizer,
            sampler,
            len: 0,
        })
    }

    /// Number of tokens in the conversation so far.
    pub fn tokens(&self) -> usize {
        self.len
    }

    /// Sends a message from the user, streaming the reply's text to `on_text`
    /// as it is generated. The reply ends early if `on_text` breaks.
    pub fn send(&mut self, message: &str, mut on_text: impl FnMut(&str) -> ControlFlow<()>) -> Result<()> {
        let t = &self.tokenizer;
        let mut prompt = vec![self.im_start];
        prompt.extend(t.encode(&format!("user\n{message}"))?);
        prompt.push(self.im_end);
        prompt.extend(t.encode("\n")?);
        prompt.push(self.im_start);
        prompt.extend(t.encode("assistant\n")?);
        // Qwen3 reasons inside <think> tags before replying; opening the reply
        // with an empty thought makes it answer directly.
        prompt.push(self.think);
        prompt.extend(t.encode("\n\n")?);
        prompt.push(self.think_end);
        prompt.extend(t.encode("\n\n")?);

        let mut logits = self.feed(&prompt)?;
        let mut text = Utf8Stream::default();
        loop {
            let token = self.sampler.sample(&logits);
            if token == self.im_end || token == self.end_of_text {
                break;
            }
            let chunk = text.push(self.tokenizer.decode(token));
            let flow = if chunk.is_empty() { ControlFlow::Continue(()) } else { on_text(&chunk) };
            // Feed the token even if the reply ends here, so that the model
            // remembers the reply exactly as far as it was shown.
            logits = self.feed(&[token])?;
            if flow.is_break() {
                break;
            }
        }

        // End the reply the way the chat format expects, ready for the next
        // message, even if it was cut short.
        let mut end = vec![self.im_end];
        end.extend(self.tokenizer.encode("\n")?);
        self.feed(&end)?;
        Ok(())
    }

    /// Runs tokens through the model, returning the logits after the last one.
    fn feed(&mut self, tokens: &[u32]) -> Result<Vec<f32>> {
        let mut logits = Vec::new();
        for &token in tokens {
            if self.len == self.state.max_len() {
                return Err("the conversation no longer fits in the context window".into());
            }
            logits = self.model.forward(&mut self.state, token, self.len);
            self.len += 1;
        }
        Ok(logits)
    }
}

/// Turns a stream of bytes into text, holding back a UTF-8 character split
/// across tokens until the rest of it arrives.
#[derive(Default)]
struct Utf8Stream {
    pending: Vec<u8>,
}

impl Utf8Stream {
    fn push(&mut self, bytes: &[u8]) -> String {
        self.pending.extend_from_slice(bytes);
        let complete = match std::str::from_utf8(&self.pending) {
            Ok(_) => self.pending.len(),
            // An incomplete character at the end: wait for the rest.
            Err(e) if e.error_len().is_none() => e.valid_up_to(),
            Err(_) => self.pending.len(),
        };
        let text = String::from_utf8_lossy(&self.pending[..complete]).into_owned();
        self.pending.drain(..complete);
        text
    }
}
