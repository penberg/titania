<div align="center">
  <img src=".github/assets/hero.png" width="280" alt="Project Titania logo">

# Project Titania

**A large language model, from transformer to transistor.**

The model, the instruction set, the compiler, the simulator, and the GPU:
every layer designed from scratch, and small enough for one person to read.

[![Rust](https://img.shields.io/badge/Rust-2024-orange?logo=rust)](https://www.rust-lang.org)
[![Model: Qwen3-0.6B](https://img.shields.io/badge/Model-Qwen3--0.6B-3ec98a)](https://huggingface.co/Qwen/Qwen3-0.6B)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE.md)

</div>

<p align="center">
  <img src=".github/assets/titania-run.gif" alt="titania run: chatting with Qwen3-0.6B on the CPU. Asked for a haiku about the moon, it answers: Silence of the night / A silver thread in the sky / And I — the moon.">
</p>

---

## What is Titania?

Titania is a from-scratch design for every layer of a language model stack.
Instead of building on someone else's GPU, instruction set, and compiler, it
defines all of them itself, and keeps each one small and clear enough that a
single person can read, understand, and implement all of it.

- **🌙 A real model.** Titania runs Qwen3-0.6B, a decoder-only transformer of
  the same kind as today's large language models, and you can chat with it.
- **🧩 Every layer, end to end.** The model's kernels are compiled to the
  Titania ISA and executed by the ISA simulator, today. An RTL GPU, an FPGA,
  and silicon come next.
- **📜 One source of truth.** The ISA simulator defines what every instruction
  does, and the hardware must produce bit-for-bit the same results.
- **📖 Small enough to read.** No layer is a black box: each is written to be
  understood, not just used.

<p align="center">
  <img src=".github/assets/architecture.svg" width="700" alt="Titania architecture: the model, a transformer, is compiled to the Titania ISA. The GPU, an RTL design, executes the ISA and is synthesized to an FPGA and then an ASIC. The ISA simulator defines the semantics of the ISA, and the GPU is verified against it.">
</p>

---

## Quick start

Clone the repository and install the `titania` command:

```console
git clone https://github.com/penberg/titania.git
cd titania
cargo install --path cli
```

Then chat with the model, which is downloaded (about 1.4 GB) on first use:

```console
titania run
```

Or run it on the Titania ISA simulator instead of the CPU:

```console
titania run --device sim
```

The model has a `bash` tool, so asking it about the files or the system you
are on makes it run a command and read the output. Commands run without
confirmation.

---

## Running on the Titania GPU

`titania run --device sim` runs the same model on a simulated Titania GPU. Each
of the model's operations is compiled into a Titania kernel the first time it
is used with a given shape, and the ISA simulator executes it instruction by
instruction. While the model thinks, a panel shows the kernel running, where
one of its warps is in the disassembly, and how fast instructions are
executing.

<p align="center">
  <img src=".github/assets/titania-run-sim.gif" alt="titania run --device sim: Qwen3-0.6B on the Titania ISA simulator. While it answers, a Titania GPU panel shows the running matvec kernel, its launch configuration, the instruction count, and the disassembly around a sampled warp's program counter.">
</p>

<p align="center"><em>Sped up: on the simulator, the model generates about a token per second.</em></p>

---

## Blueprint

Titania is made up of five layers:

| Layer | What it is | Where |
|-------|------------|-------|
| **Model** | A decoder-only transformer, written as GPU kernels | [`model/`](model) |
| **Compiler** | Lowers the model's kernels to Titania ISA programs | [`compiler/`](compiler) |
| **ISA** | The contract between software and hardware | [`docs/architecture-reference.md`](docs/architecture-reference.md) |
| **ISA Simulator** | The reference implementation of the ISA | [`simulator/`](simulator), [`runtime/`](runtime) |
| **Hardware** | The GPU itself, as an RTL design | Planned |

### 🧠 Model

The model is what the whole system exists to run. It is a decoder-only
transformer, the same kind of architecture behind today's large language
models: given a sequence of tokens, it predicts the next one, and generating
text means doing that over and over. The model is written as a set of GPU
kernels (matrix multiplications, normalization, attention, and activation
functions), and those kernels define the workload that every layer below must
support.

### ⚙️ Compiler

The compiler turns the model's kernels into programs for the Titania ISA.
Kernels are written in an ordinary high-level GPU programming language, not a
special dialect, and the compiler lowers them to Titania instructions: it
selects instructions, allocates registers, and lays out the program. Nobody
writes assembly by hand. The compiler targets the ISA rather than the hardware,
so it needs no knowledge of how the GPU is built.

### 📐 ISA

The instruction set architecture (ISA) is the contract between software and
hardware. It defines the GPU's programming model: the instructions, the
registers, the memory spaces, and how threads are grouped and executed
together. It is hardware-independent, in the spirit of NVIDIA's **PTX** and
Khronos' **SPIR-V**: the compiler targets it without knowing how the GPU is
built, and the GPU can evolve without breaking compiled programs. The
[Titania GPU Architecture Reference Manual](docs/architecture-reference.md)
defines it in full.

### 🧪 ISA Simulator

The ISA simulator is the reference implementation of the ISA: a program that
executes Titania instructions in software. It models *what* each instruction
does, not *how long* it takes, which keeps it simple enough to serve as the
definition of correct behavior. It is where the model first runs end to end,
and every result the hardware produces is checked against it.

### 🔌 Hardware

The hardware is the GPU itself: a digital design, described at the
register-transfer level (RTL), that executes the Titania ISA. It fetches
instructions, schedules threads, moves data between memory and registers, and
does the arithmetic. The design first runs in an RTL simulator, where every
program must produce the same results as on the ISA simulator. It is then
synthesized onto an FPGA, and eventually manufactured as silicon.

---

## Milestones

- [x] Model runs on CPU
- [x] Compiler emits Titania ISA
- [x] Model runs on ISA simulator
- [ ] Model runs on RTL simulator
- [ ] Model runs on FPGA
- [ ] Tapeout

---

## License

This project is licensed under the [MIT license].

### Contribution

Unless you explicitly state otherwise, any contribution intentionally submitted
for inclusion in Titania by you, shall be licensed as MIT, without any additional
terms or conditions.

[MIT license]: LICENSE.md
