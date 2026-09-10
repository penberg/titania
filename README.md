<p align="center">
  <img src=".github/assets/hero.png" width="320" alt="Project Titania logo">
</p>

<h1 align="center">Project Titania</h1>

<p align="center">
  <em>A complete LLM system, from transformer to transistor, simple enough for one person to understand.</em>
</p>

## Introduction

Titania is a from-scratch design for every layer of a language model stack:
the model, the instruction set, the compiler, the simulator, and the hardware.
Every piece is kept small and clear enough that a single person can read,
understand, and implement all of it.

<p align="center">
  <img src=".github/assets/architecture.svg" width="640" alt="Titania architecture: the model, a transformer, is compiled to the Titania ISA. The GPU, an RTL design, executes the ISA and is synthesized to an FPGA and then an ASIC. The ISA simulator defines the semantics of the ISA, and the GPU is verified against it.">
</p>

## Blueprint

Titania is made up of five layers:

* Model
* Compiler
* ISA
* ISA Simulator
* Hardware

### Model

The model is what the whole system exists to run. It is a decoder-only
transformer, the same kind of architecture behind today's large language
models: given a sequence of tokens, it predicts the next one, and generating
text means doing that over and over. The model is written as a set of GPU
kernels (matrix multiplications, normalization, attention, and activation
functions), and those kernels define the workload that every layer below must
support.

### Compiler

The compiler turns the model's kernels into programs for the Titania ISA.
Kernels are written in an ordinary high-level GPU programming language, not a
special dialect, and the compiler lowers them to Titania instructions: it
selects instructions, allocates registers, and lays out the program. Nobody
writes assembly by hand. The compiler targets the ISA rather than the hardware,
so it needs no knowledge of how the GPU is built.

### ISA

The instruction set architecture (ISA) is the contract between software and
hardware. It defines the GPU's programming model: the instructions, the
registers, the memory spaces, and how threads are grouped and executed
together. It is hardware-independent, in the spirit of NVIDIA's **PTX** and
Khronos' **SPIR-V**: the compiler targets it without knowing how the GPU is
built, and the GPU can evolve without breaking compiled programs.

### ISA Simulator

The ISA simulator is the reference implementation of the ISA: a program that
executes Titania instructions in software. It models *what* each instruction
does, not *how long* it takes, which keeps it simple enough to serve as the
definition of correct behavior. It is where the model first runs end to end,
and every result the hardware produces is checked against it.

### Hardware

The hardware is the GPU itself: a digital design, described at the
register-transfer level (RTL), that executes the Titania ISA. It fetches
instructions, schedules threads, moves data between memory and registers, and
does the arithmetic. The design first runs in an RTL simulator, where every
program must produce the same results as on the ISA simulator. It is then
synthesized onto an FPGA, and eventually manufactured as silicon.

## Getting Started

Install the `titania` command:

```console
cargo install --path cli
```

Then chat with the model, which is downloaded on first use:

```console
titania run
```

`titania models` lists the supported models, whether they are downloaded, and
where they are stored. `titania fetch` downloads a model ahead of time.

## Milestones

- [x] Model runs on CPU
- [ ] Compiler emits Titania ISA
- [ ] Model runs on ISA simulator
- [ ] Model runs on RTL simulator
- [ ] Model runs on FPGA
- [ ] Tapeout
