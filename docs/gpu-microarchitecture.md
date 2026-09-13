# Titania GPU Microarchitecture

*Version 0 (draft)*

The Titania GPU is the register-transfer level (RTL) design in `rtl/`, written
in SystemVerilog, that executes the Titania ISA. This document describes how
it is built. What it computes is defined by the [architecture reference
manual](architecture-reference.md), and the design must produce bit-for-bit
the same results as the ISA simulator for every program whose behavior that
manual defines.

The design is deliberately small. It has no caches, no instruction-level
parallelism within a warp, and no timing optimization yet: the goal of this
version is a correct, readable GPU that runs the model in an RTL simulator.
Later versions will pipeline the arithmetic units and add the memory
hierarchy an FPGA or ASIC needs.

## 1. Overview

```
                 host interface                memory ports
          (program, params, launch, errors)   (one per SM)
                       │                            ▲
                       ▼                            │
 ┌─────────────────────────────────────────────────────────────┐
 │ titania                                                     │
 │   ┌────────────┐   blocks   ┌────┐ ┌────┐ ┌────┐ ┌────┐     │
 │   │ dispatcher │───────────▶│ SM │ │ SM │ │ SM │ │ SM │ ... │
 │   └────────────┘            └────┘ └────┘ └────┘ └────┘     │
 └─────────────────────────────────────────────────────────────┘
```

| Module    | File             | Role                                                          |
|-----------|------------------|---------------------------------------------------------------|
| `titania` | `rtl/titania.sv` | The chip: host interface, block dispatcher, `NUM_SMS` SMs      |
| `sm`      | `rtl/sm.sv`      | A streaming multiprocessor: runs one block at a time           |
| `isa`     | `rtl/isa.sv`     | Opcodes, operand formats, and the instruction decoder (§7, §8) |
| `fpu`     | `rtl/fpu.sv`     | IEEE 754 binary32 arithmetic (§4)                              |

Global memory is not part of the chip. Each SM has a memory port, and
whatever the chip is mounted on serves them: in the RTL simulator, a C++
harness with a model of DRAM (`rtlsim/harness.cpp`); on an FPGA, a memory
controller.

## 2. Host Interface

The host drives the `titania` module directly:

1. It resets the chip.
2. It writes the program into instruction memory (`prog_we`, `prog_addr`,
   `prog_data`), one 64-bit instruction per cycle, and the parameters into
   parameter memory (`param_we`, `param_addr`, `param_data`), one word per
   cycle. Every SM keeps its own copy of both.
3. It presents the launch geometry (`grid_width`, `grid_height`,
   `block_size`, `shared_bytes`, `prog_len`, `param_bytes`) and pulses
   `launch`.
4. It clocks the chip until `busy` falls. If `err_valid` rises instead, the
   launch has failed: `err_code`, `err_pc`, and `err_addr` say why and where
   (§6 of the manual). The chip stays in this state until it is reset.

`retired` counts the instructions completed since the launch, once per warp
that executes each, and `sample_block`, `sample_warp`, and `sample_pc` say
where the most recently completed instruction was, for a monitor.

## 3. Block Dispatch

The dispatcher hands out the blocks of the grid in row-major order, `(0, 0)`,
`(1, 0)`, and so on along x and then y, one per cycle, to the lowest-numbered
idle SM, with each block's coordinates and its number in that order, which the
monitor reports. A launch is done when every block has been handed out and
every SM is idle. Blocks therefore run in an unspecified order and
concurrently, as §1 of the manual allows.

## 4. The Streaming Multiprocessor

An SM executes one block: up to 32 warps of 32 lanes. It holds:

| State                      | Size                  | Notes                                          |
|----------------------------|-----------------------|------------------------------------------------|
| Instruction memory         | 4096 × 64 bits        | Written by the host                            |
| Parameter memory           | 64 × 32 bits          | Written by the host                            |
| Register file              | 32 warps × 64 × 1024 bits | One 1024-bit row per register: 32 lanes of 32 bits |
| Predicate registers        | 32 warps × 8 × 32 bits | One bit per lane; entry 7 is `pt`             |
| Shared memory              | 64 KiB, in 32 banks   | Word `i` is in bank `i mod 32`                 |
| Block coordinates          | 2 × 32 bits           | `%ctaid.x` and `%ctaid.y` of the block running |
| Per-warp PC and live mask  | 32 × (12 + 32) bits   |                                                |
| Warp status                | 3 × 32 bits           | `active`, `inflight`, `waiting`                |

`r0` is never written and reads as zero. Registers, predicates, and shared
memory are not cleared between blocks: the manual leaves them undefined.

### 4.1 Pipeline

Instructions flow through three stages, one cycle each:

| Stage   | What happens                                                                                             |
|---------|----------------------------------------------------------------------------------------------------------|
| Issue   | Pick a ready warp, round robin from the one issued last; check its PC; fetch from instruction memory     |
| Decode  | Decode the instruction word; evaluate the guard against the predicate and live masks; read `ra`, `rb`, `rc` |
| Execute | Compute the result for every lane; write back the register or predicate, the PC, and the live mask      |

**One instruction in flight per warp.** A warp is *ready* only when it is
active, not waiting at a barrier, and has no instruction in flight; issuing
marks it in flight, and writeback clears the mark. So no two instructions of
one warp are ever in the pipeline together, and there are no data hazards to
detect or forward around. The cost is that a warp can issue at most every
three cycles: three ready warps keep the pipeline full, and the kernels the
compiler generates run 2 to 8 warps per block.

**Guards and masks.** The decode stage forms the lane mask of an instruction
as its guard predicate (negated if `neg`), ANDed with the warp's live mask.
Execute writes results only for lanes in the mask. `BRA` and `BAR` compare the
mask against the live mask: equal means the whole warp takes the branch or
waits; zero means it does not; anything else is a divergence error.

**Stalls.** Execute stalls the whole pipeline (issue and decode hold) while a
shared memory access still has lanes to serve, or while the load/store unit
cannot accept a global memory access.

### 4.2 Arithmetic

Every arithmetic instruction, including `FFMA`, `FDIV`, and `FSQRT`, executes
combinationally in a single cycle. This makes for a very long critical path
and is the first thing a timing-driven revision would change, but it keeps
the pipeline simple and the results obviously right.

The floating-point unit is the `fpu` package. Its operations are built on one
rounding routine, `pack`, which takes an exact result as a wide integer, a
power-of-two scale, and a sticky bit, and rounds it once to nearest-even, with
subnormals and overflow to infinity handled in the same place:

- `fma32` computes the 48-bit exact product, aligns it and the addend in a
  64-bit frame with the larger at the top (bits of the smaller that fall
  below the frame can only affect the sticky bit), adds or subtracts, and
  packs. `FADD`, `FSUB`, and `FMUL` are `fma32` with an operand of `1.0` or
  `-0.0`, which gives the same correctly rounded results.
- `fdiv32` divides the normalized significands to 26 fractional bits, with the
  remainder as the sticky bit, and packs.
- `fsqrt32` takes a restoring integer square root of the significand, scaled
  to an even exponent, with the remainder as the sticky bit, and packs.
- `i2f` packs the integer's magnitude; `f2i` rounds to nearest-even and
  saturates.

`fpu.sv` is checked against the host's IEEE arithmetic (`fmaf`, division,
`sqrtf`, conversions) on tens of millions of random and special operands, and
the whole design against the ISA simulator by the tests in `rtlsim/lib.rs`.

### 4.3 Memory

**Global memory** goes through the load/store unit. At the execute stage, an
`LDG` or `STG` sends one request to the SM's memory port with the address of
every lane in its mask (and the data, for stores) and enters a FIFO of
accesses in flight, up to 16; the warp stays in flight. Responses return in
order. A load's response writes its data to the register file through a
second write port; either response clears the warp's in-flight mark. Because
a store is not complete until its response returns, and a warp cannot pass a
barrier with an instruction in flight, every write before a barrier is
visible after it.

The memory port is a gather/scatter interface: `mem_req_valid`/`ready`,
`mem_req_we`, a 32-bit lane mask, and 32 addresses and 32 data words;
responses carry 32 data words and an error flag for accesses that were
misaligned or out of range. The RTL simulator's harness answers every request
after a fixed latency of 8 cycles.

**Shared memory** has 32 banks. An `LDS` or `STS` serves, each cycle, the
lowest pending lane of every bank, and finishes the cycle after its last lane
is served: two cycles for a conflict-free access, plus one per extra lane on
the busiest bank.

**Parameter memory** is small enough to read all 32 lanes in one cycle.

### 4.4 Barriers and Completion

A warp whose `BAR` has a true, uniform guard enters the waiting state. When
every active warp of the block is waiting, all are released at once. A warp
that executes `EXIT` in its last live lane leaves the active set, which can
itself release a barrier the other warps are waiting at. The block is done
when no warp is active, and the SM tells the dispatcher it is idle.

## 5. Errors

Errors are detected where they occur: a bad PC at issue, an invalid
instruction at decode, divergence and bad shared or parameter addresses at
execute, and bad global addresses either at execute (misalignment) or when the
memory port reports one. The SM latches the first, with its PC and address,
and the chip reports it to the host. Nothing is halted: the host stops the
clock and resets the chip before the next launch.

## 6. The RTL Simulator

`titania run --device rtlsim` runs the model on this design. The
`titania-rtlsim` crate compiles the RTL with [Verilator] into a C++ model at
build time; `rtlsim/harness.cpp` drives the host interface and models global
memory; and the runtime uses it through the same `Gpu` interface as the ISA
simulator. Global memory is a Rust `Vec<u32>` that the harness reads and
writes directly.

Simulating a GPU cycle by cycle is slow. With the default four SMs, the
simulator runs at about 300 kHz on one core, retiring about 3.6 instructions
per cycle; a Qwen3-0.6B token takes about 28 million cycles, or a minute and
a half. Loading the chat's system prompt therefore takes hours, and the model
is bit-for-bit the same as on the ISA simulator, which is checked token by
token. The build script accepts `TITANIA_SMS` (SMs to instantiate) and
`TITANIA_THREADS` (threads Verilator simulates them with); more SMs finish a
launch in fewer cycles but each cycle costs proportionally more to simulate,
and threads have not paid for their synchronization so far.

[Verilator]: https://verilator.org
