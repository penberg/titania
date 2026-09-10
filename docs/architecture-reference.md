# Titania GPU Architecture Reference Manual

*Version 0 (draft)*

This manual defines the Titania GPU architecture: its execution model, machine
state, memory model, instruction set, and binary encoding. The JIT compiler
generates code for it, the ISA simulator implements it, and the hardware must
produce bit-for-bit the same results as the simulator for every program whose
behavior this manual defines.

Every instruction has exactly one correct result. There are no approximate
instructions and no implementation-defined rounding, so that the simulator can
serve as the reference that the hardware is verified against.

**Conventions.** Bit ranges are written `[high:low]`. All multi-byte values are
little-endian. Numbers prefixed with `0x` are hexadecimal.

## 1. Execution Model

A **kernel** is a program that runs on many threads at once. Launching a kernel
creates a **grid** of **blocks**, and each block consists of **threads**:

- The grid size (number of blocks) and block size (threads per block) are
  given at launch. Blocks are numbered `0` to `grid size - 1`, and threads
  within a block `0` to `block size - 1`.
- Every thread runs the same program, starting at instruction 0, and tells
  itself apart from other threads by reading its thread and block number from
  special registers (§2.3).
- Blocks run in an unspecified order, possibly concurrently. Threads in
  different blocks cannot communicate during a launch.

### 1.1 Warps

The threads of a block are grouped into **warps** of **32 threads**: warp `w`
holds threads `32w` to `32w + 31`. If the block size is not a multiple of 32,
the last warp has fewer threads; the missing threads do not exist.

The threads of a warp execute in lockstep: the warp has a single program
counter (PC), and each instruction executes for all of the warp's threads at
once. Every instruction reads all of its source operands before it writes its
destination.

The warps of a block are interleaved in an unspecified order, except where they
synchronize with `BAR` (§1.4).

### 1.2 Predication

Every instruction has a **guard**: a predicate register, optionally negated
(§2.2). An instruction takes effect only in threads whose guard is true; in
other threads it does nothing. The guard `pt` is always true.

### 1.3 Control Flow

A thread is **live** from the start of the kernel until it executes `EXIT`. A
warp finishes when none of its threads are live, and a block finishes when all
of its warps have.

- `EXIT` ends every thread whose guard is true. Threads may exit at different
  times: the warp continues with its remaining live threads.
- `BRA` jumps if its guard is true. The guard must have the same value in every
  live thread of the warp: a **divergent branch** is an error (§6).

Branches that go the same way for a whole warp are enough for loops with
uniform bounds; anything that differs between threads uses predication.

### 1.4 Barriers

`BAR` waits until every unfinished warp of the block is waiting at a `BAR`, and
then releases them all. Its guard must have the same value in every live thread
of the warp; a warp whose guard is false does not wait.

A barrier also orders memory: every shared or global memory write a thread of
the block made before the barrier is visible to every thread of the block after
it.

## 2. Machine State

### 2.1 General-Purpose Registers

Each thread has 64 general-purpose registers `r0` to `r63` of 32 bits each.
`r0` always reads as zero, and writes to it are discarded.

Registers are untyped: an instruction interprets its operands as integers or as
floating-point numbers (§4). Registers other than `r0` are undefined when a
thread starts.

### 2.2 Predicate Registers

Each thread has 7 one-bit predicate registers `p0` to `p6`, which are set by
comparisons (`ISETP`, `FSETP`) and used as guards. The predicate `pt` is always
true; writes to it are discarded. Predicate registers are undefined when a
thread starts.

### 2.3 Special Registers

Special registers are read with `S2R`:

| Number | Name      | Value                                  |
|--------|-----------|----------------------------------------|
| 0      | `%tid`    | Thread number within the block         |
| 1      | `%ntid`   | Block size                             |
| 2      | `%ctaid`  | Block number within the grid           |
| 3      | `%nctaid` | Grid size                              |

A thread's lane within its warp is `%tid & 31`.

## 3. Memory

There are three memory spaces, each byte-addressed with 32-bit addresses:

| Space      | Scope                     | Access         | Size                                   |
|------------|---------------------------|----------------|----------------------------------------|
| **Global** | Every thread and the host | Read and write | Device memory, up to 4 GiB             |
| **Shared** | The threads of one block  | Read and write | Given at launch, up to 64 KiB          |
| **Param**  | Every thread              | Read only      | Given at launch, up to 256 bytes       |

- All loads and stores are 32 bits wide, and their addresses must be multiples
  of 4. An access that is misaligned or outside the space is an error (§6).
- **Global memory** holds the kernel's inputs and outputs. The host reads and
  writes it between launches (§5).
- **Shared memory** is private to a block, and undefined when the block starts.
- **Param memory** holds the kernel's parameters, such as buffer addresses and
  sizes, given at launch.

**Ordering.** A thread always sees its own writes. A write becomes visible to
the other threads of its block at the next barrier (§1.4). Within a launch,
threads in different blocks must not access the same global memory location if
either of them writes it. Every write of a launch is visible to every later
launch.

Programs are stored separately from the memory spaces and cannot be read or
written by instructions.

## 4. Data Types

**Integers** are 32-bit two's complement. Arithmetic wraps around on overflow.
Comparisons are signed.

**Floating-point numbers** are IEEE 754 binary32 (`f32`):

- Every operation is correctly rounded, with round-to-nearest-ties-to-even.
  `FFMA` rounds once, after both the multiplication and the addition.
- Subnormal numbers are supported, both as operands and as results.
- Every operation that produces a NaN produces the canonical NaN `0x7FC00000`.
- There are no exceptions or status flags.

**bf16** numbers are the upper 16 bits of an `f32`. They are a storage format,
not an arithmetic type: a 32-bit word holds two of them, the first in the lower
half. Shifting a bf16 number left by 16 bits (`SHL`) or masking the upper half
(`AND` with `0xFFFF0000`) turns it into an `f32`.

## 5. Kernel Launch

The host interacts with the GPU in two ways:

- It reads and writes global memory.
- It launches kernels. A launch runs to completion, or until an error (§6),
  before the next one begins.

A launch is described by:

| Field              | Meaning                                                   |
|--------------------|-----------------------------------------------------------|
| Program            | The kernel's instructions. Execution starts at instruction 0. |
| Grid size          | Number of blocks, at least 1.                             |
| Block size         | Threads per block, from 1 to 1024.                        |
| Shared memory size | Bytes of shared memory per block: a multiple of 4, up to 65536. |
| Parameters         | Up to 256 bytes, a multiple of 4, read with `LDP`.        |

## 6. Errors

The following are **errors**. An error stops the launch and is reported to the
host; global memory is then undefined.

- An instruction with an unknown opcode or a nonzero unused field.
- A PC outside the program.
- A divergent `BRA` or `BAR`: its guard differs between live threads of a warp.
- A load or store that is misaligned or outside its memory space.

The following are **undefined behavior**, and are not detected: reading a
register before writing it, reading shared memory before writing it, and
conflicting accesses to the same memory location without a barrier between
them (§3).

## 7. Instruction Encoding

Every instruction is a 64-bit word:

| Bits      | Field    | Meaning                                                      |
|-----------|----------|--------------------------------------------------------------|
| `[7:0]`   | `opcode` | The operation (§8)                                           |
| `[10:8]`  | `guard`  | Guard predicate: `0`–`6` for `p0`–`p6`, `7` for `pt`         |
| `[11]`    | `neg`    | Negates the guard                                            |
| `[17:12]` | `rd`     | Destination register; for comparisons, the destination predicate |
| `[23:18]` | `ra`     | First source register                                        |
| `[29:24]` | `rb`     | Second source register                                       |
| `[30]`    | `i`      | The last source operand is the immediate, not a register     |
| `[31]`    | —        | Unused                                                       |
| `[63:32]` | `imm`    | 32-bit immediate if `i` is set; otherwise `rc` in `[37:32]`  |

Operands are written in the tables below as:

- `rd`, `ra`, `rb`, `rc`: the registers in those fields.
- `b`: register `rb`, or `imm` if `i` is set.
- `c`: register `rc`, or `imm` if `i` is set.
- `a`: register `ra`, or `imm` if `i` is set (for instructions with a single
  source).
- `pd`: predicate register `rd[2:0]`.

Immediates are 32-bit values: for floating-point instructions, they are the
bits of an `f32`.

Memory instructions always take their address offset from `imm`, and `BRA` and
`S2R` always take their target and register number from it; for these
instructions `i` must be set. Fields an instruction does not use must be zero.

## 8. Instruction Set

### 8.1 Integer

| Opcode | Mnemonic | Operands          | Operation                              |
|--------|----------|-------------------|----------------------------------------|
| `0x01` | `IADD`   | `rd, ra, b`       | `rd = ra + b`                          |
| `0x02` | `ISUB`   | `rd, ra, b`       | `rd = ra - b`                          |
| `0x03` | `IMUL`   | `rd, ra, b`       | `rd = ra × b`, lower 32 bits           |
| `0x04` | `IMAD`   | `rd, ra, rb, c`   | `rd = ra × rb + c`, lower 32 bits      |
| `0x05` | `AND`    | `rd, ra, b`       | `rd = ra & b`                          |
| `0x06` | `OR`     | `rd, ra, b`       | `rd = ra \| b`                         |
| `0x07` | `XOR`    | `rd, ra, b`       | `rd = ra ^ b`                          |
| `0x08` | `SHL`    | `rd, ra, b`       | `rd = ra << (b & 31)`                  |
| `0x09` | `SHR`    | `rd, ra, b`       | `rd = ra >> (b & 31)`, logical         |
| `0x0A` | `SRA`    | `rd, ra, b`       | `rd = ra >> (b & 31)`, arithmetic      |

### 8.2 Floating Point

| Opcode | Mnemonic | Operands          | Operation                              |
|--------|----------|-------------------|----------------------------------------|
| `0x10` | `FADD`   | `rd, ra, b`       | `rd = ra + b`                          |
| `0x11` | `FSUB`   | `rd, ra, b`       | `rd = ra - b`                          |
| `0x12` | `FMUL`   | `rd, ra, b`       | `rd = ra × b`                          |
| `0x13` | `FFMA`   | `rd, ra, rb, c`   | `rd = ra × rb + c`, rounded once       |
| `0x14` | `FMIN`   | `rd, ra, b`       | `rd = min(ra, b)`                      |
| `0x15` | `FMAX`   | `rd, ra, b`       | `rd = max(ra, b)`                      |
| `0x16` | `FDIV`   | `rd, ra, b`       | `rd = ra / b`                          |
| `0x17` | `FSQRT`  | `rd, a`           | `rd = √a`                              |
| `0x18` | `I2F`    | `rd, a`           | Converts the integer `a` to `f32`      |
| `0x19` | `F2I`    | `rd, a`           | Converts `a` to an integer: rounds to nearest, ties to even; out-of-range values saturate, and NaN converts to 0 |

For `FMIN` and `FMAX`, `-0.0` is less than `+0.0`, and if exactly one operand
is NaN, the result is the other operand.

### 8.3 Data Movement

| Opcode | Mnemonic | Operands          | Operation                              |
|--------|----------|-------------------|----------------------------------------|
| `0x20` | `MOV`    | `rd, a`           | `rd = a`                               |
| `0x21` | `S2R`    | `rd, imm`         | `rd` = special register `imm` (§2.3)   |

### 8.4 Comparison

| Opcode      | Mnemonic                               | Operands     | Operation                     |
|-------------|----------------------------------------|--------------|-------------------------------|
| `0x28–0x2D` | `ISETP.EQ`, `.NE`, `.LT`, `.LE`, `.GT`, `.GE` | `pd, ra, b` | `pd = ra op b`, signed integers |
| `0x30–0x35` | `FSETP.EQ`, `.NE`, `.LT`, `.LE`, `.GT`, `.GE` | `pd, ra, b` | `pd = ra op b`, `f32`     |

The opcodes are in the order listed: `0x28` is `ISETP.EQ` and `0x2D` is
`ISETP.GE`. `FSETP` comparisons involving NaN are false, except `.NE`, which is
true.

### 8.5 Memory

| Opcode | Mnemonic | Operands            | Operation                                  |
|--------|----------|---------------------|--------------------------------------------|
| `0x40` | `LDG`    | `rd, [ra + imm]`    | Loads a word from global memory            |
| `0x41` | `STG`    | `[ra + imm], rb`    | Stores `rb` to global memory               |
| `0x42` | `LDS`    | `rd, [ra + imm]`    | Loads a word from shared memory            |
| `0x43` | `STS`    | `[ra + imm], rb`    | Stores `rb` to shared memory               |
| `0x44` | `LDP`    | `rd, [ra + imm]`    | Loads a word from param memory             |

The address is `ra + imm`, wrapping around at 2³².

### 8.6 Warp

| Opcode | Mnemonic    | Operands     | Operation                                             |
|--------|-------------|--------------|-------------------------------------------------------|
| `0x48` | `SHFL.IDX`  | `rd, ra, b`  | `rd` = `ra` of lane `b & 31`                          |
| `0x49` | `SHFL.BFLY` | `rd, ra, b`  | `rd` = `ra` of lane `lane ^ (b & 31)`                 |

A shuffle reads `ra` from another thread of the same warp. If that thread does
not exist or has exited, the result is the thread's own `ra`. The source thread
does not need a true guard: its `ra` is read either way.

### 8.7 Control

| Opcode | Mnemonic | Operands   | Operation                                             |
|--------|----------|------------|-------------------------------------------------------|
| `0x50` | `BRA`    | `imm`      | Jumps to instruction `imm` (§1.3)                     |
| `0x51` | `BAR`    |            | Waits for the block's other warps (§1.4)              |
| `0x52` | `EXIT`   |            | Ends the thread (§1.3)                                |

## 9. Assembly Syntax

The assembler, disassembler, and compiler output share one textual syntax: one
instruction per line, with an optional guard, the mnemonic, and its operands.

```
        S2R   r1, %tid                 ; r1 = thread number
        AND   r2, r1, 31               ; r2 = lane
        ISETP.EQ p0, r2, 0             ; p0 = lane 0?
        LDP   r3, [r0 + 0]             ; r3 = first parameter
loop:   FFMA  r4, r5, r6, r4
        ISETP.LT p1, r7, r8
   @p1  BRA   loop
  @!p0  EXIT
        STG   [r3 + 0], r4
        EXIT
```

- A guard is written `@p` or `@!p` before the mnemonic; no guard means `pt`.
- Immediates are decimal or hexadecimal integers, or `f32` literals such as
  `1.0` for floating-point instructions.
- Labels name instruction numbers, and `;` starts a comment.

## Appendix: Idioms

**Unpacking bf16 weights.** A word loaded from a bf16 tensor holds two
numbers:

```
        LDG   r1, [r2 + 0]             ; two bf16 numbers
        SHL   r3, r1, 16               ; first, as f32
        AND   r4, r1, 0xFFFF0000       ; second, as f32
```

**Summing across a warp.** Five butterfly shuffles leave the sum of `r1` over
all 32 lanes in every lane:

```
        SHFL.BFLY r2, r1, 16
        FADD  r1, r1, r2
        SHFL.BFLY r2, r1, 8
        FADD  r1, r1, r2
        SHFL.BFLY r2, r1, 4
        FADD  r1, r1, r2
        SHFL.BFLY r2, r1, 2
        FADD  r1, r1, r2
        SHFL.BFLY r2, r1, 1
        FADD  r1, r1, r2
```

**Exponentials.** There is no exponential instruction. Compute `eˣ` as `2ᵗ`
with `t = x · log₂e`: split `t` into an integer `n` (`F2I`) and a fraction
`f`, evaluate `2ᶠ` with a polynomial (`FFMA`), and add `n` to the result's
exponent bits (`SHL`, `IADD`). Clamp `t` first (`FMAX`, `FMIN`) so that `n`
stays in range.
