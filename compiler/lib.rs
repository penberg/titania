//! The Titania JIT compiler: generates Titania programs for a transformer's
//! operations, specialized to their shapes.
//!
//! Each operation in [`kernels`] is lowered, through a [`Builder`], to
//! instructions on an unlimited supply of virtual registers. `codegen` then
//! allocates physical registers, and [`insn`] encodes the result.

mod builder;
mod codegen;
pub mod insn;
pub mod kernels;

pub use builder::{Builder, Cond, Operand, Pred, Value};
pub use insn::{Insn, Instruction};
pub use kernels::Kernel;
