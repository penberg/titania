use crate::{
    codegen,
    insn::{Insn, Instruction, SR_CTAID_X, SR_CTAID_Y, SR_NCTAID_X, SR_NCTAID_Y, SR_NTID, SR_TID},
};

/// A virtual register holding a 32-bit value.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Value(u32);

impl Value {
    /// Always zero: register `r0`.
    pub const ZERO: Value = Value(0);
}

/// A virtual predicate register.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Pred(u32);

/// A source operand: a register or a 32-bit immediate.
#[derive(Clone, Copy, Debug)]
pub enum Operand {
    Value(Value),
    Imm(u32),
}

impl From<Value> for Operand {
    fn from(value: Value) -> Self {
        Operand::Value(value)
    }
}

impl From<u32> for Operand {
    fn from(imm: u32) -> Self {
        Operand::Imm(imm)
    }
}

impl From<i32> for Operand {
    fn from(imm: i32) -> Self {
        Operand::Imm(imm as u32)
    }
}

impl From<f32> for Operand {
    fn from(imm: f32) -> Self {
        Operand::Imm(imm.to_bits())
    }
}

/// A comparison.
#[derive(Clone, Copy, Debug)]
pub enum Cond {
    Eq,
    Ne,
    Lt,
    Le,
    Gt,
    Ge,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Label(u32);

/// An instruction on virtual registers.
#[derive(Clone, Debug)]
pub struct VInst {
    pub insn: Insn,
    pub guard: Option<(Pred, bool)>,
    pub dst: Option<Value>,
    pub pdst: Option<Pred>,
    /// Register sources, in the order of the `ra`, `rb`, and `rc` fields.
    pub srcs: Vec<Value>,
    pub imm: Option<u32>,
    pub target: Option<Label>,
}

pub enum Item {
    Inst(VInst),
    Label(Label),
    /// Brackets a loop body, for liveness: a register live across the
    /// loop's boundary must stay allocated for the whole loop.
    LoopStart,
    LoopEnd,
}

/// Builds a kernel as instructions on virtual registers.
///
/// Values are never reused, so each is written once, except by
/// [`assign`](Self::assign), which updates a variable in place (such as an
/// accumulator in a loop).
pub struct Builder {
    items: Vec<Item>,
    next_value: u32,
    next_pred: u32,
    next_label: u32,
    guard: Option<(Pred, bool)>,
}

impl Default for Builder {
    fn default() -> Self {
        Self::new()
    }
}

impl Builder {
    pub fn new() -> Self {
        Self {
            items: Vec::new(),
            next_value: 1,
            next_pred: 0,
            next_label: 0,
            guard: None,
        }
    }

    /// Generates the kernel's instructions.
    pub fn finish(mut self) -> Result<Vec<Instruction>, String> {
        self.emit(Insn::Exit, None, None, vec![], None);
        codegen::generate(&self.items)
    }

    fn value(&mut self) -> Value {
        self.next_value += 1;
        Value(self.next_value - 1)
    }

    fn label(&mut self) -> Label {
        self.next_label += 1;
        Label(self.next_label - 1)
    }

    fn emit(&mut self, insn: Insn, dst: Option<Value>, pdst: Option<Pred>, srcs: Vec<Value>, imm: Option<u32>) {
        self.items.push(Item::Inst(VInst {
            insn,
            guard: self.guard,
            dst,
            pdst,
            srcs,
            imm,
            target: None,
        }));
    }

    /// Emits an instruction whose last source is `last`, which may be an
    /// immediate.
    fn op(&mut self, insn: Insn, mut srcs: Vec<Value>, last: Operand) -> Value {
        let dst = self.value();
        let imm = match last {
            Operand::Value(value) => {
                srcs.push(value);
                None
            }
            Operand::Imm(imm) => Some(imm),
        };
        self.emit(insn, Some(dst), None, srcs, imm);
        dst
    }

    /// Makes `var` hold `value` from here on.
    pub fn assign(&mut self, var: Value, value: Value) {
        assert_ne!(var, Value::ZERO, "cannot assign to r0");
        // Retarget the instruction that just produced `value`, if it did,
        // rather than copying it.
        if let Some(Item::Inst(inst)) = self.items.last_mut()
            && inst.dst == Some(value)
            && inst.guard == self.guard
        {
            inst.dst = Some(var);
            return;
        }
        self.emit(Insn::Mov, Some(var), None, vec![value], None);
    }

    // Special registers and parameters.

    fn special(&mut self, sr: u32) -> Value {
        let dst = self.value();
        self.emit(Insn::S2r, Some(dst), None, vec![], Some(sr));
        dst
    }

    pub fn tid(&mut self) -> Value {
        self.special(SR_TID)
    }

    pub fn ntid(&mut self) -> Value {
        self.special(SR_NTID)
    }

    /// The block's number along x.
    pub fn ctaid_x(&mut self) -> Value {
        self.special(SR_CTAID_X)
    }

    /// The block's number along y.
    pub fn ctaid_y(&mut self) -> Value {
        self.special(SR_CTAID_Y)
    }

    /// The grid's width.
    pub fn nctaid_x(&mut self) -> Value {
        self.special(SR_NCTAID_X)
    }

    /// The grid's height.
    pub fn nctaid_y(&mut self) -> Value {
        self.special(SR_NCTAID_Y)
    }

    /// Loads the `index`th 32-bit kernel parameter.
    pub fn param(&mut self, index: u32) -> Value {
        self.ld(Insn::Ldp, Value::ZERO, index * 4)
    }

    // Arithmetic.

    pub fn mov(&mut self, a: impl Into<Operand>) -> Value {
        self.op(Insn::Mov, vec![], a.into())
    }

    pub fn iadd(&mut self, a: Value, b: impl Into<Operand>) -> Value {
        self.op(Insn::Iadd, vec![a], b.into())
    }

    pub fn isub(&mut self, a: Value, b: impl Into<Operand>) -> Value {
        self.op(Insn::Isub, vec![a], b.into())
    }

    pub fn imul(&mut self, a: Value, b: impl Into<Operand>) -> Value {
        self.op(Insn::Imul, vec![a], b.into())
    }

    /// `a * b + c`
    pub fn imad(&mut self, a: Value, b: Value, c: impl Into<Operand>) -> Value {
        self.op(Insn::Imad, vec![a, b], c.into())
    }

    pub fn and(&mut self, a: Value, b: impl Into<Operand>) -> Value {
        self.op(Insn::And, vec![a], b.into())
    }

    pub fn xor(&mut self, a: Value, b: impl Into<Operand>) -> Value {
        self.op(Insn::Xor, vec![a], b.into())
    }

    pub fn shl(&mut self, a: Value, b: impl Into<Operand>) -> Value {
        self.op(Insn::Shl, vec![a], b.into())
    }

    pub fn shr(&mut self, a: Value, b: impl Into<Operand>) -> Value {
        self.op(Insn::Shr, vec![a], b.into())
    }

    pub fn fadd(&mut self, a: Value, b: impl Into<Operand>) -> Value {
        self.op(Insn::Fadd, vec![a], b.into())
    }

    pub fn fsub(&mut self, a: Value, b: impl Into<Operand>) -> Value {
        self.op(Insn::Fsub, vec![a], b.into())
    }

    pub fn fmul(&mut self, a: Value, b: impl Into<Operand>) -> Value {
        self.op(Insn::Fmul, vec![a], b.into())
    }

    /// `a * b + c`, rounded once.
    pub fn ffma(&mut self, a: Value, b: Value, c: impl Into<Operand>) -> Value {
        self.op(Insn::Ffma, vec![a, b], c.into())
    }

    pub fn fmin(&mut self, a: Value, b: impl Into<Operand>) -> Value {
        self.op(Insn::Fmin, vec![a], b.into())
    }

    pub fn fmax(&mut self, a: Value, b: impl Into<Operand>) -> Value {
        self.op(Insn::Fmax, vec![a], b.into())
    }

    pub fn fdiv(&mut self, a: Value, b: impl Into<Operand>) -> Value {
        self.op(Insn::Fdiv, vec![a], b.into())
    }

    pub fn fsqrt(&mut self, a: Value) -> Value {
        self.op(Insn::Fsqrt, vec![], a.into())
    }

    pub fn i2f(&mut self, a: Value) -> Value {
        self.op(Insn::I2f, vec![], a.into())
    }

    pub fn f2i(&mut self, a: Value) -> Value {
        self.op(Insn::F2i, vec![], a.into())
    }

    // Comparisons and control flow.

    /// Compares integers.
    pub fn isetp(&mut self, cond: Cond, a: Value, b: impl Into<Operand>) -> Pred {
        use Cond::*;
        let insn = match cond {
            Eq => Insn::IsetpEq,
            Ne => Insn::IsetpNe,
            Lt => Insn::IsetpLt,
            Le => Insn::IsetpLe,
            Gt => Insn::IsetpGt,
            Ge => Insn::IsetpGe,
        };
        self.setp(insn, a, b.into())
    }

    /// Compares floating-point numbers.
    pub fn fsetp(&mut self, cond: Cond, a: Value, b: impl Into<Operand>) -> Pred {
        use Cond::*;
        let insn = match cond {
            Eq => Insn::FsetpEq,
            Ne => Insn::FsetpNe,
            Lt => Insn::FsetpLt,
            Le => Insn::FsetpLe,
            Gt => Insn::FsetpGt,
            Ge => Insn::FsetpGe,
        };
        self.setp(insn, a, b.into())
    }

    fn setp(&mut self, insn: Insn, a: Value, b: Operand) -> Pred {
        let pred = Pred(self.next_pred);
        self.next_pred += 1;
        let (srcs, imm) = match b {
            Operand::Value(b) => (vec![a, b], None),
            Operand::Imm(imm) => (vec![a], Some(imm)),
        };
        self.emit(insn, None, Some(pred), srcs, imm);
        pred
    }

    /// Emits the instructions built by `body` so that they take effect only in
    /// threads where `pred` is true.
    pub fn when(&mut self, pred: Pred, body: impl FnOnce(&mut Self)) {
        self.guarded(pred, false, body);
    }

    /// Like [`when`](Self::when), for threads where `pred` is false.
    pub fn unless(&mut self, pred: Pred, body: impl FnOnce(&mut Self)) {
        self.guarded(pred, true, body);
    }

    fn guarded(&mut self, pred: Pred, neg: bool, body: impl FnOnce(&mut Self)) {
        assert!(self.guard.is_none(), "guards cannot be nested");
        self.guard = Some((pred, neg));
        body(self);
        self.guard = None;
    }

    /// Ends the threads where `pred` is true.
    pub fn exit_if(&mut self, pred: Pred) {
        self.when(pred, |b| b.emit(Insn::Exit, None, None, vec![], None));
    }

    pub fn barrier(&mut self) {
        self.emit(Insn::Bar, None, None, vec![], None);
    }

    /// Runs `body` for `i` from `start` while `i < end`, stepping by `step`.
    ///
    /// Branches must be uniform across a warp, so every thread of a warp
    /// must see the same `start` and `end`.
    pub fn for_range(
        &mut self,
        start: impl Into<Operand>,
        end: impl Into<Operand>,
        step: u32,
        body: impl FnOnce(&mut Self, Value),
    ) {
        assert!(self.guard.is_none(), "loops cannot be guarded");
        let i = self.mov(start);
        let end = end.into();
        let (top, done) = (self.label(), self.label());
        self.items.push(Item::LoopStart);
        self.items.push(Item::Label(top));
        let finished = self.isetp(Cond::Ge, i, end);
        self.when(finished, |b| b.branch(done));
        body(self, i);
        let next = self.iadd(i, step);
        self.assign(i, next);
        self.branch(top);
        self.items.push(Item::LoopEnd);
        self.items.push(Item::Label(done));
    }

    fn branch(&mut self, target: Label) {
        self.items.push(Item::Inst(VInst {
            insn: Insn::Bra,
            guard: self.guard,
            dst: None,
            pdst: None,
            srcs: vec![],
            imm: None,
            target: Some(target),
        }));
    }

    // Memory.

    fn ld(&mut self, insn: Insn, addr: Value, offset: u32) -> Value {
        let dst = self.value();
        self.emit(insn, Some(dst), None, vec![addr], Some(offset));
        dst
    }

    fn st(&mut self, insn: Insn, addr: Value, offset: u32, value: Value) {
        self.emit(insn, None, None, vec![addr, value], Some(offset));
    }

    /// Loads a word from global memory at `addr + offset`.
    pub fn ldg(&mut self, addr: Value, offset: u32) -> Value {
        self.ld(Insn::Ldg, addr, offset)
    }

    /// Stores a word to global memory at `addr + offset`.
    pub fn stg(&mut self, addr: Value, offset: u32, value: Value) {
        self.st(Insn::Stg, addr, offset, value);
    }

    /// Loads a word from shared memory at `addr + offset`.
    pub fn lds(&mut self, addr: Value, offset: u32) -> Value {
        self.ld(Insn::Lds, addr, offset)
    }

    /// Stores a word to shared memory at `addr + offset`.
    pub fn sts(&mut self, addr: Value, offset: u32, value: Value) {
        self.st(Insn::Sts, addr, offset, value);
    }

    /// Reads `a` from the lane whose number is this lane's XORed with `mask`.
    pub fn shfl_bfly(&mut self, a: Value, mask: u32) -> Value {
        self.op(Insn::ShflBfly, vec![a], mask.into())
    }

    // Idioms built from the instructions above.

    /// This thread's index in the whole grid.
    pub fn global_id(&mut self) -> Value {
        let (block, size, tid) = (self.ctaid_x(), self.ntid(), self.tid());
        self.imad(block, size, tid)
    }

    /// Sums `a` across the warp, leaving the result in every lane.
    pub fn warp_sum(&mut self, a: Value) -> Value {
        self.warp_reduce(a, Self::fadd)
    }

    /// Takes the maximum of `a` across the warp, leaving it in every lane.
    pub fn warp_max(&mut self, a: Value) -> Value {
        self.warp_reduce(a, Self::fmax)
    }

    fn warp_reduce(&mut self, mut a: Value, op: fn(&mut Self, Value, Operand) -> Value) -> Value {
        for mask in [16, 8, 4, 2, 1] {
            let other = self.shfl_bfly(a, mask);
            a = op(self, a, other.into());
        }
        a
    }

    /// `eˣ`, computed as `2ⁿ · eʳ` with `n` an integer and `r` small enough
    /// for a polynomial (Cephes' `expf`).
    pub fn exp(&mut self, x: Value) -> Value {
        // Keep the result, and therefore n, within the range of f32.
        let x = self.fmax(x, -87.0f32);
        let x = self.fmin(x, 88.0f32);
        let t = self.fmul(x, std::f32::consts::LOG2_E);
        let n = self.f2i(t);
        let nf = self.i2f(n);
        // r = x - n·ln 2, with ln 2 split into two parts for precision.
        let ln2_hi = self.mov(-0.693_359_4_f32);
        let r = self.ffma(nf, ln2_hi, x);
        let ln2_lo = self.mov(2.121_944_4e-4_f32);
        let r = self.ffma(nf, ln2_lo, r);
        let mut p = self.mov(1.987_569_1e-4_f32);
        for c in [1.398_199_9e-3_f32, 8.333_452e-3, 4.166_579_6e-2, 0.166_666_65, 0.5] {
            p = self.ffma(p, r, c);
        }
        let r2 = self.fmul(r, r);
        let y = self.ffma(p, r2, r);
        let y = self.fadd(y, 1.0f32);
        // 2ⁿ, built directly from its exponent bits.
        let e = self.iadd(n, 127);
        let scale = self.shl(e, 23);
        self.fmul(y, scale)
    }
}
