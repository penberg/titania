//! Titania instructions and their binary encoding, as defined in the Titania
//! GPU Architecture Reference Manual (`docs/architecture-reference.md`).

use std::fmt;

/// Number of general-purpose registers.
pub const NUM_REGS: u8 = 64;

/// Number of predicate registers, not counting `pt`.
pub const NUM_PREDS: u8 = 7;

/// The guard field value for `pt`, the always-true predicate.
pub const PT: u8 = 7;

/// Special registers, read with `S2R`.
pub const SR_TID: u32 = 0;
pub const SR_NTID: u32 = 1;
pub const SR_CTAID: u32 = 2;
pub const SR_NCTAID: u32 = 3;

/// An operation, with its opcode (§8 of the manual).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Insn {
    Iadd = 0x01,
    Isub = 0x02,
    Imul = 0x03,
    Imad = 0x04,
    And = 0x05,
    Or = 0x06,
    Xor = 0x07,
    Shl = 0x08,
    Shr = 0x09,
    Sra = 0x0A,
    Fadd = 0x10,
    Fsub = 0x11,
    Fmul = 0x12,
    Ffma = 0x13,
    Fmin = 0x14,
    Fmax = 0x15,
    Fdiv = 0x16,
    Fsqrt = 0x17,
    I2f = 0x18,
    F2i = 0x19,
    Mov = 0x20,
    S2r = 0x21,
    IsetpEq = 0x28,
    IsetpNe = 0x29,
    IsetpLt = 0x2A,
    IsetpLe = 0x2B,
    IsetpGt = 0x2C,
    IsetpGe = 0x2D,
    FsetpEq = 0x30,
    FsetpNe = 0x31,
    FsetpLt = 0x32,
    FsetpLe = 0x33,
    FsetpGt = 0x34,
    FsetpGe = 0x35,
    Ldg = 0x40,
    Stg = 0x41,
    Lds = 0x42,
    Sts = 0x43,
    Ldp = 0x44,
    ShflIdx = 0x48,
    ShflBfly = 0x49,
    Bra = 0x50,
    Bar = 0x51,
    Exit = 0x52,
}

/// The operands an instruction takes (§7 and §8 of the manual).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Format {
    /// `rd, ra, b`
    R2,
    /// `rd, ra, rb, c`
    R3,
    /// `rd, a`
    R1,
    /// `pd, ra, b`
    Setp,
    /// `rd, [ra + imm]`
    Load,
    /// `[ra + imm], rb`
    Store,
    /// `rd, imm`
    Special,
    /// `imm`
    Branch,
    /// No operands.
    None,
}

impl Insn {
    pub fn format(self) -> Format {
        use Insn::*;
        match self {
            Iadd | Isub | Imul | And | Or | Xor | Shl | Shr | Sra => Format::R2,
            Fadd | Fsub | Fmul | Fmin | Fmax | Fdiv | ShflIdx | ShflBfly => Format::R2,
            Imad | Ffma => Format::R3,
            Fsqrt | I2f | F2i | Mov => Format::R1,
            IsetpEq | IsetpNe | IsetpLt | IsetpLe | IsetpGt | IsetpGe => Format::Setp,
            FsetpEq | FsetpNe | FsetpLt | FsetpLe | FsetpGt | FsetpGe => Format::Setp,
            Ldg | Lds | Ldp => Format::Load,
            Stg | Sts => Format::Store,
            S2r => Format::Special,
            Bra => Format::Branch,
            Bar | Exit => Format::None,
        }
    }

    pub fn mnemonic(self) -> &'static str {
        use Insn::*;
        match self {
            Iadd => "IADD",
            Isub => "ISUB",
            Imul => "IMUL",
            Imad => "IMAD",
            And => "AND",
            Or => "OR",
            Xor => "XOR",
            Shl => "SHL",
            Shr => "SHR",
            Sra => "SRA",
            Fadd => "FADD",
            Fsub => "FSUB",
            Fmul => "FMUL",
            Ffma => "FFMA",
            Fmin => "FMIN",
            Fmax => "FMAX",
            Fdiv => "FDIV",
            Fsqrt => "FSQRT",
            I2f => "I2F",
            F2i => "F2I",
            Mov => "MOV",
            S2r => "S2R",
            IsetpEq => "ISETP.EQ",
            IsetpNe => "ISETP.NE",
            IsetpLt => "ISETP.LT",
            IsetpLe => "ISETP.LE",
            IsetpGt => "ISETP.GT",
            IsetpGe => "ISETP.GE",
            FsetpEq => "FSETP.EQ",
            FsetpNe => "FSETP.NE",
            FsetpLt => "FSETP.LT",
            FsetpLe => "FSETP.LE",
            FsetpGt => "FSETP.GT",
            FsetpGe => "FSETP.GE",
            Ldg => "LDG",
            Stg => "STG",
            Lds => "LDS",
            Sts => "STS",
            Ldp => "LDP",
            ShflIdx => "SHFL.IDX",
            ShflBfly => "SHFL.BFLY",
            Bra => "BRA",
            Bar => "BAR",
            Exit => "EXIT",
        }
    }
}

/// An instruction with its operands, on physical registers.
///
/// `imm` is `Some` when the instruction's `i` bit is set. Register fields an
/// instruction doesn't use are zero.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Instruction {
    pub insn: Insn,
    /// Guard predicate: `0`–`6` for `p0`–`p6`, or [`PT`].
    pub guard: u8,
    /// Negates the guard.
    pub neg: bool,
    pub rd: u8,
    pub ra: u8,
    pub rb: u8,
    pub rc: u8,
    pub imm: Option<u32>,
}

impl Instruction {
    /// An unguarded instruction with all operand fields zero.
    pub fn new(insn: Insn) -> Self {
        Self {
            insn,
            guard: PT,
            neg: false,
            rd: 0,
            ra: 0,
            rb: 0,
            rc: 0,
            imm: None,
        }
    }

    /// Encodes the instruction as a 64-bit word (§7).
    pub fn encode(&self) -> u64 {
        let low = self.insn as u64
            | (self.guard as u64) << 8
            | (self.neg as u64) << 11
            | (self.rd as u64) << 12
            | (self.ra as u64) << 18
            | (self.rb as u64) << 24
            | (self.imm.is_some() as u64) << 30;
        let high = match self.imm {
            Some(imm) => imm as u64,
            None => self.rc as u64,
        };
        low | high << 32
    }
}

/// Disassembles the instruction in the syntax of §9 of the manual.
impl fmt::Display for Instruction {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.guard != PT || self.neg {
            let neg = if self.neg { "!" } else { "" };
            let pred = if self.guard == PT { "pt".to_string() } else { format!("p{}", self.guard) };
            write!(f, "@{neg}{pred} ")?;
        }
        write!(f, "{}", self.insn.mnemonic())?;
        let imm = self.imm.unwrap_or(0);
        let last = |reg: u8| match self.imm {
            Some(value) => format!("{value:#x}"),
            None => format!("r{reg}"),
        };
        let (rd, ra, rb, rc) = (self.rd, self.ra, self.rb, self.rc);
        match self.insn.format() {
            Format::R2 => write!(f, " r{rd}, r{ra}, {}", last(rb)),
            Format::R3 => write!(f, " r{rd}, r{ra}, r{rb}, {}", last(rc)),
            Format::R1 => write!(f, " r{rd}, {}", last(ra)),
            Format::Setp => {
                let pd = if rd == PT { "pt".to_string() } else { format!("p{rd}") };
                write!(f, " {pd}, r{ra}, {}", last(rb))
            }
            Format::Load => write!(f, " r{rd}, [r{ra} + {imm:#x}]"),
            Format::Store => write!(f, " [r{ra} + {imm:#x}], r{rb}"),
            Format::Special => {
                let name = match imm {
                    SR_TID => "%tid",
                    SR_NTID => "%ntid",
                    SR_CTAID => "%ctaid",
                    _ => "%nctaid",
                };
                write!(f, " r{rd}, {name}")
            }
            Format::Branch => write!(f, " {imm}"),
            Format::None => Ok(()),
        }
    }
}
