// The Titania instruction set as the hardware sees it: opcodes, operand
// formats, and the decoder for the 64-bit instruction word (§7 and §8 of the
// Titania GPU Architecture Reference Manual).

package isa;

  // Opcodes (§8).
  localparam logic [7:0] IADD = 8'h01;
  localparam logic [7:0] ISUB = 8'h02;
  localparam logic [7:0] IMUL = 8'h03;
  localparam logic [7:0] IMAD = 8'h04;
  localparam logic [7:0] AND = 8'h05;
  localparam logic [7:0] OR = 8'h06;
  localparam logic [7:0] XOR = 8'h07;
  localparam logic [7:0] SHL = 8'h08;
  localparam logic [7:0] SHR = 8'h09;
  localparam logic [7:0] SRA = 8'h0A;
  localparam logic [7:0] FADD = 8'h10;
  localparam logic [7:0] FSUB = 8'h11;
  localparam logic [7:0] FMUL = 8'h12;
  localparam logic [7:0] FFMA = 8'h13;
  localparam logic [7:0] FMIN = 8'h14;
  localparam logic [7:0] FMAX = 8'h15;
  localparam logic [7:0] FDIV = 8'h16;
  localparam logic [7:0] FSQRT = 8'h17;
  localparam logic [7:0] I2F = 8'h18;
  localparam logic [7:0] F2I = 8'h19;
  localparam logic [7:0] MOV = 8'h20;
  localparam logic [7:0] S2R = 8'h21;
  localparam logic [7:0] ISETP_EQ = 8'h28;
  localparam logic [7:0] ISETP_NE = 8'h29;
  localparam logic [7:0] ISETP_LT = 8'h2A;
  localparam logic [7:0] ISETP_LE = 8'h2B;
  localparam logic [7:0] ISETP_GT = 8'h2C;
  localparam logic [7:0] ISETP_GE = 8'h2D;
  localparam logic [7:0] FSETP_EQ = 8'h30;
  localparam logic [7:0] FSETP_NE = 8'h31;
  localparam logic [7:0] FSETP_LT = 8'h32;
  localparam logic [7:0] FSETP_LE = 8'h33;
  localparam logic [7:0] FSETP_GT = 8'h34;
  localparam logic [7:0] FSETP_GE = 8'h35;
  localparam logic [7:0] LDG = 8'h40;
  localparam logic [7:0] STG = 8'h41;
  localparam logic [7:0] LDS = 8'h42;
  localparam logic [7:0] STS = 8'h43;
  localparam logic [7:0] LDP = 8'h44;
  localparam logic [7:0] SHFL_IDX = 8'h48;
  localparam logic [7:0] SHFL_BFLY = 8'h49;
  localparam logic [7:0] BRA = 8'h50;
  localparam logic [7:0] BAR = 8'h51;
  localparam logic [7:0] EXIT = 8'h52;

  // The guard field value for `pt`, the always-true predicate.
  localparam logic [2:0] PT = 3'd7;

  // Errors (§6), as the hardware reports them.
  localparam logic [2:0] ERR_INVALID_INSTRUCTION = 3'd1;
  localparam logic [2:0] ERR_PC = 3'd2;
  localparam logic [2:0] ERR_DIVERGENT_BRANCH = 3'd3;
  localparam logic [2:0] ERR_DIVERGENT_BARRIER = 3'd4;
  localparam logic [2:0] ERR_GLOBAL = 3'd5;
  localparam logic [2:0] ERR_SHARED = 3'd6;
  localparam logic [2:0] ERR_PARAM = 3'd7;

  // Special registers (§2.3).
  localparam logic [31:0] SR_TID = 0;
  localparam logic [31:0] SR_NTID = 1;
  localparam logic [31:0] SR_CTAID_X = 2;
  localparam logic [31:0] SR_NCTAID_X = 3;
  localparam logic [31:0] SR_CTAID_Y = 4;
  localparam logic [31:0] SR_NCTAID_Y = 5;

  // The operands an instruction takes (§7 and §8).
  typedef enum logic [3:0] {
    F_R2,       // rd, ra, b
    F_R3,       // rd, ra, rb, c
    F_R1,       // rd, a
    F_SETP,     // pd, ra, b
    F_LOAD,     // rd, [ra + imm]
    F_STORE,    // [ra + imm], rb
    F_SPECIAL,  // rd, imm
    F_BRANCH,   // imm
    F_NONE,     // no operands
    F_INVALID   // unknown opcode
  } format_t;

  function automatic format_t format(input logic [7:0] op);
    case (op)
      IADD, ISUB, IMUL, AND, OR, XOR, SHL, SHR, SRA: return F_R2;
      FADD, FSUB, FMUL, FMIN, FMAX, FDIV, SHFL_IDX, SHFL_BFLY: return F_R2;
      IMAD, FFMA: return F_R3;
      FSQRT, I2F, F2I, MOV: return F_R1;
      ISETP_EQ, ISETP_NE, ISETP_LT, ISETP_LE, ISETP_GT, ISETP_GE: return F_SETP;
      FSETP_EQ, FSETP_NE, FSETP_LT, FSETP_LE, FSETP_GT, FSETP_GE: return F_SETP;
      LDG, LDS, LDP: return F_LOAD;
      STG, STS: return F_STORE;
      S2R: return F_SPECIAL;
      BRA: return F_BRANCH;
      BAR, EXIT: return F_NONE;
      default: return F_INVALID;
    endcase
  endfunction

  // A decoded instruction: the fields of §7.
  typedef struct packed {
    // Whether the opcode is known and every unused field is zero (§6).
    logic valid;
    logic [7:0] op;
    format_t format;
    logic [2:0] guard;
    logic neg;
    logic [5:0] rd;
    logic [5:0] ra;
    logic [5:0] rb;
    logic [5:0] rc;
    // The `i` bit: the last source operand is the immediate.
    logic has_imm;
    logic [31:0] imm;
  } inst_t;

  function automatic inst_t decode(input logic [63:0] word);
    inst_t inst;
    logic [31:0] high;
    logic fields_used, reserved_clear;
    high = word[63:32];
    inst.op = word[7:0];
    inst.format = format(inst.op);
    inst.guard = word[10:8];
    inst.neg = word[11];
    inst.rd = word[17:12];
    inst.ra = word[23:18];
    inst.rb = word[29:24];
    inst.has_imm = word[30];
    inst.rc = inst.has_imm ? 6'b0 : high[5:0];
    inst.imm = inst.has_imm ? high : 32'b0;
    case (inst.format)
      F_R2: fields_used = inst.rc == 0 && (!inst.has_imm || inst.rb == 0);
      F_R3: fields_used = 1;
      F_R1: fields_used = inst.rb == 0 && inst.rc == 0 && (!inst.has_imm || inst.ra == 0);
      F_SETP: fields_used = inst.rd < 8 && inst.rc == 0 && (!inst.has_imm || inst.rb == 0);
      F_LOAD: fields_used = inst.has_imm && inst.rb == 0;
      F_STORE: fields_used = inst.has_imm && inst.rd == 0;
      F_SPECIAL: fields_used = inst.has_imm && inst.ra == 0 && inst.rb == 0 && high <= SR_NCTAID_Y;
      F_BRANCH: fields_used = inst.has_imm && inst.rd == 0 && inst.ra == 0 && inst.rb == 0;
      F_NONE:
      fields_used = !inst.has_imm && inst.rd == 0 && inst.ra == 0 && inst.rb == 0 && inst.rc == 0;
      default: fields_used = 0;
    endcase
    reserved_clear = !word[31] && (inst.has_imm || high[31:6] == 0);
    inst.valid = fields_used && reserved_clear;
    return inst;
  endfunction

endpackage
