// Floating-point arithmetic for the Titania GPU: IEEE 754 binary32 with
// round-to-nearest-ties-to-even, subnormals, and one canonical NaN, exactly
// as §4 of the Titania GPU Architecture Reference Manual requires.
//
// Every operation is a pure function of its operands, so a lane computes it
// combinationally in the execute stage. The functions share one rounding
// routine, `pack`, which takes an exact result as an integer magnitude, a
// power-of-two scale, and a sticky bit, and rounds it once.

package fpu;

  localparam logic [31:0] NAN = 32'h7fc0_0000;

  // Width of the fixed-point frame `pack` rounds from. The larger operand of
  // a sum sits at its top; bits of the smaller that fall below it are too far
  // down to matter except as the sticky bit. 64 bits keep every intermediate
  // a machine word in simulation: a 48-bit product and a 24-bit addend that
  // nearly cancel fit with room to spare.
  localparam int FW = 64;

  localparam logic [31:0] ONE = 32'h3f80_0000;
  localparam logic [31:0] NEG_ZERO = 32'h8000_0000;

  function automatic logic is_nan(input logic [31:0] x);
    return x[30:23] == 8'hff && x[22:0] != 0;
  endfunction

  function automatic logic is_inf(input logic [31:0] x);
    return x[30:23] == 8'hff && x[22:0] == 0;
  endfunction

  function automatic logic is_zero(input logic [31:0] x);
    return x[30:0] == 0;
  endfunction

  // The significand with its hidden bit, and the exponent it is scaled by:
  // `x = sig × 2^(exp - 150)`. Subnormals have exponent 1 and no hidden bit.
  function automatic logic [23:0] significand(input logic [31:0] x);
    return {x[30:23] != 0, x[22:0]};
  endfunction

  function automatic int exponent(input logic [31:0] x);
    return x[30:23] == 0 ? 1 : int'(x[30:23]);
  endfunction

  // Index of the highest set bit of `x`, or 0 if none.
  function automatic int leading_one(input logic [FW-1:0] x);
    int lead = 0;
    logic [FW-1:0] v = x;
    for (int step = 32; step >= 1; step = step / 2) begin
      if ((v >> step) != 0) begin
        v = v >> step;
        lead = lead + step;
      end
    end
    return lead;
  endfunction

  // `x >> sh`, with `sticky` set if any bits were shifted out.
  function automatic void shift_right(input logic [FW-1:0] x, input int sh, output logic [FW-1:0] y,
                                      output logic sticky);
    if (sh >= FW) begin
      y = 0;
      sticky = x != 0;
    end else if (sh == 0) begin
      y = x;
      sticky = 0;
    end else begin
      y = x >> sh;
      sticky = (x << (FW - sh)) != 0;
    end
  endfunction

  // Rounds `sign × mag × 2^e` to the nearest f32, ties to even. `sticky` says
  // that the exact value has nonzero bits below `mag`'s lowest.
  function automatic logic [31:0] pack(input logic sign, input logic [FW-1:0] mag, input int e,
                                       input logic sticky);
    int lead, eres, k;
    logic [23:0] sig;
    logic round, st;
    logic [24:0] rounded;
    logic [FW-1:0] shifted;
    int efield;
    logic [22:0] mant;
    if (mag == 0) return {sign, 31'b0};
    lead = leading_one(mag);
    // The biased exponent of the result: that of its leading bit, or the
    // subnormal exponent if that is smaller.
    eres = e + lead + 127;
    if (eres < 1) eres = 1;
    // Bit `k` of `mag` becomes the result significand's lowest bit.
    k = eres - 150 - e;
    if (k < 0) begin
      sig = 24'(mag << (-k));
      round = 0;
      st = sticky;
    end else if (k == 0) begin
      sig = 24'(mag);
      round = 0;
      st = sticky;
    end else begin
      // Shift out `k` bits: the highest is the round bit and the rest are
      // sticky.
      shift_right(mag, k - 1, shifted, st);
      st = st | sticky;
      round = shifted[0];
      sig = 24'(shifted >> 1);
    end
    rounded = {1'b0, sig} + {24'b0, round & (st | sig[0])};
    if (rounded[24]) begin
      efield = eres + 1;
      mant = rounded[23:1];
    end else if (rounded[23]) begin
      efield = eres;
      mant = rounded[22:0];
    end else begin
      efield = 0;
      mant = rounded[22:0];
    end
    if (efield >= 255) return {sign, 8'hff, 23'b0};
    return {sign, efield[7:0], mant};
  endfunction

  // Positions `x` in the frame with its lowest bit shifted right by `sh` (or
  // left, if `sh` is negative), setting `sticky` if any bits are shifted out.
  function automatic void align(input logic [47:0] x, input int sh, output logic [FW-1:0] y,
                                output logic sticky);
    if (sh <= 0) begin
      y = {{(FW - 48) {1'b0}}, x} << (-sh);
      sticky = 0;
    end else begin
      shift_right({{(FW - 48) {1'b0}}, x}, sh, y, sticky);
    end
  endfunction

  // `a × b + c`, rounded once.
  function automatic logic [31:0] fma32(input logic [31:0] a, input logic [31:0] b,
                                        input logic [31:0] c);
    logic sa, sb, sc, sp, sign, sticky, stp, stc;
    logic [47:0] prod;
    int ep, ec, etop, elsb;
    logic [FW-1:0] pa, ca, mag;
    sa = a[31];
    sb = b[31];
    sc = c[31];
    sp = sa ^ sb;
    if (is_nan(a) || is_nan(b) || is_nan(c)) return NAN;
    if ((is_inf(a) && is_zero(b)) || (is_zero(a) && is_inf(b))) return NAN;
    if (is_inf(a) || is_inf(b)) begin
      if (is_inf(c) && sc != sp) return NAN;
      return {sp, 8'hff, 23'b0};
    end
    if (is_inf(c)) return c;
    if (is_zero(a) || is_zero(b)) begin
      // The product is an exact zero: the sum is `c`, unless that is zero
      // too, when the sign is negative only if both are.
      if (is_zero(c)) return {sp & sc, 31'b0};
      return c;
    end
    // The product and addend as integers scaled by powers of two.
    prod = 48'(significand(a)) * 48'(significand(b));
    ep = exponent(a) + exponent(b) - 300;
    ec = exponent(c) - 150;
    // Frame the larger of the two at the top, and align the other to it. Bits
    // of the smaller that fall below the frame can only be sticky: they are
    // too far down to affect anything but the rounding. A zero addend has no
    // bits at all, so the product is framed whatever its exponent.
    etop = is_zero(c) || ep + 48 > ec + 24 ? ep + 48 : ec + 24;
    elsb = etop - (FW - 2);
    align(prod, elsb - ep, pa, stp);
    align({24'b0, significand(c)}, elsb - ec, ca, stc);
    sticky = stp | stc;
    if (sp == sc) begin
      mag = pa + ca;
      sign = sp;
    end else if (pa > ca || (pa == ca && stp)) begin
      // The product is larger. Subtracting an addend with sticky bits below
      // the frame borrows one from the frame; the sticky bit then stands for
      // the fraction left over.
      mag = pa - ca - FW'(stc);
      sign = sp;
    end else begin
      mag = ca - pa - FW'(stp);
      sign = sc;
    end
    if (mag == 0 && !sticky) return {sp & sc, 31'b0};
    return pack(sign, mag, elsb, sticky);
  endfunction

  function automatic logic [31:0] fadd32(input logic [31:0] a, input logic [31:0] b);
    return fma32(a, ONE, b);
  endfunction

  function automatic logic [31:0] fsub32(input logic [31:0] a, input logic [31:0] b);
    return fma32(a, ONE, b ^ NEG_ZERO);
  endfunction

  function automatic logic [31:0] fmul32(input logic [31:0] a, input logic [31:0] b);
    // Adding -0 leaves every product, including -0, as it is.
    return fma32(a, b, NEG_ZERO);
  endfunction

  // Normalizes a subnormal significand so that division and square root see
  // a hidden bit, adjusting the exponent to match.
  function automatic void normalize(input logic [31:0] x, output logic [23:0] sig, output int e);
    sig = significand(x);
    e = exponent(x);
    for (int i = 0; i < 23; i++) begin
      if (!sig[23]) begin
        sig = sig << 1;
        e = e - 1;
      end
    end
  endfunction

  function automatic logic [31:0] fdiv32(input logic [31:0] a, input logic [31:0] b);
    logic s;
    logic [23:0] siga, sigb;
    int ea, eb;
    logic [49:0] num, q, r;
    s = a[31] ^ b[31];
    if (is_nan(a) || is_nan(b)) return NAN;
    if ((is_inf(a) && is_inf(b)) || (is_zero(a) && is_zero(b))) return NAN;
    if (is_inf(a) || is_zero(b)) return {s, 8'hff, 23'b0};
    if (is_zero(a) || is_inf(b)) return {s, 31'b0};
    normalize(a, siga, ea);
    normalize(b, sigb, eb);
    // The quotient of the significands to 26 fractional bits, which is at
    // least 2^25 since both are in [2^23, 2^24): enough to round exactly with
    // the remainder as the sticky bit.
    num = {siga, 26'b0};
    q = num / 50'(sigb);
    r = num % 50'(sigb);
    return pack(s, FW'(q), ea - eb - 26, r != 0);
  endfunction

  // Integer square root by the restoring method: `x = root² + rem`.
  function automatic void isqrt(input logic [53:0] x, output logic [26:0] root,
                                output logic [55:0] rem);
    logic [55:0] trial;
    root = 0;
    rem = 0;
    for (int i = 26; i >= 0; i--) begin
      rem = {rem[53:0], x[2*i+:2]};
      trial = {27'b0, root, 2'b01};
      if (rem >= trial) begin
        rem = rem - trial;
        root = {root[25:0], 1'b1};
      end else begin
        root = {root[25:0], 1'b0};
      end
    end
  endfunction

  function automatic logic [31:0] fsqrt32(input logic [31:0] a);
    logic [23:0] sig;
    int e;
    logic [53:0] x;
    logic [26:0] root;
    logic [55:0] rem;
    if (is_nan(a)) return NAN;
    if (is_zero(a) || is_inf(a) && !a[31]) return a;
    if (a[31]) return NAN;
    normalize(a, sig, e);
    // a = sig × 2^(e - 150). Make the exponent even, then take the root of
    // the significand to 14 fractional bits, which gives at least 26 bits.
    e = e - 150;
    if (e[0]) begin
      x = {1'b0, sig, 29'b0};
      e = e - 1;
    end else begin
      x = {2'b0, sig, 28'b0};
    end
    isqrt(x, root, rem);
    return pack(0, FW'(root), e / 2 - 14, rem != 0);
  endfunction

  // Converts a signed integer to f32.
  function automatic logic [31:0] i2f(input logic [31:0] x);
    logic [32:0] mag;
    mag = x[31] ? -{x[31], x} : {1'b0, x};
    return pack(x[31], FW'(mag), 0, 0);
  endfunction

  // Converts an f32 to a signed integer, rounding to nearest, ties to even.
  // Out-of-range values saturate, and NaN converts to 0.
  function automatic logic [31:0] f2i(input logic [31:0] x);
    int e, sh;
    logic [23:0] sig;
    logic [31:0] mag;
    logic round, sticky;
    logic [46:0] wide;
    if (is_nan(x)) return 0;
    e = int'(x[30:23]) - 127;
    if (e >= 31) return x[31] ? 32'h8000_0000 : 32'h7fff_ffff;
    if (e < -1) return 0;
    sig = significand(x);
    if (e >= 23) begin
      mag = 32'(sig) << (e - 23);
    end else begin
      sh = 23 - e;
      wide = {sig, 23'b0} >> sh;
      mag = 32'(wide[46:23]);
      round = wide[22];
      sticky = wide[21:0] != 0;
      mag = mag + {31'b0, round & (sticky | mag[0])};
    end
    return x[31] ? -mag : mag;
  endfunction

  // Comparisons, for operands that are not NaN.
  function automatic logic feq(input logic [31:0] a, input logic [31:0] b);
    return a == b || (is_zero(a) && is_zero(b));
  endfunction

  function automatic logic flt(input logic [31:0] a, input logic [31:0] b);
    if (is_zero(a) && is_zero(b)) return 0;
    if (a[31] != b[31]) return a[31];
    return a[31] ? a[30:0] > b[30:0] : a[30:0] < b[30:0];
  endfunction

  // `FMIN`: `-0.0` is less than `+0.0`, and a NaN operand yields the other.
  function automatic logic [31:0] fmin32(input logic [31:0] a, input logic [31:0] b);
    if (is_nan(a) && is_nan(b)) return NAN;
    if (is_nan(a)) return b;
    if (is_nan(b)) return a;
    if (flt(a, b)) return a;
    if (flt(b, a)) return b;
    // Equal: either the same bits, or zeros of opposite signs.
    return a | b;
  endfunction

  // `FMAX`: `+0.0` is greater than `-0.0`, and a NaN operand yields the other.
  function automatic logic [31:0] fmax32(input logic [31:0] a, input logic [31:0] b);
    if (is_nan(a) && is_nan(b)) return NAN;
    if (is_nan(a)) return b;
    if (is_nan(b)) return a;
    if (flt(b, a)) return a;
    if (flt(a, b)) return b;
    return a & b;
  endfunction

endpackage
