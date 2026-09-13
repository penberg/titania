// A streaming multiprocessor: executes one block at a time, interleaving its
// warps through a three-stage pipeline.
//
// Each warp has at most one instruction in flight, so the pipeline never
// needs to forward results or detect hazards: an instruction reads its
// operands after the warp's previous instruction has written back. With
// three or more warps ready, the pipeline issues an instruction every cycle.
//
//   Issue    picks a ready warp and fetches its instruction.
//   Decode   decodes it, evaluates its guard, and reads its operands.
//   Execute  computes the result and writes it back.
//
// Global memory loads and stores leave the pipeline at the execute stage for
// the load/store unit, which sends them to the memory port and writes loads
// back when their data returns, in order. Shared memory is banked 32 ways;
// an access whose lanes conflict on a bank takes a cycle per conflict.

module sm
  import isa::*;
  import fpu::*;
#(
    parameter int IMEM_WORDS = 4096,
    parameter int SMEM_WORDS = 16384,
    parameter int LSU_DEPTH = 16
) (
    input logic clk,
    input logic rst,

    // Program and parameter memories, loaded by the host before a launch.
    input logic prog_we,
    input logic [$clog2(IMEM_WORDS)-1:0] prog_addr,
    input logic [63:0] prog_data,
    input logic param_we,
    input logic [5:0] param_addr,
    input logic [31:0] param_data,

    // Launch geometry, held for the whole launch.
    input logic [31:0] grid_width,
    input logic [31:0] grid_height,
    input logic [10:0] block_size,
    input logic [16:0] shared_bytes,
    input logic [$clog2(IMEM_WORDS):0] prog_len,
    input logic [8:0] param_bytes,

    // Block dispatch: `start` assigns block `(block_x, block_y)` to an idle
    // SM; `block_id` is its number in row-major order, for the monitor.
    input logic start,
    input logic [31:0] block_id,
    input logic [31:0] block_x,
    input logic [31:0] block_y,
    output logic busy,
    output logic [31:0] block,

    // The first error, held until reset (§6).
    output logic err_valid,
    output logic [2:0] err_code,
    output logic [$clog2(IMEM_WORDS)-1:0] err_pc,
    output logic [31:0] err_addr,

    // Global memory port: a request accesses one word per lane in `mask`,
    // and responses return in order.
    output logic mem_req_valid,
    input logic mem_req_ready,
    output logic mem_req_we,
    output logic [31:0] mem_req_mask,
    output logic [31:0][31:0] mem_req_addr,
    output logic [31:0][31:0] mem_req_wdata,
    input logic mem_resp_valid,
    input logic [31:0][31:0] mem_resp_rdata,
    input logic mem_resp_error,
    input logic [31:0] mem_resp_addr,

    // Monitoring: pulses when an instruction completes, with where it was.
    output logic retired,
    output logic [4:0] sample_warp,
    output logic [$clog2(IMEM_WORDS)-1:0] sample_pc
);

  localparam int W = 32;  // warps per block, at most
  localparam int PCW = $clog2(IMEM_WORDS);
  localparam int ROWS = SMEM_WORDS / 32;  // words per shared memory bank
  localparam int ROWW = $clog2(ROWS);
  localparam int LSUW = $clog2(LSU_DEPTH);

  // ---------------------------------------------------------------------
  // State

  logic [63:0] imem[IMEM_WORDS];
  logic [31:0] params[64];
  // Register `r` of warp `w` is row `{w, r}`, one word per lane.
  logic [31:0][31:0] rf[W*64];
  // Shared memory: word `i` is in bank `i % 32`, row `i / 32`.
  logic [31:0] smem[32][ROWS];

  // The coordinates of the block being run.
  logic [31:0] block_x_r, block_y_r;

  logic [PCW-1:0] pc[W];
  // Lanes that exist and have not exited.
  logic [31:0] live[W];
  // Predicate registers, one bit per lane; entry 7 is `pt`, always true.
  logic [7:0][31:0] preds[W];
  // Warps that exist and have not finished; that have an instruction in
  // flight; and that are waiting at a barrier.
  logic [W-1:0] active, inflight, waiting;
  logic [4:0] last_issued;

  always_ff @(posedge clk) begin
    if (prog_we) imem[prog_addr] <= prog_data;
    if (param_we) params[param_addr] <= param_data;
  end

  // ---------------------------------------------------------------------
  // Issue: pick the ready warp after the last one issued, round robin.

  logic [W-1:0] ready;
  logic issue_valid, issue_pc_bad;
  logic [4:0] issue_warp;
  logic [PCW-1:0] issue_pc;
  logic stall;

  assign ready = active & ~inflight & ~waiting;

  always_comb begin
    issue_valid = 0;
    issue_warp = 0;
    for (int i = 1; i <= W; i++) begin
      if (!issue_valid && ready[5'(int'(last_issued)+i)]) begin
        issue_valid = 1;
        issue_warp = 5'(int'(last_issued) + i);
      end
    end
    issue_pc = pc[issue_warp];
    issue_pc_bad = {1'b0, issue_pc} >= prog_len;
  end

  // ---------------------------------------------------------------------
  // Decode: the instruction, its guard, and its operands.

  logic s1_valid;
  logic [4:0] s1_warp;
  logic [PCW-1:0] s1_pc;
  logic [63:0] s1_word;
  inst_t s1_inst;
  logic [31:0] s1_guard, s1_mask;

  assign s1_inst = decode(s1_word);

  always_comb begin
    s1_guard = preds[s1_warp][s1_inst.guard];
    s1_mask  = (s1_inst.neg ? ~s1_guard : s1_guard) & live[s1_warp];
  end

  // Register `r` of a warp; `r0` reads as zero.
  function automatic logic [31:0][31:0] rf_read(input logic [4:0] warp, input logic [5:0] r);
    return r == 0 ? '0 : rf[{warp, r}];
  endfunction

  // ---------------------------------------------------------------------
  // Execute

  logic s2_valid;
  logic [4:0] s2_warp;
  logic [PCW-1:0] s2_pc;
  inst_t s2_inst;
  logic [31:0] s2_mask;
  logic [31:0][31:0] s2_a, s2_b, s2_c;

  logic is_setp, is_global, is_shared, is_store, is_ldp, is_bra, is_bar, is_exit, writes_rd;
  logic uniform;  // the guard is true in every live lane

  always_comb begin
    is_setp   = s2_inst.format == F_SETP;
    is_global = s2_inst.op == LDG || s2_inst.op == STG;
    is_shared = s2_inst.op == LDS || s2_inst.op == STS;
    is_store  = s2_inst.op == STG || s2_inst.op == STS;
    is_ldp    = s2_inst.op == LDP;
    is_bra    = s2_inst.op == BRA;
    is_bar    = s2_inst.op == BAR;
    is_exit   = s2_inst.op == EXIT;
    writes_rd = !(is_setp || is_store || is_bra || is_bar || is_exit || is_global);
    uniform   = s2_mask == live[s2_warp];
  end

  // Source operands, with the immediate in place of the last register when
  // the `i` bit is set, and memory addresses.
  logic [31:0][31:0] op_a, op_b, op_c, addr;

  always_comb begin
    for (int l = 0; l < 32; l++) begin
      op_a[l] = s2_inst.has_imm && s2_inst.format == F_R1 ? s2_inst.imm : s2_a[l];
      op_b[l] = s2_inst.has_imm && (s2_inst.format == F_R2 || s2_inst.format == F_SETP) ?
          s2_inst.imm : s2_b[l];
      op_c[l] = s2_inst.has_imm && s2_inst.format == F_R3 ? s2_inst.imm : s2_c[l];
      addr[l] = s2_a[l] + s2_inst.imm;
    end
  end

  // Arithmetic that depends only on a lane's own operands.
  function automatic logic [31:0] alu(input logic [7:0] op, input logic [31:0] a,
                                      input logic [31:0] b, input logic [31:0] c);
    case (op)
      IADD: return a + b;
      ISUB: return a - b;
      IMUL: return a * b;
      IMAD: return a * b + c;
      AND: return a & b;
      OR: return a | b;
      XOR: return a ^ b;
      SHL: return a << b[4:0];
      SHR: return a >> b[4:0];
      SRA: return 32'($signed(a) >>> b[4:0]);
      FADD: return fadd32(a, b);
      FSUB: return fsub32(a, b);
      FMUL: return fmul32(a, b);
      FFMA: return fma32(a, b, c);
      FMIN: return fmin32(a, b);
      FMAX: return fmax32(a, b);
      FDIV: return fdiv32(a, b);
      FSQRT: return fsqrt32(a);
      I2F: return i2f(a);
      F2I: return f2i(a);
      MOV: return a;
      default: return 0;
    endcase
  endfunction

  // Comparisons: `FSETP` on a NaN is false, except `.NE`.
  function automatic logic compare(input logic [7:0] op, input logic [31:0] a, input logic [31:0] b);
    logic unordered;
    unordered = is_nan(a) || is_nan(b);
    case (op)
      ISETP_EQ: return a == b;
      ISETP_NE: return a != b;
      ISETP_LT: return $signed(a) < $signed(b);
      ISETP_LE: return $signed(a) <= $signed(b);
      ISETP_GT: return $signed(a) > $signed(b);
      ISETP_GE: return $signed(a) >= $signed(b);
      FSETP_EQ: return !unordered && feq(a, b);
      FSETP_NE: return unordered || !feq(a, b);
      FSETP_LT: return !unordered && flt(a, b);
      FSETP_LE: return !unordered && (flt(a, b) || feq(a, b));
      FSETP_GT: return !unordered && flt(b, a);
      FSETP_GE: return !unordered && (flt(b, a) || feq(a, b));
      default: return 0;
    endcase
  endfunction

  logic [31:0][31:0] result, smem_rdata;
  logic [31:0] pbits, ldp_bad;
  logic [4:0] from;

  always_comb begin
    result = '0;
    pbits = '0;
    ldp_bad = '0;
    from = 0;
    if (s2_valid) begin
      for (int l = 0; l < 32; l++) begin
        if (s2_mask[l]) begin
          case (s2_inst.op)
            S2R: begin
              case (s2_inst.imm)
                SR_TID: result[l] = {22'b0, s2_warp, 5'(l)};
                SR_NTID: result[l] = {21'b0, block_size};
                SR_CTAID_X: result[l] = block_x_r;
                SR_NCTAID_X: result[l] = grid_width;
                SR_CTAID_Y: result[l] = block_y_r;
                default: result[l] = grid_height;
              endcase
            end
            LDS: result[l] = smem_rdata[l];
            LDP: begin
              result[l] = params[addr[l][7:2]];
              ldp_bad[l] = addr[l][1:0] != 0 || addr[l] >= 32'(param_bytes);
            end
            SHFL_IDX, SHFL_BFLY: begin
              // A lane that does not exist or has exited yields the reader's
              // own value.
              from = s2_inst.op == SHFL_IDX ? op_b[l][4:0] : 5'(l) ^ op_b[l][4:0];
              result[l] = live[s2_warp][from] ? s2_a[from] : s2_a[l];
            end
            default: begin
              if (is_setp) pbits[l] = compare(s2_inst.op, op_a[l], op_b[l]);
              else result[l] = alu(s2_inst.op, op_a[l], op_b[l], op_c[l]);
            end
          endcase
        end
      end
    end
  end

  // ---------------------------------------------------------------------
  // Shared memory: each cycle, every bank serves the lowest lane that still
  // needs it. Loads land in `smem_rdata` the cycle after they are served,
  // and the access is done once nothing is pending.

  logic smem_active, smem_done;
  logic [31:0] smem_pending, smem_todo, smem_serve, smem_bad;
  logic [4:0] bank[32];
  logic [ROWW-1:0] row[32];

  always_comb begin
    smem_todo = smem_active ? smem_pending : s2_mask;
    smem_serve = '0;
    smem_bad = '0;
    for (int l = 0; l < 32; l++) begin
      bank[l] = addr[l][6:2];
      row[l] = addr[l][7+:ROWW];
    end
    if (s2_valid && is_shared) begin
      for (int l = 0; l < 32; l++) begin
        smem_bad[l] = s2_mask[l] && (addr[l][1:0] != 0 || addr[l] >= 32'(shared_bytes));
        smem_serve[l] = smem_todo[l];
        for (int k = 0; k < l; k++) begin
          if (smem_todo[k] && bank[k] == bank[l]) smem_serve[l] = 0;
        end
      end
    end
    smem_done = smem_active && smem_pending == 0;
  end

  always_ff @(posedge clk) begin
    if (rst || start) begin
      smem_active <= 0;
    end else if (s2_valid && is_shared) begin
      if (smem_done) begin
        smem_active <= 0;
      end else begin
        for (int l = 0; l < 32; l++) begin
          if (smem_serve[l]) begin
            if (is_store) smem[bank[l]][row[l]] <= s2_b[l];
            else smem_rdata[l] <= smem[bank[l]][row[l]];
          end
        end
        smem_pending <= smem_todo & ~smem_serve;
        smem_active <= 1;
      end
    end
  end

  // ---------------------------------------------------------------------
  // Load/store unit: global memory accesses in flight, in order.

  typedef struct packed {
    logic [4:0] warp;
    logic [5:0] rd;
    logic [31:0] mask;
    logic is_store;
    logic [PCW-1:0] pc;
  } lsu_entry_t;

  lsu_entry_t lsu_q[LSU_DEPTH];
  lsu_entry_t lsu_head_entry;
  logic [LSUW:0] lsu_count;
  logic [LSUW-1:0] lsu_head, lsu_tail;
  logic lsu_full, lsu_issue;
  logic [31:0] global_bad;

  always_comb begin
    lsu_full = lsu_count == (LSUW + 1)'(LSU_DEPTH);
    for (int l = 0; l < 32; l++) global_bad[l] = s2_mask[l] && addr[l][1:0] != 0;
    mem_req_valid = s2_valid && is_global && !lsu_full && global_bad == 0;
    mem_req_we = is_store;
    mem_req_mask = s2_mask;
    mem_req_addr = addr;
    mem_req_wdata = s2_b;
    lsu_issue = mem_req_valid && mem_req_ready;
    lsu_head_entry = lsu_q[lsu_head];
  end

  // ---------------------------------------------------------------------
  // Completion and errors

  logic s2_done;
  logic branch_taken, bar_wait;

  always_comb begin
    s2_done = !s2_valid || (is_global ? lsu_issue : is_shared ? smem_done : 1'b1);
    stall = s2_valid && !s2_done;
    branch_taken = is_bra && uniform;
    bar_wait = is_bar && uniform && s2_mask != 0;
  end

  logic err_hit;
  logic [2:0] err_hit_code;
  logic [PCW-1:0] err_hit_pc;
  logic [31:0] err_hit_addr;

  // The lowest lane whose address is bad.
  function automatic logic [31:0] first_bad(input logic [31:0] bad, input logic [31:0][31:0] addrs);
    logic [31:0] a = 0;
    for (int l = 31; l >= 0; l--) begin
      if (bad[l]) a = addrs[l];
    end
    return a;
  endfunction

  always_comb begin
    err_hit = 0;
    err_hit_code = 0;
    err_hit_pc = s2_pc;
    err_hit_addr = 0;
    if (issue_valid && issue_pc_bad && !stall) begin
      err_hit = 1;
      err_hit_code = ERR_PC;
      err_hit_pc = issue_pc;
    end
    if (s1_valid && !s1_inst.valid && !stall) begin
      err_hit = 1;
      err_hit_code = ERR_INVALID_INSTRUCTION;
      err_hit_pc = s1_pc;
    end
    if (mem_resp_valid && mem_resp_error) begin
      err_hit = 1;
      err_hit_code = ERR_GLOBAL;
      err_hit_pc = lsu_head_entry.pc;
      err_hit_addr = mem_resp_addr;
    end
    if (s2_valid) begin
      if (is_bra && !uniform && s2_mask != 0) begin
        err_hit = 1;
        err_hit_code = ERR_DIVERGENT_BRANCH;
        err_hit_pc = s2_pc;
      end
      if (is_bar && !uniform && s2_mask != 0) begin
        err_hit = 1;
        err_hit_code = ERR_DIVERGENT_BARRIER;
        err_hit_pc = s2_pc;
      end
      if (is_ldp && ldp_bad != 0) begin
        err_hit = 1;
        err_hit_code = ERR_PARAM;
        err_hit_pc = s2_pc;
        err_hit_addr = first_bad(ldp_bad, addr);
      end
      if (is_shared && smem_bad != 0) begin
        err_hit = 1;
        err_hit_code = ERR_SHARED;
        err_hit_pc = s2_pc;
        err_hit_addr = first_bad(smem_bad, addr);
      end
      if (is_global && global_bad != 0) begin
        err_hit = 1;
        err_hit_code = ERR_GLOBAL;
        err_hit_pc = s2_pc;
        err_hit_addr = first_bad(global_bad, addr);
      end
    end
  end

  always_ff @(posedge clk) begin
    if (rst) begin
      err_valid <= 0;
      err_code <= 0;
      err_pc <= 0;
      err_addr <= 0;
    end else if (err_hit && !err_valid) begin
      err_valid <= 1;
      err_code <= err_hit_code;
      err_pc <= err_hit_pc;
      err_addr <= err_hit_addr;
    end
  end

  // ---------------------------------------------------------------------
  // The pipeline

  // Live lanes of warp `w` at the start of a block.
  function automatic logic [31:0] lanes(input int w, input logic [10:0] size);
    int threads;
    threads = int'(size) - w * 32;
    if (threads >= 32) return 32'hffff_ffff;
    if (threads <= 0) return 0;
    return (32'd1 << threads) - 1;
  endfunction

  always_ff @(posedge clk) begin
    if (rst) begin
      busy <= 0;
      block <= 0;
      block_x_r <= 0;
      block_y_r <= 0;
      active <= 0;
      inflight <= 0;
      waiting <= 0;
      last_issued <= 5'd31;
      s1_valid <= 0;
      s2_valid <= 0;
      lsu_count <= 0;
      lsu_head <= 0;
      lsu_tail <= 0;
      retired <= 0;
      sample_warp <= 0;
      sample_pc <= 0;
    end else if (start) begin
      busy <= 1;
      block <= block_id;
      block_x_r <= block_x;
      block_y_r <= block_y;
      for (int w = 0; w < W; w++) begin
        active[w] <= lanes(w, block_size) != 0;
        pc[w] <= 0;
        live[w] <= lanes(w, block_size);
        preds[w] <= {32'hffff_ffff, 224'b0};
      end
      inflight <= 0;
      waiting <= 0;
      last_issued <= 5'd31;
      s1_valid <= 0;
      s2_valid <= 0;
      lsu_count <= 0;
      lsu_head <= 0;
      lsu_tail <= 0;
      retired <= 0;
    end else begin
      retired <= 0;

      // Issue and decode advance together unless execute is stalled.
      if (!stall) begin
        s1_valid <= issue_valid && !issue_pc_bad;
        if (issue_valid && !issue_pc_bad) begin
          s1_warp <= issue_warp;
          s1_pc <= issue_pc;
          s1_word <= imem[issue_pc];
          inflight[issue_warp] <= 1;
          last_issued <= issue_warp;
        end

        s2_valid <= s1_valid && s1_inst.valid;
        if (s1_valid) begin
          s2_warp <= s1_warp;
          s2_pc <= s1_pc;
          s2_inst <= s1_inst;
          s2_mask <= s1_mask;
          s2_a <= rf_read(s1_warp, s1_inst.ra);
          s2_b <= rf_read(s1_warp, s1_inst.rb);
          s2_c <= rf_read(s1_warp, s1_inst.rc);
        end
      end

      // Execute completes: write back and let the warp go on. Global
      // memory instructions stay in flight until their response.
      if (s2_valid && s2_done) begin
        if (writes_rd && s2_inst.rd != 0) begin
          for (int l = 0; l < 32; l++) begin
            if (s2_mask[l]) rf[{s2_warp, s2_inst.rd}][l] <= result[l];
          end
        end
        if (is_setp && s2_inst.rd[2:0] != PT) begin
          preds[s2_warp][s2_inst.rd[2:0]] <= (preds[s2_warp][s2_inst.rd[2:0]] & ~s2_mask) |
              (pbits & s2_mask);
        end
        pc[s2_warp] <= branch_taken ? s2_inst.imm[PCW-1:0] : s2_pc + 1;
        if (is_exit) begin
          live[s2_warp] <= live[s2_warp] & ~s2_mask;
          if ((live[s2_warp] & ~s2_mask) == 0) active[s2_warp] <= 0;
        end
        if (bar_wait) waiting[s2_warp] <= 1;
        if (!is_global) inflight[s2_warp] <= 0;
        if (is_global) begin
          lsu_q[lsu_tail] <= '{
              warp: s2_warp,
              rd: s2_inst.rd,
              mask: s2_mask,
              is_store: is_store,
              pc: s2_pc
          };
          lsu_tail <= lsu_tail + 1;
        end
        retired <= 1;
        sample_warp <= s2_warp;
        sample_pc <= s2_pc;
      end

      // A global memory response completes the oldest access in flight.
      if (mem_resp_valid) begin
        if (!lsu_head_entry.is_store && lsu_head_entry.rd != 0) begin
          for (int l = 0; l < 32; l++) begin
            if (lsu_head_entry.mask[l]) begin
              rf[{lsu_head_entry.warp, lsu_head_entry.rd}][l] <= mem_resp_rdata[l];
            end
          end
        end
        inflight[lsu_head_entry.warp] <= 0;
        lsu_head <= lsu_head + 1;
      end
      lsu_count <= lsu_count + (LSUW + 1)'(lsu_issue) - (LSUW + 1)'(mem_resp_valid);

      // A barrier releases once every unfinished warp is waiting at it.
      if (waiting != 0 && (active & ~waiting) == 0) waiting <= 0;

      // The block is done when every warp has finished.
      if (busy && active == 0) busy <= 0;
    end
  end

endmodule
