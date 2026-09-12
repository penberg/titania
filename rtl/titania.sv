// Titania: a GPU that executes the Titania ISA.
//
// The host loads a program and its parameters, then launches a grid. A
// dispatcher hands the grid's blocks, in row-major order, to the streaming
// multiprocessors (SMs) as they become idle, and the launch is done when
// every block has run.
// Each SM has a port to global memory, which lives outside the chip.
//
// The first error any SM raises stops the launch: `busy` stays high and
// `err_valid` reports it, until the next reset.

module titania
  import isa::*;
#(
    parameter int NUM_SMS = 4,
    parameter int IMEM_WORDS = 4096
) (
    input logic clk,
    input logic rst,

    // Program and parameter memories, written before a launch.
    input logic prog_we,
    input logic [$clog2(IMEM_WORDS)-1:0] prog_addr,
    input logic [63:0] prog_data,
    input logic param_we,
    input logic [5:0] param_addr,
    input logic [31:0] param_data,

    // Launch: pulse `launch` with the geometry (§5).
    input logic launch,
    input logic [31:0] grid_width,
    input logic [31:0] grid_height,
    input logic [10:0] block_size,
    input logic [16:0] shared_bytes,
    input logic [$clog2(IMEM_WORDS):0] prog_len,
    input logic [8:0] param_bytes,
    output logic busy,

    output logic err_valid,
    output logic [2:0] err_code,
    output logic [$clog2(IMEM_WORDS)-1:0] err_pc,
    output logic [31:0] err_addr,

    // Global memory ports, one per SM.
    output logic mem_req_valid[NUM_SMS],
    input logic mem_req_ready[NUM_SMS],
    output logic mem_req_we[NUM_SMS],
    output logic [31:0] mem_req_mask[NUM_SMS],
    output logic [31:0][31:0] mem_req_addr[NUM_SMS],
    output logic [31:0][31:0] mem_req_wdata[NUM_SMS],
    input logic mem_resp_valid[NUM_SMS],
    input logic [31:0][31:0] mem_resp_rdata[NUM_SMS],
    input logic mem_resp_error[NUM_SMS],
    input logic [31:0] mem_resp_addr[NUM_SMS],

    // Monitoring: instructions completed, and where the last one was.
    output logic [63:0] retired,
    output logic [31:0] sample_block,
    output logic [4:0] sample_warp,
    output logic [$clog2(IMEM_WORDS)-1:0] sample_pc
);

  localparam int PCW = $clog2(IMEM_WORDS);

  // The launch in progress.
  logic [31:0] grid_w_r, grid_h_r;
  logic [10:0] block_r;
  logic [16:0] shared_r;
  logic [PCW:0] prog_len_r;
  logic [8:0] param_bytes_r;
  // The next block to dispatch: its coordinates, and its number in
  // row-major order, for the monitor.
  logic [31:0] next_x, next_y, next_block;

  logic [NUM_SMS-1:0] sm_busy, sm_start, sm_err, sm_retired;
  logic [2:0] sm_err_code[NUM_SMS];
  logic [PCW-1:0] sm_err_pc[NUM_SMS];
  logic [31:0] sm_err_addr[NUM_SMS];
  logic [31:0] sm_block[NUM_SMS];
  logic [4:0] sm_sample_warp[NUM_SMS];
  logic [PCW-1:0] sm_sample_pc[NUM_SMS];

  for (genvar i = 0; i < NUM_SMS; i++) begin : sms
    sm #(
        .IMEM_WORDS(IMEM_WORDS)
    ) sm (
        .clk(clk),
        .rst(rst),
        .prog_we(prog_we),
        .prog_addr(prog_addr),
        .prog_data(prog_data),
        .param_we(param_we),
        .param_addr(param_addr),
        .param_data(param_data),
        .grid_width(grid_w_r),
        .grid_height(grid_h_r),
        .block_size(block_r),
        .shared_bytes(shared_r),
        .prog_len(prog_len_r),
        .param_bytes(param_bytes_r),
        .start(sm_start[i]),
        .block_id(next_block),
        .block_x(next_x),
        .block_y(next_y),
        .busy(sm_busy[i]),
        .block(sm_block[i]),
        .err_valid(sm_err[i]),
        .err_code(sm_err_code[i]),
        .err_pc(sm_err_pc[i]),
        .err_addr(sm_err_addr[i]),
        .mem_req_valid(mem_req_valid[i]),
        .mem_req_ready(mem_req_ready[i]),
        .mem_req_we(mem_req_we[i]),
        .mem_req_mask(mem_req_mask[i]),
        .mem_req_addr(mem_req_addr[i]),
        .mem_req_wdata(mem_req_wdata[i]),
        .mem_resp_valid(mem_resp_valid[i]),
        .mem_resp_rdata(mem_resp_rdata[i]),
        .mem_resp_error(mem_resp_error[i]),
        .mem_resp_addr(mem_resp_addr[i]),
        .retired(sm_retired[i]),
        .sample_warp(sm_sample_warp[i]),
        .sample_pc(sm_sample_pc[i])
    );
  end

  // Dispatch the next block to the first idle SM, one per cycle. The grid
  // has been handed out once `next_y` reaches its height.
  logic dispatch, found;

  always_comb begin
    dispatch = busy && !err_valid && next_y < grid_h_r;
    found = 0;
    sm_start = '0;
    for (int i = 0; i < NUM_SMS; i++) begin
      if (dispatch && !found && !sm_busy[i]) begin
        sm_start[i] = 1;
        found = 1;
      end
    end
  end

  always_ff @(posedge clk) begin
    if (rst) begin
      busy <= 0;
      next_x <= 0;
      next_y <= 0;
      next_block <= 0;
      grid_w_r <= 0;
      grid_h_r <= 0;
      block_r <= 0;
      shared_r <= 0;
      prog_len_r <= 0;
      param_bytes_r <= 0;
      err_valid <= 0;
      err_code <= 0;
      err_pc <= 0;
      err_addr <= 0;
      retired <= 0;
      sample_block <= 0;
      sample_warp <= 0;
      sample_pc <= 0;
    end else begin
      if (launch) begin
        busy <= 1;
        next_x <= 0;
        next_y <= 0;
        next_block <= 0;
        grid_w_r <= grid_width;
        grid_h_r <= grid_height;
        block_r <= block_size;
        shared_r <= shared_bytes;
        prog_len_r <= prog_len;
        param_bytes_r <= param_bytes;
        retired <= 0;
      end else if (busy) begin
        if (found) begin
          next_block <= next_block + 1;
          if (next_x + 1 == grid_w_r) begin
            next_x <= 0;
            next_y <= next_y + 1;
          end else begin
            next_x <= next_x + 1;
          end
        end
        if (next_y == grid_h_r && sm_busy == 0) busy <= 0;
      end

      if (!err_valid) begin
        for (int i = NUM_SMS - 1; i >= 0; i--) begin
          if (sm_err[i]) begin
            err_valid <= 1;
            err_code <= sm_err_code[i];
            err_pc <= sm_err_pc[i];
            err_addr <= sm_err_addr[i];
          end
        end
      end

      retired <= retired + 64'($countones(sm_retired));
      for (int i = NUM_SMS - 1; i >= 0; i--) begin
        if (sm_retired[i]) begin
          sample_block <= sm_block[i];
          sample_warp <= sm_sample_warp[i];
          sample_pc <= sm_sample_pc[i];
        end
      end
    end
  end

endmodule
