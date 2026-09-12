// The board the Titania GPU is simulated on: clocks the Verilated design,
// serves its memory ports from a model of global memory, and drives its host
// interface to load programs and launch kernels.
//
// Rust owns global memory and calls in here through the C functions at the
// bottom of the file.

#include <cstdint>
#include <cstring>
#include <deque>
#include <memory>

#include "Vtitania.h"
#include "verilated.h"

namespace {

// Cycles between a memory request and its response.
constexpr uint64_t MEM_LATENCY = 8;
// Requests a memory port can have in flight; matches the SM's load/store
// unit, which never sends more.
constexpr size_t MEM_DEPTH = 16;
// Cycles the host holds reset before a launch.
constexpr int RESET_CYCLES = 2;

constexpr int NUM_SMS = TITANIA_SMS;
constexpr uint32_t IMEM_WORDS = 4096;

// A memory request in flight: its response, due at a cycle.
struct Pending {
    uint64_t due;
    bool error;
    uint32_t error_addr;
    uint32_t data[32];
};

struct Titania {
    // Each simulated GPU has a context of its own, so that several can run
    // in different threads, each with its own simulation threads.
    std::unique_ptr<VerilatedContext> context = std::make_unique<VerilatedContext>();
    Vtitania dut{context.get()};
    std::deque<Pending> ports[NUM_SMS];
    uint32_t* mem = nullptr;
    size_t mem_words = 0;
    uint64_t cycle = 0;

    void tick() {
        dut.clk = 1;
        dut.eval();
        dut.clk = 0;
        dut.eval();
        cycle++;
    }

    void reset() {
        dut.rst = 1;
        dut.launch = 0;
        dut.prog_we = 0;
        dut.param_we = 0;
        for (int p = 0; p < NUM_SMS; p++) {
            dut.mem_req_ready[p] = 0;
            dut.mem_resp_valid[p] = 0;
            dut.mem_resp_error[p] = 0;
            ports[p].clear();
        }
        for (int i = 0; i < RESET_CYCLES; i++) tick();
        dut.rst = 0;
        cycle = 0;
    }

    // Serves the memory ports for one cycle, then clocks the design.
    //
    // The design's requests depend only on its state, so they are read as
    // its outputs stand after the last clock edge; the responses it is given
    // are sampled at the next.
    void step() {
        for (int p = 0; p < NUM_SMS; p++) {
            auto& port = ports[p];
            bool respond = !port.empty() && port.front().due <= cycle;
            dut.mem_resp_valid[p] = respond;
            if (respond) {
                const Pending& r = port.front();
                dut.mem_resp_error[p] = r.error;
                dut.mem_resp_addr[p] = r.error_addr;
                for (int l = 0; l < 32; l++) dut.mem_resp_rdata[p][l] = r.data[l];
            }
            dut.mem_req_ready[p] = port.size() < MEM_DEPTH;
            if (dut.mem_req_valid[p] && dut.mem_req_ready[p]) {
                Pending r;
                r.due = cycle + MEM_LATENCY;
                r.error = false;
                r.error_addr = 0;
                memset(r.data, 0, sizeof r.data);
                uint32_t mask = dut.mem_req_mask[p];
                bool store = dut.mem_req_we[p];
                for (int l = 0; l < 32; l++) {
                    if (!(mask >> l & 1)) continue;
                    uint32_t addr = dut.mem_req_addr[p][l];
                    if (addr % 4 != 0 || addr / 4 >= mem_words) {
                        if (!r.error) r.error_addr = addr;
                        r.error = true;
                        continue;
                    }
                    if (store) {
                        mem[addr / 4] = dut.mem_req_wdata[p][l];
                    } else {
                        r.data[l] = mem[addr / 4];
                    }
                }
                port.push_back(r);
            }
            if (dut.mem_resp_valid[p]) port.pop_front();
        }
        tick();
    }
};

}  // namespace

extern "C" {

struct titania_status {
    uint64_t cycles;
    uint64_t retired;
    uint32_t sample_block;
    uint32_t sample_warp;
    uint32_t sample_pc;
    uint32_t error_code;
    uint32_t error_pc;
    uint32_t error_addr;
};

Titania* titania_new(void) {
    return new Titania();
}

void titania_free(Titania* t) {
    delete t;
}

uint32_t titania_imem_words(void) {
    return IMEM_WORDS;
}

// Resets the GPU, loads a program and its parameters, and launches a grid
// over the given global memory, which must stay valid until the launch is
// done.
void titania_start(Titania* t, const uint64_t* program, uint32_t prog_len, const uint32_t* params,
                   uint32_t nparams, uint32_t grid_width, uint32_t grid_height, uint32_t block,
                   uint32_t shared, uint32_t* mem, size_t mem_words) {
    t->mem = mem;
    t->mem_words = mem_words;
    t->reset();
    for (uint32_t i = 0; i < prog_len; i++) {
        t->dut.prog_we = 1;
        t->dut.prog_addr = i;
        t->dut.prog_data = program[i];
        t->tick();
    }
    t->dut.prog_we = 0;
    for (uint32_t i = 0; i < nparams; i++) {
        t->dut.param_we = 1;
        t->dut.param_addr = i;
        t->dut.param_data = params[i];
        t->tick();
    }
    t->dut.param_we = 0;
    t->dut.grid_width = grid_width;
    t->dut.grid_height = grid_height;
    t->dut.block_size = block;
    t->dut.shared_bytes = shared;
    t->dut.prog_len = prog_len;
    t->dut.param_bytes = nparams * 4;
    t->dut.launch = 1;
    t->tick();
    t->dut.launch = 0;
    t->cycle = 0;
}

// Clocks the GPU for up to `max_cycles` cycles, or until the launch is done
// or fails. Returns 0 while the launch is still running, 1 when it is done,
// and 2 if it failed.
int titania_run(Titania* t, uint64_t max_cycles, titania_status* status) {
    int state = 0;
    for (uint64_t i = 0; i < max_cycles; i++) {
        if (t->dut.err_valid) {
            state = 2;
            break;
        }
        if (!t->dut.busy) {
            state = 1;
            break;
        }
        t->step();
    }
    if (state == 0 && t->dut.err_valid) state = 2;
    if (state == 0 && !t->dut.busy) state = 1;
    status->cycles = t->cycle;
    status->retired = t->dut.retired;
    status->sample_block = t->dut.sample_block;
    status->sample_warp = t->dut.sample_warp;
    status->sample_pc = t->dut.sample_pc;
    status->error_code = t->dut.err_code;
    status->error_pc = t->dut.err_pc;
    status->error_addr = t->dut.err_addr;
    return state;
}
}
