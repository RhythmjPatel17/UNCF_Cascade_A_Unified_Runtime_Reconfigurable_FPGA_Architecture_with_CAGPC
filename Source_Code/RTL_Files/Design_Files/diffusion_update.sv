import trident_pkg::*;

module diffusion_update #(
  parameter int LANES = PIX_DIM,
  parameter int NTOK  = N_TOK
)(
  input  logic                       clk,
  input  logic                       rst_n,

  input  logic                       start,
  output logic                       done,

  output logic                       a_re,
  output logic [$clog2(NTOK)-1:0]    a_raddr,
  input  logic [LANES*DATA_W-1:0]    a_rdata,

  output logic                       b_re,
  output logic [$clog2(NTOK)-1:0]    b_raddr,
  input  logic [LANES*DATA_W-1:0]    b_rdata,

  output logic                       out_we,
  output logic [$clog2(NTOK)-1:0]    out_waddr,
  output logic [LANES*DATA_W-1:0]    out_wdata
);

  localparam int TAW = $clog2(NTOK <= 1 ? 2 : NTOK);

  typedef enum logic [1:0] {S_IDLE, S_REQ, S_WAIT, S_DONE} state_e;
  state_e state, state_n;
  logic [TAW-1:0] tok_idx, tok_idx_n;
  logic [LANES*DATA_W-1:0] out_word;

  function automatic int8_t sat_sub8(input int8_t a, input int8_t b);
    logic signed [DATA_W:0] diff9;
    begin
      diff9 = {a[DATA_W-1], a} - {b[DATA_W-1], b};
      if (diff9 > 9'sd127)
        sat_sub8 = 8'sd127;
      else if (diff9 < -9'sd128)
        sat_sub8 = -8'sd128;
      else
        sat_sub8 = int8_t'(diff9[DATA_W-1:0]);
    end
  endfunction

  int8_t eps_lane;
  int8_t step_lane;

  integer li;
  always_comb begin
    for (li = 0; li < LANES; li++) begin
      eps_lane  = int8_t'(b_rdata[li*DATA_W +: DATA_W]);
      step_lane = eps_lane >>> UPD_SHIFT;
      out_word[li*DATA_W +: DATA_W] =
        sat_sub8(int8_t'(a_rdata[li*DATA_W +: DATA_W]), step_lane);
    end
  end

  always_comb begin
    state_n   = state;
    tok_idx_n = tok_idx;

    a_re = 1'b0; a_raddr = tok_idx;
    b_re = 1'b0; b_raddr = tok_idx;
    out_we = 1'b0; out_waddr = tok_idx; out_wdata = out_word;
    done = 1'b0;

    unique case (state)
      S_IDLE: begin
        if (start) begin
          tok_idx_n = '0;
          state_n   = S_REQ;
        end
      end
      S_REQ: begin
        a_re = 1'b1; a_raddr = tok_idx;
        b_re = 1'b1; b_raddr = tok_idx;
        state_n = S_WAIT;
      end
      S_WAIT: begin
        out_we    = 1'b1;
        out_waddr = tok_idx;
        out_wdata = out_word;
        if (tok_idx != TAW'(NTOK-1)) begin
          tok_idx_n = tok_idx + 1'b1;
          state_n   = S_REQ;
        end else begin
          state_n = S_DONE;
        end
      end
      S_DONE: begin
        done = 1'b1;
        if (!start) state_n = S_IDLE;
      end
      default: state_n = S_IDLE;
    endcase
  end

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      state   <= S_IDLE;
      tok_idx <= '0;
    end else begin
      state   <= state_n;
      tok_idx <= tok_idx_n;
    end
  end

endmodule : diffusion_update
