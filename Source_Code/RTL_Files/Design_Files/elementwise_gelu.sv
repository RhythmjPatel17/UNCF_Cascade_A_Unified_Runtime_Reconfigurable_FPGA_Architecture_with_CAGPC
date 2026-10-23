import trident_pkg::*;

module elementwise_gelu #(
  parameter int LANES = FFN_HID,
  parameter int NTOK  = N_TOK
)(
  input  logic                     clk,
  input  logic                     rst_n,

  input  logic                     start,
  output logic                     done,

  output logic                     in_re,
  output logic [$clog2(NTOK)-1:0]  in_raddr,
  input  logic [LANES*DATA_W-1:0]  in_rdata,

  output logic                     out_we,
  output logic [$clog2(NTOK)-1:0]  out_waddr,
  output logic [LANES*DATA_W-1:0]  out_wdata
);

  localparam int TAW = $clog2(NTOK  <= 1 ? 2 : NTOK);
  localparam int LAW = $clog2(LANES <= 1 ? 2 : LANES);

  typedef enum logic [2:0] {S_IDLE, S_LOAD_REQ, S_LOAD_WAIT, S_LUT_REQ, S_LUT_WAIT, S_WRITE, S_DONE} state_e;
  state_e state, state_n;

  logic [TAW-1:0] tok_idx, tok_idx_n;
  logic [LAW-1:0] lane,    lane_n;
  logic [LANES*DATA_W-1:0] in_word, in_word_n;
  logic [LANES*DATA_W-1:0] out_word, out_word_n;

  logic              gelu_re;
  logic [DATA_W-1:0] gelu_in, gelu_out;

  gelu_approx u_gelu (
    .clk      (clk),
    .re       (gelu_re),
    .in_bits  (gelu_in),
    .out_bits (gelu_out)
  );

  assign gelu_in = in_word[lane*DATA_W +: DATA_W];

  always_comb begin
    state_n    = state;
    tok_idx_n  = tok_idx;
    lane_n     = lane;
    in_word_n  = in_word;
    out_word_n = out_word;

    in_re     = 1'b0; in_raddr = tok_idx;
    out_we    = 1'b0; out_waddr = tok_idx; out_wdata = out_word;
    gelu_re   = 1'b0;
    done      = 1'b0;

    unique case (state)
      S_IDLE: begin
        if (start) begin
          tok_idx_n = '0;
          state_n   = S_LOAD_REQ;
        end
      end

      S_LOAD_REQ: begin
        in_re    = 1'b1; in_raddr = tok_idx;
        state_n  = S_LOAD_WAIT;
      end
      S_LOAD_WAIT: begin
        in_word_n = in_rdata;
        lane_n    = '0;
        state_n   = S_LUT_REQ;
      end

      S_LUT_REQ: begin
        gelu_re = 1'b1;
        state_n = S_LUT_WAIT;
      end
      S_LUT_WAIT: begin
        out_word_n[lane*DATA_W +: DATA_W] = gelu_out;
        if (lane != LAW'(LANES-1)) begin
          lane_n  = lane + 1'b1;
          state_n = S_LUT_REQ;
        end else begin
          state_n = S_WRITE;
        end
      end

      S_WRITE: begin
        out_we    = 1'b1;
        out_waddr = tok_idx;
        out_wdata = out_word_n;
        if (tok_idx != TAW'(NTOK-1)) begin
          tok_idx_n = tok_idx + 1'b1;
          state_n   = S_LOAD_REQ;
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
      state    <= S_IDLE;
      tok_idx  <= '0;
      lane     <= '0;
      in_word  <= '0;
      out_word <= '0;
    end else begin
      state    <= state_n;
      tok_idx  <= tok_idx_n;
      lane     <= lane_n;
      in_word  <= in_word_n;
      out_word <= out_word_n;
    end
  end

endmodule : elementwise_gelu
