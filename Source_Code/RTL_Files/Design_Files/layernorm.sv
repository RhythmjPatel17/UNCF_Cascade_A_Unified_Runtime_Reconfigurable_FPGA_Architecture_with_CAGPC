import trident_pkg::*;

module layernorm (
  input  logic                    clk,
  input  logic                    rst_n,

  input  logic                    start,
  output logic                    done,

  output logic                    in_re,
  output logic [TOK_AW-1:0]       in_raddr,
  input  logic [HID_D*DATA_W-1:0] in_rdata,

  output logic                    out_we,
  output logic [TOK_AW-1:0]       out_waddr,
  output logic [HID_D*DATA_W-1:0] out_wdata,

  output logic                    gb_re,
  output logic [$clog2(HID_D<=1?2:HID_D)-1:0] gb_raddr,
  input  logic [15:0]             gb_rdata
);

  localparam int D_AW = $clog2(HID_D <= 1 ? 2 : HID_D);

  logic [7:0] rsqrt_lut [0:7];
  initial begin
    rsqrt_lut[0] = 8'd16;
    rsqrt_lut[1] = 8'd11;
    rsqrt_lut[2] = 8'd8;
    rsqrt_lut[3] = 8'd6;
    rsqrt_lut[4] = 8'd4;
    rsqrt_lut[5] = 8'd3;
    rsqrt_lut[6] = 8'd2;
    rsqrt_lut[7] = 8'd1;
  end

  typedef enum logic [3:0] {
    S_IDLE, S_GB_REQ, S_GB_WAIT,
    S_LOAD_REQ, S_LOAD_WAIT, S_STAT_D, S_NORM_D, S_WRITE, S_DONE
  } state_e;
  state_e state, state_n;

  logic [TOK_AW-1:0] tok_idx, tok_idx_n;
  logic [D_AW-1:0]   d_idx,   d_idx_n;
  logic [D_AW-1:0]   gb_idx,  gb_idx_n;

  int8_t gamma_arr [0:HID_D-1];
  int8_t beta_arr  [0:HID_D-1];

  logic [HID_D*DATA_W-1:0] in_word, in_word_n;
  logic [HID_D*DATA_W-1:0] out_word, out_word_n;

  acc_t sum_acc,  sum_acc_n;
  acc_t sumsq_acc, sumsq_acc_n;

  int8_t cur_lane;
  assign cur_lane = int8_t'(in_word[d_idx*DATA_W +: DATA_W]);

  acc_t mean_q;
  acc_t var_q;

  logic [2:0] msb_idx;

  always_comb begin
    if      (var_q >= 32'sd128) msb_idx = 3'd7;
    else if (var_q >= 32'sd64)  msb_idx = 3'd6;
    else if (var_q >= 32'sd32)  msb_idx = 3'd5;
    else if (var_q >= 32'sd16)  msb_idx = 3'd4;
    else if (var_q >= 32'sd8)   msb_idx = 3'd3;
    else if (var_q >= 32'sd4)   msb_idx = 3'd2;
    else if (var_q >= 32'sd2)   msb_idx = 3'd1;
    else                        msb_idx = 3'd0;
  end

  assign mean_q = sum_acc / HID_D;
  assign var_q  = (sumsq_acc / HID_D) - ((mean_q*mean_q) >>> 4);

  acc_t centered, normed;
  assign centered = acc_t'(cur_lane) - mean_q;
  assign normed   = (centered * acc_t'(rsqrt_lut[msb_idx])) >>> 4;

  acc_t affine_prod;
  assign affine_prod = normed * acc_t'(gamma_arr[d_idx]);
  int8_t affine_scaled;
  assign affine_scaled = requantize(affine_prod, QSHIFT);

  always_comb begin
    state_n     = state;
    tok_idx_n   = tok_idx;
    d_idx_n     = d_idx;
    gb_idx_n    = gb_idx;
    in_word_n   = in_word;
    out_word_n  = out_word;
    sum_acc_n   = sum_acc;
    sumsq_acc_n = sumsq_acc;

    in_re     = 1'b0; in_raddr = tok_idx;
    out_we    = 1'b0; out_waddr = tok_idx; out_wdata = out_word;
    gb_re     = 1'b0; gb_raddr  = gb_idx;
    done      = 1'b0;

    unique case (state)
      S_IDLE: begin
        if (start) begin
          gb_idx_n = '0;
          state_n  = S_GB_REQ;
        end
      end

      S_GB_REQ: begin
        gb_re    = 1'b1; gb_raddr = gb_idx;
        state_n  = S_GB_WAIT;
      end
      S_GB_WAIT: begin
        if (gb_idx != D_AW'(HID_D-1)) begin
          gb_idx_n = gb_idx + 1'b1;
          state_n  = S_GB_REQ;
        end else begin
          tok_idx_n = '0;
          state_n   = S_LOAD_REQ;
        end
      end

      S_LOAD_REQ: begin
        in_re    = 1'b1; in_raddr = tok_idx;
        state_n  = S_LOAD_WAIT;
      end
      S_LOAD_WAIT: begin
        in_word_n   = in_rdata;
        d_idx_n     = '0;
        sum_acc_n   = '0;
        sumsq_acc_n = '0;
        state_n     = S_STAT_D;
      end

      S_STAT_D: begin
        sum_acc_n   = sum_acc + acc_t'(int8_t'(in_word[d_idx*DATA_W +: DATA_W]));
        sumsq_acc_n = sumsq_acc + acc_t'(int8_t'(in_word[d_idx*DATA_W +: DATA_W])) *
                                    acc_t'(int8_t'(in_word[d_idx*DATA_W +: DATA_W]));
        if (d_idx != D_AW'(HID_D-1)) begin
          d_idx_n = d_idx + 1'b1;
        end else begin
          d_idx_n = '0;
          state_n = S_NORM_D;
        end
      end

      S_NORM_D: begin
        out_word_n[d_idx*DATA_W +: DATA_W] = sat_add8(affine_scaled, beta_arr[d_idx]);
        if (d_idx != D_AW'(HID_D-1)) begin
          d_idx_n = d_idx + 1'b1;
        end else begin
          state_n = S_WRITE;
        end
      end

      S_WRITE: begin
        out_we    = 1'b1;
        out_waddr = tok_idx;
        out_wdata = out_word_n;
        if (tok_idx != TOK_AW'(N_TOK-1)) begin
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
      state      <= S_IDLE;
      tok_idx    <= '0;
      d_idx      <= '0;
      gb_idx     <= '0;
      in_word    <= '0;
      out_word   <= '0;
      sum_acc    <= '0;
      sumsq_acc  <= '0;
    end else begin
      state      <= state_n;
      tok_idx    <= tok_idx_n;
      d_idx      <= d_idx_n;
      gb_idx     <= gb_idx_n;
      in_word    <= in_word_n;
      out_word   <= out_word_n;
      sum_acc    <= sum_acc_n;
      sumsq_acc  <= sumsq_acc_n;
    end
  end

  integer gi;
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      for (gi = 0; gi < HID_D; gi++) begin
        gamma_arr[gi] <= 8'sd8;
        beta_arr[gi]  <= 8'sd0;
      end
    end else if (state == S_GB_WAIT) begin
      gamma_arr[gb_idx] <= int8_t'(gb_rdata[7:0]);
      beta_arr[gb_idx]  <= int8_t'(gb_rdata[15:8]);
    end
  end

endmodule : layernorm
