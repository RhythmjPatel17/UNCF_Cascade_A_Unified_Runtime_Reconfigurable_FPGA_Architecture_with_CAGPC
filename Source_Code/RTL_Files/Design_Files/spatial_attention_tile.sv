import trident_pkg::*;

module spatial_attention_tile (
  input  logic                    clk,
  input  logic                    rst_n,

  input  logic                    start,
  output logic                    done,

  output logic                    q_re,
  output logic [TOK_AW-1:0]       q_raddr,
  input  logic [HID_D*DATA_W-1:0] q_rdata,

  output logic                    k_re,
  output logic [TOK_AW-1:0]       k_raddr,
  input  logic [HID_D*DATA_W-1:0] k_rdata,

  output logic                    v_re,
  output logic [TOK_AW-1:0]       v_raddr,
  input  logic [HID_D*DATA_W-1:0] v_rdata,

  output logic                    out_we,
  output logic [TOK_AW-1:0]       out_waddr,
  output logic [HID_D*DATA_W-1:0] out_wdata
);

  localparam int NWIN   = N_TOK / WIN_W;
  localparam int WIN_AW = $clog2(NWIN  <= 1 ? 2 : NWIN);
  localparam int WI_AW  = $clog2(WIN_W <= 1 ? 2 : WIN_W);
  localparam int D_AW   = $clog2(HID_D <= 1 ? 2 : HID_D);
  localparam int HD     = HID_D / N_HEADS;
  localparam int HD_AW  = $clog2(HD <= 1 ? 2 : HD);
  localparam int H_AW   = $clog2(N_HEADS <= 1 ? 2 : N_HEADS);

  typedef enum logic [4:0] {
    S_IDLE,
    S_KV_REQ, S_KV_WAIT,
    S_Q_REQ,  S_Q_WAIT,
    S_SCORE_REQ_D, S_SCORE_ACC_D, S_SCORE_STORE, S_SCORE_NEXT_H,
    S_SM_H_INIT,
    S_SM_MAX_INIT, S_SM_MAX_SCAN,
    S_SM_EXP_REQ,  S_SM_EXP_WAIT,
    S_SM_DIV_START, S_SM_DIV_WAIT, S_SM_NEXT_H,
    S_OUT_H_INIT,
    S_OUT_ACC_KJ, S_OUT_STORE_D,
    S_WRITE_TOK,
    S_NEXT_Q, S_NEXT_WIN,
    S_DONE
  } state_e;
  state_e state, state_n;

  logic [WIN_AW-1:0] win_idx, win_idx_n;
  logic [WI_AW-1:0]  kv_idx,  kv_idx_n;
  logic [WI_AW-1:0]  qi,      qi_n;
  logic [WI_AW-1:0]  kj,      kj_n;
  logic [HD_AW-1:0]  d_idx,   d_idx_n;
  logic [WI_AW-1:0]  midx,    midx_n;
  logic [WI_AW-1:0]  eidx,    eidx_n;
  logic [WI_AW-1:0]  didx,    didx_n;
  logic [H_AW-1:0]   h_idx,   h_idx_n;

  int8_t k_cache    [0:WIN_W-1][0:HID_D-1];
  int8_t v_cache    [0:WIN_W-1][0:HID_D-1];
  int8_t raw_scores [0:N_HEADS-1][0:WIN_W-1];
  int8_t exp_arr    [0:N_HEADS-1][0:WIN_W-1];
  int8_t weight_arr [0:N_HEADS-1][0:WIN_W-1];

  genvar gh;
  generate
    for (gh = 0; gh < N_HEADS; gh = gh + 1) begin : g_reset_head
      integer gi;
      always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
          for (gi = 0; gi < WIN_W; gi = gi + 1) begin
            raw_scores[gh][gi] <= '0;
            exp_arr[gh][gi]    <= '0;
            weight_arr[gh][gi] <= '0;
          end
        end
      end
    end
  endgenerate

  logic [HID_D*DATA_W-1:0] q_word, q_word_n;
  logic [HID_D*DATA_W-1:0] out_word, out_word_n;

  acc_t score_acc, score_acc_n;
  acc_t out_acc,   out_acc_n;

  int8_t max_val, max_val_n;
  logic [15:0] sum_reg, sum_reg_n;

  logic [TOK_AW-1:0] cur_tok;
  assign cur_tok = TOK_AW'(win_idx * WIN_W + qi);
  logic [TOK_AW-1:0] cur_kv_tok;
  assign cur_kv_tok = TOK_AW'(win_idx * WIN_W + kv_idx);

  logic [D_AW-1:0] abs_d_idx;
  assign abs_d_idx = D_AW'(h_idx * HD + d_idx);

  int8_t q_lane;
  assign q_lane = int8_t'(q_word[abs_d_idx*DATA_W +: DATA_W]);
  acc_t score_prod;
  assign score_prod = acc_t'(q_lane) * acc_t'(k_cache[kj][abs_d_idx]);

  function automatic acc_t dot_local(
    input logic [HID_D*DATA_W-1:0] a_word,
    input int8_t b_arr [0:HID_D-1],
    input int base,
    input int width
  );
    acc_t sum;
    int i;
    int8_t av;
    begin
      sum = '0;
      for (i = 0; i < width; i = i + 1) begin
        av  = int8_t'(a_word[(base+i)*DATA_W +: DATA_W]);
        sum = sum + acc_t'(av) * acc_t'(b_arr[base+i]);
      end
      dot_local = sum;
    end
  endfunction

  acc_t mix_prod;
  assign mix_prod = acc_t'(weight_arr[h_idx][kj]) * acc_t'(v_cache[kj][abs_d_idx]);

  int8_t raw_score_val;
  assign raw_score_val = requantize(score_acc, QSHIFT);

  logic signed [8:0] diff9;
  assign diff9 = {max_val[DATA_W-1], max_val} - {raw_scores[h_idx][eidx][DATA_W-1], raw_scores[h_idx][eidx]};
  logic [7:0] diff_u;
  assign diff_u = diff9[7:0];

  logic exp_re;
  logic [7:0] exp_val_lut;
  exp_lut u_exp (.clk(clk), .re(exp_re), .diff(diff_u), .exp_val(exp_val_lut));

  logic        div_start, div_done;
  logic [15:0] div_numer, div_denom, div_quot;
  divider16 u_div (
    .clk(clk), .rst_n(rst_n), .start(div_start),
    .numer(div_numer), .denom(div_denom), .done(div_done), .quot(div_quot)
  );
  assign div_numer = {8'b0, exp_arr[h_idx][didx]} << 7;
  assign div_denom = sum_reg;

  always_comb begin
    state_n     = state;
    win_idx_n   = win_idx;
    kv_idx_n    = kv_idx;
    qi_n        = qi;
    kj_n        = kj;
    d_idx_n     = d_idx;
    midx_n      = midx;
    eidx_n      = eidx;
    didx_n      = didx;
    h_idx_n     = h_idx;
    q_word_n    = q_word;
    out_word_n  = out_word;
    score_acc_n = score_acc;
    out_acc_n   = out_acc;
    max_val_n   = max_val;
    sum_reg_n   = sum_reg;

    q_re = 1'b0; q_raddr = cur_tok;
    k_re = 1'b0; k_raddr = cur_kv_tok;
    v_re = 1'b0; v_raddr = cur_kv_tok;

    out_we    = 1'b0;
    out_waddr = cur_tok;
    out_wdata = out_word;

    exp_re    = 1'b0;
    div_start = 1'b0;

    done = 1'b0;

    unique case (state)
      S_IDLE: begin
        if (start) begin
          win_idx_n = '0;
          kv_idx_n  = '0;
          state_n   = S_KV_REQ;
        end
      end

      S_KV_REQ: begin
        k_re    = 1'b1; k_raddr = cur_kv_tok;
        v_re    = 1'b1; v_raddr = cur_kv_tok;
        state_n = S_KV_WAIT;
      end
      S_KV_WAIT: begin
        if (kv_idx != WI_AW'(WIN_W-1)) begin
          kv_idx_n = kv_idx + 1'b1;
          state_n  = S_KV_REQ;
        end else begin
          qi_n    = '0;
          state_n = S_Q_REQ;
        end
      end

      S_Q_REQ: begin
        q_re    = 1'b1; q_raddr = cur_tok;
        state_n = S_Q_WAIT;
      end
      S_Q_WAIT: begin
        q_word_n = q_rdata;
        h_idx_n  = '0;
        kj_n     = '0;
        state_n  = S_SCORE_REQ_D;
      end

      S_SCORE_REQ_D: begin
        score_acc_n = dot_local(q_word, k_cache[kj], h_idx * HD, HD);
        state_n     = S_SCORE_STORE;
      end
      S_SCORE_STORE: begin
        if (kj != WI_AW'(WIN_W-1)) begin
          kj_n    = kj + 1'b1;
          state_n = S_SCORE_REQ_D;
        end else begin
          state_n = S_SCORE_NEXT_H;
        end
      end
      S_SCORE_NEXT_H: begin
        if (h_idx != H_AW'(N_HEADS-1)) begin
          h_idx_n = h_idx + 1'b1;
          kj_n    = '0;
          state_n = S_SCORE_REQ_D;
        end else begin
          h_idx_n = '0;
          state_n = S_SM_H_INIT;
        end
      end

      S_SM_H_INIT: begin
        midx_n  = '0;
        state_n = S_SM_MAX_INIT;
      end

      S_SM_MAX_INIT: begin
        max_val_n = raw_scores[h_idx][0];
        midx_n    = 1;
        state_n   = (WIN_W == 1) ? S_SM_EXP_REQ : S_SM_MAX_SCAN;
        if (WIN_W == 1) eidx_n = '0;
      end
      S_SM_MAX_SCAN: begin
        if (raw_scores[h_idx][midx] > max_val) max_val_n = raw_scores[h_idx][midx];
        if (midx != WI_AW'(WIN_W-1)) begin
          midx_n = midx + 1'b1;
        end else begin
          eidx_n    = '0;
          sum_reg_n = '0;
          state_n   = S_SM_EXP_REQ;
        end
      end

      S_SM_EXP_REQ: begin
        exp_re  = 1'b1;
        state_n = S_SM_EXP_WAIT;
      end
      S_SM_EXP_WAIT: begin
        sum_reg_n = sum_reg + {8'b0, exp_val_lut};
        if (eidx != WI_AW'(WIN_W-1)) begin
          eidx_n  = eidx + 1'b1;
          state_n = S_SM_EXP_REQ;
        end else begin
          didx_n  = '0;
          state_n = S_SM_DIV_START;
        end
      end

      S_SM_DIV_START: begin
        div_start = 1'b1;
        state_n   = S_SM_DIV_WAIT;
      end
      S_SM_DIV_WAIT: begin
        if (div_done) begin
          if (didx != WI_AW'(WIN_W-1)) begin
            didx_n  = didx + 1'b1;
            state_n = S_SM_DIV_START;
          end else begin
            state_n = S_SM_NEXT_H;
          end
        end
      end
      S_SM_NEXT_H: begin
        if (h_idx != H_AW'(N_HEADS-1)) begin
          h_idx_n = h_idx + 1'b1;
          state_n = S_SM_H_INIT;
        end else begin
          h_idx_n = '0;
          state_n = S_OUT_H_INIT;
        end
      end

      S_OUT_H_INIT: begin
        d_idx_n = '0;
        kj_n    = '0;
        state_n = S_OUT_ACC_KJ;
      end

      S_OUT_ACC_KJ: begin
        out_acc_n = (kj == '0) ? mix_prod : (out_acc + mix_prod);
        if (kj != WI_AW'(WIN_W-1)) begin
          kj_n = kj + 1'b1;
        end else begin
          state_n = S_OUT_STORE_D;
        end
      end
      S_OUT_STORE_D: begin
        out_word_n[abs_d_idx*DATA_W +: DATA_W] = requantize(out_acc_n, QSHIFT);
        if (d_idx != HD_AW'(HD-1)) begin
          d_idx_n = d_idx + 1'b1;
          kj_n    = '0;
          state_n = S_OUT_ACC_KJ;
        end else if (h_idx != H_AW'(N_HEADS-1)) begin
          h_idx_n = h_idx + 1'b1;
          d_idx_n = '0;
          kj_n    = '0;
          state_n = S_OUT_ACC_KJ;
        end else begin
          state_n = S_WRITE_TOK;
        end
      end

      S_WRITE_TOK: begin
        out_we    = 1'b1;
        out_waddr = cur_tok;
        out_wdata = out_word_n;
        state_n   = S_NEXT_Q;
      end

      S_NEXT_Q: begin
        if (qi != WI_AW'(WIN_W-1)) begin
          qi_n    = qi + 1'b1;
          state_n = S_Q_REQ;
        end else begin
          state_n = S_NEXT_WIN;
        end
      end

      S_NEXT_WIN: begin
        if (win_idx != WIN_AW'(NWIN-1)) begin
          win_idx_n = win_idx + 1'b1;
          kv_idx_n  = '0;
          state_n   = S_KV_REQ;
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
      state     <= S_IDLE;
      win_idx   <= '0;
      kv_idx    <= '0;
      qi        <= '0;
      kj        <= '0;
      d_idx     <= '0;
      midx      <= '0;
      eidx      <= '0;
      didx      <= '0;
      h_idx     <= '0;
      q_word    <= '0;
      out_word  <= '0;
      score_acc <= '0;
      out_acc   <= '0;
      max_val   <= '0;
      sum_reg   <= '0;
    end else begin
      state     <= state_n;
      win_idx   <= win_idx_n;
      kv_idx    <= kv_idx_n;
      qi        <= qi_n;
      kj        <= kj_n;
      d_idx     <= d_idx_n;
      midx      <= midx_n;
      eidx      <= eidx_n;
      didx      <= didx_n;
      h_idx     <= h_idx_n;
      q_word    <= q_word_n;
      out_word  <= out_word_n;
      score_acc <= score_acc_n;
      out_acc   <= out_acc_n;
      max_val   <= max_val_n;
      sum_reg   <= sum_reg_n;
    end
  end

  integer ci, cj;
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      for (ci = 0; ci < WIN_W; ci++) begin
        for (cj = 0; cj < HID_D; cj++) begin
          k_cache[ci][cj] <= '0;
          v_cache[ci][cj] <= '0;
        end
      end
    end else begin
      if (state == S_KV_WAIT) begin
        for (cj = 0; cj < HID_D; cj++) begin
          k_cache[kv_idx][cj] <= int8_t'(k_rdata[cj*DATA_W +: DATA_W]);
          v_cache[kv_idx][cj] <= int8_t'(v_rdata[cj*DATA_W +: DATA_W]);
        end
      end
      if (state == S_SCORE_STORE) begin
        raw_scores[h_idx][kj] <= raw_score_val;
      end
      if (state == S_SM_EXP_WAIT) begin
        exp_arr[h_idx][eidx] <= int8_t'(exp_val_lut);
      end
      if (state == S_SM_DIV_WAIT && div_done) begin
        weight_arr[h_idx][didx] <= (div_quot > 16'd127) ? 8'sd127 : int8_t'(div_quot[7:0]);
      end
    end
  end

endmodule : spatial_attention_tile
