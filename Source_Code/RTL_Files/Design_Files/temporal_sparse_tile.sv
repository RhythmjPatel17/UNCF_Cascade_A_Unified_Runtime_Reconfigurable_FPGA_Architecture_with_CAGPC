import trident_pkg::*;

module temporal_sparse_tile (
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

  output logic                    attn_re,
  output logic [TOK_AW-1:0]       attn_raddr,
  input  logic [HID_D*DATA_W-1:0] attn_rdata,
  output logic                    attn_we,
  output logic [TOK_AW-1:0]       attn_waddr,
  output logic [HID_D*DATA_W-1:0] attn_wdata,

  input  logic                    reuse_mode,

  output logic                              cache_we,
  output logic [TOK_AW-1:0]                 cache_w_tok,
  output logic [$clog2(N_HEADS<=1?2:N_HEADS)-1:0] cache_w_head,
  output logic [$clog2(TOPK_K<=1?2:TOPK_K)-1:0]   cache_w_iter,
  output logic [TOK_AW-1:0]                 cache_w_idx,

  output logic                              cache_re,
  output logic [TOK_AW-1:0]                 cache_r_tok,
  output logic [$clog2(N_HEADS<=1?2:N_HEADS)-1:0] cache_r_head,
  output logic [$clog2(TOPK_K<=1?2:TOPK_K)-1:0]   cache_r_iter,
  input  logic [TOK_AW-1:0]                 cache_r_idx
);

  localparam int D_AW   = $clog2(HID_D  <= 1 ? 2 : HID_D);
  localparam int K_AW   = $clog2(TOPK_K <= 1 ? 2 : TOPK_K);
  localparam int HD     = HID_D / N_HEADS;
  localparam int HD_AW  = $clog2(HD <= 1 ? 2 : HD);
  localparam int H_AW   = $clog2(N_HEADS <= 1 ? 2 : N_HEADS);

  typedef enum logic [5:0] {
    S_IDLE,
    S_Q_REQ, S_Q_WAIT, S_SCORE_CLEAR,
    S_KEY_REQ, S_KEY_WAIT,
    S_SCORE_ALL, S_SCORE_NEXT_KI,
    S_CACHE_WRITE_H_INIT, S_CACHE_WRITE_ITER, S_CACHE_WRITE_NEXT_H,
    S_RESCORE_H_INIT, S_RESCORE_ITER_INIT, S_RESCORE_CACHE_WAIT,
    S_RESCORE_KEY_REQ, S_RESCORE_KEY_WAIT, S_RESCORE_D, S_RESCORE_STORE,
    S_RESCORE_NEXT_ITER, S_RESCORE_NEXT_H,
    S_SM_H_INIT,
    S_SM_MAX_INIT, S_SM_MAX_SCAN,
    S_SM_EXP_REQ, S_SM_EXP_WAIT,
    S_SM_DIV_START, S_SM_DIV_WAIT, S_SM_NEXT_H,
    S_MIX_H_INIT,
    S_GATHER_REQ, S_GATHER_WAIT,
    S_MIX_D,
    S_COMBINE_REQ, S_COMBINE_WAIT, S_COMBINE_WRITE,
    S_NEXT_Q,
    S_DONE
  } state_e;
  state_e state, state_n;

  logic [TOK_AW-1:0] qi,      qi_n;
  logic [TOK_AW-1:0] ki,      ki_n;
  logic [HD_AW-1:0]  d_idx,   d_idx_n;
  logic [K_AW-1:0]   iter,    iter_n;
  logic [K_AW-1:0]   smax_i,  smax_i_n;
  logic [K_AW-1:0]   sexp_i,  sexp_i_n;
  logic [K_AW-1:0]   sdiv_i,  sdiv_i_n;
  logic [H_AW-1:0]   h_idx,   h_idx_n;

  logic [HID_D*DATA_W-1:0] q_word, q_word_n;
  logic [HID_D*DATA_W-1:0] out_word, out_word_n;

  acc_t score_acc, score_acc_n;
  acc_t mix_acc,   mix_acc_n;

  int8_t             sel_val    [0:N_HEADS-1][0:TOPK_K-1];
  logic [TOK_AW-1:0] sel_idx    [0:N_HEADS-1][0:TOPK_K-1];
  int8_t             exp_arr    [0:N_HEADS-1][0:TOPK_K-1];
  int8_t             weight_arr [0:N_HEADS-1][0:TOPK_K-1];

  logic clear_sel;

  genvar gh;
  generate
    for (gh = 0; gh < N_HEADS; gh = gh + 1) begin : g_reset_head
      integer gi;
      always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
          for (gi = 0; gi < TOPK_K; gi = gi + 1) begin
            sel_val[gh][gi]    <= -8'sd128;
            sel_idx[gh][gi]    <= '0;
            exp_arr[gh][gi]    <= '0;
            weight_arr[gh][gi] <= '0;
          end
        end else if (clear_sel) begin
          for (gi = 0; gi < TOPK_K; gi = gi + 1) begin
            sel_val[gh][gi] <= -8'sd128;
            sel_idx[gh][gi] <= '0;
          end
        end
      end
    end
  endgenerate

  int8_t smax_val, smax_val_n;
  logic [15:0] sum_reg, sum_reg_n;

  logic [D_AW-1:0] abs_d_idx;
  assign abs_d_idx = D_AW'(h_idx * HD + d_idx);

  int8_t raw_score;
  assign raw_score = requantize(score_acc, QSHIFT);

  acc_t mix_prod;
  assign mix_prod = acc_t'(weight_arr[h_idx][iter]) * acc_t'(int8_t'(v_rdata[abs_d_idx*DATA_W +: DATA_W]));

  logic signed [8:0] diff9;
  assign diff9 = {smax_val[DATA_W-1], smax_val} - {sel_val[h_idx][sexp_i][DATA_W-1], sel_val[h_idx][sexp_i]};
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
  assign div_numer = {8'b0, exp_arr[h_idx][sdiv_i]} << 7;
  assign div_denom = sum_reg;

  int8_t             ins_val [0:TOPK_K-1];
  logic [TOK_AW-1:0] ins_idx [0:TOPK_K-1];

  int8_t             score_all [0:N_HEADS-1];
  int8_t             ins_val_all [0:N_HEADS-1][0:TOPK_K-1];
  logic [TOK_AW-1:0] ins_idx_all [0:N_HEADS-1][0:TOPK_K-1];

  task automatic topk_insert(
    input  int8_t             val_in [0:TOPK_K-1],
    input  logic [TOK_AW-1:0] idx_in [0:TOPK_K-1],
    input  int8_t             new_score,
    input  logic [TOK_AW-1:0] new_idx,
    output int8_t             val_out [0:TOPK_K-1],
    output logic [TOK_AW-1:0] idx_out [0:TOPK_K-1]
  );
    int p, i;
    logic found;
    begin
      val_out = val_in;
      idx_out = idx_in;
      if (new_score > val_in[TOPK_K-1]) begin
        found = 1'b0;
        p = 0;
        for (i = 0; i < TOPK_K; i = i + 1) begin
          if (!found && val_in[i] < new_score) begin
            p = i;
            found = 1'b1;
          end
        end
        for (i = TOPK_K-1; i > p; i = i - 1) begin
          val_out[i] = val_in[i-1];
          idx_out[i] = idx_in[i-1];
        end
        val_out[p] = new_score;
        idx_out[p] = new_idx;
      end
    end
  endtask

  integer si;
  integer hh;
  integer whh;

  always_comb begin
    state_n     = state;
    qi_n        = qi;
    ki_n        = ki;
    d_idx_n     = d_idx;
    iter_n      = iter;
    smax_i_n    = smax_i;
    sexp_i_n    = sexp_i;
    sdiv_i_n    = sdiv_i;
    h_idx_n     = h_idx;
    q_word_n    = q_word;
    out_word_n  = out_word;
    score_acc_n = score_acc;
    mix_acc_n   = mix_acc;
    smax_val_n  = smax_val;
    sum_reg_n   = sum_reg;

    q_re = 1'b0; q_raddr = qi;
    k_re = 1'b0; k_raddr = ki;
    v_re = 1'b0; v_raddr = sel_idx[h_idx][iter];

    attn_re    = 1'b0; attn_raddr = qi;
    attn_we    = 1'b0; attn_waddr = qi; attn_wdata = out_word;

    exp_re    = 1'b0;
    div_start = 1'b0;

    clear_sel = 1'b0;

    cache_we     = 1'b0;
    cache_w_tok  = qi;
    cache_w_head = h_idx;
    cache_w_iter = iter;
    cache_w_idx  = sel_idx[h_idx][iter];

    cache_re     = 1'b0;
    cache_r_tok  = qi;
    cache_r_head = h_idx;
    cache_r_iter = iter;

    ins_val = sel_val[h_idx];
    ins_idx = sel_idx[h_idx];

    for (hh = 0; hh < N_HEADS; hh = hh + 1) begin
      score_all[hh] = requantize(dot_slice(q_word, k_rdata, hh * HD, HD), QSHIFT);
      topk_insert(sel_val[hh], sel_idx[hh], score_all[hh], ki,
                  ins_val_all[hh], ins_idx_all[hh]);
    end

    done = 1'b0;

    unique case (state)
      S_IDLE: begin
        if (start) begin
          qi_n = '0;
          state_n = S_Q_REQ;
        end
      end

      S_Q_REQ: begin
        q_re = 1'b1; q_raddr = qi;
        state_n = S_Q_WAIT;
      end
      S_Q_WAIT: begin
        q_word_n = q_rdata;
        if (reuse_mode) begin
          h_idx_n = '0;
          iter_n  = '0;
          state_n = S_RESCORE_H_INIT;
        end else begin
          state_n = S_SCORE_CLEAR;
        end
      end
      S_SCORE_CLEAR: begin
        clear_sel = 1'b1;
        ki_n      = '0;
        state_n   = S_KEY_REQ;
      end

      S_KEY_REQ: begin
        k_re    = 1'b1; k_raddr = ki;
        state_n = S_KEY_WAIT;
      end
      S_KEY_WAIT: begin
        state_n = S_SCORE_ALL;
      end
      S_SCORE_ALL: begin
        state_n = S_SCORE_NEXT_KI;
      end
      S_SCORE_NEXT_KI: begin
        if (ki != TOK_AW'(N_TOK-1)) begin
          ki_n    = ki + 1'b1;
          state_n = S_KEY_REQ;
        end else begin
          h_idx_n = '0;
          iter_n  = '0;
          state_n = S_CACHE_WRITE_H_INIT;
        end
      end

      S_CACHE_WRITE_H_INIT: begin
        iter_n  = '0;
        state_n = S_CACHE_WRITE_ITER;
      end
      S_CACHE_WRITE_ITER: begin
        cache_we     = 1'b1;
        cache_w_tok  = qi;
        cache_w_head = h_idx;
        cache_w_iter = iter;
        cache_w_idx  = sel_idx[h_idx][iter];
        if (iter != K_AW'(TOPK_K-1)) begin
          iter_n  = iter + 1'b1;
          state_n = S_CACHE_WRITE_ITER;
        end else begin
          state_n = S_CACHE_WRITE_NEXT_H;
        end
      end
      S_CACHE_WRITE_NEXT_H: begin
        if (h_idx != H_AW'(N_HEADS-1)) begin
          h_idx_n = h_idx + 1'b1;
          iter_n  = '0;
          state_n = S_CACHE_WRITE_ITER;
        end else begin
          h_idx_n  = '0;
          smax_i_n = '0;
          state_n  = S_SM_MAX_INIT;
        end
      end

      S_RESCORE_H_INIT: begin
        iter_n  = '0;
        state_n = S_RESCORE_ITER_INIT;
      end
      S_RESCORE_ITER_INIT: begin
        cache_re     = 1'b1;
        cache_r_tok  = qi;
        cache_r_head = h_idx;
        cache_r_iter = iter;
        state_n      = S_RESCORE_CACHE_WAIT;
      end
      S_RESCORE_CACHE_WAIT: begin
        state_n = S_RESCORE_KEY_REQ;
      end
      S_RESCORE_KEY_REQ: begin
        k_re    = 1'b1; k_raddr = cache_r_idx;
        state_n = S_RESCORE_KEY_WAIT;
      end
      S_RESCORE_KEY_WAIT: begin
        ki_n    = cache_r_idx;
        state_n = S_RESCORE_D;
      end
      S_RESCORE_D: begin
        score_acc_n = dot_slice(q_word, k_rdata, h_idx * HD, HD);
        state_n     = S_RESCORE_STORE;
      end
      S_RESCORE_STORE: begin
        state_n = S_RESCORE_NEXT_ITER;
      end
      S_RESCORE_NEXT_ITER: begin
        if (iter != K_AW'(TOPK_K-1)) begin
          iter_n  = iter + 1'b1;
          state_n = S_RESCORE_ITER_INIT;
        end else begin
          state_n = S_RESCORE_NEXT_H;
        end
      end
      S_RESCORE_NEXT_H: begin
        if (h_idx != H_AW'(N_HEADS-1)) begin
          h_idx_n = h_idx + 1'b1;
          state_n = S_RESCORE_H_INIT;
        end else begin
          h_idx_n  = '0;
          smax_i_n = '0;
          state_n  = S_SM_MAX_INIT;
        end
      end

      S_SM_MAX_INIT: begin
        smax_val_n = sel_val[h_idx][0];
        smax_i_n   = 1;
        if (TOPK_K == 1) begin
          sexp_i_n  = '0;
          sum_reg_n = '0;
          state_n   = S_SM_EXP_REQ;
        end else begin
          state_n = S_SM_MAX_SCAN;
        end
      end
      S_SM_MAX_SCAN: begin
        if (sel_val[h_idx][smax_i] > smax_val) smax_val_n = sel_val[h_idx][smax_i];
        if (smax_i != K_AW'(TOPK_K-1)) begin
          smax_i_n = smax_i + 1'b1;
        end else begin
          sexp_i_n  = '0;
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
        if (sexp_i != K_AW'(TOPK_K-1)) begin
          sexp_i_n = sexp_i + 1'b1;
          state_n  = S_SM_EXP_REQ;
        end else begin
          sdiv_i_n = '0;
          state_n  = S_SM_DIV_START;
        end
      end

      S_SM_DIV_START: begin
        div_start = 1'b1;
        state_n   = S_SM_DIV_WAIT;
      end
      S_SM_DIV_WAIT: begin
        if (div_done) begin
          if (sdiv_i != K_AW'(TOPK_K-1)) begin
            sdiv_i_n = sdiv_i + 1'b1;
            state_n  = S_SM_DIV_START;
          end else begin
            state_n = S_SM_NEXT_H;
          end
        end
      end
      S_SM_NEXT_H: begin
        if (h_idx != H_AW'(N_HEADS-1)) begin
          h_idx_n  = h_idx + 1'b1;
          smax_i_n = '0;
          state_n  = S_SM_MAX_INIT;
        end else begin
          h_idx_n = '0;
          iter_n  = '0;
          d_idx_n = '0;
          state_n = S_GATHER_REQ;
        end
      end

      S_GATHER_REQ: begin
        v_re    = 1'b1; v_raddr = sel_idx[h_idx][iter];
        state_n = S_GATHER_WAIT;
      end
      S_GATHER_WAIT: begin
        state_n = S_MIX_D;
      end
      S_MIX_D: begin
        mix_acc_n = (iter == '0) ? mix_prod : (mix_acc + mix_prod);
        if (iter != K_AW'(TOPK_K-1)) begin
          iter_n  = iter + 1'b1;
          state_n = S_GATHER_REQ;
        end else begin
          out_word_n[abs_d_idx*DATA_W +: DATA_W] = requantize(mix_acc_n, QSHIFT);
          if (d_idx != HD_AW'(HD-1)) begin
            d_idx_n = d_idx + 1'b1;
            iter_n  = '0;
            state_n = S_GATHER_REQ;
          end else if (h_idx != H_AW'(N_HEADS-1)) begin
            h_idx_n = h_idx + 1'b1;
            d_idx_n = '0;
            iter_n  = '0;
            state_n = S_GATHER_REQ;
          end else begin
            state_n = S_COMBINE_REQ;
          end
        end
      end

      S_COMBINE_REQ: begin
        attn_re    = 1'b1; attn_raddr = qi;
        state_n    = S_COMBINE_WAIT;
      end
      S_COMBINE_WAIT: begin
        state_n = S_COMBINE_WRITE;
      end
      S_COMBINE_WRITE: begin
        for (si = 0; si < HID_D; si++) begin
          out_word_n[si*DATA_W +: DATA_W] =
            sat_add8(int8_t'(attn_rdata[si*DATA_W +: DATA_W]),
                      int8_t'(out_word[si*DATA_W +: DATA_W]));
        end
        attn_we    = 1'b1;
        attn_waddr = qi;
        attn_wdata = out_word_n;
        state_n    = S_NEXT_Q;
      end

      S_NEXT_Q: begin
        if (qi != TOK_AW'(N_TOK-1)) begin
          qi_n    = qi + 1'b1;
          state_n = S_Q_REQ;
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
      qi        <= '0;
      ki        <= '0;
      d_idx     <= '0;
      iter      <= '0;
      smax_i    <= '0;
      sexp_i    <= '0;
      sdiv_i    <= '0;
      h_idx     <= '0;
      q_word    <= '0;
      out_word  <= '0;
      score_acc <= '0;
      mix_acc   <= '0;
      smax_val  <= '0;
      sum_reg   <= '0;
    end else begin
      state     <= state_n;
      qi        <= qi_n;
      ki        <= ki_n;
      d_idx     <= d_idx_n;
      iter      <= iter_n;
      smax_i    <= smax_i_n;
      sexp_i    <= sexp_i_n;
      sdiv_i    <= sdiv_i_n;
      h_idx     <= h_idx_n;
      q_word    <= q_word_n;
      out_word  <= out_word_n;
      score_acc <= score_acc_n;
      mix_acc   <= mix_acc_n;
      smax_val  <= smax_val_n;
      sum_reg   <= sum_reg_n;
    end
  end

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
    end else begin
      if (state == S_SCORE_ALL) begin
        for (whh = 0; whh < N_HEADS; whh = whh + 1) begin
          sel_val[whh] <= ins_val_all[whh];
          sel_idx[whh] <= ins_idx_all[whh];
        end
      end
      if (state == S_RESCORE_STORE) begin
        sel_val[h_idx][iter] <= raw_score;
        sel_idx[h_idx][iter] <= ki;
      end
      if (state == S_SM_EXP_WAIT) begin
        exp_arr[h_idx][sexp_i] <= int8_t'(exp_val_lut);
      end
      if (state == S_SM_DIV_WAIT && div_done) begin
        weight_arr[h_idx][sdiv_i] <= (div_quot > 16'd127) ? 8'sd127 : int8_t'(div_quot[7:0]);
      end
    end
  end

endmodule : temporal_sparse_tile
