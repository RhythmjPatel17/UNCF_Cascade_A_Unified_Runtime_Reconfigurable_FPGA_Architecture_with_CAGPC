import trident_pkg::*;

module linear_engine #(
  parameter int IN_DIM   = HID_D,
  parameter int OUT_DIM  = HID_D,
  parameter int W_GBASE  = 0,
  parameter int NTOK     = N_TOK
)(
  input  logic                          clk,
  input  logic                          rst_n,

  input  logic                          start,
  output logic                          done,

  output logic                          in_re,
  output logic [$clog2(NTOK)-1:0]       in_raddr,
  input  logic [IN_DIM*DATA_W-1:0]      in_rdata,

  output logic                          wc_re,
  output logic [WCACHE_GAW-1:0]         wc_raddr,
  input  logic [PE_PAR*DATA_W-1:0]      wc_rdata,

  output logic                          out_we,
  output logic [$clog2(NTOK)-1:0]       out_waddr,
  output logic [OUT_DIM*DATA_W-1:0]     out_wdata
);

  localparam int TAW    = $clog2(NTOK  <= 1 ? 2 : NTOK);
  localparam int IAW    = $clog2(IN_DIM  <= 1 ? 2 : IN_DIM);
  localparam int OGRP   = (OUT_DIM + PE_PAR - 1) / PE_PAR;
  localparam int OGAW   = $clog2(OGRP <= 1 ? 2 : OGRP);

  typedef enum logic [3:0] {
    S_IDLE, S_LOAD_REQ, S_LOAD_WAIT, S_GRP_INIT, S_W_REQ, S_W_WAIT, S_WRITE_GRP, S_NEXT_TOK, S_DONE
  } state_e;
  state_e state, state_n;

  logic [TAW-1:0]  tok_idx, tok_idx_n;
  logic [OGAW-1:0] out_grp, out_grp_n;
  logic [IAW-1:0]  in_ch,   in_ch_n;

  logic [IN_DIM*DATA_W-1:0]  in_word, in_word_n;
  logic [OUT_DIM*DATA_W-1:0] out_word, out_word_n;

  int8_t cur_in_val;
  assign cur_in_val = int8_t'(in_word[in_ch*DATA_W +: DATA_W]);

  logic clear_lanes, en_lanes;
  acc_t lane_acc [0:PE_PAR-1];

  genvar gi;
  generate
    for (gi = 0; gi < PE_PAR; gi++) begin : g_lane
      mac_pe u_pe (
        .clk   (clk),
        .rst_n (rst_n),
        .clear (clear_lanes),
        .en    (en_lanes),
        .a     (cur_in_val),
        .b     (int8_t'(wc_rdata[gi*DATA_W +: DATA_W])),
        .acc   (lane_acc[gi])
      );
    end
  endgenerate

  always_comb begin
    state_n    = state;
    tok_idx_n  = tok_idx;
    out_grp_n  = out_grp;
    in_ch_n    = in_ch;
    in_word_n  = in_word;
    out_word_n = out_word;

    in_re     = 1'b0;
    in_raddr  = tok_idx;

    wc_re     = 1'b0;
    wc_raddr  = WCACHE_GAW'(W_GBASE + in_ch * OGRP + out_grp);

    out_we    = 1'b0;
    out_waddr = tok_idx;
    out_wdata = out_word;

    clear_lanes = 1'b0;
    en_lanes    = 1'b0;

    done = 1'b0;

    unique case (state)
      S_IDLE: begin
        if (start) begin
          tok_idx_n = '0;
          state_n   = S_LOAD_REQ;
        end
      end

      S_LOAD_REQ: begin
        in_re    = 1'b1;
        in_raddr = tok_idx;
        state_n  = S_LOAD_WAIT;
      end

      S_LOAD_WAIT: begin
        in_word_n = in_rdata;
        out_grp_n = '0;
        state_n   = S_GRP_INIT;
      end

      S_GRP_INIT: begin
        clear_lanes = 1'b1;
        in_ch_n     = '0;
        state_n     = S_W_REQ;
      end

      S_W_REQ: begin
        wc_re    = 1'b1;
        wc_raddr = WCACHE_GAW'(W_GBASE + in_ch * OGRP + out_grp);
        state_n  = S_W_WAIT;
      end

      S_W_WAIT: begin
        en_lanes = 1'b1;
        if (in_ch != IAW'(IN_DIM-1)) begin
          in_ch_n = in_ch + 1'b1;
          state_n = S_W_REQ;
        end else begin
          state_n = S_WRITE_GRP;
        end
      end

      S_WRITE_GRP: begin
        for (int li = 0; li < PE_PAR; li++) begin
          if ((out_grp * PE_PAR + li) < OUT_DIM) begin
            out_word_n[(out_grp*PE_PAR + li)*DATA_W +: DATA_W] = requantize(lane_acc[li], QSHIFT);
          end
        end
        if (out_grp != OGAW'(OGRP-1)) begin
          out_grp_n = out_grp + 1'b1;
          state_n   = S_GRP_INIT;
        end else begin
          out_we    = 1'b1;
          out_waddr = tok_idx;
          out_wdata = out_word_n;
          state_n   = S_NEXT_TOK;
        end
      end

      S_NEXT_TOK: begin
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
      out_grp  <= '0;
      in_ch    <= '0;
      in_word  <= '0;
      out_word <= '0;
    end else begin
      state    <= state_n;
      tok_idx  <= tok_idx_n;
      out_grp  <= out_grp_n;
      in_ch    <= in_ch_n;
      in_word  <= in_word_n;
      out_word <= out_word_n;
    end
  end

endmodule : linear_engine
