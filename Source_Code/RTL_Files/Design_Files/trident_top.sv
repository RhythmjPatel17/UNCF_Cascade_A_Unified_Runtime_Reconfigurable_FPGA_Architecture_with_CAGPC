import trident_pkg::*;

module trident_top (
  input  logic clk,
  input  logic rst_n,

  input  logic start,
  output logic busy,
  output logic done,

  input  decomp_mode_e decomp_mode,

  output logic                    ext_mem_req,
  output logic [31:0]             ext_mem_addr,
  input  logic                    ext_mem_valid,
  input  logic [COMP_WORD_W-1:0]  ext_mem_rdata,

  input  logic                    dbg_tok_we,
  input  logic [TOK_AW-1:0]       dbg_tok_waddr,
  input  logic [HID_D*DATA_W-1:0] dbg_tok_wdata,
  input  logic                    dbg_tok_re,
  input  logic [TOK_AW-1:0]       dbg_tok_raddr,
  output logic [HID_D*DATA_W-1:0] dbg_tok_rdata,

  input  logic                     dbg_pix_we,
  input  logic [TOK_AW-1:0]        dbg_pix_waddr,
  input  logic [PIX_DIM*DATA_W-1:0] dbg_pix_wdata,
  input  logic                     dbg_pix_re,
  input  logic [TOK_AW-1:0]        dbg_pix_raddr,
  output logic [PIX_DIM*DATA_W-1:0] dbg_pix_rdata,

  output logic [$clog2(DEPTH<=1?2:DEPTH)-1:0]     dbg_layer_idx,
  output logic [$clog2(T_STEPS<=1?2:T_STEPS)-1:0] dbg_step_idx
);

  logic dma_start, dma_done;
  logic recon_start, recon_done;
  logic patch_start, patch_done;
  logic ln0_start, ln0_done, oproj_start, oproj_done;
  logic outp_start, outp_done, dupd_start, dupd_done;
  logic qproj_start, qproj_done, kproj_start, kproj_done, vproj_start, vproj_done;
  logic spatial_start, spatial_done, temporal_start, temporal_done;
  logic resid1_start, resid1_done, ln1_start, ln1_done;
  logic ffnu1_start, ffnu1_done, ffnu2_start, ffnu2_done;
  logic gelu_start, gelu_done, ffnd1_start, ffnd1_done, ffnd2_start, ffnd2_done;
  logic resid2_start, resid2_done;
  logic [$clog2(DEPTH<=1?2:DEPTH)-1:0]    layer_idx;
  logic [$clog2(T_STEPS<=1?2:T_STEPS)-1:0] step_idx;

  global_scheduler u_sched (
    .clk (clk), .rst_n (rst_n),
    .start (start), .busy (busy), .done (done),
    .dma_start (dma_start), .dma_done (dma_done),
    .recon_start (recon_start), .recon_done (recon_done),
    .patch_start (patch_start), .patch_done (patch_done),
    .ln0_start (ln0_start), .ln0_done (ln0_done),
    .oproj_start (oproj_start), .oproj_done (oproj_done),
    .outp_start (outp_start), .outp_done (outp_done),
    .dupd_start (dupd_start), .dupd_done (dupd_done),
    .qproj_start (qproj_start), .qproj_done (qproj_done),
    .kproj_start (kproj_start), .kproj_done (kproj_done),
    .vproj_start (vproj_start), .vproj_done (vproj_done),
    .spatial_start (spatial_start), .spatial_done (spatial_done),
    .temporal_start (temporal_start), .temporal_done (temporal_done),
    .resid1_start (resid1_start), .resid1_done (resid1_done),
    .ln1_start (ln1_start), .ln1_done (ln1_done),
    .ffnu1_start (ffnu1_start), .ffnu1_done (ffnu1_done),
    .ffnu2_start (ffnu2_start), .ffnu2_done (ffnu2_done),
    .gelu_start (gelu_start), .gelu_done (gelu_done),
    .ffnd1_start (ffnd1_start), .ffnd1_done (ffnd1_done),
    .ffnd2_start (ffnd2_start), .ffnd2_done (ffnd2_done),
    .resid2_start (resid2_start), .resid2_done (resid2_done),
    .layer_idx (layer_idx), .step_idx (step_idx)
  );

  assign dbg_layer_idx = layer_idx;
  assign dbg_step_idx  = step_idx;

  logic                          crecon_req;
  logic [31:0]                   crecon_addr;
  logic                          crecon_valid;
  logic [COMP_WORD_W-1:0]        crecon_data;

  compressed_weight_store u_cstore (
    .clk (clk), .rst_n (rst_n),
    .dma_start (dma_start), .dma_done (dma_done),
    .dma_req (ext_mem_req), .dma_addr (ext_mem_addr),
    .dma_valid (ext_mem_valid), .dma_rdata (ext_mem_rdata),
    .rd_req (crecon_req), .rd_layer_idx (layer_idx), .rd_byte_addr (crecon_addr),
    .rd_valid (crecon_valid), .rd_data (crecon_data)
  );

  logic                 wc_we;
  logic [WCACHE_AW-1:0] wc_waddr;
  logic [DATA_W-1:0]    wc_wdata_recon;

  int8_reconstruction u_recon (
    .clk (clk), .rst_n (rst_n),
    .start (recon_start), .mode (decomp_mode), .done (recon_done),
    .mem_req (crecon_req), .mem_addr (crecon_addr),
    .mem_valid (crecon_valid), .mem_rdata (crecon_data),
    .wc_we (wc_we), .wc_waddr (wc_waddr), .wc_wdata (wc_wdata_recon)
  );

  logic                  wc_re_patch, wc_re_q, wc_re_k, wc_re_v, wc_re_o, wc_re_u1, wc_re_u2, wc_re_d1, wc_re_d2, wc_re_out;
  logic [WCACHE_GAW-1:0] wc_raddr_patch, wc_raddr_q, wc_raddr_k, wc_raddr_v, wc_raddr_o, wc_raddr_u1, wc_raddr_u2, wc_raddr_d1, wc_raddr_d2, wc_raddr_out;
  logic                  wc_re_mux;
  logic [WCACHE_GAW-1:0] wc_raddr_mux;
  logic [PE_PAR*DATA_W-1:0] wc_rdata;

  always_comb begin
    if      (wc_re_patch) begin wc_re_mux = 1'b1; wc_raddr_mux = wc_raddr_patch; end
    else if (wc_re_q)  begin wc_re_mux = 1'b1; wc_raddr_mux = wc_raddr_q;  end
    else if (wc_re_k)  begin wc_re_mux = 1'b1; wc_raddr_mux = wc_raddr_k;  end
    else if (wc_re_v)  begin wc_re_mux = 1'b1; wc_raddr_mux = wc_raddr_v;  end
    else if (wc_re_o)  begin wc_re_mux = 1'b1; wc_raddr_mux = wc_raddr_o;  end
    else if (wc_re_out) begin wc_re_mux = 1'b1; wc_raddr_mux = wc_raddr_out; end
    else if (wc_re_u1) begin wc_re_mux = 1'b1; wc_raddr_mux = wc_raddr_u1; end
    else if (wc_re_u2) begin wc_re_mux = 1'b1; wc_raddr_mux = wc_raddr_u2; end
    else if (wc_re_d1) begin wc_re_mux = 1'b1; wc_raddr_mux = wc_raddr_d1; end
    else if (wc_re_d2) begin wc_re_mux = 1'b1; wc_raddr_mux = wc_raddr_d2; end
    else               begin wc_re_mux = 1'b0; wc_raddr_mux = '0;          end
  end

  weight_cache u_wcache (
    .clk (clk),
    .we (wc_we), .waddr (wc_waddr), .wdata (wc_wdata_recon),
    .re (wc_re_mux), .raddr (wc_raddr_mux), .rdata (wc_rdata)
  );

  logic                     pix_re;
  logic [TOK_AW-1:0]        pix_raddr;
  logic [PIX_DIM*DATA_W-1:0] pix_rdata;

  logic                     du_a_re;
  logic [TOK_AW-1:0]        du_a_raddr;
  logic                     du_out_we;
  logic [TOK_AW-1:0]        du_out_waddr;
  logic [PIX_DIM*DATA_W-1:0] du_out_wdata;

  logic                      pix_we_mux;
  logic [TOK_AW-1:0]         pix_waddr_mux;
  logic [PIX_DIM*DATA_W-1:0] pix_wdata_mux;
  always_comb begin
    if (dbg_pix_we) begin
      pix_we_mux = 1'b1; pix_waddr_mux = dbg_pix_waddr; pix_wdata_mux = dbg_pix_wdata;
    end else begin
      pix_we_mux = du_out_we; pix_waddr_mux = du_out_waddr; pix_wdata_mux = du_out_wdata;
    end
  end

  logic              pix_re_mux;
  logic [TOK_AW-1:0] pix_raddr_mux;
  always_comb begin
    if (dbg_pix_re)    begin pix_re_mux = 1'b1; pix_raddr_mux = dbg_pix_raddr; end
    else if (pix_re)   begin pix_re_mux = 1'b1; pix_raddr_mux = pix_raddr;     end
    else               begin pix_re_mux = du_a_re; pix_raddr_mux = du_a_raddr; end
  end
  assign dbg_pix_rdata = pix_rdata;

  token_buffer #(.LANES(PIX_DIM)) u_pix_buf (
    .clk (clk),
    .we (pix_we_mux), .waddr (pix_waddr_mux), .wdata (pix_wdata_mux),
    .re (pix_re_mux), .raddr (pix_raddr_mux), .rdata (pix_rdata)
  );

  logic                    tokbuf_we;
  logic [TOK_AW-1:0]       tokbuf_waddr;
  logic [HID_D*DATA_W-1:0] tokbuf_wdata;
  logic                    tokbuf_re_ln0, tokbuf_re_r1, tokbuf_re_outp;
  logic [TOK_AW-1:0]       tokbuf_raddr_ln0, tokbuf_raddr_r1, tokbuf_raddr_outp;
  logic                    tokbuf_re_mux;
  logic [TOK_AW-1:0]       tokbuf_raddr_mux;
  logic [HID_D*DATA_W-1:0] tokbuf_rdata;

  always_comb begin
    if      (tokbuf_re_ln0)  begin tokbuf_re_mux = 1'b1; tokbuf_raddr_mux = tokbuf_raddr_ln0;  end
    else if (tokbuf_re_r1)   begin tokbuf_re_mux = 1'b1; tokbuf_raddr_mux = tokbuf_raddr_r1;   end
    else if (tokbuf_re_outp) begin tokbuf_re_mux = 1'b1; tokbuf_raddr_mux = tokbuf_raddr_outp; end
    else                    begin tokbuf_re_mux = 1'b0; tokbuf_raddr_mux = '0;               end
  end

  logic                    patch_out_we;
  logic [TOK_AW-1:0]       patch_out_waddr;
  logic [HID_D*DATA_W-1:0] patch_out_wdata;

  logic                    tokbuf_we_mux;
  logic [TOK_AW-1:0]       tokbuf_waddr_mux;
  logic [HID_D*DATA_W-1:0] tokbuf_wdata_mux;
  always_comb begin
    if (dbg_tok_we) begin
      tokbuf_we_mux    = 1'b1;
      tokbuf_waddr_mux = dbg_tok_waddr;
      tokbuf_wdata_mux = dbg_tok_wdata;
    end else if (patch_out_we) begin
      tokbuf_we_mux    = 1'b1;
      tokbuf_waddr_mux = patch_out_waddr;
      tokbuf_wdata_mux = patch_out_wdata;
    end else begin
      tokbuf_we_mux    = tokbuf_we;
      tokbuf_waddr_mux = tokbuf_waddr;
      tokbuf_wdata_mux = tokbuf_wdata;
    end
  end

  assign dbg_tok_rdata = tokbuf_rdata;

  token_buffer #(.LANES(HID_D)) u_tok_buf (
    .clk (clk),
    .we (tokbuf_we_mux), .waddr (tokbuf_waddr_mux), .wdata (tokbuf_wdata_mux),
    .re (tokbuf_re_mux | dbg_tok_re), .raddr (dbg_tok_re ? dbg_tok_raddr : tokbuf_raddr_mux),
    .rdata (tokbuf_rdata)
  );

  logic ln0_out_we; logic [TOK_AW-1:0] ln0_out_waddr; logic [HID_D*DATA_W-1:0] ln0_out_wdata;
  logic ln0_in_re;  logic [TOK_AW-1:0] ln0_in_raddr;
  logic tokbuf_re_q, tokbuf_re_k, tokbuf_re_v;
  logic [TOK_AW-1:0] tokbuf_raddr_q, tokbuf_raddr_k, tokbuf_raddr_v;
  logic ln0buf_re_mux; logic [TOK_AW-1:0] ln0buf_raddr_mux;
  logic [HID_D*DATA_W-1:0] ln0_rdata;
  always_comb begin
    if      (tokbuf_re_q) begin ln0buf_re_mux = 1'b1; ln0buf_raddr_mux = tokbuf_raddr_q; end
    else if (tokbuf_re_k) begin ln0buf_re_mux = 1'b1; ln0buf_raddr_mux = tokbuf_raddr_k; end
    else                  begin ln0buf_re_mux = tokbuf_re_v; ln0buf_raddr_mux = tokbuf_raddr_v; end
  end
  token_buffer #(.LANES(HID_D)) u_ln0_buf (
    .clk(clk), .we(ln0_out_we), .waddr(ln0_out_waddr), .wdata(ln0_out_wdata),
    .re(ln0buf_re_mux), .raddr(ln0buf_raddr_mux), .rdata(ln0_rdata)
  );

  logic q_we; logic [TOK_AW-1:0] q_waddr; logic [HID_D*DATA_W-1:0] q_wdata;
  logic k_we; logic [TOK_AW-1:0] k_waddr; logic [HID_D*DATA_W-1:0] k_wdata;
  logic v_we; logic [TOK_AW-1:0] v_waddr; logic [HID_D*DATA_W-1:0] v_wdata;

  logic q_re_sp, q_re_tp; logic [TOK_AW-1:0] q_raddr_sp, q_raddr_tp;
  logic k_re_sp, k_re_tp; logic [TOK_AW-1:0] k_raddr_sp, k_raddr_tp;
  logic v_re_sp, v_re_tp; logic [TOK_AW-1:0] v_raddr_sp, v_raddr_tp;

  logic q_re_mux, k_re_mux, v_re_mux;
  logic [TOK_AW-1:0] q_raddr_mux, k_raddr_mux, v_raddr_mux;
  logic [HID_D*DATA_W-1:0] q_rdata, k_rdata, v_rdata;

  always_comb begin
    if (q_re_sp) begin q_re_mux = 1'b1; q_raddr_mux = q_raddr_sp; end
    else         begin q_re_mux = q_re_tp; q_raddr_mux = q_raddr_tp; end
  end
  always_comb begin
    if (k_re_sp) begin k_re_mux = 1'b1; k_raddr_mux = k_raddr_sp; end
    else         begin k_re_mux = k_re_tp; k_raddr_mux = k_raddr_tp; end
  end
  always_comb begin
    if (v_re_sp) begin v_re_mux = 1'b1; v_raddr_mux = v_raddr_sp; end
    else         begin v_re_mux = v_re_tp; v_raddr_mux = v_raddr_tp; end
  end

  token_buffer #(.LANES(HID_D)) u_q_buf (
    .clk(clk), .we(q_we), .waddr(q_waddr), .wdata(q_wdata),
    .re(q_re_mux), .raddr(q_raddr_mux), .rdata(q_rdata)
  );
  token_buffer #(.LANES(HID_D)) u_k_buf (
    .clk(clk), .we(k_we), .waddr(k_waddr), .wdata(k_wdata),
    .re(k_re_mux), .raddr(k_raddr_mux), .rdata(k_rdata)
  );
  token_buffer #(.LANES(HID_D)) u_v_buf (
    .clk(clk), .we(v_we), .waddr(v_waddr), .wdata(v_wdata),
    .re(v_re_mux), .raddr(v_raddr_mux), .rdata(v_rdata)
  );

  logic sp_out_we; logic [TOK_AW-1:0] sp_out_waddr; logic [HID_D*DATA_W-1:0] sp_out_wdata;
  logic tp_attn_we; logic [TOK_AW-1:0] tp_attn_waddr; logic [HID_D*DATA_W-1:0] tp_attn_wdata;
  logic tp_attn_re; logic [TOK_AW-1:0] tp_attn_raddr;
  logic r1_b_re;    logic [TOK_AW-1:0] r1_b_raddr;
  logic op_in_re;   logic [TOK_AW-1:0] op_in_raddr;

  logic                    attn_we_mux;
  logic [TOK_AW-1:0]       attn_waddr_mux;
  logic [HID_D*DATA_W-1:0] attn_wdata_mux;
  always_comb begin
    if (sp_out_we) begin
      attn_we_mux = 1'b1; attn_waddr_mux = sp_out_waddr; attn_wdata_mux = sp_out_wdata;
    end else begin
      attn_we_mux = tp_attn_we; attn_waddr_mux = tp_attn_waddr; attn_wdata_mux = tp_attn_wdata;
    end
  end
  logic              attn_re_mux;
  logic [TOK_AW-1:0] attn_raddr_mux;
  always_comb begin
    if (tp_attn_re) begin attn_re_mux = 1'b1; attn_raddr_mux = tp_attn_raddr; end
    else            begin attn_re_mux = op_in_re; attn_raddr_mux = op_in_raddr; end
  end
  logic [HID_D*DATA_W-1:0] attn_rdata;

  token_buffer #(.LANES(HID_D)) u_attn_buf (
    .clk(clk), .we(attn_we_mux), .waddr(attn_waddr_mux), .wdata(attn_wdata_mux),
    .re(attn_re_mux), .raddr(attn_raddr_mux), .rdata(attn_rdata)
  );

  logic op_out_we; logic [TOK_AW-1:0] op_out_waddr; logic [HID_D*DATA_W-1:0] op_out_wdata;
  logic [HID_D*DATA_W-1:0] o_rdata;
  token_buffer #(.LANES(HID_D)) u_o_buf (
    .clk(clk), .we(op_out_we), .waddr(op_out_waddr), .wdata(op_out_wdata),
    .re(r1_b_re), .raddr(r1_b_raddr), .rdata(o_rdata)
  );

  logic r1_we; logic [TOK_AW-1:0] r1_waddr; logic [HID_D*DATA_W-1:0] r1_wdata;
  logic ln1_in_re; logic [TOK_AW-1:0] ln1_in_raddr;
  logic r2_a_re;   logic [TOK_AW-1:0] r2_a_raddr;
  logic r1_re_mux; logic [TOK_AW-1:0] r1_raddr_mux;
  always_comb begin
    if (ln1_in_re) begin r1_re_mux = 1'b1; r1_raddr_mux = ln1_in_raddr; end
    else           begin r1_re_mux = r2_a_re; r1_raddr_mux = r2_a_raddr; end
  end
  logic [HID_D*DATA_W-1:0] r1_rdata;
  token_buffer #(.LANES(HID_D)) u_resid1_buf (
    .clk(clk), .we(r1_we), .waddr(r1_waddr), .wdata(r1_wdata),
    .re(r1_re_mux), .raddr(r1_raddr_mux), .rdata(r1_rdata)
  );

  logic ln1_out_we; logic [TOK_AW-1:0] ln1_out_waddr; logic [HID_D*DATA_W-1:0] ln1_out_wdata;
  logic u1_in_re;   logic [TOK_AW-1:0] u1_in_raddr;
  logic [HID_D*DATA_W-1:0] ln1_rdata;
  token_buffer #(.LANES(HID_D)) u_ln1_buf (
    .clk(clk), .we(ln1_out_we), .waddr(ln1_out_waddr), .wdata(ln1_out_wdata),
    .re(u1_in_re), .raddr(u1_in_raddr), .rdata(ln1_rdata)
  );

  logic u1_out_we; logic [TOK_AW-1:0] u1_out_waddr; logic [RANK_R*DATA_W-1:0] u1_out_wdata;
  logic u2_in_re;  logic [TOK_AW-1:0] u2_in_raddr;
  logic [RANK_R*DATA_W-1:0] xu_rdata;
  token_buffer #(.LANES(RANK_R)) u_xu_buf (
    .clk(clk), .we(u1_out_we), .waddr(u1_out_waddr), .wdata(u1_out_wdata),
    .re(u2_in_re), .raddr(u2_in_raddr), .rdata(xu_rdata)
  );

  logic u2_out_we; logic [TOK_AW-1:0] u2_out_waddr; logic [FFN_HID*DATA_W-1:0] u2_out_wdata;
  logic gelu_in_re; logic [TOK_AW-1:0] gelu_in_raddr;
  logic [FFN_HID*DATA_W-1:0] h_rdata;
  token_buffer #(.LANES(FFN_HID)) u_h_buf (
    .clk(clk), .we(u2_out_we), .waddr(u2_out_waddr), .wdata(u2_out_wdata),
    .re(gelu_in_re), .raddr(gelu_in_raddr), .rdata(h_rdata)
  );

  logic gelu_out_we; logic [TOK_AW-1:0] gelu_out_waddr; logic [FFN_HID*DATA_W-1:0] gelu_out_wdata;
  logic d1_in_re;    logic [TOK_AW-1:0] d1_in_raddr;
  logic [FFN_HID*DATA_W-1:0] gelu_rdata;
  token_buffer #(.LANES(FFN_HID)) u_gelu_buf (
    .clk(clk), .we(gelu_out_we), .waddr(gelu_out_waddr), .wdata(gelu_out_wdata),
    .re(d1_in_re), .raddr(d1_in_raddr), .rdata(gelu_rdata)
  );

  logic d1_out_we; logic [TOK_AW-1:0] d1_out_waddr; logic [RANK_R*DATA_W-1:0] d1_out_wdata;
  logic d2_in_re;  logic [TOK_AW-1:0] d2_in_raddr;
  logic [RANK_R*DATA_W-1:0] hu_rdata;
  token_buffer #(.LANES(RANK_R)) u_hu_buf (
    .clk(clk), .we(d1_out_we), .waddr(d1_out_waddr), .wdata(d1_out_wdata),
    .re(d2_in_re), .raddr(d2_in_raddr), .rdata(hu_rdata)
  );

  logic d2_out_we; logic [TOK_AW-1:0] d2_out_waddr; logic [HID_D*DATA_W-1:0] d2_out_wdata;
  logic r2_b_re;   logic [TOK_AW-1:0] r2_b_raddr;
  logic [HID_D*DATA_W-1:0] ffn_out_rdata;
  token_buffer #(.LANES(HID_D)) u_ffn_out_buf (
    .clk(clk), .we(d2_out_we), .waddr(d2_out_waddr), .wdata(d2_out_wdata),
    .re(r2_b_re), .raddr(r2_b_raddr), .rdata(ffn_out_rdata)
  );

  patch_embed u_patch_embed (
    .clk(clk), .rst_n(rst_n), .start(patch_start), .done(patch_done),
    .pix_re(pix_re), .pix_raddr(pix_raddr), .pix_rdata(pix_rdata),
    .wc_re(wc_re_patch), .wc_raddr(wc_raddr_patch), .wc_rdata(wc_rdata),
    .out_we(patch_out_we), .out_waddr(patch_out_waddr), .out_wdata(patch_out_wdata)
  );

  linear_engine #(.IN_DIM(HID_D), .OUT_DIM(HID_D), .W_GBASE(WGOFF_Q)) u_qproj (
    .clk(clk), .rst_n(rst_n), .start(qproj_start), .done(qproj_done),
    .in_re(tokbuf_re_q), .in_raddr(tokbuf_raddr_q), .in_rdata(ln0_rdata),
    .wc_re(wc_re_q), .wc_raddr(wc_raddr_q), .wc_rdata(wc_rdata),
    .out_we(q_we), .out_waddr(q_waddr), .out_wdata(q_wdata)
  );
  linear_engine #(.IN_DIM(HID_D), .OUT_DIM(HID_D), .W_GBASE(WGOFF_K)) u_kproj (
    .clk(clk), .rst_n(rst_n), .start(kproj_start), .done(kproj_done),
    .in_re(tokbuf_re_k), .in_raddr(tokbuf_raddr_k), .in_rdata(ln0_rdata),
    .wc_re(wc_re_k), .wc_raddr(wc_raddr_k), .wc_rdata(wc_rdata),
    .out_we(k_we), .out_waddr(k_waddr), .out_wdata(k_wdata)
  );
  linear_engine #(.IN_DIM(HID_D), .OUT_DIM(HID_D), .W_GBASE(WGOFF_V)) u_vproj (
    .clk(clk), .rst_n(rst_n), .start(vproj_start), .done(vproj_done),
    .in_re(tokbuf_re_v), .in_raddr(tokbuf_raddr_v), .in_rdata(ln0_rdata),
    .wc_re(wc_re_v), .wc_raddr(wc_raddr_v), .wc_rdata(wc_rdata),
    .out_we(v_we), .out_waddr(v_waddr), .out_wdata(v_wdata)
  );

  linear_engine #(.IN_DIM(HID_D), .OUT_DIM(HID_D), .W_GBASE(WGOFF_O)) u_oproj (
    .clk(clk), .rst_n(rst_n), .start(oproj_start), .done(oproj_done),
    .in_re(op_in_re), .in_raddr(op_in_raddr), .in_rdata(attn_rdata),
    .wc_re(wc_re_o), .wc_raddr(wc_raddr_o), .wc_rdata(wc_rdata),
    .out_we(op_out_we), .out_waddr(op_out_waddr), .out_wdata(op_out_wdata)
  );

  logic ln0_gb_re;
  logic [$clog2(HID_D<=1?2:HID_D)-1:0] ln0_gb_raddr;
  logic [15:0] ln0_gb_rdata;

  ln_affine_store #(.GAMMA_FILE("ln0_gamma_init.hex"), .BETA_FILE("ln0_beta_init.hex")) u_ln0_affine (
    .clk(clk),
    .layer_idx(layer_idx),
    .gb_re(ln0_gb_re), .gb_raddr(ln0_gb_raddr), .gb_rdata(ln0_gb_rdata)
  );

  layernorm u_ln0 (
    .clk(clk), .rst_n(rst_n), .start(ln0_start), .done(ln0_done),
    .in_re(tokbuf_re_ln0), .in_raddr(tokbuf_raddr_ln0), .in_rdata(tokbuf_rdata),
    .out_we(ln0_out_we), .out_waddr(ln0_out_waddr), .out_wdata(ln0_out_wdata),
    .gb_re(ln0_gb_re), .gb_raddr(ln0_gb_raddr), .gb_rdata(ln0_gb_rdata)
  );

  spatial_attention_tile u_spatial (
    .clk(clk), .rst_n(rst_n), .start(spatial_start), .done(spatial_done),
    .q_re(q_re_sp), .q_raddr(q_raddr_sp), .q_rdata(q_rdata),
    .k_re(k_re_sp), .k_raddr(k_raddr_sp), .k_rdata(k_rdata),
    .v_re(v_re_sp), .v_raddr(v_raddr_sp), .v_rdata(v_rdata),
    .out_we(sp_out_we), .out_waddr(sp_out_waddr), .out_wdata(sp_out_wdata)
  );

  parameter int REFRESH_INTERVAL = 8;
  logic temporal_reuse_mode;
  assign temporal_reuse_mode = (REFRESH_INTERVAL <= 1) ? 1'b0
                              : ((step_idx % REFRESH_INTERVAL) != 0);

  logic tc_we, tc_re;
  logic [TOK_AW-1:0] tc_w_tok, tc_r_tok;
  logic [$clog2(N_HEADS<=1?2:N_HEADS)-1:0] tc_w_head, tc_r_head;
  logic [$clog2(TOPK_K<=1?2:TOPK_K)-1:0]   tc_w_iter, tc_r_iter;
  logic [TOK_AW-1:0] tc_w_idx, tc_r_idx;

  topk_cache u_topk_cache (
    .clk(clk), .layer_idx(layer_idx),
    .we(tc_we), .w_tok(tc_w_tok), .w_head(tc_w_head), .w_iter(tc_w_iter), .w_idx(tc_w_idx),
    .re(tc_re), .r_tok(tc_r_tok), .r_head(tc_r_head), .r_iter(tc_r_iter), .r_idx(tc_r_idx)
  );

  temporal_sparse_tile u_temporal (
    .clk(clk), .rst_n(rst_n), .start(temporal_start), .done(temporal_done),
    .q_re(q_re_tp), .q_raddr(q_raddr_tp), .q_rdata(q_rdata),
    .k_re(k_re_tp), .k_raddr(k_raddr_tp), .k_rdata(k_rdata),
    .v_re(v_re_tp), .v_raddr(v_raddr_tp), .v_rdata(v_rdata),
    .attn_re(tp_attn_re), .attn_raddr(tp_attn_raddr), .attn_rdata(attn_rdata),
    .attn_we(tp_attn_we), .attn_waddr(tp_attn_waddr), .attn_wdata(tp_attn_wdata),
    .reuse_mode(temporal_reuse_mode),
    .cache_we(tc_we), .cache_w_tok(tc_w_tok), .cache_w_head(tc_w_head),
    .cache_w_iter(tc_w_iter), .cache_w_idx(tc_w_idx),
    .cache_re(tc_re), .cache_r_tok(tc_r_tok), .cache_r_head(tc_r_head),
    .cache_r_iter(tc_r_iter), .cache_r_idx(tc_r_idx)
  );

  residual_add #(.LANES(HID_D)) u_resid1 (
    .clk(clk), .rst_n(rst_n), .start(resid1_start), .done(resid1_done),
    .a_re(tokbuf_re_r1), .a_raddr(tokbuf_raddr_r1), .a_rdata(tokbuf_rdata),
    .b_re(r1_b_re), .b_raddr(r1_b_raddr), .b_rdata(o_rdata),
    .out_we(r1_we), .out_waddr(r1_waddr), .out_wdata(r1_wdata)
  );

  logic ln1_gb_re;
  logic [$clog2(HID_D<=1?2:HID_D)-1:0] ln1_gb_raddr;
  logic [15:0] ln1_gb_rdata;

  ln_affine_store u_ln1_affine (
    .clk(clk),
    .layer_idx(layer_idx),
    .gb_re(ln1_gb_re), .gb_raddr(ln1_gb_raddr), .gb_rdata(ln1_gb_rdata)
  );

  layernorm u_ln1 (
    .clk(clk), .rst_n(rst_n), .start(ln1_start), .done(ln1_done),
    .in_re(ln1_in_re), .in_raddr(ln1_in_raddr), .in_rdata(r1_rdata),
    .out_we(ln1_out_we), .out_waddr(ln1_out_waddr), .out_wdata(ln1_out_wdata),
    .gb_re(ln1_gb_re), .gb_raddr(ln1_gb_raddr), .gb_rdata(ln1_gb_rdata)
  );

  linear_engine #(.IN_DIM(HID_D), .OUT_DIM(RANK_R), .W_GBASE(WGOFF_FFN_U)) u_ffn_up1 (
    .clk(clk), .rst_n(rst_n), .start(ffnu1_start), .done(ffnu1_done),
    .in_re(u1_in_re), .in_raddr(u1_in_raddr), .in_rdata(ln1_rdata),
    .wc_re(wc_re_u1), .wc_raddr(wc_raddr_u1), .wc_rdata(wc_rdata),
    .out_we(u1_out_we), .out_waddr(u1_out_waddr), .out_wdata(u1_out_wdata)
  );

  linear_engine #(.IN_DIM(RANK_R), .OUT_DIM(FFN_HID), .W_GBASE(WGOFF_FFN_V)) u_ffn_up2 (
    .clk(clk), .rst_n(rst_n), .start(ffnu2_start), .done(ffnu2_done),
    .in_re(u2_in_re), .in_raddr(u2_in_raddr), .in_rdata(xu_rdata),
    .wc_re(wc_re_u2), .wc_raddr(wc_raddr_u2), .wc_rdata(wc_rdata),
    .out_we(u2_out_we), .out_waddr(u2_out_waddr), .out_wdata(u2_out_wdata)
  );

  elementwise_gelu #(.LANES(FFN_HID)) u_gelu (
    .clk(clk), .rst_n(rst_n), .start(gelu_start), .done(gelu_done),
    .in_re(gelu_in_re), .in_raddr(gelu_in_raddr), .in_rdata(h_rdata),
    .out_we(gelu_out_we), .out_waddr(gelu_out_waddr), .out_wdata(gelu_out_wdata)
  );

  linear_engine #(.IN_DIM(FFN_HID), .OUT_DIM(RANK_R), .W_GBASE(WGOFF_FDN_U)) u_ffn_dn1 (
    .clk(clk), .rst_n(rst_n), .start(ffnd1_start), .done(ffnd1_done),
    .in_re(d1_in_re), .in_raddr(d1_in_raddr), .in_rdata(gelu_rdata),
    .wc_re(wc_re_d1), .wc_raddr(wc_raddr_d1), .wc_rdata(wc_rdata),
    .out_we(d1_out_we), .out_waddr(d1_out_waddr), .out_wdata(d1_out_wdata)
  );

  linear_engine #(.IN_DIM(RANK_R), .OUT_DIM(HID_D), .W_GBASE(WGOFF_FDN_V)) u_ffn_dn2 (
    .clk(clk), .rst_n(rst_n), .start(ffnd2_start), .done(ffnd2_done),
    .in_re(d2_in_re), .in_raddr(d2_in_raddr), .in_rdata(hu_rdata),
    .wc_re(wc_re_d2), .wc_raddr(wc_raddr_d2), .wc_rdata(wc_rdata),
    .out_we(d2_out_we), .out_waddr(d2_out_waddr), .out_wdata(d2_out_wdata)
  );

  residual_add #(.LANES(HID_D)) u_resid2 (
    .clk(clk), .rst_n(rst_n), .start(resid2_start), .done(resid2_done),
    .a_re(r2_a_re), .a_raddr(r2_a_raddr), .a_rdata(r1_rdata),
    .b_re(r2_b_re), .b_raddr(r2_b_raddr), .b_rdata(ffn_out_rdata),
    .out_we(tokbuf_we), .out_waddr(tokbuf_waddr), .out_wdata(tokbuf_wdata)
  );

  logic eps_we; logic [TOK_AW-1:0] eps_waddr; logic [PIX_DIM*DATA_W-1:0] eps_wdata;
  logic du_b_re; logic [TOK_AW-1:0] du_b_raddr;
  logic [PIX_DIM*DATA_W-1:0] eps_rdata;

  token_buffer #(.LANES(PIX_DIM)) u_eps_buf (
    .clk(clk), .we(eps_we), .waddr(eps_waddr), .wdata(eps_wdata),
    .re(du_b_re), .raddr(du_b_raddr), .rdata(eps_rdata)
  );

  linear_engine #(.IN_DIM(HID_D), .OUT_DIM(PIX_DIM), .W_GBASE(WGOFF_OUT)) u_outproj (
    .clk(clk), .rst_n(rst_n), .start(outp_start), .done(outp_done),
    .in_re(tokbuf_re_outp), .in_raddr(tokbuf_raddr_outp), .in_rdata(tokbuf_rdata),
    .wc_re(wc_re_out), .wc_raddr(wc_raddr_out), .wc_rdata(wc_rdata),
    .out_we(eps_we), .out_waddr(eps_waddr), .out_wdata(eps_wdata)
  );

  diffusion_update #(.LANES(PIX_DIM)) u_dupd (
    .clk(clk), .rst_n(rst_n), .start(dupd_start), .done(dupd_done),
    .a_re(du_a_re), .a_raddr(du_a_raddr), .a_rdata(pix_rdata),
    .b_re(du_b_re), .b_raddr(du_b_raddr), .b_rdata(eps_rdata),
    .out_we(du_out_we), .out_waddr(du_out_waddr), .out_wdata(du_out_wdata)
  );

endmodule : trident_top
