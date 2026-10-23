import trident_pkg::*;

module patch_embed (
  input  logic                      clk,
  input  logic                      rst_n,

  input  logic                      start,
  output logic                      done,

  output logic                      pix_re,
  output logic [TOK_AW-1:0]         pix_raddr,
  input  logic [PIX_DIM*DATA_W-1:0] pix_rdata,

  output logic                      wc_re,
  output logic [WCACHE_GAW-1:0]     wc_raddr,
  input  logic [PE_PAR*DATA_W-1:0]  wc_rdata,

  output logic                      out_we,
  output logic [TOK_AW-1:0]         out_waddr,
  output logic [HID_D*DATA_W-1:0]   out_wdata
);

  linear_engine #(
    .IN_DIM  (PIX_DIM),
    .OUT_DIM (HID_D),
    .W_GBASE (WGOFF_PATCH)
  ) u_engine (
    .clk       (clk),
    .rst_n     (rst_n),
    .start     (start),
    .done      (done),
    .in_re     (pix_re),
    .in_raddr  (pix_raddr),
    .in_rdata  (pix_rdata),
    .wc_re     (wc_re),
    .wc_raddr  (wc_raddr),
    .wc_rdata  (wc_rdata),
    .out_we    (out_we),
    .out_waddr (out_waddr),
    .out_wdata (out_wdata)
  );

endmodule : patch_embed
