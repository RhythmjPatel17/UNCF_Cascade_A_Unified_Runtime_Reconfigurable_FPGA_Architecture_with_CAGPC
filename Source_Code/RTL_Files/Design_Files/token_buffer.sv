import trident_pkg::*;

module token_buffer #(
  parameter int LANES = HID_D,
  parameter int DEPTH = N_TOK
)(
  input  logic                              clk,
  input  logic                              we,
  input  logic [$clog2(DEPTH)-1:0]          waddr,
  input  logic [LANES*DATA_W-1:0]           wdata,
  input  logic                              re,
  input  logic [$clog2(DEPTH)-1:0]          raddr,
  output logic [LANES*DATA_W-1:0]           rdata
);

  bram_wide #(
    .DATA_W (DATA_W),
    .NLANES (LANES),
    .DEPTH  (DEPTH)
  ) u_buf (
    .clk   (clk),
    .we    (we),
    .waddr (waddr),
    .wdata (wdata),
    .re    (re),
    .raddr (raddr),
    .rdata (rdata)
  );

endmodule : token_buffer
