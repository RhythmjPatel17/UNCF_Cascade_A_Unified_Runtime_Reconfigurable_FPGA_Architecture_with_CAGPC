module bram_sdp #(
  parameter int DATA_W = 8,
  parameter int DEPTH  = 256,
  parameter int ADDR_W = (DEPTH <= 1) ? 1 : $clog2(DEPTH)
)(
  input  logic                  clk,

  input  logic                  we,
  input  logic [ADDR_W-1:0]     waddr,
  input  logic [DATA_W-1:0]     wdata,

  input  logic                  re,
  input  logic [ADDR_W-1:0]     raddr,
  output logic [DATA_W-1:0]     rdata
);

  (* ram_style = "block" *) logic [DATA_W-1:0] mem [0:DEPTH-1];

  always_ff @(posedge clk) begin
    if (we) mem[waddr] <= wdata;
  end

  always_ff @(posedge clk) begin
    if (re) rdata <= mem[raddr];
  end

endmodule : bram_sdp
