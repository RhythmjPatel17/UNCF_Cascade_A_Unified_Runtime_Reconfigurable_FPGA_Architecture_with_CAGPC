module sim_mem #(
  parameter int DATA_W = 32,
  parameter int DEPTH  = 256,
  parameter int ADDR_W = (DEPTH <= 1) ? 1 : $clog2(DEPTH),
  parameter string INIT_FILE = ""
)(
  input  logic                clk,
  input  logic                req,
  input  logic [31:0]         addr,
  output logic                valid,
  output logic [DATA_W-1:0]   rdata
);

  logic [DATA_W-1:0] mem [0:DEPTH-1];

  initial begin
    if (INIT_FILE != "") $readmemh(INIT_FILE, mem);
  end

  always_ff @(posedge clk) begin
    valid <= req;
    if (req) rdata <= mem[addr[ADDR_W+1:2]];
  end

endmodule : sim_mem
