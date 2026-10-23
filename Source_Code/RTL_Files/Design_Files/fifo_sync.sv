module fifo_sync #(
  parameter int DATA_W = 8,
  parameter int DEPTH  = 32,
  parameter int ADDR_W = $clog2(DEPTH)
)(
  input  logic                 clk,
  input  logic                 rst_n,

  input  logic                 wr_en,
  input  logic [DATA_W-1:0]    wr_data,
  output logic                 full,

  input  logic                 rd_en,
  output logic [DATA_W-1:0]    rd_data,
  output logic                 empty
);

  logic [ADDR_W:0] wr_ptr, rd_ptr;
  logic            mem_we;
  logic [ADDR_W-1:0] mem_waddr, mem_raddr;

  assign mem_we    = wr_en && !full;
  assign mem_waddr = wr_ptr[ADDR_W-1:0];
  assign mem_raddr = rd_ptr[ADDR_W-1:0];

  bram_sdp #(
    .DATA_W (DATA_W),
    .DEPTH  (DEPTH),
    .ADDR_W (ADDR_W)
  ) u_storage (
    .clk    (clk),
    .we     (mem_we),
    .waddr  (mem_waddr),
    .wdata  (wr_data),
    .re     (rd_en && !empty),
    .raddr  (mem_raddr),
    .rdata  (rd_data)
  );

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n)      wr_ptr <= '0;
    else if (mem_we) wr_ptr <= wr_ptr + 1'b1;
  end

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n)                  rd_ptr <= '0;
    else if (rd_en && !empty)    rd_ptr <= rd_ptr + 1'b1;
  end

  assign empty = (wr_ptr == rd_ptr);
  assign full  = (wr_ptr[ADDR_W-1:0] == rd_ptr[ADDR_W-1:0]) &&
                 (wr_ptr[ADDR_W]     != rd_ptr[ADDR_W]);

endmodule : fifo_sync
