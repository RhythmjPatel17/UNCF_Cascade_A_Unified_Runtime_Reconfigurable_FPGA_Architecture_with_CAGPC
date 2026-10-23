module tb_fifo_sync;

  localparam int DW = 8;
  localparam int DEPTH = 8;

  logic clk = 0;
  always #5 clk = ~clk;
  logic rst_n;

  logic wr_en, full, rd_en, empty;
  logic [DW-1:0] wr_data, rd_data;

  fifo_sync #(.DATA_W(DW), .DEPTH(DEPTH)) dut (
    .clk (clk), .rst_n (rst_n),
    .wr_en (wr_en), .wr_data (wr_data), .full (full),
    .rd_en (rd_en), .rd_data (rd_data), .empty (empty)
  );

  integer errors = 0;
  integer i;

  initial begin
    rst_n = 0; wr_en = 0; rd_en = 0; wr_data = '0;
    repeat (3) @(posedge clk);
    rst_n = 1;
    @(posedge clk);

    if (!empty) begin $display("FAIL: fifo not empty after reset"); errors++; end

    for (i = 0; i < DEPTH; i++) begin
      @(posedge clk);
      wr_en   = 1'b1;
      wr_data = i[DW-1:0];
    end
    @(posedge clk);
    wr_en = 1'b0;
    @(posedge clk);

    if (!full) begin $display("FAIL: fifo should be full after %0d pushes", DEPTH); errors++; end

    for (i = 0; i < DEPTH; i++) begin
      rd_en = 1'b1;
      @(posedge clk);
      rd_en = 1'b0;
      @(posedge clk);
      if (rd_data !== i[DW-1:0]) begin
        $display("FAIL: pop %0d expected=%0d got=%0d", i, i, rd_data);
        errors++;
      end
    end

    @(posedge clk);
    if (!empty) begin $display("FAIL: fifo not empty after draining"); errors++; end

    if (errors == 0) $display("tb_fifo_sync: *** PASS ***");
    else begin
      $display("tb_fifo_sync: *** FAIL (%0d errors) ***", errors);
      $fatal(1);
    end
    $finish;
  end

endmodule : tb_fifo_sync
