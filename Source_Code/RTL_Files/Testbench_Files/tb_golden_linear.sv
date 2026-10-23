import trident_pkg::*;

module tb_golden_linear;

  localparam int IN_DIM  = 8;
  localparam int OUT_DIM = 8;
  localparam int NTOK    = 16;
  localparam int OGRP    = OUT_DIM / PE_PAR;

  logic clk = 0;
  logic rst_n = 0;
  always #5 clk = ~clk;

  logic start, done;
  logic in_re;
  logic [$clog2(NTOK)-1:0] in_raddr;
  logic [IN_DIM*DATA_W-1:0] in_rdata;

  logic wc_re;
  logic [WCACHE_GAW-1:0] wc_raddr;
  logic [PE_PAR*DATA_W-1:0] wc_rdata;

  logic out_we;
  logic [$clog2(NTOK)-1:0] out_waddr;
  logic [OUT_DIM*DATA_W-1:0] out_wdata;

  logic [IN_DIM*DATA_W-1:0] in_mem [0:NTOK-1];
  logic [PE_PAR*DATA_W-1:0] wc_mem [0:IN_DIM*OGRP-1];
  logic [OUT_DIM*DATA_W-1:0] out_mem [0:NTOK-1];
  logic [OUT_DIM*DATA_W-1:0] golden_mem [0:NTOK-1];

  initial begin
    $readmemh("in_tokens.hex", in_mem);
    $readmemh("wc_mem.hex", wc_mem);
    $readmemh("golden_out.hex", golden_mem);
  end

  always_ff @(posedge clk) begin
    if (in_re) in_rdata <= in_mem[in_raddr];
    if (wc_re) wc_rdata <= wc_mem[wc_raddr];
  end

  always_ff @(posedge clk) begin
    if (out_we) out_mem[out_waddr] <= out_wdata;
  end

  linear_engine #(
    .IN_DIM  (IN_DIM),
    .OUT_DIM (OUT_DIM),
    .W_GBASE (0),
    .NTOK    (NTOK)
  ) dut (
    .clk      (clk),
    .rst_n    (rst_n),
    .start    (start),
    .done     (done),
    .in_re    (in_re),
    .in_raddr (in_raddr),
    .in_rdata (in_rdata),
    .wc_re    (wc_re),
    .wc_raddr (wc_raddr),
    .wc_rdata (wc_rdata),
    .out_we   (out_we),
    .out_waddr(out_waddr),
    .out_wdata(out_wdata)
  );

  int mismatches;
  longint cyc_start, cyc_done;
  initial begin
    start = 0;
    mismatches = 0;
    repeat (3) @(posedge clk);
    rst_n = 1;
    repeat (3) @(posedge clk);
    start = 1;
    cyc_start = $time;
    wait (done);
    cyc_done = $time;
    repeat (2) @(posedge clk);

    for (int t = 0; t < NTOK; t++) begin
      if (out_mem[t] !== golden_mem[t]) begin
        mismatches++;
        $display("MISMATCH tok=%0d rtl=%h golden=%h", t, out_mem[t], golden_mem[t]);
      end
    end

    $display("CYCLE_COUNT %0d", (cyc_done - cyc_start) / 10);

    if (mismatches == 0)
      $display("GOLDEN_CHECK PASS: all %0d tokens bit-exact match", NTOK);
    else
      $display("GOLDEN_CHECK FAIL: %0d/%0d tokens mismatched", mismatches, NTOK);

    $finish;
  end

endmodule
