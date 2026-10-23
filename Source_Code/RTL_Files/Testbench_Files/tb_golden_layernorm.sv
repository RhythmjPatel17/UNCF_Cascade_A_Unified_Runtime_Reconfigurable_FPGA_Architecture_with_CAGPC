import trident_pkg::*;

module tb_golden_layernorm;

  logic clk = 0;
  logic rst_n = 0;
  always #5 clk = ~clk;

  logic start, done;
  logic in_re;
  logic [TOK_AW-1:0] in_raddr;
  logic [HID_D*DATA_W-1:0] in_rdata;
  logic out_we;
  logic [TOK_AW-1:0] out_waddr;
  logic [HID_D*DATA_W-1:0] out_wdata;
  logic gb_re;
  logic [$clog2(HID_D<=1?2:HID_D)-1:0] gb_raddr;
  logic [15:0] gb_rdata;

  logic [HID_D*DATA_W-1:0] in_mem [0:N_TOK-1];
  logic [15:0] gb_mem [0:HID_D-1];
  logic [HID_D*DATA_W-1:0] out_mem [0:N_TOK-1];
  logic [HID_D*DATA_W-1:0] golden_mem [0:N_TOK-1];

  initial begin
    $readmemh("ln_in.hex", in_mem);
    $readmemh("ln_gb.hex", gb_mem);
    $readmemh("ln_golden_out.hex", golden_mem);
  end

  always_ff @(posedge clk) begin
    if (in_re) in_rdata <= in_mem[in_raddr];
    if (gb_re) gb_rdata <= gb_mem[gb_raddr];
  end

  always_ff @(posedge clk) begin
    if (out_we) out_mem[out_waddr] <= out_wdata;
  end

  layernorm dut (
    .clk(clk), .rst_n(rst_n),
    .start(start), .done(done),
    .in_re(in_re), .in_raddr(in_raddr), .in_rdata(in_rdata),
    .out_we(out_we), .out_waddr(out_waddr), .out_wdata(out_wdata),
    .gb_re(gb_re), .gb_raddr(gb_raddr), .gb_rdata(gb_rdata)
  );

  int mismatches;
  initial begin
    start = 0;
    mismatches = 0;
    repeat (3) @(posedge clk);
    rst_n = 1;
    repeat (3) @(posedge clk);
    start = 1;
    wait (done);
    repeat (2) @(posedge clk);

    for (int t = 0; t < N_TOK; t++) begin
      if (out_mem[t] !== golden_mem[t]) begin
        mismatches++;
        $display("MISMATCH tok=%0d rtl=%h golden=%h", t, out_mem[t], golden_mem[t]);
      end
    end

    if (mismatches == 0)
      $display("GOLDEN_CHECK PASS: layernorm all %0d tokens bit-exact match", N_TOK);
    else
      $display("GOLDEN_CHECK FAIL: layernorm %0d/%0d tokens mismatched", mismatches, N_TOK);

    $finish;
  end

endmodule
