import trident_pkg::*;

module tb_golden_spatial;

  logic clk = 0;
  logic rst_n = 0;
  always #5 clk = ~clk;

  logic start, done;
  logic q_re, k_re, v_re;
  logic [TOK_AW-1:0] q_raddr, k_raddr, v_raddr;
  logic [HID_D*DATA_W-1:0] q_rdata, k_rdata, v_rdata;
  logic out_we;
  logic [TOK_AW-1:0] out_waddr;
  logic [HID_D*DATA_W-1:0] out_wdata;

  logic [HID_D*DATA_W-1:0] q_mem [0:N_TOK-1];
  logic [HID_D*DATA_W-1:0] k_mem [0:N_TOK-1];
  logic [HID_D*DATA_W-1:0] v_mem [0:N_TOK-1];
  logic [HID_D*DATA_W-1:0] out_mem [0:N_TOK-1];
  logic [HID_D*DATA_W-1:0] golden_mem [0:N_TOK-1];

  initial begin
    $readmemh("q_mem.hex", q_mem);
    $readmemh("k_mem.hex", k_mem);
    $readmemh("v_mem.hex", v_mem);
    $readmemh("golden_spatial_out.hex", golden_mem);
  end

  always_ff @(posedge clk) begin
    if (q_re) q_rdata <= q_mem[q_raddr];
    if (k_re) k_rdata <= k_mem[k_raddr];
    if (v_re) v_rdata <= v_mem[v_raddr];
  end

  always_ff @(posedge clk) begin
    if (out_we) out_mem[out_waddr] <= out_wdata;
  end

  spatial_attention_tile dut (
    .clk(clk), .rst_n(rst_n),
    .start(start), .done(done),
    .q_re(q_re), .q_raddr(q_raddr), .q_rdata(q_rdata),
    .k_re(k_re), .k_raddr(k_raddr), .k_rdata(k_rdata),
    .v_re(v_re), .v_raddr(v_raddr), .v_rdata(v_rdata),
    .out_we(out_we), .out_waddr(out_waddr), .out_wdata(out_wdata)
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

    for (int t = 0; t < N_TOK; t++) begin
      if (out_mem[t] !== golden_mem[t]) begin
        mismatches++;
        $display("MISMATCH tok=%0d rtl=%h golden=%h", t, out_mem[t], golden_mem[t]);
      end
    end

    $display("CYCLE_COUNT %0d", (cyc_done - cyc_start) / 10);

    if (mismatches == 0)
      $display("GOLDEN_CHECK PASS: spatial all %0d tokens bit-exact match", N_TOK);
    else
      $display("GOLDEN_CHECK FAIL: spatial %0d/%0d tokens mismatched", mismatches, N_TOK);

    $finish;
  end

endmodule
