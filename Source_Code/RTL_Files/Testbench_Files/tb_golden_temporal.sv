import trident_pkg::*;

module tb_golden_temporal;

  logic clk = 0;
  logic rst_n = 0;
  always #5 clk = ~clk;

  logic start, done;
  logic q_re, k_re, v_re, attn_re, attn_we;
  logic [TOK_AW-1:0] q_raddr, k_raddr, v_raddr, attn_raddr, attn_waddr;
  logic [HID_D*DATA_W-1:0] q_rdata, k_rdata, v_rdata, attn_rdata, attn_wdata;

  logic [HID_D*DATA_W-1:0] q_mem [0:N_TOK-1];
  logic [HID_D*DATA_W-1:0] k_mem [0:N_TOK-1];
  logic [HID_D*DATA_W-1:0] v_mem [0:N_TOK-1];
  logic [HID_D*DATA_W-1:0] attn_mem [0:N_TOK-1];
  logic [HID_D*DATA_W-1:0] golden_mem [0:N_TOK-1];

  initial begin
    $readmemh("q_mem_t.hex", q_mem);
    $readmemh("k_mem_t.hex", k_mem);
    $readmemh("v_mem_t.hex", v_mem);
    $readmemh("attn_mem_t.hex", attn_mem);
    $readmemh("golden_temporal_out.hex", golden_mem);
  end

  always_ff @(posedge clk) begin
    if (q_re) q_rdata <= q_mem[q_raddr];
    if (k_re) k_rdata <= k_mem[k_raddr];
    if (v_re) v_rdata <= v_mem[v_raddr];
    if (attn_re) attn_rdata <= attn_mem[attn_raddr];
  end

  always_ff @(posedge clk) begin
    if (attn_we) attn_mem[attn_waddr] <= attn_wdata;
  end

  temporal_sparse_tile dut (
    .clk(clk), .rst_n(rst_n),
    .start(start), .done(done),
    .q_re(q_re), .q_raddr(q_raddr), .q_rdata(q_rdata),
    .k_re(k_re), .k_raddr(k_raddr), .k_rdata(k_rdata),
    .v_re(v_re), .v_raddr(v_raddr), .v_rdata(v_rdata),
    .attn_re(attn_re), .attn_raddr(attn_raddr), .attn_rdata(attn_rdata),
    .attn_we(attn_we), .attn_waddr(attn_waddr), .attn_wdata(attn_wdata),
    .reuse_mode(1'b0),
    .cache_we(), .cache_w_tok(), .cache_w_head(), .cache_w_iter(), .cache_w_idx(),
    .cache_re(), .cache_r_tok(), .cache_r_head(), .cache_r_iter(), .cache_r_idx('0)
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
      if (attn_mem[t] !== golden_mem[t]) begin
        mismatches++;
        $display("MISMATCH tok=%0d rtl=%h golden=%h", t, attn_mem[t], golden_mem[t]);
      end
    end

    $display("CYCLE_COUNT %0d", (cyc_done - cyc_start) / 10);

    if (mismatches == 0)
      $display("GOLDEN_CHECK PASS: temporal all %0d tokens bit-exact match", N_TOK);
    else
      $display("GOLDEN_CHECK FAIL: temporal %0d/%0d tokens mismatched", mismatches, N_TOK);

    $finish;
  end

endmodule
