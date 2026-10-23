import trident_pkg::*;

module tb_golden_temporal_reuse;

  logic clk = 0;
  logic rst_n = 0;
  always #5 clk = ~clk;

  logic start, done, reuse_mode;
  logic q_re, k_re, v_re, attn_re, attn_we;
  logic [TOK_AW-1:0] q_raddr, k_raddr, v_raddr, attn_raddr, attn_waddr;
  logic [HID_D*DATA_W-1:0] q_rdata, k_rdata, v_rdata, attn_rdata, attn_wdata;

  logic cache_we, cache_re;
  logic [TOK_AW-1:0] cache_w_tok, cache_r_tok;
  logic [$clog2(N_HEADS<=1?2:N_HEADS)-1:0] cache_w_head, cache_r_head;
  logic [$clog2(TOPK_K<=1?2:TOPK_K)-1:0]   cache_w_iter, cache_r_iter;
  logic [TOK_AW-1:0] cache_w_idx, cache_r_idx;

  logic [HID_D*DATA_W-1:0] q_mem [0:N_TOK-1];
  logic [HID_D*DATA_W-1:0] k_mem [0:N_TOK-1];
  logic [HID_D*DATA_W-1:0] v_mem [0:N_TOK-1];
  logic [HID_D*DATA_W-1:0] attn_mem [0:N_TOK-1];
  logic [HID_D*DATA_W-1:0] golden_mem [0:N_TOK-1];

  temporal_sparse_tile dut (
    .clk(clk), .rst_n(rst_n),
    .start(start), .done(done),
    .q_re(q_re), .q_raddr(q_raddr), .q_rdata(q_rdata),
    .k_re(k_re), .k_raddr(k_raddr), .k_rdata(k_rdata),
    .v_re(v_re), .v_raddr(v_raddr), .v_rdata(v_rdata),
    .attn_re(attn_re), .attn_raddr(attn_raddr), .attn_rdata(attn_rdata),
    .attn_we(attn_we), .attn_waddr(attn_waddr), .attn_wdata(attn_wdata),
    .reuse_mode(reuse_mode),
    .cache_we(cache_we), .cache_w_tok(cache_w_tok), .cache_w_head(cache_w_head),
    .cache_w_iter(cache_w_iter), .cache_w_idx(cache_w_idx),
    .cache_re(cache_re), .cache_r_tok(cache_r_tok), .cache_r_head(cache_r_head),
    .cache_r_iter(cache_r_iter), .cache_r_idx(cache_r_idx)
  );

  topk_cache u_cache (
    .clk(clk),
    .layer_idx('0),
    .we(cache_we), .w_tok(cache_w_tok), .w_head(cache_w_head),
    .w_iter(cache_w_iter), .w_idx(cache_w_idx),
    .re(cache_re), .r_tok(cache_r_tok), .r_head(cache_r_head),
    .r_iter(cache_r_iter), .r_idx(cache_r_idx)
  );

  always_ff @(posedge clk) begin
    if (q_re) q_rdata <= q_mem[q_raddr];
    if (k_re) k_rdata <= k_mem[k_raddr];
    if (v_re) v_rdata <= v_mem[v_raddr];
    if (attn_re) attn_rdata <= attn_mem[attn_raddr];
  end

  always_ff @(posedge clk) begin
    if (attn_we) attn_mem[attn_waddr] <= attn_wdata;
  end

  task automatic run_pass(input string qf, input string kf, input string vf,
                           input string af, input string gf, input bit reuse,
                           output int mismatches, output longint cycles);
    longint t0;
    begin
      $readmemh(qf, q_mem);
      $readmemh(kf, k_mem);
      $readmemh(vf, v_mem);
      $readmemh(af, attn_mem);
      $readmemh(gf, golden_mem);
      reuse_mode = reuse;
      start = 0;
      @(posedge clk);
      start = 1;
      t0 = $time;
      wait (done);
      cycles = ($time - t0) / 10;
      @(posedge clk);
      start = 0;
      @(posedge clk);
      mismatches = 0;
      for (int t = 0; t < N_TOK; t++) begin
        if (attn_mem[t] !== golden_mem[t]) begin
          mismatches++;
          $display("MISMATCH tok=%0d rtl=%h golden=%h", t, attn_mem[t], golden_mem[t]);
        end
      end
    end
  endtask

  int m1, m2;
  longint c1, c2;
  initial begin
    rst_n = 0;
    repeat (3) @(posedge clk);
    rst_n = 1;
    repeat (3) @(posedge clk);

    run_pass("q1_mem.hex", "k1_mem.hex", "v1_mem.hex", "attn1_mem.hex",
              "golden1_out.hex", 1'b0, m1, c1);
    $display("PASS1 (refresh/full-scan) CYCLE_COUNT %0d", c1);
    if (m1 == 0) $display("GOLDEN_CHECK PASS: reuse-pass1(refresh) all %0d tokens bit-exact", N_TOK);
    else $display("GOLDEN_CHECK FAIL: reuse-pass1(refresh) %0d/%0d mismatched", m1, N_TOK);

    run_pass("q2_mem.hex", "k2_mem.hex", "v2_mem.hex", "attn2_mem.hex",
              "golden2_out.hex", 1'b1, m2, c2);
    $display("PASS2 (reuse cached candidates) CYCLE_COUNT %0d", c2);
    if (m2 == 0) $display("GOLDEN_CHECK PASS: reuse-pass2(reuse) all %0d tokens bit-exact", N_TOK);
    else $display("GOLDEN_CHECK FAIL: reuse-pass2(reuse) %0d/%0d mismatched", m2, N_TOK);

    $finish;
  end

endmodule
