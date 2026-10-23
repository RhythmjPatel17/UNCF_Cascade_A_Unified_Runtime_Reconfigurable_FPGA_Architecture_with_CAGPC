import trident_pkg::*;

module tb_trident_top;

  logic clk = 0;
  logic rst_n;
  always #5 clk = ~clk;

  logic start, busy, done;
  decomp_mode_e decomp_mode;

  logic                   ext_mem_req;
  logic [31:0]            ext_mem_addr;
  logic                   ext_mem_valid;
  logic [COMP_WORD_W-1:0] ext_mem_rdata;

  logic                    dbg_tok_we;
  logic [TOK_AW-1:0]       dbg_tok_waddr;
  logic [HID_D*DATA_W-1:0] dbg_tok_wdata;
  logic                    dbg_tok_re;
  logic [TOK_AW-1:0]       dbg_tok_raddr;
  logic [HID_D*DATA_W-1:0] dbg_tok_rdata;
  logic [$clog2(DEPTH<=1?2:DEPTH)-1:0]     dbg_layer_idx;
  logic [$clog2(T_STEPS<=1?2:T_STEPS)-1:0] dbg_step_idx;

  logic                     dbg_pix_we;
  logic [TOK_AW-1:0]        dbg_pix_waddr;
  logic [PIX_DIM*DATA_W-1:0] dbg_pix_wdata;
  logic                      dbg_pix_re;
  logic [TOK_AW-1:0]         dbg_pix_raddr;
  logic [PIX_DIM*DATA_W-1:0] dbg_pix_rdata;

  trident_top dut (
    .clk (clk), .rst_n (rst_n),
    .start (start), .busy (busy), .done (done),
    .decomp_mode (decomp_mode),
    .ext_mem_req (ext_mem_req), .ext_mem_addr (ext_mem_addr),
    .ext_mem_valid (ext_mem_valid), .ext_mem_rdata (ext_mem_rdata),
    .dbg_tok_we (dbg_tok_we), .dbg_tok_waddr (dbg_tok_waddr), .dbg_tok_wdata (dbg_tok_wdata),
    .dbg_tok_re (dbg_tok_re), .dbg_tok_raddr (dbg_tok_raddr), .dbg_tok_rdata (dbg_tok_rdata),
    .dbg_pix_we (dbg_pix_we), .dbg_pix_waddr (dbg_pix_waddr), .dbg_pix_wdata (dbg_pix_wdata),
    .dbg_pix_re (dbg_pix_re), .dbg_pix_raddr (dbg_pix_raddr), .dbg_pix_rdata (dbg_pix_rdata),
    .dbg_layer_idx (dbg_layer_idx), .dbg_step_idx (dbg_step_idx)
  );

  sim_mem #(
    .DATA_W (COMP_WORD_W),
    .DEPTH  (COMP_WORDS_TOTAL),
    .INIT_FILE ("compressed_weights.hex")
  ) u_mem (
    .clk   (clk),
    .req   (ext_mem_req),
    .addr  (ext_mem_addr),
    .valid (ext_mem_valid),
    .rdata (ext_mem_rdata)
  );

  integer i, d;
  integer errors;

  initial begin
    rst_n = 0;
    start = 0;
    decomp_mode = MODE_INT4;
    dbg_tok_we = 0; dbg_tok_waddr = '0; dbg_tok_wdata = '0;
    dbg_tok_re = 0; dbg_tok_raddr = '0;
    dbg_pix_we = 0; dbg_pix_waddr = '0; dbg_pix_wdata = '0;
    dbg_pix_re = 0; dbg_pix_raddr = '0;
    errors = 0;

    repeat (5) @(posedge clk);
    rst_n = 1;
    repeat (5) @(posedge clk);

    for (i = 0; i < N_TOK; i++) begin
      @(posedge clk);
      dbg_pix_we    = 1'b1;
      dbg_pix_waddr = i[TOK_AW-1:0];
      for (d = 0; d < PIX_DIM; d++) begin
        dbg_pix_wdata[d*DATA_W +: DATA_W] = int8_t'(((i*PIX_DIM + d) % 32) - 16);
      end
    end
    @(posedge clk);
    dbg_pix_we = 1'b0;

    $display("[%0t] TB: synthetic patch data loaded, starting inference", $time);

    @(posedge clk);
    start = 1'b1;
    @(posedge clk);
    start = 1'b0;

    fork
      begin
        int wcount;
        for (wcount = 0; wcount < 2_000_000_000; wcount++) @(posedge clk);
        $display("[%0t] TB: *** TIMEOUT waiting for done -- possible deadlock ***", $time);
        errors = errors + 1;
      end
      begin
        @(posedge done);
      end
    join_any
    disable fork;

    if (errors == 0) begin
      $display("[%0t] TB: inference completed (done asserted)", $time);
    end

    for (i = 0; i < N_TOK; i++) begin
      @(posedge clk);
      dbg_tok_re    = 1'b1;
      dbg_tok_raddr = i[TOK_AW-1:0];
      @(posedge clk);
      dbg_tok_re = 1'b0;
      @(posedge clk);
      $write("[%0t] TB: token[%0d] =", $time, i);
      for (d = 0; d < HID_D; d++) begin
        $write(" %0d", $signed(dbg_tok_rdata[d*DATA_W +: DATA_W]));
      end
      $write("\n");
    end

    if (errors == 0) begin
      $display("TB: *** PASS (pipeline ran to completion, no deadlock/timeout) ***");
    end else begin
      $display("TB: *** FAIL (%0d error(s)) ***", errors);
      $fatal(1);
    end

    $finish;
  end

  logic seen_done;
  always_ff @(posedge clk) begin
    if (!rst_n) seen_done <= 1'b0;
    else if (done) seen_done <= 1'b1;
  end

endmodule : tb_trident_top
