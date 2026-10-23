import trident_pkg::*;

module tb_golden_top;

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

  logic                      dbg_pix_we;
  logic [TOK_AW-1:0]         dbg_pix_waddr;
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
    .clk (clk), .req (ext_mem_req), .addr (ext_mem_addr),
    .valid (ext_mem_valid), .rdata (ext_mem_rdata)
  );

  logic [PIX_DIM*DATA_W-1:0] pix_in_mem [0:N_TOK-1];
  integer i, fp, ft;
  longint cyc;

  initial begin
    $readmemh("pix_in.hex", pix_in_mem);
    rst_n = 0;
    start = 0;
    decomp_mode = MODE_INT4;
    dbg_tok_we = 0; dbg_tok_waddr = '0; dbg_tok_wdata = '0;
    dbg_tok_re = 0; dbg_tok_raddr = '0;
    dbg_pix_we = 0; dbg_pix_waddr = '0; dbg_pix_wdata = '0;
    dbg_pix_re = 0; dbg_pix_raddr = '0;

    repeat (5) @(posedge clk);
    rst_n = 1;
    repeat (5) @(posedge clk);

    for (i = 0; i < N_TOK; i++) begin
      @(posedge clk);
      dbg_pix_we    = 1'b1;
      dbg_pix_waddr = i[TOK_AW-1:0];
      dbg_pix_wdata = pix_in_mem[i];
    end
    @(posedge clk);
    dbg_pix_we = 1'b0;

    @(posedge clk);
    start = 1'b1;
    @(posedge clk);
    start = 1'b0;

    cyc = 0;
    while (!done && cyc < 500_000_000) begin
      @(posedge clk);
      cyc++;
    end
    $display("CYCLES_TO_DONE %0d", cyc);

    fp = $fopen("pix_out_rtl.hex", "w");
    ft = $fopen("tok_out_rtl.hex", "w");
    for (i = 0; i < N_TOK; i++) begin
      @(posedge clk);
      dbg_pix_re = 1'b1; dbg_pix_raddr = i[TOK_AW-1:0];
      dbg_tok_re = 1'b1; dbg_tok_raddr = i[TOK_AW-1:0];
      @(posedge clk);
      dbg_pix_re = 1'b0; dbg_tok_re = 1'b0;
      @(posedge clk);
      $fwrite(fp, "%x\n", dbg_pix_rdata);
      $fwrite(ft, "%x\n", dbg_tok_rdata);
    end
    $fclose(fp);
    $fclose(ft);
    $finish;
  end

endmodule
