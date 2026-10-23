module tb_divider16;
  logic clk = 0;
  always #5 clk = ~clk;
  logic rst_n, start, done;
  logic [15:0] numer, denom, quot;

  divider16 dut (
    .clk(clk), .rst_n(rst_n), .start(start),
    .numer(numer), .denom(denom), .done(done), .quot(quot)
  );

  integer errors = 0;

  task automatic check(input [15:0] n, input [15:0] d, input [15:0] expect_q);
    begin
      @(posedge clk); start = 1; numer = n; denom = d;
      @(posedge clk); start = 0;
      wait (done);
      @(posedge clk);
      if (quot !== expect_q) begin
        $display("FAIL: %0d/%0d expected %0d got %0d", n, d, expect_q, quot);
        errors = errors + 1;
      end else begin
        $display("PASS: %0d/%0d = %0d", n, d, quot);
      end
      @(posedge clk);
    end
  endtask

  initial begin
    rst_n = 0; start = 0; numer = 0; denom = 0;
    repeat (3) @(posedge clk);
    rst_n = 1;
    repeat (3) @(posedge clk);

    check(16'd10, 16'd3, 16'd3);
    check(16'd32640, 16'd8160, 16'd4);
    check(16'd255, 16'd1, 16'd255);
    check(16'd0, 16'd7, 16'd0);
    check(16'd32640, 16'd1, 16'd32640);
    check(16'd100, 16'd200, 16'd0);
    check(16'd999, 16'd37, 16'd27);

    if (errors == 0) $display("TB: *** ALL PASS ***");
    else begin
      $display("TB: *** %0d FAILURES ***", errors);
      $fatal(1);
    end
    $finish;
  end
endmodule
