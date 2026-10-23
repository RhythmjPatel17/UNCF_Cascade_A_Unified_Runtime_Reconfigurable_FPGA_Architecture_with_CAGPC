import trident_pkg::*;

module tb_bitslice_decoder;

  logic [COMP_WORD_W-1:0] packed_word;
  logic [3:0]             slice_idx;
  decomp_mode_e           mode;
  int8_t                  operand;

  bitslice_decoder dut (
    .packed_word (packed_word),
    .slice_idx   (slice_idx),
    .mode        (mode),
    .operand     (operand)
  );

  integer errors = 0;

  task automatic check_int4(input int nibble_val);
    begin
      packed_word = 32'h0;
      packed_word[3:0] = nibble_val[3:0];
      mode = MODE_INT4;
      slice_idx = 4'd0;
      #1;
      if (operand !== $signed(nibble_val[3:0])) begin
        $display("FAIL INT4 nibble=%0d expected=%0d got=%0d", nibble_val, $signed(nibble_val[3:0]), operand);
        errors++;
      end
    end
  endtask

  initial begin

    for (int n = 0; n < 16; n++) check_int4(n);

    mode = MODE_TERNARY;
    packed_word = 32'h0;
    packed_word[1:0] = 2'b00; slice_idx = 4'd0; #1;
    if (operand !== 8'sd0) begin $display("FAIL ternary 00 got=%0d", operand); errors++; end

    packed_word = 32'h0;
    packed_word[1:0] = 2'b01; slice_idx = 4'd0; #1;
    if (operand !== 8'sd127) begin $display("FAIL ternary 01 got=%0d", operand); errors++; end

    packed_word = 32'h0;
    packed_word[1:0] = 2'b10; slice_idx = 4'd0; #1;
    if (operand !== -8'sd127) begin $display("FAIL ternary 10 got=%0d", operand); errors++; end

    mode = MODE_INT4;
    packed_word = 32'h0;
    packed_word[23:20] = 4'hD;
    slice_idx = 4'd5;
    #1;
    if (operand !== -8'sd3) begin
      $display("FAIL slice5 alignment expected=-3 got=%0d", operand);
      errors++;
    end

    if (errors == 0) $display("tb_bitslice_decoder: *** PASS ***");
    else begin
      $display("tb_bitslice_decoder: *** FAIL (%0d errors) ***", errors);
      $fatal(1);
    end
    $finish;
  end

endmodule : tb_bitslice_decoder
