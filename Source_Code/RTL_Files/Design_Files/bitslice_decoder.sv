import trident_pkg::*;

module bitslice_decoder (
  input  logic [COMP_WORD_W-1:0] packed_word,
  input  logic [3:0]             slice_idx,
  input  decomp_mode_e           mode,
  output int8_t                  operand
);

  logic [COMP_W-1:0]  raw_int4_slice;
  logic [1:0]         raw_tern_slice;
  logic [4:0]          barrel_shift_amt_int4;
  logic [4:0]          barrel_shift_amt_tern;

  assign barrel_shift_amt_int4 = {2'b00, slice_idx[2:0]} * 5'd4;
  assign barrel_shift_amt_tern = {1'b0, slice_idx}        * 5'd2;

  assign raw_int4_slice = packed_word[barrel_shift_amt_int4 +: COMP_W];
  assign raw_tern_slice = packed_word[barrel_shift_amt_tern +: 2];

  int8_t int4_expanded;
  always_comb begin
    int4_expanded = { {(DATA_W-COMP_W){raw_int4_slice[COMP_W-1]}}, raw_int4_slice };
  end

  int8_t tern_expanded;
  always_comb begin
    unique case (raw_tern_slice)
      2'b00:   tern_expanded =  8'sd0;
      2'b01:   tern_expanded =  8'sd127;
      2'b10:   tern_expanded = -8'sd127;
      default: tern_expanded =  8'sd0;
    endcase
  end

  always_comb begin
    operand = (mode == MODE_INT4) ? int4_expanded : tern_expanded;
  end

endmodule : bitslice_decoder
