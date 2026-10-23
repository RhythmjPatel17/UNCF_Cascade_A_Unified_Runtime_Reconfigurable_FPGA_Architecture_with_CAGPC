`include "trident_params.svh"

package trident_pkg;

  parameter int DATA_W   = 8;
  parameter int ACC_W    = 32;
  parameter int COMP_W   = 4;

  typedef logic signed [DATA_W-1:0] int8_t;
  typedef logic signed [ACC_W-1:0]  acc_t;

  parameter int N_TOK      = `TRIDENT_N_TOK;
  parameter int HID_D      = `TRIDENT_HID_D;
  parameter int N_HEADS    = `TRIDENT_N_HEADS;
  parameter int WIN_W      = `TRIDENT_WIN_W;
  parameter int TOPK_K     = `TRIDENT_TOPK_K;
  parameter int DEPTH      = `TRIDENT_DEPTH;
  parameter int MLP_RATIO  = `TRIDENT_MLP_RATIO;
  parameter int RANK_R     = `TRIDENT_RANK_R;
  parameter int T_STEPS    = `TRIDENT_T_STEPS;

  parameter int PE_PAR     = `TRIDENT_PE_PAR;

  parameter int CH         = 3;
  parameter int PATCH      = 2;
  parameter int PIX_DIM    = CH * PATCH * PATCH;

  parameter int FFN_HID    = HID_D * MLP_RATIO;
  parameter int QSHIFT     = 3;
  parameter int UPD_SHIFT  = 2;

  parameter int TOK_AW     = $clog2(N_TOK  <= 1 ? 2 : N_TOK);
  parameter int DIM_AW     = $clog2(HID_D  <= 1 ? 2 : HID_D);
  parameter int FFN_AW     = $clog2(FFN_HID<= 1 ? 2 : FFN_HID);

  parameter int WCACHE_DEPTH =
      (PIX_DIM*HID_D)
    + (HID_D*HID_D)*4
    + (HID_D*RANK_R)
    + (FFN_HID*RANK_R)
    + (FFN_HID*RANK_R)
    + (HID_D*RANK_R)
    + (HID_D*PIX_DIM);
  parameter int WCACHE_AW = $clog2(WCACHE_DEPTH <= 1 ? 2 : WCACHE_DEPTH);

  parameter int PE_PAR_AW    = $clog2(PE_PAR <= 1 ? 2 : PE_PAR);
  parameter int WCACHE_GDEPTH = (WCACHE_DEPTH + PE_PAR - 1) / PE_PAR;
  parameter int WCACHE_GAW    = $clog2(WCACHE_GDEPTH <= 1 ? 2 : WCACHE_GDEPTH);

  parameter int WOFF_PATCH = 0;
  parameter int WOFF_Q     = WOFF_PATCH + PIX_DIM*HID_D;
  parameter int WOFF_K     = WOFF_Q + HID_D*HID_D;
  parameter int WOFF_V     = WOFF_K + HID_D*HID_D;
  parameter int WOFF_O     = WOFF_V + HID_D*HID_D;
  parameter int WOFF_FFN_U = WOFF_O + HID_D*HID_D;
  parameter int WOFF_FFN_V = WOFF_FFN_U + HID_D*RANK_R;
  parameter int WOFF_FDN_U = WOFF_FFN_V + FFN_HID*RANK_R;
  parameter int WOFF_FDN_V = WOFF_FDN_U + FFN_HID*RANK_R;
  parameter int WOFF_OUT   = WOFF_FDN_V + HID_D*RANK_R;

  parameter int WGOFF_PATCH = WOFF_PATCH / PE_PAR;
  parameter int WGOFF_Q     = WOFF_Q     / PE_PAR;
  parameter int WGOFF_K     = WOFF_K     / PE_PAR;
  parameter int WGOFF_V     = WOFF_V     / PE_PAR;
  parameter int WGOFF_O     = WOFF_O     / PE_PAR;
  parameter int WGOFF_OUT   = WOFF_OUT   / PE_PAR;
  parameter int WGOFF_FFN_U = WOFF_FFN_U / PE_PAR;
  parameter int WGOFF_FFN_V = WOFF_FFN_V / PE_PAR;
  parameter int WGOFF_FDN_U = WOFF_FDN_U / PE_PAR;
  parameter int WGOFF_FDN_V = WOFF_FDN_V / PE_PAR;

  parameter int COMP_WORD_W  = 32;
  parameter int SLICES_INT4  = COMP_WORD_W / 4;
  parameter int SLICES_TERN  = COMP_WORD_W / 2;

  typedef enum logic [0:0] {MODE_INT4 = 1'b0, MODE_TERNARY = 1'b1} decomp_mode_e;

  parameter decomp_mode_e DEPLOY_MODE = MODE_INT4;
  parameter int DEPLOY_SLICES_PER_WORD = (DEPLOY_MODE == MODE_INT4) ? SLICES_INT4 : SLICES_TERN;
  parameter int COMP_WORDS_PER_LAYER = (WCACHE_DEPTH + DEPLOY_SLICES_PER_WORD - 1) / DEPLOY_SLICES_PER_WORD;
  parameter int COMP_WORDS_TOTAL     = COMP_WORDS_PER_LAYER * DEPTH;
  parameter int COMP_WORD_AW         = $clog2(COMP_WORDS_TOTAL <= 1 ? 2 : COMP_WORDS_TOTAL);
  parameter int COMP_LAYER_WORD_AW   = $clog2(COMP_WORDS_PER_LAYER <= 1 ? 2 : COMP_WORDS_PER_LAYER);

  function automatic int8_t requantize(input acc_t val, input int unsigned shift);
    acc_t shifted;
    begin
      shifted = val >>> shift;
      if (shifted > 32'sd127)
        requantize = 8'sd127;
      else if (shifted < -32'sd128)
        requantize = -8'sd128;
      else
        requantize = int8_t'(shifted[7:0]);
    end
  endfunction

  function automatic int8_t sat_add8(input int8_t a, input int8_t b);
    logic signed [DATA_W:0] sum9;
    begin
      sum9 = {a[DATA_W-1], a} + {b[DATA_W-1], b};
      if (sum9 > 9'sd127)
        sat_add8 = 8'sd127;
      else if (sum9 < -9'sd128)
        sat_add8 = -8'sd128;
      else
        sat_add8 = int8_t'(sum9[DATA_W-1:0]);
    end
  endfunction

  function automatic acc_t dot_slice(
    input logic [HID_D*DATA_W-1:0] a_word,
    input logic [HID_D*DATA_W-1:0] b_word,
    input int base,
    input int width
  );
    acc_t sum;
    int i;
    int8_t av, bv;
    begin
      sum = '0;
      for (i = 0; i < width; i = i + 1) begin
        av  = int8_t'(a_word[(base+i)*DATA_W +: DATA_W]);
        bv  = int8_t'(b_word[(base+i)*DATA_W +: DATA_W]);
        sum = sum + acc_t'(av) * acc_t'(bv);
      end
      dot_slice = sum;
    end
  endfunction

endpackage : trident_pkg
