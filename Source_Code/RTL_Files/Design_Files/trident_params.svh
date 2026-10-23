`ifndef TRIDENT_PARAMS_SVH
`define TRIDENT_PARAMS_SVH

`ifdef TRIDENT_SMOKE_SCALE
  `define TRIDENT_N_TOK      16
  `define TRIDENT_HID_D      8
  `define TRIDENT_N_HEADS    2
  `define TRIDENT_WIN_W      2
  `define TRIDENT_TOPK_K     4
  `define TRIDENT_DEPTH      2
  `define TRIDENT_MLP_RATIO  2
  `define TRIDENT_RANK_R     4
  `define TRIDENT_T_STEPS    2
  `define TRIDENT_PE_PAR     4
`elsif TRIDENT_MEASURE_SCALE
  `define TRIDENT_N_TOK      2304
  `define TRIDENT_HID_D      384
  `define TRIDENT_N_HEADS    6
  `define TRIDENT_WIN_W      4
  `define TRIDENT_TOPK_K     32
  `define TRIDENT_DEPTH      1
  `define TRIDENT_MLP_RATIO  4
  `define TRIDENT_RANK_R     64
  `define TRIDENT_T_STEPS    1
  `define TRIDENT_PE_PAR     64
`else
  `define TRIDENT_N_TOK      2304
  `define TRIDENT_HID_D      384
  `define TRIDENT_N_HEADS    6
  `define TRIDENT_WIN_W      4
  `define TRIDENT_TOPK_K     32
  `define TRIDENT_DEPTH      12
  `define TRIDENT_MLP_RATIO  4
  `define TRIDENT_RANK_R     64
  `define TRIDENT_T_STEPS    50
  `define TRIDENT_PE_PAR     64
`endif

`endif
