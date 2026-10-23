import trident_pkg::*;

module topk_cache (
  input  logic clk,

  input  logic [$clog2(DEPTH<=1?2:DEPTH)-1:0] layer_idx,

  input  logic                    we,
  input  logic [TOK_AW-1:0]       w_tok,
  input  logic [$clog2(N_HEADS<=1?2:N_HEADS)-1:0] w_head,
  input  logic [$clog2(TOPK_K<=1?2:TOPK_K)-1:0]   w_iter,
  input  logic [TOK_AW-1:0]       w_idx,

  input  logic                    re,
  input  logic [TOK_AW-1:0]       r_tok,
  input  logic [$clog2(N_HEADS<=1?2:N_HEADS)-1:0] r_head,
  input  logic [$clog2(TOPK_K<=1?2:TOPK_K)-1:0]   r_iter,
  output logic [TOK_AW-1:0]       r_idx
);

  localparam int PER_LAYER = N_TOK * N_HEADS * TOPK_K;
  localparam int TOTAL     = DEPTH * PER_LAYER;
  localparam int AW        = $clog2(TOTAL <= 1 ? 2 : TOTAL);

  (* ram_style = "ultra" *) logic [TOK_AW-1:0] mem [0:TOTAL-1];

  logic [AW-1:0] waddr, raddr;
  assign waddr = AW'(layer_idx * PER_LAYER + (w_tok * N_HEADS + w_head) * TOPK_K + w_iter);
  assign raddr = AW'(layer_idx * PER_LAYER + (r_tok * N_HEADS + r_head) * TOPK_K + r_iter);

  always_ff @(posedge clk) begin
    if (we) mem[waddr] <= w_idx;
    if (re) r_idx <= mem[raddr];
  end

endmodule : topk_cache
