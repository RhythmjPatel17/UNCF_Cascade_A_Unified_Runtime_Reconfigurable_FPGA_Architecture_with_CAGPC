import trident_pkg::*;

module mac_pe (
  input  logic       clk,
  input  logic       rst_n,

  input  logic       clear,
  input  logic       en,
  input  int8_t      a,
  input  int8_t      b,

  output acc_t       acc
);

  acc_t acc_q;
  acc_t product_ext;

  assign product_ext = acc_t'(a) * acc_t'(b);

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n)       acc_q <= '0;
    else if (clear)   acc_q <= '0;
    else if (en)      acc_q <= acc_q + product_ext;
  end

  assign acc = acc_q;

endmodule : mac_pe
