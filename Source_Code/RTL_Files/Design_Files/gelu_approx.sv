import trident_pkg::*;

module gelu_approx (
  input  logic              clk,
  input  logic               re,
  input  logic [DATA_W-1:0]  in_bits,
  output logic [DATA_W-1:0]  out_bits
);

  (* rom_style = "block" *) logic [DATA_W-1:0] rom [0:255];

  initial begin
    rom = '{
    8'h00, 8'h01, 8'h01, 8'h02, 8'h02, 8'h03, 8'h04, 8'h05, 8'h06, 8'h06, 8'h07, 8'h08, 8'h09, 8'h0a, 8'h0b, 8'h0c,
    8'h0d, 8'h0f, 8'h10, 8'h11, 8'h12, 8'h13, 8'h14, 8'h15, 8'h16, 8'h18, 8'h19, 8'h1a, 8'h1b, 8'h1c, 8'h1d, 8'h1e,
    8'h1f, 8'h20, 8'h21, 8'h23, 8'h24, 8'h25, 8'h26, 8'h27, 8'h28, 8'h29, 8'h2a, 8'h2b, 8'h2c, 8'h2d, 8'h2e, 8'h2f,
    8'h30, 8'h31, 8'h32, 8'h33, 8'h34, 8'h35, 8'h36, 8'h37, 8'h38, 8'h39, 8'h3a, 8'h3b, 8'h3c, 8'h3d, 8'h3e, 8'h3f,
    8'h40, 8'h41, 8'h42, 8'h43, 8'h44, 8'h45, 8'h46, 8'h47, 8'h48, 8'h49, 8'h4a, 8'h4b, 8'h4c, 8'h4d, 8'h4e, 8'h4f,
    8'h50, 8'h51, 8'h52, 8'h53, 8'h54, 8'h55, 8'h56, 8'h57, 8'h58, 8'h59, 8'h5a, 8'h5b, 8'h5c, 8'h5d, 8'h5e, 8'h5f,
    8'h60, 8'h61, 8'h62, 8'h63, 8'h64, 8'h65, 8'h66, 8'h67, 8'h68, 8'h69, 8'h6a, 8'h6b, 8'h6c, 8'h6d, 8'h6e, 8'h6f,
    8'h70, 8'h71, 8'h72, 8'h73, 8'h74, 8'h75, 8'h76, 8'h77, 8'h78, 8'h79, 8'h7a, 8'h7b, 8'h7c, 8'h7d, 8'h7e, 8'h7f,
    8'h00, 8'h00, 8'h00, 8'h00, 8'h00, 8'h00, 8'h00, 8'h00, 8'h00, 8'h00, 8'h00, 8'h00, 8'h00, 8'h00, 8'h00, 8'h00,
    8'h00, 8'h00, 8'h00, 8'h00, 8'h00, 8'h00, 8'h00, 8'h00, 8'h00, 8'h00, 8'h00, 8'h00, 8'h00, 8'h00, 8'h00, 8'h00,
    8'h00, 8'h00, 8'h00, 8'h00, 8'h00, 8'h00, 8'h00, 8'h00, 8'h00, 8'h00, 8'h00, 8'h00, 8'h00, 8'h00, 8'h00, 8'h00,
    8'h00, 8'h00, 8'h00, 8'h00, 8'h00, 8'h00, 8'h00, 8'h00, 8'h00, 8'h00, 8'h00, 8'h00, 8'h00, 8'h00, 8'h00, 8'h00,
    8'h00, 8'h00, 8'h00, 8'h00, 8'h00, 8'h00, 8'h00, 8'h00, 8'h00, 8'h00, 8'h00, 8'h00, 8'h00, 8'h00, 8'h00, 8'h00,
    8'h00, 8'h00, 8'h00, 8'h00, 8'h00, 8'h00, 8'h00, 8'h00, 8'h00, 8'h00, 8'h00, 8'h00, 8'h00, 8'h00, 8'hff, 8'hff,
    8'hff, 8'hff, 8'hff, 8'hff, 8'hff, 8'hff, 8'hff, 8'hff, 8'hfe, 8'hfe, 8'hfe, 8'hfe, 8'hfe, 8'hfe, 8'hfe, 8'hfe,
    8'hfd, 8'hfd, 8'hfd, 8'hfd, 8'hfd, 8'hfd, 8'hfd, 8'hfd, 8'hfe, 8'hfe, 8'hfe, 8'hfe, 8'hfe, 8'hff, 8'hff, 8'h00
    };
  end

  always_ff @(posedge clk) begin
    if (re) out_bits <= rom[in_bits];
  end

endmodule : gelu_approx
