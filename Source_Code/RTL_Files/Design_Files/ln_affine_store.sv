import trident_pkg::*;

module ln_affine_store #(
  parameter string GAMMA_FILE = "ln_gamma_init.hex",
  parameter string BETA_FILE  = "ln_beta_init.hex"
)(
  input  logic                    clk,

  input  logic [$clog2(DEPTH<=1?2:DEPTH)-1:0] layer_idx,

  input  logic                    gb_re,
  input  logic [$clog2(HID_D<=1?2:HID_D)-1:0] gb_raddr,
  output logic [15:0]             gb_rdata
);

  localparam int TOTAL = DEPTH * HID_D;
  localparam int D_AW  = $clog2(HID_D <= 1 ? 2 : HID_D);
  localparam int F_AW  = $clog2(TOTAL <= 1 ? 2 : TOTAL);

  (* rom_style = "block" *) logic [7:0] gamma_mem [0:TOTAL-1];
  (* rom_style = "block" *) logic [7:0] beta_mem  [0:TOTAL-1];

  initial begin
    for (int i = 0; i < TOTAL; i++) begin
      gamma_mem[i] = 8'sd8;
      beta_mem[i]  = 8'sd0;
    end
    $readmemh(GAMMA_FILE, gamma_mem);
    $readmemh(BETA_FILE, beta_mem);
  end

  logic [F_AW-1:0] flat_addr;
  assign flat_addr = F_AW'(layer_idx * HID_D + gb_raddr);

  always_ff @(posedge clk) begin
    if (gb_re) begin
      gb_rdata <= {beta_mem[flat_addr], gamma_mem[flat_addr]};
    end
  end

endmodule : ln_affine_store
