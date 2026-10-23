import trident_pkg::*;

module weight_cache (
  input  logic                        clk,

  input  logic                        we,
  input  logic [WCACHE_AW-1:0]        waddr,
  input  logic [DATA_W-1:0]           wdata,

  input  logic                        re,
  input  logic [WCACHE_GAW-1:0]       raddr,
  output logic [PE_PAR*DATA_W-1:0]    rdata
);

  genvar gi;
  generate
    for (gi = 0; gi < PE_PAR; gi++) begin : g_bank
      logic bank_we;
      assign bank_we = we && (waddr[PE_PAR_AW-1:0] == gi[PE_PAR_AW-1:0]);

      bram_sdp #(
        .DATA_W (DATA_W),
        .DEPTH  (WCACHE_GDEPTH),
        .ADDR_W (WCACHE_GAW)
      ) u_bank (
        .clk   (clk),
        .we    (bank_we),
        .waddr (waddr[WCACHE_AW-1:PE_PAR_AW]),
        .wdata (wdata),
        .re    (re),
        .raddr (raddr),
        .rdata (rdata[gi*DATA_W +: DATA_W])
      );
    end
  endgenerate

endmodule : weight_cache
