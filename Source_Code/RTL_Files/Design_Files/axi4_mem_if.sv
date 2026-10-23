module axi4_mem_if #(
  parameter int ADDR_W = 32,
  parameter int DATA_W = 32
)(
  input  logic                  clk,
  input  logic                  rst_n,

  input  logic                  req,
  input  logic [ADDR_W-1:0]     req_addr,
  output logic                  rvalid,
  output logic [DATA_W-1:0]     rdata,

  output logic                  m_axi_arvalid,
  input  logic                  m_axi_arready,
  output logic [ADDR_W-1:0]     m_axi_araddr,
  output logic [7:0]            m_axi_arlen,
  output logic [2:0]            m_axi_arsize,
  output logic [1:0]            m_axi_arburst,

  input  logic                  m_axi_rvalid,
  output logic                  m_axi_rready,
  input  logic [DATA_W-1:0]     m_axi_rdata,
  input  logic                  m_axi_rlast
);

  typedef enum logic [1:0] {S_IDLE, S_ADDR, S_DATA} state_e;
  state_e state, state_n;
  logic [ADDR_W-1:0] addr_reg, addr_reg_n;

  always_comb begin
    state_n    = state;
    addr_reg_n = addr_reg;

    m_axi_arvalid = 1'b0;
    m_axi_araddr  = addr_reg;
    m_axi_rready  = 1'b0;
    rvalid        = 1'b0;
    rdata         = m_axi_rdata;

    unique case (state)
      S_IDLE: begin
        if (req) begin
          addr_reg_n = req_addr;
          state_n    = S_ADDR;
        end
      end

      S_ADDR: begin
        m_axi_arvalid = 1'b1;
        m_axi_araddr  = addr_reg;
        if (m_axi_arready) state_n = S_DATA;
      end

      S_DATA: begin
        m_axi_rready = 1'b1;
        if (m_axi_rvalid) begin
          rvalid  = 1'b1;
          state_n = S_IDLE;
        end
      end

      default: state_n = S_IDLE;
    endcase
  end

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      state    <= S_IDLE;
      addr_reg <= '0;
    end else begin
      state    <= state_n;
      addr_reg <= addr_reg_n;
    end
  end

  assign m_axi_arlen   = 8'd0;
  assign m_axi_arsize  = 3'b010;
  assign m_axi_arburst = 2'b01;

endmodule : axi4_mem_if
