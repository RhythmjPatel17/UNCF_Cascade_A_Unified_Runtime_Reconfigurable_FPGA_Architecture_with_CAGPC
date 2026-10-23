import trident_pkg::*;

module compressed_weight_store (
  input  logic                          clk,
  input  logic                          rst_n,

  input  logic                          dma_start,
  output logic                          dma_done,

  output logic                          dma_req,
  output logic [31:0]                   dma_addr,
  input  logic                          dma_valid,
  input  logic [COMP_WORD_W-1:0]        dma_rdata,

  input  logic                          rd_req,
  input  logic [$clog2(DEPTH<=1?2:DEPTH)-1:0] rd_layer_idx,
  input  logic [31:0]                   rd_byte_addr,
  output logic                          rd_valid,
  output logic [COMP_WORD_W-1:0]        rd_data
);

  localparam int LAW = $clog2(DEPTH<=1?2:DEPTH);

  (* ram_style = "block" *) logic [COMP_WORD_W-1:0] mem [0:COMP_WORDS_TOTAL-1];

  typedef enum logic [1:0] {D_IDLE, D_REQ, D_WAIT, D_DONE} dstate_e;
  dstate_e dstate, dstate_n;

  logic [COMP_WORD_AW-1:0] word_cnt, word_cnt_n;

  always_comb begin
    dstate_n  = dstate;
    word_cnt_n = word_cnt;
    dma_req   = 1'b0;
    dma_addr  = {word_cnt, 2'b00};
    dma_done  = 1'b0;

    unique case (dstate)
      D_IDLE: begin
        if (dma_start) begin
          word_cnt_n = '0;
          dstate_n   = D_REQ;
        end
      end
      D_REQ: begin
        dma_req  = 1'b1;
        dma_addr = {word_cnt, 2'b00};
        dstate_n = D_WAIT;
      end
      D_WAIT: begin
        if (dma_valid) begin
          if (word_cnt == COMP_WORD_AW'(COMP_WORDS_TOTAL-1)) begin
            dstate_n = D_DONE;
          end else begin
            word_cnt_n = word_cnt + 1'b1;
            dstate_n   = D_REQ;
          end
        end
      end
      D_DONE: begin
        dma_done = 1'b1;
        if (!dma_start) dstate_n = D_IDLE;
      end
      default: dstate_n = D_IDLE;
    endcase
  end

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      dstate   <= D_IDLE;
      word_cnt <= '0;
    end else begin
      dstate   <= dstate_n;
      word_cnt <= word_cnt_n;
      if (dstate == D_WAIT && dma_valid) begin
        mem[word_cnt] <= dma_rdata;
      end
    end
  end

  logic [COMP_LAYER_WORD_AW-1:0] rd_word_idx;
  assign rd_word_idx = rd_byte_addr[COMP_LAYER_WORD_AW+1:2];

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      rd_valid <= 1'b0;
      rd_data  <= '0;
    end else begin
      rd_valid <= rd_req;
      if (rd_req) begin
        rd_data <= mem[rd_layer_idx * COMP_WORDS_PER_LAYER + rd_word_idx];
      end
    end
  end

endmodule : compressed_weight_store
