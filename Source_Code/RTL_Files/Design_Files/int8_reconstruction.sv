import trident_pkg::*;

module int8_reconstruction (
  input  logic                    clk,
  input  logic                    rst_n,

  input  logic                    start,
  input  decomp_mode_e            mode,
  output logic                    done,

  output logic                    mem_req,
  output logic [31:0]             mem_addr,
  input  logic                    mem_valid,
  input  logic [COMP_WORD_W-1:0]  mem_rdata,

  output logic                    wc_we,
  output logic [WCACHE_AW-1:0]    wc_waddr,
  output logic [DATA_W-1:0]       wc_wdata
);

  typedef enum logic [2:0] {S_IDLE, S_REQ, S_WAIT, S_SLICE, S_DONE} state_e;
  state_e state, state_n;

  logic [31:0]            word_cnt, word_cnt_n;
  logic [WCACHE_AW-1:0]   out_cnt,  out_cnt_n;
  logic [3:0]             slice_idx, slice_idx_n;
  logic [3:0]             slices_per_word;
  logic [COMP_WORD_W-1:0] word_reg, word_reg_n;

  assign slices_per_word = (mode == MODE_INT4) ? SLICES_INT4[3:0] : SLICES_TERN[3:0];

  int8_t decoded_operand;
  bitslice_decoder u_decoder (
    .packed_word (word_reg),
    .slice_idx   (slice_idx),
    .mode        (mode),
    .operand     (decoded_operand)
  );

  always_comb begin
    state_n     = state;
    word_cnt_n  = word_cnt;
    out_cnt_n   = out_cnt;
    slice_idx_n = slice_idx;
    word_reg_n  = word_reg;

    mem_req  = 1'b0;
    mem_addr = word_cnt << 2;
    wc_we    = 1'b0;
    wc_waddr = out_cnt;
    wc_wdata = decoded_operand;
    done     = 1'b0;

    unique case (state)
      S_IDLE: begin
        if (start) begin
          state_n    = S_REQ;
          word_cnt_n = '0;
          out_cnt_n  = '0;
        end
      end

      S_REQ: begin
        mem_req = 1'b1;
        state_n = S_WAIT;
      end

      S_WAIT: begin
        if (mem_valid) begin
          word_reg_n  = mem_rdata;
          slice_idx_n = '0;
          state_n     = S_SLICE;
        end
      end

      S_SLICE: begin
        wc_we    = 1'b1;
        wc_waddr = out_cnt;
        wc_wdata = decoded_operand;
        out_cnt_n = out_cnt + 1'b1;

        if ((slice_idx == slices_per_word - 1) ||
            (out_cnt   == WCACHE_AW'(WCACHE_DEPTH - 1))) begin

          if (out_cnt == WCACHE_AW'(WCACHE_DEPTH - 1)) begin
            state_n = S_DONE;
          end else begin
            word_cnt_n = word_cnt + 1'b1;
            state_n    = S_REQ;
          end
        end else begin
          slice_idx_n = slice_idx + 1'b1;
        end
      end

      S_DONE: begin
        done = 1'b1;
        if (!start) state_n = S_IDLE;
      end

      default: state_n = S_IDLE;
    endcase
  end

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      state     <= S_IDLE;
      word_cnt  <= '0;
      out_cnt   <= '0;
      slice_idx <= '0;
      word_reg  <= '0;
    end else begin
      state     <= state_n;
      word_cnt  <= word_cnt_n;
      out_cnt   <= out_cnt_n;
      slice_idx <= slice_idx_n;
      word_reg  <= word_reg_n;
    end
  end

endmodule : int8_reconstruction
