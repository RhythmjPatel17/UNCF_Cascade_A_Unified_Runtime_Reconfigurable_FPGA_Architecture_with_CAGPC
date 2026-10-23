import trident_pkg::*;

module global_scheduler (
  input  logic clk,
  input  logic rst_n,

  input  logic start,
  output logic busy,
  output logic done,

  output logic dma_start,   input logic dma_done,
  output logic recon_start, input logic recon_done,
  output logic patch_start, input logic patch_done,
  output logic ln0_start,    input logic ln0_done,
  output logic qproj_start, input logic qproj_done,
  output logic kproj_start, input logic kproj_done,
  output logic vproj_start, input logic vproj_done,
  output logic spatial_start, input logic spatial_done,
  output logic temporal_start, input logic temporal_done,
  output logic oproj_start,  input logic oproj_done,
  output logic resid1_start, input logic resid1_done,
  output logic ln1_start,    input logic ln1_done,
  output logic ffnu1_start,  input logic ffnu1_done,
  output logic ffnu2_start,  input logic ffnu2_done,
  output logic gelu_start,   input logic gelu_done,
  output logic ffnd1_start,  input logic ffnd1_done,
  output logic ffnd2_start,  input logic ffnd2_done,
  output logic resid2_start, input logic resid2_done,
  output logic outp_start,   input logic outp_done,
  output logic dupd_start,   input logic dupd_done,

  output logic [$clog2(DEPTH<=1?2:DEPTH)-1:0]     layer_idx,
  output logic [$clog2(T_STEPS<=1?2:T_STEPS)-1:0]  step_idx
);

  localparam int LAW = $clog2(DEPTH   <= 1 ? 2 : DEPTH);
  localparam int SAW = $clog2(T_STEPS <= 1 ? 2 : T_STEPS);

  typedef enum logic [5:0] {
    S_IDLE, S_BOOT_GO, S_BOOT_WAIT,
    S_RECON_GO, S_RECON_WAIT,
    S_PATCH_GO, S_PATCH_WAIT,
    S_LN0_GO, S_LN0_WAIT,
    S_QPROJ_GO, S_QPROJ_WAIT,
    S_KPROJ_GO, S_KPROJ_WAIT,
    S_VPROJ_GO, S_VPROJ_WAIT,
    S_SPATIAL_GO, S_SPATIAL_WAIT,
    S_TEMPORAL_GO, S_TEMPORAL_WAIT,
    S_OPROJ_GO, S_OPROJ_WAIT,
    S_RESID1_GO, S_RESID1_WAIT,
    S_LN1_GO, S_LN1_WAIT,
    S_FFNU1_GO, S_FFNU1_WAIT,
    S_FFNU2_GO, S_FFNU2_WAIT,
    S_GELU_GO, S_GELU_WAIT,
    S_FFND1_GO, S_FFND1_WAIT,
    S_FFND2_GO, S_FFND2_WAIT,
    S_RESID2_GO, S_RESID2_WAIT,
    S_NEXT_LAYER,
    S_OUTP_GO, S_OUTP_WAIT,
    S_DUPD_GO, S_DUPD_WAIT,
    S_NEXT_STEP,
    S_DONE
  } state_e;
  state_e state, state_n;

  logic [LAW-1:0] layer_cnt, layer_cnt_n;
  logic [SAW-1:0] step_cnt,  step_cnt_n;

  assign layer_idx = layer_cnt;
  assign step_idx  = step_cnt;

  always_comb begin
    state_n     = state;
    layer_cnt_n = layer_cnt;
    step_cnt_n  = step_cnt;

    dma_start      = 1'b0;
    recon_start    = 1'b0;
    patch_start    = 1'b0;
    ln0_start      = 1'b0;
    qproj_start    = 1'b0;
    kproj_start    = 1'b0;
    vproj_start    = 1'b0;
    spatial_start  = 1'b0;
    temporal_start = 1'b0;
    oproj_start    = 1'b0;
    resid1_start   = 1'b0;
    ln1_start      = 1'b0;
    ffnu1_start    = 1'b0;
    ffnu2_start    = 1'b0;
    gelu_start     = 1'b0;
    ffnd1_start    = 1'b0;
    ffnd2_start    = 1'b0;
    resid2_start   = 1'b0;
    outp_start     = 1'b0;
    dupd_start     = 1'b0;

    busy = (state != S_IDLE) && (state != S_DONE);
    done = 1'b0;

    unique case (state)
      S_IDLE: begin
        if (start) begin
          step_cnt_n  = '0;
          layer_cnt_n = '0;
          state_n     = S_BOOT_GO;
        end
      end

      S_BOOT_GO:   begin dma_start = 1'b1; state_n = S_BOOT_WAIT; end
      S_BOOT_WAIT: if (dma_done) state_n = S_RECON_GO;

      S_RECON_GO:   begin recon_start = 1'b1; state_n = S_RECON_WAIT; end
      S_RECON_WAIT: if (recon_done) state_n = (layer_cnt == LAW'(0)) ? S_PATCH_GO : S_LN0_GO;

      S_PATCH_GO:   begin patch_start = 1'b1; state_n = S_PATCH_WAIT; end
      S_PATCH_WAIT: if (patch_done) state_n = S_LN0_GO;

      S_LN0_GO:   begin ln0_start = 1'b1; state_n = S_LN0_WAIT; end
      S_LN0_WAIT: if (ln0_done) state_n = S_QPROJ_GO;

      S_QPROJ_GO:   begin qproj_start = 1'b1; state_n = S_QPROJ_WAIT; end
      S_QPROJ_WAIT: if (qproj_done) state_n = S_KPROJ_GO;

      S_KPROJ_GO:   begin kproj_start = 1'b1; state_n = S_KPROJ_WAIT; end
      S_KPROJ_WAIT: if (kproj_done) state_n = S_VPROJ_GO;

      S_VPROJ_GO:   begin vproj_start = 1'b1; state_n = S_VPROJ_WAIT; end
      S_VPROJ_WAIT: if (vproj_done) state_n = S_SPATIAL_GO;

      S_SPATIAL_GO:   begin spatial_start = 1'b1; state_n = S_SPATIAL_WAIT; end
      S_SPATIAL_WAIT: if (spatial_done) state_n = S_TEMPORAL_GO;

      S_TEMPORAL_GO:   begin temporal_start = 1'b1; state_n = S_TEMPORAL_WAIT; end
      S_TEMPORAL_WAIT: if (temporal_done) state_n = S_OPROJ_GO;

      S_OPROJ_GO:   begin oproj_start = 1'b1; state_n = S_OPROJ_WAIT; end
      S_OPROJ_WAIT: if (oproj_done) state_n = S_RESID1_GO;

      S_RESID1_GO:   begin resid1_start = 1'b1; state_n = S_RESID1_WAIT; end
      S_RESID1_WAIT: if (resid1_done) state_n = S_LN1_GO;

      S_LN1_GO:   begin ln1_start = 1'b1; state_n = S_LN1_WAIT; end
      S_LN1_WAIT: if (ln1_done) state_n = S_FFNU1_GO;

      S_FFNU1_GO:   begin ffnu1_start = 1'b1; state_n = S_FFNU1_WAIT; end
      S_FFNU1_WAIT: if (ffnu1_done) state_n = S_FFNU2_GO;

      S_FFNU2_GO:   begin ffnu2_start = 1'b1; state_n = S_FFNU2_WAIT; end
      S_FFNU2_WAIT: if (ffnu2_done) state_n = S_GELU_GO;

      S_GELU_GO:   begin gelu_start = 1'b1; state_n = S_GELU_WAIT; end
      S_GELU_WAIT: if (gelu_done) state_n = S_FFND1_GO;

      S_FFND1_GO:   begin ffnd1_start = 1'b1; state_n = S_FFND1_WAIT; end
      S_FFND1_WAIT: if (ffnd1_done) state_n = S_FFND2_GO;

      S_FFND2_GO:   begin ffnd2_start = 1'b1; state_n = S_FFND2_WAIT; end
      S_FFND2_WAIT: if (ffnd2_done) state_n = S_RESID2_GO;

      S_RESID2_GO:   begin resid2_start = 1'b1; state_n = S_RESID2_WAIT; end
      S_RESID2_WAIT: if (resid2_done) state_n = S_NEXT_LAYER;

      S_NEXT_LAYER: begin
        if (layer_cnt != LAW'(DEPTH-1)) begin
          layer_cnt_n = layer_cnt + 1'b1;
          state_n     = S_RECON_GO;
        end else begin
          state_n = S_OUTP_GO;
        end
      end

      S_OUTP_GO:   begin outp_start = 1'b1; state_n = S_OUTP_WAIT; end
      S_OUTP_WAIT: if (outp_done) state_n = S_DUPD_GO;

      S_DUPD_GO:   begin dupd_start = 1'b1; state_n = S_DUPD_WAIT; end
      S_DUPD_WAIT: if (dupd_done) state_n = S_NEXT_STEP;

      S_NEXT_STEP: begin
        layer_cnt_n = '0;
        if (step_cnt != SAW'(T_STEPS-1)) begin
          step_cnt_n = step_cnt + 1'b1;
          state_n    = S_RECON_GO;
        end else begin
          state_n = S_DONE;
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
      layer_cnt <= '0;
      step_cnt  <= '0;
    end else begin
      state     <= state_n;
      layer_cnt <= layer_cnt_n;
      step_cnt  <= step_cnt_n;
    end
  end

endmodule : global_scheduler
