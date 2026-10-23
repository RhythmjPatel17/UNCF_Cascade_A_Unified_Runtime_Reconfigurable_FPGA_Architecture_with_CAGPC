module divider16 (
  input  logic         clk,
  input  logic         rst_n,

  input  logic         start,
  input  logic [15:0]  numer,
  input  logic [15:0]  denom,

  output logic         done,
  output logic [15:0]  quot
);

  typedef enum logic [1:0] {D_IDLE, D_RUN, D_DONE} state_e;
  state_e state, state_n;

  logic [4:0] cnt, cnt_n;
  logic [31:0] acc, acc_n;
  logic [15:0] denom_r, denom_n;

  always_comb begin
    state_n  = state;
    cnt_n    = cnt;
    acc_n    = acc;
    denom_n  = denom_r;
    done     = 1'b0;
    quot     = acc[15:0];

    unique case (state)
      D_IDLE: begin
        if (start) begin
          acc_n   = {16'b0, numer};
          denom_n = (denom == 16'd0) ? 16'd1 : denom;
          cnt_n   = 5'd0;
          state_n = D_RUN;
        end
      end
      D_RUN: begin
        logic [31:0] shifted;
        logic [31:0] trial;
        shifted = acc << 1;
        trial   = shifted;
        if (trial[31:16] >= {16'b0, denom_r}) begin
          trial[31:16] = trial[31:16] - denom_r;
          trial[0]     = 1'b1;
        end
        acc_n = trial;
        if (cnt == 5'd15) begin
          state_n = D_DONE;
        end else begin
          cnt_n = cnt + 1'b1;
        end
      end
      D_DONE: begin
        done = 1'b1;
        quot = acc[15:0];
        if (!start) state_n = D_IDLE;
      end
      default: state_n = D_IDLE;
    endcase
  end

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      state   <= D_IDLE;
      cnt     <= '0;
      acc     <= '0;
      denom_r <= '0;
    end else begin
      state   <= state_n;
      cnt     <= cnt_n;
      acc     <= acc_n;
      denom_r <= denom_n;
    end
  end

endmodule : divider16
