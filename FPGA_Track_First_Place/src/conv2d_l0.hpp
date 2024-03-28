#ifndef __CONV2D_L0_HPP__
#define __CONV2D_L0_HPP__

#include <ap_int.h>
#include <hls_stream.h>
using namespace hls;

#include "function.h"
#include "stream_tools.h"

template <unsigned IN_W, unsigned IN_BIT>
void stream_in_row_l0(stream<ap_uint<3 * IN_BIT>> &in,
                      ap_uint<3 * IN_BIT> row_buffer[4][IN_W + 1],
                      ap_uint<2> rowBufferIdx) {

  for (unsigned w = 0; w < IN_W + 1; w++) {
#pragma HLS pipeline
    ap_uint<3 * IN_BIT> data;
    if (w != IN_W) {
      data = in.read();
    } else {
      data = 0;
    }
    row_buffer[rowBufferIdx][w] = data;
  }
}

template <unsigned IN_H, unsigned IN_W, unsigned IN_BIT, unsigned OUTPENUM>
void stream_out_data_l0(stream<ap_uint<3 * IN_BIT * 3>> &out,
                        ap_uint<3 * IN_BIT> row_buffer[4][IN_W + 1],
                        ap_int<12> outRowIdx,
                        ap_uint<2> centerRowBufferIdx) {
#pragma HLS array_partition variable = row_buffer dim = 1 complete

  for (unsigned peIdx = 0; peIdx < OUTPENUM; peIdx++)
    for (unsigned c = 0; c < IN_W; c += 2)
      for (unsigned kc = 0; kc < 3; kc++) {
#pragma HLS pipeline
        ap_uint<3 * IN_BIT> data[4];
#pragma HLS array_partition variable = data dim = 1 complete
        for (unsigned i = 0; i < 4; i++) {
          data[i] = row_buffer[i][c + kc];
        }
        ap_uint<2> row_sel0, row_sel1, row_sel2;
        row_sel0 = centerRowBufferIdx - 1;
        row_sel1 = centerRowBufferIdx;
        row_sel2 = centerRowBufferIdx + 1;
        ap_uint<3 * IN_BIT> data0, data1, data2;

        if (outRowIdx == 0)
          data0 = 0;
        else
          data0 = data[row_sel0];
        data1 = data[row_sel1];
        if (outRowIdx == IN_H - 1)
          data2 = 0;
        else
          data2 = data[row_sel2];
        out.write((data2, data1, data0));
      }
}

template <unsigned K, unsigned IN_H, unsigned IN_W, unsigned IN_CH,
          unsigned IN_BIT, unsigned OUTPENUM>
void conv3padding_l0(stream<ap_uint<3 * IN_BIT>> &in,
                     stream<ap_uint<3 * IN_BIT * 3>> &out,
                     const unsigned reps = 1) {
  static_assert(K == 3, "K!=3");
  ap_uint<IN_CH * IN_BIT> row_buffer[4][IN_W + 1];
#pragma HLS ARRAY_PARTITION variable = row_buffer dim = 1 complete
#pragma HLS RESOURCE variable = row_buffer core = RAM_S2P_BRAM

  ap_uint<2> storeBufferIdx = 0;
  ap_uint<2> loadBufferIdx = 0;
  ap_int<10> rowIdx = 0;

  stream_in_row_l0<IN_W, IN_BIT>(in, row_buffer, storeBufferIdx);
  storeBufferIdx++;
  stream_in_row_l0<IN_W, IN_BIT>(in, row_buffer, storeBufferIdx);
  storeBufferIdx++;
  
  for (unsigned rep = 0; rep < reps * IN_H - 2; rep++) {
#pragma HLS dependence intra false variable = row_buffer

    stream_in_row_l0<IN_W, IN_BIT>(in, row_buffer, storeBufferIdx);
    stream_out_data_l0<IN_H, IN_W, IN_BIT, OUTPENUM>(out, row_buffer, rowIdx, loadBufferIdx);
    loadBufferIdx++;
    storeBufferIdx++;
    if (rowIdx == IN_H - 1) {
      rowIdx = 0;
    } else {
      rowIdx++;
    }
  }
  stream_out_data_l0<IN_H, IN_W, IN_BIT, OUTPENUM>(out, row_buffer, rowIdx, loadBufferIdx);
  loadBufferIdx++;
  if (rowIdx == IN_H - 1) {
    rowIdx = 0;
  } else {
    rowIdx++;
  }
  
  stream_out_data_l0<IN_H, IN_W, IN_BIT, OUTPENUM>(out, row_buffer, rowIdx, loadBufferIdx);
  loadBufferIdx++;
  if (rowIdx == IN_H - 1) {
    rowIdx = 0;
  } else {
    rowIdx++;
  }

}

template <unsigned IN_BIT, unsigned W_BIT>
ap_int<IN_BIT + W_BIT> convl0_mul_lut(ap_uint<IN_BIT> in, ap_int<W_BIT> w) {
  ap_int<IN_BIT + W_BIT> out;
#pragma HLS RESOURCE variable=return core=Mul_LUT
#pragma HLS inline off
  out = in * w;
  return out;
}

template <unsigned IN_BIT, unsigned W_BIT, unsigned PROD_BIT>
void simd_mac9_LUT(ap_uint<IN_BIT> invec[9], ap_int<W_BIT> w0vec[9],
                    ap_int<W_BIT> w1vec[9], ap_int<PROD_BIT> &out0,
                    ap_int<PROD_BIT> &out1) {
#pragma HLS array_partition variable = invec
#pragma HLS array_partition variable = w1vec
#pragma HLS array_partition variable = w0vec

  ap_int<PROD_BIT> acc0 = 0;
  ap_int<PROD_BIT> acc1 = 0;

  for (int i = 0; i < 9; i++) {
    ap_int<IN_BIT + W_BIT> m0 = convl0_mul_lut<IN_BIT, W_BIT>(invec[i], w0vec[i]);
    acc0 += m0;
  }

  for (int i = 0; i < 9; i++) {
    ap_int<IN_BIT + W_BIT> m1 = convl0_mul_lut<IN_BIT, W_BIT>(invec[i], w1vec[i]);
    acc1 += m1;
  }

  out0 = acc0;
  out1 = acc1;
}


template <unsigned IN_BIT>
void loadInReg9(ap_uint<IN_BIT * 9> inData, ap_uint<IN_BIT> ivec0[9], ap_uint<IN_BIT> ivec1[9]) {
#pragma HLS pipeline II = 1
#pragma HLS inline off
#pragma HLS ARRAY_PARTITION variable = ivec0 complete dim = 1
#pragma HLS ARRAY_PARTITION variable = ivec1 complete dim = 1

  for (unsigned s = 0; s < 9; s++) {
    ivec0[s] = ivec1[s];
    ivec1[s] = inData((s + 1) * IN_BIT - 1, s * IN_BIT);
  }
}

template <unsigned OUT_ROW, unsigned OUT_COL, unsigned OUT_CH, unsigned PE,
          unsigned IN_BIT, unsigned W_BIT, unsigned M_BIT,
          unsigned OUT_BIT>
void convLUTOpt_l0(stream<ap_uint<IN_BIT * 9>> &in,
                   const ap_uint<3 * W_BIT> weights[PE][3][3 * (OUT_CH / PE)],
                   stream<ap_uint<M_BIT * PE * 2>> &out, const unsigned reps = 1) {
#pragma HLS ARRAY_PARTITION variable = weights complete dim = 1
#pragma HLS ARRAY_PARTITION variable = weights complete dim = 2

  const unsigned PROD_BIT = IN_BIT + W_BIT + 4;

  ap_int<M_BIT> outPartialArr0[PE];
#pragma HLS ARRAY_PARTITION variable = outPartialArr0 complete dim = 1

  ap_int<M_BIT> outPartialArr1[PE];
#pragma HLS ARRAY_PARTITION variable = outPartialArr1 complete dim = 1

  ap_uint<IN_BIT> ivec0[9];
#pragma HLS ARRAY_PARTITION variable = ivec0 complete dim = 1
  ap_uint<IN_BIT> ivec1[9];
#pragma HLS ARRAY_PARTITION variable = ivec1 complete dim = 1
  for (int i = 0; i < 9; i++){
#pragma HLS unroll
    ivec1[i] = 0;
  }
  for (unsigned int h = 0; h < OUT_ROW * reps; h++) {
    for (unsigned peIdx = 0; peIdx < OUT_CH / PE; peIdx++)
      for (unsigned int w = 0; w < OUT_COL / 2; w++) {
        for (unsigned int kc = 0; kc < 3; kc++) {
#pragma HLS pipeline II = 1

          ap_int<W_BIT> wvec[PE][9];
#pragma HLS ARRAY_PARTITION variable = wvec complete dim = 1
#pragma HLS ARRAY_PARTITION variable = wvec complete dim = 2
          
          ap_uint<IN_BIT * 9> inData;

          inData = in.read();
          if (w == 0 || kc != 0){ 
            loadInReg9<IN_BIT>(inData, ivec0, ivec1);
          }

          for (int i = 0; i < PE; i++) {
            for (int s = 0; s < 9; s++) {
              wvec[i][s] = weights[i][s / 3][peIdx * 3 + kc](
                  (s % 3 + 1) * W_BIT - 1, s % 3 * W_BIT);
            }
          }

          for (int p = 0; p < PE; p += 2) {
#pragma HLS unroll 
            ap_int<PROD_BIT> outPartial00;
            ap_int<PROD_BIT> outPartial01;

            ap_int<PROD_BIT> outPartial10;
            ap_int<PROD_BIT> outPartial11;

            simd_mac9_LUT<IN_BIT, W_BIT, PROD_BIT>(ivec0, wvec[p], wvec[p + 1],
                                                    outPartial00, outPartial01);
            simd_mac9_LUT<IN_BIT, W_BIT, PROD_BIT>(ivec1, wvec[p], wvec[p + 1],
                                                    outPartial10, outPartial11);

            if (kc == 0) {
              outPartialArr0[p] = outPartial00;
              outPartialArr0[p + 1] = outPartial01;
            } else {
              outPartialArr0[p] += outPartial00;
              outPartialArr0[p + 1] += outPartial01;
            }
            
            if (kc == 0) {
              outPartialArr1[p] = outPartial10;
              outPartialArr1[p + 1] = outPartial11;
            } else {
              outPartialArr1[p] += outPartial10;
              outPartialArr1[p + 1] += outPartial11;
            }
          }
          ap_uint<M_BIT * PE> odata0, odata1;
          if (kc == 2) {
            for (int i = 0; i < PE; i++) {
              odata0((i + 1) * M_BIT - 1, i * M_BIT) = outPartialArr0[i];
            }

            for (int i = 0; i < PE; i++) {
              odata1((i + 1) * M_BIT - 1, i * M_BIT) = outPartialArr1[i];
            }

            out.write((odata1, odata0));
          }
        }
      }

  }
}

template <unsigned OUT_ROW, unsigned OUT_COL, unsigned OUT_CH, unsigned M_BIT,
          unsigned OUT_BIT, unsigned INC_BIT, unsigned BIAS_BIT,
          unsigned IN_BIT, unsigned W_BIT, unsigned L_SHIFT, unsigned PE>
void streamRelu_l0(stream<ap_uint<PE * M_BIT * 2>> &in,
                     const ap_int<INC_BIT> inc[PE][OUT_CH / PE],
                     const ap_int<BIAS_BIT> bias[PE][OUT_CH / PE],
                     stream<ap_uint<PE * OUT_BIT * 2>> &out,
                     const unsigned rep = 1) {
#pragma HLS ARRAY_PARTITION variable = inc complete dim = 1
#pragma HLS ARRAY_PARTITION variable = bias complete dim = 1

  for (unsigned r = 0; r < OUT_ROW * rep; r++)
    for (unsigned peIdx = 0; peIdx < OUT_CH / PE; peIdx++)
      for (unsigned w = 0; w < OUT_COL / 2; w++) {
#pragma HLS pipeline II = 3
        ap_uint<M_BIT * PE * 2> data_in;
        ap_uint<OUT_BIT * PE * 2> data_out;
        ap_int<M_BIT> invec[2 * PE];
#pragma HLS array_partition variable = invec dim = 1 complete
        data_in = in.read();
        for (int i = 0; i < PE * 2; i++) {
          invec[i] = data_in((i + 1) * M_BIT - 1, i * M_BIT);
        }
        for (int i = 0; i < PE * 2; i++) {
          data_out((i + 1) * OUT_BIT - 1, i * OUT_BIT) =
              qurelu_fixed<M_BIT, OUT_BIT, INC_BIT, BIAS_BIT, IN_BIT, W_BIT,
                              L_SHIFT>(invec[i], inc[i % PE][peIdx], bias[i % PE][peIdx]);
        }

        out.write(data_out);
      }
}

template <unsigned OUT_ROW, unsigned OUT_COL, unsigned OUT_CH, unsigned M_BIT,
          unsigned OUT_BIT, unsigned INC_BIT, unsigned BIAS_BIT,
          unsigned IN_BIT, unsigned W_BIT, unsigned L_SHIFT, unsigned PE>
void streamBnRelu_l0(stream<ap_uint<PE * M_BIT * 2>> &in,
                     const ap_int<INC_BIT> inc[PE][OUT_CH / PE],
                     const ap_int<BIAS_BIT> bias[PE][OUT_CH / PE],
                     stream<ap_uint<PE * OUT_BIT * 2>> &out,
                     const unsigned rep = 1) {
#pragma HLS ARRAY_PARTITION variable = inc complete dim = 1
#pragma HLS ARRAY_PARTITION variable = bias complete dim = 1

  for (unsigned r = 0; r < OUT_ROW * rep; r++)
    for (unsigned peIdx = 0; peIdx < OUT_CH / PE; peIdx++)
      for (unsigned w = 0; w < OUT_COL / 2; w++) {
#pragma HLS pipeline II = 3
        ap_uint<M_BIT * PE * 2> data_in;
        ap_uint<OUT_BIT * PE * 2> data_out;
        ap_int<M_BIT> invec[2 * PE];
#pragma HLS array_partition variable = invec dim = 1 complete
        data_in = in.read();
        for (int i = 0; i < PE * 2; i++) {
          invec[i] = data_in((i + 1) * M_BIT - 1, i * M_BIT);
        }
        for (int i = 0; i < PE * 2; i++) {
          data_out((i + 1) * OUT_BIT - 1, i * OUT_BIT) =
              bn_qurelu_fixed<M_BIT, OUT_BIT, INC_BIT, BIAS_BIT, IN_BIT, W_BIT,
                              L_SHIFT>(invec[i], inc[i % PE][peIdx], bias[i % PE][peIdx]);
        }

        out.write(data_out);
      }
}


template <unsigned IN_ROW, unsigned IN_COL, unsigned IN_CH, unsigned IN_BIT,

          unsigned OUT_CH,
          unsigned OUT_BIT, 

          unsigned W_BIT, unsigned M_BIT, unsigned INC_BIT, unsigned BIAS_BIT,

          unsigned SIMD, unsigned CASCADE, unsigned IN_PE, unsigned PE,
          unsigned L_SHIFT>
void conv3x3_l0_bn_act_LUTopt(
    stream<ap_uint<IN_BIT * IN_CH>> &in,
    const ap_uint<IN_CH * W_BIT> weights[PE][3][3 * (OUT_CH / PE)],
    const ap_int<INC_BIT> inc[PE][OUT_CH / PE],
    const ap_int<BIAS_BIT> bias[PE][OUT_CH / PE],

    stream<ap_uint<OUT_BIT * PE * 2>> &out, const unsigned reps = 1) {
#pragma HLS DATAFLOW

  const unsigned OUT_ROW = IN_ROW;
  const unsigned OUT_COL = IN_COL;

  stream<ap_uint<SIMD * IN_BIT * 3>> padding_out("pad_l0_out");
  conv3padding_l0<3, IN_ROW, IN_COL, IN_CH, IN_BIT, OUT_CH / PE>(
      in, padding_out, reps);

  stream<ap_uint<M_BIT * PE * 2>> conv_l0_out("conv_l0_out");
  convLUTOpt_l0<OUT_ROW, OUT_COL, OUT_CH, PE, IN_BIT, W_BIT, M_BIT, OUT_BIT>(
      padding_out, weights, conv_l0_out, reps);

  streamBnRelu_l0<OUT_ROW, OUT_COL, OUT_CH, M_BIT, OUT_BIT, INC_BIT, BIAS_BIT,
                  L_SHIFT, IN_BIT, W_BIT, PE>(conv_l0_out, inc, bias, out,
                                              reps);
}

#endif
