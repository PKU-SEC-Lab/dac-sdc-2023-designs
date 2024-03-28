#ifndef __CONV2D_HPP__
#define __CONV2D_HPP__

#include <ap_int.h>
#include <hls_stream.h>
using namespace hls;

#include "function.h"
#include "stream_tools.h"

#define CEILDIV(x, y) (((x) + (y)-1) / (y))

template <unsigned IN_W, unsigned IN_CH, unsigned IN_BIT, unsigned IN_PE,
          unsigned SIMD>
void stream_in_row(
    stream<ap_uint<IN_PE * IN_BIT * 2>> &in,
    ap_uint<IN_PE * IN_BIT * 2> row_buffer[SIMD / IN_PE][4]
                                          [IN_W / 2 * IN_CH / SIMD],
    ap_uint<2> rowBufferIdx) {

  for (ap_uint<7> peIdx = 0; peIdx < IN_CH / IN_PE; peIdx++)
    for (ap_uint<9> w = 0; w < IN_W / 2; w++) {
#pragma HLS pipeline
      ap_uint<IN_PE * IN_BIT * 2> data;
      ap_uint<IN_PE * IN_BIT> data0, data1;
      data = in.read();
      row_buffer[peIdx % (SIMD / IN_PE)][rowBufferIdx]
                [w * IN_CH / SIMD + peIdx / (SIMD / IN_PE)] = data;
    }
}


template <unsigned K, unsigned IN_H, unsigned IN_W, unsigned IN_CH,
          unsigned IN_BIT, unsigned IN_PE, unsigned SIMD, unsigned OUTPENUM>
void stream_out_data(
    stream<ap_uint<SIMD * IN_BIT * 2>> &out,
    ap_uint<IN_PE * IN_BIT * 2> row_buffer[SIMD / IN_PE][4]
                                          [IN_W / 2 * IN_CH / SIMD],
    ap_int<12> outRowIdx, ap_uint<2> startRowBufferIdx) {
#pragma HLS array_partition variable = row_buffer dim = 1 complete

  const unsigned IN_PE_BIT = IN_PE * IN_BIT;
  const unsigned SIMDNUM = IN_CH / SIMD;
  const unsigned WLEN = IN_W / 2;

  ap_uint<4> infoldIdx = 0;
  ap_uint<8> w = 0;

  for (ap_uint<7> peIdx = 0; peIdx < OUTPENUM; peIdx++) {
    for (ap_uint<12> cycle = 0; cycle < WLEN * K * SIMDNUM; cycle++) {
      ap_uint<2> wr = infoldIdx / SIMDNUM;
      ap_uint<6> simdIdx = infoldIdx % SIMDNUM;
#pragma HLS pipeline
      ap_uint<SIMD * IN_BIT> data0;
      ap_uint<SIMD * IN_BIT> data1;
      ap_uint<IN_PE * IN_BIT * 2> buffer_data[SIMD / IN_PE];
#pragma HLS array_partition variable = buffer_data complete
      ap_uint<2> rowBufferIdx = startRowBufferIdx + wr;
      for (int i = 0; i < SIMD / IN_PE; i++) {
#pragma HLS unroll
        buffer_data[i] = row_buffer[i][rowBufferIdx][w * SIMDNUM + simdIdx];
      }

      if (outRowIdx + wr == 0 || outRowIdx + wr == IN_H + 1) {
        data0 = 0;
        data1 = 0;
      } else {
        for (int i = 0; i < SIMD / IN_PE; i++) {
          data0((i + 1) * IN_PE_BIT - 1, i * IN_PE_BIT) =
              buffer_data[i](IN_PE_BIT - 1, 0);
          data1((i + 1) * IN_PE_BIT - 1, i * IN_PE_BIT) =
              buffer_data[i](IN_PE_BIT * 2 - 1, IN_PE_BIT);
        }
      }
      out.write((data1, data0));

      if (cycle == WLEN * K * SIMDNUM - 1) {
        w = 0;
      } else if (infoldIdx == K * SIMDNUM - 1) {
        w++;
      }

      if (infoldIdx == K * SIMDNUM - 1) {
        infoldIdx = 0;
      } else {
        infoldIdx++;
      }
    }
  }
}

template <unsigned K, unsigned IN_H, unsigned IN_W, unsigned IN_CH,
          unsigned IN_BIT, unsigned IN_PE, unsigned SIMD, unsigned OUTPENUM>
void conv3padding(stream<ap_uint<IN_PE * IN_BIT * 2>> &in,
                  stream<ap_uint<SIMD * IN_BIT * 2>> &out,
                  const unsigned reps = 1) {
  static_assert(SIMD % IN_PE == 0, "SIMD %IN_PE !=0");
  static_assert(K == 3, "K!=3");

  ap_uint<IN_PE * IN_BIT * 2> row_buffer[SIMD / IN_PE][4]
                                        [IN_W / 2 * IN_CH / SIMD];
#pragma HLS ARRAY_PARTITION variable = row_buffer dim = 1 complete
#pragma HLS RESOURCE variable = row_buffer core = RAM_S2P_BRAM

  ap_uint<8> inh = 0;
  ap_uint<8> outh = 0;

  ap_uint<2> storeBufferIdx = 0;
  ap_uint<2> loadBufferIdx = 3;
  ap_int<10> rowIdx = 0;
  
  stream_in_row<IN_W, IN_CH, IN_BIT, IN_PE, SIMD>(
    in, row_buffer, storeBufferIdx);
  storeBufferIdx++;
  
  stream_in_row<IN_W, IN_CH, IN_BIT, IN_PE, SIMD>(
    in, row_buffer, storeBufferIdx);
  storeBufferIdx++;

  for (unsigned rep = 0; rep < reps * IN_H - 2; rep++) {
#pragma HLS dependence intra false variable = row_buffer
    stream_in_row<IN_W, IN_CH, IN_BIT, IN_PE, SIMD>(
        in, row_buffer, storeBufferIdx);
    stream_out_data<K, IN_H, IN_W, IN_CH, IN_BIT, IN_PE, SIMD, OUTPENUM>(
        out, row_buffer, rowIdx, loadBufferIdx);
    loadBufferIdx++;
    storeBufferIdx++;

    if (rowIdx == IN_H - 1) {
      rowIdx = 0;
    } else {
      rowIdx++;
    }
  }
  stream_out_data<K, IN_H, IN_W, IN_CH, IN_BIT, IN_PE, SIMD, OUTPENUM>(
      out, row_buffer, rowIdx, loadBufferIdx);
  
  loadBufferIdx++;
  if (rowIdx == IN_H - 1) {
    rowIdx = 0;
  } else {
    rowIdx++;
  }
  
  stream_out_data<K, IN_H, IN_W, IN_CH, IN_BIT, IN_PE, SIMD, OUTPENUM>(
      out, row_buffer, rowIdx, loadBufferIdx);
  
  loadBufferIdx++;
  if (rowIdx == IN_H - 1) {
    rowIdx = 0;
  } else {
    rowIdx++;
  }
}

template <unsigned OUT_ROW, unsigned OUT_COL, unsigned OUT_CH, unsigned M_BIT,
          unsigned OUT_BIT, unsigned INC_BIT, unsigned BIAS_BIT,
          unsigned IN_BIT, unsigned W_BIT, unsigned L_SHIFT, unsigned INFOLD, unsigned PE>
void streamRelu(stream<ap_uint<PE * M_BIT * 2>> &in,
                  const ap_int<INC_BIT> inc[PE][OUT_CH / PE],
                  const ap_int<BIAS_BIT> bias[PE][OUT_CH / PE],
                  stream<ap_uint<PE * OUT_BIT * 2>> &out,
                  const unsigned rep = 1) {
#pragma HLS ARRAY_PARTITION variable = inc complete dim = 1
#pragma HLS ARRAY_PARTITION variable = bias complete dim = 1
  ap_uint<PE * M_BIT> reg;
  ap_uint<PE * M_BIT * 2> data_in;
  ap_uint<PE * M_BIT> data0, data1;
  ap_uint<PE * OUT_BIT * 2> data_out;
  (data1, data0) = in.read();
  reg = data1;


  for (int r = 0; r < OUT_ROW * rep; r++)
    for (ap_uint<7> peIdx = 0; peIdx < OUT_CH / PE; peIdx++)
      for (ap_uint<9> w = 0; w < OUT_COL / 2; w++) {
#pragma HLS pipeline II = INFOLD
        ap_int<M_BIT> invec[2 * PE];
#pragma HLS array_partition variable = invec dim = 1 complete
        (data1, data0) = in.read();
        data_in = (data0, reg);
        reg = data1;
        for (int i = 0; i < PE * 2; i++) {
          invec[i] = data_in((i + 1) * M_BIT - 1, i * M_BIT);
        }
        for (int i = 0; i < PE * 2; i++) {
          data_out((i + 1) * OUT_BIT - 1, i * OUT_BIT) =
              qurelu_fixed<M_BIT, OUT_BIT, INC_BIT, BIAS_BIT, IN_BIT, W_BIT,
                              L_SHIFT>(invec[i], inc[i % PE][peIdx],
                                       bias[i % PE][peIdx]);
        }
        out.write(data_out);
      }
}


template <unsigned OUT_ROW, unsigned OUT_COL, unsigned OUT_CH, unsigned M_BIT,
          unsigned OUT_BIT, unsigned INC_BIT, unsigned BIAS_BIT,
          unsigned IN_BIT, unsigned W_BIT, unsigned L_SHIFT, unsigned INFOLD, unsigned PE>
void streamBnRelu(stream<ap_uint<PE * M_BIT * 2>> &in,
                  const ap_int<INC_BIT> inc[PE][OUT_CH / PE],
                  const ap_int<BIAS_BIT> bias[PE][OUT_CH / PE],
                  stream<ap_uint<PE * OUT_BIT * 2>> &out,
                  const unsigned rep = 1) {
#pragma HLS ARRAY_PARTITION variable = inc complete dim = 1
#pragma HLS ARRAY_PARTITION variable = bias complete dim = 1
  ap_uint<PE * M_BIT> reg;
  ap_uint<PE * M_BIT * 2> data_in;
  ap_uint<PE * M_BIT> data0, data1;
  ap_uint<PE * OUT_BIT * 2> data_out;
  (data1, data0) = in.read();
  reg = data1;


  for (int r = 0; r < OUT_ROW * rep; r++)
    for (ap_uint<7> peIdx = 0; peIdx < OUT_CH / PE; peIdx++)
      for (ap_uint<9> w = 0; w < OUT_COL / 2; w++) {
#pragma HLS pipeline II = INFOLD
        ap_int<M_BIT> invec[2 * PE];
#pragma HLS array_partition variable = invec dim = 1 complete
        (data1, data0) = in.read();
        data_in = (data0, reg);
        reg = data1;
        for (int i = 0; i < PE * 2; i++) {
          invec[i] = data_in((i + 1) * M_BIT - 1, i * M_BIT);
        }
        for (int i = 0; i < PE * 2; i++) {
          data_out((i + 1) * OUT_BIT - 1, i * OUT_BIT) =
              bn_qurelu_fixed<M_BIT, OUT_BIT, INC_BIT, BIAS_BIT, IN_BIT, W_BIT,
                              L_SHIFT>(invec[i], inc[i % PE][peIdx],
                                       bias[i % PE][peIdx]);
        }
        out.write(data_out);
      }
}

template <unsigned IN_BIT, unsigned SIMD, unsigned PROD_BIT>
void pack_input_data(ap_uint<IN_BIT * SIMD> A, ap_uint<IN_BIT * SIMD> B,
                     ap_uint<PROD_BIT + IN_BIT> ipack[SIMD],
                     ap_uint<PROD_BIT + 5> subdata[2]) {
#pragma HLS array_partition variable = ipack
#pragma HLS array_partition variable = subdata
  subdata[1] = 0;
  subdata[0] = 0;
  for (int i = 0; i < SIMD; i++) {
    ap_uint<IN_BIT> i1 = A(i * IN_BIT + IN_BIT - 1, i * IN_BIT);
    ap_uint<IN_BIT> i0 = B(i * IN_BIT + IN_BIT - 1, i * IN_BIT);
    ipack[i] = (i1, (ap_uint<PROD_BIT - IN_BIT>)0, i0);
    subdata[1] += i1;
    subdata[0] += i0;
  }
}

template <unsigned W_BIT, unsigned SIMD, unsigned PROD_BIT>
void pack_weight_data(ap_uint<W_BIT * SIMD> w2, ap_uint<W_BIT * SIMD> w1,
                      ap_uint<W_BIT * SIMD> w0,
                      ap_int<PROD_BIT * 2 + W_BIT> wpack[SIMD]) {
#pragma HLS array_partition variable = wpack

  for (int i = 0; i < SIMD; i++) {
    ap_int<W_BIT> w2_seg = w2(i * W_BIT + W_BIT - 1, i * W_BIT);
    ap_int<W_BIT> w1_seg = w1(i * W_BIT + W_BIT - 1, i * W_BIT);
    ap_int<W_BIT> w0_seg = w0(i * W_BIT + W_BIT - 1, i * W_BIT);
    wpack[i] =
        (w0_seg * (1 << (PROD_BIT * 2))) + (w1_seg * (1 << PROD_BIT)) + w2_seg;
  }
}

template <unsigned W_BIT, unsigned SIMD, unsigned PROD_BIT>
void pack_shiftweight_data(ap_uint<W_BIT * SIMD> w2, ap_uint<W_BIT * SIMD> w1,
                      ap_uint<W_BIT * SIMD> w0,
                      ap_uint<PROD_BIT * 2 + W_BIT> wpack[SIMD]) {
#pragma HLS array_partition variable = wpack

  for (int i = 0; i < SIMD; i++) {
    ap_uint<W_BIT> w2_seg = w2(i * W_BIT + W_BIT - 1, i * W_BIT);
    ap_uint<W_BIT> w1_seg = w1(i * W_BIT + W_BIT - 1, i * W_BIT);
    ap_uint<W_BIT> w0_seg = w0(i * W_BIT + W_BIT - 1, i * W_BIT);
    wpack[i] = (w0_seg, (ap_uint<PROD_BIT - W_BIT>)0, w1_seg, (ap_uint<PROD_BIT - W_BIT>)0, w2_seg);
  }
}

template <unsigned W_BIT, unsigned IN_BIT, unsigned PROD_BIT, unsigned SIMD,
          unsigned CASCADE>
void simd_MAC(ap_uint<PROD_BIT * 2 + W_BIT> wpack[SIMD],
              ap_uint<PROD_BIT + IN_BIT> ipack[SIMD],
              ap_uint<PROD_BIT + 5> sub0,
              ap_uint<PROD_BIT + 5> sub1,
              ap_int<PROD_BIT + 5> &partial0, ap_int<PROD_BIT + 5> &partial1,
              ap_int<PROD_BIT + 5> &partial2, ap_int<PROD_BIT + 5> &partial3) {
#pragma HLS ARRAY_PARTITION variable = wpack complete
#pragma HLS ARRAY_PARTITION variable = ipack complete
  ap_int<PROD_BIT + 5> r0, r1, r2, r3;
  r0 = 0;
  r1 = 0;
  r2 = 0;
  r3 = 0;
  for (int i = 0; i < SIMD; i += CASCADE) {
#pragma HLS unroll
    ap_uint<PROD_BIT * 4> m = 0;
    for (int cs = 0; cs < CASCADE; cs++) {
      m += wpack[i + cs] * ipack[i + cs];
    }

    ap_uint<PROD_BIT> p0 = m(PROD_BIT - 1, 0);
    ap_uint<PROD_BIT> p1 = m(PROD_BIT * 2 - 1, PROD_BIT);
    ap_uint<PROD_BIT> p2 = m(PROD_BIT * 3 - 1, PROD_BIT * 2);
    ap_uint<PROD_BIT> p3 = m(PROD_BIT * 4 - 1, PROD_BIT * 3);

    r0 += p0;
    r1 += p1;
    r2 += p2;
    r3 += p3;
  }
  partial0 = r0 - sub0;
  partial1 = r1 - sub0 - sub1;
  partial2 = r2 - sub0 - sub1;
  partial3 = r3 - sub1;
}

template <unsigned IN_BIT, unsigned W_BIT>
ap_int<IN_BIT + W_BIT> conv_mul_lut(ap_uint<IN_BIT> in, ap_int<W_BIT> w) {
  ap_int<IN_BIT + W_BIT> out;
#pragma HLS RESOURCE variable=return core=Mul_LUT
#pragma HLS inline off
  out = in * w;
  return out;
}

template <unsigned W_BIT, unsigned IN_BIT, unsigned SIMD, unsigned PROD_BIT>
void simd_MAC_DSPLUT(ap_int<W_BIT * SIMD> w0, ap_int<W_BIT * SIMD> w1,
                     ap_int<W_BIT * SIMD> w2, ap_uint<IN_BIT * SIMD> i0,
                     ap_uint<IN_BIT * SIMD> i1, ap_int<PROD_BIT + 5> &partial0,
                     ap_int<PROD_BIT + 5> &partial1,
                     ap_int<PROD_BIT + 5> &partial2,
                     ap_int<PROD_BIT + 5> &partial3) {
  ap_int<PROD_BIT + 5> r0, r1, r2, r3;
  r0 = 0;
  r1 = 0;
  r2 = 0;
  r3 = 0;
  for (int i = 0; i < SIMD; i++) {
    ap_int<W_BIT> w0_seg = w0((i + 1) * W_BIT - 1, i * W_BIT);
    ap_int<W_BIT> w1_seg = w1((i + 1) * W_BIT - 1, i * W_BIT);
    ap_int<W_BIT> w2_seg = w2((i + 1) * W_BIT - 1, i * W_BIT);
    ap_uint<IN_BIT> x0_seg = i0((i + 1) * IN_BIT - 1, i * IN_BIT);
    ap_uint<IN_BIT> x1_seg = i1((i + 1) * IN_BIT - 1, i * IN_BIT);

    r0 += conv_mul_lut<IN_BIT, W_BIT>(x0_seg, w2_seg);
    r1 += conv_mul_lut<IN_BIT, W_BIT>(x0_seg, w1_seg) + conv_mul_lut<IN_BIT, W_BIT>(x1_seg, w2_seg);
    r2 += conv_mul_lut<IN_BIT, W_BIT>(x0_seg, w0_seg) + conv_mul_lut<IN_BIT, W_BIT>(x1_seg, w1_seg);
    r3 += conv_mul_lut<IN_BIT, W_BIT>(x1_seg, w0_seg);
  }
  partial0 = r0;
  partial1 = r1;
  partial2 = r2;
  partial3 = r3;
}

template <unsigned K, unsigned IN_BIT, unsigned IN_CH, unsigned OUT_BIT,
          unsigned OUT_H, unsigned OUT_W, unsigned OUT_CH, unsigned W_BIT,
          unsigned GUARD_BIT, unsigned M_BIT, unsigned INC_BIT,
          unsigned BIAS_BIT, unsigned SIMD, unsigned CASCADE, unsigned PE,
          unsigned L_SHIFT>
void convDSPOpt(
    stream<ap_uint<SIMD * IN_BIT * 2>> &vec,
    const ap_uint<SIMD * W_BIT> weights[PE][3][K * IN_CH / SIMD * OUT_CH / PE],
    const ap_int<INC_BIT> inc[PE][OUT_CH / PE],
    const ap_int<BIAS_BIT> bias[PE][OUT_CH / PE],
    stream<ap_uint<PE * M_BIT * 2>> &out,
    const unsigned reps = 1) {

  static_assert(IN_CH % SIMD == 0, "IN_CH % SIMD !=0");
  static_assert(SIMD % CASCADE == 0, "SIMD % CASCADE != 0");
  static_assert(CASCADE <= 4, "SIMD % CASCADE != 0");
  const unsigned PENUM = OUT_CH / PE;
  const unsigned SIMDNUM = IN_CH / SIMD;
  const unsigned PROD_BIT = W_BIT + IN_BIT + GUARD_BIT;
  const unsigned WPACK_BIT = W_BIT * 3 + IN_BIT * 2 + GUARD_BIT * 2;
  const unsigned IPACK_BIT = IN_BIT * 2 + W_BIT + GUARD_BIT * 1;
  const unsigned OUT_WNUM = OUT_W / 2;
  const unsigned INFOLD = K * SIMDNUM;

#pragma HLS ARRAY_PARTITION variable = weights complete dim = 1
#pragma HLS ARRAY_PARTITION variable = weights complete dim = 2
#pragma HLS ARRAY_PARTITION variable = inc complete dim = 1
#pragma HLS ARRAY_PARTITION variable = bias complete dim = 1

  ap_uint<WPACK_BIT> wpacks[PE][SIMD];
#pragma HLS ARRAY_PARTITION variable = wpacks complete dim = 1
#pragma HLS ARRAY_PARTITION variable = wpacks complete dim = 2

  ap_uint<IPACK_BIT> ipack[SIMD];
#pragma HLS ARRAY_PARTITION variable = ipack complete dim = 1

  ap_int<M_BIT> firPartialRes0[PE];
#pragma HLS ARRAY_PARTITION variable = firPartialRes0 complete dim = 1
  ap_int<M_BIT> firPartialRes1[PE];
#pragma HLS ARRAY_PARTITION variable = firPartialRes1 complete dim = 1

  ap_int<M_BIT> outPartialArr0[PE];
#pragma HLS ARRAY_PARTITION variable = outPartialArr0 complete dim = 1
  ap_int<M_BIT> outPartialArr1[PE];
#pragma HLS ARRAY_PARTITION variable = outPartialArr1 complete dim = 1

  ap_int<PROD_BIT + 5> firPartial0;
  ap_int<PROD_BIT + 5> firPartial1;
  ap_int<PROD_BIT + 5> firPartial2;
  ap_int<PROD_BIT + 5> firPartial3;
  for (unsigned int h = 0; h < OUT_H * reps; h++) {
    for (ap_uint<7> peIdx = 0; peIdx < PENUM; peIdx++) {
      for (ap_uint<9> w = 0; w < OUT_WNUM; w++) {
        for (ap_uint<5> infoldIdx = 0; infoldIdx < INFOLD; infoldIdx++) {
#pragma HLS pipeline
          bool m_clear = (w == 0);
          bool o_clear = (infoldIdx == 0);
          bool o_out = (infoldIdx == INFOLD - 1);
          ap_uint<SIMD * IN_BIT> data1, data0;
          (data1, data0) = vec.read();
          ap_uint<PROD_BIT + 5> subdata[2];
#pragma HLS ARRAY_PARTITION variable = subdata complete dim = 1
          pack_input_data<IN_BIT, SIMD, PROD_BIT>(data1, data0, ipack, subdata);
          subdata[0] = subdata[0] << (W_BIT - 1);
          subdata[1] = subdata[1] << (W_BIT - 1);
          for (int p = 0; p < PE; p++) {
#pragma HLS unroll
            pack_shiftweight_data<W_BIT, SIMD, PROD_BIT>(
                weights[p][2][peIdx * INFOLD + infoldIdx],
                weights[p][1][peIdx * INFOLD + infoldIdx],
                weights[p][0][peIdx * INFOLD + infoldIdx], wpacks[p]);
          }

          for (int p = 0; p < PE; p++) {
#pragma HLS unroll 

            simd_MAC<W_BIT, IN_BIT, PROD_BIT, SIMD, CASCADE>(
                wpacks[p], ipack, subdata[0], subdata[1], firPartial0, firPartial1, firPartial2,
                firPartial3);

            if (m_clear & o_clear) {
              outPartialArr0[p] = firPartialRes0[p];
              outPartialArr1[p] = firPartial1;
            }
            if (m_clear & !o_clear) {
              outPartialArr0[p] = outPartialArr0[p];
              outPartialArr1[p] += firPartial1;
            } 
            if (!m_clear & o_clear) {
              outPartialArr0[p] = firPartial0 + firPartialRes0[p];
              outPartialArr1[p] = firPartial1 + firPartialRes1[p];
            }
            if (!m_clear & !o_clear) {
              outPartialArr0[p] += firPartial0;
              outPartialArr1[p] += firPartial1;
            }

            if (o_clear) {
              firPartialRes0[p] = firPartial2;
              firPartialRes1[p] = firPartial3;
            }
            else {
              firPartialRes0[p] += firPartial2;
              firPartialRes1[p] += firPartial3;
            }
          }

          if (o_out) {
            ap_uint<PE * M_BIT> out_buf0;
            ap_uint<PE * M_BIT> out_buf1;
            for (int p = 0; p < PE; p++) {
#pragma HLS unroll 
              out_buf0(p * M_BIT + M_BIT - 1, p * M_BIT) = outPartialArr0[p];
              out_buf1(p * M_BIT + M_BIT - 1, p * M_BIT) = outPartialArr1[p];

            }
            out.write((out_buf1, out_buf0));

          }
        }
      }
    }
  }  
  ap_uint<PE * M_BIT> out_buf2;
  for (ap_uint<4> p = 0; p < PE; p++) {
#pragma HLS unroll
    out_buf2(p * M_BIT + M_BIT - 1, p * M_BIT) = firPartialRes0[p];
  }
  out.write((0, out_buf2));
}

template <unsigned IN_ROW, unsigned IN_COL, unsigned IN_CH, unsigned IN_BIT,

          unsigned OUT_CH,
          unsigned OUT_BIT, 
          unsigned W_BIT, unsigned M_BIT, unsigned INC_BIT, unsigned BIAS_BIT,

          unsigned SIMD, unsigned CASCADE, unsigned IN_PE, unsigned PE,
          unsigned L_SHIFT>
void conv3x3_bn_act_DSPopt(
    stream<ap_uint<IN_BIT * IN_PE * 2>> &in,
    const ap_uint<SIMD * W_BIT> weights[PE][3]
                                       [((IN_CH * 3) / SIMD) * (OUT_CH / PE)],
    const ap_int<INC_BIT> inc[PE][OUT_CH / PE],
    const ap_int<BIAS_BIT> bias[PE][OUT_CH / PE],
    stream<ap_uint<OUT_BIT * PE * 2>> &out, const unsigned reps = 1) {
#pragma HLS DATAFLOW

  const unsigned INTER_ROW = IN_ROW + 2;
  const unsigned INTER_COL = IN_COL + 2;
  const unsigned OUT_ROW = IN_ROW;
  const unsigned OUT_COL = IN_COL;
  const unsigned INFOLD = 3 * IN_CH / SIMD;


  stream<ap_uint<SIMD * IN_BIT * 2>> padding_out("padding_out");
  conv3padding<3, IN_ROW, IN_COL, IN_CH, IN_BIT, IN_PE, SIMD, OUT_CH / PE>(
      in, padding_out, reps);

  stream<ap_uint<PE * M_BIT * 2>> conv_out("conv_out");
  convDSPOpt<3, IN_BIT, IN_CH, OUT_BIT, OUT_ROW, OUT_COL, OUT_CH, W_BIT, 3,
             M_BIT, INC_BIT, BIAS_BIT, SIMD, CASCADE, PE, L_SHIFT>(
      padding_out, weights, inc, bias, conv_out, reps);
  streamBnRelu<OUT_ROW, OUT_COL, OUT_CH, M_BIT, OUT_BIT, INC_BIT, BIAS_BIT,
        L_SHIFT, IN_BIT, W_BIT, INFOLD, PE>(conv_out, inc, bias, out,
                                    reps);
}

#endif
