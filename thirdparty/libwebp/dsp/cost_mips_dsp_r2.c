// Copyright 2014 Google Inc. All Rights Reserved.
//
// Use of this source code is governed by a BSD-style license
// that can be found in the COPYING file in the root of the source
// tree. An additional intellectual property rights grant can be found
// in the file PATENTS. All contributing project authors may
// be found in the AUTHORS file in the root of the source tree.
// -----------------------------------------------------------------------------
//
// Author: Djordje Pesut (djordje.pesut@imgtec.com)

#include "./dsp.h"

#if defined(WEBP_USE_MIPS_DSP_R2)

#include "../enc/cost_enc.h"

static int GetResidualCost(int ctx0, const VP8Residual* const res) {
  int temp0, temp1;
  int v_reg, ctx_reg;
  int n = res->first;
  // should be prob[VP8EncBands[n]], but it's equivalent for n=0 or 1
  int p0 = res->prob[n][ctx0][0];
  CostArrayPtr const costs = res->costs;
  const uint16_t* t = costs[n][ctx0];
  // bit_cost(1, p0) is already incorporated in t[] tables, but only if ctx != 0
  // (as required by the syntax). For ctx0 == 0, we need to add it here or it'll
  // be missing during the loop.
  int cost = (ctx0 == 0) ? VP8BitCost(1, p0) : 0;
  const int16_t* res_coeffs = res->coeffs;
  const int res_last = res->last;
  const int const_max_level = MAX_VARIABLE_LEVEL;
  const int const_2 = 2;
  const uint16_t** p_costs = &costs[n][0];
  const size_t inc_p_costs = NUM_CTX * sizeof(*p_costs);

  if (res->last < 0) {
    return VP8BitCost(0, p0);
  }

  __asm__ volatile (
    ".set      push                                                     \n\t"
    ".set      noreorder                                                \n\t"
    "subu      %[temp1],        %[res_last],        %[n]                \n\t"
    "blez      %[temp1],        2f                                      \n\t"
    " nop                                                               \n\t"
  "1:                                                                   \n\t"
    "sll       %[temp0],        %[n],               1                   \n\t"
    "lhx       %[v_reg],        %[temp0](%[res_coeffs])                 \n\t"
    "addiu     %[n],            %[n],               1                   \n\t"
    "absq_s.w  %[v_reg],        %[v_reg]                                \n\t"
    "sltiu     %[temp0],        %[v_reg],           2                   \n\t"
    "move      %[ctx_reg],      %[v_reg]                                \n\t"
    "movz      %[ctx_reg],      %[const_2],         %[temp0]            \n\t"
    "sll       %[temp1],        %[v_reg],           1                   \n\t"
    "lhx       %[temp1],        %[temp1](%[VP8LevelFixedCosts])         \n\t"
    "slt       %[temp0],        %[v_reg],           %[const_max_level]  \n\t"
    "movz      %[v_reg],        %[const_max_level], %[temp0]            \n\t"
    "addu      %[cost],         %[cost],            %[temp1]            \n\t"
    "sll       %[v_reg],        %[v_reg],           1                   \n\t"
    "sll       %[ctx_reg],      %[ctx_reg],         2                   \n\t"
    "lhx       %[temp0],        %[v_reg](%[t])                          \n\t"
    "addu      %[p_costs],      %[p_costs],         %[inc_p_costs]      \n\t"
    "addu      %[t],            %[p_costs],         %[ctx_reg]          \n\t"
    "addu      %[cost],         %[cost],            %[temp0]            \n\t"
    "bne       %[n],            %[res_last],        1b                  \n\t"
    " lw       %[t],            0(%[t])                                 \n\t"
  "2:                                                                   \n\t"
    ".set      pop                                                      \n\t"
    : [cost]"+&r"(cost), [t]"+&r"(t), [n]"+&r"(n), [v_reg]"=&r"(v_reg),
      [ctx_reg]"=&r"(ctx_reg), [p_costs]"+&r"(p_costs), [temp0]"=&r"(temp0),
      [temp1]"=&r"(temp1)
    : [const_2]"r"(const_2), [const_max_level]"r"(const_max_level),
      [VP8LevelFixedCosts]"r"(VP8LevelFixedCosts), [res_last]"r"(res_last),
      [res_coeffs]"r"(res_coeffs), [inc_p_costs]"r"(inc_p_costs)
    : "memory"
  );

  // Last coefficient is always non-zero
  {
    const int v = abs(res->coeffs[n]);
    assert(v != 0);
    cost += VP8LevelCost(t, v);
    if (n < 15) {
      const int b = VP8EncBands[n + 1];
      const int ctx = (v == 1) ? 1 : 2;
      const int last_p0 = res->prob[b][ctx][0];
      cost += VP8BitCost(0, last_p0);
    }
  }
  return cost;
}

//------------------------------------------------------------------------------
// Entry point

extern void VP8EncDspCostInitMIPSdspR2(void);

WEBP_TSAN_IGNORE_FUNCTION void VP8EncDspCostInitMIPSdspR2(void) {
  VP8GetResidualCost = GetResidualCost;
}

#else  // !WEBP_USE_MIPS_DSP_R2

WEBP_DSP_INIT_STUB(VP8EncDspCostInitMIPSdspR2)

#endif  // WEBP_USE_MIPS_DSP_R2
