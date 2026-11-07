// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "lib/jxl/enc_ans_simd.h"

#include <cstdint>

#include "lib/jxl/base/compiler_specific.h"
#include "lib/jxl/base/status.h"
#include "lib/jxl/dec_ans.h"
#include "lib/jxl/memory_manager_internal.h"

#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "lib/jxl/enc_ans_simd.cc"
#include <hwy/foreach_target.h>
#include <hwy/highway.h>

HWY_BEFORE_NAMESPACE();
namespace jxl {
namespace HWY_NAMESPACE {

// These templates are not found via ADL.
using hwy::HWY_NAMESPACE::Add;
using hwy::HWY_NAMESPACE::And;
using hwy::HWY_NAMESPACE::Ge;
using hwy::HWY_NAMESPACE::GetLane;
using hwy::HWY_NAMESPACE::Gt;
using hwy::HWY_NAMESPACE::IfThenElse;
using hwy::HWY_NAMESPACE::IfThenElseZero;
using hwy::HWY_NAMESPACE::Iota;
using hwy::HWY_NAMESPACE::LoadU;
using hwy::HWY_NAMESPACE::Lt;
using hwy::HWY_NAMESPACE::Mul;
using hwy::HWY_NAMESPACE::Or;
using hwy::HWY_NAMESPACE::Set;
using hwy::HWY_NAMESPACE::ShiftRight;
using hwy::HWY_NAMESPACE::Store;
using hwy::HWY_NAMESPACE::Sub;
using hwy::HWY_NAMESPACE::Zero;

template <size_t E, size_t M, size_t L>
uint32_t EstimateTokenCostImpl(uint32_t* JXL_RESTRICT values, size_t len,
                               uint32_t* JXL_RESTRICT out) {
  const HWY_FULL(uint32_t) du;
  const HWY_FULL(float) df;
  const auto kZero = Zero(du);
  const auto kSplit = Set(du, 1 << E);
  const auto kExpOffset = Set(du, 127);
  const auto kEBOffset = Set(du, 127 + M + L);
  const auto kBase = Set(du, static_cast<uint32_t>((1 << E) - (E << (M + L))));
  const auto kMulN = Set(du, 1 << (M + L));
  const auto kMaskL = Set(du, (1 << L) - 1);
  const auto kMaskM = Set(du, ((1 << M) - 1) << L);
  const auto kLargeThreshold = Set(du, (1 << 22) - 1);
  constexpr size_t kLargeShiftVal = 10;
  const auto kLargeShift = Set(du, kLargeShiftVal);

  auto extra_bits = kZero;
  size_t last_full = Lanes(du) * (len / Lanes(du));
  for (size_t i = 0; i < last_full; i += Lanes(du)) {
    const auto val = LoadU(du, values + i);
    const auto is_large = Gt(val, kLargeThreshold);
    const auto val_shifted = ShiftRight<kLargeShiftVal>(val);
    const auto not_literal = Ge(val, kSplit);
    const auto val_fixed = IfThenElse(is_large, val_shifted, val);
    const auto b = BitCast(du, ConvertTo(df, val_fixed));
    const auto l = And(val, kMaskL);
    const auto exp = ShiftRight<23>(b);
    const auto exp_fixed = IfThenElse(is_large, Add(exp, kLargeShift), exp);
    const auto n = Sub(exp_fixed, kExpOffset);
    const auto eb = Sub(exp_fixed, kEBOffset);
    const auto m = ShiftRight<23 - M - L>(b);
    const auto a = Add(kBase, Mul(n, kMulN));
    const auto d = And(m, kMaskM);
    const auto eb_fixed = IfThenElseZero(not_literal, eb);
    const auto c = Or(a, l);
    extra_bits = Add(extra_bits, eb_fixed);
    const auto t = Or(c, d);
    const auto t_fixed = IfThenElse(not_literal, t, val);
    Store(t_fixed, du, out + i);
  }
  if (last_full < len) {
    const auto stop = Set(du, len);
    const auto fence = Iota(du, last_full);
    const auto take = Lt(fence, stop);
    const auto val = LoadU(du, values + last_full);
    const auto is_large = Gt(val, kLargeThreshold);
    const auto val_shifted = ShiftRight<kLargeShiftVal>(val);
    const auto not_literal = Ge(val, kSplit);
    const auto val_fixed = IfThenElse(is_large, val_shifted, val);
    const auto b = BitCast(du, ConvertTo(df, val_fixed));
    const auto l = And(val, kMaskL);
    const auto exp = ShiftRight<23>(b);
    const auto exp_fixed = IfThenElse(is_large, Add(exp, kLargeShift), exp);
    const auto n = Sub(exp_fixed, kExpOffset);
    const auto eb = Sub(exp_fixed, kEBOffset);
    const auto m = ShiftRight<23 - M - L>(b);
    const auto a = Add(kBase, Mul(n, kMulN));
    const auto d = And(m, kMaskM);
    const auto eb_fixed = IfThenElseZero(not_literal, eb);
    const auto eb_masked = IfThenElseZero(take, eb_fixed);
    const auto c = Or(a, l);
    extra_bits = Add(extra_bits, eb_masked);
    const auto t = Or(c, d);
    const auto t_fixed = IfThenElse(not_literal, t, val);
    Store(t_fixed, du, out + last_full);
  }
  return GetLane(SumOfLanes(du, extra_bits));
}

uint32_t EstimateTokenCost(uint32_t* JXL_RESTRICT values, size_t len,
                           HybridUintConfig cfg, AlignedMemory& tokens) {
  uint32_t* JXL_RESTRICT out = tokens.address<uint32_t>();
#if HWY_TARGET == HWY_SCALAR
  uint32_t extra_bits = 0;
  for (size_t i = 0; i < len; ++i) {
    uint32_t v = values[i];
    uint32_t tok, nbits, bits;
    cfg.Encode(v, &tok, &nbits, &bits);
    extra_bits += nbits;
    out[i] = tok;
  }
  return extra_bits;
#else
  if (cfg.split_exponent == 0) {
    return EstimateTokenCostImpl<0, 0, 0>(values, len, out);
  } else if (cfg.split_exponent == 2) {
    JXL_DASSERT((cfg.msb_in_token == 0) && (cfg.lsb_in_token == 1));
    return EstimateTokenCostImpl<2, 0, 1>(values, len, out);
  } else if (cfg.split_exponent == 3) {
    if (cfg.msb_in_token == 1) {
      if (cfg.lsb_in_token == 0) {
        return EstimateTokenCostImpl<3, 1, 0>(values, len, out);
      } else {
        JXL_DASSERT(cfg.lsb_in_token == 2);
        return EstimateTokenCostImpl<3, 1, 2>(values, len, out);
      }
    } else {
      JXL_DASSERT(cfg.msb_in_token == 2);
      if (cfg.lsb_in_token == 0) {
        return EstimateTokenCostImpl<3, 2, 0>(values, len, out);
      } else {
        JXL_DASSERT(cfg.lsb_in_token == 1);
        return EstimateTokenCostImpl<3, 2, 1>(values, len, out);
      }
    }
  } else if (cfg.split_exponent == 4) {
    if (cfg.msb_in_token == 1) {
      if (cfg.lsb_in_token == 0) {
        return EstimateTokenCostImpl<4, 1, 0>(values, len, out);
      } else if (cfg.lsb_in_token == 2) {
        return EstimateTokenCostImpl<4, 1, 2>(values, len, out);
      } else {
        JXL_DASSERT(cfg.lsb_in_token == 3);
        return EstimateTokenCostImpl<4, 1, 3>(values, len, out);
      }
    } else {
      JXL_DASSERT(cfg.msb_in_token == 2);
      if (cfg.lsb_in_token == 0) {
        return EstimateTokenCostImpl<4, 2, 0>(values, len, out);
      } else if (cfg.lsb_in_token == 1) {
        return EstimateTokenCostImpl<4, 2, 1>(values, len, out);
      } else {
        JXL_DASSERT(cfg.lsb_in_token == 2);
        return EstimateTokenCostImpl<4, 2, 2>(values, len, out);
      }
    }
  } else if (cfg.split_exponent == 5) {
    if (cfg.msb_in_token == 1) {
      if (cfg.lsb_in_token == 0) {
        return EstimateTokenCostImpl<5, 1, 0>(values, len, out);
      } else if (cfg.lsb_in_token == 2) {
        return EstimateTokenCostImpl<5, 1, 2>(values, len, out);
      } else {
        JXL_DASSERT(cfg.lsb_in_token == 4);
        return EstimateTokenCostImpl<5, 1, 4>(values, len, out);
      }
    } else {
      JXL_DASSERT(cfg.msb_in_token == 2);
      if (cfg.lsb_in_token == 0) {
        return EstimateTokenCostImpl<5, 2, 0>(values, len, out);
      } else if (cfg.lsb_in_token == 1) {
        return EstimateTokenCostImpl<5, 2, 1>(values, len, out);
      } else if (cfg.lsb_in_token == 2) {
        return EstimateTokenCostImpl<5, 2, 2>(values, len, out);
      } else {
        JXL_DASSERT(cfg.lsb_in_token == 3);
        return EstimateTokenCostImpl<5, 2, 3>(values, len, out);
      }
    }
  } else if (cfg.split_exponent == 6) {
    if (cfg.msb_in_token == 0) {
      JXL_DASSERT(cfg.lsb_in_token == 0);
      return EstimateTokenCostImpl<6, 0, 0>(values, len, out);
    } else if (cfg.msb_in_token == 1) {
      JXL_DASSERT(cfg.lsb_in_token == 5);
      return EstimateTokenCostImpl<6, 1, 5>(values, len, out);
    } else {
      JXL_DASSERT(cfg.msb_in_token == 2);
      JXL_DASSERT(cfg.lsb_in_token == 4);
      return EstimateTokenCostImpl<6, 2, 4>(values, len, out);
    }
  } else if (cfg.split_exponent >= 7 && cfg.split_exponent <= 12) {
    JXL_DASSERT(cfg.msb_in_token == 0);
    JXL_DASSERT(cfg.lsb_in_token == 0);
    if (cfg.split_exponent == 7) {
      return EstimateTokenCostImpl<7, 0, 0>(values, len, out);
    } else if (cfg.split_exponent == 8) {
      return EstimateTokenCostImpl<8, 0, 0>(values, len, out);
    } else if (cfg.split_exponent == 9) {
      return EstimateTokenCostImpl<9, 0, 0>(values, len, out);
    } else if (cfg.split_exponent == 10) {
      return EstimateTokenCostImpl<10, 0, 0>(values, len, out);
    } else if (cfg.split_exponent == 11) {
      return EstimateTokenCostImpl<11, 0, 0>(values, len, out);
    } else {
      return EstimateTokenCostImpl<12, 0, 0>(values, len, out);
    }
  } else {
    JXL_DASSERT(false);
  }
  return ~0;
#endif
}

// NOLINTNEXTLINE(google-readability-namespace-comments)
}  // namespace HWY_NAMESPACE
}  // namespace jxl
HWY_AFTER_NAMESPACE();

#if HWY_ONCE
namespace jxl {

HWY_EXPORT(EstimateTokenCost);

uint32_t EstimateTokenCost(uint32_t* JXL_RESTRICT values, size_t len,
                           HybridUintConfig cfg, AlignedMemory& tokens) {
  JXL_DASSERT(cfg.lsb_in_token + cfg.msb_in_token <= cfg.split_exponent);
  return HWY_DYNAMIC_DISPATCH(EstimateTokenCost)(values, len, cfg, tokens);
}

}  // namespace jxl
#endif
