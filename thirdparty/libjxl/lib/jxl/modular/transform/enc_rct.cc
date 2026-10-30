// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "lib/jxl/modular/transform/enc_rct.h"

#include <array>
#include <cstddef>
#include <cstdint>

#include "lib/jxl/base/compiler_specific.h"
#include "lib/jxl/base/data_parallel.h"
#include "lib/jxl/base/status.h"
#include "lib/jxl/modular/modular_image.h"
#include "lib/jxl/modular/transform/transform.h"  // CheckEqualChannels

#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "lib/jxl/modular/transform/enc_rct.cc"
#include <hwy/foreach_target.h>
#include <hwy/highway.h>

HWY_BEFORE_NAMESPACE();
namespace jxl {
namespace HWY_NAMESPACE {

using hwy::HWY_NAMESPACE::Add;
using hwy::HWY_NAMESPACE::Load;
using hwy::HWY_NAMESPACE::ShiftRight;
using hwy::HWY_NAMESPACE::Store;
using hwy::HWY_NAMESPACE::Sub;

template <size_t transform>
bool FwdRctRow(size_t w, const pixel_type* in0, const pixel_type* in1,
               const pixel_type* in2, pixel_type* out0, pixel_type* out1,
               pixel_type* out2) {
  static_assert(transform <= 6, "Invalid RCT transform");
  constexpr size_t second = transform >> 1;
  constexpr size_t third = transform & 1;
#if HWY_TARGET == HWY_SCALAR
  if (transform == 6) {
    for (size_t x = 0; x < w; x++) {
      pixel_type R = in0[x];
      pixel_type G = in1[x];
      pixel_type B = in2[x];
      pixel_type o1 = R - B;
      pixel_type tmp = B + (o1 >> 1);
      pixel_type o2 = G - tmp;
      out0[x] = tmp + (o2 >> 1);
      out1[x] = o1;
      out2[x] = o2;
    }
    return true;
  }

  for (size_t x = 0; x < w; x++) {
    pixel_type First = in0[x];
    pixel_type Second = in1[x];
    pixel_type Third = in2[x];
    if (second == 1) {
      Second = Second - First;
    } else if (second == 2) {
      Second = Second - ((First + Third) >> 1);
    }
    if (third) Third = Third - First;
    out0[x] = First;
    out1[x] = Second;
    out2[x] = Third;
  }
  return true;
#else
  const HWY_FULL(int32_t) di;

  if (transform == 6) {
    for (size_t x = 0; x < w; x += Lanes(di)) {
      const auto R = Load(di, in0 + x);
      const auto G = Load(di, in1 + x);
      const auto B = Load(di, in2 + x);
      const auto o1 = Sub(R, B);
      const auto tmp = Add(B, ShiftRight<1>(o1));
      const auto o2 = Sub(G, tmp);
      const auto o0 = Add(tmp, ShiftRight<1>(o2));
      Store(o0, di, out0 + x);
      Store(o1, di, out1 + x);
      Store(o2, di, out2 + x);
    }
    return true;
  }

  for (size_t x = 0; x < w; x += Lanes(di)) {
    const auto i0 = Load(di, in0 + x);
    const auto i1 = Load(di, in1 + x);
    const auto i2 = Load(di, in2 + x);
    auto o1 = i1;
    if (second == 1) {
      o1 = Sub(o1, i0);
    } else if (second == 2) {
      o1 = Sub(o1, ShiftRight<1>(Add(i0, i2)));
    }
    auto o2 = i2;
    if (third) o2 = Sub(o2, i0);
    Store(i0, di, out0 + x);
    Store(o1, di, out1 + x);
    Store(o2, di, out2 + x);
  }
  return true;
#endif
}

std::array<const Channel*, 3> RctPermute(
    const std::array<const Channel*, 3>& in, size_t permutation) {
  return {in[permutation % 3], in[(permutation + 1 + permutation / 3) % 3],
          in[(permutation + 2 - permutation / 3) % 3]};
}

Status FwdRctImpl(const std::array<const Channel*, 3>& in,
                  const std::array<Channel*, 3>& out, size_t rct_type,
                  ThreadPool* pool) {
  // Permutation: 0=RGB, 1=GBR, 2=BRG, 3=RBG, 4=GRB, 5=BGR
  const size_t permutation = rct_type / 7;
  // 0-5 values have the low bit corresponding to Third and the high bits
  // corresponding to Second. 6 corresponds to YCoCg.
  //
  // Second: 0=nop, 1=SubtractFirst, 2=SubtractAvgFirstThird
  //
  // Third: 0=nop, 1=SubtractFirst
  const size_t transform = rct_type % 7;
  std::array<const Channel*, 3> inp = RctPermute(in, permutation);
  const size_t w = out[0]->w;
  const size_t h = out[0]->h;
  if (transform == 0) {
    const auto do_rct = [&](const int y, const int thread) -> Status {
      return FwdRctRow<0>(w, inp[0]->Row(y), inp[1]->Row(y), inp[2]->Row(y),
                          out[0]->Row(y), out[1]->Row(y), out[2]->Row(y));
    };
    JXL_RETURN_IF_ERROR(
        RunOnPool(pool, 0, h, ThreadPool::NoInit, do_rct, "FwdRct"));
  } else if (transform == 1) {
    const auto do_rct = [&](const int y, const int thread) -> Status {
      return FwdRctRow<1>(w, inp[0]->Row(y), inp[1]->Row(y), inp[2]->Row(y),
                          out[0]->Row(y), out[1]->Row(y), out[2]->Row(y));
    };
    JXL_RETURN_IF_ERROR(
        RunOnPool(pool, 0, h, ThreadPool::NoInit, do_rct, "FwdRct"));
  } else if (transform == 2) {
    const auto do_rct = [&](const int y, const int thread) -> Status {
      return FwdRctRow<2>(w, inp[0]->Row(y), inp[1]->Row(y), inp[2]->Row(y),
                          out[0]->Row(y), out[1]->Row(y), out[2]->Row(y));
    };
    JXL_RETURN_IF_ERROR(
        RunOnPool(pool, 0, h, ThreadPool::NoInit, do_rct, "FwdRct"));
  } else if (transform == 3) {
    const auto do_rct = [&](const int y, const int thread) -> Status {
      return FwdRctRow<3>(w, inp[0]->Row(y), inp[1]->Row(y), inp[2]->Row(y),
                          out[0]->Row(y), out[1]->Row(y), out[2]->Row(y));
    };
    JXL_RETURN_IF_ERROR(
        RunOnPool(pool, 0, h, ThreadPool::NoInit, do_rct, "FwdRct"));
  } else if (transform == 4) {
    const auto do_rct = [&](const int y, const int thread) -> Status {
      return FwdRctRow<4>(w, inp[0]->Row(y), inp[1]->Row(y), inp[2]->Row(y),
                          out[0]->Row(y), out[1]->Row(y), out[2]->Row(y));
    };
    JXL_RETURN_IF_ERROR(
        RunOnPool(pool, 0, h, ThreadPool::NoInit, do_rct, "FwdRct"));
  } else if (transform == 5) {
    const auto do_rct = [&](const int y, const int thread) -> Status {
      return FwdRctRow<5>(w, inp[0]->Row(y), inp[1]->Row(y), inp[2]->Row(y),
                          out[0]->Row(y), out[1]->Row(y), out[2]->Row(y));
    };
    JXL_RETURN_IF_ERROR(
        RunOnPool(pool, 0, h, ThreadPool::NoInit, do_rct, "FwdRct"));
  } else {
    JXL_DASSERT(transform == 6);
    const auto do_rct = [&](const int y, const int thread) -> Status {
      return FwdRctRow<6>(w, inp[0]->Row(y), inp[1]->Row(y), inp[2]->Row(y),
                          out[0]->Row(y), out[1]->Row(y), out[2]->Row(y));
    };
    JXL_RETURN_IF_ERROR(
        RunOnPool(pool, 0, h, ThreadPool::NoInit, do_rct, "FwdRct"));
  }
  return true;
}

// NOLINTNEXTLINE(google-readability-namespace-comments)
}  // namespace HWY_NAMESPACE
}  // namespace jxl
HWY_AFTER_NAMESPACE();

#if HWY_ONCE

namespace jxl {

HWY_EXPORT(FwdRctImpl);

Status FwdRct(const std::array<const Channel*, 3>& in,
              const std::array<Channel*, 3>& out, size_t rct_type,
              ThreadPool* pool) {
  return HWY_DYNAMIC_DISPATCH(FwdRctImpl)(in, out, rct_type, pool);
}

Status FwdRct(Image& input, size_t begin_c, size_t rct_type, ThreadPool* pool) {
  JXL_RETURN_IF_ERROR(CheckEqualChannels(input, begin_c, begin_c + 2));
  if (rct_type == 0) {  // noop
    return false;
  }
  std::array<Channel*, 3> chs = {&input.channel[begin_c + 0],
                                 &input.channel[begin_c + 1],
                                 &input.channel[begin_c + 2]};
  return FwdRct({chs[0], chs[1], chs[2]}, chs, rct_type, pool);
}

}  // namespace jxl
#endif
