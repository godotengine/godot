// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "lib/jxl/modular/transform/rct.h"
#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "lib/jxl/modular/transform/rct.cc"
#include <hwy/foreach_target.h>
#include <hwy/highway.h>
HWY_BEFORE_NAMESPACE();
namespace jxl {
namespace HWY_NAMESPACE {

// These templates are not found via ADL.
using hwy::HWY_NAMESPACE::Add;
using hwy::HWY_NAMESPACE::ShiftRight;
using hwy::HWY_NAMESPACE::Sub;

template <int transform_type>
void InvRCTRow(const pixel_type* in0, const pixel_type* in1,
               const pixel_type* in2, pixel_type* out0, pixel_type* out1,
               pixel_type* out2, size_t w) {
  static_assert(transform_type >= 0 && transform_type < 7,
                "Invalid transform type");
  int second = transform_type >> 1;
  int third = transform_type & 1;

  size_t x = 0;
  const HWY_FULL(pixel_type) d;
  const size_t N = Lanes(d);
  for (; x + N - 1 < w; x += N) {
    if (transform_type == 6) {
      auto Y = Load(d, in0 + x);
      auto Co = Load(d, in1 + x);
      auto Cg = Load(d, in2 + x);
      Y = Sub(Y, ShiftRight<1>(Cg));
      auto G = Add(Cg, Y);
      Y = Sub(Y, ShiftRight<1>(Co));
      auto R = Add(Y, Co);
      Store(R, d, out0 + x);
      Store(G, d, out1 + x);
      Store(Y, d, out2 + x);
    } else {
      auto First = Load(d, in0 + x);
      auto Second = Load(d, in1 + x);
      auto Third = Load(d, in2 + x);
      if (third) Third = Add(Third, First);
      if (second == 1) {
        Second = Add(Second, First);
      } else if (second == 2) {
        Second = Add(Second, ShiftRight<1>(Add(First, Third)));
      }
      Store(First, d, out0 + x);
      Store(Second, d, out1 + x);
      Store(Third, d, out2 + x);
    }
  }
  for (; x < w; x++) {
    if (transform_type == 6) {
      pixel_type Y = in0[x];
      pixel_type Co = in1[x];
      pixel_type Cg = in2[x];
      pixel_type tmp = PixelAdd(Y, -(Cg >> 1));
      pixel_type G = PixelAdd(Cg, tmp);
      pixel_type B = PixelAdd(tmp, -(Co >> 1));
      pixel_type R = PixelAdd(B, Co);
      out0[x] = R;
      out1[x] = G;
      out2[x] = B;
    } else {
      pixel_type First = in0[x];
      pixel_type Second = in1[x];
      pixel_type Third = in2[x];
      if (third) Third = PixelAdd(Third, First);
      if (second == 1) {
        Second = PixelAdd(Second, First);
      } else if (second == 2) {
        Second = PixelAdd(Second, (PixelAdd(First, Third) >> 1));
      }
      out0[x] = First;
      out1[x] = Second;
      out2[x] = Third;
    }
  }
}

Status InvRCT(Image& input, size_t begin_c, size_t rct_type, ThreadPool* pool) {
  JXL_RETURN_IF_ERROR(CheckEqualChannels(input, begin_c, begin_c + 2));
  size_t m = begin_c;
  Channel& c0 = input.channel[m + 0];
  size_t w = c0.w;
  size_t h = c0.h;
  if (rct_type == 0) {  // noop
    return true;
  }
  // Permutation: 0=RGB, 1=GBR, 2=BRG, 3=RBG, 4=GRB, 5=BGR
  int permutation = rct_type / 7;
  JXL_ENSURE(permutation < 6);
  // 0-5 values have the low bit corresponding to Third and the high bits
  // corresponding to Second. 6 corresponds to YCoCg.
  //
  // Second: 0=nop, 1=SubtractFirst, 2=SubtractAvgFirstThird
  //
  // Third: 0=nop, 1=SubtractFirst
  int custom = rct_type % 7;
  // Special case: permute-only. Swap channels around.
  if (custom == 0) {
    Channel ch0 = std::move(input.channel[m]);
    Channel ch1 = std::move(input.channel[m + 1]);
    Channel ch2 = std::move(input.channel[m + 2]);
    input.channel[m + (permutation % 3)] = std::move(ch0);
    input.channel[m + ((permutation + 1 + permutation / 3) % 3)] =
        std::move(ch1);
    input.channel[m + ((permutation + 2 - permutation / 3) % 3)] =
        std::move(ch2);
    return true;
  }
  constexpr decltype(&InvRCTRow<0>) inv_rct_row[] = {
      InvRCTRow<0>, InvRCTRow<1>, InvRCTRow<2>, InvRCTRow<3>,
      InvRCTRow<4>, InvRCTRow<5>, InvRCTRow<6>};
  const auto process_row = [&](const uint32_t task,
                               size_t /* thread */) -> Status {
    const size_t y = task;
    const pixel_type* in0 = input.channel[m].Row(y);
    const pixel_type* in1 = input.channel[m + 1].Row(y);
    const pixel_type* in2 = input.channel[m + 2].Row(y);
    pixel_type* out0 = input.channel[m + (permutation % 3)].Row(y);
    pixel_type* out1 =
        input.channel[m + ((permutation + 1 + permutation / 3) % 3)].Row(y);
    pixel_type* out2 =
        input.channel[m + ((permutation + 2 - permutation / 3) % 3)].Row(y);
    inv_rct_row[custom](in0, in1, in2, out0, out1, out2, w);
    return true;
  };
  JXL_RETURN_IF_ERROR(
      RunOnPool(pool, 0, h, ThreadPool::NoInit, process_row, "InvRCT"));
  return true;
}

}  // namespace HWY_NAMESPACE
}  // namespace jxl
HWY_AFTER_NAMESPACE();

#if HWY_ONCE
namespace jxl {

HWY_EXPORT(InvRCT);
Status InvRCT(Image& input, size_t begin_c, size_t rct_type, ThreadPool* pool) {
  return HWY_DYNAMIC_DISPATCH(InvRCT)(input, begin_c, rct_type, pool);
}

}  // namespace jxl
#endif
