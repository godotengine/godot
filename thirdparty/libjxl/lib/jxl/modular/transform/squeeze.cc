// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "lib/jxl/modular/transform/squeeze.h"

#include <jxl/memory_manager.h>

#include <cstdlib>

#include "lib/jxl/base/common.h"
#include "lib/jxl/base/data_parallel.h"
#include "lib/jxl/base/printf_macros.h"
#include "lib/jxl/modular/modular_image.h"
#include "lib/jxl/modular/transform/transform.h"
#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "lib/jxl/modular/transform/squeeze.cc"
#include <hwy/foreach_target.h>
#include <hwy/highway.h>

#include "lib/jxl/simd_util-inl.h"

HWY_BEFORE_NAMESPACE();
namespace jxl {
namespace HWY_NAMESPACE {

// These templates are not found via ADL.
using hwy::HWY_NAMESPACE::Abs;
using hwy::HWY_NAMESPACE::Add;
using hwy::HWY_NAMESPACE::And;
using hwy::HWY_NAMESPACE::Gt;
using hwy::HWY_NAMESPACE::IfThenElse;
using hwy::HWY_NAMESPACE::IfThenZeroElse;
using hwy::HWY_NAMESPACE::Lt;
using hwy::HWY_NAMESPACE::MulEven;
using hwy::HWY_NAMESPACE::Ne;
using hwy::HWY_NAMESPACE::Neg;
using hwy::HWY_NAMESPACE::OddEven;
using hwy::HWY_NAMESPACE::RebindToUnsigned;
using hwy::HWY_NAMESPACE::ShiftLeft;
using hwy::HWY_NAMESPACE::ShiftRight;
using hwy::HWY_NAMESPACE::Sub;
using hwy::HWY_NAMESPACE::Xor;

#if HWY_TARGET != HWY_SCALAR

JXL_INLINE void FastUnsqueeze(const pixel_type *JXL_RESTRICT p_residual,
                              const pixel_type *JXL_RESTRICT p_avg,
                              const pixel_type *JXL_RESTRICT p_navg,
                              const pixel_type *p_pout,
                              pixel_type *JXL_RESTRICT p_out,
                              pixel_type *p_nout) {
  const HWY_CAPPED(pixel_type, 8) d;
  const RebindToUnsigned<decltype(d)> du;
  const size_t N = Lanes(d);
  auto onethird = Set(d, 0x55555556);
  for (size_t x = 0; x < 8; x += N) {
    auto avg = Load(d, p_avg + x);
    auto next_avg = Load(d, p_navg + x);
    auto top = Load(d, p_pout + x);
    // Equivalent to SmoothTendency(top,avg,next_avg), but without branches
    // typo:off
    auto Ba = Sub(top, avg);
    auto an = Sub(avg, next_avg);
    auto nonmono = Xor(Ba, an);
    auto absBa = Abs(Ba);
    auto absan = Abs(an);
    auto absBn = Abs(Sub(top, next_avg));
    // Compute a3 = absBa / 3
    auto a3e = BitCast(d, ShiftRight<32>(MulEven(absBa, onethird)));
    auto a3oi = MulEven(Reverse(d, absBa), onethird);
    auto a3o = BitCast(
        d, Reverse(hwy::HWY_NAMESPACE::Repartition<pixel_type_w, decltype(d)>(),
                   a3oi));
    auto a3 = OddEven(a3o, a3e);
    a3 = Add(a3, Add(absBn, Set(d, 2)));
    auto absdiff = ShiftRight<2>(a3);
    auto skipdiff = Ne(Ba, Zero(d));
    skipdiff = And(skipdiff, Ne(an, Zero(d)));
    skipdiff = And(skipdiff, Lt(nonmono, Zero(d)));
    auto absBa2 = Add(ShiftLeft<1>(absBa), And(absdiff, Set(d, 1)));
    absdiff = IfThenElse(Gt(absdiff, absBa2),
                         Add(ShiftLeft<1>(absBa), Set(d, 1)), absdiff);
    // typo:on
    auto absan2 = ShiftLeft<1>(absan);
    absdiff = IfThenElse(Gt(Add(absdiff, And(absdiff, Set(d, 1))), absan2),
                         absan2, absdiff);
    auto diff1 = IfThenElse(Lt(top, next_avg), Neg(absdiff), absdiff);
    auto tendency = IfThenZeroElse(skipdiff, diff1);

    auto diff_minus_tendency = Load(d, p_residual + x);
    auto diff = Add(diff_minus_tendency, tendency);
    auto out =
        Add(avg, ShiftRight<1>(
                     Add(diff, BitCast(d, ShiftRight<31>(BitCast(du, diff))))));
    Store(out, d, p_out + x);
    Store(Sub(out, diff), d, p_nout + x);
  }
}

#endif

Status InvHSqueeze(Image &input, uint32_t c, uint32_t rc, ThreadPool *pool) {
  JXL_ENSURE(c < input.channel.size());
  JXL_ENSURE(rc < input.channel.size());
  Channel &chin = input.channel[c];
  const Channel &chin_residual = input.channel[rc];
  // These must be valid since we ran MetaApply already.
  JXL_ENSURE(chin.w == DivCeil(chin.w + chin_residual.w, 2));
  JXL_ENSURE(chin.h == chin_residual.h);
  JxlMemoryManager *memory_manager = input.memory_manager();

  if (chin_residual.w == 0) {
    // Short-circuit: output channel has same dimensions as input.
    input.channel[c].hshift--;
    return true;
  }

  // Note: chin.w >= chin_residual.w and at most 1 different.
  JXL_ASSIGN_OR_RETURN(Channel chout,
                       Channel::Create(memory_manager, chin.w + chin_residual.w,
                                       chin.h, chin.hshift - 1, chin.vshift));
  JXL_DEBUG_V(4,
              "Undoing horizontal squeeze of channel %i using residuals in "
              "channel %i (going from width %" PRIuS " to %" PRIuS ")",
              c, rc, chin.w, chout.w);

  if (chin_residual.h == 0) {
    // Short-circuit: channel with no pixels.
    input.channel[c] = std::move(chout);
    return true;
  }
  auto unsqueeze_row = [&](size_t y, size_t x0) {
    const pixel_type *JXL_RESTRICT p_residual = chin_residual.Row(y);
    const pixel_type *JXL_RESTRICT p_avg = chin.Row(y);
    pixel_type *JXL_RESTRICT p_out = chout.Row(y);
    for (size_t x = x0; x < chin_residual.w; x++) {
      pixel_type_w diff_minus_tendency = p_residual[x];
      pixel_type_w avg = p_avg[x];
      pixel_type_w next_avg = (x + 1 < chin.w ? p_avg[x + 1] : avg);
      pixel_type_w left = (x ? p_out[(x << 1) - 1] : avg);
      pixel_type_w tendency = SmoothTendency(left, avg, next_avg);
      pixel_type_w diff = diff_minus_tendency + tendency;
      pixel_type_w A = avg + (diff / 2);
      p_out[(x << 1)] = A;
      pixel_type_w B = A - diff;
      p_out[(x << 1) + 1] = B;
    }
    if (chout.w & 1) p_out[chout.w - 1] = p_avg[chin.w - 1];
  };

  // somewhat complicated trickery just to be able to SIMD this.
  // Horizontal unsqueeze has horizontal data dependencies, so we do
  // 8 rows at a time and treat it as a vertical unsqueeze of a
  // transposed 8x8 block (or 9x8 for one input).
  static constexpr const size_t kRowsPerThread = 8;
  const auto unsqueeze_span = [&](const uint32_t task,
                                  size_t /* thread */) -> Status {
    const size_t y0 = task * kRowsPerThread;
    const size_t rows = std::min(kRowsPerThread, chin.h - y0);
    size_t x = 0;

#if HWY_TARGET != HWY_SCALAR
    intptr_t onerow_in = chin.plane.PixelsPerRow();
    intptr_t onerow_inr = chin_residual.plane.PixelsPerRow();
    intptr_t onerow_out = chout.plane.PixelsPerRow();
    const pixel_type *JXL_RESTRICT p_residual = chin_residual.Row(y0);
    const pixel_type *JXL_RESTRICT p_avg = chin.Row(y0);
    pixel_type *JXL_RESTRICT p_out = chout.Row(y0);
    HWY_ALIGN pixel_type b_p_avg[9 * kRowsPerThread];
    HWY_ALIGN pixel_type b_p_residual[8 * kRowsPerThread];
    HWY_ALIGN pixel_type b_p_out_even[8 * kRowsPerThread];
    HWY_ALIGN pixel_type b_p_out_odd[8 * kRowsPerThread];
    HWY_ALIGN pixel_type b_p_out_evenT[8 * kRowsPerThread];
    HWY_ALIGN pixel_type b_p_out_oddT[8 * kRowsPerThread];
    const HWY_CAPPED(pixel_type, 8) d;
    const size_t N = Lanes(d);
    if (chin_residual.w > 16 && rows == kRowsPerThread) {
      for (; x < chin_residual.w - 9; x += 8) {
        Transpose8x8Block(p_residual + x, b_p_residual, onerow_inr);
        Transpose8x8Block(p_avg + x, b_p_avg, onerow_in);
        for (size_t y = 0; y < kRowsPerThread; y++) {
          b_p_avg[8 * 8 + y] = p_avg[x + 8 + onerow_in * y];
        }
        for (size_t i = 0; i < 8; i++) {
          FastUnsqueeze(
              b_p_residual + 8 * i, b_p_avg + 8 * i, b_p_avg + 8 * (i + 1),
              (x + i ? b_p_out_odd + 8 * ((x + i - 1) & 7) : b_p_avg + 8 * i),
              b_p_out_even + 8 * i, b_p_out_odd + 8 * i);
        }

        Transpose8x8Block(b_p_out_even, b_p_out_evenT, 8);
        Transpose8x8Block(b_p_out_odd, b_p_out_oddT, 8);
        for (size_t y = 0; y < kRowsPerThread; y++) {
          for (size_t i = 0; i < kRowsPerThread; i += N) {
            auto even = Load(d, b_p_out_evenT + 8 * y + i);
            auto odd = Load(d, b_p_out_oddT + 8 * y + i);
            StoreInterleaved(d, even, odd,
                             p_out + ((x + i) << 1) + onerow_out * y);
          }
        }
      }
    }
#endif
    for (size_t y = 0; y < rows; y++) {
      unsqueeze_row(y0 + y, x);
    }
    return true;
  };
  JXL_RETURN_IF_ERROR(RunOnPool(pool, 0, DivCeil(chin.h, kRowsPerThread),
                                ThreadPool::NoInit, unsqueeze_span,
                                "InvHorizontalSqueeze"));
  input.channel[c] = std::move(chout);
  return true;
}

Status InvVSqueeze(Image &input, uint32_t c, uint32_t rc, ThreadPool *pool) {
  JXL_ENSURE(c < input.channel.size());
  JXL_ENSURE(rc < input.channel.size());
  const Channel &chin = input.channel[c];
  const Channel &chin_residual = input.channel[rc];
  // These must be valid since we ran MetaApply already.
  JXL_ENSURE(chin.h == DivCeil(chin.h + chin_residual.h, 2));
  JXL_ENSURE(chin.w == chin_residual.w);
  JxlMemoryManager *memory_manager = input.memory_manager();

  if (chin_residual.h == 0) {
    // Short-circuit: output channel has same dimensions as input.
    input.channel[c].vshift--;
    return true;
  }

  // Note: chin.h >= chin_residual.h and at most 1 different.
  JXL_ASSIGN_OR_RETURN(
      Channel chout,
      Channel::Create(memory_manager, chin.w, chin.h + chin_residual.h,
                      chin.hshift, chin.vshift - 1));
  JXL_DEBUG_V(
      4,
      "Undoing vertical squeeze of channel %i using residuals in channel "
      "%i (going from height %" PRIuS " to %" PRIuS ")",
      c, rc, chin.h, chout.h);

  if (chin_residual.w == 0) {
    // Short-circuit: channel with no pixels.
    input.channel[c] = std::move(chout);
    return true;
  }

  static constexpr const int kColsPerThread = 64;
  const auto unsqueeze_slice = [&](const uint32_t task,
                                   size_t /* thread */) -> Status {
    const size_t x0 = task * kColsPerThread;
    const size_t x1 =
        std::min(static_cast<size_t>(task + 1) * kColsPerThread, chin.w);
    const size_t w = x1 - x0;
    // We only iterate up to std::min(chin_residual.h, chin.h) which is
    // always chin_residual.h.
    for (size_t y = 0; y < chin_residual.h; y++) {
      const pixel_type *JXL_RESTRICT p_residual = chin_residual.Row(y) + x0;
      const pixel_type *JXL_RESTRICT p_avg = chin.Row(y) + x0;
      const pixel_type *JXL_RESTRICT p_navg =
          chin.Row(y + 1 < chin.h ? y + 1 : y) + x0;
      pixel_type *JXL_RESTRICT p_out = chout.Row(y << 1) + x0;
      pixel_type *JXL_RESTRICT p_nout = chout.Row((y << 1) + 1) + x0;
      const pixel_type *p_pout = y > 0 ? chout.Row((y << 1) - 1) + x0 : p_avg;
      size_t x = 0;
#if HWY_TARGET != HWY_SCALAR
      for (; x + 7 < w; x += 8) {
        FastUnsqueeze(p_residual + x, p_avg + x, p_navg + x, p_pout + x,
                      p_out + x, p_nout + x);
      }
#endif
      for (; x < w; x++) {
        pixel_type_w avg = p_avg[x];
        pixel_type_w next_avg = p_navg[x];
        pixel_type_w top = p_pout[x];
        pixel_type_w tendency = SmoothTendency(top, avg, next_avg);
        pixel_type_w diff_minus_tendency = p_residual[x];
        pixel_type_w diff = diff_minus_tendency + tendency;
        pixel_type_w out = avg + (diff / 2);
        p_out[x] = out;
        // If the chin_residual.h == chin.h, the output has an even number
        // of rows so the next line is fine. Otherwise, this loop won't
        // write to the last output row which is handled separately.
        p_nout[x] = out - diff;
      }
    }
    return true;
  };
  JXL_RETURN_IF_ERROR(RunOnPool(pool, 0, DivCeil(chin.w, kColsPerThread),
                                ThreadPool::NoInit, unsqueeze_slice,
                                "InvVertSqueeze"));

  if (chout.h & 1) {
    size_t y = chin.h - 1;
    const pixel_type *p_avg = chin.Row(y);
    pixel_type *p_out = chout.Row(y << 1);
    for (size_t x = 0; x < chin.w; x++) {
      p_out[x] = p_avg[x];
    }
  }
  input.channel[c] = std::move(chout);
  return true;
}

Status InvSqueeze(Image &input, const std::vector<SqueezeParams> &parameters,
                  ThreadPool *pool) {
  for (int i = parameters.size() - 1; i >= 0; i--) {
    JXL_RETURN_IF_ERROR(
        CheckMetaSqueezeParams(parameters[i], input.channel.size()));
    bool horizontal = parameters[i].horizontal;
    bool in_place = parameters[i].in_place;
    uint32_t beginc = parameters[i].begin_c;
    uint32_t endc = parameters[i].begin_c + parameters[i].num_c - 1;
    uint32_t offset;
    if (in_place) {
      offset = endc + 1;
    } else {
      offset = input.channel.size() + beginc - endc - 1;
    }
    if (beginc < input.nb_meta_channels) {
      // This is checked in MetaSqueeze.
      JXL_ENSURE(input.nb_meta_channels > parameters[i].num_c);
      input.nb_meta_channels -= parameters[i].num_c;
    }

    for (uint32_t c = beginc; c <= endc; c++) {
      uint32_t rc = offset + c - beginc;
      // MetaApply should imply that `rc` is within range, otherwise there's a
      // programming bug.
      JXL_ENSURE(rc < input.channel.size());
      if ((input.channel[c].w < input.channel[rc].w) ||
          (input.channel[c].h < input.channel[rc].h)) {
        return JXL_FAILURE("Corrupted squeeze transform");
      }
      if (horizontal) {
        JXL_RETURN_IF_ERROR(InvHSqueeze(input, c, rc, pool));
      } else {
        JXL_RETURN_IF_ERROR(InvVSqueeze(input, c, rc, pool));
      }
    }
    input.channel.erase(input.channel.begin() + offset,
                        input.channel.begin() + offset + (endc - beginc + 1));
  }
  return true;
}

}  // namespace HWY_NAMESPACE
}  // namespace jxl
HWY_AFTER_NAMESPACE();

#if HWY_ONCE

namespace jxl {

HWY_EXPORT(InvSqueeze);
Status InvSqueeze(Image &input, const std::vector<SqueezeParams> &parameters,
                  ThreadPool *pool) {
  return HWY_DYNAMIC_DISPATCH(InvSqueeze)(input, parameters, pool);
}

void DefaultSqueezeParameters(std::vector<SqueezeParams> *parameters,
                              const Image &image) {
  int nb_channels = image.channel.size() - image.nb_meta_channels;

  parameters->clear();
  size_t w = image.channel[image.nb_meta_channels].w;
  size_t h = image.channel[image.nb_meta_channels].h;
  JXL_DEBUG_V(
      7, "Default squeeze parameters for %" PRIuS "x%" PRIuS " image: ", w, h);

  // do horizontal first on wide images; vertical first on tall images
  bool wide = (w > h);

  if (nb_channels > 2 && image.channel[image.nb_meta_channels + 1].w == w &&
      image.channel[image.nb_meta_channels + 1].h == h) {
    // assume channels 1 and 2 are chroma, and can be squeezed first for 4:2:0
    // previews
    JXL_DEBUG_V(7, "(4:2:0 chroma), %" PRIuS "x%" PRIuS " image", w, h);
    SqueezeParams params;
    // horizontal chroma squeeze
    params.horizontal = true;
    params.in_place = false;
    params.begin_c = image.nb_meta_channels + 1;
    params.num_c = 2;
    parameters->push_back(params);
    params.horizontal = false;
    // vertical chroma squeeze
    parameters->push_back(params);
  }
  SqueezeParams params;
  params.begin_c = image.nb_meta_channels;
  params.num_c = nb_channels;
  params.in_place = true;

  if (!wide) {
    if (h > kMaxFirstPreviewSize) {
      params.horizontal = false;
      parameters->push_back(params);
      h = (h + 1) / 2;
      JXL_DEBUG_V(7, "Vertical (%" PRIuS "x%" PRIuS "), ", w, h);
    }
  }
  while (w > kMaxFirstPreviewSize || h > kMaxFirstPreviewSize) {
    if (w > kMaxFirstPreviewSize) {
      params.horizontal = true;
      parameters->push_back(params);
      w = (w + 1) / 2;
      JXL_DEBUG_V(7, "Horizontal (%" PRIuS "x%" PRIuS "), ", w, h);
    }
    if (h > kMaxFirstPreviewSize) {
      params.horizontal = false;
      parameters->push_back(params);
      h = (h + 1) / 2;
      JXL_DEBUG_V(7, "Vertical (%" PRIuS "x%" PRIuS "), ", w, h);
    }
  }
  JXL_DEBUG_V(7, "that's it");
}

Status CheckMetaSqueezeParams(const SqueezeParams &parameter,
                              int num_channels) {
  int c1 = parameter.begin_c;
  int c2 = parameter.begin_c + parameter.num_c - 1;
  if (c1 < 0 || c1 >= num_channels || c2 < 0 || c2 >= num_channels || c2 < c1) {
    return JXL_FAILURE("Invalid channel range");
  }
  return true;
}

Status MetaSqueeze(Image &image, std::vector<SqueezeParams> *parameters) {
  JxlMemoryManager *memory_manager = image.memory_manager();
  if (parameters->empty()) {
    DefaultSqueezeParameters(parameters, image);
  }

  for (auto &parameter : *parameters) {
    JXL_RETURN_IF_ERROR(
        CheckMetaSqueezeParams(parameter, image.channel.size()));
    bool horizontal = parameter.horizontal;
    bool in_place = parameter.in_place;
    uint32_t beginc = parameter.begin_c;
    uint32_t endc = parameter.begin_c + parameter.num_c - 1;

    uint32_t offset;
    if (beginc < image.nb_meta_channels) {
      if (endc >= image.nb_meta_channels) {
        return JXL_FAILURE("Invalid squeeze: mix of meta and nonmeta channels");
      }
      if (!in_place) {
        return JXL_FAILURE(
            "Invalid squeeze: meta channels require in-place residuals");
      }
      image.nb_meta_channels += parameter.num_c;
    }
    if (in_place) {
      offset = endc + 1;
    } else {
      offset = image.channel.size();
    }
    for (uint32_t c = beginc; c <= endc; c++) {
      if (image.channel[c].hshift > 30 || image.channel[c].vshift > 30) {
        return JXL_FAILURE("Too many squeezes: shift > 30");
      }
      size_t w = image.channel[c].w;
      size_t h = image.channel[c].h;
      if (w == 0 || h == 0) return JXL_FAILURE("Squeezing empty channel");
      if (horizontal) {
        image.channel[c].w = (w + 1) / 2;
        if (image.channel[c].hshift >= 0) image.channel[c].hshift++;
        w = w - (w + 1) / 2;
      } else {
        image.channel[c].h = (h + 1) / 2;
        if (image.channel[c].vshift >= 0) image.channel[c].vshift++;
        h = h - (h + 1) / 2;
      }
      JXL_RETURN_IF_ERROR(image.channel[c].shrink());
      JXL_ASSIGN_OR_RETURN(Channel placeholder,
                           Channel::Create(memory_manager, w, h));
      placeholder.hshift = image.channel[c].hshift;
      placeholder.vshift = image.channel[c].vshift;

      image.channel.insert(image.channel.begin() + offset + (c - beginc),
                           std::move(placeholder));
      JXL_DEBUG_V(8, "MetaSqueeze applied, current image: %s",
                  image.DebugString().c_str());
    }
  }
  return true;
}

}  // namespace jxl

#endif
