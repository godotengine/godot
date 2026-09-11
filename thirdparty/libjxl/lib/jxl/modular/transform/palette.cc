// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "lib/jxl/modular/transform/palette.h"

#include <jxl/memory_manager.h>

#include "lib/jxl/base/status.h"
#include "lib/jxl/modular/transform/transform.h"  // CheckEqualChannels

namespace jxl {

Status InvPalette(Image &input, uint32_t begin_c, uint32_t nb_colors,
                  uint32_t nb_deltas, Predictor predictor,
                  const weighted::Header &wp_header, ThreadPool *pool) {
  JxlMemoryManager *memory_manager = input.memory_manager();
  if (input.nb_meta_channels < 1) {
    return JXL_FAILURE("Error: Palette transform without palette.");
  }
  int nb = input.channel[0].h;
  uint32_t c0 = begin_c + 1;
  if (c0 >= input.channel.size()) {
    return JXL_FAILURE("Channel is out of range.");
  }
  size_t w = input.channel[c0].w;
  size_t h = input.channel[c0].h;
  if (nb < 1) return JXL_FAILURE("Corrupted transforms");
  for (int i = 1; i < nb; i++) {
    JXL_ASSIGN_OR_RETURN(Channel c, Channel::Create(memory_manager, w, h,
                                                    input.channel[c0].hshift,
                                                    input.channel[c0].vshift));
    input.channel.insert(input.channel.begin() + c0 + 1, std::move(c));
  }
  const Channel &palette = input.channel[0];
  const pixel_type *JXL_RESTRICT p_palette = input.channel[0].Row(0);
  intptr_t onerow = input.channel[0].plane.PixelsPerRow();
  intptr_t onerow_image = input.channel[c0].plane.PixelsPerRow();
  const int bit_depth = std::min(input.bitdepth, 24);

  if (w == 0) {
    // Nothing to do.
    // Avoid touching "empty" channels with non-zero height.
  } else if (nb_deltas == 0 && predictor == Predictor::Zero) {
    if (nb == 1) {
      const auto process_row = [&](const uint32_t task,
                                   size_t /* thread */) -> Status {
        const size_t y = task;
        pixel_type *p = input.channel[c0].Row(y);
        for (size_t x = 0; x < w; x++) {
          const int index =
              Clamp1<int>(p[x], 0, static_cast<pixel_type>(palette.w) - 1);
          p[x] = palette_internal::GetPaletteValue(p_palette, index, /*c=*/0,
                                                   /*palette_size=*/palette.w,
                                                   /*onerow=*/onerow,
                                                   /*bit_depth=*/bit_depth);
        }
        return true;
      };
      JXL_RETURN_IF_ERROR(RunOnPool(pool, 0, h, ThreadPool::NoInit, process_row,
                                    "UndoChannelPalette"));
    } else {
      const auto process_row = [&](const uint32_t task,
                                   size_t /* thread */) -> Status {
        const size_t y = task;
        std::vector<pixel_type *> p_out(nb);
        const pixel_type *p_index = input.channel[c0].Row(y);
        for (int c = 0; c < nb; c++) p_out[c] = input.channel[c0 + c].Row(y);
        for (size_t x = 0; x < w; x++) {
          const int index = p_index[x];
          for (int c = 0; c < nb; c++) {
            p_out[c][x] = palette_internal::GetPaletteValue(
                p_palette, index, /*c=*/c,
                /*palette_size=*/palette.w,
                /*onerow=*/onerow, /*bit_depth=*/bit_depth);
          }
        }
        return true;
      };
      JXL_RETURN_IF_ERROR(RunOnPool(pool, 0, h, ThreadPool::NoInit, process_row,
                                    "UndoPalette"));
    }
  } else {
    // Parallelized per channel.
    ImageI indices;
    ImageI &plane = input.channel[c0].plane;
    JXL_ASSIGN_OR_RETURN(
        indices, ImageI::Create(memory_manager, plane.xsize(), plane.ysize()));
    plane.Swap(indices);
    if (predictor == Predictor::Weighted) {
      const auto process_row = [&](const uint32_t c,
                                   size_t /* thread */) -> Status {
        Channel &channel = input.channel[c0 + c];
        weighted::State wp_state(wp_header, channel.w, channel.h);
        for (size_t y = 0; y < channel.h; y++) {
          pixel_type *JXL_RESTRICT p = channel.Row(y);
          const pixel_type *JXL_RESTRICT idx = indices.Row(y);
          for (size_t x = 0; x < channel.w; x++) {
            int index = idx[x];
            pixel_type_w val = 0;
            const pixel_type palette_entry = palette_internal::GetPaletteValue(
                p_palette, index, /*c=*/c,
                /*palette_size=*/palette.w, /*onerow=*/onerow,
                /*bit_depth=*/bit_depth);
            if (index < static_cast<int32_t>(nb_deltas)) {
              PredictionResult pred = PredictNoTreeWP(
                  channel.w, p + x, onerow_image, x, y, predictor, &wp_state);
              val = pred.guess + palette_entry;
            } else {
              val = palette_entry;
            }
            p[x] = val;
            wp_state.UpdateErrors(p[x], x, y, channel.w);
          }
        }
        return true;
      };
      JXL_RETURN_IF_ERROR(RunOnPool(pool, 0, nb, ThreadPool::NoInit,
                                    process_row, "UndoDeltaPaletteWP"));
    } else {
      const auto process_row = [&](const uint32_t c,
                                   size_t /* thread */) -> Status {
        Channel &channel = input.channel[c0 + c];
        for (size_t y = 0; y < channel.h; y++) {
          pixel_type *JXL_RESTRICT p = channel.Row(y);
          const pixel_type *JXL_RESTRICT idx = indices.Row(y);
          for (size_t x = 0; x < channel.w; x++) {
            int index = idx[x];
            pixel_type_w val = 0;
            const pixel_type palette_entry = palette_internal::GetPaletteValue(
                p_palette, index, /*c=*/c,
                /*palette_size=*/palette.w,
                /*onerow=*/onerow, /*bit_depth=*/bit_depth);
            if (index < static_cast<int32_t>(nb_deltas)) {
              PredictionResult pred = PredictNoTreeNoWP(
                  channel.w, p + x, onerow_image, x, y, predictor);
              val = pred.guess + palette_entry;
            } else {
              val = palette_entry;
            }
            p[x] = val;
          }
        }
        return true;
      };
      JXL_RETURN_IF_ERROR(RunOnPool(pool, 0, nb, ThreadPool::NoInit,
                                    process_row, "UndoDeltaPaletteNoWP"));
    }
  }
  if (c0 >= input.nb_meta_channels) {
    // Palette was done on normal channels
    input.nb_meta_channels--;
  } else {
    // Palette was done on metachannels
    JXL_ENSURE(static_cast<int>(input.nb_meta_channels) >= 2 - nb);
    input.nb_meta_channels -= 2 - nb;
    JXL_ENSURE(begin_c + nb - 1 < input.nb_meta_channels);
  }
  input.channel.erase(input.channel.begin(), input.channel.begin() + 1);
  return true;
}

Status MetaPalette(Image &input, uint32_t begin_c, uint32_t end_c,
                   uint32_t nb_colors, uint32_t nb_deltas, bool lossy) {
  JXL_RETURN_IF_ERROR(CheckEqualChannels(input, begin_c, end_c));
  JxlMemoryManager *memory_manager = input.memory_manager();

  size_t nb = end_c - begin_c + 1;
  if (begin_c >= input.nb_meta_channels) {
    // Palette was done on normal channels
    input.nb_meta_channels++;
  } else {
    // Palette was done on metachannels
    JXL_ENSURE(end_c < input.nb_meta_channels);
    // we remove nb-1 metachannels and add one
    input.nb_meta_channels += 2 - nb;
  }
  input.channel.erase(input.channel.begin() + begin_c + 1,
                      input.channel.begin() + end_c + 1);
  JXL_ASSIGN_OR_RETURN(
      Channel pch, Channel::Create(memory_manager, nb_colors + nb_deltas, nb));
  pch.hshift = -1;
  pch.vshift = -1;
  input.channel.insert(input.channel.begin(), std::move(pch));
  return true;
}

}  // namespace jxl
