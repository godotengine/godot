// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#ifndef LIB_JXL_RENDER_PIPELINE_RENDER_PIPELINE_STAGE_H_
#define LIB_JXL_RENDER_PIPELINE_RENDER_PIPELINE_STAGE_H_

#include <cstddef>
#include <utility>
#include <vector>

#include "lib/jxl/base/arch_macros.h"
#include "lib/jxl/base/status.h"
#include "lib/jxl/frame_header.h"

namespace jxl {

// The first pixel in the input to RenderPipelineStage will be located at
// this position. Pixels before this position may be accessed as padding.
// This should be at least the RoundUpTo(maximum padding / 2, maximum vector
// size) times 2: this is realized when using Gaborish + EPF + upsampling +
// chroma subsampling.
#if JXL_ARCH_ARM
constexpr size_t kRenderPipelineXOffset = 16;
#else
constexpr size_t kRenderPipelineXOffset = 32;
#endif

enum class RenderPipelineChannelMode {
  // This channel is not modified by this stage.
  kIgnored = 0,
  // This channel is modified in-place.
  kInPlace = 1,
  // This channel is modified and written to a new buffer.
  kInOut = 2,
  // This channel is only read. These are the only stages that are assumed to
  // have observable effects, i.e. calls to ProcessRow for other stages may be
  // omitted if it can be shown they can't affect any kInput stage ProcessRow
  // call that happens inside image boundaries.
  kInput = 3,
};

class RenderPipeline;

class RenderPipelineStage {
 protected:
  using Row = float*;
  using ChannelRows = std::vector<Row>;

 public:
  using RowInfo = std::vector<ChannelRows>;
  struct Settings {
    // Amount of padding required in the various directions by all channels
    // that have kInOut mode.
    size_t border_x = 0;
    size_t border_y = 0;

    // Log2 of the number of columns/rows of output that this stage will produce
    // for every input row for kInOut channels.
    size_t shift_x = 0;
    size_t shift_y = 0;

    static Settings ShiftX(size_t shift, size_t border) {
      Settings settings;
      settings.border_x = border;
      settings.shift_x = shift;
      return settings;
    }

    static Settings ShiftY(size_t shift, size_t border) {
      Settings settings;
      settings.border_y = border;
      settings.shift_y = shift;
      return settings;
    }

    static Settings Symmetric(size_t shift, size_t border) {
      Settings settings;
      settings.border_x = settings.border_y = border;
      settings.shift_x = settings.shift_y = shift;
      return settings;
    }

    static Settings SymmetricBorderOnly(size_t border) {
      return Symmetric(0, border);
    }
  };

  virtual ~RenderPipelineStage() = default;

  // Processes one row of input, producing the appropriate number of rows of
  // output. Input/output rows can be obtained by calls to
  // `GetInputRow`/`GetOutputRow`. `xsize+2*xextra` represents the total number
  // of pixels to be processed in the input row, where the first pixel is at
  // position `kRenderPipelineXOffset-xextra`. All pixels in the
  // `[kRenderPipelineXOffset-xextra-border_x,
  // kRenderPipelineXOffset+xsize+xextra+border_x)` range are initialized and
  // accessible. `xpos` and `ypos` represent the position of the first
  // (non-extra, i.e. in position kRenderPipelineXOffset) pixel in the center
  // row of the input in the full image. `xpos` is a multiple of
  // `GroupBorderAssigner::kPaddingXRound`. If `settings_.temp_buffer_size` is
  // nonzero, `temp` will point to an HWY-aligned buffer of at least that number
  // of floats; concurrent calls will have different buffers.
  virtual Status ProcessRow(const RowInfo& input_rows,
                            const RowInfo& output_rows, size_t xextra,
                            size_t xsize, size_t xpos, size_t ypos,
                            size_t thread_id) const = 0;

  // How each channel will be processed. Channels are numbered starting from
  // color channels (always 3) and followed by all other channels.
  virtual RenderPipelineChannelMode GetChannelMode(size_t c) const = 0;

 protected:
  explicit RenderPipelineStage(Settings settings) : settings_(settings) {}

  virtual Status IsInitialized() const { return true; }

  // Informs the stage about the total size of each channel. Few stages will
  // actually need to use this information.
  virtual Status SetInputSizes(
      const std::vector<std::pair<size_t, size_t>>& input_sizes) {
    return true;
  }

  virtual Status PrepareForThreads(size_t num_threads) { return true; }

  // Returns a pointer to the input row of channel `c` with offset `y`.
  // `y` must be in [-settings_.border_y, settings_.border_y]. `c` must be such
  // that `GetChannelMode(c) != kIgnored`. The returned pointer points to the
  // offset-ed row (i.e. kRenderPipelineXOffset has been applied).
  float* GetInputRow(const RowInfo& input_rows, size_t c, int offset) const {
    JXL_DASSERT(GetChannelMode(c) != RenderPipelineChannelMode::kIgnored);
    JXL_DASSERT(-offset <= static_cast<int>(settings_.border_y));
    JXL_DASSERT(offset <= static_cast<int>(settings_.border_y));
    return input_rows[c][settings_.border_y + offset] + kRenderPipelineXOffset;
  }
  // Similar to `GetInputRow`, but can only be used if `GetChannelMode(c) ==
  // kInOut`. Offset must be less than `1<<settings_.shift_y`.. The returned
  // pointer points to the offset-ed row (i.e. kRenderPipelineXOffset has been
  // applied).
  float* GetOutputRow(const RowInfo& output_rows, size_t c,
                      size_t offset) const {
    JXL_DASSERT(GetChannelMode(c) == RenderPipelineChannelMode::kInOut);
    JXL_DASSERT(offset <= 1ul << settings_.shift_y);
    return output_rows[c][offset] + kRenderPipelineXOffset;
  }

  // Indicates whether, from this stage on, the pipeline will operate on an
  // image- rather than frame-sized buffer. Only one stage in the pipeline
  // should return true, and it should implement ProcessPaddingRow below too.
  // It is assumed that, if there is a SwitchToImageDimensions() == true stage,
  // all kInput stages appear after it.
  virtual bool SwitchToImageDimensions() const { return false; }

  // If SwitchToImageDimensions returns true, then this should set xsize and
  // ysize to the image size, and frame_origin to the location of the frame
  // within the image. Otherwise, this is not called at all.
  virtual void GetImageDimensions(size_t* xsize, size_t* ysize,
                                  FrameOrigin* frame_origin) const {}

  // Produces the appropriate output data outside of the frame dimensions. xpos
  // and ypos are now relative to the full image.
  virtual void ProcessPaddingRow(const RowInfo& output_rows, size_t xsize,
                                 size_t xpos, size_t ypos) const {}

  virtual const char* GetName() const = 0;

  Settings settings_;
  friend class RenderPipeline;
  friend class SimpleRenderPipeline;
  friend class LowMemoryRenderPipeline;
};

}  // namespace jxl

#endif  // LIB_JXL_RENDER_PIPELINE_RENDER_PIPELINE_STAGE_H_
