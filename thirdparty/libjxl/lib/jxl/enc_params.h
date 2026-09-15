// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#ifndef LIB_JXL_ENC_PARAMS_H_
#define LIB_JXL_ENC_PARAMS_H_

// Parameters and flags that govern JXL compression.

#include <jxl/cms_interface.h>
#include <jxl/encode.h>
#include <stddef.h>

#include <vector>

#include "lib/jxl/base/override.h"
#include "lib/jxl/common.h"
#include "lib/jxl/enc_progressive_split.h"
#include "lib/jxl/frame_dimensions.h"
#include "lib/jxl/frame_header.h"
#include "lib/jxl/modular/encoding/dec_ma.h"
#include "lib/jxl/modular/options.h"
#include "lib/jxl/splines.h"

namespace jxl {

// NOLINTNEXTLINE(clang-analyzer-optin.performance.Padding)
struct CompressParams {
  float butteraugli_distance = 1.0f;

  // explicit distances for extra channels (defaults to butteraugli_distance
  // when not set; value of -1 can be used to represent 'default')
  std::vector<float> ec_distance;

  // Try to achieve a maximum pixel-by-pixel error on each channel.
  bool max_error_mode = false;
  float max_error[3] = {0.0, 0.0, 0.0};

  bool disable_perceptual_optimizations = false;

  SpeedTier speed_tier = SpeedTier::kSquirrel;
  int brotli_effort = -1;

  // 0 = default.
  // 1 = slightly worse quality.
  // 4 = fastest speed, lowest quality
  size_t decoding_speed_tier = 0;

  ColorTransform color_transform = ColorTransform::kXYB;

  // If true, the "modular mode options" members below are used.
  bool modular_mode = false;

  // Change group size in modular mode (0=128, 1=256, 2=512, 3=1024, -1=encoder
  // chooses).
  int modular_group_size_shift = -1;

  Override preview = Override::kDefault;
  Override noise = Override::kDefault;
  Override dots = Override::kDefault;
  Override patches = Override::kDefault;
  Override gaborish = Override::kDefault;
  int epf = -1;

  // Progressive mode.
  Override progressive_mode = Override::kDefault;

  // Quantized-progressive mode.
  Override qprogressive_mode = Override::kDefault;

  // Put center groups first in the bitstream.
  bool centerfirst = false;

  // Pixel coordinates of the center. First group will contain that center.
  size_t center_x = static_cast<size_t>(-1);
  size_t center_y = static_cast<size_t>(-1);

  int progressive_dc = -1;

  // If on: preserve color of invisible pixels (if off: don't care)
  // Default: on
  Override keep_invisible = Override::kDefault;

  JxlCmsInterface cms;
  bool cms_set = false;
  void SetCms(const JxlCmsInterface& cms) {
    this->cms = cms;
    cms_set = true;
  }

  // Force usage of CfL when doing JPEG recompression. This can have unexpected
  // effects on the decoded pixels, while still being JPEG-compliant and
  // allowing reconstruction of the original JPEG.
  bool force_cfl_jpeg_recompression = true;

  // Use brotli compression for any boxes derived from a JPEG frame.
  bool jpeg_compress_boxes = true;

  // Preserve this metadata when doing JPEG recompression.
  bool jpeg_keep_exif = true;
  bool jpeg_keep_xmp = true;
  bool jpeg_keep_jumbf = true;

  // Set the noise to what it would approximately be if shooting at the nominal
  // exposure for a given ISO setting on a 35mm camera.
  float photon_noise_iso = 0;

  // modular mode options below
  ModularOptions options;

  // TODO(eustas): use Override?
  int responsive = -1;
  int colorspace = -1;
  int move_to_front_from_channel = -1;

  // Use Global channel palette if #colors < this percentage of range
  float channel_colors_pre_transform_percent = 95.f;
  // Use Local channel palette if #colors < this percentage of range
  float channel_colors_percent = 80.f;
  int palette_colors = 1 << 10;  // up to 10-bit palette is probably worthwhile
  bool lossy_palette = false;

  // Returns whether these params are lossless as defined by SetLossless();
  bool IsLossless() const { return modular_mode && ModularPartIsLossless(); }

  bool ModularPartIsLossless() const {
    if (modular_mode) {
      // YCbCr is also considered lossless here since it's intended for
      // source material that is already YCbCr (we don't do the fwd transform)
      if (butteraugli_distance != 0 ||
          color_transform == jxl::ColorTransform::kXYB)
        return false;
    }
    for (float f : ec_distance) {
      if (f > 0) return false;
      if (f < 0 && butteraugli_distance != 0) return false;
    }
    // all modular channels are encoded at distance 0
    return true;
  }

  // Sets the parameters required to make the codec lossless.
  void SetLossless() {
    modular_mode = true;
    butteraugli_distance = 0.0f;
    for (float& f : ec_distance) f = 0.0f;
    color_transform = jxl::ColorTransform::kNone;
  }

  // Down/upsample the image before encoding / after decoding by this factor.
  // The resampling value can also be set to <= 0 to automatically choose based
  // on distance, however EncodeFrame doesn't support this, so it is
  // required to call PostInit() to set a valid positive resampling
  // value and altered butteraugli score if this is used.
  int resampling = -1;
  int ec_resampling = -1;
  // Skip the downsampling before encoding if this is true.
  bool already_downsampled = false;
  // Butteraugli target distance on the original full size image, this can be
  // different from butteraugli_distance if resampling was used.
  float original_butteraugli_distance = -1.0f;

  float quant_ac_rescale = 1.0;

  // Codestream level to conform to.
  // -1: don't care
  int level = -1;

  // See JXL_ENC_FRAME_SETTING_BUFFERING option value.
  int buffering = -1;
  // See JXL_ENC_FRAME_SETTING_USE_FULL_IMAGE_HEURISTICS option value.
  bool use_full_image_heuristics = true;

  std::vector<float> manual_noise;
  std::vector<float> manual_xyb_factors;

  // If not empty, this tree will be used for dc global section.
  // Used in jxl_from_tree tool.
  Tree custom_fixed_tree;
  // If not empty, these custom splines will be used instead of the computed
  // ones. Used in jxl_from_tee tool.
  Splines custom_splines;
  // If not null, overrides progressive mode settings. Used in decode_test.
  const ProgressiveMode* custom_progressive_mode = nullptr;

  JxlDebugImageCallback debug_image = nullptr;
  void* debug_image_opaque;
};

static constexpr float kMinButteraugliForDynamicAR = 0.5f;
static constexpr float kMinButteraugliForDots = 3.0f;
static constexpr float kMinButteraugliToSubtractOriginalPatches = 3.0f;

// Always off
static constexpr float kMinButteraugliForNoise = 99.0f;

// Minimum butteraugli distance the encoder accepts.
static constexpr float kMinButteraugliDistance = 0.001f;

// Tile size for encoder-side processing. Must be equal to color tile dim in the
// current implementation.
static constexpr size_t kEncTileDim = 64;
static constexpr size_t kEncTileDimInBlocks = kEncTileDim / kBlockDim;

}  // namespace jxl

#endif  // LIB_JXL_ENC_PARAMS_H_
