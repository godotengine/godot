// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#ifndef LIB_JXL_COMMON_H_
#define LIB_JXL_COMMON_H_

// Shared constants.

#include <cstddef>

#ifndef JXL_HIGH_PRECISION
#define JXL_HIGH_PRECISION 1
#endif

// Macro that defines whether support for decoding JXL files to JPEG is enabled.
#ifndef JPEGXL_ENABLE_TRANSCODE_JPEG
#define JPEGXL_ENABLE_TRANSCODE_JPEG 1
#endif  // JPEGXL_ENABLE_TRANSCODE_JPEG

// Macro that defines whether support for decoding boxes is enabled.
#ifndef JPEGXL_ENABLE_BOXES
#define JPEGXL_ENABLE_BOXES 1
#endif  // JPEGXL_ENABLE_BOXES

namespace jxl {
// Some enums and typedefs used by more than one header file.

// Maximum number of passes in an image.
constexpr size_t kMaxNumPasses = 11;

// Maximum number of reference frames.
constexpr size_t kMaxNumReferenceFrames = 4;

enum class SpeedTier {
  // Try multiple combinations of Glacier flags for modular mode. Otherwise
  // like kGlacier.
  kTectonicPlate = -1,
  // Learn a global tree in Modular mode.
  kGlacier = 0,
  // Turns on FindBestQuantizationHQ loop. Equivalent to "guetzli" mode.
  kTortoise = 1,
  // Turns on FindBestQuantization butteraugli loop.
  kKitten = 2,
  // Turns on dots, patches, and spline detection by default, as well as full
  // context clustering. Default.
  kSquirrel = 3,
  // Turns on error diffusion and full AC strategy heuristics. Equivalent to
  // "fast" mode.
  kWombat = 4,
  // Turns on gaborish by default, non-default cmap, initial quant field.
  kHare = 5,
  // Turns on simple heuristics for AC strategy, quant field, and clustering;
  // also enables coefficient reordering.
  kCheetah = 6,
  // Turns off most encoder features. Does context clustering.
  // Modular: uses fixed tree with Weighted predictor.
  kFalcon = 7,
  // Currently fastest possible setting for VarDCT.
  // Modular: uses fixed tree with Gradient predictor.
  kThunder = 8,
  // VarDCT: same as kThunder.
  // Modular: no tree, Gradient predictor, fast histograms
  kLightning = 9
};

}  // namespace jxl

#endif  // LIB_JXL_COMMON_H_
