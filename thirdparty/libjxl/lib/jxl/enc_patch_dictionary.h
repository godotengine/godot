// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#ifndef LIB_JXL_ENC_PATCH_DICTIONARY_H_
#define LIB_JXL_ENC_PATCH_DICTIONARY_H_

// Chooses reference patches, and avoids encoding them once per occurrence.

#include <jxl/cms_interface.h>
#include <sys/types.h>

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <utility>
#include <vector>

#include "lib/jxl/base/compiler_specific.h"
#include "lib/jxl/base/data_parallel.h"
#include "lib/jxl/base/status.h"
#include "lib/jxl/dec_patch_dictionary.h"
#include "lib/jxl/enc_bit_writer.h"
#include "lib/jxl/enc_cache.h"
#include "lib/jxl/enc_params.h"
#include "lib/jxl/image.h"

namespace jxl {

struct AuxOut;
enum class LayerType : uint8_t;

constexpr size_t kMaxPatchSize = 32;

struct QuantizedPatch {
  size_t xsize;
  size_t ysize;
  QuantizedPatch() {
    for (size_t i = 0; i < 3; i++) {
      pixels[i].resize(kMaxPatchSize * kMaxPatchSize);
      fpixels[i].resize(kMaxPatchSize * kMaxPatchSize);
    }
  }
  std::vector<int8_t> pixels[3] = {};
  // Not compared. Used only to retrieve original pixels to construct the
  // reference image.
  std::vector<float> fpixels[3] = {};
  bool operator==(const QuantizedPatch& other) const {
    if (xsize != other.xsize) return false;
    if (ysize != other.ysize) return false;
    for (size_t c = 0; c < 3; c++) {
      if (memcmp(pixels[c].data(), other.pixels[c].data(),
                 sizeof(int8_t) * xsize * ysize) != 0)
        return false;
    }
    return true;
  }

  bool operator<(const QuantizedPatch& other) const {
    if (xsize != other.xsize) return xsize < other.xsize;
    if (ysize != other.ysize) return ysize < other.ysize;
    for (size_t c = 0; c < 3; c++) {
      int cmp = memcmp(pixels[c].data(), other.pixels[c].data(),
                       sizeof(int8_t) * xsize * ysize);
      if (cmp > 0) return false;
      if (cmp < 0) return true;
    }
    return false;
  }
};

// Pair (patch, vector of occurrences).
using PatchInfo =
    std::pair<QuantizedPatch, std::vector<std::pair<uint32_t, uint32_t>>>;

// Friend class of PatchDictionary.
class PatchDictionaryEncoder {
 public:
  // Only call if HasAny().
  static Status Encode(const PatchDictionary& pdic, BitWriter* writer,
                       LayerType layer, AuxOut* aux_out);

  static void SetPositions(PatchDictionary* pdic,
                           std::vector<PatchPosition> positions,
                           std::vector<PatchReferencePosition> ref_positions,
                           std::vector<PatchBlending> blendings,
                           size_t blendings_stride) {
    pdic->positions_ = std::move(positions);
    pdic->ref_positions_ = std::move(ref_positions);
    pdic->blendings_ = std::move(blendings);
    pdic->blendings_stride_ = blendings_stride;
    pdic->ComputePatchTree();
  }

  static Status SubtractFrom(const PatchDictionary& pdic, Image3F* opsin);
};

Status FindBestPatchDictionary(const Image3F& opsin,
                               PassesEncoderState* JXL_RESTRICT state,
                               const JxlCmsInterface& cms, ThreadPool* pool,
                               AuxOut* aux_out, bool is_xyb = true);

Status RoundtripPatchFrame(Image3F* reference_frame,
                           PassesEncoderState* JXL_RESTRICT state, int idx,
                           CompressParams& cparams, const JxlCmsInterface& cms,
                           ThreadPool* pool, AuxOut* aux_out, bool subtract);

}  // namespace jxl

#endif  // LIB_JXL_ENC_PATCH_DICTIONARY_H_
