// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#ifndef LIB_JXL_MODULAR_ENCODING_ENC_ENCODING_H_
#define LIB_JXL_MODULAR_ENCODING_ENC_ENCODING_H_

#include <cstddef>
#include <cstdint>
#include <vector>

#include "lib/jxl/base/status.h"
#include "lib/jxl/enc_ans.h"
#include "lib/jxl/enc_bit_writer.h"
#include "lib/jxl/modular/encoding/dec_ma.h"
#include "lib/jxl/modular/modular_image.h"
#include "lib/jxl/modular/options.h"

namespace jxl {

struct AuxOut;
enum class LayerType : uint8_t;
struct GroupHeader;

Tree PredefinedTree(ModularOptions::TreeKind tree_kind, size_t total_pixels,
                    int bitdepth, int prevprop);

StatusOr<Tree> LearnTree(
    const Image *images, const ModularOptions *opts, uint32_t start,
    uint32_t stop,
    const std::vector<ModularMultiplierInfo> &multiplier_info = {});

// Default single-image compress.
Status ModularGenericCompress(const Image &image, const ModularOptions &opts,
                              BitWriter &writer, AuxOut *aux_out = nullptr,
                              LayerType layer = static_cast<LayerType>(0),
                              size_t group_id = 0);

// For encoding with a given tree.
Status ModularCompress(const Image &image, const ModularOptions &opts,
                       size_t group_id, const Tree &tree, GroupHeader &header,
                       std::vector<Token> &tokens, size_t *width);
}  // namespace jxl

#endif  // LIB_JXL_MODULAR_ENCODING_ENC_ENCODING_H_
