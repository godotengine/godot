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
#include "lib/jxl/modular/encoding/enc_ma.h"
#include "lib/jxl/modular/modular_image.h"
#include "lib/jxl/modular/options.h"

namespace jxl {

struct AuxOut;
enum class LayerType : uint8_t;
struct GroupHeader;

Tree PredefinedTree(ModularOptions::TreeKind tree_kind, size_t total_pixels,
                    int bitdepth, int prevprop);

StatusOr<Tree> LearnTree(
    TreeSamples &&tree_samples, size_t total_pixels,
    const ModularOptions &options,
    const std::vector<ModularMultiplierInfo> &multiplier_info = {},
    StaticPropRange static_prop_range = {});

// TODO(veluca): make cleaner interfaces.

Status ModularGenericCompress(
    Image &image, const ModularOptions &opts, BitWriter *writer,
    AuxOut *aux_out = nullptr, LayerType layer = static_cast<LayerType>(0),
    size_t group_id = 0,
    // For gathering data for producing a global tree.
    TreeSamples *tree_samples = nullptr, size_t *total_pixels = nullptr,
    // For encoding with global tree.
    const Tree *tree = nullptr, GroupHeader *header = nullptr,
    std::vector<Token> *tokens = nullptr, size_t *widths = nullptr);
}  // namespace jxl

#endif  // LIB_JXL_MODULAR_ENCODING_ENC_ENCODING_H_
