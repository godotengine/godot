// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#ifndef LIB_JXL_MODULAR_ENCODING_ENC_MA_H_
#define LIB_JXL_MODULAR_ENCODING_ENC_MA_H_

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "lib/jxl/base/status.h"
#include "lib/jxl/enc_ans.h"
#include "lib/jxl/modular/encoding/dec_ma.h"
#include "lib/jxl/modular/modular_image.h"
#include "lib/jxl/modular/options.h"

namespace jxl {

// Struct to collect all the data needed to build a tree.
struct TreeSamples {
  bool HasSamples() const {
    return !residuals.empty() && !residuals[0].empty();
  }
  size_t NumDistinctSamples() const { return sample_counts.size(); }
  size_t NumSamples() const { return num_samples; }
  // Set the predictor to use. Must be called before adding any samples.
  Status SetPredictor(Predictor predictor,
                      ModularOptions::TreeMode wp_tree_mode);
  // Set the properties to use. Must be called before adding any samples.
  Status SetProperties(const std::vector<uint32_t> &properties,
                       ModularOptions::TreeMode wp_tree_mode);

  size_t Token(size_t pred, size_t i) const { return residuals[pred][i].tok; }
  size_t NBits(size_t pred, size_t i) const { return residuals[pred][i].nbits; }
  size_t Count(size_t i) const { return sample_counts[i]; }
  size_t PredictorIndex(Predictor predictor) const {
    const auto predictor_elem =
        std::find(predictors.begin(), predictors.end(), predictor);
    JXL_DASSERT(predictor_elem != predictors.end());
    return predictor_elem - predictors.begin();
  }
  size_t PropertyIndex(size_t property) const {
    const auto property_elem =
        std::find(props_to_use.begin(), props_to_use.end(), property);
    JXL_DASSERT(property_elem != props_to_use.end());
    return property_elem - props_to_use.begin();
  }
  size_t NumPropertyValues(size_t property_index) const {
    return compact_properties[property_index].size() + 1;
  }
  // Returns the *quantized* property value.
  size_t Property(size_t property_index, size_t i) const {
    return props[property_index][i];
  }
  int UnquantizeProperty(size_t property_index, uint32_t quant) const {
    JXL_DASSERT(quant < compact_properties[property_index].size());
    return compact_properties[property_index][quant];
  }

  Predictor PredictorFromIndex(size_t index) const {
    JXL_DASSERT(index < predictors.size());
    return predictors[index];
  }
  size_t PropertyFromIndex(size_t index) const {
    JXL_DASSERT(index < props_to_use.size());
    return props_to_use[index];
  }
  size_t NumPredictors() const { return predictors.size(); }
  size_t NumProperties() const { return props_to_use.size(); }

  // Preallocate data for a given number of samples. MUST be called before
  // adding any sample.
  void PrepareForSamples(size_t num_samples);
  // Add a sample.
  void AddSample(pixel_type_w pixel, const Properties &properties,
                 const pixel_type_w *predictions);
  // Pre-cluster property values.
  void PreQuantizeProperties(
      const StaticPropRange &range,
      const std::vector<ModularMultiplierInfo> &multiplier_info,
      const std::vector<uint32_t> &group_pixel_count,
      const std::vector<uint32_t> &channel_pixel_count,
      std::vector<pixel_type> &pixel_samples,
      std::vector<pixel_type> &diff_samples, size_t max_property_values);

  void AllSamplesDone() { dedup_table_ = std::vector<uint32_t>(); }

  uint32_t QuantizeProperty(uint32_t prop, pixel_type v) const {
    v = std::min(std::max(v, -kPropertyRange), kPropertyRange) + kPropertyRange;
    return property_mapping[prop][v];
  }

  // Swaps samples in position a and b. Does nothing if a == b.
  void Swap(size_t a, size_t b);

  // Cycles samples: a -> b -> c -> a. We assume a <= b <= c, so that we can
  // just call Swap(a, b) if b==c.
  void ThreeShuffle(size_t a, size_t b, size_t c);

 private:
  // TODO(veluca): as the total number of properties and predictors are known
  // before adding any samples, it might be better to interleave predictors,
  // properties and counts in a single vector to improve locality.
  // A first attempt at doing this actually results in much slower encoding,
  // possibly because of the more complex addressing.
  struct ResidualToken {
    uint8_t tok;
    uint8_t nbits;
  };
  // Residual information: token and number of extra bits, per predictor.
  std::vector<std::vector<ResidualToken>> residuals;
  // Number of occurrences of each sample.
  std::vector<uint16_t> sample_counts;
  // Property values, quantized to at most 256 distinct values.
  std::vector<std::vector<uint8_t>> props;
  // Decompactification info for `props`.
  std::vector<std::vector<int32_t>> compact_properties;
  // List of properties to use.
  std::vector<uint32_t> props_to_use;
  // List of predictors to use.
  std::vector<Predictor> predictors;
  // Mapping property value -> quantized property value.
  static constexpr int32_t kPropertyRange = 511;
  std::vector<std::vector<uint8_t>> property_mapping;
  // Number of samples seen.
  size_t num_samples = 0;
  // Table for deduplication.
  static constexpr uint32_t kDedupEntryUnused{static_cast<uint32_t>(-1)};
  std::vector<uint32_t> dedup_table_;

  // Functions for sample deduplication.
  bool IsSameSample(size_t a, size_t b) const;
  size_t Hash1(size_t a) const;
  size_t Hash2(size_t a) const;
  void InitTable(size_t log_size);
  // Returns true if `a` was already present in the table.
  bool AddToTableAndMerge(size_t a);
  void AddToTable(size_t a);
};

Status TokenizeTree(const Tree &tree, std::vector<Token> *tokens,
                    Tree *decoder_tree);

void CollectPixelSamples(const Image &image, const ModularOptions &options,
                         uint32_t group_id,
                         std::vector<uint32_t> &group_pixel_count,
                         std::vector<uint32_t> &channel_pixel_count,
                         std::vector<pixel_type> &pixel_samples,
                         std::vector<pixel_type> &diff_samples);

Status ComputeBestTree(TreeSamples &tree_samples, float threshold,
                       const std::vector<ModularMultiplierInfo> &mul_info,
                       StaticPropRange static_prop_range,
                       float fast_decode_multiplier, Tree *tree);

}  // namespace jxl
#endif  // LIB_JXL_MODULAR_ENCODING_ENC_MA_H_
