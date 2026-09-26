// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#ifndef LIB_JXL_QUANT_WEIGHTS_H_
#define LIB_JXL_QUANT_WEIGHTS_H_

#include <jxl/memory_manager.h>

#include <array>
#include <cstdint>
#include <cstring>
#include <vector>

#include "lib/jxl/ac_strategy.h"
#include "lib/jxl/base/common.h"
#include "lib/jxl/base/compiler_specific.h"
#include "lib/jxl/base/status.h"
#include "lib/jxl/dec_bit_reader.h"
#include "lib/jxl/frame_dimensions.h"
#include "lib/jxl/memory_manager_internal.h"

namespace jxl {

static constexpr size_t kMaxQuantTableSize = AcStrategy::kMaxCoeffArea;
static constexpr size_t kNumPredefinedTables = 1;
static constexpr size_t kCeilLog2NumPredefinedTables = 0;
static constexpr size_t kLog2NumQuantModes = 3;

struct DctQuantWeightParams {
  static constexpr size_t kLog2MaxDistanceBands = 4;
  static constexpr size_t kMaxDistanceBands = 1 + (1 << kLog2MaxDistanceBands);
  typedef std::array<std::array<float, kMaxDistanceBands>, 3>
      DistanceBandsArray;

  size_t num_distance_bands = 0;
  DistanceBandsArray distance_bands = {};

  constexpr DctQuantWeightParams() : num_distance_bands(0) {}

  constexpr DctQuantWeightParams(const DistanceBandsArray& dist_bands,
                                 size_t num_dist_bands)
      : num_distance_bands(num_dist_bands), distance_bands(dist_bands) {}

  template <size_t num_dist_bands>
  explicit DctQuantWeightParams(const float dist_bands[3][num_dist_bands]) {
    num_distance_bands = num_dist_bands;
    for (size_t c = 0; c < 3; c++) {
      memcpy(distance_bands[c].data(), dist_bands[c],
             sizeof(float) * num_dist_bands);
    }
  }
};

// NOLINTNEXTLINE(clang-analyzer-optin.performance.Padding)
struct QuantEncodingInternal {
  enum Mode {
    kQuantModeLibrary,
    kQuantModeID,
    kQuantModeDCT2,
    kQuantModeDCT4,
    kQuantModeDCT4X8,
    kQuantModeAFV,
    kQuantModeDCT,
    kQuantModeRAW,
  };

  template <Mode mode>
  struct Tag {};

  typedef std::array<std::array<float, 3>, 3> IdWeights;
  typedef std::array<std::array<float, 6>, 3> DCT2Weights;
  typedef std::array<std::array<float, 2>, 3> DCT4Multipliers;
  typedef std::array<std::array<float, 9>, 3> AFVWeights;
  typedef std::array<float, 3> DCT4x8Multipliers;

  template <size_t A>
  static constexpr QuantEncodingInternal Library() {
    static_assert(A < kNumPredefinedTables);
    return QuantEncodingInternal(Tag<kQuantModeLibrary>(), A);
  }
  constexpr QuantEncodingInternal(Tag<kQuantModeLibrary> /* tag */,
                                  uint8_t predefined)
      : mode(kQuantModeLibrary), predefined(predefined) {}

  // Identity
  // xybweights is an array of {xweights, yweights, bweights}.
  static constexpr QuantEncodingInternal Identity(const IdWeights& xybweights) {
    return QuantEncodingInternal(Tag<kQuantModeID>(), xybweights);
  }
  constexpr QuantEncodingInternal(Tag<kQuantModeID> /* tag */,
                                  const IdWeights& xybweights)
      : mode(kQuantModeID), idweights(xybweights) {}

  // DCT2
  static constexpr QuantEncodingInternal DCT2(const DCT2Weights& xybweights) {
    return QuantEncodingInternal(Tag<kQuantModeDCT2>(), xybweights);
  }
  constexpr QuantEncodingInternal(Tag<kQuantModeDCT2> /* tag */,
                                  const DCT2Weights& xybweights)
      : mode(kQuantModeDCT2), dct2weights(xybweights) {}

  // DCT4
  static constexpr QuantEncodingInternal DCT4(
      const DctQuantWeightParams& params, const DCT4Multipliers& xybmul) {
    return QuantEncodingInternal(Tag<kQuantModeDCT4>(), params, xybmul);
  }
  constexpr QuantEncodingInternal(Tag<kQuantModeDCT4> /* tag */,
                                  const DctQuantWeightParams& params,
                                  const DCT4Multipliers& xybmul)
      : mode(kQuantModeDCT4), dct_params(params), dct4multipliers(xybmul) {}

  // DCT4x8
  static constexpr QuantEncodingInternal DCT4X8(
      const DctQuantWeightParams& params, const DCT4x8Multipliers& xybmul) {
    return QuantEncodingInternal(Tag<kQuantModeDCT4X8>(), params, xybmul);
  }
  constexpr QuantEncodingInternal(Tag<kQuantModeDCT4X8> /* tag */,
                                  const DctQuantWeightParams& params,
                                  const DCT4x8Multipliers& xybmul)
      : mode(kQuantModeDCT4X8), dct_params(params), dct4x8multipliers(xybmul) {}

  // DCT
  static constexpr QuantEncodingInternal DCT(
      const DctQuantWeightParams& params) {
    return QuantEncodingInternal(Tag<kQuantModeDCT>(), params);
  }
  constexpr QuantEncodingInternal(Tag<kQuantModeDCT> /* tag */,
                                  const DctQuantWeightParams& params)
      : mode(kQuantModeDCT), dct_params(params) {}

  // AFV
  static constexpr QuantEncodingInternal AFV(
      const DctQuantWeightParams& params4x8,
      const DctQuantWeightParams& params4x4, const AFVWeights& weights) {
    return QuantEncodingInternal(Tag<kQuantModeAFV>(), params4x8, params4x4,
                                 weights);
  }
  constexpr QuantEncodingInternal(Tag<kQuantModeAFV> /* tag */,
                                  const DctQuantWeightParams& params4x8,
                                  const DctQuantWeightParams& params4x4,
                                  const AFVWeights& weights)
      : mode(kQuantModeAFV),
        dct_params(params4x8),
        afv_weights(weights),
        dct_params_afv_4x4(params4x4) {}

  // This constructor is not constexpr so it can't be used in any of the
  // constexpr cases above.
  explicit QuantEncodingInternal(Mode mode) : mode(mode) {}

  Mode mode;

  // Weights for DCT4+ tables.
  DctQuantWeightParams dct_params;

  union {
    // Weights for identity.
    IdWeights idweights;

    // Weights for DCT2.
    DCT2Weights dct2weights;

    // Extra multipliers for coefficients 01/10 and 11 for DCT4 and AFV.
    DCT4Multipliers dct4multipliers;

    // Weights for AFV. {0, 1} are used directly for coefficients (0, 1) and (1,
    // 0);  {2, 3, 4} are used directly corner DC, (1,0) - (0,1) and (0, 1) +
    // (1, 0) - (0, 0) inside the AFV block. Values from 5 to 8 are interpolated
    // as in GetQuantWeights for DC and are used for other coefficients.
    AFVWeights afv_weights = {};

    // Extra multipliers for coefficients 01 or 10 for DCT4X8 and DCT8X4.
    DCT4x8Multipliers dct4x8multipliers;

    // Only used in kQuantModeRAW mode.
    struct {
      // explicit quantization table (like in JPEG)
      std::vector<int>* qtable = nullptr;
      float qtable_den = 1.f / (8 * 255);
    } qraw;
  };

  // Weights for 4x4 sub-block in AFV.
  DctQuantWeightParams dct_params_afv_4x4;

  union {
    // Which predefined table to use. Only used if mode is kQuantModeLibrary.
    uint8_t predefined = 0;

    // Which other quant table to copy; must copy from a table that comes before
    // the current one. Only used if mode is kQuantModeCopy.
    uint8_t source;
  };
};

class QuantEncoding final : public QuantEncodingInternal {
 public:
  QuantEncoding(const QuantEncoding& other)
      : QuantEncodingInternal(
            static_cast<const QuantEncodingInternal&>(other)) {
    if (mode == kQuantModeRAW && qraw.qtable) {
      // Need to make a copy of the passed *qtable.
      qraw.qtable = new std::vector<int>(*other.qraw.qtable);
    }
  }
  QuantEncoding(QuantEncoding&& other) noexcept
      : QuantEncodingInternal(
            static_cast<const QuantEncodingInternal&>(other)) {
    // Steal the qtable from the other object if any.
    if (mode == kQuantModeRAW) {
      other.qraw.qtable = nullptr;
    }
  }
  QuantEncoding& operator=(const QuantEncoding& other) {
    if (mode == kQuantModeRAW && qraw.qtable) {
      delete qraw.qtable;
    }
    *static_cast<QuantEncodingInternal*>(this) =
        QuantEncodingInternal(static_cast<const QuantEncodingInternal&>(other));
    if (mode == kQuantModeRAW && qraw.qtable) {
      // Need to make a copy of the passed *qtable.
      qraw.qtable = new std::vector<int>(*other.qraw.qtable);
    }
    return *this;
  }

  ~QuantEncoding() {
    if (mode == kQuantModeRAW && qraw.qtable) {
      delete qraw.qtable;
    }
  }

  // Wrappers of the QuantEncodingInternal:: static functions that return a
  // QuantEncoding instead. This is using the explicit and private cast from
  // QuantEncodingInternal to QuantEncoding, which would be inlined anyway.
  // In general, you should use this wrappers. The only reason to directly
  // create a QuantEncodingInternal instance is if you need a constexpr version
  // of this class. Note that RAW() is not supported in that case since it uses
  // a std::vector.
  template <size_t A>
  static QuantEncoding Library() {
    return QuantEncoding(QuantEncodingInternal::Library<A>());
  }
  static QuantEncoding Identity(const IdWeights& xybweights) {
    return QuantEncoding(QuantEncodingInternal::Identity(xybweights));
  }
  static QuantEncoding DCT2(const DCT2Weights& xybweights) {
    return QuantEncoding(QuantEncodingInternal::DCT2(xybweights));
  }
  static QuantEncoding DCT4(const DctQuantWeightParams& params,
                            const DCT4Multipliers& xybmul) {
    return QuantEncoding(QuantEncodingInternal::DCT4(params, xybmul));
  }
  static QuantEncoding DCT4X8(const DctQuantWeightParams& params,
                              const DCT4x8Multipliers& xybmul) {
    return QuantEncoding(QuantEncodingInternal::DCT4X8(params, xybmul));
  }
  static QuantEncoding DCT(const DctQuantWeightParams& params) {
    return QuantEncoding(QuantEncodingInternal::DCT(params));
  }
  static QuantEncoding AFV(const DctQuantWeightParams& params4x8,
                           const DctQuantWeightParams& params4x4,
                           const AFVWeights& weights) {
    return QuantEncoding(
        QuantEncodingInternal::AFV(params4x8, params4x4, weights));
  }

  // RAW, note that this one is not a constexpr one.
  static QuantEncoding RAW(std::vector<int>&& qtable, int shift = 0) {
    QuantEncoding encoding(kQuantModeRAW);
    encoding.qraw.qtable = new std::vector<int>();
    *encoding.qraw.qtable = qtable;
    encoding.qraw.qtable_den = (1 << shift) * (1.f / (8 * 255));
    return encoding;
  }

 private:
  explicit QuantEncoding(const QuantEncodingInternal& other)
      : QuantEncodingInternal(other) {}

  explicit QuantEncoding(QuantEncodingInternal::Mode mode_arg)
      : QuantEncodingInternal(mode_arg) {}
};

// A constexpr QuantEncodingInternal instance is often downcasted to the
// QuantEncoding subclass even if the instance wasn't an instance of the
// subclass. This is safe because user will upcast to QuantEncodingInternal to
// access any of its members.
static_assert(sizeof(QuantEncoding) == sizeof(QuantEncodingInternal),
              "Don't add any members to QuantEncoding");

// Let's try to keep these 2**N for possible future simplicity.
const float kInvDCQuant[3] = {
    4096.0f,
    512.0f,
    256.0f,
};

const float kDCQuant[3] = {
    1.0f / kInvDCQuant[0],
    1.0f / kInvDCQuant[1],
    1.0f / kInvDCQuant[2],
};

class ModularFrameEncoder;
class ModularFrameDecoder;

enum class QuantTable : size_t {
  DCT = 0,
  IDENTITY,
  DCT2X2,
  DCT4X4,
  DCT16X16,
  DCT32X32,
  // DCT16X8
  DCT8X16,
  // DCT32X8
  DCT8X32,
  // DCT32X16
  DCT16X32,
  DCT4X8,
  // DCT8X4
  AFV0,
  // AFV1
  // AFV2
  // AFV3
  DCT64X64,
  // DCT64X32,
  DCT32X64,
  DCT128X128,
  // DCT128X64,
  DCT64X128,
  DCT256X256,
  // DCT256X128,
  DCT128X256
};

static constexpr uint8_t kNumQuantTables =
    static_cast<uint8_t>(QuantTable::DCT128X256) + 1;

static const std::array<QuantTable, AcStrategy::kNumValidStrategies>
    kAcStrategyToQuantTableMap = {
        QuantTable::DCT,        QuantTable::IDENTITY,   QuantTable::DCT2X2,
        QuantTable::DCT4X4,     QuantTable::DCT16X16,   QuantTable::DCT32X32,
        QuantTable::DCT8X16,    QuantTable::DCT8X16,    QuantTable::DCT8X32,
        QuantTable::DCT8X32,    QuantTable::DCT16X32,   QuantTable::DCT16X32,
        QuantTable::DCT4X8,     QuantTable::DCT4X8,     QuantTable::AFV0,
        QuantTable::AFV0,       QuantTable::AFV0,       QuantTable::AFV0,
        QuantTable::DCT64X64,   QuantTable::DCT32X64,   QuantTable::DCT32X64,
        QuantTable::DCT128X128, QuantTable::DCT64X128,  QuantTable::DCT64X128,
        QuantTable::DCT256X256, QuantTable::DCT128X256, QuantTable::DCT128X256,
};

class DequantMatrices {
 public:
  DequantMatrices();

  static const QuantEncoding* Library();

  typedef std::array<QuantEncodingInternal,
                     kNumPredefinedTables * kNumQuantTables>
      DequantLibraryInternal;
  // Return the array of library kNumPredefinedTables QuantEncoding entries as
  // a constexpr array. Use Library() to obtain a pointer to the copy in the
  // .cc file.
  static DequantLibraryInternal LibraryInit();

  // Returns aligned memory.
  JXL_INLINE const float* Matrix(AcStrategyType quant_kind, size_t c) const {
    JXL_DASSERT((1 << static_cast<uint32_t>(quant_kind)) & computed_mask_);
    return &table_[table_offsets_[static_cast<size_t>(quant_kind) * 3 + c]];
  }

  JXL_INLINE const float* InvMatrix(AcStrategyType quant_kind, size_t c) const {
    size_t quant_table_idx = static_cast<uint32_t>(quant_kind);
    JXL_DASSERT((1 << quant_table_idx) & computed_mask_);
    return &inv_table_[table_offsets_[quant_table_idx * 3 + c]];
  }

  // DC quants are used in modular mode for XYB multipliers.
  JXL_INLINE float DCQuant(size_t c) const { return dc_quant_[c]; }
  JXL_INLINE const float* DCQuants() const { return dc_quant_; }

  JXL_INLINE float InvDCQuant(size_t c) const { return inv_dc_quant_[c]; }

  // For encoder.
  void SetEncodings(const std::vector<QuantEncoding>& encodings) {
    encodings_ = encodings;
    computed_mask_ = 0;
  }

  // For encoder.
  void SetDCQuant(const float dc[3]) {
    for (size_t c = 0; c < 3; c++) {
      dc_quant_[c] = 1.0f / dc[c];
      inv_dc_quant_[c] = dc[c];
    }
  }

  Status Decode(JxlMemoryManager* memory_manager, BitReader* br,
                ModularFrameDecoder* modular_frame_decoder = nullptr);
  Status DecodeDC(BitReader* br);

  const std::vector<QuantEncoding>& encodings() const { return encodings_; }

  static constexpr auto required_size_x =
      to_array<int>({1, 1, 1, 1, 2, 4, 1, 1, 2, 1, 1, 8, 4, 16, 8, 32, 16});
  static_assert(kNumQuantTables == required_size_x.size(),
                "Update this array when adding or removing quant tables.");

  static constexpr auto required_size_y =
      to_array<int>({1, 1, 1, 1, 2, 4, 2, 4, 4, 1, 1, 8, 8, 16, 16, 32, 32});
  static_assert(kNumQuantTables == required_size_y.size(),
                "Update this array when adding or removing quant tables.");

  // MUST be equal `sum(dot(required_size_x, required_size_y))`.
  static constexpr size_t kSumRequiredXy = 2056;

  Status EnsureComputed(JxlMemoryManager* memory_manager, uint32_t acs_mask);

 private:
  static constexpr size_t kTotalTableSize = kSumRequiredXy * kDCTBlockSize * 3;

  uint32_t computed_mask_ = 0;
  // kTotalTableSize entries followed by kTotalTableSize for inv_table
  AlignedMemory table_storage_;
  const float* table_;
  const float* inv_table_;
  float dc_quant_[3] = {kDCQuant[0], kDCQuant[1], kDCQuant[2]};
  float inv_dc_quant_[3] = {kInvDCQuant[0], kInvDCQuant[1], kInvDCQuant[2]};
  size_t table_offsets_[AcStrategy::kNumValidStrategies * 3];
  std::vector<QuantEncoding> encodings_;
};

}  // namespace jxl

#endif  // LIB_JXL_QUANT_WEIGHTS_H_
