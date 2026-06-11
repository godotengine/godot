// Copyright 2016 The Draco Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// See dynamic_integer_points_kd_tree_encoder.h for documentation.

#ifndef DRACO_COMPRESSION_POINT_CLOUD_ALGORITHMS_DYNAMIC_INTEGER_POINTS_KD_TREE_DECODER_H_
#define DRACO_COMPRESSION_POINT_CLOUD_ALGORITHMS_DYNAMIC_INTEGER_POINTS_KD_TREE_DECODER_H_

#include <array>
#include <limits>
#include <memory>
#include <stack>
#include <vector>

#include "draco/compression/bit_coders/adaptive_rans_bit_decoder.h"
#include "draco/compression/bit_coders/direct_bit_decoder.h"
#include "draco/compression/bit_coders/folded_integer_bit_decoder.h"
#include "draco/compression/bit_coders/rans_bit_decoder.h"
#include "draco/compression/point_cloud/algorithms/point_cloud_types.h"
#include "draco/core/bit_utils.h"
#include "draco/core/decoder_buffer.h"
#include "draco/core/math_utils.h"

namespace draco {

template <int compression_level_t>
struct DynamicIntegerPointsKdTreeDecoderCompressionPolicy
    : public DynamicIntegerPointsKdTreeDecoderCompressionPolicy<
          compression_level_t - 1> {};

template <>
struct DynamicIntegerPointsKdTreeDecoderCompressionPolicy<0> {
  typedef DirectBitDecoder NumbersDecoder;
  typedef DirectBitDecoder AxisDecoder;
  typedef DirectBitDecoder HalfDecoder;
  typedef DirectBitDecoder RemainingBitsDecoder;
  static constexpr bool select_axis = false;
};

template <>
struct DynamicIntegerPointsKdTreeDecoderCompressionPolicy<2>
    : public DynamicIntegerPointsKdTreeDecoderCompressionPolicy<1> {
  typedef RAnsBitDecoder NumbersDecoder;
};

template <>
struct DynamicIntegerPointsKdTreeDecoderCompressionPolicy<4>
    : public DynamicIntegerPointsKdTreeDecoderCompressionPolicy<3> {
  typedef FoldedBit32Decoder<RAnsBitDecoder> NumbersDecoder;
};

template <>
struct DynamicIntegerPointsKdTreeDecoderCompressionPolicy<6>
    : public DynamicIntegerPointsKdTreeDecoderCompressionPolicy<5> {
  static constexpr bool select_axis = true;
};

// Decodes a point cloud encoded by DynamicIntegerPointsKdTreeEncoder.
template <int compression_level_t>
class DynamicIntegerPointsKdTreeDecoder {
  static_assert(compression_level_t >= 0, "Compression level must in [0..6].");
  static_assert(compression_level_t <= 6, "Compression level must in [0..6].");
  typedef DynamicIntegerPointsKdTreeDecoderCompressionPolicy<
      compression_level_t>
      Policy;

  typedef typename Policy::NumbersDecoder NumbersDecoder;
  typedef typename Policy::AxisDecoder AxisDecoder;
  typedef typename Policy::HalfDecoder HalfDecoder;
  typedef typename Policy::RemainingBitsDecoder RemainingBitsDecoder;
  typedef std::vector<uint32_t> VectorUint32;

 public:
  explicit DynamicIntegerPointsKdTreeDecoder(uint32_t dimension)
      : bit_length_(0),
        num_points_(0),
        num_decoded_points_(0),
        dimension_(dimension),
        p_(dimension, 0),
        axes_(dimension, 0),
        // Init the stack with the maximum depth of the tree.
        // +1 for a second leaf.
        base_stack_(32 * dimension + 1, VectorUint32(dimension, 0)),
        levels_stack_(32 * dimension + 1, VectorUint32(dimension, 0)) {}

  // Decodes an integer point cloud from |buffer|. Optional |oit_max_points| can
  // be used to tell the decoder the maximum number of points accepted by the
  // iterator.
  template <class OutputIteratorT>
  bool DecodePoints(DecoderBuffer *buffer, OutputIteratorT &oit);

  template <class OutputIteratorT>
  bool DecodePoints(DecoderBuffer *buffer, OutputIteratorT &oit,
                    uint32_t oit_max_points);

#ifndef DRACO_OLD_GCC
  template <class OutputIteratorT>
  bool DecodePoints(DecoderBuffer *buffer, OutputIteratorT &&oit);
  template <class OutputIteratorT>
  bool DecodePoints(DecoderBuffer *buffer, OutputIteratorT &&oit,
                    uint32_t oit_max_points);
#endif  // DRACO_OLD_GCC

  const uint32_t dimension() const { return dimension_; }

  // Returns the number of decoded points. Must be called after DecodePoints().
  uint32_t num_decoded_points() const { return num_decoded_points_; }

 private:
  uint32_t GetAxis(uint32_t num_remaining_points, const VectorUint32 &levels,
                   uint32_t last_axis);

  template <class OutputIteratorT>
  bool DecodeInternal(uint32_t num_points, OutputIteratorT &oit);

  void DecodeNumber(int nbits, uint32_t *value) {
    numbers_decoder_.DecodeLeastSignificantBits32(nbits, value);
  }

  struct DecodingStatus {
    DecodingStatus(uint32_t num_remaining_points_, uint32_t last_axis_,
                   uint32_t stack_pos_)
        : num_remaining_points(num_remaining_points_),
          last_axis(last_axis_),
          stack_pos(stack_pos_) {}

    uint32_t num_remaining_points;
    uint32_t last_axis;
    uint32_t stack_pos;  // used to get base and levels
  };

  uint32_t bit_length_;
  uint32_t num_points_;
  uint32_t num_decoded_points_;
  uint32_t dimension_;
  NumbersDecoder numbers_decoder_;
  RemainingBitsDecoder remaining_bits_decoder_;
  AxisDecoder axis_decoder_;
  HalfDecoder half_decoder_;
  VectorUint32 p_;
  VectorUint32 axes_;
  std::vector<VectorUint32> base_stack_;
  std::vector<VectorUint32> levels_stack_;
};

// Decodes a point cloud from |buffer|.
#ifndef DRACO_OLD_GCC
template <int compression_level_t>
template <class OutputIteratorT>
bool DynamicIntegerPointsKdTreeDecoder<compression_level_t>::DecodePoints(
    DecoderBuffer *buffer, OutputIteratorT &&oit) {
  return DecodePoints(buffer, oit, std::numeric_limits<uint32_t>::max());
}

template <int compression_level_t>
template <class OutputIteratorT>
bool DynamicIntegerPointsKdTreeDecoder<compression_level_t>::DecodePoints(
    DecoderBuffer *buffer, OutputIteratorT &&oit, uint32_t oit_max_points) {
  OutputIteratorT local = std::forward<OutputIteratorT>(oit);
  return DecodePoints(buffer, local, oit_max_points);
}
#endif  // DRACO_OLD_GCC

template <int compression_level_t>
template <class OutputIteratorT>
bool DynamicIntegerPointsKdTreeDecoder<compression_level_t>::DecodePoints(
    DecoderBuffer *buffer, OutputIteratorT &oit) {
  return DecodePoints(buffer, oit, std::numeric_limits<uint32_t>::max());
}

template <int compression_level_t>
template <class OutputIteratorT>
bool DynamicIntegerPointsKdTreeDecoder<compression_level_t>::DecodePoints(
    DecoderBuffer *buffer, OutputIteratorT &oit, uint32_t oit_max_points) {
  if (!buffer->Decode(&bit_length_)) {
    return false;
  }
  if (bit_length_ > 32) {
    return false;
  }
  if (!buffer->Decode(&num_points_)) {
    return false;
  }
  if (num_points_ == 0) {
    return true;
  }
  if (num_points_ > oit_max_points) {
    return false;
  }
  num_decoded_points_ = 0;

  if (!numbers_decoder_.StartDecoding(buffer)) {
    return false;
  }
  if (!remaining_bits_decoder_.StartDecoding(buffer)) {
    return false;
  }
  if (!axis_decoder_.StartDecoding(buffer)) {
    return false;
  }
  if (!half_decoder_.StartDecoding(buffer)) {
    return false;
  }

  if (!DecodeInternal(num_points_, oit)) {
    return false;
  }

  numbers_decoder_.EndDecoding();
  remaining_bits_decoder_.EndDecoding();
  axis_decoder_.EndDecoding();
  half_decoder_.EndDecoding();

  return true;
}

template <int compression_level_t>
uint32_t DynamicIntegerPointsKdTreeDecoder<compression_level_t>::GetAxis(
    uint32_t num_remaining_points, const VectorUint32 &levels,
    uint32_t last_axis) {
  if (!Policy::select_axis) {
    return DRACO_INCREMENT_MOD(last_axis, dimension_);
  }

  uint32_t best_axis = 0;
  if (num_remaining_points < 64) {
    for (uint32_t axis = 1; axis < dimension_; ++axis) {
      if (levels[best_axis] > levels[axis]) {
        best_axis = axis;
      }
    }
  } else {
    axis_decoder_.DecodeLeastSignificantBits32(4, &best_axis);
  }

  return best_axis;
}

template <int compression_level_t>
template <class OutputIteratorT>
bool DynamicIntegerPointsKdTreeDecoder<compression_level_t>::DecodeInternal(
    uint32_t num_points, OutputIteratorT &oit) {
  typedef DecodingStatus Status;
  base_stack_[0] = VectorUint32(dimension_, 0);
  levels_stack_[0] = VectorUint32(dimension_, 0);
  DecodingStatus init_status(num_points, 0, 0);
  std::stack<Status> status_stack;
  status_stack.push(init_status);

  // TODO(b/199760123): Use preallocated vector instead of stack.
  while (!status_stack.empty()) {
    const DecodingStatus status = status_stack.top();
    status_stack.pop();

    const uint32_t num_remaining_points = status.num_remaining_points;
    const uint32_t last_axis = status.last_axis;
    const uint32_t stack_pos = status.stack_pos;
    const VectorUint32 &old_base = base_stack_[stack_pos];
    const VectorUint32 &levels = levels_stack_[stack_pos];

    if (num_remaining_points > num_points) {
      return false;
    }

    const uint32_t axis = GetAxis(num_remaining_points, levels, last_axis);
    if (axis >= dimension_) {
      return false;
    }

    const uint32_t level = levels[axis];

    // All axes have been fully subdivided, just output points.
    if ((bit_length_ - level) == 0) {
      for (uint32_t i = 0; i < num_remaining_points; i++) {
        *oit = old_base;
        ++oit;
        ++num_decoded_points_;
      }
      continue;
    }

    DRACO_DCHECK_EQ(true, num_remaining_points != 0);

    // Fast decoding of remaining bits if number of points is 1 or 2.
    if (num_remaining_points <= 2) {
      // TODO(b/199760123): |axes_| not necessary, remove would change
      // bitstream!
      axes_[0] = axis;
      for (uint32_t i = 1; i < dimension_; i++) {
        axes_[i] = DRACO_INCREMENT_MOD(axes_[i - 1], dimension_);
      }
      for (uint32_t i = 0; i < num_remaining_points; ++i) {
        for (uint32_t j = 0; j < dimension_; j++) {
          p_[axes_[j]] = 0;
          const uint32_t num_remaining_bits = bit_length_ - levels[axes_[j]];
          if (num_remaining_bits) {
            if (!remaining_bits_decoder_.DecodeLeastSignificantBits32(
                    num_remaining_bits, &p_[axes_[j]])) {
              return false;
            }
          }
          p_[axes_[j]] = old_base[axes_[j]] | p_[axes_[j]];
        }
        *oit = p_;
        ++oit;
        ++num_decoded_points_;
      }
      continue;
    }

    if (num_decoded_points_ > num_points_) {
      return false;
    }

    const int num_remaining_bits = bit_length_ - level;
    const uint32_t modifier = 1 << (num_remaining_bits - 1);
    base_stack_[stack_pos + 1] = old_base;         // copy
    base_stack_[stack_pos + 1][axis] += modifier;  // new base

    const int incoming_bits = MostSignificantBit(num_remaining_points);

    uint32_t number = 0;
    DecodeNumber(incoming_bits, &number);

    uint32_t first_half = num_remaining_points / 2;
    if (first_half < number) {
      // Invalid |number|.
      return false;
    }
    first_half -= number;
    uint32_t second_half = num_remaining_points - first_half;

    if (first_half != second_half) {
      if (!half_decoder_.DecodeNextBit()) {
        std::swap(first_half, second_half);
      }
    }

    levels_stack_[stack_pos][axis] += 1;
    levels_stack_[stack_pos + 1] = levels_stack_[stack_pos];  // copy
    if (first_half) {
      status_stack.push(DecodingStatus(first_half, axis, stack_pos));
    }
    if (second_half) {
      status_stack.push(DecodingStatus(second_half, axis, stack_pos + 1));
    }
  }
  return true;
}

extern template class DynamicIntegerPointsKdTreeDecoder<0>;
extern template class DynamicIntegerPointsKdTreeDecoder<2>;
extern template class DynamicIntegerPointsKdTreeDecoder<4>;
extern template class DynamicIntegerPointsKdTreeDecoder<6>;

}  // namespace draco

#endif  // DRACO_COMPRESSION_POINT_CLOUD_ALGORITHMS_DYNAMIC_INTEGER_POINTS_KD_TREE_DECODER_H_
