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
// TODO(b/199760123): Make this a wrapper using
// DynamicIntegerPointsKdTreeDecoder.
//
// See integer_points_kd_tree_encoder.h for documentation.

#ifndef DRACO_COMPRESSION_POINT_CLOUD_ALGORITHMS_INTEGER_POINTS_KD_TREE_DECODER_H_
#define DRACO_COMPRESSION_POINT_CLOUD_ALGORITHMS_INTEGER_POINTS_KD_TREE_DECODER_H_

#include <array>
#include <memory>

#include "draco/compression/bit_coders/adaptive_rans_bit_decoder.h"
#include "draco/compression/bit_coders/direct_bit_decoder.h"
#include "draco/compression/bit_coders/folded_integer_bit_decoder.h"
#include "draco/compression/bit_coders/rans_bit_decoder.h"
#include "draco/compression/point_cloud/algorithms/point_cloud_types.h"
#include "draco/compression/point_cloud/algorithms/queuing_policy.h"
#include "draco/core/bit_utils.h"
#include "draco/core/decoder_buffer.h"
#include "draco/core/math_utils.h"

namespace draco {

template <int compression_level_t>
struct IntegerPointsKdTreeDecoderCompressionPolicy
    : public IntegerPointsKdTreeDecoderCompressionPolicy<compression_level_t -
                                                         1> {};

template <>
struct IntegerPointsKdTreeDecoderCompressionPolicy<0> {
  typedef DirectBitDecoder NumbersDecoder;
  typedef DirectBitDecoder AxisDecoder;
  typedef DirectBitDecoder HalfDecoder;
  typedef DirectBitDecoder RemainingBitsDecoder;
  static constexpr bool select_axis = false;

  template <class T>
  using QueuingStrategy = Stack<T>;
};

template <>
struct IntegerPointsKdTreeDecoderCompressionPolicy<2>
    : public IntegerPointsKdTreeDecoderCompressionPolicy<1> {
  typedef RAnsBitDecoder NumbersDecoder;
};

template <>
struct IntegerPointsKdTreeDecoderCompressionPolicy<4>
    : public IntegerPointsKdTreeDecoderCompressionPolicy<3> {
  typedef FoldedBit32Decoder<RAnsBitDecoder> NumbersDecoder;
};

template <>
struct IntegerPointsKdTreeDecoderCompressionPolicy<6>
    : public IntegerPointsKdTreeDecoderCompressionPolicy<5> {
  static constexpr bool select_axis = true;
};

template <>
struct IntegerPointsKdTreeDecoderCompressionPolicy<8>
    : public IntegerPointsKdTreeDecoderCompressionPolicy<7> {
  typedef FoldedBit32Decoder<AdaptiveRAnsBitDecoder> NumbersDecoder;
  template <class T>
  using QueuingStrategy = Queue<T>;
};

template <>
struct IntegerPointsKdTreeDecoderCompressionPolicy<10>
    : public IntegerPointsKdTreeDecoderCompressionPolicy<9> {
  template <class T>
  using QueuingStrategy = PriorityQueue<T>;
};

// Decodes a point cloud encoded by IntegerPointsKdTreeEncoder.
// |PointDiT| is a type representing a point with uint32_t coordinates.
// must provide construction from three uint32_t and operator[].
template <class PointDiT, int compression_level_t>
class IntegerPointsKdTreeDecoder {
  typedef IntegerPointsKdTreeDecoderCompressionPolicy<compression_level_t>
      Policy;

  typedef typename Policy::NumbersDecoder NumbersDecoder;
  typedef typename Policy::AxisDecoder AxisDecoder;
  typedef typename Policy::HalfDecoder HalfDecoder;
  typedef typename Policy::RemainingBitsDecoder RemainingBitsDecoder;

 public:
  IntegerPointsKdTreeDecoder() : bit_length_(0) {}

  // Decodes a integer point cloud from |buffer|.
  template <class OutputIteratorT>
  bool DecodePoints(DecoderBuffer *buffer, OutputIteratorT oit);

 private:
  // For the sake of readability of code, we decided to make this exception
  // from the naming scheme.
  static constexpr int D = PointTraits<PointDiT>::Dimension();

  uint32_t GetAxis(uint32_t num_remaining_points, const PointDiT &base,
                   std::array<uint32_t, D> levels, uint32_t last_axis);

  template <class OutputIteratorT>
  void DecodeInternal(uint32_t num_remaining_points, PointDiT base,
                      std::array<uint32_t, D> levels, uint32_t last_axis,
                      OutputIteratorT oit);

  void DecodeNumber(int nbits, uint32_t *value) {
    numbers_decoder_.DecodeLeastSignificantBits32(nbits, value);
  }

  struct DecodingStatus {
    DecodingStatus(
        uint32_t num_remaining_points_, const PointDiT &old_base_,
        std::array<uint32_t, PointTraits<PointDiT>::Dimension()> levels_,
        uint32_t last_axis_)
        : num_remaining_points(num_remaining_points_),
          old_base(old_base_),
          levels(levels_),
          last_axis(last_axis_) {}

    uint32_t num_remaining_points;
    PointDiT old_base;
    std::array<uint32_t, D> levels;
    uint32_t last_axis;
    friend bool operator<(const DecodingStatus &l, const DecodingStatus &r) {
      return l.num_remaining_points < r.num_remaining_points;
    }
  };

  uint32_t bit_length_;
  uint32_t num_points_;
  NumbersDecoder numbers_decoder_;
  RemainingBitsDecoder remaining_bits_decoder_;
  AxisDecoder axis_decoder_;
  HalfDecoder half_decoder_;
};

// Decodes a point cloud from |buffer|.
template <class PointDiT, int compression_level_t>
template <class OutputIteratorT>
bool IntegerPointsKdTreeDecoder<PointDiT, compression_level_t>::DecodePoints(
    DecoderBuffer *buffer, OutputIteratorT oit) {
  if (!buffer->Decode(&bit_length_)) {
    return false;
  }
  if (!buffer->Decode(&num_points_)) {
    return false;
  }
  if (num_points_ == 0) {
    return true;
  }

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

  DecodeInternal(num_points_, PointTraits<PointDiT>::Origin(),
                 PointTraits<PointDiT>::ZeroArray(), 0, oit);

  numbers_decoder_.EndDecoding();
  remaining_bits_decoder_.EndDecoding();
  axis_decoder_.EndDecoding();
  half_decoder_.EndDecoding();

  return true;
}

template <class PointDiT, int compression_level_t>
uint32_t IntegerPointsKdTreeDecoder<PointDiT, compression_level_t>::GetAxis(
    uint32_t num_remaining_points, const PointDiT & /* base */,
    std::array<uint32_t, D> levels, uint32_t last_axis) {
  if (!Policy::select_axis) {
    return DRACO_INCREMENT_MOD(last_axis, D);
  }

  uint32_t best_axis = 0;
  if (num_remaining_points < 64) {
    for (uint32_t axis = 1; axis < D; ++axis) {
      if (levels[best_axis] > levels[axis]) {
        best_axis = axis;
      }
    }
  } else {
    axis_decoder_.DecodeLeastSignificantBits32(4, &best_axis);
  }

  return best_axis;
}

template <class PointDiT, int compression_level_t>
template <class OutputIteratorT>
void IntegerPointsKdTreeDecoder<PointDiT, compression_level_t>::DecodeInternal(
    uint32_t num_remaining_points, PointDiT old_base,
    std::array<uint32_t, D> levels, uint32_t last_axis, OutputIteratorT oit) {
  DecodingStatus init_status(num_remaining_points, old_base, levels, last_axis);
  typename Policy::template QueuingStrategy<DecodingStatus> status_q;
  status_q.push(init_status);

  while (!status_q.empty()) {
    const DecodingStatus status = status_q.front();
    status_q.pop();

    num_remaining_points = status.num_remaining_points;
    old_base = status.old_base;
    levels = status.levels;
    last_axis = status.last_axis;

    const uint32_t axis =
        GetAxis(num_remaining_points, old_base, levels, last_axis);

    const uint32_t level = levels[axis];

    // All axes have been fully subdivided, just output points.
    if ((bit_length_ - level) == 0) {
      for (int i = 0; i < static_cast<int>(num_remaining_points); i++) {
        *oit++ = old_base;
      }
      continue;
    }

    DRACO_DCHECK_EQ(true, num_remaining_points != 0);
    if (num_remaining_points <= 2) {
      std::array<uint32_t, D> axes;
      axes[0] = axis;
      for (int i = 1; i < D; i++) {
        axes[i] = DRACO_INCREMENT_MOD(axes[i - 1], D);
      }

      std::array<uint32_t, D> num_remaining_bits;
      for (int i = 0; i < D; i++) {
        num_remaining_bits[i] = bit_length_ - levels[axes[i]];
      }

      for (uint32_t i = 0; i < num_remaining_points; ++i) {
        // Get remaining bits, mind the carry if not starting at x.
        PointDiT p = PointTraits<PointDiT>::Origin();
        for (int j = 0; j < static_cast<int>(D); j++) {
          if (num_remaining_bits[j]) {
            remaining_bits_decoder_.DecodeLeastSignificantBits32(
                num_remaining_bits[j], &p[axes[j]]);
          }
          p[axes[j]] = old_base[axes[j]] | p[axes[j]];
        }
        *oit++ = p;
      }
      continue;
    }

    const int num_remaining_bits = bit_length_ - level;
    const uint32_t modifier = 1 << (num_remaining_bits - 1);
    PointDiT new_base(old_base);
    new_base[axis] += modifier;

    const int incoming_bits = MostSignificantBit(num_remaining_points);

    uint32_t number = 0;
    DecodeNumber(incoming_bits, &number);

    uint32_t first_half = num_remaining_points / 2 - number;
    uint32_t second_half = num_remaining_points - first_half;

    if (first_half != second_half) {
      if (!half_decoder_.DecodeNextBit()) {
        std::swap(first_half, second_half);
      }
    }

    levels[axis] += 1;
    if (first_half) {
      status_q.push(DecodingStatus(first_half, old_base, levels, axis));
    }
    if (second_half) {
      status_q.push(DecodingStatus(second_half, new_base, levels, axis));
    }
  }
}

extern template class IntegerPointsKdTreeDecoder<Point3ui, 0>;
extern template class IntegerPointsKdTreeDecoder<Point3ui, 1>;
extern template class IntegerPointsKdTreeDecoder<Point3ui, 2>;
extern template class IntegerPointsKdTreeDecoder<Point3ui, 3>;
extern template class IntegerPointsKdTreeDecoder<Point3ui, 4>;
extern template class IntegerPointsKdTreeDecoder<Point3ui, 5>;
extern template class IntegerPointsKdTreeDecoder<Point3ui, 6>;
extern template class IntegerPointsKdTreeDecoder<Point3ui, 7>;
extern template class IntegerPointsKdTreeDecoder<Point3ui, 8>;
extern template class IntegerPointsKdTreeDecoder<Point3ui, 9>;
extern template class IntegerPointsKdTreeDecoder<Point3ui, 10>;

}  // namespace draco

#endif  // DRACO_COMPRESSION_POINT_CLOUD_ALGORITHMS_INTEGER_POINTS_KD_TREE_DECODER_H_
