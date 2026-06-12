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
#include "draco/compression/point_cloud/algorithms/float_points_tree_decoder.h"

#include <algorithm>

#include "draco/compression/point_cloud/algorithms/dynamic_integer_points_kd_tree_decoder.h"
#include "draco/compression/point_cloud/algorithms/quantize_points_3.h"
#include "draco/core/math_utils.h"
#include "draco/core/quantization_utils.h"

namespace draco {

struct Converter {
  typedef std::vector<uint32_t> SourceType;
  typedef Point3ui TargetType;
  Point3ui operator()(const std::vector<uint32_t> &v) {
    return Point3ui(v[0], v[1], v[2]);
  }
};

// Output iterator that is used to decode values directly into the data buffer
// of the modified PointAttribute.
template <class OutputIterator, class Converter>
class ConversionOutputIterator {
  typedef ConversionOutputIterator<OutputIterator, Converter> Self;
  typedef typename Converter::SourceType SourceType;
  typedef typename Converter::TargetType TargetType;

 public:
  explicit ConversionOutputIterator(OutputIterator oit) : oit_(oit) {}

  const Self &operator++() {
    ++oit_;
    return *this;
  }
  Self operator++(int) {
    Self copy = *this;
    ++oit_;
    return copy;
  }
  Self &operator*() { return *this; }
  const Self &operator=(const SourceType &source) {
    *oit_ = Converter()(source);
    return *this;
  }

 private:
  OutputIterator oit_;
};

FloatPointsTreeDecoder::FloatPointsTreeDecoder()
    : num_points_(0), compression_level_(0), num_points_from_header_(0) {
  qinfo_.quantization_bits = 0;
  qinfo_.range = 0;
}

bool FloatPointsTreeDecoder::DecodePointCloudKdTreeInternal(
    DecoderBuffer *buffer, std::vector<Point3ui> *qpoints) {
  if (!buffer->Decode(&qinfo_.quantization_bits)) {
    return false;
  }
  if (qinfo_.quantization_bits > 31) {
    return false;
  }
  if (!buffer->Decode(&qinfo_.range)) {
    return false;
  }
  if (!buffer->Decode(&num_points_)) {
    return false;
  }
  if (num_points_from_header_ > 0 && num_points_ != num_points_from_header_) {
    return false;
  }
  if (!buffer->Decode(&compression_level_)) {
    return false;
  }

  // Only allow compression level in [0..6].
  if (6 < compression_level_) {
    DRACO_LOGE("FloatPointsTreeDecoder: compression level %i not supported.\n",
               compression_level_);
    return false;
  }

  std::back_insert_iterator<std::vector<Point3ui>> oit_qpoints =
      std::back_inserter(*qpoints);
  ConversionOutputIterator<std::back_insert_iterator<std::vector<Point3ui>>,
                           Converter>
      oit(oit_qpoints);
  if (num_points_ > 0) {
    qpoints->reserve(num_points_);
    switch (compression_level_) {
      case 0: {
        DynamicIntegerPointsKdTreeDecoder<0> qpoints_decoder(3);
        qpoints_decoder.DecodePoints(buffer, oit);
        break;
      }
      case 1: {
        DynamicIntegerPointsKdTreeDecoder<1> qpoints_decoder(3);
        qpoints_decoder.DecodePoints(buffer, oit);
        break;
      }
      case 2: {
        DynamicIntegerPointsKdTreeDecoder<2> qpoints_decoder(3);
        qpoints_decoder.DecodePoints(buffer, oit);
        break;
      }
      case 3: {
        DynamicIntegerPointsKdTreeDecoder<3> qpoints_decoder(3);
        qpoints_decoder.DecodePoints(buffer, oit);
        break;
      }
      case 4: {
        DynamicIntegerPointsKdTreeDecoder<4> qpoints_decoder(3);
        qpoints_decoder.DecodePoints(buffer, oit);
        break;
      }
      case 5: {
        DynamicIntegerPointsKdTreeDecoder<5> qpoints_decoder(3);
        qpoints_decoder.DecodePoints(buffer, oit);
        break;
      }
      case 6: {
        DynamicIntegerPointsKdTreeDecoder<6> qpoints_decoder(3);
        qpoints_decoder.DecodePoints(buffer, oit);
        break;
      }
      default:
        return false;
    }
  }

  if (qpoints->size() != num_points_) {
    return false;
  }
  return true;
}

}  // namespace draco
