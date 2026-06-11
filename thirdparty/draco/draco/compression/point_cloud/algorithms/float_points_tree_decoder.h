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
#ifndef DRACO_COMPRESSION_POINT_CLOUD_ALGORITHMS_FLOAT_POINTS_TREE_DECODER_H_
#define DRACO_COMPRESSION_POINT_CLOUD_ALGORITHMS_FLOAT_POINTS_TREE_DECODER_H_

#include <memory>

#include "draco/compression/config/compression_shared.h"
#include "draco/compression/point_cloud/algorithms/point_cloud_compression_method.h"
#include "draco/compression/point_cloud/algorithms/point_cloud_types.h"
#include "draco/compression/point_cloud/algorithms/quantize_points_3.h"
#include "draco/core/decoder_buffer.h"

namespace draco {

// Decodes a point cloud encoded by PointCloudTreeEncoder.
class FloatPointsTreeDecoder {
 public:
  FloatPointsTreeDecoder();

  // Decodes a point cloud from |buffer|.
  template <class OutputIteratorT>
  bool DecodePointCloud(DecoderBuffer *buffer, OutputIteratorT &out);

#ifndef DRACO_OLD_GCC
  template <class OutputIteratorT>
  bool DecodePointCloud(DecoderBuffer *buffer, OutputIteratorT &&out);
#endif  // DRACO_OLD_GCC

  // Initializes a DecoderBuffer from |data|, and calls function above.
  template <class OutputIteratorT>
  bool DecodePointCloud(const char *data, size_t data_size,
                        OutputIteratorT out) {
    if (data == 0 || data_size <= 0) {
      return false;
    }

    DecoderBuffer buffer;
    buffer.Init(data, data_size);
    buffer.set_bitstream_version(kDracoPointCloudBitstreamVersion);
    return DecodePointCloud(&buffer, out);
  }

  uint32_t quantization_bits() const { return qinfo_.quantization_bits; }
  uint32_t compression_level() const { return compression_level_; }
  float range() const { return qinfo_.range; }
  uint32_t num_points() const { return num_points_; }
  uint32_t version() const { return version_; }
  std::string identification_string() const {
    if (method_ == KDTREE) {
      return "FloatPointsTreeDecoder: IntegerPointsKDTreeDecoder";
    } else {
      return "FloatPointsTreeDecoder: Unsupported Method";
    }
  }

  void set_num_points_from_header(uint32_t num_points) {
    num_points_from_header_ = num_points;
  }

 private:
  bool DecodePointCloudKdTreeInternal(DecoderBuffer *buffer,
                                      std::vector<Point3ui> *qpoints);

  static const uint32_t version_ = 3;
  QuantizationInfo qinfo_;
  int8_t method_;
  uint32_t num_points_;
  uint32_t compression_level_;

  // Member variable to check if the number of points from the file header
  // matches the number of points in the compression header. If
  // |num_points_from_header_| is 0, do not perform the check. Defaults to 0.
  uint32_t num_points_from_header_;
};

#ifndef DRACO_OLD_GCC
// TODO(vytyaz): Reenable once USD migrates from GCC 4.8 to a higher version
// that can disambiguate calls to overloaded methods taking rvalue reference.
template <class OutputIteratorT>
bool FloatPointsTreeDecoder::DecodePointCloud(DecoderBuffer *buffer,
                                              OutputIteratorT &&out) {
  OutputIteratorT local = std::forward<OutputIteratorT>(out);
  return DecodePointCloud(buffer, local);
}
#endif  // DRACO_OLD_GCC

template <class OutputIteratorT>
bool FloatPointsTreeDecoder::DecodePointCloud(DecoderBuffer *buffer,
                                              OutputIteratorT &out) {
  std::vector<Point3ui> qpoints;

  uint32_t decoded_version;
  if (!buffer->Decode(&decoded_version)) {
    return false;
  }

  if (decoded_version == 3) {
    int8_t method_number;
    if (!buffer->Decode(&method_number)) {
      return false;
    }

    method_ = method_number;

    if (method_ == KDTREE) {
      if (!DecodePointCloudKdTreeInternal(buffer, &qpoints)) {
        return false;
      }
    } else {  // Unsupported method.
      fprintf(stderr, "Method not supported. \n");
      return false;
    }
  } else if (decoded_version == 2) {  // Version 2 only uses KDTREE method.
    if (!DecodePointCloudKdTreeInternal(buffer, &qpoints)) {
      return false;
    }
  } else {  // Unsupported version.
    fprintf(stderr, "Version not supported. \n");
    return false;
  }

  DequantizePoints3(qpoints.begin(), qpoints.end(), qinfo_, out);
  return true;
}

}  // namespace draco

#endif  // DRACO_COMPRESSION_POINT_CLOUD_ALGORITHMS_FLOAT_POINTS_TREE_DECODER_H_
