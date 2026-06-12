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
#ifndef DRACO_COMPRESSION_POINT_CLOUD_POINT_CLOUD_SEQUENTIAL_DECODER_H_
#define DRACO_COMPRESSION_POINT_CLOUD_POINT_CLOUD_SEQUENTIAL_DECODER_H_

#include "draco/compression/point_cloud/point_cloud_decoder.h"

namespace draco {

// Point cloud decoder for data encoded by the PointCloudSequentialEncoder.
// All attribute values are decoded using an identity mapping between point ids
// and attribute value ids.
class PointCloudSequentialDecoder : public PointCloudDecoder {
 protected:
  bool DecodeGeometryData() override;
  bool CreateAttributesDecoder(int32_t att_decoder_id) override;
};

}  // namespace draco

#endif  // DRACO_COMPRESSION_POINT_CLOUD_POINT_CLOUD_SEQUENTIAL_DECODER_H_
