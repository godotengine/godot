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
#ifndef DRACO_COMPRESSION_POINT_CLOUD_ALGORITHMS_POINT_CLOUD_COMPRESSION_METHOD_H_
#define DRACO_COMPRESSION_POINT_CLOUD_ALGORITHMS_POINT_CLOUD_COMPRESSION_METHOD_H_

namespace draco {

// Enum indicating the used compression method, used by Encoder and Decoder.
enum PointCloudCompressionMethod {
  RESERVED_POINT_CLOUD_METHOD_0 = 0,  // Reserved for internal use.
  // Generalized version of Encoding using the Octree method by Olivier
  // Devillers to d dimensions.
  // "Progressive lossless compression of arbitrary simplicial complexes"
  // https://doi.org/10.1145/566570.566591
  KDTREE = 1,
  RESERVED_POINT_CLOUD_METHOD_2 = 2,  // Reserved for internal use.
  RESERVED_POINT_CLOUD_METHOD_3 = 0,  // Reserved for internal use.
};

}  // namespace draco

#endif  // DRACO_COMPRESSION_POINT_CLOUD_ALGORITHMS_POINT_CLOUD_COMPRESSION_METHOD_H_
