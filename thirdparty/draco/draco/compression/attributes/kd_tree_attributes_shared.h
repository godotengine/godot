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
#ifndef DRACO_COMPRESSION_ATTRIBUTES_KD_TREE_ATTRIBUTES_SHARED_H_
#define DRACO_COMPRESSION_ATTRIBUTES_KD_TREE_ATTRIBUTES_SHARED_H_

namespace draco {

// Defines types of kD-tree compression
enum KdTreeAttributesEncodingMethod {
  kKdTreeQuantizationEncoding = 0,
  kKdTreeIntegerEncoding
};

}  // namespace draco

#endif  // DRACO_COMPRESSION_ATTRIBUTES_KD_TREE_ATTRIBUTES_SHARED_H_
