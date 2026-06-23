// Copyright 2017 The Draco Authors.
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
#ifndef DRACO_COMPRESSION_CONFIG_DECODER_OPTIONS_H_
#define DRACO_COMPRESSION_CONFIG_DECODER_OPTIONS_H_

#include <map>
#include <memory>

#include "draco/attributes/geometry_attribute.h"
#include "draco/compression/config/draco_options.h"

namespace draco {

// Class containing options that can be passed to PointCloudDecoder to control
// decoding of the input geometry. The options can be specified either for the
// whole geometry or for a specific attribute type. Each option is identified
// by a unique name stored as an std::string.
typedef DracoOptions<GeometryAttribute::Type> DecoderOptions;

}  // namespace draco

#endif  // DRACO_COMPRESSION_CONFIG_DECODER_OPTIONS_H_
