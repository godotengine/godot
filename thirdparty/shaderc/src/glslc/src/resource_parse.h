// Copyright 2016 The Shaderc Authors. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef GLSLC_RESOURCE_PARSE_H
#define GLSLC_RESOURCE_PARSE_H

#include <string>
#include <vector>

#include "shaderc/shaderc.h"

namespace glslc {

// A resource limit setting.
struct ResourceSetting {
  shaderc_limit limit;
  int value;
};


// Returns true when two resource setting structures are equal.
inline bool operator==(const ResourceSetting& lhs, const ResourceSetting& rhs) {
  return (lhs.limit == rhs.limit) && (lhs.value == rhs.value);
}


// Parses a resource limit setting string.  On success, returns true and populates
// the limits parameter.  On failure returns failure and emits a message to err.
// The setting string should be a seqeuence of pairs, where each pair
// is a limit name followed by a decimal integer.  Tokens should be separated
// by whitespace.  In particular, this function accepts Glslang's configuration
// file syntax.  If a limit is mentioned multiple times, then the last setting
// takes effect.  Ignore settings for:
//   nonInductiveForLoops
//   whileLoops
//   doWhileLoops
//   generalUniformIndexing
//   generalAttributeMatrixVectorIndexing
//   generalVaryingIndexing
//   generalSamplerIndexing
//   generalVariableIndexing
//   generalConstantMatrixVectorIndexing
bool ParseResourceSettings(const std::string& input,
                           std::vector<ResourceSetting>* limits,
                           std::string* err);
}  // namespace glslc


#endif  // GLSLC_FILE_H_
