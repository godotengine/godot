// Copyright (c) 2017 Google Inc.
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

#ifndef TOOLS_COMP_MARKV_MODEL_SHADER_H_
#define TOOLS_COMP_MARKV_MODEL_SHADER_H_

#include "source/comp/markv_model.h"

namespace spvtools {
namespace comp {

// MARK-V shader compression model, which only uses fast and lightweight
// algorithms, which do not require training and are not heavily dependent on
// SPIR-V grammar. Compression ratio is worse than by other models.
class MarkvModelShaderLite : public MarkvModel {
 public:
  MarkvModelShaderLite();
};

// MARK-V shader compression model with balanced compression ratio and runtime
// performance.
class MarkvModelShaderMid : public MarkvModel {
 public:
  MarkvModelShaderMid();
};

// MARK-V shader compression model designed for maximum compression.
class MarkvModelShaderMax : public MarkvModel {
 public:
  MarkvModelShaderMax();
};

}  // namespace comp
}  // namespace spvtools

#endif  // TOOLS_COMP_MARKV_MODEL_SHADER_H_
