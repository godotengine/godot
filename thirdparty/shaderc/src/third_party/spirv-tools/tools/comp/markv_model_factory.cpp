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

#include "tools/comp/markv_model_factory.h"

#include "source/util/make_unique.h"
#include "tools/comp/markv_model_shader.h"

namespace spvtools {
namespace comp {

std::unique_ptr<MarkvModel> CreateMarkvModel(MarkvModelType type) {
  std::unique_ptr<MarkvModel> model;
  switch (type) {
    case kMarkvModelShaderLite: {
      model = MakeUnique<MarkvModelShaderLite>();
      break;
    }
    case kMarkvModelShaderMid: {
      model = MakeUnique<MarkvModelShaderMid>();
      break;
    }
    case kMarkvModelShaderMax: {
      model = MakeUnique<MarkvModelShaderMax>();
      break;
    }
    case kMarkvModelUnknown: {
      assert(0 && "kMarkvModelUnknown supplied to CreateMarkvModel");
      return model;
    }
  }

  model->SetModelType(static_cast<uint32_t>(type));

  return model;
}

}  // namespace comp
}  // namespace spvtools
