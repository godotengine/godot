// Copyright (c) 2022 The Khronos Group Inc.
// Copyright (c) 2022 LunarG Inc.
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

#include "source/opt/analyze_live_input_pass.h"

#include "source/opt/ir_context.h"

namespace spvtools {
namespace opt {

Pass::Status AnalyzeLiveInputPass::Process() {
  // Current functionality assumes shader capability
  if (!context()->get_feature_mgr()->HasCapability(spv::Capability::Shader))
    return Status::SuccessWithoutChange;
  Pass::Status status = DoLiveInputAnalysis();
  return status;
}

Pass::Status AnalyzeLiveInputPass::DoLiveInputAnalysis() {
  // Current functionality only supports frag, tesc, tese or geom shaders.
  // Report failure for any other stage.
  auto stage = context()->GetStage();
  if (stage != spv::ExecutionModel::Fragment &&
      stage != spv::ExecutionModel::TessellationControl &&
      stage != spv::ExecutionModel::TessellationEvaluation &&
      stage != spv::ExecutionModel::Geometry)
    return Status::Failure;
  context()->get_liveness_mgr()->GetLiveness(live_locs_, live_builtins_);
  return Status::SuccessWithoutChange;
}

}  // namespace opt
}  // namespace spvtools
