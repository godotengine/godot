// Copyright (c) 2020 Vasyl Teliman
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

#include "source/fuzz/fuzzer_pass_replace_params_with_struct.h"

#include <numeric>
#include <vector>

#include "source/fuzz/fuzzer_context.h"
#include "source/fuzz/fuzzer_util.h"
#include "source/fuzz/transformation_replace_params_with_struct.h"

namespace spvtools {
namespace fuzz {

FuzzerPassReplaceParamsWithStruct::FuzzerPassReplaceParamsWithStruct(
    opt::IRContext* ir_context, TransformationContext* transformation_context,
    FuzzerContext* fuzzer_context,
    protobufs::TransformationSequence* transformations,
    bool ignore_inapplicable_transformations)
    : FuzzerPass(ir_context, transformation_context, fuzzer_context,
                 transformations, ignore_inapplicable_transformations) {}

void FuzzerPassReplaceParamsWithStruct::Apply() {
  for (const auto& function : *GetIRContext()->module()) {
    auto params =
        fuzzerutil::GetParameters(GetIRContext(), function.result_id());

    if (params.empty() || fuzzerutil::FunctionIsEntryPoint(
                              GetIRContext(), function.result_id())) {
      continue;
    }

    if (!GetFuzzerContext()->ChoosePercentage(
            GetFuzzerContext()->GetChanceOfReplacingParametersWithStruct())) {
      continue;
    }

    std::vector<uint32_t> parameter_index(params.size());
    std::iota(parameter_index.begin(), parameter_index.end(), 0);

    // Remove the indices of unsupported parameters.
    auto new_end =
        std::remove_if(parameter_index.begin(), parameter_index.end(),
                       [this, &params](uint32_t index) {
                         return !TransformationReplaceParamsWithStruct::
                             IsParameterTypeSupported(GetIRContext(),
                                                      params[index]->type_id());
                       });

    // std::remove_if changes the vector so that removed elements are placed at
    // the end (i.e. [new_end, parameter_index.end()) is a range of removed
    // elements). However, the size of the vector is not changed so we need to
    // change that explicitly by popping those elements from the vector.
    parameter_index.erase(new_end, parameter_index.end());

    if (parameter_index.empty()) {
      continue;
    }

    // Select |num_replaced_params| parameters at random. We shuffle the vector
    // of indices for randomization and shrink it to select first
    // |num_replaced_params| parameters.
    auto num_replaced_params = std::min<size_t>(
        parameter_index.size(),
        GetFuzzerContext()->GetRandomNumberOfParametersReplacedWithStruct(
            static_cast<uint32_t>(params.size())));

    GetFuzzerContext()->Shuffle(&parameter_index);
    parameter_index.resize(num_replaced_params);

    // Make sure OpTypeStruct exists in the module.
    std::vector<uint32_t> component_type_ids;
    for (auto index : parameter_index) {
      component_type_ids.push_back(params[index]->type_id());
    }

    FindOrCreateStructType(component_type_ids);

    // Map parameters' indices to parameters' ids.
    std::vector<uint32_t> parameter_id;
    for (auto index : parameter_index) {
      parameter_id.push_back(params[index]->result_id());
    }

    std::map<uint32_t, uint32_t> caller_id_to_fresh_id;
    for (const auto* inst :
         fuzzerutil::GetCallers(GetIRContext(), function.result_id())) {
      caller_id_to_fresh_id[inst->result_id()] =
          GetFuzzerContext()->GetFreshId();
    }

    ApplyTransformation(TransformationReplaceParamsWithStruct(
        parameter_id, GetFuzzerContext()->GetFreshId(),
        GetFuzzerContext()->GetFreshId(), caller_id_to_fresh_id));
  }
}

}  // namespace fuzz
}  // namespace spvtools
