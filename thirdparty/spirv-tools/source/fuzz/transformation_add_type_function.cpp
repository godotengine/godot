// Copyright (c) 2019 Google LLC
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

#include "source/fuzz/transformation_add_type_function.h"

#include <vector>

#include "source/fuzz/fuzzer_util.h"

namespace spvtools {
namespace fuzz {

TransformationAddTypeFunction::TransformationAddTypeFunction(
    protobufs::TransformationAddTypeFunction message)
    : message_(std::move(message)) {}

TransformationAddTypeFunction::TransformationAddTypeFunction(
    uint32_t fresh_id, uint32_t return_type_id,
    const std::vector<uint32_t>& argument_type_ids) {
  message_.set_fresh_id(fresh_id);
  message_.set_return_type_id(return_type_id);
  for (auto id : argument_type_ids) {
    message_.add_argument_type_id(id);
  }
}

bool TransformationAddTypeFunction::IsApplicable(
    opt::IRContext* ir_context, const TransformationContext& /*unused*/) const {
  // The result id must be fresh.
  if (!fuzzerutil::IsFreshId(ir_context, message_.fresh_id())) {
    return false;
  }
  // The return and argument types must be type ids but not not be function
  // type ids.
  if (!fuzzerutil::IsNonFunctionTypeId(ir_context, message_.return_type_id())) {
    return false;
  }
  for (auto argument_type_id : message_.argument_type_id()) {
    if (!fuzzerutil::IsNonFunctionTypeId(ir_context, argument_type_id)) {
      return false;
    }
  }
  // Check whether there is already an OpTypeFunction definition that uses
  // exactly the same return and argument type ids.  (Note that the type manager
  // does not allow us to check this, as it does not distinguish between
  // function types with different but isomorphic pointer argument types.)
  std::vector<uint32_t> type_ids = {message_.return_type_id()};
  type_ids.insert(type_ids.end(), message_.argument_type_id().begin(),
                  message_.argument_type_id().end());
  return fuzzerutil::FindFunctionType(ir_context, type_ids) == 0;
}

void TransformationAddTypeFunction::Apply(
    opt::IRContext* ir_context, TransformationContext* /*unused*/) const {
  std::vector<uint32_t> type_ids = {message_.return_type_id()};
  type_ids.insert(type_ids.end(), message_.argument_type_id().begin(),
                  message_.argument_type_id().end());

  fuzzerutil::AddFunctionType(ir_context, message_.fresh_id(), type_ids);
  // We have added an instruction to the module, so need to be careful about the
  // validity of existing analyses.
  ir_context->InvalidateAnalysesExceptFor(
      opt::IRContext::Analysis::kAnalysisNone);
}

protobufs::Transformation TransformationAddTypeFunction::ToMessage() const {
  protobufs::Transformation result;
  *result.mutable_add_type_function() = message_;
  return result;
}

std::unordered_set<uint32_t> TransformationAddTypeFunction::GetFreshIds()
    const {
  return {message_.fresh_id()};
}

}  // namespace fuzz
}  // namespace spvtools
