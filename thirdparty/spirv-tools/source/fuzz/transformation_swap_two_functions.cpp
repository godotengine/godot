// Copyright (c) 2021 Shiyu Liu
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

#include "source/fuzz/transformation_swap_two_functions.h"

#include "source/opt/function.h"
#include "source/opt/module.h"

#include "source/fuzz/fuzzer_util.h"

namespace spvtools {
namespace fuzz {

TransformationSwapTwoFunctions::TransformationSwapTwoFunctions(
    protobufs::TransformationSwapTwoFunctions message)
    : message_(std::move(message)) {}

TransformationSwapTwoFunctions::TransformationSwapTwoFunctions(uint32_t id1,
                                                               uint32_t id2) {
  assert(id1 != id2 && "The two function ids cannot be the same.");
  message_.set_function_id1(id1);
  message_.set_function_id2(id2);
}

bool TransformationSwapTwoFunctions::IsApplicable(
    opt::IRContext* ir_context, const TransformationContext& /*unused*/) const {
  auto func1_ptr = ir_context->GetFunction(message_.function_id1());
  auto func2_ptr = ir_context->GetFunction(message_.function_id2());
  return func1_ptr != nullptr && func2_ptr != nullptr;
}

void TransformationSwapTwoFunctions::Apply(
    opt::IRContext* ir_context, TransformationContext* /*unused*/) const {
  opt::Module::iterator func1_it =
      fuzzerutil::GetFunctionIterator(ir_context, message_.function_id1());
  opt::Module::iterator func2_it =
      fuzzerutil::GetFunctionIterator(ir_context, message_.function_id2());

  assert(func1_it != ir_context->module()->end() &&
         "Could not find function 1.");
  assert(func2_it != ir_context->module()->end() &&
         "Could not find function 2.");

  // Two function pointers are all set, swap the two functions within the
  // module.
  std::iter_swap(func1_it.Get(), func2_it.Get());
}

protobufs::Transformation TransformationSwapTwoFunctions::ToMessage() const {
  protobufs::Transformation result;
  *result.mutable_swap_two_functions() = message_;
  return result;
}

std::unordered_set<uint32_t> TransformationSwapTwoFunctions::GetFreshIds()
    const {
  return std::unordered_set<uint32_t>();
}

}  // namespace fuzz
}  // namespace spvtools
