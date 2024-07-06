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

#include "source/fuzz/fact_manager/livesafe_function_facts.h"

#include "source/fuzz/fuzzer_util.h"

namespace spvtools {
namespace fuzz {
namespace fact_manager {

LivesafeFunctionFacts::LivesafeFunctionFacts(opt::IRContext* ir_context)
    : ir_context_(ir_context) {}

bool LivesafeFunctionFacts::MaybeAddFact(
    const protobufs::FactFunctionIsLivesafe& fact) {
  if (!fuzzerutil::FindFunction(ir_context_, fact.function_id())) {
    return false;
  }

  if (fuzzerutil::FunctionIsEntryPoint(ir_context_, fact.function_id())) {
    return false;
  }

  livesafe_function_ids_.insert(fact.function_id());
  return true;
}

bool LivesafeFunctionFacts::FunctionIsLivesafe(uint32_t function_id) const {
  return livesafe_function_ids_.count(function_id) != 0;
}

}  // namespace fact_manager
}  // namespace fuzz
}  // namespace spvtools
