// Copyright (c) 2021 Google LLC.
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

#include "spirv-tools/linter.hpp"

#include "source/lint/lints.h"
#include "source/opt/build_module.h"
#include "source/opt/ir_context.h"
#include "spirv-tools/libspirv.h"
#include "spirv-tools/libspirv.hpp"

namespace spvtools {

struct Linter::Impl {
  explicit Impl(spv_target_env env) : target_env(env) {
    message_consumer = [](spv_message_level_t /*level*/, const char* /*source*/,
                          const spv_position_t& /*position*/,
                          const char* /*message*/) {};
  }

  spv_target_env target_env;         // Target environment.
  MessageConsumer message_consumer;  // Message consumer.
};

Linter::Linter(spv_target_env env) : impl_(new Impl(env)) {}

Linter::~Linter() {}

void Linter::SetMessageConsumer(MessageConsumer consumer) {
  impl_->message_consumer = std::move(consumer);
}

const MessageConsumer& Linter::Consumer() const {
  return impl_->message_consumer;
}

bool Linter::Run(const uint32_t* binary, size_t binary_size) {
  std::unique_ptr<opt::IRContext> context =
      BuildModule(SPV_ENV_VULKAN_1_2, Consumer(), binary, binary_size);
  if (context == nullptr) return false;

  bool result = true;
  result &= lint::lints::CheckDivergentDerivatives(context.get());

  return result;
}

}  // namespace spvtools
