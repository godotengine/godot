// Copyright (c) 2016 Google Inc.
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

#include "source/opt/build_module.h"

#include <utility>
#include <vector>

#include "source/opt/ir_context.h"
#include "source/opt/ir_loader.h"
#include "source/table.h"
#include "source/util/make_unique.h"

namespace spvtools {
namespace {

// Sets the module header for IrLoader. Meets the interface requirement of
// spvBinaryParse().
spv_result_t SetSpvHeader(void* builder, spv_endianness_t, uint32_t magic,
                          uint32_t version, uint32_t generator,
                          uint32_t id_bound, uint32_t reserved) {
  reinterpret_cast<opt::IrLoader*>(builder)->SetModuleHeader(
      magic, version, generator, id_bound, reserved);
  return SPV_SUCCESS;
}

// Processes a parsed instruction for IrLoader. Meets the interface requirement
// of spvBinaryParse().
spv_result_t SetSpvInst(void* builder, const spv_parsed_instruction_t* inst) {
  if (reinterpret_cast<opt::IrLoader*>(builder)->AddInstruction(inst)) {
    return SPV_SUCCESS;
  }
  return SPV_ERROR_INVALID_BINARY;
}

}  // namespace

std::unique_ptr<opt::IRContext> BuildModule(spv_target_env env,
                                            MessageConsumer consumer,
                                            const uint32_t* binary,
                                            const size_t size) {
  return BuildModule(env, consumer, binary, size, true);
}

std::unique_ptr<opt::IRContext> BuildModule(spv_target_env env,
                                            MessageConsumer consumer,
                                            const uint32_t* binary,
                                            const size_t size,
                                            bool extra_line_tracking) {
  auto context = spvContextCreate(env);
  SetContextMessageConsumer(context, consumer);

  auto irContext = MakeUnique<opt::IRContext>(env, consumer);
  opt::IrLoader loader(consumer, irContext->module());
  loader.SetExtraLineTracking(extra_line_tracking);

  spv_result_t status = spvBinaryParse(context, &loader, binary, size,
                                       SetSpvHeader, SetSpvInst, nullptr);
  loader.EndModule();

  spvContextDestroy(context);

  return status == SPV_SUCCESS ? std::move(irContext) : nullptr;
}

std::unique_ptr<opt::IRContext> BuildModule(spv_target_env env,
                                            MessageConsumer consumer,
                                            const std::string& text,
                                            uint32_t assemble_options) {
  SpirvTools t(env);
  t.SetMessageConsumer(consumer);
  std::vector<uint32_t> binary;
  if (!t.Assemble(text, &binary, assemble_options)) return nullptr;
  return BuildModule(env, consumer, binary.data(), binary.size());
}

}  // namespace spvtools
