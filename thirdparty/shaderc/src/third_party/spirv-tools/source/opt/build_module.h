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

#ifndef SOURCE_OPT_BUILD_MODULE_H_
#define SOURCE_OPT_BUILD_MODULE_H_

#include <memory>
#include <string>

#include "source/opt/ir_context.h"
#include "source/opt/module.h"
#include "spirv-tools/libspirv.hpp"

namespace spvtools {

// Builds an Module returns the owning IRContext from the given SPIR-V
// |binary|. |size| specifies number of words in |binary|. The |binary| will be
// decoded according to the given target |env|. Returns nullptr if errors occur
// and sends the errors to |consumer|.
std::unique_ptr<opt::IRContext> BuildModule(spv_target_env env,
                                            MessageConsumer consumer,
                                            const uint32_t* binary,
                                            size_t size);

// Builds an Module and returns the owning IRContext from the given
// SPIR-V assembly |text|.  The |text| will be encoded according to the given
// target |env|. Returns nullptr if errors occur and sends the errors to
// |consumer|.
std::unique_ptr<opt::IRContext> BuildModule(
    spv_target_env env, MessageConsumer consumer, const std::string& text,
    uint32_t assemble_options = SpirvTools::kDefaultAssembleOption);

}  // namespace spvtools

#endif  // SOURCE_OPT_BUILD_MODULE_H_
