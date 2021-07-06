/*
 * Copyright 2015-2020 Arm Limited
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef SPIRV_CROSS_UTIL_HPP
#define SPIRV_CROSS_UTIL_HPP

#include "spirv_cross.hpp"

namespace spirv_cross_util
{
void rename_interface_variable(SPIRV_CROSS_NAMESPACE::Compiler &compiler,
                               const SPIRV_CROSS_NAMESPACE::SmallVector<SPIRV_CROSS_NAMESPACE::Resource> &resources,
                               uint32_t location, const std::string &name);
void inherit_combined_sampler_bindings(SPIRV_CROSS_NAMESPACE::Compiler &compiler);
} // namespace spirv_cross_util

#endif
