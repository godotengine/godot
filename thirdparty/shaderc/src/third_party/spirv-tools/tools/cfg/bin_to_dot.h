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

#ifndef TOOLS_CFG_BIN_TO_DOT_H_
#define TOOLS_CFG_BIN_TO_DOT_H_

#include <iostream>

#include "spirv-tools/libspirv.h"

// Dumps the control flow graph for the given module to the output stream.
// Returns SPV_SUCCESS on succes.
spv_result_t BinaryToDot(const spv_const_context context, const uint32_t* words,
                         size_t num_words, std::iostream* out,
                         spv_diagnostic* diagnostic);

#endif  // TOOLS_CFG_BIN_TO_DOT_H_
