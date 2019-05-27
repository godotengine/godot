// Copyright (c) 2017 Google Inc.
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

#ifndef SOURCE_EXTENSIONS_H_
#define SOURCE_EXTENSIONS_H_

#include <string>

#include "source/enum_set.h"
#include "spirv-tools/libspirv.h"

namespace spvtools {

// The known SPIR-V extensions.
enum Extension {
#include "extension_enum.inc"
};

using ExtensionSet = EnumSet<Extension>;

// Returns literal string operand of OpExtension instruction.
std::string GetExtensionString(const spv_parsed_instruction_t* inst);

// Returns text string listing |extensions| separated by whitespace.
std::string ExtensionSetToString(const ExtensionSet& extensions);

}  // namespace spvtools

#endif  // SOURCE_EXTENSIONS_H_
