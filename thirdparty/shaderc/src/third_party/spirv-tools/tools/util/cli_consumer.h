// Copyright (c) 2018 Google LLC
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

#ifndef SOURCE_UTIL_CLI_CONSUMMER_H_
#define SOURCE_UTIL_CLI_CONSUMMER_H_

#include <include/spirv-tools/libspirv.h>

namespace spvtools {
namespace utils {

// A message consumer that can be used by command line tools like spirv-opt and
// spirv-val to display messages.
void CLIMessageConsumer(spv_message_level_t level, const char*,
                        const spv_position_t& position, const char* message);

}  // namespace utils
}  // namespace spvtools

#endif  // SOURCE_UTIL_CLI_CONSUMMER_H_
