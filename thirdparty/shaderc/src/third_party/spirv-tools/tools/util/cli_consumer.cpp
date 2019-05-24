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

#include "tools/util/cli_consumer.h"

#include <iostream>

namespace spvtools {
namespace utils {

void CLIMessageConsumer(spv_message_level_t level, const char*,
                        const spv_position_t& position, const char* message) {
  switch (level) {
    case SPV_MSG_FATAL:
    case SPV_MSG_INTERNAL_ERROR:
    case SPV_MSG_ERROR:
      std::cerr << "error: line " << position.index << ": " << message
                << std::endl;
      break;
    case SPV_MSG_WARNING:
      std::cout << "warning: line " << position.index << ": " << message
                << std::endl;
      break;
    case SPV_MSG_INFO:
      std::cout << "info: line " << position.index << ": " << message
                << std::endl;
      break;
    default:
      break;
  }
}

}  // namespace utils
}  // namespace spvtools
