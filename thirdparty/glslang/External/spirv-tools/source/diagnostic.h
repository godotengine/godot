// Copyright (c) 2015-2016 The Khronos Group Inc.
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

#ifndef SOURCE_DIAGNOSTIC_H_
#define SOURCE_DIAGNOSTIC_H_

#include <sstream>
#include <string>

#include "spirv-tools/libspirv.hpp"

namespace spvtools {

// A DiagnosticStream remembers the current position of the input and an error
// code, and captures diagnostic messages via the left-shift operator.
// If the error code is not SPV_FAILED_MATCH, then captured messages are
// emitted during the destructor.
class DiagnosticStream {
 public:
  DiagnosticStream(spv_position_t position, const MessageConsumer& consumer,
                   const std::string& disassembled_instruction,
                   spv_result_t error)
      : position_(position),
        consumer_(consumer),
        disassembled_instruction_(disassembled_instruction),
        error_(error) {}

  // Creates a DiagnosticStream from an expiring DiagnosticStream.
  // The new object takes the contents of the other, and prevents the
  // other from emitting anything during destruction.
  DiagnosticStream(DiagnosticStream&& other);

  // Destroys a DiagnosticStream.
  // If its status code is something other than SPV_FAILED_MATCH
  // then emit the accumulated message to the consumer.
  ~DiagnosticStream();

  // Adds the given value to the diagnostic message to be written.
  template <typename T>
  DiagnosticStream& operator<<(const T& val) {
    stream_ << val;
    return *this;
  }

  // Conversion operator to spv_result, returning the error code.
  operator spv_result_t() { return error_; }

 private:
  std::ostringstream stream_;
  spv_position_t position_;
  MessageConsumer consumer_;  // Message consumer callback.
  std::string disassembled_instruction_;
  spv_result_t error_;
};

// Changes the MessageConsumer in |context| to one that updates |diagnostic|
// with the last message received.
//
// This function expects that |diagnostic| is not nullptr and its content is a
// nullptr.
void UseDiagnosticAsMessageConsumer(spv_context context,
                                    spv_diagnostic* diagnostic);

std::string spvResultToString(spv_result_t res);

}  // namespace spvtools

#endif  // SOURCE_DIAGNOSTIC_H_
