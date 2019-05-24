// Copyright 2015 The Shaderc Authors. All rights reserved.
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

#ifndef LIBSHADERC_UTIL_SRC_MESSAGE_H_
#define LIBSHADERC_UTIL_SRC_MESSAGE_H_

#include "libshaderc_util/string_piece.h"

namespace shaderc_util {

// TODO(antiagainst): document the differences of the following message types.
enum class MessageType {
  Warning,
  Error,
  ErrorSummary,
  WarningSummary,
  GlobalWarning,
  GlobalError,
  Unknown,
  Ignored
};

// Given a glslang warning/error message, processes it in the following way and
// returns its message type.
//
// * Places the source name into the source_name parameter, if found.
//   Otherwise, clears the source_name parameter.
// * Places the line number into the line_number parameter, if found.
//   Otherwise, clears the line_number parameter.
// * Places the rest of the message (the text past warning/error prefix, source
//   name, and line number) into the rest parameter.
//
// If warnings_as_errors is set to true, then all warnings will be treated as
// errors.
// If suppress_warnings is set to true, then no warnings will be emitted. This
// takes precedence over warnings_as_errors.
//
// Examples:
// "ERROR: 0:2: Message"
//   source_name="0", line_number="2", rest="Message"
// "Warning, Message"
//   source_name="", line_number="", rest="Message"
// "ERROR: 2 errors found."
//   source_name="2", line_number="", rest="errors found".
MessageType ParseGlslangOutput(const shaderc_util::string_piece& message,
                               bool warnings_as_errors, bool suppress_warnings,
                               shaderc_util::string_piece* source_name,
                               shaderc_util::string_piece* line_number,
                               shaderc_util::string_piece* rest);

// Filters error_messages received from glslang, and outputs, to error_stream,
// any that are not ignored in a clang like format. If the warnings_as_errors
// boolean is set, then all warnings will be treated as errors. If the
// suppress_warnings boolean is set then any warning messages are ignored. This
// takes precedence over warnings_as_errors. Increments total_warnings and
// total_errors based on the message types.
// Returns true if no new errors were found when parsing the messages.
// "<command line>" will substitute "-1" appearing at the string name/number
// segment.
bool PrintFilteredErrors(const shaderc_util::string_piece& file_name,
                         std::ostream* error_stream, bool warnings_as_errors,
                         bool suppress_warnings, const char* error_list,
                         size_t* total_warnings, size_t* total_errors);

// Outputs, to error_stream,  the number of warnings and errors if there are
// any.
void OutputMessages(std::ostream* error_stream, size_t total_warnings,
                    size_t total_errors);

}  // namespace glslc

#endif  // LIBSHADERC_UTIL_SRC_MESSAGE_H_
