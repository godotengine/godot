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

#include "source/diagnostic.h"

#include <cassert>
#include <cstring>
#include <iostream>
#include <sstream>
#include <utility>

#include "source/table.h"

// Diagnostic API

spv_diagnostic spvDiagnosticCreate(const spv_position position,
                                   const char* message) {
  spv_diagnostic diagnostic = new spv_diagnostic_t;
  if (!diagnostic) return nullptr;
  size_t length = strlen(message) + 1;
  diagnostic->error = new char[length];
  if (!diagnostic->error) {
    delete diagnostic;
    return nullptr;
  }
  diagnostic->position = *position;
  diagnostic->isTextSource = false;
  memset(diagnostic->error, 0, length);
  strncpy(diagnostic->error, message, length);
  return diagnostic;
}

void spvDiagnosticDestroy(spv_diagnostic diagnostic) {
  if (!diagnostic) return;
  delete[] diagnostic->error;
  delete diagnostic;
}

spv_result_t spvDiagnosticPrint(const spv_diagnostic diagnostic) {
  if (!diagnostic) return SPV_ERROR_INVALID_DIAGNOSTIC;

  if (diagnostic->isTextSource) {
    // NOTE: This is a text position
    // NOTE: add 1 to the line as editors start at line 1, we are counting new
    // line characters to start at line 0
    std::cerr << "error: " << diagnostic->position.line + 1 << ": "
              << diagnostic->position.column + 1 << ": " << diagnostic->error
              << "\n";
    return SPV_SUCCESS;
  }

  // NOTE: Assume this is a binary position
  std::cerr << "error: ";
  if (diagnostic->position.index > 0)
    std::cerr << diagnostic->position.index << ": ";
  std::cerr << diagnostic->error << "\n";
  return SPV_SUCCESS;
}

namespace spvtools {

DiagnosticStream::DiagnosticStream(DiagnosticStream&& other)
    : stream_(),
      position_(other.position_),
      consumer_(other.consumer_),
      disassembled_instruction_(std::move(other.disassembled_instruction_)),
      error_(other.error_) {
  // Prevent the other object from emitting output during destruction.
  other.error_ = SPV_FAILED_MATCH;
  // Some platforms are missing support for std::ostringstream functionality,
  // including:  move constructor, swap method.  Either would have been a
  // better choice than copying the string.
  stream_ << other.stream_.str();
}

DiagnosticStream::~DiagnosticStream() {
  if (error_ != SPV_FAILED_MATCH && consumer_ != nullptr) {
    auto level = SPV_MSG_ERROR;
    switch (error_) {
      case SPV_SUCCESS:
      case SPV_REQUESTED_TERMINATION:  // Essentially success.
        level = SPV_MSG_INFO;
        break;
      case SPV_WARNING:
        level = SPV_MSG_WARNING;
        break;
      case SPV_UNSUPPORTED:
      case SPV_ERROR_INTERNAL:
      case SPV_ERROR_INVALID_TABLE:
        level = SPV_MSG_INTERNAL_ERROR;
        break;
      case SPV_ERROR_OUT_OF_MEMORY:
        level = SPV_MSG_FATAL;
        break;
      default:
        break;
    }
    if (disassembled_instruction_.size() > 0)
      stream_ << std::endl << "  " << disassembled_instruction_ << std::endl;

    consumer_(level, "input", position_, stream_.str().c_str());
  }
}

void UseDiagnosticAsMessageConsumer(spv_context context,
                                    spv_diagnostic* diagnostic) {
  assert(diagnostic && *diagnostic == nullptr);

  auto create_diagnostic = [diagnostic](spv_message_level_t, const char*,
                                        const spv_position_t& position,
                                        const char* message) {
    auto p = position;
    spvDiagnosticDestroy(*diagnostic);  // Avoid memory leak.
    *diagnostic = spvDiagnosticCreate(&p, message);
  };
  SetContextMessageConsumer(context, std::move(create_diagnostic));
}

std::string spvResultToString(spv_result_t res) {
  std::string out;
  switch (res) {
    case SPV_SUCCESS:
      out = "SPV_SUCCESS";
      break;
    case SPV_UNSUPPORTED:
      out = "SPV_UNSUPPORTED";
      break;
    case SPV_END_OF_STREAM:
      out = "SPV_END_OF_STREAM";
      break;
    case SPV_WARNING:
      out = "SPV_WARNING";
      break;
    case SPV_FAILED_MATCH:
      out = "SPV_FAILED_MATCH";
      break;
    case SPV_REQUESTED_TERMINATION:
      out = "SPV_REQUESTED_TERMINATION";
      break;
    case SPV_ERROR_INTERNAL:
      out = "SPV_ERROR_INTERNAL";
      break;
    case SPV_ERROR_OUT_OF_MEMORY:
      out = "SPV_ERROR_OUT_OF_MEMORY";
      break;
    case SPV_ERROR_INVALID_POINTER:
      out = "SPV_ERROR_INVALID_POINTER";
      break;
    case SPV_ERROR_INVALID_BINARY:
      out = "SPV_ERROR_INVALID_BINARY";
      break;
    case SPV_ERROR_INVALID_TEXT:
      out = "SPV_ERROR_INVALID_TEXT";
      break;
    case SPV_ERROR_INVALID_TABLE:
      out = "SPV_ERROR_INVALID_TABLE";
      break;
    case SPV_ERROR_INVALID_VALUE:
      out = "SPV_ERROR_INVALID_VALUE";
      break;
    case SPV_ERROR_INVALID_DIAGNOSTIC:
      out = "SPV_ERROR_INVALID_DIAGNOSTIC";
      break;
    case SPV_ERROR_INVALID_LOOKUP:
      out = "SPV_ERROR_INVALID_LOOKUP";
      break;
    case SPV_ERROR_INVALID_ID:
      out = "SPV_ERROR_INVALID_ID";
      break;
    case SPV_ERROR_INVALID_CFG:
      out = "SPV_ERROR_INVALID_CFG";
      break;
    case SPV_ERROR_INVALID_LAYOUT:
      out = "SPV_ERROR_INVALID_LAYOUT";
      break;
    default:
      out = "Unknown Error";
  }
  return out;
}

}  // namespace spvtools
