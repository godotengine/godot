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

#include "libshaderc_util/message.h"

#include <algorithm>
#include <cstring>
#include <iostream>
#include <iterator>

namespace shaderc_util {

namespace {

// Given a message, deduces and returns its type. If the message type is
// recognized, advances *message past the prefix indicating the type. Otherwise,
// leaves *message unchanged and returns MessageType::Unknown.
MessageType DeduceMessageType(string_piece* message) {
  static const char kErrorMessage[] = "ERROR: ";
  static const char kWarningMessage[] = "WARNING: ";
  static const char kGlobalWarningMessage[] = "Warning, ";

  if (message->starts_with(kErrorMessage)) {
    *message = message->substr(::strlen(kErrorMessage));
    return MessageType::Error;
  } else if (message->starts_with(kWarningMessage)) {
    *message = message->substr(::strlen(kWarningMessage));
    return MessageType::Warning;
  } else if (message->starts_with(kGlobalWarningMessage)) {
    *message = message->substr(::strlen(kGlobalWarningMessage));
    return MessageType::GlobalWarning;
  }
  return MessageType::Unknown;
}

// Deduces a location specification from the given message. A location
// specification is of the form "<source-name>:<line-number>:". If the deduction
// is successful, returns true and updates source_name and line_number to the
// deduced source name and line numer respectively. The prefix standing for the
// location specification in message is skipped. Otherwise, returns false and
// keeps all parameters untouched.
bool DeduceLocationSpec(string_piece* message, string_piece* source_name,
                        string_piece* line_number) {
  // TODO(antiagainst): we use ':' as a delimiter here. It may be a valid
  // character in the filename. Also be aware of other special characters,
  // for example, ' '.
  string_piece rest(*message);
  size_t colon_after_source = rest.find_first_of(':');
  if (colon_after_source == string_piece::npos) return false;

  string_piece source = rest.substr(0, colon_after_source);
  rest = rest.substr(colon_after_source + 1);
  size_t colon_after_line = rest.find_first_of(':');
  if (source.size() == 1 && ::isalpha(source.front()) && rest.size() > 0 &&
      rest.front() == '\\') {
    // Handle Windows path.
    colon_after_source += colon_after_line + 1;
    source = message->substr(0, colon_after_source);
    rest = rest.substr(colon_after_line + 1);
    colon_after_line = rest.find_first_of(':');
  }

  if (colon_after_line == string_piece::npos) return false;
  const string_piece line = rest.substr(0, colon_after_line);

  if (!std::all_of(line.begin(), line.end(), ::isdigit)) return false;

  *source_name = source;
  *line_number = line;
  *message = rest.substr(colon_after_line + 1).strip_whitespace();
  return true;
}

// Returns true if the given message is a summary message.
bool IsSummaryMessage(const string_piece& message) {
  const size_t space_loc = message.find_first_of(' ');
  if (space_loc == string_piece::npos) return false;
  const string_piece number = message.substr(0, space_loc);
  const string_piece rest = message.substr(space_loc + 1);
  if (!std::all_of(number.begin(), number.end(), ::isdigit)) return false;
  if (!rest.starts_with("compilation errors.")) return false;
  return true;
}

}  // anonymous namespace

MessageType ParseGlslangOutput(const string_piece& message,
                               bool warnings_as_errors, bool suppress_warnings,
                               string_piece* source_name,
                               string_piece* line_number, string_piece* rest) {
  string_piece rest_of_message(message);
  source_name->clear();
  line_number->clear();
  rest->clear();

  // The glslang warning/error messages are typically of the following form:
  //   <message-type> <location-specification> <message-body>
  //
  // <message-type> can be "WARNING:", "ERROR:", or "Warning, ". "WARNING:"
  // means a warning message for a certain line, while "Warning, " means a
  // global one.
  //
  // <location-specification> is of the form:
  //   <filename-or-string-number>:<line-number>:
  // It doesn't exist if the warning/error message is a global one.

  bool is_error = false;

  // Handle <message-type>.
  switch (DeduceMessageType(&rest_of_message)) {
    case MessageType::Warning:
      if (suppress_warnings) return MessageType::Ignored;
      break;
    case MessageType::Error:
      is_error = true;
      break;
    case MessageType::GlobalWarning:
      if (suppress_warnings) return MessageType::Ignored;
      *rest = rest_of_message;
      return warnings_as_errors ? MessageType::GlobalError
                                : MessageType::GlobalWarning;
    case MessageType::Unknown:
      *rest = rest_of_message;
      return MessageType::Unknown;
    default:
      break;
  }

  rest_of_message = rest_of_message.strip_whitespace();
  if (rest_of_message.empty()) return MessageType::Unknown;

  // Now we have stripped the <message-type>. Try to see if we can find
  // a <location-specification>.
  if (DeduceLocationSpec(&rest_of_message, source_name, line_number)) {
    *rest = rest_of_message;
    return (is_error || warnings_as_errors) ? MessageType::Error
                                            : MessageType::Warning;
  } else {
    // No <location-specification>. This is a global warning/error message.
    // A special kind of global message is summary message, which should
    // start with a number.
    *rest = rest_of_message;
    if (IsSummaryMessage(rest_of_message)) {
      return (is_error || warnings_as_errors) ? MessageType::ErrorSummary
                                              : MessageType::WarningSummary;
    }
    return (is_error || warnings_as_errors) ? MessageType::GlobalError
                                            : MessageType::GlobalWarning;
  }
  return MessageType::Unknown;
}

bool PrintFilteredErrors(const string_piece& file_name,
                         std::ostream* error_stream, bool warnings_as_errors,
                         bool suppress_warnings, const char* error_list,
                         size_t* total_warnings, size_t* total_errors) {
  const char* ignored_error_strings[] = {
      "Warning, version 310 is not yet complete; most version-specific "
      "features are present, but some are missing.",
      "Warning, version 400 is not yet complete; most version-specific "
      "features are present, but some are missing.",
      "Warning, version 410 is not yet complete; most version-specific "
      "features are present, but some are missing.",
      "Warning, version 420 is not yet complete; most version-specific "
      "features are present, but some are missing.",
      "Warning, version 430 is not yet complete; most version-specific "
      "features are present, but some are missing.",
      "Warning, version 440 is not yet complete; most version-specific "
      "features are present, but some are missing.",
      "Warning, version 450 is not yet complete; most version-specific "
      "features are present, but some are missing.",
      "Linked vertex stage:", "Linked fragment stage:",
      "Linked tessellation control stage:",
      "Linked tessellation evaluation stage:", "Linked geometry stage:",
      "Linked compute stage:", ""};
  size_t existing_total_errors = *total_errors;
  string_piece error_messages(error_list);
  for (const string_piece& message : error_messages.get_fields('\n')) {
    if (std::find(std::begin(ignored_error_strings),
                  std::end(ignored_error_strings),
                  message) == std::end(ignored_error_strings)) {
      string_piece source_name;
      string_piece line_number;
      string_piece rest;
      const MessageType type =
          ParseGlslangOutput(message, warnings_as_errors, suppress_warnings,
                             &source_name, &line_number, &rest);
      string_piece name = file_name;
      if (!source_name.empty()) {
        // -1 is the string number for the preamble injected by us.
        name = source_name == "-1" ? "<command line>" : source_name;
      }
      switch (type) {
        case MessageType::Error:
        case MessageType::Warning:
          assert(!name.empty() && !line_number.empty() && !rest.empty());
          *error_stream << name << ":" << line_number << ": "
                        << (type == MessageType::Error ? "error: "
                                                       : "warning: ")
                        << rest.strip_whitespace() << std::endl;
          *total_errors += type == MessageType::Error;
          *total_warnings += type == MessageType::Warning;
          break;
        case MessageType::ErrorSummary:
        case MessageType::WarningSummary:
          break;
        case MessageType::GlobalError:
        case MessageType::GlobalWarning:
          assert(!rest.empty());
          *total_errors += type == MessageType::GlobalError;
          *total_warnings += type == MessageType::GlobalWarning;
          *error_stream << name << ": "
                        << (type == MessageType::GlobalError ? "error"
                                                             : "warning")
                        << ": " << rest.strip_whitespace() << std::endl;
          break;
        case MessageType::Unknown:
          *error_stream << name << ":";
          *error_stream << " " << message << std::endl;
          break;
        case MessageType::Ignored:
          break;
      }
    }
  }
  return (existing_total_errors == *total_errors);
}

// Outputs the number of warnings and errors if there are any.
void OutputMessages(std::ostream* error_stream, size_t total_warnings,
                    size_t total_errors) {
  if (total_warnings > 0 || total_errors > 0) {
    if (total_warnings > 0 && total_errors > 0) {
      *error_stream << total_warnings << " warning"
                    << (total_warnings > 1 ? "s" : "") << " and "
                    << total_errors << " error" << (total_errors > 1 ? "s" : "")
                    << " generated." << std::endl;
    } else if (total_warnings > 0) {
      *error_stream << total_warnings << " warning"
                    << (total_warnings > 1 ? "s" : "") << " generated."
                    << std::endl;
    } else if (total_errors > 0) {
      *error_stream << total_errors << " error" << (total_errors > 1 ? "s" : "")
                    << " generated." << std::endl;
    }
  }
}

}  // namespace glslc
