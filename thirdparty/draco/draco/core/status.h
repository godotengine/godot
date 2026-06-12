// Copyright 2017 The Draco Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
#ifndef DRACO_CORE_STATUS_H_
#define DRACO_CORE_STATUS_H_

#include <ostream>
#include <string>

namespace draco {

// Class encapsulating a return status of an operation with an optional error
// message. Intended to be used as a return type for functions instead of bool.
class Status {
 public:
  enum Code {
    OK = 0,
    DRACO_ERROR = -1,          // Used for general errors.
    IO_ERROR = -2,             // Error when handling input or output stream.
    INVALID_PARAMETER = -3,    // Invalid parameter passed to a function.
    UNSUPPORTED_VERSION = -4,  // Input not compatible with the current version.
    UNKNOWN_VERSION = -5,      // Input was created with an unknown version of
                               // the library.
    UNSUPPORTED_FEATURE = -6,  // Input contains feature that is not supported.
  };

  Status() : code_(OK) {}
  Status(const Status &status) = default;
  Status(Status &&status) = default;
  explicit Status(Code code) : code_(code) {}
  Status(Code code, const std::string &error_msg)
      : code_(code), error_msg_(error_msg) {}

  Code code() const { return code_; }
  const std::string &error_msg_string() const { return error_msg_; }
  const char *error_msg() const { return error_msg_.c_str(); }
  std::string code_string() const;
  std::string code_and_error_string() const;

  bool operator==(Code code) const { return code == code_; }
  bool ok() const { return code_ == OK; }

  Status &operator=(const Status &) = default;

 private:
  Code code_;
  std::string error_msg_;
};

inline std::ostream &operator<<(std::ostream &os, const Status &status) {
  os << status.error_msg_string();
  return os;
}

inline Status OkStatus() { return Status(Status::OK); }
inline Status ErrorStatus(const std::string &msg) {
  return Status(Status::DRACO_ERROR, msg);
}

// Evaluates an expression that returns draco::Status. If the status is not OK,
// the macro returns the status object.
#define DRACO_RETURN_IF_ERROR(expression)             \
  {                                                   \
    const draco::Status _local_status = (expression); \
    if (!_local_status.ok()) {                        \
      return _local_status;                           \
    }                                                 \
  }

}  // namespace draco

#endif  // DRACO_CORE_STATUS_H_
