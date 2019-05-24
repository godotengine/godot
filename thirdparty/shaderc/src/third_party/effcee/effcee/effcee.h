// Copyright 2017 The Effcee Authors.
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

#ifndef EFFCEE_EFFCEE_H
#define EFFCEE_EFFCEE_H

#include <string>
#include "re2/re2.h"

namespace effcee {

// TODO(dneto): Provide a check language tutorial / manual.

// This does not implement the equivalents of FileCheck options:
//   --match-full-lines
//   --strict-whitespace
//   --implicit-ch3eck-not
//   --enable-var-scope

using StringPiece = re2::StringPiece;

// Options for matching.
class Options {
 public:
  Options()
      : prefix_("CHECK"), input_name_("<stdin>"), checks_name_("<stdin>") {}

  // Sets rule prefix to a copy of |prefix|.  Returns this object.
  Options& SetPrefix(StringPiece prefix) {
    prefix_ = std::string(prefix.begin(), prefix.end());
    return *this;
  }
  const std::string& prefix() const { return prefix_; }

  // Sets the input name.  Returns this object.
  // Use this for file names, for example.
  Options& SetInputName(StringPiece name) {
    input_name_ = std::string(name.begin(), name.end());
    return *this;
  }
  const std::string& input_name() const { return input_name_; }

  // Sets the checks input name.  Returns this object.
  // Use this for file names, for example.
  Options& SetChecksName(StringPiece name) {
    checks_name_ = std::string(name.begin(), name.end());
    return *this;
  }
  const std::string& checks_name() const { return checks_name_; }

 private:
  std::string prefix_;
  std::string input_name_;
  std::string checks_name_;
};

// The result of an attempted match.
class Result {
 public:
  enum class Status {
    Ok = 0,
    Fail,       // A failure to match
    BadOption,  // A bad option was specified
    NoRules,    // No rules were specified
    BadRule,    // A bad rule was specified
  };

  // Constructs a result with a given status.
  explicit Result(Status status) : status_(status) {}
  // Constructs a result with the given message.  Keeps a copy of the message.
  Result(Status status, StringPiece message)
      : status_(status), message_({message.begin(), message.end()}) {}

  Status status() const { return status_; }

  // Returns true if the match was successful.
  operator bool() const { return status_ == Status::Ok; }

  const std::string& message() const { return message_; }

  // Sets the error message to a copy of |message|.  Returns this object.
  Result& SetMessage(StringPiece message) {
    message_ = std::string(message.begin(), message.end());
    return *this;
  }

 private:
  // Status code indicating success, or kind of failure.
  Status status_;

  // Message describing the failure, if any.  On success, this is empty.
  std::string message_;
};

// Returns the result of attempting to match |text| against the pattern
// program in |checks|, with the given |options|.
Result Match(StringPiece text, StringPiece checks,
             const Options& options = Options());

}  // namespace effcee

#endif
