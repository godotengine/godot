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
#ifndef DRACO_CORE_STATUS_OR_H_
#define DRACO_CORE_STATUS_OR_H_

#include "draco/core/macros.h"
#include "draco/core/status.h"

namespace draco {

// Class StatusOr is used to wrap a Status along with a value of a specified
// type |T|. StatusOr is intended to be returned from functions in situations
// where it is desirable to carry over more information about the potential
// errors encountered during the function execution. If there are not errors,
// the caller can simply use the return value, otherwise the Status object
// provides more info about the encountered problem.
template <class T>
class StatusOr {
 public:
  StatusOr() {}
  // Note: Constructors are intentionally not explicit to allow returning
  // Status or the return value directly from functions.
  StatusOr(const StatusOr &) = default;
  StatusOr(StatusOr &&) = default;
  StatusOr(const Status &status) : status_(status) {}
  StatusOr(const T &value) : status_(OkStatus()), value_(value) {}
  StatusOr(T &&value) : status_(OkStatus()), value_(std::move(value)) {}
  StatusOr(const Status &status, const T &value)
      : status_(status), value_(value) {}

  const Status &status() const { return status_; }
  const T &value() const & { return value_; }
  const T &&value() const && { return std::move(value_); }
  T &&value() && { return std::move(value_); }

  // For consistency with existing Google StatusOr API we also include
  // ValueOrDie() that currently returns the value().
  const T &ValueOrDie() const & { return value(); }
  T &&ValueOrDie() && { return std::move(value()); }

  bool ok() const { return status_.ok(); }

 private:
  Status status_;
  T value_;
};

// In case StatusOr<T> is ok(), this macro assigns value stored in StatusOr<T>
// to |lhs|, otherwise it returns the error Status.
//
//   DRACO_ASSIGN_OR_RETURN(lhs, expression)
//
#define DRACO_ASSIGN_OR_RETURN(lhs, expression)                                \
  DRACO_ASSIGN_OR_RETURN_IMPL_(DRACO_MACROS_IMPL_CONCAT_(_statusor, __LINE__), \
                               lhs, expression, _status)

// The actual implementation of the above macro.
#define DRACO_ASSIGN_OR_RETURN_IMPL_(statusor, lhs, expression, error_expr) \
  auto statusor = (expression);                                             \
  if (!statusor.ok()) {                                                     \
    auto _status = std::move(statusor.status());                            \
    (void)_status; /* error_expression may not use it */                    \
    return error_expr;                                                      \
  }                                                                         \
  lhs = std::move(statusor).value();

}  // namespace draco

#endif  // DRACO_CORE_STATUS_OR_H_
