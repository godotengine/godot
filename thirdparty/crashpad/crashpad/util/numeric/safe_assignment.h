// Copyright 2014 The Crashpad Authors. All rights reserved.
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

#ifndef CRASHPAD_UTIL_NUMERIC_SAFE_ASSIGNMENT_H_
#define CRASHPAD_UTIL_NUMERIC_SAFE_ASSIGNMENT_H_

#include "base/numerics/safe_conversions.h"

namespace crashpad {

//! \brief Performs an assignment if it can be done safely, and signals if it
//!     cannot be done safely.
//!
//! \param[out] destination A pointer to the variable to be assigned to.
//! \param[in] source The value to assign.
//!
//! \return `true` if \a source is in the range supported by the type of \a
//!     *destination, with the assignment to \a *destination having been
//!     performed. `false` if the assignment cannot be completed safely because
//!     \a source is outside of this range.
template <typename Destination, typename Source>
bool AssignIfInRange(Destination* destination, Source source) {
  if (!base::IsValueInRangeForNumericType<Destination>(source)) {
    return false;
  }

  *destination = static_cast<Destination>(source);
  return true;
}

}  // namespace crashpad

#endif  // CRASHPAD_UTIL_NUMERIC_SAFE_ASSIGNMENT_H_
