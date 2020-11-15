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

#ifndef CRASHPAD_UTIL_NUMERIC_IN_RANGE_CAST_H_
#define CRASHPAD_UTIL_NUMERIC_IN_RANGE_CAST_H_

#include "base/logging.h"
#include "base/numerics/safe_conversions.h"

namespace crashpad {

//! \brief Casts to a different type if it can be done without data loss,
//!     logging a warning message and returing a default value otherwise.
//!
//! \param[in] source The value to convert and return.
//! \param[in] default_value The default value to return, in the event that \a
//!     source cannot be represented in the destination type.
//!
//! \return \a source if it can be represented in the destination type,
//!     otherwise \a default_value.
template <typename Destination, typename Source>
Destination InRangeCast(Source source, Destination default_value) {
  if (base::IsValueInRangeForNumericType<Destination>(source)) {
    return static_cast<Destination>(source);
  }

  LOG(WARNING) << "value " << source << " out of range";
  return static_cast<Destination>(default_value);
}

}  // namespace crashpad

#endif  // CRASHPAD_UTIL_NUMERIC_IN_RANGE_CAST_H_
