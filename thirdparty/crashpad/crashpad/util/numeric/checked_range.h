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

#ifndef CRASHPAD_UTIL_NUMERIC_CHECKED_RANGE_H_
#define CRASHPAD_UTIL_NUMERIC_CHECKED_RANGE_H_

#include <limits>
#include <tuple>

#include "base/logging.h"
#include "base/numerics/safe_conversions.h"
#include "base/numerics/safe_math.h"
#include "util/misc/implicit_cast.h"

namespace crashpad {

//! \brief Ensures that a range, composed of a base and size, does not overflow
//!     its data type.
template <typename ValueType, typename SizeType = ValueType>
class CheckedRange {
 public:
  CheckedRange(ValueType base, SizeType size) {
    static_assert(!std::numeric_limits<SizeType>::is_signed,
                  "SizeType must be unsigned");
    SetRange(base, size);
  }

  //! \brief Sets the range’s base and size to \a base and \a size,
  //!     respectively.
  void SetRange(ValueType base, SizeType size) {
    base_ = base;
    size_ = size;
  }

  //! \brief The range’s base.
  ValueType base() const { return base_; }

  //! \brief The range’s size.
  SizeType size() const { return size_; }

  //! \brief The range’s end (its base plus its size).
  ValueType end() const { return base_ + size_; }

  //! \brief Returns the validity of the range.
  //!
  //! \return `true` if the range is valid, `false` otherwise.
  //!
  //! A range is valid if its size can be converted to the range’s data type
  //! without data loss, and if its end (base plus size) can be computed without
  //! overflowing its data type.
  bool IsValid() const {
    if (!base::IsValueInRangeForNumericType<ValueType, SizeType>(size_)) {
      return false;
    }
    base::CheckedNumeric<ValueType> checked_end(base_);
    checked_end += implicit_cast<ValueType>(size_);
    return checked_end.IsValid();
  }

  //! \brief Returns whether the range contains another value.
  //!
  //! \param[in] value The (possibly) contained value.
  //!
  //! \return `true` if the range contains \a value, `false` otherwise.
  //!
  //! A range contains a value if the value is greater than or equal to its
  //! base, and less than its end (base plus size).
  //!
  //! This method must only be called if IsValid() would return `true`.
  bool ContainsValue(ValueType value) const {
    DCHECK(IsValid());

    return value >= base() && value < end();
  }

  //! \brief Returns whether the range contains another range.
  //!
  //! \param[in] that The (possibly) contained range.
  //!
  //! \return `true` if `this` range, the containing range, contains \a that,
  //!     the contained range. `false` otherwise.
  //!
  //! A range contains another range when the contained range’s base is greater
  //! than or equal to the containing range’s base, and the contained range’s
  //! end is less than or equal to the containing range’s end.
  //!
  //! This method must only be called if IsValid() would return `true` for both
  //! CheckedRange objects involved.
  bool ContainsRange(const CheckedRange<ValueType, SizeType>& that) const {
    DCHECK(IsValid());
    DCHECK(that.IsValid());

    return that.base() >= base() && that.end() <= end();
  }

  //! \brief Returns whether the range overlaps another range.
  //!
  //! \param[in] that The (possibly) overlapping range.
  //!
  //! \return `true` if `this` range, the first range, overlaps \a that,
  //!     the provided range. `false` otherwise.
  //!
  //! Ranges are considered to be closed-open [base, end) for this test. Zero
  //! length ranges are never considered to overlap another range.
  //!
  //! This method must only be called if IsValid() would return `true` for both
  //! CheckedRange objects involved.
  bool OverlapsRange(const CheckedRange<ValueType, SizeType>& that) const {
    DCHECK(IsValid());
    DCHECK(that.IsValid());

    if (size() == 0 || that.size() == 0)
      return false;

    return base() < that.end() && that.base() < end();
  }

  bool operator<(const CheckedRange& other) const {
    return std::tie(base_, size_) < std::tie(other.base_, other.size_);
  }

 private:
  ValueType base_;
  SizeType size_;
};

}  // namespace crashpad

#endif  // CRASHPAD_UTIL_NUMERIC_CHECKED_RANGE_H_
