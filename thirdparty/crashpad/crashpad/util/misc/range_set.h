// Copyright 2018 The Crashpad Authors. All rights reserved.
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

#ifndef CRASHPAD_UTIL_MISC_RANGE_SET_H_
#define CRASHPAD_UTIL_MISC_RANGE_SET_H_

#include <map>

#include "base/macros.h"
#include "util/misc/address_types.h"

namespace crashpad {

//! \brief A set of VMAddress ranges.
class RangeSet {
 public:
  RangeSet();
  ~RangeSet();

  //! \brief Inserts a range into the set.
  //!
  //! \param[in] base The low address of the range.
  //! \param[in] size The size of the range.
  void Insert(VMAddress base, VMSize size);

  //! \brief Returns `true` if \a address falls within a range in this set.
  bool Contains(VMAddress address) const;

 private:
  // Keys are the highest address in the range. Values are the base address of
  // the range. Overlapping ranges are merged on insertion. Adjacent ranges may
  // be merged.
  std::map<VMAddress, VMAddress> ranges_;

  DISALLOW_COPY_AND_ASSIGN(RangeSet);
};

}  // namespace crashpad

#endif  // CRASHPAD_UTIL_MISC_RANGE_SET_H_
