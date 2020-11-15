// Copyright 2015 The Crashpad Authors. All rights reserved.
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

#include "util/numeric/checked_address_range.h"

#include "base/format_macros.h"
#include "base/strings/stringprintf.h"

#if defined(OS_MACOSX)
#include <mach/mach.h>
#elif defined(OS_WIN)
#include "util/win/address_types.h"
#elif defined(OS_LINUX) || defined(OS_ANDROID)
#include "util/linux/address_types.h"
#elif defined(OS_FUCHSIA)
#include <zircon/types.h>
#endif  // OS_MACOSX

namespace crashpad {
namespace internal {

template <class ValueType, class SizeType>
CheckedAddressRangeGeneric<ValueType, SizeType>::CheckedAddressRangeGeneric()
    : range_32_(0, 0),
#if defined(COMPILER_MSVC)
      range_64_(0, 0),
#endif  // COMPILER_MSVC
      is_64_bit_(false),
      range_ok_(true) {
}

template <class ValueType, class SizeType>
CheckedAddressRangeGeneric<ValueType, SizeType>::CheckedAddressRangeGeneric(
    bool is_64_bit,
    ValueType base,
    SizeType size)
#if defined(COMPILER_MSVC)
    : range_32_(0, 0),
      range_64_(0, 0)
#endif  // COMPILER_MSVC
{
  SetRange(is_64_bit, base, size);
}

template <class ValueType, class SizeType>
ValueType CheckedAddressRangeGeneric<ValueType, SizeType>::Base() const {
  return is_64_bit_ ? range_64_.base() : range_32_.base();
}

template <class ValueType, class SizeType>
SizeType CheckedAddressRangeGeneric<ValueType, SizeType>::Size() const {
  return is_64_bit_ ? range_64_.size() : range_32_.size();
}

template <class ValueType, class SizeType>
ValueType CheckedAddressRangeGeneric<ValueType, SizeType>::End() const {
  return is_64_bit_ ? range_64_.end() : range_32_.end();
}

template <class ValueType, class SizeType>
bool CheckedAddressRangeGeneric<ValueType, SizeType>::IsValid() const {
  return range_ok_ && (is_64_bit_ ? range_64_.IsValid() : range_32_.IsValid());
}

template <class ValueType, class SizeType>
void CheckedAddressRangeGeneric<ValueType, SizeType>::SetRange(bool is_64_bit,
                                                               ValueType base,
                                                               SizeType size) {
  is_64_bit_ = is_64_bit;
  if (is_64_bit_) {
    range_64_.SetRange(base, size);
    range_ok_ = true;
  } else {
    range_32_.SetRange(static_cast<uint32_t>(base),
                       static_cast<uint32_t>(size));
    range_ok_ = base::IsValueInRangeForNumericType<uint32_t>(base) &&
                base::IsValueInRangeForNumericType<uint32_t>(size);
  }
}

template <class ValueType, class SizeType>
bool CheckedAddressRangeGeneric<ValueType, SizeType>::ContainsValue(
    ValueType value) const {
  DCHECK(range_ok_);

  if (is_64_bit_) {
    return range_64_.ContainsValue(value);
  }

  if (!base::IsValueInRangeForNumericType<uint32_t>(value)) {
    return false;
  }

  return range_32_.ContainsValue(static_cast<uint32_t>(value));
}

template <class ValueType, class SizeType>
bool CheckedAddressRangeGeneric<ValueType, SizeType>::ContainsRange(
    const CheckedAddressRangeGeneric& that) const {
  DCHECK_EQ(is_64_bit_, that.is_64_bit_);
  DCHECK(range_ok_);
  DCHECK(that.range_ok_);

  return is_64_bit_ ? range_64_.ContainsRange(that.range_64_)
                    : range_32_.ContainsRange(that.range_32_);
}

template <class ValueType, class SizeType>
std::string CheckedAddressRangeGeneric<ValueType, SizeType>::AsString() const {
  return base::StringPrintf("0x%" PRIx64 " + 0x%" PRIx64 " (%s)",
                            uint64_t{Base()},
                            uint64_t{Size()},
                            Is64Bit() ? "64" : "32");
}

// Explicit instantiations for the cases we use.
#if defined(OS_MACOSX)
template class CheckedAddressRangeGeneric<mach_vm_address_t, mach_vm_size_t>;
#elif defined(OS_WIN)
template class CheckedAddressRangeGeneric<WinVMAddress, WinVMSize>;
#elif defined(OS_LINUX) || defined(OS_ANDROID)
template class CheckedAddressRangeGeneric<LinuxVMAddress, LinuxVMSize>;
#elif defined(OS_FUCHSIA)
template class CheckedAddressRangeGeneric<zx_vaddr_t, size_t>;
#endif  // OS_MACOSX

}  // namespace internal
}  // namespace crashpad
