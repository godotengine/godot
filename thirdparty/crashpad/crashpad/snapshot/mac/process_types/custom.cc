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

#include "snapshot/mac/process_types.h"

#include <stddef.h>
#include <string.h>
#include <sys/types.h>

#include <algorithm>
#include <type_traits>

#include "base/logging.h"
#include "base/numerics/safe_math.h"
#include "base/strings/stringprintf.h"
#include "snapshot/mac/process_types/internal.h"
#include "util/mach/task_memory.h"

#if !DOXYGEN

namespace crashpad {
namespace process_types {
namespace internal {

namespace {

template <typename T>
bool ReadIntoAndZero(TaskMemory* task_memory,
                     mach_vm_address_t address,
                     mach_vm_size_t size,
                     T* specific) {
  DCHECK_LE(size, sizeof(*specific));

  if (!task_memory->Read(address, size, specific)) {
    return false;
  }

  // Zero out the rest of the structure in case anything accesses fields without
  // checking the version or size.
  const size_t remaining = sizeof(*specific) - size;
  if (remaining > 0) {
    char* const start = reinterpret_cast<char*>(specific) + size;
    memset(start, 0, remaining);
  }

  return true;
}

template <typename T>
bool FieldAddressIfInRange(mach_vm_address_t address,
                           size_t offset,
                           mach_vm_address_t* field_address) {
  base::CheckedNumeric<typename T::Pointer> checked_field_address(address);
  checked_field_address += offset;
  typename T::Pointer local_field_address;
  if (!checked_field_address.AssignIfValid(&local_field_address)) {
    LOG(ERROR) << base::StringPrintf(
        "address 0x%llx + offset 0x%zx out of range", address, offset);
    return false;
  }

  *field_address = local_field_address;
  return true;
}

template <typename T>
bool ReadIntoVersioned(ProcessReaderMac* process_reader,
                       mach_vm_address_t address,
                       T* specific) {
  mach_vm_address_t field_address;
  if (!FieldAddressIfInRange<T>(
          address, offsetof(T, version), &field_address)) {
    return false;
  }

  TaskMemory* task_memory = process_reader->Memory();
  decltype(specific->version) version;
  if (!task_memory->Read(field_address, sizeof(version), &version)) {
    return false;
  }

  const size_t size = T::ExpectedSizeForVersion(version);
  return ReadIntoAndZero(task_memory, address, size, specific);
}

template <typename T>
bool ReadIntoSized(ProcessReaderMac* process_reader,
                   mach_vm_address_t address,
                   T* specific) {
  mach_vm_address_t field_address;
  if (!FieldAddressIfInRange<T>(address, offsetof(T, size), &field_address)) {
    return false;
  }

  TaskMemory* task_memory = process_reader->Memory();
  decltype(specific->size) size;
  if (!task_memory->Read(address + offsetof(T, size), sizeof(size), &size)) {
    return false;
  }

  if (size < T::MinimumSize()) {
    LOG(ERROR) << "small size " << size;
    return false;
  }

  size = std::min(static_cast<size_t>(size), sizeof(*specific));
  return ReadIntoAndZero(task_memory, address, size, specific);
}

}  // namespace

// static
template <typename Traits>
size_t dyld_all_image_infos<Traits>::ExpectedSizeForVersion(
    decltype(dyld_all_image_infos<Traits>::version) version) {
  static constexpr size_t kSizeForVersion[] = {
      offsetof(dyld_all_image_infos<Traits>, infoArrayCount),  // 0
      offsetof(dyld_all_image_infos<Traits>, libSystemInitialized),  // 1
      offsetof(dyld_all_image_infos<Traits>, jitInfo),  // 2
      offsetof(dyld_all_image_infos<Traits>, dyldVersion),  // 3
      offsetof(dyld_all_image_infos<Traits>, dyldVersion),  // 4
      offsetof(dyld_all_image_infos<Traits>, coreSymbolicationShmPage),  // 5
      offsetof(dyld_all_image_infos<Traits>, systemOrderFlag),  // 6
      offsetof(dyld_all_image_infos<Traits>, uuidArrayCount),  // 7
      offsetof(dyld_all_image_infos<Traits>, dyldAllImageInfosAddress),  // 8
      offsetof(dyld_all_image_infos<Traits>, initialImageCount),  // 9
      offsetof(dyld_all_image_infos<Traits>, errorKind),  // 10
      offsetof(dyld_all_image_infos<Traits>, sharedCacheSlide),  // 11
      offsetof(dyld_all_image_infos<Traits>, sharedCacheUUID),  // 12
      offsetof(dyld_all_image_infos<Traits>, infoArrayChangeTimestamp),  // 13
      offsetof(dyld_all_image_infos<Traits>, end_14_15),  // 14
      offsetof(dyld_all_image_infos<Traits>, end_14_15),  // 15
      sizeof(dyld_all_image_infos<Traits>),  // 16
  };

  if (version >= arraysize(kSizeForVersion)) {
    return kSizeForVersion[arraysize(kSizeForVersion) - 1];
  }

  static_assert(std::is_unsigned<decltype(version)>::value,
                "version must be unsigned");
  return kSizeForVersion[version];
}

// static
template <typename Traits>
bool dyld_all_image_infos<Traits>::ReadInto(
    ProcessReaderMac* process_reader,
    mach_vm_address_t address,
    dyld_all_image_infos<Traits>* specific) {
  return ReadIntoVersioned(process_reader, address, specific);
}

// static
template <typename Traits>
size_t crashreporter_annotations_t<Traits>::ExpectedSizeForVersion(
    decltype(crashreporter_annotations_t<Traits>::version) version) {
  if (version >= 5) {
    return sizeof(crashreporter_annotations_t<Traits>);
  }
  if (version >= 4) {
    return offsetof(crashreporter_annotations_t<Traits>, unknown_0);
  }
  return offsetof(crashreporter_annotations_t<Traits>, message);
}

// static
template <typename Traits>
bool crashreporter_annotations_t<Traits>::ReadInto(
    ProcessReaderMac* process_reader,
    mach_vm_address_t address,
    crashreporter_annotations_t<Traits>* specific) {
  return ReadIntoVersioned(process_reader, address, specific);
}

// static
template <typename Traits>
bool CrashpadInfo<Traits>::ReadInto(ProcessReaderMac* process_reader,
                                    mach_vm_address_t address,
                                    CrashpadInfo<Traits>* specific) {
  return ReadIntoSized(process_reader, address, specific);
}

// Explicit template instantiation of the above.
#define PROCESS_TYPE_FLAVOR_TRAITS(lp_bits)                             \
  template size_t                                                       \
  dyld_all_image_infos<Traits##lp_bits>::ExpectedSizeForVersion(        \
      decltype(dyld_all_image_infos<Traits##lp_bits>::version));        \
  template bool dyld_all_image_infos<Traits##lp_bits>::ReadInto(        \
      ProcessReaderMac*,                                                \
      mach_vm_address_t,                                                \
      dyld_all_image_infos<Traits##lp_bits>*);                          \
  template size_t                                                       \
  crashreporter_annotations_t<Traits##lp_bits>::ExpectedSizeForVersion( \
      decltype(crashreporter_annotations_t<Traits##lp_bits>::version)); \
  template bool crashreporter_annotations_t<Traits##lp_bits>::ReadInto( \
      ProcessReaderMac*,                                                \
      mach_vm_address_t,                                                \
      crashreporter_annotations_t<Traits##lp_bits>*);                   \
  template bool CrashpadInfo<Traits##lp_bits>::ReadInto(                \
      ProcessReaderMac*, mach_vm_address_t, CrashpadInfo<Traits##lp_bits>*);

#include "snapshot/mac/process_types/flavors.h"

#undef PROCESS_TYPE_FLAVOR_TRAITS

}  // namespace internal
}  // namespace process_types
}  // namespace crashpad

#endif  // !DOXYGEN
