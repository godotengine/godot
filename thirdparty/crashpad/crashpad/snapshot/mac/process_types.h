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

#ifndef CRASHPAD_SNAPSHOT_MAC_PROCESS_TYPES_H_
#define CRASHPAD_SNAPSHOT_MAC_PROCESS_TYPES_H_

#include <mach/mach.h>
#include <mach-o/dyld_images.h>
#include <mach-o/loader.h>
#include <stdint.h>
#include <sys/types.h>

#include "snapshot/mac/process_reader_mac.h"

namespace crashpad {
namespace process_types {
namespace internal {

// Some structure definitions have tail padding when built in a 64-bit
// environment, but will have less (or no) tail padding in a 32-bit environment.
// These structure’s apparent sizes will differ between these two environments,
// which is incorrect. Use this “Nothing” type to place an “end” marker after
// all of the real members of such structures, and use offsetof(type, end) to
// compute their actual sizes.
using Nothing = char[0];

// Some structure definitions differ in 32-bit and 64-bit environments by having
// additional “reserved” padding fields present only in the 64-bit environment.
// These Reserved*_*Only* types allow the process_types system to replicate
// these structures more precisely.
using Reserved32_32Only32 = uint32_t;
using Reserved32_32Only64 = Nothing;
using Reserved32_64Only32 = Nothing;
using Reserved32_64Only64 = uint32_t;
using Reserved64_64Only32 = Nothing;
using Reserved64_64Only64 = uint64_t;

}  // namespace internal
}  // namespace process_types
}  // namespace crashpad

#include "snapshot/mac/process_types/traits.h"

// Creates the traits type crashpad::process_types::internal::TraitsGeneric.
DECLARE_PROCESS_TYPE_TRAITS_CLASS(Generic, 64)

#undef DECLARE_PROCESS_TYPE_TRAITS_CLASS

// Declare the crashpad::process_types::struct_name structs. These are the
// user-visible generic structs that callers will interact with. They read data
// from 32-bit or 64-bit processes by using the specific internal templatized
// structs below.
#define PROCESS_TYPE_STRUCT_DECLARE 1

#define PROCESS_TYPE_STRUCT_BEGIN(struct_name)                                 \
  namespace crashpad {                                                         \
  namespace process_types {                                                    \
  struct struct_name {                                                         \
   public:                                                                     \
    using Long = internal::TraitsGeneric::Long;                                \
    using ULong = internal::TraitsGeneric::ULong;                              \
    using Pointer = internal::TraitsGeneric::Pointer;                          \
    using IntPtr = internal::TraitsGeneric::IntPtr;                            \
    using UIntPtr = internal::TraitsGeneric::UIntPtr;                          \
    using Reserved32_32Only = internal::TraitsGeneric::Reserved32_32Only;      \
    using Reserved32_64Only = internal::TraitsGeneric::Reserved32_64Only;      \
    using Reserved64_64Only = internal::TraitsGeneric::Reserved64_64Only;      \
    using Nothing = internal::TraitsGeneric::Nothing;                          \
                                                                               \
    /* Initializes an object with data read from |process_reader| at           \
     * |address|, properly genericized. */                                     \
    bool Read(ProcessReaderMac* process_reader, mach_vm_address_t address) {   \
      return ReadInto(process_reader, address, this);                          \
    }                                                                          \
                                                                               \
    /* Reads |count| objects from |process_reader| beginning at |address|, and \
     * genericizes the objects. The caller must provide storage for |count|    \
     * objects in |generic|. */                                                \
    static bool ReadArrayInto(ProcessReaderMac* process_reader,                \
                              mach_vm_address_t address,                       \
                              size_t count,                                    \
                              struct_name* generic);                           \
                                                                               \
    /* Returns the size of the object that was read. This is the size of the   \
     * storage in the process that the data is read from, and is not the same  \
     * as the size of the generic struct. For versioned and sized structures,  \
     * the size of the full structure is returned. */                          \
    size_t Size() const { return size_; }                                      \
                                                                               \
    /* Similar to Size(), but computes the expected size of a structure based  \
     * on the process’ bitness. This can be used prior to reading any data     \
     * from a process. For versioned and sized structures,                     \
     * ExpectedSizeForVersion() and MinimumSize() may also be useful. */       \
    static size_t ExpectedSize(ProcessReaderMac* process_reader);

#define PROCESS_TYPE_STRUCT_MEMBER(member_type, member_name, ...)              \
    member_type member_name __VA_ARGS__;

#define PROCESS_TYPE_STRUCT_VERSIONED(struct_name, version_field)            \
  /* Similar to ExpectedSize(), but computes the expected size of a          \
   * structure based on the process’ bitness and a custom value, such as a   \
   * structure version number. This can be used prior to reading any data    \
   * from a process. */                                                      \
  static size_t ExpectedSizeForVersion(                                      \
      ProcessReaderMac* process_reader,                                      \
      decltype(struct_name::version_field) version);

#define PROCESS_TYPE_STRUCT_SIZED(struct_name, size_field)                    \
  /* Similar to ExpectedSize(), but computes the minimum size of a            \
   * structure based on the process’ bitness, typically including enough of   \
   * a structure to contain its size field. This can be used prior to         \
   * reading any data from a process. */                                      \
  static size_t MinimumSize(ProcessReaderMac* process_reader);

#define PROCESS_TYPE_STRUCT_END(struct_name)                          \
 private:                                                             \
  /* The static form of Read(). Populates the struct at |generic|. */ \
  static bool ReadInto(ProcessReaderMac* process_reader,              \
                       mach_vm_address_t address,                     \
                       struct_name* generic);                         \
                                                                      \
  template <typename T>                                               \
  static bool ReadIntoInternal(ProcessReaderMac* process_reader,      \
                               mach_vm_address_t address,             \
                               struct_name* generic);                 \
  template <typename T>                                               \
  static bool ReadArrayIntoInternal(ProcessReaderMac* process_reader, \
                                    mach_vm_address_t address,        \
                                    size_t count,                     \
                                    struct_name* generic);            \
  size_t size_;                                                       \
  }                                                                   \
  ;                                                                   \
  } /* namespace process_types */                                     \
  } /* namespace crashpad */

#include "snapshot/mac/process_types/all.proctype"

#undef PROCESS_TYPE_STRUCT_BEGIN
#undef PROCESS_TYPE_STRUCT_MEMBER
#undef PROCESS_TYPE_STRUCT_VERSIONED
#undef PROCESS_TYPE_STRUCT_SIZED
#undef PROCESS_TYPE_STRUCT_END
#undef PROCESS_TYPE_STRUCT_DECLARE

// Declare the templatized crashpad::process_types::internal::struct_name<>
// structs. These are the 32-bit and 64-bit specific structs that describe the
// layout of objects in another process. This is repeated instead of being
// shared with the generic declaration above because both the generic and
// templatized specific structs need all of the struct members declared.
//
// GenericizeInto() translates a struct from the representation used in the
// remote process into the generic form.
#define PROCESS_TYPE_STRUCT_DECLARE_INTERNAL 1

#define PROCESS_TYPE_STRUCT_BEGIN(struct_name)                                \
  namespace crashpad {                                                        \
  namespace process_types {                                                   \
  namespace internal {                                                        \
  template <typename Traits>                                                  \
  struct struct_name {                                                        \
   public:                                                                    \
    using Long = typename Traits::Long;                                       \
    using ULong = typename Traits::ULong;                                     \
    using Pointer = typename Traits::Pointer;                                 \
    using IntPtr = typename Traits::IntPtr;                                   \
    using UIntPtr = typename Traits::UIntPtr;                                 \
    using Reserved32_32Only = typename Traits::Reserved32_32Only;             \
    using Reserved32_64Only = typename Traits::Reserved32_64Only;             \
    using Reserved64_64Only = typename Traits::Reserved64_64Only;             \
    using Nothing = typename Traits::Nothing;                                 \
                                                                              \
    /* Read(), ReadArrayInto(), and Size() are as in the generic user-visible \
     * struct above. */                                                       \
    bool Read(ProcessReaderMac* process_reader, mach_vm_address_t address) {  \
      return ReadInto(process_reader, address, this);                         \
    }                                                                         \
    static bool ReadArrayInto(ProcessReaderMac* process_reader,               \
                              mach_vm_address_t address,                      \
                              size_t count,                                   \
                              struct_name<Traits>* specific);                 \
    static size_t Size() { return sizeof(struct_name<Traits>); }              \
                                                                              \
    /* Translates a struct from the representation used in the remote process \
     * into the generic form. */                                              \
    void GenericizeInto(process_types::struct_name* generic,                  \
                        size_t* specific_size);

#define PROCESS_TYPE_STRUCT_MEMBER(member_type, member_name, ...)              \
    member_type member_name __VA_ARGS__;

#define PROCESS_TYPE_STRUCT_VERSIONED(struct_name, version_field)              \
    /* ExpectedSizeForVersion() is as in the generic user-visible struct       \
     * above. */                                                               \
    static size_t ExpectedSizeForVersion(                                      \
        decltype(struct_name::version_field) version);

#define PROCESS_TYPE_STRUCT_SIZED(struct_name, size_field)                     \
  /* MinimumSize() is as in the generic user-visible struct above. */          \
  static size_t MinimumSize();

#define PROCESS_TYPE_STRUCT_END(struct_name)                       \
 private:                                                          \
  /* ReadInto() is as in the generic user-visible struct above. */ \
  static bool ReadInto(ProcessReaderMac* process_reader,           \
                       mach_vm_address_t address,                  \
                       struct_name<Traits>* specific);             \
  }                                                                \
  ;                                                                \
  } /* namespace internal */                                       \
  } /* namespace process_types */                                  \
  } /* namespace crashpad */

#include "snapshot/mac/process_types/all.proctype"

#undef PROCESS_TYPE_STRUCT_BEGIN
#undef PROCESS_TYPE_STRUCT_MEMBER
#undef PROCESS_TYPE_STRUCT_VERSIONED
#undef PROCESS_TYPE_STRUCT_SIZED
#undef PROCESS_TYPE_STRUCT_END
#undef PROCESS_TYPE_STRUCT_DECLARE_INTERNAL

#endif  // CRASHPAD_SNAPSHOT_MAC_PROCESS_TYPES_H_
