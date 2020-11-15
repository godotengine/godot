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
#include <uuid/uuid.h>

#include <memory>

#include "base/macros.h"
#include "snapshot/mac/process_types/internal.h"
#include "util/mach/task_memory.h"

namespace crashpad {
namespace {

// Assign() is used by each flavor's ReadInto implementation to copy data from a
// specific struct to a generic struct. For fundamental types, the assignment
// operator suffices. For other types such as arrays, an explicit Assign
// specialization is needed, typically performing a copy.

template <typename DestinationType, typename SourceType>
inline void Assign(DestinationType* destination, const SourceType& source) {
  *destination = source;
}

template <typename Type>
inline void Assign(Type* destination, const Type& source) {
  memcpy(destination, &source, sizeof(source));
}

template <>
inline void Assign<process_types::internal::Reserved32_32Only64,
                   process_types::internal::Reserved32_32Only32>(
    process_types::internal::Reserved32_32Only64* destination,
    const process_types::internal::Reserved32_32Only32& source) {
  // Reserved32_32Only32 carries no data and has no storage in the 64-bit
  // structure.
}

template <>
inline void Assign<process_types::internal::Reserved32_64Only64,
                   process_types::internal::Reserved32_64Only32>(
    process_types::internal::Reserved32_64Only64* destination,
    const process_types::internal::Reserved32_64Only32& source) {
  // Reserved32_64Only32 carries no data.
  *destination = 0;
}

template <>
inline void Assign<process_types::internal::Reserved64_64Only64,
                   process_types::internal::Reserved64_64Only32>(
    process_types::internal::Reserved64_64Only64* destination,
    const process_types::internal::Reserved64_64Only32& source) {
  // Reserved64_64Only32 carries no data.
  *destination = 0;
}

using UInt32Array4 = uint32_t[4];
using UInt64Array4 = uint64_t[4];
template <>
inline void Assign<UInt64Array4, UInt32Array4>(UInt64Array4* destination,
                                               const UInt32Array4& source) {
  for (size_t index = 0; index < arraysize(source); ++index) {
    (*destination)[index] = source[index];
  }
}

}  // namespace
}  // namespace crashpad

// Implement the generic crashpad::process_types::struct_name ReadInto(), which
// delegates to the templatized ReadIntoInternal(), which reads the specific
// struct type from the remote process and genericizes it. Also implement
// crashpad::process_types::internal::struct_name<> GenericizeInto(), which
// operates on each member in the struct.
#define PROCESS_TYPE_STRUCT_IMPLEMENT 1

#define PROCESS_TYPE_STRUCT_BEGIN(struct_name)                            \
  namespace crashpad {                                                    \
  namespace process_types {                                               \
                                                                          \
  /* static */                                                            \
  size_t struct_name::ExpectedSize(ProcessReaderMac* process_reader) {    \
    if (!process_reader->Is64Bit()) {                                     \
      return internal::struct_name<internal::Traits32>::Size();           \
    } else {                                                              \
      return internal::struct_name<internal::Traits64>::Size();           \
    }                                                                     \
  }                                                                       \
                                                                          \
  /* static */                                                            \
  bool struct_name::ReadInto(ProcessReaderMac* process_reader,            \
                             mach_vm_address_t address,                   \
                             struct_name* generic) {                      \
    if (!process_reader->Is64Bit()) {                                     \
      return ReadIntoInternal<internal::struct_name<internal::Traits32>>( \
          process_reader, address, generic);                              \
    } else {                                                              \
      return ReadIntoInternal<internal::struct_name<internal::Traits64>>( \
          process_reader, address, generic);                              \
    }                                                                     \
  }                                                                       \
                                                                          \
  /* static */                                                            \
  template <typename T>                                                   \
  bool struct_name::ReadIntoInternal(ProcessReaderMac* process_reader,    \
                                     mach_vm_address_t address,           \
                                     struct_name* generic) {              \
    T specific;                                                           \
    if (!specific.Read(process_reader, address)) {                        \
      return false;                                                       \
    }                                                                     \
    specific.GenericizeInto(generic, &generic->size_);                    \
    return true;                                                          \
  }                                                                       \
                                                                          \
  namespace internal {                                                    \
                                                                          \
  template <typename Traits>                                              \
  void struct_name<Traits>::GenericizeInto(                               \
      process_types::struct_name* generic,                                \
      size_t* specific_size) {                                            \
    *specific_size = Size();

#define PROCESS_TYPE_STRUCT_MEMBER(member_type, member_name, ...) \
    Assign(&generic->member_name, member_name);

#define PROCESS_TYPE_STRUCT_VERSIONED(struct_name, version_field)

#define PROCESS_TYPE_STRUCT_SIZED(struct_name, size_field)

#define PROCESS_TYPE_STRUCT_END(struct_name) \
  }                                          \
  }  /* namespace internal */                \
  }  /* namespace process_types */           \
  }  /* namespace crashpad */

#include "snapshot/mac/process_types/all.proctype"

#undef PROCESS_TYPE_STRUCT_BEGIN
#undef PROCESS_TYPE_STRUCT_MEMBER
#undef PROCESS_TYPE_STRUCT_VERSIONED
#undef PROCESS_TYPE_STRUCT_SIZED
#undef PROCESS_TYPE_STRUCT_END
#undef PROCESS_TYPE_STRUCT_IMPLEMENT

// Implement the specific crashpad::process_types::internal::struct_name<>
// ReadInto(). The default implementation simply reads the struct from the
// remote process. This is separated from other method implementations because
// some types may wish to provide custom readers. This can be done by guarding
// such types’ proctype definitions against this macro and providing custom
// implementations in snapshot/mac/process_types/custom.cc.
#define PROCESS_TYPE_STRUCT_IMPLEMENT_INTERNAL_READ_INTO 1

#define PROCESS_TYPE_STRUCT_BEGIN(struct_name)                         \
  namespace crashpad {                                                 \
  namespace process_types {                                            \
  namespace internal {                                                 \
                                                                       \
  /* static */                                                         \
  template <typename Traits>                                           \
  bool struct_name<Traits>::ReadInto(ProcessReaderMac* process_reader, \
                                     mach_vm_address_t address,        \
                                     struct_name<Traits>* specific) {  \
    return process_reader->Memory()->Read(                             \
        address, sizeof(*specific), specific);                         \
  }                                                                    \
  } /* namespace internal */                                           \
  } /* namespace process_types */                                      \
  } /* namespace crashpad */

#define PROCESS_TYPE_STRUCT_MEMBER(member_type, member_name, ...)

#define PROCESS_TYPE_STRUCT_VERSIONED(struct_name, version_field)

#define PROCESS_TYPE_STRUCT_SIZED(struct_name, size_field)

#define PROCESS_TYPE_STRUCT_END(struct_name)

#include "snapshot/mac/process_types/all.proctype"

#undef PROCESS_TYPE_STRUCT_BEGIN
#undef PROCESS_TYPE_STRUCT_MEMBER
#undef PROCESS_TYPE_STRUCT_VERSIONED
#undef PROCESS_TYPE_STRUCT_SIZED
#undef PROCESS_TYPE_STRUCT_END
#undef PROCESS_TYPE_STRUCT_IMPLEMENT_INTERNAL_READ_INTO

// Implement the array operations. These are separated from other method
// implementations because some types are variable-length and are never stored
// as direct-access arrays. It would be incorrect to provide reader
// implementations for such types. Types that wish to suppress array operations
// can do so by guarding their proctype definitions against this macro.
#define PROCESS_TYPE_STRUCT_IMPLEMENT_ARRAY 1

#define PROCESS_TYPE_STRUCT_BEGIN(struct_name)                                 \
  namespace crashpad {                                                         \
  namespace process_types {                                                    \
  namespace internal {                                                         \
                                                                               \
  /* static */                                                                 \
  template <typename Traits>                                                   \
  bool struct_name<Traits>::ReadArrayInto(ProcessReaderMac* process_reader,    \
                                          mach_vm_address_t address,           \
                                          size_t count,                        \
                                          struct_name<Traits>* specific) {     \
    return process_reader->Memory()->Read(                                     \
        address, sizeof(struct_name<Traits>[count]), specific);                \
  }                                                                            \
                                                                               \
  } /* namespace internal */                                                   \
                                                                               \
  /* static */                                                                 \
  bool struct_name::ReadArrayInto(ProcessReaderMac* process_reader,            \
                                  mach_vm_address_t address,                   \
                                  size_t count,                                \
                                  struct_name* generic) {                      \
    if (!process_reader->Is64Bit()) {                                          \
      return ReadArrayIntoInternal<internal::struct_name<internal::Traits32>>( \
          process_reader, address, count, generic);                            \
    } else {                                                                   \
      return ReadArrayIntoInternal<internal::struct_name<internal::Traits64>>( \
          process_reader, address, count, generic);                            \
    }                                                                          \
    return true;                                                               \
  }                                                                            \
                                                                               \
  /* static */                                                                 \
  template <typename T>                                                        \
  bool struct_name::ReadArrayIntoInternal(ProcessReaderMac* process_reader,    \
                                          mach_vm_address_t address,           \
                                          size_t count,                        \
                                          struct_name* generic) {              \
    std::unique_ptr<T[]> specific(new T[count]);                               \
    if (!T::ReadArrayInto(process_reader, address, count, &specific[0])) {     \
      return false;                                                            \
    }                                                                          \
    for (size_t index = 0; index < count; ++index) {                           \
      specific[index].GenericizeInto(&generic[index], &generic[index].size_);  \
    }                                                                          \
    return true;                                                               \
  }                                                                            \
  } /* namespace process_types */                                              \
  } /* namespace crashpad */

#define PROCESS_TYPE_STRUCT_MEMBER(member_type, member_name, ...)

#define PROCESS_TYPE_STRUCT_VERSIONED(struct_name, version_field)

#define PROCESS_TYPE_STRUCT_SIZED(struct_name, size_field)

#define PROCESS_TYPE_STRUCT_END(struct_name)

#include "snapshot/mac/process_types/all.proctype"

#undef PROCESS_TYPE_STRUCT_BEGIN
#undef PROCESS_TYPE_STRUCT_MEMBER
#undef PROCESS_TYPE_STRUCT_VERSIONED
#undef PROCESS_TYPE_STRUCT_SIZED
#undef PROCESS_TYPE_STRUCT_END
#undef PROCESS_TYPE_STRUCT_IMPLEMENT_ARRAY

// Implement the generic crashpad::process_types::struct_name
// ExpectedSizeForVersion(), which delegates to the templatized
// ExpectedSizeForVersion(), which returns the expected size of a versioned
// structure given a version parameter. This is only implemented for structures
// that use PROCESS_TYPE_STRUCT_VERSIONED(), and implementations of the internal
// templatized functions must be provided in
// snapshot/mac/process_types/custom.cc.
#define PROCESS_TYPE_STRUCT_IMPLEMENT_VERSIONED 1

#define PROCESS_TYPE_STRUCT_BEGIN(struct_name)

#define PROCESS_TYPE_STRUCT_MEMBER(member_type, member_name, ...)

#define PROCESS_TYPE_STRUCT_VERSIONED(struct_name, version_field) \
  namespace crashpad {                                            \
  namespace process_types {                                       \
                                                                  \
  /* static */                                                    \
  size_t struct_name::ExpectedSizeForVersion(                     \
      ProcessReaderMac* process_reader,                           \
      decltype(struct_name::version_field) version) {             \
    if (!process_reader->Is64Bit()) {                             \
      return internal::struct_name<                               \
          internal::Traits32>::ExpectedSizeForVersion(version);   \
    } else {                                                      \
      return internal::struct_name<                               \
          internal::Traits64>::ExpectedSizeForVersion(version);   \
    }                                                             \
  }                                                               \
                                                                  \
  } /* namespace process_types */                                 \
  } /* namespace crashpad */

#define PROCESS_TYPE_STRUCT_SIZED(struct_name, size_field)

#define PROCESS_TYPE_STRUCT_END(struct_name)

#include "snapshot/mac/process_types/all.proctype"

#undef PROCESS_TYPE_STRUCT_BEGIN
#undef PROCESS_TYPE_STRUCT_MEMBER
#undef PROCESS_TYPE_STRUCT_VERSIONED
#undef PROCESS_TYPE_STRUCT_SIZED
#undef PROCESS_TYPE_STRUCT_END
#undef PROCESS_TYPE_STRUCT_IMPLEMENT_VERSIONED

// Implement the generic crashpad::process_types::struct_name MinimumSize() and
// its templatized equivalent. The generic version delegates to the templatized
// one, which returns the minimum size of a sized structure. This can be used to
// ensure that enough of a sized structure is available to interpret its size
// field. This is only implemented for structures that use
// PROCESS_TYPE_STRUCT_SIZED().
#define PROCESS_TYPE_STRUCT_IMPLEMENT_SIZED 1

#define PROCESS_TYPE_STRUCT_BEGIN(struct_name)

#define PROCESS_TYPE_STRUCT_MEMBER(member_type, member_name, ...)

#define PROCESS_TYPE_STRUCT_VERSIONED(struct_name, version_field)

#define PROCESS_TYPE_STRUCT_SIZED(struct_name, size_field)             \
  namespace crashpad {                                                 \
  namespace process_types {                                            \
                                                                       \
  namespace internal {                                                 \
                                                                       \
  /* static */                                                         \
  template <typename Traits>                                           \
  size_t struct_name<Traits>::MinimumSize() {                          \
    return offsetof(struct_name<Traits>, size_field) +                 \
           sizeof(struct_name<Traits>::size_field);                    \
  }                                                                    \
                                                                       \
  /* Explicit instantiations of the above. */                          \
  template size_t struct_name<Traits32>::MinimumSize();                \
  template size_t struct_name<Traits64>::MinimumSize();                \
                                                                       \
  } /* namespace internal */                                           \
                                                                       \
  /* static */                                                         \
  size_t struct_name::MinimumSize(ProcessReaderMac* process_reader) {  \
    if (!process_reader->Is64Bit()) {                                  \
      return internal::struct_name<internal::Traits32>::MinimumSize(); \
    } else {                                                           \
      return internal::struct_name<internal::Traits64>::MinimumSize(); \
    }                                                                  \
  }                                                                    \
                                                                       \
  } /* namespace process_types */                                      \
  } /* namespace crashpad */

#define PROCESS_TYPE_STRUCT_END(struct_name)

#include "snapshot/mac/process_types/all.proctype"

#undef PROCESS_TYPE_STRUCT_BEGIN
#undef PROCESS_TYPE_STRUCT_MEMBER
#undef PROCESS_TYPE_STRUCT_VERSIONED
#undef PROCESS_TYPE_STRUCT_SIZED
#undef PROCESS_TYPE_STRUCT_END
#undef PROCESS_TYPE_STRUCT_IMPLEMENT_SIZED
