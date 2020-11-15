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

#ifndef CRASHPAD_MINIDUMP_TEST_MINIDUMP_WRITABLE_TEST_UTIL_H_
#define CRASHPAD_MINIDUMP_TEST_MINIDUMP_WRITABLE_TEST_UTIL_H_

#include <windows.h>
#include <dbghelp.h>
#include <stdint.h>
#include <sys/types.h>

#include <string>

#include "base/macros.h"
#include "gtest/gtest.h"
#include "minidump/minidump_extensions.h"
#include "minidump/minidump_writable.h"

namespace crashpad {

class FileWriterInterface;

namespace test {

//! \brief Returns an untyped minidump object located within a minidump file’s
//!     contents, where the offset and size of the object are known.
//!
//! \param[in] file_contents The contents of the minidump file.
//! \param[in] location A MINIDUMP_LOCATION_DESCRIPTOR giving the offset within
//!     the minidump file of the desired object, as well as its size.
//! \param[in] expected_size The expected size of the object. If \a
//!     allow_oversized_data is `true`, \a expected_size is treated as the
//!     minimum size of \a location, but it is permitted to be larger. If \a
//!     allow_oversized_data is `false`, the size of \a location must match
//!     \a expected_size exactly.
//! \param[in] allow_oversized_data Controls whether \a expected_size is a
//!     minimum limit (`true`) or an exact match is required (`false`).
//!
//! \return If the size of \a location is agrees with \a expected_size, and if
//!     \a location is within the range of \a file_contents, returns a pointer
//!     into \a file_contents at offset \a rva. Otherwise, raises a gtest
//!     assertion failure and returns `nullptr`.
//!
//! Do not call this function. Use the typed version,
//! MinidumpWritableAtLocationDescriptor<>(), or another type-specific function.
const void* MinidumpWritableAtLocationDescriptorInternal(
    const std::string& file_contents,
    const MINIDUMP_LOCATION_DESCRIPTOR& location,
    size_t expected_size,
    bool allow_oversized_data);

//! \brief A traits class defining whether a minidump object type is required to
//!     appear only as a fixed-size object or if it is variable-sized.
//!
//! Variable-sized data is data referenced by a MINIDUMP_LOCATION_DESCRIPTOR
//! whose DataSize field may be larger than the size of the basic object type’s
//! structure. This can happen for types that appear only as variable-sized
//! lists, or types whose final fields are variable-sized lists or other
//! variable-sized data.
template <typename T>
struct MinidumpWritableTraits {
  //! \brief `true` if \a T should be treated as a variable-sized data type,
  //!     where its base size is used solely as a minimum bound. `false` if \a
  //!     T is a fixed-sized type, which should only appear at its base size.
  static const bool kAllowOversizedData = false;
};

#define MINIDUMP_ALLOW_OVERSIZED_DATA(x)          \
  template <>                                     \
  struct MinidumpWritableTraits<x> {              \
    static const bool kAllowOversizedData = true; \
  }

// This type appears only as a variable-sized list.
MINIDUMP_ALLOW_OVERSIZED_DATA(MINIDUMP_DIRECTORY);

// These types are permitted to be oversized because their final fields are
// variable-sized lists.
MINIDUMP_ALLOW_OVERSIZED_DATA(MINIDUMP_MEMORY_LIST);
MINIDUMP_ALLOW_OVERSIZED_DATA(MINIDUMP_MODULE_LIST);
MINIDUMP_ALLOW_OVERSIZED_DATA(MINIDUMP_UNLOADED_MODULE_LIST);
MINIDUMP_ALLOW_OVERSIZED_DATA(MINIDUMP_THREAD_LIST);
MINIDUMP_ALLOW_OVERSIZED_DATA(MINIDUMP_HANDLE_DATA_STREAM);
MINIDUMP_ALLOW_OVERSIZED_DATA(MINIDUMP_MEMORY_INFO_LIST);
MINIDUMP_ALLOW_OVERSIZED_DATA(MinidumpModuleCrashpadInfoList);
MINIDUMP_ALLOW_OVERSIZED_DATA(MinidumpRVAList);
MINIDUMP_ALLOW_OVERSIZED_DATA(MinidumpSimpleStringDictionary);
MINIDUMP_ALLOW_OVERSIZED_DATA(MinidumpAnnotationList);

// These types have final fields carrying variable-sized data (typically string
// data).
MINIDUMP_ALLOW_OVERSIZED_DATA(IMAGE_DEBUG_MISC);
MINIDUMP_ALLOW_OVERSIZED_DATA(MINIDUMP_STRING);
MINIDUMP_ALLOW_OVERSIZED_DATA(CodeViewRecordPDB20);
MINIDUMP_ALLOW_OVERSIZED_DATA(CodeViewRecordPDB70);
MINIDUMP_ALLOW_OVERSIZED_DATA(MinidumpUTF8String);

// minidump_file_writer_test accesses its variable-sized test streams via a
// uint8_t*.
MINIDUMP_ALLOW_OVERSIZED_DATA(uint8_t);

#undef MINIDUMP_ALLOW_OVERSIZED_DATA

//! \brief Returns a typed minidump object located within a minidump file’s
//!     contents, where the offset and size of the object are known.
//!
//! This function is similar to MinidumpWritableAtLocationDescriptor<>() and is
//! used to implement that function. It exists independently so that template
//! specializations are able to call this function, which provides the default
//! implementation.
//!
//! Do not call this function directly. Use
//! MinidumpWritableAtLocationDescriptor<>() instead.
template <typename T>
const T* TMinidumpWritableAtLocationDescriptor(
    const std::string& file_contents,
    const MINIDUMP_LOCATION_DESCRIPTOR& location) {
  return reinterpret_cast<const T*>(
      MinidumpWritableAtLocationDescriptorInternal(
          file_contents,
          location,
          sizeof(T),
          MinidumpWritableTraits<T>::kAllowOversizedData));
}

//! \brief Returns a typed minidump object located within a minidump file’s
//!     contents, where the offset and size of the object are known.
//!
//! This function has template specializations that perform more stringent
//! checking than the default implementation:
//!  - With a MINIDUMP_HEADER template parameter, a template specialization
//!    ensures that the structure’s magic number and version fields are correct.
//!  - With a MINIDUMP_MEMORY_LIST, MINIDUMP_THREAD_LIST, MINIDUMP_MODULE_LIST,
//!    MINIDUMP_MEMORY_INFO_LIST, MinidumpSimpleStringDictionary, or
//!    MinidumpAnnotationList template parameter, template specializations
//!    ensure that the size given by \a location matches the size expected of a
//!    stream containing the number of elements it claims to have.
//!  - With an IMAGE_DEBUG_MISC, CodeViewRecordPDB20, or CodeViewRecordPDB70
//!    template parameter, template specializations ensure that the structure
//!    has the expected format including any magic number and the `NUL`-
//!    terminated string.
//!
//! \param[in] file_contents The contents of the minidump file.
//! \param[in] location A MINIDUMP_LOCATION_DESCRIPTOR giving the offset within
//!     the minidump file of the desired object, as well as its size.
//!
//! \return If the size of \a location is at least as big as the size of the
//!     requested object, and if \a location is within the range of \a
//!     file_contents, returns a pointer into \a file_contents at offset \a rva.
//!     Otherwise, raises a gtest assertion failure and returns `nullptr`.
//!
//! \sa MinidumpWritableAtRVA()
template <typename T>
const T* MinidumpWritableAtLocationDescriptor(
    const std::string& file_contents,
    const MINIDUMP_LOCATION_DESCRIPTOR& location) {
  return TMinidumpWritableAtLocationDescriptor<T>(file_contents, location);
}

template <>
const IMAGE_DEBUG_MISC* MinidumpWritableAtLocationDescriptor<IMAGE_DEBUG_MISC>(
    const std::string& file_contents,
    const MINIDUMP_LOCATION_DESCRIPTOR& location);

template <>
const MINIDUMP_HEADER* MinidumpWritableAtLocationDescriptor<MINIDUMP_HEADER>(
    const std::string& file_contents,
    const MINIDUMP_LOCATION_DESCRIPTOR& location);

template <>
const MINIDUMP_MEMORY_LIST* MinidumpWritableAtLocationDescriptor<
    MINIDUMP_MEMORY_LIST>(const std::string& file_contents,
                          const MINIDUMP_LOCATION_DESCRIPTOR& location);

template <>
const MINIDUMP_MODULE_LIST* MinidumpWritableAtLocationDescriptor<
    MINIDUMP_MODULE_LIST>(const std::string& file_contents,
                          const MINIDUMP_LOCATION_DESCRIPTOR& location);

template <>
const MINIDUMP_UNLOADED_MODULE_LIST*
MinidumpWritableAtLocationDescriptor<MINIDUMP_UNLOADED_MODULE_LIST>(
    const std::string& file_contents,
    const MINIDUMP_LOCATION_DESCRIPTOR& location);

template <>
const MINIDUMP_THREAD_LIST* MinidumpWritableAtLocationDescriptor<
    MINIDUMP_THREAD_LIST>(const std::string& file_contents,
                          const MINIDUMP_LOCATION_DESCRIPTOR& location);

template <>
const MINIDUMP_HANDLE_DATA_STREAM* MinidumpWritableAtLocationDescriptor<
    MINIDUMP_HANDLE_DATA_STREAM>(const std::string& file_contents,
                                 const MINIDUMP_LOCATION_DESCRIPTOR& location);

template <>
const MINIDUMP_MEMORY_INFO_LIST* MinidumpWritableAtLocationDescriptor<
    MINIDUMP_MEMORY_INFO_LIST>(const std::string& file_contents,
                               const MINIDUMP_LOCATION_DESCRIPTOR& location);

template <>
const CodeViewRecordPDB20* MinidumpWritableAtLocationDescriptor<
    CodeViewRecordPDB20>(const std::string& file_contents,
                         const MINIDUMP_LOCATION_DESCRIPTOR& location);

template <>
const CodeViewRecordPDB70* MinidumpWritableAtLocationDescriptor<
    CodeViewRecordPDB70>(const std::string& file_contents,
                         const MINIDUMP_LOCATION_DESCRIPTOR& location);

template <>
const MinidumpModuleCrashpadInfoList*
MinidumpWritableAtLocationDescriptor<MinidumpModuleCrashpadInfoList>(
    const std::string& file_contents,
    const MINIDUMP_LOCATION_DESCRIPTOR& location);

template <>
const MinidumpSimpleStringDictionary*
MinidumpWritableAtLocationDescriptor<MinidumpSimpleStringDictionary>(
    const std::string& file_contents,
    const MINIDUMP_LOCATION_DESCRIPTOR& location);

template <>
const MinidumpAnnotationList*
MinidumpWritableAtLocationDescriptor<MinidumpAnnotationList>(
    const std::string& file_contents,
    const MINIDUMP_LOCATION_DESCRIPTOR& location);

//! \brief Returns a typed minidump object located within a minidump file’s
//!     contents, where the offset of the object is known.
//!
//! \param[in] file_contents The contents of the minidump file.
//! \param[in] rva The offset within the minidump file of the desired object.
//!
//! \return If \a rva plus the size of an object of type \a T is within the
//!     range of \a file_contents, returns a pointer into \a file_contents at
//!     offset \a rva. Otherwise, raises a gtest assertion failure and returns
//!     `nullptr`.
//!
//! \sa MinidumpWritableAtLocationDescriptor<>()
template <typename T>
const T* MinidumpWritableAtRVA(const std::string& file_contents, RVA rva) {
  MINIDUMP_LOCATION_DESCRIPTOR location;
  location.DataSize = sizeof(T);
  location.Rva = rva;
  return MinidumpWritableAtLocationDescriptor<T>(file_contents, location);
}

//! \brief An internal::MinidumpWritable that carries a `uint32_t` for testing.
class TestUInt32MinidumpWritable final : public internal::MinidumpWritable {
 public:
  //! \brief Constructs the object to write a `uint32_t` with value \a value.
  explicit TestUInt32MinidumpWritable(uint32_t value);

  ~TestUInt32MinidumpWritable() override;

 protected:
  // MinidumpWritable:
  size_t SizeOfObject() override;
  bool WriteObject(FileWriterInterface* file_writer) override;

 private:
  uint32_t value_;

  DISALLOW_COPY_AND_ASSIGN(TestUInt32MinidumpWritable);
};

}  // namespace test
}  // namespace crashpad

#endif  // CRASHPAD_MINIDUMP_TEST_MINIDUMP_WRITABLE_TEST_UTIL_H_
