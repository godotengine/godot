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

#include "minidump/test/minidump_writable_test_util.h"

#include <stddef.h>

#include <string>

#include "base/strings/string16.h"
#include "gtest/gtest.h"
#include "util/file/file_writer.h"
#include "util/misc/implicit_cast.h"

namespace crashpad {
namespace test {

namespace {

//! \brief Returns an untyped minidump object located within a minidump file’s
//!     contents, where the offset of the object is known.
//!
//! \param[in] file_contents The contents of the minidump file.
//! \param[in] rva The offset within the minidump file of the desired object.
//!
//! \return If \a rva is within the range of \a file_contents, returns a pointer
//!     into \a file_contents at offset \a rva. Otherwise, raises a gtest
//!     assertion failure and returns `nullptr`.
//!
//! Do not call this function. Use the typed version, MinidumpWritableAtRVA<>(),
//! or another type-specific function.
const void* MinidumpWritableAtRVAInternal(const std::string& file_contents,
                                          RVA rva) {
  if (rva >= file_contents.size()) {
    EXPECT_LT(rva, file_contents.size());
    return nullptr;
  }

  return &file_contents[rva];
}

}  // namespace

const void* MinidumpWritableAtLocationDescriptorInternal(
    const std::string& file_contents,
    const MINIDUMP_LOCATION_DESCRIPTOR& location,
    size_t expected_size,
    bool allow_oversized_data) {
  if (location.DataSize == 0) {
    EXPECT_EQ(location.Rva, 0u);
    return nullptr;
  }

  if (allow_oversized_data) {
    if (location.DataSize < expected_size) {
      EXPECT_GE(location.DataSize, expected_size);
      return nullptr;
    }
  } else if (location.DataSize != expected_size) {
    EXPECT_EQ(location.DataSize, expected_size);
    return nullptr;
  }

  RVA end = location.Rva + location.DataSize;
  if (end > file_contents.size()) {
    EXPECT_LE(end, file_contents.size());
    return nullptr;
  }

  const void* rv = MinidumpWritableAtRVAInternal(file_contents, location.Rva);

  return rv;
}

template <>
const IMAGE_DEBUG_MISC* MinidumpWritableAtLocationDescriptor<IMAGE_DEBUG_MISC>(
    const std::string& file_contents,
    const MINIDUMP_LOCATION_DESCRIPTOR& location) {
  const IMAGE_DEBUG_MISC* misc =
      TMinidumpWritableAtLocationDescriptor<IMAGE_DEBUG_MISC>(file_contents,
                                                              location);
  if (!misc) {
    return nullptr;
  }

  if (misc->DataType != IMAGE_DEBUG_MISC_EXENAME) {
    EXPECT_EQ(misc->DataType,
              implicit_cast<uint32_t>(IMAGE_DEBUG_MISC_EXENAME));
    return nullptr;
  }

  if (misc->Length != location.DataSize) {
    EXPECT_EQ(misc->Length, location.DataSize);
    return nullptr;
  }

  if (misc->Unicode == 0) {
    size_t string_length = misc->Length - offsetof(IMAGE_DEBUG_MISC, Data) - 1;
    if (misc->Data[string_length] != '\0') {
      EXPECT_EQ(misc->Data[string_length], '\0');
      return nullptr;
    }
  } else if (misc->Unicode == 1) {
    if (misc->Length % sizeof(base::char16) != 0) {
      EXPECT_EQ(misc->Length % sizeof(base::char16), 0u);
      return nullptr;
    }

    size_t string_length = (misc->Length - offsetof(IMAGE_DEBUG_MISC, Data)) /
                               sizeof(base::char16) -
                           1;
    const base::char16* data16 =
        reinterpret_cast<const base::char16*>(misc->Data);
    if (data16[string_length] != '\0') {
      EXPECT_EQ(data16[string_length], '\0');
      return nullptr;
    }
  } else {
    ADD_FAILURE() << "misc->Unicode " << misc->Unicode;
    return nullptr;
  }

  return misc;
}

template <>
const MINIDUMP_HEADER* MinidumpWritableAtLocationDescriptor<MINIDUMP_HEADER>(
    const std::string& file_contents,
    const MINIDUMP_LOCATION_DESCRIPTOR& location) {
  const MINIDUMP_HEADER* header =
      TMinidumpWritableAtLocationDescriptor<MINIDUMP_HEADER>(file_contents,
                                                             location);
  if (!header) {
    return nullptr;
  }

  if (header->Signature != MINIDUMP_SIGNATURE) {
    EXPECT_EQ(header->Signature, implicit_cast<uint32_t>(MINIDUMP_SIGNATURE));
    return nullptr;
  }
  if (header->Version != MINIDUMP_VERSION) {
    EXPECT_EQ(header->Version, implicit_cast<uint32_t>(MINIDUMP_VERSION));
    return nullptr;
  }

  return header;
}

namespace {

struct MinidumpMemoryListTraits {
  using ListType = MINIDUMP_MEMORY_LIST;
  enum : size_t { kElementSize = sizeof(MINIDUMP_MEMORY_DESCRIPTOR) };
  static size_t ElementCount(const ListType* list) {
    return list->NumberOfMemoryRanges;
  }
};

struct MinidumpModuleListTraits {
  using ListType = MINIDUMP_MODULE_LIST;
  enum : size_t { kElementSize = sizeof(MINIDUMP_MODULE) };
  static size_t ElementCount(const ListType* list) {
    return list->NumberOfModules;
  }
};

struct MinidumpUnloadedModuleListTraits {
  using ListType = MINIDUMP_UNLOADED_MODULE_LIST;
  enum : size_t { kElementSize = sizeof(MINIDUMP_UNLOADED_MODULE) };
  static size_t ElementCount(const ListType* list) {
    return list->NumberOfEntries;
  }
};

struct MinidumpThreadListTraits {
  using ListType = MINIDUMP_THREAD_LIST;
  enum : size_t { kElementSize = sizeof(MINIDUMP_THREAD) };
  static size_t ElementCount(const ListType* list) {
    return list->NumberOfThreads;
  }
};

struct MinidumpHandleDataStreamTraits {
  using ListType = MINIDUMP_HANDLE_DATA_STREAM;
  enum : size_t { kElementSize = sizeof(MINIDUMP_HANDLE_DESCRIPTOR) };
  static size_t ElementCount(const ListType* list) {
    return static_cast<size_t>(list->NumberOfDescriptors);
  }
};

struct MinidumpMemoryInfoListTraits {
  using ListType = MINIDUMP_MEMORY_INFO_LIST;
  enum : size_t { kElementSize = sizeof(MINIDUMP_MEMORY_INFO) };
  static size_t ElementCount(const ListType* list) {
    return static_cast<size_t>(list->NumberOfEntries);
  }
};

struct MinidumpModuleCrashpadInfoListTraits {
  using ListType = MinidumpModuleCrashpadInfoList;
  enum : size_t { kElementSize = sizeof(MinidumpModuleCrashpadInfoLink) };
  static size_t ElementCount(const ListType* list) {
    return list->count;
  }
};

struct MinidumpSimpleStringDictionaryListTraits {
  using ListType = MinidumpSimpleStringDictionary;
  enum : size_t { kElementSize = sizeof(MinidumpSimpleStringDictionaryEntry) };
  static size_t ElementCount(const ListType* list) {
    return list->count;
  }
};

struct MinidumpAnnotationListObjectsTraits {
  using ListType = MinidumpAnnotationList;
  enum : size_t { kElementSize = sizeof(MinidumpAnnotation) };
  static size_t ElementCount(const ListType* list) { return list->count; }
};

template <typename T>
const typename T::ListType* MinidumpListAtLocationDescriptor(
    const std::string& file_contents,
    const MINIDUMP_LOCATION_DESCRIPTOR& location) {
  const typename T::ListType* list =
      TMinidumpWritableAtLocationDescriptor<typename T::ListType>(file_contents,
                                                                  location);
  if (!list) {
    return nullptr;
  }

  size_t expected_size =
      sizeof(typename T::ListType) + T::ElementCount(list) * T::kElementSize;
  if (location.DataSize != expected_size) {
    EXPECT_EQ(location.DataSize, expected_size);
    return nullptr;
  }

  return list;
}

}  // namespace

template <>
const MINIDUMP_MEMORY_LIST* MinidumpWritableAtLocationDescriptor<
    MINIDUMP_MEMORY_LIST>(const std::string& file_contents,
                          const MINIDUMP_LOCATION_DESCRIPTOR& location) {
  return MinidumpListAtLocationDescriptor<MinidumpMemoryListTraits>(
      file_contents, location);
}

template <>
const MINIDUMP_MODULE_LIST* MinidumpWritableAtLocationDescriptor<
    MINIDUMP_MODULE_LIST>(const std::string& file_contents,
                          const MINIDUMP_LOCATION_DESCRIPTOR& location) {
  return MinidumpListAtLocationDescriptor<MinidumpModuleListTraits>(
      file_contents, location);
}

template <>
const MINIDUMP_UNLOADED_MODULE_LIST*
MinidumpWritableAtLocationDescriptor<MINIDUMP_UNLOADED_MODULE_LIST>(
    const std::string& file_contents,
    const MINIDUMP_LOCATION_DESCRIPTOR& location) {
  return MinidumpListAtLocationDescriptor<MinidumpUnloadedModuleListTraits>(
      file_contents, location);
}

template <>
const MINIDUMP_THREAD_LIST* MinidumpWritableAtLocationDescriptor<
    MINIDUMP_THREAD_LIST>(const std::string& file_contents,
                          const MINIDUMP_LOCATION_DESCRIPTOR& location) {
  return MinidumpListAtLocationDescriptor<MinidumpThreadListTraits>(
      file_contents, location);
}

template <>
const MINIDUMP_HANDLE_DATA_STREAM* MinidumpWritableAtLocationDescriptor<
    MINIDUMP_HANDLE_DATA_STREAM>(const std::string& file_contents,
                                 const MINIDUMP_LOCATION_DESCRIPTOR& location) {
  return MinidumpListAtLocationDescriptor<MinidumpHandleDataStreamTraits>(
      file_contents, location);
}

template <>
const MINIDUMP_MEMORY_INFO_LIST* MinidumpWritableAtLocationDescriptor<
    MINIDUMP_MEMORY_INFO_LIST>(const std::string& file_contents,
                               const MINIDUMP_LOCATION_DESCRIPTOR& location) {
  return MinidumpListAtLocationDescriptor<MinidumpMemoryInfoListTraits>(
      file_contents, location);
}

template <>
const MinidumpModuleCrashpadInfoList*
MinidumpWritableAtLocationDescriptor<MinidumpModuleCrashpadInfoList>(
    const std::string& file_contents,
    const MINIDUMP_LOCATION_DESCRIPTOR& location) {
  return MinidumpListAtLocationDescriptor<MinidumpModuleCrashpadInfoListTraits>(
      file_contents, location);
}

template <>
const MinidumpSimpleStringDictionary*
MinidumpWritableAtLocationDescriptor<MinidumpSimpleStringDictionary>(
    const std::string& file_contents,
    const MINIDUMP_LOCATION_DESCRIPTOR& location) {
  return MinidumpListAtLocationDescriptor<
      MinidumpSimpleStringDictionaryListTraits>(file_contents, location);
}

template <>
const MinidumpAnnotationList*
MinidumpWritableAtLocationDescriptor<MinidumpAnnotationList>(
    const std::string& file_contents,
    const MINIDUMP_LOCATION_DESCRIPTOR& location) {
  return MinidumpListAtLocationDescriptor<MinidumpAnnotationListObjectsTraits>(
      file_contents, location);
}

namespace {

template <typename T>
const T* MinidumpCVPDBAtLocationDescriptor(
    const std::string& file_contents,
    const MINIDUMP_LOCATION_DESCRIPTOR& location) {
  const T* cv_pdb =
      TMinidumpWritableAtLocationDescriptor<T>(file_contents, location);
  if (!cv_pdb) {
    return nullptr;
  }

  if (cv_pdb->signature != T::kSignature) {
    EXPECT_EQ(cv_pdb->signature, T::kSignature);
    return nullptr;
  }

  size_t string_length = location.DataSize - offsetof(T, pdb_name) - 1;
  if (cv_pdb->pdb_name[string_length] != '\0') {
    EXPECT_EQ(cv_pdb->pdb_name[string_length], '\0');
    return nullptr;
  }

  return cv_pdb;
}

}  // namespace

template <>
const CodeViewRecordPDB20* MinidumpWritableAtLocationDescriptor<
    CodeViewRecordPDB20>(const std::string& file_contents,
                         const MINIDUMP_LOCATION_DESCRIPTOR& location) {
  return MinidumpCVPDBAtLocationDescriptor<CodeViewRecordPDB20>(file_contents,
                                                                location);
}

template <>
const CodeViewRecordPDB70* MinidumpWritableAtLocationDescriptor<
    CodeViewRecordPDB70>(const std::string& file_contents,
                         const MINIDUMP_LOCATION_DESCRIPTOR& location) {
  return MinidumpCVPDBAtLocationDescriptor<CodeViewRecordPDB70>(file_contents,
                                                                location);
}

TestUInt32MinidumpWritable::TestUInt32MinidumpWritable(uint32_t value)
    : MinidumpWritable(),
      value_(value) {
}

TestUInt32MinidumpWritable::~TestUInt32MinidumpWritable() {
}

size_t TestUInt32MinidumpWritable::SizeOfObject() {
  return sizeof(value_);
}

bool TestUInt32MinidumpWritable::WriteObject(FileWriterInterface* file_writer) {
  return file_writer->Write(&value_, sizeof(value_));
}

}  // namespace test
}  // namespace crashpad
