// Copyright 2017 The Crashpad Authors. All rights reserved.
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

#include "snapshot/minidump/minidump_annotation_reader.h"

#include <stdint.h>

#include "base/logging.h"
#include "minidump/minidump_extensions.h"
#include "snapshot/minidump/minidump_string_reader.h"

namespace crashpad {
namespace internal {

namespace {

bool ReadMinidumpByteArray(FileReaderInterface* file_reader,
                           RVA rva,
                           std::vector<uint8_t>* data) {
  if (rva == 0) {
    data->clear();
    return true;
  }

  if (!file_reader->SeekSet(rva)) {
    return false;
  }

  uint32_t length;
  if (!file_reader->ReadExactly(&length, sizeof(length))) {
    return false;
  }

  std::vector<uint8_t> local_data(length);
  if (!file_reader->ReadExactly(local_data.data(), length)) {
    return false;
  }

  data->swap(local_data);
  return true;
}

}  // namespace

bool ReadMinidumpAnnotationList(FileReaderInterface* file_reader,
                                const MINIDUMP_LOCATION_DESCRIPTOR& location,
                                std::vector<AnnotationSnapshot>* list) {
  if (location.Rva == 0) {
    list->clear();
    return true;
  }

  if (location.DataSize < sizeof(MinidumpAnnotationList)) {
    LOG(ERROR) << "annotation list size mismatch";
    return false;
  }

  if (!file_reader->SeekSet(location.Rva)) {
    return false;
  }

  uint32_t count;
  if (!file_reader->ReadExactly(&count, sizeof(count))) {
    return false;
  }

  if (location.DataSize !=
      sizeof(MinidumpAnnotationList) + count * sizeof(MinidumpAnnotation)) {
    LOG(ERROR) << "annotation object size mismatch";
    return false;
  }

  std::vector<MinidumpAnnotation> minidump_annotations(count);
  if (!file_reader->ReadExactly(minidump_annotations.data(),
                                count * sizeof(MinidumpAnnotation))) {
    return false;
  }

  std::vector<AnnotationSnapshot> annotations;
  annotations.reserve(count);

  for (size_t i = 0; i < count; ++i) {
    const MinidumpAnnotation* minidump_annotation = &minidump_annotations[i];

    AnnotationSnapshot annotation;
    // The client-exposed size of this field is 16-bit, but the minidump field
    // is 32-bit for padding. Take just the lower part.
    annotation.type = static_cast<uint16_t>(minidump_annotation->type);

    if (!ReadMinidumpUTF8String(
            file_reader, minidump_annotation->name, &annotation.name)) {
      return false;
    }

    if (!ReadMinidumpByteArray(
            file_reader, minidump_annotation->value, &annotation.value)) {
      return false;
    }

    annotations.push_back(std::move(annotation));
  }

  list->swap(annotations);
  return true;
}

}  // namespace internal
}  // namespace crashpad
