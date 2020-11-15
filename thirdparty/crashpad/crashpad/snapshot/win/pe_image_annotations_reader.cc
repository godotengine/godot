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

#include "snapshot/win/pe_image_annotations_reader.h"

#include <string.h>
#include <sys/types.h>

#include "base/strings/utf_string_conversions.h"
#include "client/annotation.h"
#include "client/simple_string_dictionary.h"
#include "snapshot/snapshot_constants.h"
#include "snapshot/win/pe_image_reader.h"
#include "snapshot/win/process_reader_win.h"
#include "util/win/process_structs.h"

namespace crashpad {

namespace process_types {

template <class Traits>
struct Annotation {
  typename Traits::Pointer link_node;
  typename Traits::Pointer name;
  typename Traits::Pointer value;
  uint32_t size;
  uint16_t type;
};

template <class Traits>
struct AnnotationList {
  typename Traits::Pointer tail_pointer;
  Annotation<Traits> head;
  Annotation<Traits> tail;
};

}  // namespace process_types

PEImageAnnotationsReader::PEImageAnnotationsReader(
    ProcessReaderWin* process_reader,
    const PEImageReader* pe_image_reader,
    const std::wstring& name)
    : name_(name),
      process_reader_(process_reader),
      pe_image_reader_(pe_image_reader) {
}

std::map<std::string, std::string> PEImageAnnotationsReader::SimpleMap() const {
  std::map<std::string, std::string> simple_map_annotations;
  if (process_reader_->Is64Bit()) {
    ReadCrashpadSimpleAnnotations<process_types::internal::Traits64>(
        &simple_map_annotations);
  } else {
    ReadCrashpadSimpleAnnotations<process_types::internal::Traits32>(
        &simple_map_annotations);
  }
  return simple_map_annotations;
}

std::vector<AnnotationSnapshot> PEImageAnnotationsReader::AnnotationsList()
    const {
  std::vector<AnnotationSnapshot> annotations;
  if (process_reader_->Is64Bit()) {
    ReadCrashpadAnnotationsList<process_types::internal::Traits64>(
        &annotations);
  } else {
    ReadCrashpadAnnotationsList<process_types::internal::Traits32>(
        &annotations);
  }
  return annotations;
}

template <class Traits>
void PEImageAnnotationsReader::ReadCrashpadSimpleAnnotations(
    std::map<std::string, std::string>* simple_map_annotations) const {
  process_types::CrashpadInfo<Traits> crashpad_info;
  if (!pe_image_reader_->GetCrashpadInfo(&crashpad_info) ||
      !crashpad_info.simple_annotations) {
    return;
  }

  std::vector<SimpleStringDictionary::Entry>
      simple_annotations(SimpleStringDictionary::num_entries);
  if (!process_reader_->ReadMemory(
          crashpad_info.simple_annotations,
          simple_annotations.size() * sizeof(simple_annotations[0]),
          &simple_annotations[0])) {
    LOG(WARNING) << "could not read simple annotations from "
                 << base::UTF16ToUTF8(name_);
    return;
  }

  for (const auto& entry : simple_annotations) {
    size_t key_length = strnlen(entry.key, sizeof(entry.key));
    if (key_length) {
      std::string key(entry.key, key_length);
      std::string value(entry.value, strnlen(entry.value, sizeof(entry.value)));
      if (!simple_map_annotations->insert(std::make_pair(key, value)).second) {
        LOG(INFO) << "duplicate simple annotation " << key << " in "
                  << base::UTF16ToUTF8(name_);
      }
    }
  }
}

// TODO(rsesek): When there is a platform-agnostic remote memory reader
// interface available, use it so that the implementation is not duplicated
// in the MachOImageAnnotationsReader.
template <class Traits>
void PEImageAnnotationsReader::ReadCrashpadAnnotationsList(
    std::vector<AnnotationSnapshot>* vector_annotations) const {
  process_types::CrashpadInfo<Traits> crashpad_info;
  if (!pe_image_reader_->GetCrashpadInfo(&crashpad_info) ||
      !crashpad_info.annotations_list) {
    return;
  }

  process_types::AnnotationList<Traits> annotation_list_object;
  if (!process_reader_->ReadMemory(crashpad_info.annotations_list,
                                   sizeof(annotation_list_object),
                                   &annotation_list_object)) {
    LOG(WARNING) << "could not read annotations list object in "
                 << base::UTF16ToUTF8(name_);
    return;
  }

  process_types::Annotation<Traits> current = annotation_list_object.head;
  for (size_t index = 0;
       current.link_node != annotation_list_object.tail_pointer &&
       index < kMaxNumberOfAnnotations;
       ++index) {
    if (!process_reader_->ReadMemory(
            current.link_node, sizeof(current), &current)) {
      LOG(WARNING) << "could not read annotation at index " << index << " in "
                   << base::UTF16ToUTF8(name_);
      return;
    }

    if (current.size == 0) {
      continue;
    }

    AnnotationSnapshot snapshot;
    snapshot.type = current.type;

    char name[Annotation::kNameMaxLength];
    if (!process_reader_->ReadMemory(current.name, arraysize(name), name)) {
      LOG(WARNING) << "could not read annotation name at index " << index
                   << " in " << base::UTF16ToUTF8(name_);
      continue;
    }

    size_t name_length = strnlen(name, Annotation::kNameMaxLength);
    snapshot.name = std::string(name, name_length);

    size_t value_length =
        std::min(static_cast<size_t>(current.size), Annotation::kValueMaxSize);
    snapshot.value.resize(value_length);
    if (!process_reader_->ReadMemory(
            current.value, value_length, snapshot.value.data())) {
      LOG(WARNING) << "could not read annotation value at index " << index
                   << " in " << base::UTF16ToUTF8(name_);
      continue;
    }

    vector_annotations->push_back(std::move(snapshot));
  }
}

}  // namespace crashpad
