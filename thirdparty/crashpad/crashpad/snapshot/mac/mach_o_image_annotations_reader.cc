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

#include "snapshot/mac/mach_o_image_annotations_reader.h"

#include <mach-o/loader.h>
#include <mach/mach.h>
#include <sys/types.h>

#include <utility>

#include "base/logging.h"
#include "client/crashpad_info.h"
#include "client/simple_string_dictionary.h"
#include "snapshot/mac/mach_o_image_reader.h"
#include "snapshot/mac/process_reader_mac.h"
#include "snapshot/snapshot_constants.h"
#include "util/mach/task_memory.h"
#include "util/stdlib/strnlen.h"

namespace crashpad {

MachOImageAnnotationsReader::MachOImageAnnotationsReader(
    ProcessReaderMac* process_reader,
    const MachOImageReader* image_reader,
    const std::string& name)
    : name_(name),
      process_reader_(process_reader),
      image_reader_(image_reader) {}

std::vector<std::string> MachOImageAnnotationsReader::Vector() const {
  std::vector<std::string> vector_annotations;

  ReadCrashReporterClientAnnotations(&vector_annotations);
  ReadDyldErrorStringAnnotation(&vector_annotations);

  return vector_annotations;
}

std::map<std::string, std::string> MachOImageAnnotationsReader::SimpleMap()
    const {
  std::map<std::string, std::string> simple_map_annotations;

  ReadCrashpadSimpleAnnotations(&simple_map_annotations);

  return simple_map_annotations;
}

std::vector<AnnotationSnapshot> MachOImageAnnotationsReader::AnnotationsList()
    const {
  std::vector<AnnotationSnapshot> annotations;

  ReadCrashpadAnnotationsList(&annotations);

  return annotations;
}

void MachOImageAnnotationsReader::ReadCrashReporterClientAnnotations(
    std::vector<std::string>* vector_annotations) const {
  mach_vm_address_t crash_info_address;
  const process_types::section* crash_info_section =
      image_reader_->GetSectionByName(
          SEG_DATA, "__crash_info", &crash_info_address);
  if (!crash_info_section) {
    return;
  }

  process_types::crashreporter_annotations_t crash_info;
  if (!crash_info.Read(process_reader_, crash_info_address)) {
    LOG(WARNING) << "could not read crash info from " << name_;
    return;
  }

  if (crash_info.version != 4 && crash_info.version != 5) {
    LOG(WARNING) << "unexpected crash info version " << crash_info.version
                 << " in " << name_;
    return;
  }

  size_t expected_size =
      process_types::crashreporter_annotations_t::ExpectedSizeForVersion(
          process_reader_, crash_info.version);
  if (crash_info_section->size < expected_size) {
    LOG(WARNING) << "small crash info section size " << crash_info_section->size
                 << " < " << expected_size << " for version "
                 << crash_info.version << " in " << name_;
    return;
  }

  // This number was totally made up out of nowhere, but it seems prudent to
  // enforce some limit.
  constexpr size_t kMaxMessageSize = 1024;
  if (crash_info.message) {
    std::string message;
    if (process_reader_->Memory()->ReadCStringSizeLimited(
            crash_info.message, kMaxMessageSize, &message)) {
      vector_annotations->push_back(message);
    } else {
      LOG(WARNING) << "could not read crash message in " << name_;
    }
  }

  if (crash_info.message2) {
    std::string message;
    if (process_reader_->Memory()->ReadCStringSizeLimited(
            crash_info.message2, kMaxMessageSize, &message)) {
      vector_annotations->push_back(message);
    } else {
      LOG(WARNING) << "could not read crash message 2 in " << name_;
    }
  }
}

void MachOImageAnnotationsReader::ReadDyldErrorStringAnnotation(
    std::vector<std::string>* vector_annotations) const {
  // dyld stores its error string at the external symbol for |const char
  // error_string[1024]|. See 10.9.5 dyld-239.4/src/dyld.cpp error_string.
  if (image_reader_->FileType() != MH_DYLINKER) {
    return;
  }

  mach_vm_address_t error_string_address;
  if (!image_reader_->LookUpExternalDefinedSymbol("_error_string",
                                                  &error_string_address)) {
    return;
  }

  std::string message;
  // 1024 here is distinct from kMaxMessageSize above, because it refers to a
  // precisely-sized buffer inside dyld.
  if (process_reader_->Memory()->ReadCStringSizeLimited(
          error_string_address, 1024, &message)) {
    if (!message.empty()) {
      vector_annotations->push_back(message);
    }
  } else {
    LOG(WARNING) << "could not read dylinker error string from " << name_;
  }
}

void MachOImageAnnotationsReader::ReadCrashpadSimpleAnnotations(
    std::map<std::string, std::string>* simple_map_annotations) const {
  process_types::CrashpadInfo crashpad_info;
  if (!image_reader_->GetCrashpadInfo(&crashpad_info) ||
      !crashpad_info.simple_annotations) {
    return;
  }

  std::vector<SimpleStringDictionary::Entry>
      simple_annotations(SimpleStringDictionary::num_entries);
  if (!process_reader_->Memory()->Read(
          crashpad_info.simple_annotations,
          simple_annotations.size() * sizeof(simple_annotations[0]),
          &simple_annotations[0])) {
    LOG(WARNING) << "could not read simple annotations from " << name_;
    return;
  }

  for (const auto& entry : simple_annotations) {
    size_t key_length = strnlen(entry.key, sizeof(entry.key));
    if (key_length) {
      std::string key(entry.key, key_length);
      std::string value(entry.value, strnlen(entry.value, sizeof(entry.value)));
      if (!simple_map_annotations->insert(std::make_pair(key, value)).second) {
        LOG(INFO) << "duplicate simple annotation " << key << " in " << name_;
      }
    }
  }
}

// TODO(rsesek): When there is a platform-agnostic remote memory reader
// interface available, use it so that the implementation is not duplicated
// in the PEImageAnnotationsReader.
void MachOImageAnnotationsReader::ReadCrashpadAnnotationsList(
    std::vector<AnnotationSnapshot>* annotations) const {
  process_types::CrashpadInfo crashpad_info;
  if (!image_reader_->GetCrashpadInfo(&crashpad_info) ||
      !crashpad_info.annotations_list) {
    return;
  }

  process_types::AnnotationList annotation_list_object;
  if (!annotation_list_object.Read(process_reader_,
                                   crashpad_info.annotations_list)) {
    LOG(WARNING) << "could not read annotations list object in " << name_;
    return;
  }

  process_types::Annotation current = annotation_list_object.head;
  for (size_t index = 0;
       current.link_node != annotation_list_object.tail_pointer &&
       index < kMaxNumberOfAnnotations;
       ++index) {
    if (!current.Read(process_reader_, current.link_node)) {
      LOG(WARNING) << "could not read annotation at index " << index << " in "
                   << name_;
      return;
    }

    if (current.size == 0) {
      continue;
    }

    AnnotationSnapshot snapshot;
    snapshot.type = current.type;
    snapshot.value.resize(current.size);

    if (!process_reader_->Memory()->ReadCStringSizeLimited(
            current.name, Annotation::kNameMaxLength, &snapshot.name)) {
      LOG(WARNING) << "could not read annotation name at index " << index
                   << " in " << name_;
      continue;
    }

    size_t size =
        std::min(static_cast<size_t>(current.size), Annotation::kValueMaxSize);
    if (!process_reader_->Memory()->Read(
            current.value, size, snapshot.value.data())) {
      LOG(WARNING) << "could not read annotation value at index " << index
                   << " in " << name_;
      continue;
    }

    annotations->push_back(std::move(snapshot));
  }
}

}  // namespace crashpad
