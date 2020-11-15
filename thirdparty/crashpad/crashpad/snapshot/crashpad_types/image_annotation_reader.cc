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

#include "snapshot/crashpad_types/image_annotation_reader.h"

#include <string.h>
#include <sys/types.h>

#include <algorithm>
#include <utility>

#include "base/logging.h"
#include "build/build_config.h"
#include "client/annotation.h"
#include "client/annotation_list.h"
#include "client/simple_string_dictionary.h"
#include "snapshot/snapshot_constants.h"
#include "util/linux/traits.h"

namespace crashpad {

namespace process_types {

template <class Traits>
struct Annotation {
  typename Traits::Address link_node;
  typename Traits::Address name;
  typename Traits::Address value;
  uint32_t size;
  uint16_t type;
};

template <class Traits>
struct AnnotationList {
  typename Traits::Address tail_pointer;
  Annotation<Traits> head;
  Annotation<Traits> tail;
};

}  // namespace process_types

#if defined(ARCH_CPU_64_BITS)
#define NATIVE_TRAITS Traits64
#else
#define NATIVE_TRAITS Traits32
#endif  // ARCH_CPU_64_BITS

static_assert(sizeof(process_types::Annotation<NATIVE_TRAITS>) ==
                  sizeof(Annotation),
              "Annotation size mismatch");

static_assert(sizeof(process_types::AnnotationList<NATIVE_TRAITS>) ==
                  sizeof(AnnotationList),
              "AnnotationList size mismatch");

#undef NATIVE_TRAITS

ImageAnnotationReader::ImageAnnotationReader(const ProcessMemoryRange* memory)
    : memory_(memory) {}

ImageAnnotationReader::~ImageAnnotationReader() = default;

bool ImageAnnotationReader::SimpleMap(
    VMAddress address,
    std::map<std::string, std::string>* annotations) const {
  std::vector<SimpleStringDictionary::Entry> simple_annotations(
      SimpleStringDictionary::num_entries);

  if (!memory_->Read(address,
                     simple_annotations.size() * sizeof(simple_annotations[0]),
                     &simple_annotations[0])) {
    return false;
  }

  for (const auto& entry : simple_annotations) {
    size_t key_length = strnlen(entry.key, sizeof(entry.key));
    if (key_length) {
      std::string key(entry.key, key_length);
      std::string value(entry.value, strnlen(entry.value, sizeof(entry.value)));
      if (!annotations->insert(std::make_pair(key, value)).second) {
        LOG(WARNING) << "duplicate simple annotation " << key << " " << value;
      }
    }
  }
  return true;
}

bool ImageAnnotationReader::AnnotationsList(
    VMAddress address,
    std::vector<AnnotationSnapshot>* annotations) const {
  return memory_->Is64Bit()
             ? ReadAnnotationList<Traits64>(address, annotations)
             : ReadAnnotationList<Traits32>(address, annotations);
}

template <class Traits>
bool ImageAnnotationReader::ReadAnnotationList(
    VMAddress address,
    std::vector<AnnotationSnapshot>* annotations) const {
  process_types::AnnotationList<Traits> annotation_list;
  if (!memory_->Read(address, sizeof(annotation_list), &annotation_list)) {
    LOG(ERROR) << "could not read annotation list";
    return false;
  }

  process_types::Annotation<Traits> current = annotation_list.head;
  for (size_t index = 0; current.link_node != annotation_list.tail_pointer &&
                         index < kMaxNumberOfAnnotations;
       ++index) {
    if (!memory_->Read(current.link_node, sizeof(current), &current)) {
      LOG(ERROR) << "could not read annotation at index " << index;
      return false;
    }

    if (current.size == 0) {
      continue;
    }

    AnnotationSnapshot snapshot;
    snapshot.type = current.type;

    if (!memory_->ReadCStringSizeLimited(
            current.name, Annotation::kNameMaxLength, &snapshot.name)) {
      LOG(WARNING) << "could not read annotation name at index " << index;
      continue;
    }

    size_t value_length =
        std::min(static_cast<size_t>(current.size), Annotation::kValueMaxSize);
    snapshot.value.resize(value_length);
    if (!memory_->Read(current.value, value_length, snapshot.value.data())) {
      LOG(WARNING) << "could not read annotation value at index " << index;
      continue;
    }

    annotations->push_back(std::move(snapshot));
  }

  return true;
}

}  // namespace crashpad
