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

#include "client/annotation_list.h"

#include "base/logging.h"
#include "client/crashpad_info.h"

namespace crashpad {

AnnotationList::AnnotationList()
    : tail_pointer_(&tail_),
      head_(Annotation::Type::kInvalid, nullptr, nullptr),
      tail_(Annotation::Type::kInvalid, nullptr, nullptr) {
  head_.link_node().store(&tail_);
}

AnnotationList::~AnnotationList() {}

// static
AnnotationList* AnnotationList::Get() {
  return CrashpadInfo::GetCrashpadInfo()->annotations_list();
}

// static
AnnotationList* AnnotationList::Register() {
  AnnotationList* list = Get();
  if (!list) {
    list = new AnnotationList();
    CrashpadInfo::GetCrashpadInfo()->set_annotations_list(list);
  }
  return list;
}

void AnnotationList::Add(Annotation* annotation) {
  Annotation* null = nullptr;
  Annotation* head_next = head_.link_node().load(std::memory_order_relaxed);
  if (!annotation->link_node().compare_exchange_strong(null, head_next)) {
    // If |annotation|'s link node is not null, then it has been added to the
    // list already and no work needs to be done.
    return;
  }

  // Check that the annotation's name is less than the maximum size. This is
  // done here, since the Annotation constructor must be constexpr and this
  // path is taken once per annotation.
  DCHECK_LT(strlen(annotation->name_), Annotation::kNameMaxLength);

  // Update the head link to point to the new |annotation|.
  while (!head_.link_node().compare_exchange_weak(head_next, annotation)) {
    // Another thread has updated the head-next pointer, so try again with the
    // re-loaded |head_next|.
    annotation->link_node().store(head_next, std::memory_order_relaxed);
  }
}

AnnotationList::Iterator::Iterator(Annotation* head, const Annotation* tail)
    : curr_(head), tail_(tail) {}

AnnotationList::Iterator::~Iterator() = default;

Annotation* AnnotationList::Iterator::operator*() const {
  CHECK_NE(curr_, tail_);
  return curr_;
}

AnnotationList::Iterator& AnnotationList::Iterator::operator++() {
  CHECK_NE(curr_, tail_);
  curr_ = curr_->link_node();
  return *this;
}

bool AnnotationList::Iterator::operator==(
    const AnnotationList::Iterator& other) const {
  return curr_ == other.curr_;
}

AnnotationList::Iterator AnnotationList::begin() {
  return Iterator(head_.link_node(), tail_pointer_);
}

AnnotationList::Iterator AnnotationList::end() {
  return Iterator(&tail_, tail_pointer_);
}

}  // namespace crashpad
