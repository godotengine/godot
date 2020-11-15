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

#include "minidump/minidump_annotation_writer.h"

#include <memory>

#include "base/logging.h"
#include "util/file/file_writer.h"
#include "util/numeric/safe_assignment.h"

namespace crashpad {

MinidumpAnnotationWriter::MinidumpAnnotationWriter() = default;

MinidumpAnnotationWriter::~MinidumpAnnotationWriter() = default;

void MinidumpAnnotationWriter::InitializeFromSnapshot(
    const AnnotationSnapshot& snapshot) {
  DCHECK_EQ(state(), kStateMutable);

  name_.SetUTF8(snapshot.name);
  annotation_.type = snapshot.type;
  annotation_.reserved = 0;
  value_.set_data(snapshot.value);
}

void MinidumpAnnotationWriter::InitializeWithData(
    const std::string& name,
    uint16_t type,
    const std::vector<uint8_t>& data) {
  DCHECK_EQ(state(), kStateMutable);

  name_.SetUTF8(name);
  annotation_.type = type;
  annotation_.reserved = 0;
  value_.set_data(data);
}

bool MinidumpAnnotationWriter::Freeze() {
  DCHECK_EQ(state(), kStateMutable);

  if (!MinidumpWritable::Freeze()) {
    return false;
  }

  name_.RegisterRVA(&annotation_.name);
  value_.RegisterRVA(&annotation_.value);

  return true;
}

size_t MinidumpAnnotationWriter::SizeOfObject() {
  DCHECK_GE(state(), kStateFrozen);

  // This object is written by the MinidumpAnnotationListWriter, and its
  // children write themselves.
  return 0;
}

std::vector<internal::MinidumpWritable*> MinidumpAnnotationWriter::Children() {
  DCHECK_GE(state(), kStateFrozen);

  return {&name_, &value_};
}

bool MinidumpAnnotationWriter::WriteObject(FileWriterInterface* file_writer) {
  DCHECK_EQ(state(), kStateWritable);

  // This object is written by the MinidumpAnnotationListWriter, and its
  // children write themselves.
  return true;
}

MinidumpAnnotationListWriter::MinidumpAnnotationListWriter()
    : minidump_list_(new MinidumpAnnotationList()) {}

MinidumpAnnotationListWriter::~MinidumpAnnotationListWriter() = default;

void MinidumpAnnotationListWriter::InitializeFromList(
    const std::vector<AnnotationSnapshot>& list) {
  DCHECK_EQ(state(), kStateMutable);
  for (const auto& annotation : list) {
    auto writer = std::make_unique<MinidumpAnnotationWriter>();
    writer->InitializeFromSnapshot(annotation);
    AddObject(std::move(writer));
  }
}

void MinidumpAnnotationListWriter::AddObject(
    std::unique_ptr<MinidumpAnnotationWriter> annotation_writer) {
  DCHECK_EQ(state(), kStateMutable);

  objects_.push_back(std::move(annotation_writer));
}

bool MinidumpAnnotationListWriter::IsUseful() const {
  return !objects_.empty();
}

bool MinidumpAnnotationListWriter::Freeze() {
  DCHECK_EQ(state(), kStateMutable);

  if (!MinidumpWritable::Freeze()) {
    return false;
  }

  if (!AssignIfInRange(&minidump_list_->count, objects_.size())) {
    LOG(ERROR) << "annotation list size " << objects_.size()
               << " is out of range";
    return false;
  }

  return true;
}

size_t MinidumpAnnotationListWriter::SizeOfObject() {
  DCHECK_GE(state(), kStateFrozen);

  return sizeof(*minidump_list_) + sizeof(MinidumpAnnotation) * objects_.size();
}

std::vector<internal::MinidumpWritable*>
MinidumpAnnotationListWriter::Children() {
  DCHECK_GE(state(), kStateFrozen);

  std::vector<internal::MinidumpWritable*> children(objects_.size());
  for (size_t i = 0; i < objects_.size(); ++i) {
    children[i] = objects_[i].get();
  }

  return children;
}

bool MinidumpAnnotationListWriter::WriteObject(
    FileWriterInterface* file_writer) {
  DCHECK_EQ(state(), kStateWritable);

  std::vector<WritableIoVec> iov(1 + objects_.size());
  iov[0].iov_base = minidump_list_.get();
  iov[0].iov_len = sizeof(*minidump_list_);

  for (const auto& object : objects_) {
    iov.emplace_back(WritableIoVec{object->minidump_annotation(),
                                   sizeof(MinidumpAnnotation)});
  }

  return file_writer->WriteIoVec(&iov);
}

}  // namespace crashpad
