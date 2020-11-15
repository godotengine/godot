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

#include "minidump/minidump_module_crashpad_info_writer.h"

#include <utility>

#include "base/logging.h"
#include "minidump/minidump_annotation_writer.h"
#include "minidump/minidump_simple_string_dictionary_writer.h"
#include "snapshot/module_snapshot.h"
#include "util/file/file_writer.h"
#include "util/numeric/safe_assignment.h"

namespace crashpad {

MinidumpModuleCrashpadInfoWriter::MinidumpModuleCrashpadInfoWriter()
    : MinidumpWritable(),
      module_(),
      list_annotations_(),
      simple_annotations_(),
      annotation_objects_() {
  module_.version = MinidumpModuleCrashpadInfo::kVersion;
}

MinidumpModuleCrashpadInfoWriter::~MinidumpModuleCrashpadInfoWriter() {
}

void MinidumpModuleCrashpadInfoWriter::InitializeFromSnapshot(
    const ModuleSnapshot* module_snapshot) {
  DCHECK_EQ(state(), kStateMutable);
  DCHECK(!list_annotations_);
  DCHECK(!simple_annotations_);

  auto list_annotations = std::make_unique<MinidumpUTF8StringListWriter>();
  list_annotations->InitializeFromVector(module_snapshot->AnnotationsVector());
  if (list_annotations->IsUseful()) {
    SetListAnnotations(std::move(list_annotations));
  }

  auto simple_annotations =
      std::make_unique<MinidumpSimpleStringDictionaryWriter>();
  simple_annotations->InitializeFromMap(
      module_snapshot->AnnotationsSimpleMap());
  if (simple_annotations->IsUseful()) {
    SetSimpleAnnotations(std::move(simple_annotations));
  }

  auto annotation_objects = std::make_unique<MinidumpAnnotationListWriter>();
  annotation_objects->InitializeFromList(module_snapshot->AnnotationObjects());
  if (annotation_objects->IsUseful()) {
    SetAnnotationObjects(std::move(annotation_objects));
  }
}

void MinidumpModuleCrashpadInfoWriter::SetListAnnotations(
    std::unique_ptr<MinidumpUTF8StringListWriter> list_annotations) {
  DCHECK_EQ(state(), kStateMutable);

  list_annotations_ = std::move(list_annotations);
}

void MinidumpModuleCrashpadInfoWriter::SetSimpleAnnotations(
    std::unique_ptr<MinidumpSimpleStringDictionaryWriter> simple_annotations) {
  DCHECK_EQ(state(), kStateMutable);

  simple_annotations_ = std::move(simple_annotations);
}

void MinidumpModuleCrashpadInfoWriter::SetAnnotationObjects(
    std::unique_ptr<MinidumpAnnotationListWriter> annotation_objects) {
  DCHECK_EQ(state(), kStateMutable);

  annotation_objects_ = std::move(annotation_objects);
}

bool MinidumpModuleCrashpadInfoWriter::IsUseful() const {
  return list_annotations_ || simple_annotations_ || annotation_objects_;
}

bool MinidumpModuleCrashpadInfoWriter::Freeze() {
  DCHECK_EQ(state(), kStateMutable);

  if (!MinidumpWritable::Freeze()) {
    return false;
  }

  if (list_annotations_) {
    list_annotations_->RegisterLocationDescriptor(&module_.list_annotations);
  }

  if (simple_annotations_) {
    simple_annotations_->RegisterLocationDescriptor(
        &module_.simple_annotations);
  }

  if (annotation_objects_) {
    annotation_objects_->RegisterLocationDescriptor(
        &module_.annotation_objects);
  }

  return true;
}

size_t MinidumpModuleCrashpadInfoWriter::SizeOfObject() {
  DCHECK_GE(state(), kStateFrozen);

  return sizeof(module_);
}

std::vector<internal::MinidumpWritable*>
MinidumpModuleCrashpadInfoWriter::Children() {
  DCHECK_GE(state(), kStateFrozen);

  std::vector<MinidumpWritable*> children;
  if (list_annotations_) {
    children.push_back(list_annotations_.get());
  }
  if (simple_annotations_) {
    children.push_back(simple_annotations_.get());
  }
  if (annotation_objects_) {
    children.push_back(annotation_objects_.get());
  }

  return children;
}

bool MinidumpModuleCrashpadInfoWriter::WriteObject(
    FileWriterInterface* file_writer) {
  DCHECK_EQ(state(), kStateWritable);

  return file_writer->Write(&module_, sizeof(module_));
}

MinidumpModuleCrashpadInfoListWriter::MinidumpModuleCrashpadInfoListWriter()
    : MinidumpWritable(),
      module_crashpad_infos_(),
      module_crashpad_info_links_(),
      module_crashpad_info_list_base_() {
}

MinidumpModuleCrashpadInfoListWriter::~MinidumpModuleCrashpadInfoListWriter() {
}

void MinidumpModuleCrashpadInfoListWriter::InitializeFromSnapshot(
    const std::vector<const ModuleSnapshot*>& module_snapshots) {
  DCHECK_EQ(state(), kStateMutable);
  DCHECK(module_crashpad_infos_.empty());
  DCHECK(module_crashpad_info_links_.empty());

  size_t count = module_snapshots.size();
  for (size_t index = 0; index < count; ++index) {
    const ModuleSnapshot* module_snapshot = module_snapshots[index];

    auto module = std::make_unique<MinidumpModuleCrashpadInfoWriter>();
    module->InitializeFromSnapshot(module_snapshot);
    if (module->IsUseful()) {
      AddModule(std::move(module), index);
    }
  }
}

void MinidumpModuleCrashpadInfoListWriter::AddModule(
    std::unique_ptr<MinidumpModuleCrashpadInfoWriter> module_crashpad_info,
    size_t minidump_module_list_index) {
  DCHECK_EQ(state(), kStateMutable);
  DCHECK_EQ(module_crashpad_infos_.size(), module_crashpad_info_links_.size());

  MinidumpModuleCrashpadInfoLink module_crashpad_info_link = {};
  if (!AssignIfInRange(&module_crashpad_info_link.minidump_module_list_index,
                       minidump_module_list_index)) {
    LOG(ERROR) << "minidump_module_list_index " << minidump_module_list_index
               << " out of range";
    return;
  }

  module_crashpad_info_links_.push_back(module_crashpad_info_link);
  module_crashpad_infos_.push_back(std::move(module_crashpad_info));
}

bool MinidumpModuleCrashpadInfoListWriter::IsUseful() const {
  DCHECK_EQ(module_crashpad_infos_.size(), module_crashpad_info_links_.size());
  return !module_crashpad_infos_.empty();
}

bool MinidumpModuleCrashpadInfoListWriter::Freeze() {
  DCHECK_EQ(state(), kStateMutable);
  CHECK_EQ(module_crashpad_infos_.size(), module_crashpad_info_links_.size());

  if (!MinidumpWritable::Freeze()) {
    return false;
  }

  size_t module_count = module_crashpad_infos_.size();
  if (!AssignIfInRange(&module_crashpad_info_list_base_.count, module_count)) {
    LOG(ERROR) << "module_count " << module_count << " out of range";
    return false;
  }

  for (size_t index = 0; index < module_count; ++index) {
    module_crashpad_infos_[index]->RegisterLocationDescriptor(
        &module_crashpad_info_links_[index].location);
  }

  return true;
}

size_t MinidumpModuleCrashpadInfoListWriter::SizeOfObject() {
  DCHECK_GE(state(), kStateFrozen);
  DCHECK_EQ(module_crashpad_infos_.size(), module_crashpad_info_links_.size());

  return sizeof(module_crashpad_info_list_base_) +
         module_crashpad_info_links_.size() *
             sizeof(module_crashpad_info_links_[0]);
}

std::vector<internal::MinidumpWritable*>
MinidumpModuleCrashpadInfoListWriter::Children() {
  DCHECK_GE(state(), kStateFrozen);
  DCHECK_EQ(module_crashpad_infos_.size(), module_crashpad_info_links_.size());

  std::vector<MinidumpWritable*> children;
  for (const auto& module : module_crashpad_infos_) {
    children.push_back(module.get());
  }

  return children;
}

bool MinidumpModuleCrashpadInfoListWriter::WriteObject(
    FileWriterInterface* file_writer) {
  DCHECK_EQ(state(), kStateWritable);
  DCHECK_EQ(module_crashpad_infos_.size(), module_crashpad_info_links_.size());

  WritableIoVec iov;
  iov.iov_base = &module_crashpad_info_list_base_;
  iov.iov_len = sizeof(module_crashpad_info_list_base_);
  std::vector<WritableIoVec> iovecs(1, iov);

  if (!module_crashpad_info_links_.empty()) {
    iov.iov_base = &module_crashpad_info_links_[0];
    iov.iov_len = module_crashpad_info_links_.size() *
                  sizeof(module_crashpad_info_links_[0]);
    iovecs.push_back(iov);
  }

  return file_writer->WriteIoVec(&iovecs);
}

}  // namespace crashpad
