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

#include "minidump/minidump_crashpad_info_writer.h"

#include <utility>

#include "base/logging.h"
#include "minidump/minidump_module_crashpad_info_writer.h"
#include "minidump/minidump_simple_string_dictionary_writer.h"
#include "snapshot/process_snapshot.h"
#include "util/file/file_writer.h"

namespace crashpad {

MinidumpCrashpadInfoWriter::MinidumpCrashpadInfoWriter()
    : MinidumpStreamWriter(),
      crashpad_info_(),
      simple_annotations_(nullptr),
      module_list_(nullptr) {
  crashpad_info_.version = MinidumpCrashpadInfo::kVersion;
}

MinidumpCrashpadInfoWriter::~MinidumpCrashpadInfoWriter() {
}

void MinidumpCrashpadInfoWriter::InitializeFromSnapshot(
    const ProcessSnapshot* process_snapshot) {
  DCHECK_EQ(state(), kStateMutable);
  DCHECK(!module_list_);

  UUID report_id;
  process_snapshot->ReportID(&report_id);
  SetReportID(report_id);

  UUID client_id;
  process_snapshot->ClientID(&client_id);
  SetClientID(client_id);

  auto simple_annotations =
      std::make_unique<MinidumpSimpleStringDictionaryWriter>();
  simple_annotations->InitializeFromMap(
      process_snapshot->AnnotationsSimpleMap());
  if (simple_annotations->IsUseful()) {
    SetSimpleAnnotations(std::move(simple_annotations));
  }

  auto modules = std::make_unique<MinidumpModuleCrashpadInfoListWriter>();
  modules->InitializeFromSnapshot(process_snapshot->Modules());

  if (modules->IsUseful()) {
    SetModuleList(std::move(modules));
  }
}

void MinidumpCrashpadInfoWriter::SetReportID(const UUID& report_id) {
  DCHECK_EQ(state(), kStateMutable);

  crashpad_info_.report_id = report_id;
}

void MinidumpCrashpadInfoWriter::SetClientID(const UUID& client_id) {
  DCHECK_EQ(state(), kStateMutable);

  crashpad_info_.client_id = client_id;
}

void MinidumpCrashpadInfoWriter::SetSimpleAnnotations(
    std::unique_ptr<MinidumpSimpleStringDictionaryWriter> simple_annotations) {
  DCHECK_EQ(state(), kStateMutable);

  simple_annotations_ = std::move(simple_annotations);
}

void MinidumpCrashpadInfoWriter::SetModuleList(
    std::unique_ptr<MinidumpModuleCrashpadInfoListWriter> module_list) {
  DCHECK_EQ(state(), kStateMutable);

  module_list_ = std::move(module_list);
}

bool MinidumpCrashpadInfoWriter::Freeze() {
  DCHECK_EQ(state(), kStateMutable);

  if (!MinidumpStreamWriter::Freeze()) {
    return false;
  }

  if (simple_annotations_) {
    simple_annotations_->RegisterLocationDescriptor(
        &crashpad_info_.simple_annotations);
  }
  if (module_list_) {
    module_list_->RegisterLocationDescriptor(&crashpad_info_.module_list);
  }

  return true;
}

size_t MinidumpCrashpadInfoWriter::SizeOfObject() {
  DCHECK_GE(state(), kStateFrozen);

  return sizeof(crashpad_info_);
}

std::vector<internal::MinidumpWritable*>
MinidumpCrashpadInfoWriter::Children() {
  DCHECK_GE(state(), kStateFrozen);

  std::vector<MinidumpWritable*> children;
  if (simple_annotations_) {
    children.push_back(simple_annotations_.get());
  }
  if (module_list_) {
    children.push_back(module_list_.get());
  }

  return children;
}

bool MinidumpCrashpadInfoWriter::WriteObject(FileWriterInterface* file_writer) {
  DCHECK_EQ(state(), kStateWritable);

  return file_writer->Write(&crashpad_info_, sizeof(crashpad_info_));
}

MinidumpStreamType MinidumpCrashpadInfoWriter::StreamType() const {
  return kMinidumpStreamTypeCrashpadInfo;
}

bool MinidumpCrashpadInfoWriter::IsUseful() const {
  return crashpad_info_.report_id != UUID() ||
         crashpad_info_.client_id != UUID() ||
         simple_annotations_ ||
         module_list_;
}

}  // namespace crashpad
