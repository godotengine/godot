// Copyright 2016 The Crashpad Authors. All rights reserved.
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

#include "minidump/minidump_unloaded_module_writer.h"

#include <limits>
#include <utility>

#include "minidump/minidump_writer_util.h"
#include "util/file/file_writer.h"
#include "util/numeric/in_range_cast.h"
#include "util/numeric/safe_assignment.h"

namespace crashpad {

MinidumpUnloadedModuleWriter::MinidumpUnloadedModuleWriter()
    : MinidumpWritable(), unloaded_module_(), name_() {}

MinidumpUnloadedModuleWriter::~MinidumpUnloadedModuleWriter() {
}

void MinidumpUnloadedModuleWriter::InitializeFromSnapshot(
    const UnloadedModuleSnapshot& unloaded_module_snapshot) {
  DCHECK_EQ(state(), kStateMutable);
  DCHECK(!name_);

  SetName(unloaded_module_snapshot.Name());

  SetImageBaseAddress(unloaded_module_snapshot.Address());
  SetImageSize(InRangeCast<uint32_t>(unloaded_module_snapshot.Size(),
                                     std::numeric_limits<uint32_t>::max()));
  SetTimestamp(unloaded_module_snapshot.Timestamp());
  SetChecksum(unloaded_module_snapshot.Checksum());
}

const MINIDUMP_UNLOADED_MODULE*
MinidumpUnloadedModuleWriter::MinidumpUnloadedModule() const {
  DCHECK_EQ(state(), kStateWritable);

  return &unloaded_module_;
}

void MinidumpUnloadedModuleWriter::SetName(const std::string& name) {
  DCHECK_EQ(state(), kStateMutable);

  if (!name_) {
    name_.reset(new internal::MinidumpUTF16StringWriter());
  }
  name_->SetUTF8(name);
}

void MinidumpUnloadedModuleWriter::SetTimestamp(time_t timestamp) {
  DCHECK_EQ(state(), kStateMutable);

  internal::MinidumpWriterUtil::AssignTimeT(&unloaded_module_.TimeDateStamp,
                                            timestamp);
}

bool MinidumpUnloadedModuleWriter::Freeze() {
  DCHECK_EQ(state(), kStateMutable);
  CHECK(name_);

  if (!MinidumpWritable::Freeze()) {
    return false;
  }

  name_->RegisterRVA(&unloaded_module_.ModuleNameRva);

  return true;
}

size_t MinidumpUnloadedModuleWriter::SizeOfObject() {
  DCHECK_GE(state(), kStateFrozen);

  // This object doesn’t directly write anything itself. Its
  // MINIDUMP_UNLOADED_MODULE is written by its parent as part of a
  // MINIDUMP_UNLOADED_MODULE_LIST, and its children are responsible for writing
  // themselves.
  return 0;
}

std::vector<internal::MinidumpWritable*>
MinidumpUnloadedModuleWriter::Children() {
  DCHECK_GE(state(), kStateFrozen);
  DCHECK(name_);

  std::vector<MinidumpWritable*> children(1, name_.get());
  return children;
}

bool MinidumpUnloadedModuleWriter::WriteObject(
    FileWriterInterface* file_writer) {
  DCHECK_EQ(state(), kStateWritable);

  // This object doesn’t directly write anything itself. Its
  // MINIDUMP_UNLOADED_MODULE is written by its parent as part of a
  // MINIDUMP_UNLOADED_MODULE_LIST, and its children are responsible for writing
  // themselves.
  return true;
}

MinidumpUnloadedModuleListWriter::MinidumpUnloadedModuleListWriter()
    : MinidumpStreamWriter(),
      unloaded_modules_(),
      unloaded_module_list_base_() {}

MinidumpUnloadedModuleListWriter::~MinidumpUnloadedModuleListWriter() {
}

void MinidumpUnloadedModuleListWriter::InitializeFromSnapshot(
    const std::vector<UnloadedModuleSnapshot>& unloaded_module_snapshots) {
  DCHECK_EQ(state(), kStateMutable);
  DCHECK(unloaded_modules_.empty());

  for (auto unloaded_module_snapshot : unloaded_module_snapshots) {
    auto unloaded_module = std::make_unique<MinidumpUnloadedModuleWriter>();
    unloaded_module->InitializeFromSnapshot(unloaded_module_snapshot);
    AddUnloadedModule(std::move(unloaded_module));
  }
}

void MinidumpUnloadedModuleListWriter::AddUnloadedModule(
    std::unique_ptr<MinidumpUnloadedModuleWriter> unloaded_module) {
  DCHECK_EQ(state(), kStateMutable);

  unloaded_modules_.push_back(std::move(unloaded_module));
}

bool MinidumpUnloadedModuleListWriter::Freeze() {
  DCHECK_EQ(state(), kStateMutable);

  if (!MinidumpStreamWriter::Freeze()) {
    return false;
  }

  unloaded_module_list_base_.SizeOfHeader =
      sizeof(MINIDUMP_UNLOADED_MODULE_LIST);
  unloaded_module_list_base_.SizeOfEntry = sizeof(MINIDUMP_UNLOADED_MODULE);

  size_t unloaded_module_count = unloaded_modules_.size();
  if (!AssignIfInRange(&unloaded_module_list_base_.NumberOfEntries,
                       unloaded_module_count)) {
    LOG(ERROR) << "unloaded_module_count " << unloaded_module_count
               << " out of range";
    return false;
  }

  return true;
}

size_t MinidumpUnloadedModuleListWriter::SizeOfObject() {
  DCHECK_GE(state(), kStateFrozen);

  return sizeof(unloaded_module_list_base_) +
         unloaded_modules_.size() * sizeof(MINIDUMP_UNLOADED_MODULE);
}

std::vector<internal::MinidumpWritable*>
MinidumpUnloadedModuleListWriter::Children() {
  DCHECK_GE(state(), kStateFrozen);

  std::vector<MinidumpWritable*> children;
  for (const auto& unloaded_module : unloaded_modules_) {
    children.push_back(unloaded_module.get());
  }

  return children;
}

bool MinidumpUnloadedModuleListWriter::WriteObject(
    FileWriterInterface* file_writer) {
  DCHECK_EQ(state(), kStateWritable);

  WritableIoVec iov;
  iov.iov_base = &unloaded_module_list_base_;
  iov.iov_len = sizeof(unloaded_module_list_base_);
  std::vector<WritableIoVec> iovecs(1, iov);

  for (const auto& unloaded_module : unloaded_modules_) {
    iov.iov_base = unloaded_module->MinidumpUnloadedModule();
    iov.iov_len = sizeof(MINIDUMP_UNLOADED_MODULE);
    iovecs.push_back(iov);
  }

  return file_writer->WriteIoVec(&iovecs);
}

MinidumpStreamType MinidumpUnloadedModuleListWriter::StreamType() const {
  return kMinidumpStreamTypeUnloadedModuleList;
}

}  // namespace crashpad
