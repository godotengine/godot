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

#include "minidump/minidump_module_writer.h"

#include <stddef.h>

#include <limits>
#include <utility>

#include "base/logging.h"
#include "base/numerics/safe_conversions.h"
#include "minidump/minidump_string_writer.h"
#include "minidump/minidump_writer_util.h"
#include "snapshot/module_snapshot.h"
#include "util/file/file_writer.h"
#include "util/misc/implicit_cast.h"
#include "util/numeric/in_range_cast.h"
#include "util/numeric/safe_assignment.h"

namespace crashpad {

MinidumpModuleCodeViewRecordWriter::~MinidumpModuleCodeViewRecordWriter() {
}

namespace internal {

template <typename CodeViewRecordType>
MinidumpModuleCodeViewRecordPDBLinkWriter<
    CodeViewRecordType>::MinidumpModuleCodeViewRecordPDBLinkWriter()
    : MinidumpModuleCodeViewRecordWriter(), codeview_record_(), pdb_name_() {
  codeview_record_.signature = CodeViewRecordType::kSignature;
}

template <typename CodeViewRecordType>
MinidumpModuleCodeViewRecordPDBLinkWriter<
    CodeViewRecordType>::~MinidumpModuleCodeViewRecordPDBLinkWriter() {
}

template <typename CodeViewRecordType>
size_t
MinidumpModuleCodeViewRecordPDBLinkWriter<CodeViewRecordType>::SizeOfObject() {
  DCHECK_GE(state(), kStateFrozen);

  // NUL-terminate.
  return offsetof(decltype(codeview_record_), pdb_name) +
         (pdb_name_.size() + 1) * sizeof(pdb_name_[0]);
}

template <typename CodeViewRecordType>
bool MinidumpModuleCodeViewRecordPDBLinkWriter<CodeViewRecordType>::WriteObject(
    FileWriterInterface* file_writer) {
  DCHECK_EQ(state(), kStateWritable);

  WritableIoVec iov;
  iov.iov_base = &codeview_record_;
  iov.iov_len = offsetof(decltype(codeview_record_), pdb_name);
  std::vector<WritableIoVec> iovecs(1, iov);

  // NUL-terminate.
  iov.iov_base = &pdb_name_[0];
  iov.iov_len = (pdb_name_.size() + 1) * sizeof(pdb_name_[0]);
  iovecs.push_back(iov);

  return file_writer->WriteIoVec(&iovecs);
}

}  // namespace internal

template class internal::MinidumpModuleCodeViewRecordPDBLinkWriter<
    CodeViewRecordPDB20>;

MinidumpModuleCodeViewRecordPDB20Writer::
    ~MinidumpModuleCodeViewRecordPDB20Writer() {
}

void MinidumpModuleCodeViewRecordPDB20Writer::SetTimestampAndAge(
    time_t timestamp,
    uint32_t age) {
  DCHECK_EQ(state(), kStateMutable);

  internal::MinidumpWriterUtil::AssignTimeT(&codeview_record()->timestamp,
                                            timestamp);

  codeview_record()->age = age;
}

template class internal::MinidumpModuleCodeViewRecordPDBLinkWriter<
    CodeViewRecordPDB70>;

MinidumpModuleCodeViewRecordPDB70Writer::
    ~MinidumpModuleCodeViewRecordPDB70Writer() {
}

void MinidumpModuleCodeViewRecordPDB70Writer::InitializeFromSnapshot(
    const ModuleSnapshot* module_snapshot) {
  DCHECK_EQ(state(), kStateMutable);

  SetPDBName(module_snapshot->DebugFileName());

  UUID uuid;
  uint32_t age;
  module_snapshot->UUIDAndAge(&uuid, &age);
  SetUUIDAndAge(uuid, age);
}

MinidumpModuleMiscDebugRecordWriter::MinidumpModuleMiscDebugRecordWriter()
    : internal::MinidumpWritable(),
      image_debug_misc_(),
      data_(),
      data_utf16_() {
}

MinidumpModuleMiscDebugRecordWriter::~MinidumpModuleMiscDebugRecordWriter() {
}

void MinidumpModuleMiscDebugRecordWriter::SetData(const std::string& data,
                                                  bool utf16) {
  DCHECK_EQ(state(), kStateMutable);

  if (!utf16) {
    data_utf16_.clear();
    image_debug_misc_.Unicode = 0;
    data_ = data;
  } else {
    data_.clear();
    image_debug_misc_.Unicode = 1;
    data_utf16_ = internal::MinidumpWriterUtil::ConvertUTF8ToUTF16(data);
  }
}

bool MinidumpModuleMiscDebugRecordWriter::Freeze() {
  DCHECK_EQ(state(), kStateMutable);

  if (!MinidumpWritable::Freeze()) {
    return false;
  }

  // NUL-terminate.
  if (!image_debug_misc_.Unicode) {
    DCHECK(data_utf16_.empty());
    image_debug_misc_.Length = base::checked_cast<uint32_t>(
        offsetof(decltype(image_debug_misc_), Data) +
        (data_.size() + 1) * sizeof(data_[0]));
  } else {
    DCHECK(data_.empty());
    image_debug_misc_.Length = base::checked_cast<uint32_t>(
        offsetof(decltype(image_debug_misc_), Data) +
        (data_utf16_.size() + 1) * sizeof(data_utf16_[0]));
  }

  return true;
}

size_t MinidumpModuleMiscDebugRecordWriter::SizeOfObject() {
  DCHECK_GE(state(), kStateFrozen);

  return image_debug_misc_.Length;
}

bool MinidumpModuleMiscDebugRecordWriter::WriteObject(
    FileWriterInterface* file_writer) {
  DCHECK_EQ(state(), kStateWritable);

  const size_t base_length = offsetof(decltype(image_debug_misc_), Data);

  WritableIoVec iov;
  iov.iov_base = &image_debug_misc_;
  iov.iov_len = base_length;
  std::vector<WritableIoVec> iovecs(1, iov);

  if (!image_debug_misc_.Unicode) {
    DCHECK(data_utf16_.empty());
    iov.iov_base = &data_[0];
  } else {
    DCHECK(data_.empty());
    iov.iov_base = &data_utf16_[0];
  }
  iov.iov_len = image_debug_misc_.Length - base_length;
  iovecs.push_back(iov);

  return file_writer->WriteIoVec(&iovecs);
}

MinidumpModuleWriter::MinidumpModuleWriter()
    : MinidumpWritable(),
      module_(),
      name_(),
      codeview_record_(nullptr),
      misc_debug_record_(nullptr) {
  module_.VersionInfo.dwSignature = VS_FFI_SIGNATURE;
  module_.VersionInfo.dwStrucVersion = VS_FFI_STRUCVERSION;
}

MinidumpModuleWriter::~MinidumpModuleWriter() {
}

void MinidumpModuleWriter::InitializeFromSnapshot(
    const ModuleSnapshot* module_snapshot) {
  DCHECK_EQ(state(), kStateMutable);
  DCHECK(!name_);
  DCHECK(!codeview_record_);
  DCHECK(!misc_debug_record_);

  SetName(module_snapshot->Name());

  SetImageBaseAddress(module_snapshot->Address());
  SetImageSize(InRangeCast<uint32_t>(module_snapshot->Size(),
                                     std::numeric_limits<uint32_t>::max()));
  SetTimestamp(module_snapshot->Timestamp());

  uint16_t v[4];
  module_snapshot->FileVersion(&v[0], &v[1], &v[2], &v[3]);
  SetFileVersion(v[0], v[1], v[2], v[3]);

  module_snapshot->SourceVersion(&v[0], &v[1], &v[2], &v[3]);
  SetProductVersion(v[0], v[1], v[2], v[3]);

  uint32_t file_type;
  switch (module_snapshot->GetModuleType()) {
    case ModuleSnapshot::kModuleTypeExecutable:
      file_type = VFT_APP;
      break;
    case ModuleSnapshot::kModuleTypeSharedLibrary:
    case ModuleSnapshot::kModuleTypeLoadableModule:
      file_type = VFT_DLL;
      break;
    default:
      file_type = VFT_UNKNOWN;
      break;
  }
  SetFileTypeAndSubtype(file_type, VFT2_UNKNOWN);

  auto codeview_record =
      std::make_unique<MinidumpModuleCodeViewRecordPDB70Writer>();
  codeview_record->InitializeFromSnapshot(module_snapshot);
  SetCodeViewRecord(std::move(codeview_record));
}

const MINIDUMP_MODULE* MinidumpModuleWriter::MinidumpModule() const {
  DCHECK_EQ(state(), kStateWritable);

  return &module_;
}

void MinidumpModuleWriter::SetName(const std::string& name) {
  DCHECK_EQ(state(), kStateMutable);

  if (!name_) {
    name_.reset(new internal::MinidumpUTF16StringWriter());
  }
  name_->SetUTF8(name);
}

void MinidumpModuleWriter::SetCodeViewRecord(
    std::unique_ptr<MinidumpModuleCodeViewRecordWriter> codeview_record) {
  DCHECK_EQ(state(), kStateMutable);

  codeview_record_ = std::move(codeview_record);
}

void MinidumpModuleWriter::SetMiscDebugRecord(
    std::unique_ptr<MinidumpModuleMiscDebugRecordWriter> misc_debug_record) {
  DCHECK_EQ(state(), kStateMutable);

  misc_debug_record_ = std::move(misc_debug_record);
}

void MinidumpModuleWriter::SetTimestamp(time_t timestamp) {
  DCHECK_EQ(state(), kStateMutable);

  internal::MinidumpWriterUtil::AssignTimeT(&module_.TimeDateStamp, timestamp);
}

void MinidumpModuleWriter::SetFileVersion(uint16_t version_0,
                                          uint16_t version_1,
                                          uint16_t version_2,
                                          uint16_t version_3) {
  DCHECK_EQ(state(), kStateMutable);

  module_.VersionInfo.dwFileVersionMS =
      (implicit_cast<uint32_t>(version_0) << 16) | version_1;
  module_.VersionInfo.dwFileVersionLS =
      (implicit_cast<uint32_t>(version_2) << 16) | version_3;
}

void MinidumpModuleWriter::SetProductVersion(uint16_t version_0,
                                             uint16_t version_1,
                                             uint16_t version_2,
                                             uint16_t version_3) {
  DCHECK_EQ(state(), kStateMutable);

  module_.VersionInfo.dwProductVersionMS =
      (implicit_cast<uint32_t>(version_0) << 16) | version_1;
  module_.VersionInfo.dwProductVersionLS =
      (implicit_cast<uint32_t>(version_2) << 16) | version_3;
}

void MinidumpModuleWriter::SetFileFlagsAndMask(uint32_t file_flags,
                                               uint32_t file_flags_mask) {
  DCHECK_EQ(state(), kStateMutable);
  DCHECK_EQ(file_flags & ~file_flags_mask, 0u);

  module_.VersionInfo.dwFileFlags = file_flags;
  module_.VersionInfo.dwFileFlagsMask = file_flags_mask;
}

bool MinidumpModuleWriter::Freeze() {
  DCHECK_EQ(state(), kStateMutable);
  CHECK(name_);

  if (!MinidumpWritable::Freeze()) {
    return false;
  }

  name_->RegisterRVA(&module_.ModuleNameRva);

  if (codeview_record_) {
    codeview_record_->RegisterLocationDescriptor(&module_.CvRecord);
  }

  if (misc_debug_record_) {
    misc_debug_record_->RegisterLocationDescriptor(&module_.MiscRecord);
  }

  return true;
}

size_t MinidumpModuleWriter::SizeOfObject() {
  DCHECK_GE(state(), kStateFrozen);

  // This object doesn’t directly write anything itself. Its MINIDUMP_MODULE is
  // written by its parent as part of a MINIDUMP_MODULE_LIST, and its children
  // are responsible for writing themselves.
  return 0;
}

std::vector<internal::MinidumpWritable*> MinidumpModuleWriter::Children() {
  DCHECK_GE(state(), kStateFrozen);
  DCHECK(name_);

  std::vector<MinidumpWritable*> children;
  children.push_back(name_.get());
  if (codeview_record_) {
    children.push_back(codeview_record_.get());
  }
  if (misc_debug_record_) {
    children.push_back(misc_debug_record_.get());
  }

  return children;
}

bool MinidumpModuleWriter::WriteObject(FileWriterInterface* file_writer) {
  DCHECK_EQ(state(), kStateWritable);

  // This object doesn’t directly write anything itself. Its MINIDUMP_MODULE is
  // written by its parent as part of a MINIDUMP_MODULE_LIST, and its children
  // are responsible for writing themselves.
  return true;
}

MinidumpModuleListWriter::MinidumpModuleListWriter()
    : MinidumpStreamWriter(), modules_(), module_list_base_() {
}

MinidumpModuleListWriter::~MinidumpModuleListWriter() {
}

void MinidumpModuleListWriter::InitializeFromSnapshot(
    const std::vector<const ModuleSnapshot*>& module_snapshots) {
  DCHECK_EQ(state(), kStateMutable);
  DCHECK(modules_.empty());

  for (const ModuleSnapshot* module_snapshot : module_snapshots) {
    auto module = std::make_unique<MinidumpModuleWriter>();
    module->InitializeFromSnapshot(module_snapshot);
    AddModule(std::move(module));
  }
}

void MinidumpModuleListWriter::AddModule(
    std::unique_ptr<MinidumpModuleWriter> module) {
  DCHECK_EQ(state(), kStateMutable);

  modules_.push_back(std::move(module));
}

bool MinidumpModuleListWriter::Freeze() {
  DCHECK_EQ(state(), kStateMutable);

  if (!MinidumpStreamWriter::Freeze()) {
    return false;
  }

  size_t module_count = modules_.size();
  if (!AssignIfInRange(&module_list_base_.NumberOfModules, module_count)) {
    LOG(ERROR) << "module_count " << module_count << " out of range";
    return false;
  }

  return true;
}

size_t MinidumpModuleListWriter::SizeOfObject() {
  DCHECK_GE(state(), kStateFrozen);

  return sizeof(module_list_base_) + modules_.size() * sizeof(MINIDUMP_MODULE);
}

std::vector<internal::MinidumpWritable*> MinidumpModuleListWriter::Children() {
  DCHECK_GE(state(), kStateFrozen);

  std::vector<MinidumpWritable*> children;
  for (const auto& module : modules_) {
    children.push_back(module.get());
  }

  return children;
}

bool MinidumpModuleListWriter::WriteObject(FileWriterInterface* file_writer) {
  DCHECK_EQ(state(), kStateWritable);

  WritableIoVec iov;
  iov.iov_base = &module_list_base_;
  iov.iov_len = sizeof(module_list_base_);
  std::vector<WritableIoVec> iovecs(1, iov);

  for (const auto& module : modules_) {
    iov.iov_base = module->MinidumpModule();
    iov.iov_len = sizeof(MINIDUMP_MODULE);
    iovecs.push_back(iov);
  }

  return file_writer->WriteIoVec(&iovecs);
}

MinidumpStreamType MinidumpModuleListWriter::StreamType() const {
  return kMinidumpStreamTypeModuleList;
}

}  // namespace crashpad
