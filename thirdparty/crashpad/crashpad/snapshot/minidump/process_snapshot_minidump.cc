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

#include "snapshot/minidump/process_snapshot_minidump.h"

#include <utility>

#include "snapshot/minidump/minidump_simple_string_dictionary_reader.h"
#include "util/file/file_io.h"

namespace crashpad {

ProcessSnapshotMinidump::ProcessSnapshotMinidump()
    : ProcessSnapshot(),
      header_(),
      stream_directory_(),
      stream_map_(),
      modules_(),
      unloaded_modules_(),
      crashpad_info_(),
      annotations_simple_map_(),
      file_reader_(nullptr),
      process_id_(static_cast<pid_t>(-1)),
      initialized_() {
}

ProcessSnapshotMinidump::~ProcessSnapshotMinidump() {
}

bool ProcessSnapshotMinidump::Initialize(FileReaderInterface* file_reader) {
  INITIALIZATION_STATE_SET_INITIALIZING(initialized_);

  file_reader_ = file_reader;

  if (!file_reader_->SeekSet(0)) {
    return false;
  }

  if (!file_reader_->ReadExactly(&header_, sizeof(header_))) {
    return false;
  }

  if (header_.Signature != MINIDUMP_SIGNATURE) {
    LOG(ERROR) << "minidump signature mismatch";
    return false;
  }

  if (header_.Version != MINIDUMP_VERSION) {
    LOG(ERROR) << "minidump version mismatch";
    return false;
  }

  if (!file_reader->SeekSet(header_.StreamDirectoryRva)) {
    return false;
  }

  stream_directory_.resize(header_.NumberOfStreams);
  if (!stream_directory_.empty() &&
      !file_reader_->ReadExactly(
          &stream_directory_[0],
          header_.NumberOfStreams * sizeof(stream_directory_[0]))) {
    return false;
  }

  for (const MINIDUMP_DIRECTORY& directory : stream_directory_) {
    const MinidumpStreamType stream_type =
        static_cast<MinidumpStreamType>(directory.StreamType);
    if (stream_map_.find(stream_type) != stream_map_.end()) {
      LOG(ERROR) << "duplicate streams for type " << directory.StreamType;
      return false;
    }

    stream_map_[stream_type] = &directory.Location;
  }

  if (!InitializeCrashpadInfo() ||
      !InitializeMiscInfo() ||
      !InitializeModules()) {
    return false;
  }

  INITIALIZATION_STATE_SET_VALID(initialized_);
  return true;
}

pid_t ProcessSnapshotMinidump::ProcessID() const {
  INITIALIZATION_STATE_DCHECK_VALID(initialized_);
  return process_id_;
}

pid_t ProcessSnapshotMinidump::ParentProcessID() const {
  INITIALIZATION_STATE_DCHECK_VALID(initialized_);
  NOTREACHED();  // https://crashpad.chromium.org/bug/10
  return 0;
}

void ProcessSnapshotMinidump::SnapshotTime(timeval* snapshot_time) const {
  INITIALIZATION_STATE_DCHECK_VALID(initialized_);
  NOTREACHED();  // https://crashpad.chromium.org/bug/10
  snapshot_time->tv_sec = 0;
  snapshot_time->tv_usec = 0;
}

void ProcessSnapshotMinidump::ProcessStartTime(timeval* start_time) const {
  INITIALIZATION_STATE_DCHECK_VALID(initialized_);
  NOTREACHED();  // https://crashpad.chromium.org/bug/10
  start_time->tv_sec = 0;
  start_time->tv_usec = 0;
}

void ProcessSnapshotMinidump::ProcessCPUTimes(timeval* user_time,
                                              timeval* system_time) const {
  INITIALIZATION_STATE_DCHECK_VALID(initialized_);
  NOTREACHED();  // https://crashpad.chromium.org/bug/10
  user_time->tv_sec = 0;
  user_time->tv_usec = 0;
  system_time->tv_sec = 0;
  system_time->tv_usec = 0;
}

void ProcessSnapshotMinidump::ReportID(UUID* report_id) const {
  INITIALIZATION_STATE_DCHECK_VALID(initialized_);
  *report_id = crashpad_info_.report_id;
}

void ProcessSnapshotMinidump::ClientID(UUID* client_id) const {
  INITIALIZATION_STATE_DCHECK_VALID(initialized_);
  *client_id = crashpad_info_.client_id;
}

const std::map<std::string, std::string>&
ProcessSnapshotMinidump::AnnotationsSimpleMap() const {
  // TODO(mark): This method should not be const, although the interface
  // currently imposes this requirement. Making it non-const would allow
  // annotations_simple_map_ to be lazily constructed: InitializeCrashpadInfo()
  // could be called here, and from other locations that require it, rather than
  // calling it from Initialize().
  // https://crashpad.chromium.org/bug/9
  INITIALIZATION_STATE_DCHECK_VALID(initialized_);
  return annotations_simple_map_;
}

const SystemSnapshot* ProcessSnapshotMinidump::System() const {
  INITIALIZATION_STATE_DCHECK_VALID(initialized_);
  NOTREACHED();  // https://crashpad.chromium.org/bug/10
  return nullptr;
}

std::vector<const ThreadSnapshot*> ProcessSnapshotMinidump::Threads() const {
  INITIALIZATION_STATE_DCHECK_VALID(initialized_);
  NOTREACHED();  // https://crashpad.chromium.org/bug/10
  return std::vector<const ThreadSnapshot*>();
}

std::vector<const ModuleSnapshot*> ProcessSnapshotMinidump::Modules() const {
  INITIALIZATION_STATE_DCHECK_VALID(initialized_);
  std::vector<const ModuleSnapshot*> modules;
  for (const auto& module : modules_) {
    modules.push_back(module.get());
  }
  return modules;
}

std::vector<UnloadedModuleSnapshot> ProcessSnapshotMinidump::UnloadedModules()
    const {
  INITIALIZATION_STATE_DCHECK_VALID(initialized_);
  NOTREACHED();  // https://crashpad.chromium.org/bug/10
  return unloaded_modules_;
}

const ExceptionSnapshot* ProcessSnapshotMinidump::Exception() const {
  INITIALIZATION_STATE_DCHECK_VALID(initialized_);
  NOTREACHED();  // https://crashpad.chromium.org/bug/10
  return nullptr;
}

std::vector<const MemoryMapRegionSnapshot*> ProcessSnapshotMinidump::MemoryMap()
    const {
  INITIALIZATION_STATE_DCHECK_VALID(initialized_);
  NOTREACHED();  // https://crashpad.chromium.org/bug/10
  return std::vector<const MemoryMapRegionSnapshot*>();
}

std::vector<HandleSnapshot> ProcessSnapshotMinidump::Handles() const {
  INITIALIZATION_STATE_DCHECK_VALID(initialized_);
  NOTREACHED();  // https://crashpad.chromium.org/bug/10
  return std::vector<HandleSnapshot>();
}

std::vector<const MemorySnapshot*> ProcessSnapshotMinidump::ExtraMemory()
    const {
  INITIALIZATION_STATE_DCHECK_VALID(initialized_);
  NOTREACHED();  // https://crashpad.chromium.org/bug/10
  return std::vector<const MemorySnapshot*>();
}

bool ProcessSnapshotMinidump::InitializeCrashpadInfo() {
  const auto& stream_it = stream_map_.find(kMinidumpStreamTypeCrashpadInfo);
  if (stream_it == stream_map_.end()) {
    return true;
  }

  if (stream_it->second->DataSize < sizeof(crashpad_info_)) {
    LOG(ERROR) << "crashpad_info size mismatch";
    return false;
  }

  if (!file_reader_->SeekSet(stream_it->second->Rva)) {
    return false;
  }

  if (!file_reader_->ReadExactly(&crashpad_info_, sizeof(crashpad_info_))) {
    return false;
  }

  if (crashpad_info_.version != MinidumpCrashpadInfo::kVersion) {
    LOG(ERROR) << "crashpad_info version mismatch";
    return false;
  }

  return internal::ReadMinidumpSimpleStringDictionary(
      file_reader_,
      crashpad_info_.simple_annotations,
      &annotations_simple_map_);
}

bool ProcessSnapshotMinidump::InitializeMiscInfo() {
  const auto& stream_it = stream_map_.find(kMinidumpStreamTypeMiscInfo);
  if (stream_it == stream_map_.end()) {
    return true;
  }

  if (!file_reader_->SeekSet(stream_it->second->Rva)) {
    return false;
  }

  const size_t size = stream_it->second->DataSize;
  if (size != sizeof(MINIDUMP_MISC_INFO_5) &&
      size != sizeof(MINIDUMP_MISC_INFO_4) &&
      size != sizeof(MINIDUMP_MISC_INFO_3) &&
      size != sizeof(MINIDUMP_MISC_INFO_2) &&
      size != sizeof(MINIDUMP_MISC_INFO)) {
    LOG(ERROR) << "misc_info size mismatch";
    return false;
  }

  MINIDUMP_MISC_INFO_5 info;
  if (!file_reader_->ReadExactly(&info, size)) {
    return false;
  }

  switch (stream_it->second->DataSize) {
    case sizeof(MINIDUMP_MISC_INFO_5):
    case sizeof(MINIDUMP_MISC_INFO_4):
    case sizeof(MINIDUMP_MISC_INFO_3):
    case sizeof(MINIDUMP_MISC_INFO_2):
    case sizeof(MINIDUMP_MISC_INFO):
      // TODO(jperaza): Save the remaining misc info.
      // https://crashpad.chromium.org/bug/10
      process_id_ = info.ProcessId;
  }

  return true;
}

bool ProcessSnapshotMinidump::InitializeModules() {
  const auto& stream_it = stream_map_.find(kMinidumpStreamTypeModuleList);
  if (stream_it == stream_map_.end()) {
    return true;
  }

  std::map<uint32_t, MINIDUMP_LOCATION_DESCRIPTOR> module_crashpad_info_links;
  if (!InitializeModulesCrashpadInfo(&module_crashpad_info_links)) {
    return false;
  }

  if (stream_it->second->DataSize < sizeof(MINIDUMP_MODULE_LIST)) {
    LOG(ERROR) << "module_list size mismatch";
    return false;
  }

  if (!file_reader_->SeekSet(stream_it->second->Rva)) {
    return false;
  }

  uint32_t module_count;
  if (!file_reader_->ReadExactly(&module_count, sizeof(module_count))) {
    return false;
  }

  if (sizeof(MINIDUMP_MODULE_LIST) + module_count * sizeof(MINIDUMP_MODULE) !=
          stream_it->second->DataSize) {
    LOG(ERROR) << "module_list size mismatch";
    return false;
  }

  for (uint32_t module_index = 0; module_index < module_count; ++module_index) {
    const RVA module_rva = stream_it->second->Rva + sizeof(module_count) +
                           module_index * sizeof(MINIDUMP_MODULE);

    const auto& module_crashpad_info_it =
        module_crashpad_info_links.find(module_index);
    const MINIDUMP_LOCATION_DESCRIPTOR* module_crashpad_info_location =
        module_crashpad_info_it != module_crashpad_info_links.end()
            ? &module_crashpad_info_it->second
            : nullptr;

    auto module = std::make_unique<internal::ModuleSnapshotMinidump>();
    if (!module->Initialize(
            file_reader_, module_rva, module_crashpad_info_location)) {
      return false;
    }

    modules_.push_back(std::move(module));
  }

  return true;
}

bool ProcessSnapshotMinidump::InitializeModulesCrashpadInfo(
    std::map<uint32_t, MINIDUMP_LOCATION_DESCRIPTOR>*
        module_crashpad_info_links) {
  module_crashpad_info_links->clear();

  if (crashpad_info_.version != MinidumpCrashpadInfo::kVersion) {
    return false;
  }

  if (crashpad_info_.module_list.Rva == 0) {
    return true;
  }

  if (crashpad_info_.module_list.DataSize <
          sizeof(MinidumpModuleCrashpadInfoList)) {
    LOG(ERROR) << "module_crashpad_info_list size mismatch";
    return false;
  }

  if (!file_reader_->SeekSet(crashpad_info_.module_list.Rva)) {
    return false;
  }

  uint32_t crashpad_module_count;
  if (!file_reader_->ReadExactly(&crashpad_module_count,
                                 sizeof(crashpad_module_count))) {
    return false;
  }

  if (crashpad_info_.module_list.DataSize !=
          sizeof(MinidumpModuleCrashpadInfoList) +
              crashpad_module_count * sizeof(MinidumpModuleCrashpadInfoLink)) {
    LOG(ERROR) << "module_crashpad_info_list size mismatch";
    return false;
  }

  std::unique_ptr<MinidumpModuleCrashpadInfoLink[]> minidump_links(
      new MinidumpModuleCrashpadInfoLink[crashpad_module_count]);
  if (!file_reader_->ReadExactly(
          &minidump_links[0],
          crashpad_module_count * sizeof(MinidumpModuleCrashpadInfoLink))) {
    return false;
  }

  for (uint32_t crashpad_module_index = 0;
       crashpad_module_index < crashpad_module_count;
       ++crashpad_module_index) {
    const MinidumpModuleCrashpadInfoLink& minidump_link =
        minidump_links[crashpad_module_index];
    if (!module_crashpad_info_links
             ->insert(std::make_pair(minidump_link.minidump_module_list_index,
                                     minidump_link.location)).second) {
      LOG(WARNING)
          << "duplicate module_crashpad_info_list minidump_module_list_index "
          << minidump_link.minidump_module_list_index;
      return false;
    }
  }

  return true;
}

}  // namespace crashpad
