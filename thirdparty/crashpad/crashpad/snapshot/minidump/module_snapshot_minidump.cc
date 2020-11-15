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

#include "snapshot/minidump/module_snapshot_minidump.h"

#include "minidump/minidump_extensions.h"
#include "snapshot/minidump/minidump_annotation_reader.h"
#include "snapshot/minidump/minidump_simple_string_dictionary_reader.h"
#include "snapshot/minidump/minidump_string_list_reader.h"

namespace crashpad {
namespace internal {

ModuleSnapshotMinidump::ModuleSnapshotMinidump()
    : ModuleSnapshot(),
      minidump_module_(),
      annotations_vector_(),
      annotations_simple_map_(),
      initialized_() {
}

ModuleSnapshotMinidump::~ModuleSnapshotMinidump() {
}

bool ModuleSnapshotMinidump::Initialize(
    FileReaderInterface* file_reader,
    RVA minidump_module_rva,
    const MINIDUMP_LOCATION_DESCRIPTOR*
        minidump_module_crashpad_info_location) {
  INITIALIZATION_STATE_SET_INITIALIZING(initialized_);

  if (!file_reader->SeekSet(minidump_module_rva)) {
    return false;
  }

  if (!file_reader->ReadExactly(&minidump_module_, sizeof(minidump_module_))) {
    return false;
  }

  if (!InitializeModuleCrashpadInfo(file_reader,
                                    minidump_module_crashpad_info_location)) {
    return false;
  }

  INITIALIZATION_STATE_SET_VALID(initialized_);
  return true;
}

std::string ModuleSnapshotMinidump::Name() const {
  INITIALIZATION_STATE_DCHECK_VALID(initialized_);
  NOTREACHED();  // https://crashpad.chromium.org/bug/10
  return std::string();
}

uint64_t ModuleSnapshotMinidump::Address() const {
  INITIALIZATION_STATE_DCHECK_VALID(initialized_);
  NOTREACHED();  // https://crashpad.chromium.org/bug/10
  return 0;
}

uint64_t ModuleSnapshotMinidump::Size() const {
  INITIALIZATION_STATE_DCHECK_VALID(initialized_);
  NOTREACHED();  // https://crashpad.chromium.org/bug/10
  return 0;
}

time_t ModuleSnapshotMinidump::Timestamp() const {
  INITIALIZATION_STATE_DCHECK_VALID(initialized_);
  NOTREACHED();  // https://crashpad.chromium.org/bug/10
  return 0;
}

void ModuleSnapshotMinidump::FileVersion(uint16_t* version_0,
                                         uint16_t* version_1,
                                         uint16_t* version_2,
                                         uint16_t* version_3) const {
  INITIALIZATION_STATE_DCHECK_VALID(initialized_);
  NOTREACHED();  // https://crashpad.chromium.org/bug/10
  *version_0 = 0;
  *version_1 = 0;
  *version_2 = 0;
  *version_3 = 0;
}

void ModuleSnapshotMinidump::SourceVersion(uint16_t* version_0,
                                           uint16_t* version_1,
                                           uint16_t* version_2,
                                           uint16_t* version_3) const {
  INITIALIZATION_STATE_DCHECK_VALID(initialized_);
  NOTREACHED();  // https://crashpad.chromium.org/bug/10
  *version_0 = 0;
  *version_1 = 0;
  *version_2 = 0;
  *version_3 = 0;
}

ModuleSnapshot::ModuleType ModuleSnapshotMinidump::GetModuleType() const {
  INITIALIZATION_STATE_DCHECK_VALID(initialized_);
  NOTREACHED();  // https://crashpad.chromium.org/bug/10
  return kModuleTypeUnknown;
}

void ModuleSnapshotMinidump::UUIDAndAge(crashpad::UUID* uuid,
                                        uint32_t* age) const {
  INITIALIZATION_STATE_DCHECK_VALID(initialized_);
  NOTREACHED();  // https://crashpad.chromium.org/bug/10
  *uuid = crashpad::UUID();
  *age = 0;
}

std::string ModuleSnapshotMinidump::DebugFileName() const {
  INITIALIZATION_STATE_DCHECK_VALID(initialized_);
  NOTREACHED();  // https://crashpad.chromium.org/bug/10
  return std::string();
}

std::vector<std::string> ModuleSnapshotMinidump::AnnotationsVector() const {
  INITIALIZATION_STATE_DCHECK_VALID(initialized_);
  return annotations_vector_;
}

std::map<std::string, std::string>
ModuleSnapshotMinidump::AnnotationsSimpleMap() const {
  INITIALIZATION_STATE_DCHECK_VALID(initialized_);
  return annotations_simple_map_;
}

std::vector<AnnotationSnapshot> ModuleSnapshotMinidump::AnnotationObjects()
    const {
  INITIALIZATION_STATE_DCHECK_VALID(initialized_);
  return annotation_objects_;
}

std::set<CheckedRange<uint64_t>> ModuleSnapshotMinidump::ExtraMemoryRanges()
    const {
  INITIALIZATION_STATE_DCHECK_VALID(initialized_);
  NOTREACHED();  // https://crashpad.chromium.org/bug/10
  return std::set<CheckedRange<uint64_t>>();
}

std::vector<const UserMinidumpStream*>
ModuleSnapshotMinidump::CustomMinidumpStreams() const {
  INITIALIZATION_STATE_DCHECK_VALID(initialized_);
  NOTREACHED();  // https://crashpad.chromium.org/bug/10
  return std::vector<const UserMinidumpStream*>();
}

bool ModuleSnapshotMinidump::InitializeModuleCrashpadInfo(
    FileReaderInterface* file_reader,
    const MINIDUMP_LOCATION_DESCRIPTOR*
        minidump_module_crashpad_info_location) {
  if (!minidump_module_crashpad_info_location ||
      minidump_module_crashpad_info_location->Rva == 0) {
    return true;
  }

  MinidumpModuleCrashpadInfo minidump_module_crashpad_info;
  if (minidump_module_crashpad_info_location->DataSize <
          sizeof(minidump_module_crashpad_info)) {
    LOG(ERROR) << "minidump_module_crashpad_info size mismatch";
    return false;
  }

  if (!file_reader->SeekSet(minidump_module_crashpad_info_location->Rva)) {
    return false;
  }

  if (!file_reader->ReadExactly(&minidump_module_crashpad_info,
                                sizeof(minidump_module_crashpad_info))) {
    return false;
  }

  if (minidump_module_crashpad_info.version !=
          MinidumpModuleCrashpadInfo::kVersion) {
    LOG(ERROR) << "minidump_module_crashpad_info version mismatch";
    return false;
  }

  if (!ReadMinidumpStringList(file_reader,
                              minidump_module_crashpad_info.list_annotations,
                              &annotations_vector_)) {
    return false;
  }

  if (!ReadMinidumpSimpleStringDictionary(
          file_reader,
          minidump_module_crashpad_info.simple_annotations,
          &annotations_simple_map_)) {
    return false;
  }

  if (!ReadMinidumpAnnotationList(
          file_reader,
          minidump_module_crashpad_info.annotation_objects,
          &annotation_objects_)) {
    return false;
  }

  return true;
}

}  // namespace internal
}  // namespace crashpad
