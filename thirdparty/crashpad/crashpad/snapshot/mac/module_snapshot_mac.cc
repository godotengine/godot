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

#include "snapshot/mac/module_snapshot_mac.h"

#include <mach/mach.h>
#include <mach-o/loader.h>

#include "base/files/file_path.h"
#include "base/strings/stringprintf.h"
#include "snapshot/mac/mach_o_image_annotations_reader.h"
#include "snapshot/mac/mach_o_image_reader.h"
#include "util/misc/tri_state.h"
#include "util/misc/uuid.h"
#include "util/stdlib/strnlen.h"

namespace crashpad {
namespace internal {

ModuleSnapshotMac::ModuleSnapshotMac()
    : ModuleSnapshot(),
      name_(),
      timestamp_(0),
      mach_o_image_reader_(nullptr),
      process_reader_(nullptr),
      initialized_() {
}

ModuleSnapshotMac::~ModuleSnapshotMac() {
}

bool ModuleSnapshotMac::Initialize(
    ProcessReaderMac* process_reader,
    const ProcessReaderMac::Module& process_reader_module) {
  INITIALIZATION_STATE_SET_INITIALIZING(initialized_);

  process_reader_ = process_reader;
  name_ = process_reader_module.name;
  timestamp_ = process_reader_module.timestamp;
  mach_o_image_reader_ = process_reader_module.reader;
  if (!mach_o_image_reader_) {
    return false;
  }

  INITIALIZATION_STATE_SET_VALID(initialized_);
  return true;
}

void ModuleSnapshotMac::GetCrashpadOptions(CrashpadInfoClientOptions* options) {
  INITIALIZATION_STATE_DCHECK_VALID(initialized_);

  process_types::CrashpadInfo crashpad_info;
  if (!mach_o_image_reader_->GetCrashpadInfo(&crashpad_info)) {
    options->crashpad_handler_behavior = TriState::kUnset;
    options->system_crash_reporter_forwarding = TriState::kUnset;
    options->gather_indirectly_referenced_memory = TriState::kUnset;
    return;
  }

  options->crashpad_handler_behavior =
      CrashpadInfoClientOptions::TriStateFromCrashpadInfo(
          crashpad_info.crashpad_handler_behavior);

  options->system_crash_reporter_forwarding =
      CrashpadInfoClientOptions::TriStateFromCrashpadInfo(
          crashpad_info.system_crash_reporter_forwarding);

  options->gather_indirectly_referenced_memory =
      CrashpadInfoClientOptions::TriStateFromCrashpadInfo(
          crashpad_info.gather_indirectly_referenced_memory);

  options->indirectly_referenced_memory_cap =
      crashpad_info.indirectly_referenced_memory_cap;
}

std::string ModuleSnapshotMac::Name() const {
  INITIALIZATION_STATE_DCHECK_VALID(initialized_);
  return name_;
}

uint64_t ModuleSnapshotMac::Address() const {
  INITIALIZATION_STATE_DCHECK_VALID(initialized_);
  return mach_o_image_reader_->Address();
}

uint64_t ModuleSnapshotMac::Size() const {
  INITIALIZATION_STATE_DCHECK_VALID(initialized_);
  return mach_o_image_reader_->Size();
}

time_t ModuleSnapshotMac::Timestamp() const {
  INITIALIZATION_STATE_DCHECK_VALID(initialized_);
  return timestamp_;
}

void ModuleSnapshotMac::FileVersion(uint16_t* version_0,
                                    uint16_t* version_1,
                                    uint16_t* version_2,
                                    uint16_t* version_3) const {
  INITIALIZATION_STATE_DCHECK_VALID(initialized_);
  if (mach_o_image_reader_->FileType() == MH_DYLIB) {
    uint32_t dylib_version = mach_o_image_reader_->DylibVersion();
    *version_0 = (dylib_version & 0xffff0000) >> 16;
    *version_1 = (dylib_version & 0x0000ff00) >> 8;
    *version_2 = (dylib_version & 0x000000ff);
    *version_3 = 0;
  } else {
    *version_0 = 0;
    *version_1 = 0;
    *version_2 = 0;
    *version_3 = 0;
  }
}

void ModuleSnapshotMac::SourceVersion(uint16_t* version_0,
                                      uint16_t* version_1,
                                      uint16_t* version_2,
                                      uint16_t* version_3) const {
  INITIALIZATION_STATE_DCHECK_VALID(initialized_);

  // LC_SOURCE_VERSION is supposed to be interpreted as a 5-component version
  // number, 24 bits for the first component and 10 for the others, per
  // <mach-o/loader.h>. To preserve the full range of possible version numbers
  // without data loss, map it to the 4 16-bit fields mandated by the interface
  // here, which was informed by the minidump file format.
  uint64_t source_version = mach_o_image_reader_->SourceVersion();
  *version_0 = (source_version & 0xffff000000000000u) >> 48;
  *version_1 = (source_version & 0x0000ffff00000000u) >> 32;
  *version_2 = (source_version & 0x00000000ffff0000u) >> 16;
  *version_3 = source_version & 0x000000000000ffffu;
}

ModuleSnapshot::ModuleType ModuleSnapshotMac::GetModuleType() const {
  INITIALIZATION_STATE_DCHECK_VALID(initialized_);

  uint32_t file_type = mach_o_image_reader_->FileType();
  switch (file_type) {
    case MH_EXECUTE:
      return kModuleTypeExecutable;
    case MH_DYLIB:
      return kModuleTypeSharedLibrary;
    case MH_DYLINKER:
      return kModuleTypeDynamicLoader;
    case MH_BUNDLE:
      return kModuleTypeLoadableModule;
    default:
      return kModuleTypeUnknown;
  }
}

void ModuleSnapshotMac::UUIDAndAge(crashpad::UUID* uuid, uint32_t* age) const {
  INITIALIZATION_STATE_DCHECK_VALID(initialized_);
  mach_o_image_reader_->UUID(uuid);
  *age = 0;
}

std::string ModuleSnapshotMac::DebugFileName() const {
  INITIALIZATION_STATE_DCHECK_VALID(initialized_);
  return base::FilePath(Name()).BaseName().value();
}

std::vector<std::string> ModuleSnapshotMac::AnnotationsVector() const {
  INITIALIZATION_STATE_DCHECK_VALID(initialized_);
  MachOImageAnnotationsReader annotations_reader(
      process_reader_, mach_o_image_reader_, name_);
  return annotations_reader.Vector();
}

std::map<std::string, std::string> ModuleSnapshotMac::AnnotationsSimpleMap()
    const {
  INITIALIZATION_STATE_DCHECK_VALID(initialized_);
  MachOImageAnnotationsReader annotations_reader(
      process_reader_, mach_o_image_reader_, name_);
  return annotations_reader.SimpleMap();
}

std::vector<AnnotationSnapshot> ModuleSnapshotMac::AnnotationObjects() const {
  INITIALIZATION_STATE_DCHECK_VALID(initialized_);
  MachOImageAnnotationsReader annotations_reader(
      process_reader_, mach_o_image_reader_, name_);
  return annotations_reader.AnnotationsList();
}

std::set<CheckedRange<uint64_t>> ModuleSnapshotMac::ExtraMemoryRanges() const {
  INITIALIZATION_STATE_DCHECK_VALID(initialized_);
  return std::set<CheckedRange<uint64_t>>();
}

std::vector<const UserMinidumpStream*>
ModuleSnapshotMac::CustomMinidumpStreams() const {
  return std::vector<const UserMinidumpStream*>();
}

}  // namespace internal
}  // namespace crashpad
