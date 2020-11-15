// Copyright 2018 The Crashpad Authors. All rights reserved.
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

#include "snapshot/elf/module_snapshot_elf.h"

#include <endian.h>

#include <algorithm>

#include "base/files/file_path.h"
#include "snapshot/crashpad_types/image_annotation_reader.h"
#include "util/misc/elf_note_types.h"

namespace crashpad {
namespace internal {

ModuleSnapshotElf::ModuleSnapshotElf(const std::string& name,
                                     ElfImageReader* elf_reader,
                                     ModuleSnapshot::ModuleType type,
                                     ProcessMemoryRange* process_memory_range)
    : ModuleSnapshot(),
      name_(name),
      elf_reader_(elf_reader),
      process_memory_range_(process_memory_range),
      crashpad_info_(),
      type_(type),
      initialized_() {}

ModuleSnapshotElf::~ModuleSnapshotElf() = default;

bool ModuleSnapshotElf::Initialize() {
  INITIALIZATION_STATE_SET_INITIALIZING(initialized_);

  if (!elf_reader_) {
    LOG(ERROR) << "no elf reader";
    return false;
  }

  // The data payload is only sizeof(VMAddress) in the note, but add a bit to
  // account for the name, header, and padding.
  constexpr ssize_t kMaxNoteSize = 256;
  std::unique_ptr<ElfImageReader::NoteReader> notes =
      elf_reader_->NotesWithNameAndType(CRASHPAD_ELF_NOTE_NAME,
                                        CRASHPAD_ELF_NOTE_TYPE_CRASHPAD_INFO,
                                        kMaxNoteSize);
  std::string desc;
  VMAddress info_address;
  if (notes->NextNote(nullptr, nullptr, &desc) ==
      ElfImageReader::NoteReader::Result::kSuccess) {
    info_address = *reinterpret_cast<VMAddress*>(&desc[0]);

    ProcessMemoryRange range;
    if (range.Initialize(*elf_reader_->Memory())) {
      auto info = std::make_unique<CrashpadInfoReader>();
      if (info->Initialize(&range, info_address)) {
        crashpad_info_ = std::move(info);
      }
    }
  }

  INITIALIZATION_STATE_SET_VALID(initialized_);
  return true;
}

bool ModuleSnapshotElf::GetCrashpadOptions(CrashpadInfoClientOptions* options) {
  INITIALIZATION_STATE_DCHECK_VALID(initialized_);

  if (!crashpad_info_) {
    return false;
  }

  options->crashpad_handler_behavior =
      crashpad_info_->CrashpadHandlerBehavior();
  options->system_crash_reporter_forwarding =
      crashpad_info_->SystemCrashReporterForwarding();
  options->gather_indirectly_referenced_memory =
      crashpad_info_->GatherIndirectlyReferencedMemory();
  options->indirectly_referenced_memory_cap =
      crashpad_info_->IndirectlyReferencedMemoryCap();
  return true;
}

std::string ModuleSnapshotElf::Name() const {
  INITIALIZATION_STATE_DCHECK_VALID(initialized_);
  return name_;
}

uint64_t ModuleSnapshotElf::Address() const {
  INITIALIZATION_STATE_DCHECK_VALID(initialized_);
  return elf_reader_->Address();
}

uint64_t ModuleSnapshotElf::Size() const {
  INITIALIZATION_STATE_DCHECK_VALID(initialized_);
  return elf_reader_->Size();
}

time_t ModuleSnapshotElf::Timestamp() const {
  INITIALIZATION_STATE_DCHECK_VALID(initialized_);
  return 0;
}

void ModuleSnapshotElf::FileVersion(uint16_t* version_0,
                                    uint16_t* version_1,
                                    uint16_t* version_2,
                                    uint16_t* version_3) const {
  INITIALIZATION_STATE_DCHECK_VALID(initialized_);
  *version_0 = 0;
  *version_1 = 0;
  *version_2 = 0;
  *version_3 = 0;
}

void ModuleSnapshotElf::SourceVersion(uint16_t* version_0,
                                      uint16_t* version_1,
                                      uint16_t* version_2,
                                      uint16_t* version_3) const {
  INITIALIZATION_STATE_DCHECK_VALID(initialized_);
  *version_0 = 0;
  *version_1 = 0;
  *version_2 = 0;
  *version_3 = 0;
}

ModuleSnapshot::ModuleType ModuleSnapshotElf::GetModuleType() const {
  INITIALIZATION_STATE_DCHECK_VALID(initialized_);
  return type_;
}

void ModuleSnapshotElf::UUIDAndAge(crashpad::UUID* uuid, uint32_t* age) const {
  INITIALIZATION_STATE_DCHECK_VALID(initialized_);
  *age = 0;

  std::unique_ptr<ElfImageReader::NoteReader> notes =
      elf_reader_->NotesWithNameAndType(ELF_NOTE_GNU, NT_GNU_BUILD_ID, 64);
  std::string desc;
  notes->NextNote(nullptr, nullptr, &desc);
  desc.insert(desc.end(), 16 - std::min(desc.size(), size_t{16}), '\0');
  uuid->InitializeFromBytes(reinterpret_cast<const uint8_t*>(&desc[0]));

  // TODO(scottmg): https://crashpad.chromium.org/bug/229. These are
  // endian-swapped to match FileID::ConvertIdentifierToUUIDString() in
  // Breakpad. This is necessary as this identifier is used for symbol lookup.
  uuid->data_1 = htobe32(uuid->data_1);
  uuid->data_2 = htobe16(uuid->data_2);
  uuid->data_3 = htobe16(uuid->data_3);
}

std::string ModuleSnapshotElf::DebugFileName() const {
  INITIALIZATION_STATE_DCHECK_VALID(initialized_);
  return base::FilePath(Name()).BaseName().value();
}

std::vector<std::string> ModuleSnapshotElf::AnnotationsVector() const {
  INITIALIZATION_STATE_DCHECK_VALID(initialized_);
  return std::vector<std::string>();
}

std::map<std::string, std::string> ModuleSnapshotElf::AnnotationsSimpleMap()
    const {
  INITIALIZATION_STATE_DCHECK_VALID(initialized_);
  std::map<std::string, std::string> annotations;
  if (crashpad_info_ && crashpad_info_->SimpleAnnotations()) {
    ImageAnnotationReader reader(process_memory_range_);
    reader.SimpleMap(crashpad_info_->SimpleAnnotations(), &annotations);
  }
  return annotations;
}

std::vector<AnnotationSnapshot> ModuleSnapshotElf::AnnotationObjects() const {
  INITIALIZATION_STATE_DCHECK_VALID(initialized_);
  std::vector<AnnotationSnapshot> annotations;
  if (crashpad_info_ && crashpad_info_->AnnotationsList()) {
    ImageAnnotationReader reader(process_memory_range_);
    reader.AnnotationsList(crashpad_info_->AnnotationsList(), &annotations);
  }
  return annotations;
}

std::set<CheckedRange<uint64_t>> ModuleSnapshotElf::ExtraMemoryRanges() const {
  INITIALIZATION_STATE_DCHECK_VALID(initialized_);
  return std::set<CheckedRange<uint64_t>>();
}

std::vector<const UserMinidumpStream*>
ModuleSnapshotElf::CustomMinidumpStreams() const {
  return std::vector<const UserMinidumpStream*>();
}

}  // namespace internal
}  // namespace crashpad
