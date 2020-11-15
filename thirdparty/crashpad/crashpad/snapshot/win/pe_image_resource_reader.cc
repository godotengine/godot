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

#include "snapshot/win/pe_image_resource_reader.h"

#include <algorithm>
#include <memory>

#include "base/logging.h"

namespace {

void AddLanguageAndNeutralSublanguage(std::vector<uint16_t>* languages,
                                      uint16_t language) {
  languages->push_back(language);
  if (SUBLANGID(language) != SUBLANG_NEUTRAL) {
    languages->push_back(MAKELANGID(PRIMARYLANGID(language), SUBLANG_NEUTRAL));
  }
}

}  // namespace

namespace crashpad {

PEImageResourceReader::PEImageResourceReader()
    : resources_subrange_reader_(),
      module_base_(0),
      initialized_() {
}

PEImageResourceReader::~PEImageResourceReader() {}

bool PEImageResourceReader::Initialize(
    const ProcessSubrangeReader& module_subrange_reader,
    const IMAGE_DATA_DIRECTORY& resources_directory_entry) {
  INITIALIZATION_STATE_SET_INITIALIZING(initialized_);

  module_base_ = module_subrange_reader.Base();

  if (!resources_subrange_reader_.InitializeSubrange(
          module_subrange_reader,
          module_base_ + resources_directory_entry.VirtualAddress,
          resources_directory_entry.Size,
          "resources")) {
    return false;
  }

  INITIALIZATION_STATE_SET_VALID(initialized_);
  return true;
}

bool PEImageResourceReader::FindResourceByID(uint16_t type,
                                             uint16_t name,
                                             uint16_t language,
                                             WinVMAddress* address,
                                             WinVMSize* size,
                                             uint32_t* code_page) const {
  INITIALIZATION_STATE_DCHECK_VALID(initialized_);

  // The root resource directory is at the beginning of the resources area
  // within the module.
  const uint32_t name_directory_offset =
      GetEntryFromResourceDirectoryByID(0, type, true);
  if (!name_directory_offset) {
    return false;
  }

  const uint32_t language_directory_offset =
      GetEntryFromResourceDirectoryByID(name_directory_offset, name, true);
  if (!language_directory_offset) {
    return false;
  }

  // The definition of IMAGE_RESOURCE_DIRECTORY_ENTRY in <winnt.h> has a comment
  // saying that its offsets are relative to “the resource directory of the data
  // associated with this directory entry”. That could be interpreted to mean
  // that language_directory_offset is relative to name_directory_offset, since
  // the language directory entry is found within the name directory. This is
  // not correct. All resource offsets are relative to the resources area within
  // the module.
  const uint32_t data_offset = GetEntryFromResourceDirectoryByLanguage(
      language_directory_offset, language);
  if (!data_offset) {
    return false;
  }

  IMAGE_RESOURCE_DATA_ENTRY data_entry;
  if (!resources_subrange_reader_.ReadMemory(
          resources_subrange_reader_.Base() + data_offset,
          sizeof(data_entry),
          &data_entry)) {
    LOG(WARNING) << "could not read resource data entry from "
                 << resources_subrange_reader_.name();
    return false;
  }

  // The definition of IMAGE_RESOURCE_DATA_ENTRY in <winnt.h> has a comment
  // saying that OffsetToData is relative to the beginning of the resource data.
  // This is not correct. It’s module-relative.
  *address = module_base_ + data_entry.OffsetToData;
  *size = data_entry.Size;
  if (code_page) {
    *code_page = data_entry.CodePage;
  }

  return true;
}

uint32_t PEImageResourceReader::GetEntryFromResourceDirectoryByID(
    uint32_t language_directory_offset,
    uint16_t id,
    bool want_subdirectory) const {
  INITIALIZATION_STATE_DCHECK_VALID(initialized_);

  std::vector<IMAGE_RESOURCE_DIRECTORY_ENTRY> entries_by_id;
  if (!ReadResourceDirectory(
          language_directory_offset, nullptr, nullptr, &entries_by_id)) {
    return 0;
  }

  const auto entry_it =
      std::find_if(entries_by_id.begin(),
                   entries_by_id.end(),
                   [id](const IMAGE_RESOURCE_DIRECTORY_ENTRY& entry) {
                     return !entry.NameIsString && entry.Id == id;
                   });
  if (entry_it != entries_by_id.end()) {
    if ((entry_it->DataIsDirectory != 0) != want_subdirectory) {
      LOG(WARNING) << "expected " << (want_subdirectory ? "" : "non-")
                   << "directory for entry id " << id << " in "
                   << resources_subrange_reader_.name();
      return 0;
    }

    return entry_it->DataIsDirectory ? entry_it->OffsetToDirectory
                                     : entry_it->OffsetToData;
  }

  return 0;
}

uint32_t PEImageResourceReader::GetEntryFromResourceDirectoryByLanguage(
    uint32_t resource_directory_offset,
    uint16_t language) const {
  INITIALIZATION_STATE_DCHECK_VALID(initialized_);

  std::vector<IMAGE_RESOURCE_DIRECTORY_ENTRY> entries_by_language;
  if (!ReadResourceDirectory(
          resource_directory_offset, nullptr, nullptr, &entries_by_language)) {
    return 0;
  }

  if (entries_by_language.empty()) {
    return 0;
  }

  // https://msdn.microsoft.com/library/cc194810.aspx
  //
  // TODO(mark): It seems like FindResourceEx() might do something more complex.
  // It would be best to mimic its behavior.
  std::vector<uint16_t> try_languages;
  if (PRIMARYLANGID(language) != LANG_NEUTRAL) {
    AddLanguageAndNeutralSublanguage(&try_languages, language);
  } else {
    if (SUBLANGID(language) != SUBLANG_SYS_DEFAULT) {
      AddLanguageAndNeutralSublanguage(&try_languages,
                                       LANGIDFROMLCID(GetThreadLocale()));
      AddLanguageAndNeutralSublanguage(&try_languages,
                                       LANGIDFROMLCID(GetUserDefaultLCID()));
    }
    if (SUBLANGID(language) != SUBLANG_DEFAULT) {
      AddLanguageAndNeutralSublanguage(&try_languages,
                                       LANGIDFROMLCID(GetSystemDefaultLCID()));
    }
  }

  try_languages.push_back(MAKELANGID(LANG_NEUTRAL, SUBLANG_NEUTRAL));
  try_languages.push_back(MAKELANGID(LANG_ENGLISH, SUBLANG_DEFAULT));

  for (const auto try_language : try_languages) {
    const auto entry_it = std::find_if(
        entries_by_language.begin(),
        entries_by_language.end(),
        [try_language](const IMAGE_RESOURCE_DIRECTORY_ENTRY& entry) {
          return !entry.NameIsString && entry.Id == try_language;
        });
    if (entry_it != entries_by_language.end()) {
      if (entry_it->DataIsDirectory) {
        LOG(WARNING) << "expected non-directory for entry language "
                     << try_language << " in "
                     << resources_subrange_reader_.name();
        return 0;
      }

      return entry_it->OffsetToData;
    }
  }

  // Fall back to the first entry in the list.
  const auto& entry = entries_by_language.front();
  if (entry.DataIsDirectory) {
    LOG(WARNING) << "expected non-directory for entry in "
                 << resources_subrange_reader_.name();
    return 0;
  }

  return entry.OffsetToData;
}

bool PEImageResourceReader::ReadResourceDirectory(
    uint32_t resource_directory_offset,
    IMAGE_RESOURCE_DIRECTORY* resource_directory,
    std::vector<IMAGE_RESOURCE_DIRECTORY_ENTRY>* named_entries,
    std::vector<IMAGE_RESOURCE_DIRECTORY_ENTRY>* id_entries) const {
  INITIALIZATION_STATE_DCHECK_VALID(initialized_);

  // resource_directory is optional, but it’s still needed locally even if the
  // caller isn’t interested in it.
  std::unique_ptr<IMAGE_RESOURCE_DIRECTORY> local_resource_directory;
  if (!resource_directory) {
    local_resource_directory.reset(new IMAGE_RESOURCE_DIRECTORY);
    resource_directory = local_resource_directory.get();
  }

  const WinVMAddress address =
      resources_subrange_reader_.Base() + resource_directory_offset;

  if (!resources_subrange_reader_.ReadMemory(
          address, sizeof(*resource_directory), resource_directory)) {
    LOG(WARNING) << "could not read resource directory from "
                 << resources_subrange_reader_.name();
    return false;
  }

  if (named_entries) {
    named_entries->clear();
    named_entries->resize(resource_directory->NumberOfNamedEntries);
    if (!named_entries->empty() &&
        !resources_subrange_reader_.ReadMemory(
            address + sizeof(*resource_directory),
            named_entries->size() * sizeof((*named_entries)[0]),
            &(*named_entries)[0])) {
      LOG(WARNING) << "could not read resource directory named entries from "
                   << resources_subrange_reader_.name();
      return false;
    }
  }

  if (id_entries) {
    id_entries->clear();
    id_entries->resize(resource_directory->NumberOfIdEntries);
    if (!id_entries->empty() &&
        !resources_subrange_reader_.ReadMemory(
            address + sizeof(*resource_directory) +
                resource_directory->NumberOfNamedEntries *
                    sizeof(IMAGE_RESOURCE_DIRECTORY_ENTRY),
            id_entries->size() * sizeof((*id_entries)[0]),
            &(*id_entries)[0])) {
      LOG(WARNING) << "could not read resource directory ID entries from "
                   << resources_subrange_reader_.name();
      return false;
    }
  }

  return true;
}

}  // namespace crashpad
