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

#include "snapshot/win/pe_image_reader.h"

#include <stddef.h>
#include <string.h>

#include <algorithm>
#include <memory>

#include "base/logging.h"
#include "base/strings/stringprintf.h"
#include "client/crashpad_info.h"
#include "snapshot/win/pe_image_resource_reader.h"
#include "util/misc/from_pointer_cast.h"
#include "util/misc/pdb_structures.h"
#include "util/win/process_structs.h"

namespace crashpad {

namespace {

// Map from Traits to an IMAGE_NT_HEADERSxx.
template <class Traits>
struct NtHeadersForTraits;

template <>
struct NtHeadersForTraits<process_types::internal::Traits32> {
  using type = IMAGE_NT_HEADERS32;
};

template <>
struct NtHeadersForTraits<process_types::internal::Traits64> {
  using type = IMAGE_NT_HEADERS64;
};

}  // namespace

PEImageReader::PEImageReader()
    : module_subrange_reader_(),
      initialized_() {
}

PEImageReader::~PEImageReader() {
}

bool PEImageReader::Initialize(ProcessReaderWin* process_reader,
                               WinVMAddress address,
                               WinVMSize size,
                               const std::string& module_name) {
  INITIALIZATION_STATE_SET_INITIALIZING(initialized_);

  if (!module_subrange_reader_.Initialize(
          process_reader, address, size, module_name)) {
    return false;
  }

  INITIALIZATION_STATE_SET_VALID(initialized_);
  return true;
}

template <class Traits>
bool PEImageReader::GetCrashpadInfo(
    process_types::CrashpadInfo<Traits>* crashpad_info) const {
  INITIALIZATION_STATE_DCHECK_VALID(initialized_);

  IMAGE_SECTION_HEADER section;
  if (!GetSectionByName<typename NtHeadersForTraits<Traits>::type>("CPADinfo",
                                                                   &section)) {
    return false;
  }

  if (section.Misc.VirtualSize <
      offsetof(process_types::CrashpadInfo<Traits>, size) +
          sizeof(crashpad_info->size)) {
    LOG(WARNING) << "small crashpad info section size "
                 << section.Misc.VirtualSize << ", "
                 << module_subrange_reader_.name();
    return false;
  }

  const WinVMAddress crashpad_info_address = Address() + section.VirtualAddress;
  const WinVMSize crashpad_info_size =
      std::min(static_cast<WinVMSize>(sizeof(*crashpad_info)),
               static_cast<WinVMSize>(section.Misc.VirtualSize));
  if (!module_subrange_reader_.ReadMemory(
          crashpad_info_address, crashpad_info_size, crashpad_info)) {
    LOG(WARNING) << "could not read crashpad info from "
                 << module_subrange_reader_.name();
    return false;
  }

  if (crashpad_info->size < sizeof(*crashpad_info)) {
    // Zero out anything beyond the structure’s declared size.
    memset(reinterpret_cast<char*>(crashpad_info) + crashpad_info->size,
           0,
           sizeof(*crashpad_info) - crashpad_info->size);
  }

  if (crashpad_info->signature != CrashpadInfo::kSignature ||
      crashpad_info->version != 1) {
    LOG(WARNING) << base::StringPrintf(
        "unexpected crashpad info signature 0x%x, version %u in %s",
        crashpad_info->signature,
        crashpad_info->version,
        module_subrange_reader_.name().c_str());
    return false;
  }

  // Don’t require strict equality, to leave wiggle room for sloppy linkers.
  if (crashpad_info->size > section.Misc.VirtualSize) {
    LOG(WARNING) << "crashpad info struct size " << crashpad_info->size
                 << " large for section size " << section.Misc.VirtualSize
                 << " in " << module_subrange_reader_.name();
    return false;
  }

  if (crashpad_info->size > sizeof(*crashpad_info)) {
    // This isn’t strictly a problem, because unknown fields will simply be
    // ignored, but it may be of diagnostic interest.
    LOG(INFO) << "large crashpad info size " << crashpad_info->size << ", "
              << module_subrange_reader_.name();
  }

  return true;
}

bool PEImageReader::DebugDirectoryInformation(UUID* uuid,
                                              DWORD* age,
                                              std::string* pdbname) const {
  INITIALIZATION_STATE_DCHECK_VALID(initialized_);

  IMAGE_DATA_DIRECTORY data_directory;
  if (!ImageDataDirectoryEntry(IMAGE_DIRECTORY_ENTRY_DEBUG, &data_directory))
    return false;

  IMAGE_DEBUG_DIRECTORY debug_directory;
  if (data_directory.Size % sizeof(debug_directory) != 0)
    return false;
  for (size_t offset = 0; offset < data_directory.Size;
       offset += sizeof(debug_directory)) {
    if (!module_subrange_reader_.ReadMemory(
            Address() + data_directory.VirtualAddress + offset,
            sizeof(debug_directory),
            &debug_directory)) {
      LOG(WARNING) << "could not read data directory from "
                   << module_subrange_reader_.name();
      return false;
    }

    if (debug_directory.Type != IMAGE_DEBUG_TYPE_CODEVIEW)
      continue;

    if (debug_directory.AddressOfRawData) {
      if (debug_directory.SizeOfData < sizeof(CodeViewRecordPDB70)) {
        LOG(WARNING) << "CodeView debug entry of unexpected size in "
                     << module_subrange_reader_.name();
        continue;
      }

      std::unique_ptr<char[]> data(new char[debug_directory.SizeOfData]);
      if (!module_subrange_reader_.ReadMemory(
              Address() + debug_directory.AddressOfRawData,
              debug_directory.SizeOfData,
              data.get())) {
        LOG(WARNING) << "could not read debug directory from "
                     << module_subrange_reader_.name();
        return false;
      }

      if (*reinterpret_cast<DWORD*>(data.get()) !=
          CodeViewRecordPDB70::kSignature) {
        LOG(WARNING) << "encountered non-7.0 CodeView debug record in "
                     << module_subrange_reader_.name();
        continue;
      }

      CodeViewRecordPDB70* codeview =
          reinterpret_cast<CodeViewRecordPDB70*>(data.get());
      *uuid = codeview->uuid;
      *age = codeview->age;
      // This is a NUL-terminated string encoded in the codepage of the system
      // where the binary was linked. We have no idea what that was, so we just
      // assume ASCII.
      *pdbname = std::string(reinterpret_cast<char*>(&codeview->pdb_name[0]));
      return true;
    } else if (debug_directory.PointerToRawData) {
      // This occurs for non-PDB based debug information. We simply ignore these
      // as we don't expect to encounter modules that will be in this format
      // for which we'll actually have symbols. See
      // https://crashpad.chromium.org/bug/47.
    }
  }

  return false;
}

bool PEImageReader::VSFixedFileInfo(
    VS_FIXEDFILEINFO* vs_fixed_file_info) const {
  INITIALIZATION_STATE_DCHECK_VALID(initialized_);

  IMAGE_DATA_DIRECTORY data_directory;
  if (!ImageDataDirectoryEntry(IMAGE_DIRECTORY_ENTRY_RESOURCE,
                               &data_directory)) {
    return false;
  }

  PEImageResourceReader resource_reader;
  if (!resource_reader.Initialize(module_subrange_reader_, data_directory)) {
    return false;
  }

  WinVMAddress address;
  WinVMSize size;
  if (!resource_reader.FindResourceByID(
          FromPointerCast<uint16_t>(VS_FILE_INFO),  // RT_VERSION
          VS_VERSION_INFO,
          MAKELANGID(LANG_NEUTRAL, SUBLANG_NEUTRAL),
          &address,
          &size,
          nullptr)) {
    return false;
  }

  // This structure is not declared anywhere in the SDK, but is documented at
  // https://msdn.microsoft.com/library/ms647001.aspx.
  struct VS_VERSIONINFO {
    WORD wLength;
    WORD wValueLength;
    WORD wType;

    // The structure documentation on MSDN doesn’t show the [16], but it does
    // say that it’s supposed to be L"VS_VERSION_INFO", which is is in fact a
    // 16-character string (including its NUL terminator).
    WCHAR szKey[16];

    WORD Padding1;
    VS_FIXEDFILEINFO Value;

    // Don’t include Children or the Padding2 that precedes it, because they may
    // not be present.
    // WORD Padding2;
    // WORD Children;
  };
  VS_VERSIONINFO version_info;

  if (size < sizeof(version_info)) {
    LOG(WARNING) << "version info size " << size
                 << " too small for structure of size " << sizeof(version_info)
                 << " in " << module_subrange_reader_.name();
    return false;
  }

  if (!module_subrange_reader_.ReadMemory(
          address, sizeof(version_info), &version_info)) {
    LOG(WARNING) << "could not read version info from "
                 << module_subrange_reader_.name();
    return false;
  }

  if (version_info.wLength < sizeof(version_info) ||
      version_info.wValueLength != sizeof(version_info.Value) ||
      version_info.wType != 0 ||
      wcsncmp(version_info.szKey,
              L"VS_VERSION_INFO",
              arraysize(version_info.szKey)) != 0) {
    LOG(WARNING) << "unexpected VS_VERSIONINFO in "
                 << module_subrange_reader_.name();
    return false;
  }

  if (version_info.Value.dwSignature != VS_FFI_SIGNATURE ||
      version_info.Value.dwStrucVersion != VS_FFI_STRUCVERSION) {
    LOG(WARNING) << "unexpected VS_FIXEDFILEINFO in "
                 << module_subrange_reader_.name();
    return false;
  }

  *vs_fixed_file_info = version_info.Value;
  vs_fixed_file_info->dwFileFlags &= vs_fixed_file_info->dwFileFlagsMask;
  return true;
}

template <class NtHeadersType>
bool PEImageReader::ReadNtHeaders(NtHeadersType* nt_headers,
                                  WinVMAddress* nt_headers_address) const {
  IMAGE_DOS_HEADER dos_header;
  if (!module_subrange_reader_.ReadMemory(
          Address(), sizeof(IMAGE_DOS_HEADER), &dos_header)) {
    LOG(WARNING) << "could not read dos header from "
                 << module_subrange_reader_.name();
    return false;
  }

  if (dos_header.e_magic != IMAGE_DOS_SIGNATURE) {
    LOG(WARNING) << "invalid e_magic in dos header of "
                 << module_subrange_reader_.name();
    return false;
  }

  WinVMAddress local_nt_headers_address = Address() + dos_header.e_lfanew;
  if (!module_subrange_reader_.ReadMemory(
          local_nt_headers_address, sizeof(NtHeadersType), nt_headers)) {
    LOG(WARNING) << "could not read nt headers from "
                 << module_subrange_reader_.name();
    return false;
  }

  if (nt_headers->Signature != IMAGE_NT_SIGNATURE) {
    LOG(WARNING) << "invalid signature in nt headers of "
                 << module_subrange_reader_.name();
    return false;
  }

  if (nt_headers_address)
    *nt_headers_address = local_nt_headers_address;

  return true;
}

template <class NtHeadersType>
bool PEImageReader::GetSectionByName(const std::string& name,
                                     IMAGE_SECTION_HEADER* section) const {
  if (name.size() > sizeof(section->Name)) {
    LOG(WARNING) << "supplied section name too long " << name;
    return false;
  }

  NtHeadersType nt_headers;
  WinVMAddress nt_headers_address;
  if (!ReadNtHeaders(&nt_headers, &nt_headers_address))
    return false;

  WinVMAddress first_section_address =
      nt_headers_address + offsetof(NtHeadersType, OptionalHeader) +
      nt_headers.FileHeader.SizeOfOptionalHeader;
  for (DWORD i = 0; i < nt_headers.FileHeader.NumberOfSections; ++i) {
    WinVMAddress section_address =
        first_section_address + sizeof(IMAGE_SECTION_HEADER) * i;
    if (!module_subrange_reader_.ReadMemory(
            section_address, sizeof(IMAGE_SECTION_HEADER), section)) {
      LOG(WARNING) << "could not read section " << i << " from "
                   << module_subrange_reader_.name();
      return false;
    }
    if (strncmp(reinterpret_cast<const char*>(section->Name),
                name.c_str(),
                sizeof(section->Name)) == 0) {
      return true;
    }
  }

  return false;
}

bool PEImageReader::ImageDataDirectoryEntry(size_t index,
                                            IMAGE_DATA_DIRECTORY* entry) const {
  bool rv;
  if (module_subrange_reader_.Is64Bit()) {
    rv = ImageDataDirectoryEntryT<IMAGE_NT_HEADERS64>(index, entry);
  } else {
    rv = ImageDataDirectoryEntryT<IMAGE_NT_HEADERS32>(index, entry);
  }

  return rv && entry->VirtualAddress != 0 && entry->Size != 0;
}

template <class NtHeadersType>
bool PEImageReader::ImageDataDirectoryEntryT(
    size_t index,
    IMAGE_DATA_DIRECTORY* entry) const {
  NtHeadersType nt_headers;
  if (!ReadNtHeaders(&nt_headers, nullptr)) {
    return false;
  }

  if (nt_headers.FileHeader.SizeOfOptionalHeader <
          offsetof(decltype(nt_headers.OptionalHeader), DataDirectory[index]) +
              sizeof(nt_headers.OptionalHeader.DataDirectory[index]) ||
      nt_headers.OptionalHeader.NumberOfRvaAndSizes <= index) {
    return false;
  }

  *entry = nt_headers.OptionalHeader.DataDirectory[index];
  return true;
}

// Explicit instantiations with the only 2 valid template arguments to avoid
// putting the body of the function in the header.
template bool PEImageReader::GetCrashpadInfo<process_types::internal::Traits32>(
    process_types::CrashpadInfo<process_types::internal::Traits32>*
        crashpad_info) const;
template bool PEImageReader::GetCrashpadInfo<process_types::internal::Traits64>(
    process_types::CrashpadInfo<process_types::internal::Traits64>*
        crashpad_info) const;

}  // namespace crashpad
