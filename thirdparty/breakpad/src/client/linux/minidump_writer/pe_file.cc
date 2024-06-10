// Copyright 2022 Google LLC
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
//     * Redistributions of source code must retain the above copyright
// notice, this list of conditions and the following disclaimer.
//     * Redistributions in binary form must reproduce the above
// copyright notice, this list of conditions and the following disclaimer
// in the documentation and/or other materials provided with the
// distribution.
//     * Neither the name of Google LLC nor the names of its
// contributors may be used to endorse or promote products derived from
// this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
// "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
// LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
// A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
// OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
// SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
// LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
// DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
// THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#ifdef HAVE_CONFIG_H
#include <config.h>  // Must come first
#endif

#include <string.h>

#include "client/linux/minidump_writer/pe_file.h"
#include "client/linux/minidump_writer/pe_structs.h"
#include "common/linux/memory_mapped_file.h"

namespace google_breakpad {

PEFileFormat PEFile::TryGetDebugInfo(const char* filename,
                                      PRSDS_DEBUG_FORMAT debug_info) {
  MemoryMappedFile mapped_file(filename, 0);
  if (!mapped_file.data())
    return PEFileFormat::notPeCoff;
  const void* base = mapped_file.data();
  const size_t file_size = mapped_file.size();

  const IMAGE_DOS_HEADER* header =
      TryReadStruct<IMAGE_DOS_HEADER>(base, 0, file_size);
  if (!header || (header->e_magic != IMAGE_DOS_SIGNATURE)) {
    return PEFileFormat::notPeCoff;
  }

  // NTHeader is at position 'e_lfanew'.
  DWORD nt_header_offset = header->e_lfanew;
  // First, read a common IMAGE_NT_HEADERS structure. It should contain a
  // special flag marking whether PE module is x64 (OptionalHeader.Magic)
  // and so-called NT_SIGNATURE in Signature field.
  const IMAGE_NT_HEADERS* nt_header =
      TryReadStruct<IMAGE_NT_HEADERS>(base, nt_header_offset, file_size);
  if (!nt_header || (nt_header->Signature != IMAGE_NT_SIGNATURE)
     || ((nt_header->OptionalHeader.Magic != IMAGE_NT_OPTIONAL_HDR64_MAGIC)
     &&  (nt_header->OptionalHeader.Magic != IMAGE_NT_OPTIONAL_HDR32_MAGIC)))
    return PEFileFormat::notPeCoff;

  bool x64 = nt_header->OptionalHeader.Magic == IMAGE_NT_OPTIONAL_HDR64_MAGIC;
  WORD sections_number = nt_header->FileHeader.NumberOfSections;
  DWORD debug_offset;
  DWORD debug_size;
  DWORD section_offset;
  if (x64) {
    const IMAGE_NT_HEADERS64* header_64 =
        TryReadStruct<IMAGE_NT_HEADERS64>(base, nt_header_offset, file_size);
    if (!header_64)
      return PEFileFormat::peWithoutBuildId;
    debug_offset =
        header_64->OptionalHeader.DataDirectory[IMAGE_DIRECTORY_ENTRY_DEBUG]
            .VirtualAddress;
    debug_size =
        header_64->OptionalHeader.DataDirectory[IMAGE_DIRECTORY_ENTRY_DEBUG]
            .Size;
    section_offset = nt_header_offset + sizeof(IMAGE_NT_HEADERS64);
  } else {
    const IMAGE_NT_HEADERS32* header_32 =
        TryReadStruct<IMAGE_NT_HEADERS32>(base, nt_header_offset, file_size);
    if (!header_32)
      return PEFileFormat::peWithoutBuildId;
    debug_offset =
        header_32->OptionalHeader.DataDirectory[IMAGE_DIRECTORY_ENTRY_DEBUG]
            .VirtualAddress;
    debug_size =
        header_32->OptionalHeader.DataDirectory[IMAGE_DIRECTORY_ENTRY_DEBUG]
            .Size;
    section_offset = nt_header_offset + sizeof(IMAGE_NT_HEADERS32);
  }

  DWORD debug_end_pos = debug_offset + debug_size;
  while (debug_offset < debug_end_pos) {
    for (WORD i = 0; i < sections_number; ++i) {
      // Section headers are placed sequentially after the NT_HEADER (32/64).
      const IMAGE_SECTION_HEADER* section =
          TryReadStruct<IMAGE_SECTION_HEADER>(base, section_offset, file_size);
      if (!section)
        return PEFileFormat::peWithoutBuildId;

      section_offset += sizeof(IMAGE_SECTION_HEADER);

      // Current `debug_offset` should be inside a section, stop if we find
      // a suitable one (we don't consider any malformed sections here).
      if ((section->VirtualAddress <= debug_offset) &&
          (debug_offset < section->VirtualAddress + section->SizeOfRawData)) {
        DWORD offset =
            section->PointerToRawData + debug_offset - section->VirtualAddress;
        // Go to the position of current ImageDebugDirectory (offset).
        const IMAGE_DEBUG_DIRECTORY* debug_directory =
            TryReadStruct<IMAGE_DEBUG_DIRECTORY>(base, offset, file_size);
        if (!debug_directory)
          return PEFileFormat::peWithoutBuildId;
        // Process ImageDebugDirectory with CodeViewRecord type and skip
        // all others.
        if (debug_directory->Type == IMAGE_DEBUG_TYPE_CODEVIEW) {
          DWORD debug_directory_size = debug_directory->SizeOfData;
          if (debug_directory_size < sizeof(RSDS_DEBUG_FORMAT))
            // RSDS section is malformed.
            return PEFileFormat::peWithoutBuildId;
          // Go to the position of current ImageDebugDirectory Raw Data
          // (debug_directory->PointerToRawData) and read the RSDS section.
          const RSDS_DEBUG_FORMAT* rsds =
              TryReadStruct<RSDS_DEBUG_FORMAT>(
                base, debug_directory->PointerToRawData, file_size);

          if (!rsds)
            return PEFileFormat::peWithoutBuildId;

          memcpy(debug_info->guid, rsds->guid, sizeof(rsds->guid));
          memcpy(debug_info->age, rsds->age, sizeof(rsds->age));
          return PEFileFormat::peWithBuildId;
        }

        break;
      }
    }

    debug_offset += sizeof(IMAGE_DEBUG_DIRECTORY);
  }

  return PEFileFormat::peWithoutBuildId;
}

} // namespace google_breakpad