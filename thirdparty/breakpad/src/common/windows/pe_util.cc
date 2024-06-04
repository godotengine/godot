// Copyright (c) 2019, Google Inc.
// All rights reserved.
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
//     * Neither the name of Google Inc. nor the names of its
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

#include "pe_util.h"

#include <windows.h>
#include <winnt.h>
#include <atlbase.h>
#include <ImageHlp.h>

#include <functional>

#include "common/windows/string_utils-inl.h"
#include "common/windows/guid_string.h"

namespace {

/*
 * Not defined in WinNT.h for some reason. Definitions taken from:
 * http://uninformed.org/index.cgi?v=4&a=1&p=13
 *
 */
typedef unsigned char UBYTE;

#if !defined(_WIN64)
#define UNW_FLAG_EHANDLER  0x01
#define UNW_FLAG_UHANDLER  0x02
#define UNW_FLAG_CHAININFO 0x04
#endif

union UnwindCode {
  struct {
    UBYTE offset_in_prolog;
    UBYTE unwind_operation_code : 4;
    UBYTE operation_info : 4;
  };
  USHORT frame_offset;
};

enum UnwindOperationCodes {
  UWOP_PUSH_NONVOL = 0, /* info == register number */
  UWOP_ALLOC_LARGE,     /* no info, alloc size in next 2 slots */
  UWOP_ALLOC_SMALL,     /* info == size of allocation / 8 - 1 */
  UWOP_SET_FPREG,       /* no info, FP = RSP + UNWIND_INFO.FPRegOffset*16 */
  UWOP_SAVE_NONVOL,     /* info == register number, offset in next slot */
  UWOP_SAVE_NONVOL_FAR, /* info == register number, offset in next 2 slots */
  // XXX: these are missing from MSDN!
  // See: http://www.osronline.com/ddkx/kmarch/64bitamd_4rs7.htm
  UWOP_SAVE_XMM,
  UWOP_SAVE_XMM_FAR,
  UWOP_SAVE_XMM128,     /* info == XMM reg number, offset in next slot */
  UWOP_SAVE_XMM128_FAR, /* info == XMM reg number, offset in next 2 slots */
  UWOP_PUSH_MACHFRAME   /* info == 0: no error-code, 1: error-code */
};

// See: http://msdn.microsoft.com/en-us/library/ddssxxy8.aspx
// Note: some fields removed as we don't use them.
struct UnwindInfo {
  UBYTE version : 3;
  UBYTE flags : 5;
  UBYTE size_of_prolog;
  UBYTE count_of_codes;
  UBYTE frame_register : 4;
  UBYTE frame_offset : 4;
  UnwindCode unwind_code[1];
};

struct CV_INFO_PDB70 {
  ULONG cv_signature;
  GUID signature;
  ULONG age;
  CHAR pdb_filename[ANYSIZE_ARRAY];
};

#define CV_SIGNATURE_RSDS 'SDSR'

// A helper class to scope a PLOADED_IMAGE.
class AutoImage {
public:
  explicit AutoImage(PLOADED_IMAGE img) : img_(img) {}
  ~AutoImage() {
    if (img_)
      ImageUnload(img_);
  }

  operator PLOADED_IMAGE() { return img_; }
  PLOADED_IMAGE operator->() { return img_; }

private:
  PLOADED_IMAGE img_;
};
}  // namespace

namespace google_breakpad {

using std::unique_ptr;
using google_breakpad::GUIDString;

bool ReadModuleInfo(const wstring & pe_file, PDBModuleInfo * info) {
  // Convert wchar to native charset because ImageLoad only takes
  // a PSTR as input.
  string img_file;
  if (!WindowsStringUtils::safe_wcstombs(pe_file, &img_file)) {
    fprintf(stderr, "Image path '%S' contains unrecognized characters.\n",
        pe_file.c_str());
    return false;
  }

  AutoImage img(ImageLoad((PSTR)img_file.c_str(), NULL));
  if (!img) {
    fprintf(stderr, "Failed to load %s\n", img_file.c_str());
    return false;
  }

  info->cpu = FileHeaderMachineToCpuString(
      img->FileHeader->FileHeader.Machine);

  PIMAGE_OPTIONAL_HEADER64 optional_header =
      &(reinterpret_cast<PIMAGE_NT_HEADERS64>(img->FileHeader))->OptionalHeader;

  // Search debug directories for a guid signature & age
  DWORD debug_rva = optional_header->
    DataDirectory[IMAGE_DIRECTORY_ENTRY_DEBUG].VirtualAddress;
  DWORD debug_size = optional_header->
    DataDirectory[IMAGE_DIRECTORY_ENTRY_DEBUG].Size;
  PIMAGE_DEBUG_DIRECTORY debug_directories =
    static_cast<PIMAGE_DEBUG_DIRECTORY>(
      ImageRvaToVa(img->FileHeader,
        img->MappedAddress,
        debug_rva,
        &img->LastRvaSection));

  for (DWORD i = 0; i < debug_size / sizeof(*debug_directories); i++) {
    if (debug_directories[i].Type != IMAGE_DEBUG_TYPE_CODEVIEW ||
        debug_directories[i].SizeOfData < sizeof(CV_INFO_PDB70)) {
      continue;
    }

    struct CV_INFO_PDB70* cv_info = static_cast<CV_INFO_PDB70*>(ImageRvaToVa(
        img->FileHeader,
        img->MappedAddress,
        debug_directories[i].AddressOfRawData,
        &img->LastRvaSection));
    if (cv_info->cv_signature != CV_SIGNATURE_RSDS) {
      continue;
    }

    info->debug_identifier = GenerateDebugIdentifier(cv_info->age,
        cv_info->signature);

    // This code assumes that the pdb_filename is stored as ASCII without
    // multibyte characters, but it's not clear if that's true.
    size_t debug_file_length = strnlen_s(cv_info->pdb_filename, MAX_PATH);
    if (debug_file_length < 0 || debug_file_length >= MAX_PATH) {
      fprintf(stderr, "PE debug directory is corrupt.\n");
      return false;
    }
    std::string debug_file(cv_info->pdb_filename, debug_file_length);
    if (!WindowsStringUtils::safe_mbstowcs(debug_file, &info->debug_file)) {
      fprintf(stderr, "PDB filename '%s' contains unrecognized characters.\n",
          debug_file.c_str());
      return false;
    }
    info->debug_file = WindowsStringUtils::GetBaseName(info->debug_file);

    return true;
  }

  fprintf(stderr, "Image is missing debug information.\n");
  return false;
}

bool ReadPEInfo(const wstring & pe_file, PEModuleInfo * info) {
  // Convert wchar to native charset because ImageLoad only takes
  // a PSTR as input.
  string img_file;
  if (!WindowsStringUtils::safe_wcstombs(pe_file, &img_file)) {
    fprintf(stderr, "Image path '%S' contains unrecognized characters.\n",
        pe_file.c_str());
    return false;
  }

  AutoImage img(ImageLoad((PSTR)img_file.c_str(), NULL));
  if (!img) {
    fprintf(stderr, "Failed to open PE file: %S\n", pe_file.c_str());
    return false;
  }

  info->code_file = WindowsStringUtils::GetBaseName(pe_file);

  // The date and time that the file was created by the linker.
  DWORD TimeDateStamp = img->FileHeader->FileHeader.TimeDateStamp;
  // The size of the file in bytes, including all headers.
  DWORD SizeOfImage = 0;
  PIMAGE_OPTIONAL_HEADER64 opt =
    &((PIMAGE_NT_HEADERS64)img->FileHeader)->OptionalHeader;
  if (opt->Magic == IMAGE_NT_OPTIONAL_HDR64_MAGIC) {
    // 64-bit PE file.
    SizeOfImage = opt->SizeOfImage;
  }
  else {
    // 32-bit PE file.
    SizeOfImage = img->FileHeader->OptionalHeader.SizeOfImage;
  }
  wchar_t code_identifier[32];
  swprintf(code_identifier,
    sizeof(code_identifier) / sizeof(code_identifier[0]),
    L"%08X%X", TimeDateStamp, SizeOfImage);
  info->code_identifier = code_identifier;

  return true;
}

bool PrintPEFrameData(const wstring & pe_file, FILE * out_file)
{
  // Convert wchar to native charset because ImageLoad only takes
  // a PSTR as input.
  string img_file;
  if (!WindowsStringUtils::safe_wcstombs(pe_file, &img_file)) {
    fprintf(stderr, "Image path '%S' contains unrecognized characters.\n",
        pe_file.c_str());
    return false;
  }

  AutoImage img(ImageLoad((PSTR)img_file.c_str(), NULL));
  if (!img) {
    fprintf(stderr, "Failed to load %s\n", img_file.c_str());
    return false;
  }
  PIMAGE_OPTIONAL_HEADER64 optional_header =
    &(reinterpret_cast<PIMAGE_NT_HEADERS64>(img->FileHeader))->OptionalHeader;
  if (optional_header->Magic != IMAGE_NT_OPTIONAL_HDR64_MAGIC) {
    fprintf(stderr, "Not a PE32+ image\n");
    return false;
  }

  // Read Exception Directory
  DWORD exception_rva = optional_header->
    DataDirectory[IMAGE_DIRECTORY_ENTRY_EXCEPTION].VirtualAddress;
  DWORD exception_size = optional_header->
    DataDirectory[IMAGE_DIRECTORY_ENTRY_EXCEPTION].Size;
  PIMAGE_RUNTIME_FUNCTION_ENTRY funcs =
    static_cast<PIMAGE_RUNTIME_FUNCTION_ENTRY>(
      ImageRvaToVa(img->FileHeader,
        img->MappedAddress,
        exception_rva,
        &img->LastRvaSection));
  for (DWORD i = 0; i < exception_size / sizeof(*funcs); i++) {
    DWORD unwind_rva = funcs[i].UnwindInfoAddress;
    // handle chaining
    while (unwind_rva & 0x1) {
      unwind_rva ^= 0x1;
      PIMAGE_RUNTIME_FUNCTION_ENTRY chained_func =
        static_cast<PIMAGE_RUNTIME_FUNCTION_ENTRY>(
          ImageRvaToVa(img->FileHeader,
            img->MappedAddress,
            unwind_rva,
            &img->LastRvaSection));
      unwind_rva = chained_func->UnwindInfoAddress;
    }

    UnwindInfo *unwind_info = static_cast<UnwindInfo*>(
      ImageRvaToVa(img->FileHeader,
        img->MappedAddress,
        unwind_rva,
        &img->LastRvaSection));

    DWORD stack_size = 8;  // minimal stack size is 8 for RIP
    DWORD rip_offset = 8;
    do {
      for (UBYTE c = 0; c < unwind_info->count_of_codes; c++) {
        UnwindCode *unwind_code = &unwind_info->unwind_code[c];
        switch (unwind_code->unwind_operation_code) {
        case UWOP_PUSH_NONVOL: {
          stack_size += 8;
          break;
        }
        case UWOP_ALLOC_LARGE: {
          if (unwind_code->operation_info == 0) {
            c++;
            if (c < unwind_info->count_of_codes)
              stack_size += (unwind_code + 1)->frame_offset * 8;
          }
          else {
            c += 2;
            if (c < unwind_info->count_of_codes)
              stack_size += (unwind_code + 1)->frame_offset |
              ((unwind_code + 2)->frame_offset << 16);
          }
          break;
        }
        case UWOP_ALLOC_SMALL: {
          stack_size += unwind_code->operation_info * 8 + 8;
          break;
        }
        case UWOP_SET_FPREG:
        case UWOP_SAVE_XMM:
        case UWOP_SAVE_XMM_FAR:
          break;
        case UWOP_SAVE_NONVOL:
        case UWOP_SAVE_XMM128: {
          c++;  // skip slot with offset
          break;
        }
        case UWOP_SAVE_NONVOL_FAR:
        case UWOP_SAVE_XMM128_FAR: {
          c += 2;  // skip 2 slots with offset
          break;
        }
        case UWOP_PUSH_MACHFRAME: {
          if (unwind_code->operation_info) {
            stack_size += 88;
          }
          else {
            stack_size += 80;
          }
          rip_offset += 80;
          break;
        }
        }
      }
      if (unwind_info->flags & UNW_FLAG_CHAININFO) {
        PIMAGE_RUNTIME_FUNCTION_ENTRY chained_func =
          reinterpret_cast<PIMAGE_RUNTIME_FUNCTION_ENTRY>(
          (unwind_info->unwind_code +
            ((unwind_info->count_of_codes + 1) & ~1)));

        unwind_info = static_cast<UnwindInfo*>(
          ImageRvaToVa(img->FileHeader,
            img->MappedAddress,
            chained_func->UnwindInfoAddress,
            &img->LastRvaSection));
      }
      else {
        unwind_info = NULL;
      }
    } while (unwind_info);
    fprintf(out_file, "STACK CFI INIT %lx %lx .cfa: $rsp .ra: .cfa %lu - ^\n",
      funcs[i].BeginAddress,
      funcs[i].EndAddress - funcs[i].BeginAddress, rip_offset);
    fprintf(out_file, "STACK CFI %lx .cfa: $rsp %lu +\n",
      funcs[i].BeginAddress, stack_size);
  }

  return true;
}

wstring GenerateDebugIdentifier(DWORD age, GUID signature)
{
  // Use the same format that the MS symbol server uses in filesystem
  // hierarchies.
  wchar_t age_string[9];
  swprintf(age_string, sizeof(age_string) / sizeof(age_string[0]),
    L"%x", age);

  // remove when VC++7.1 is no longer supported
  age_string[sizeof(age_string) / sizeof(age_string[0]) - 1] = L'\0';

  wstring debug_identifier = GUIDString::GUIDToSymbolServerWString(&signature);
  debug_identifier.append(age_string);

  return debug_identifier;
}

wstring GenerateDebugIdentifier(DWORD age, DWORD signature)
{
  // Use the same format that the MS symbol server uses in filesystem
  // hierarchies.
  wchar_t identifier_string[17];
  swprintf(identifier_string,
    sizeof(identifier_string) / sizeof(identifier_string[0]),
    L"%08X%x", signature, age);

  // remove when VC++7.1 is no longer supported
  identifier_string[sizeof(identifier_string) /
    sizeof(identifier_string[0]) - 1] = L'\0';

  return wstring(identifier_string);
}

}  // namespace google_breakpad
