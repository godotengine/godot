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

#include <intrin.h>
#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <wchar.h>
#include <windows.h>
#include <winternl.h>

namespace {

bool UnicodeStringEndsWithCaseInsensitive(const UNICODE_STRING& us,
                                          const wchar_t* ends_with) {
  const size_t len = wcslen(ends_with);
  // Recall that UNICODE_STRING.Length is in bytes, not characters.
  const size_t us_len_in_chars = us.Length / sizeof(wchar_t);
  if (us_len_in_chars < len)
    return false;
  return _wcsnicmp(&us.Buffer[us_len_in_chars - len], ends_with, len) == 0;
}

}  // namespace

// A simple binary to be loaded and inspected by ProcessInfo.
int wmain(int argc, wchar_t** argv) {
  if (argc != 2)
    abort();

  // Get a handle to the event we use to communicate with our parent.
  HANDLE done_event = CreateEvent(nullptr, true, false, argv[1]);
  if (!done_event)
    abort();

  // Load an unusual module (that we don't depend upon) so we can do an
  // existence check. It's also important that these DLLs don't depend on
  // any other DLLs, otherwise there'll be additional modules in the list, which
  // the test expects not to be there.
  if (!LoadLibrary(L"lz32.dll"))
    abort();

  // Load another unusual module so we can destroy its FullDllName field in the
  // PEB to test corrupted name reads.
  static constexpr wchar_t kCorruptableDll[] = L"kbdurdu.dll";
  if (!LoadLibrary(kCorruptableDll))
    abort();

  // Find and corrupt the buffer pointer to the name in the PEB.
  HINSTANCE ntdll = GetModuleHandle(L"ntdll.dll");
  decltype(NtQueryInformationProcess)* nt_query_information_process =
      reinterpret_cast<decltype(NtQueryInformationProcess)*>(
          GetProcAddress(ntdll, "NtQueryInformationProcess"));
  if (!nt_query_information_process)
    abort();

  PROCESS_BASIC_INFORMATION pbi;
  if (nt_query_information_process(GetCurrentProcess(),
                                   ProcessBasicInformation,
                                   &pbi,
                                   sizeof(pbi),
                                   nullptr) < 0) {
    abort();
  }

  PEB_LDR_DATA* ldr = pbi.PebBaseAddress->Ldr;
  LIST_ENTRY* head = &ldr->InMemoryOrderModuleList;
  LIST_ENTRY* next = head->Flink;
  while (next != head) {
    LDR_DATA_TABLE_ENTRY* entry =
        CONTAINING_RECORD(next, LDR_DATA_TABLE_ENTRY, InMemoryOrderLinks);
    if (UnicodeStringEndsWithCaseInsensitive(entry->FullDllName,
                                             kCorruptableDll)) {
      // Corrupt the pointer to the name.
      entry->FullDllName.Buffer = 0;
    }
    next = next->Flink;
  }

  HANDLE out = GetStdHandle(STD_OUTPUT_HANDLE);
  if (out == INVALID_HANDLE_VALUE)
    abort();
  // We just want any valid address that's known to be code.
  uint64_t code_address = reinterpret_cast<uint64_t>(_ReturnAddress());
  DWORD bytes_written;
  if (!WriteFile(
          out, &code_address, sizeof(code_address), &bytes_written, nullptr) ||
      bytes_written != sizeof(code_address)) {
    abort();
  }

  if (WaitForSingleObject(done_event, INFINITE) != WAIT_OBJECT_0)
    abort();

  CloseHandle(done_event);

  return EXIT_SUCCESS;
}
