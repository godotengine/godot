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
#include <sys/types.h>
#include <windows.h>
#include <winternl.h>

#include <map>
#include <string>
#include <type_traits>
#include <vector>

#include "base/files/file_path.h"
#include "base/logging.h"
#include "base/macros.h"
#include "build/build_config.h"
#include "client/crashpad_client.h"
#include "client/crashpad_info.h"
#include "util/win/critical_section_with_debug_info.h"
#include "util/win/get_function.h"

// ntstatus.h conflicts with windows.h so define this locally.
#ifndef STATUS_NO_SUCH_FILE
#define STATUS_NO_SUCH_FILE static_cast<NTSTATUS>(0xC000000F)
#endif

namespace crashpad {

uint32_t* g_extra_memory_pointer;
uint32_t* g_extra_memory_not_saved;

namespace {

CRITICAL_SECTION g_test_critical_section;

unsigned char g_test_memory[] = {
  99, 98, 97, 96, 95, 94, 93, 92, 91, 90,
  89, 88, 87, 86, 85, 84, 83, 82, 81, 80,
};

ULONG RtlNtStatusToDosError(NTSTATUS status) {
  static const auto rtl_nt_status_to_dos_error =
      GET_FUNCTION_REQUIRED(L"ntdll.dll", ::RtlNtStatusToDosError);
  return rtl_nt_status_to_dos_error(status);
}

void AllocateMemoryOfVariousProtections() {
  SYSTEM_INFO system_info;
  GetSystemInfo(&system_info);

  const size_t kPageSize = system_info.dwPageSize;

  static constexpr uint32_t kPageTypes[] = {
      PAGE_NOACCESS,
      PAGE_READONLY,
      PAGE_READWRITE,
      PAGE_EXECUTE,
      PAGE_EXECUTE_READ,
      PAGE_EXECUTE_READWRITE,

      // PAGE_NOACCESS is invalid with PAGE_GUARD.
      PAGE_READONLY | PAGE_GUARD,
      PAGE_READWRITE | PAGE_GUARD,
      PAGE_EXECUTE | PAGE_GUARD,
      PAGE_EXECUTE_READ | PAGE_GUARD,
      PAGE_EXECUTE_READWRITE | PAGE_GUARD,
  };

  // All of these allocations are leaked, we want to view them in windbg via
  // !vprot.
  void* reserve = VirtualAlloc(
      nullptr, arraysize(kPageTypes) * kPageSize, MEM_RESERVE, PAGE_READWRITE);
  PCHECK(reserve) << "VirtualAlloc MEM_RESERVE";
  uintptr_t reserve_as_int = reinterpret_cast<uintptr_t>(reserve);

  for (size_t i = 0; i < arraysize(kPageTypes); ++i) {
    void* result =
        VirtualAlloc(reinterpret_cast<void*>(reserve_as_int + (kPageSize * i)),
                     kPageSize,
                     MEM_COMMIT,
                     kPageTypes[i]);
    PCHECK(result) << "VirtualAlloc MEM_COMMIT " << i;
  }
}

DWORD WINAPI NullThreadProc(void* param) {
  return 0;
}

// Creates a suspended background thread, and sets EDI/RDI to point at
// g_test_memory so we can confirm it's available in the minidump.
bool CreateThreadWithRegisterPointingToTestMemory() {
  HANDLE thread = CreateThread(
      nullptr, 0, &NullThreadProc, nullptr, CREATE_SUSPENDED, nullptr);
  if (!thread) {
    PLOG(ERROR) << "CreateThread";
    return false;
  }

  CONTEXT context = {0};
  context.ContextFlags = CONTEXT_INTEGER;
  if (!GetThreadContext(thread, &context)) {
    PLOG(ERROR) << "GetThreadContext";
    return false;
  }
#if defined(ARCH_CPU_X86_64)
  context.Rdi = reinterpret_cast<DWORD64>(g_test_memory);
#elif defined(ARCH_CPU_X86)
  context.Edi = reinterpret_cast<DWORD>(g_test_memory);
#endif
  if (!SetThreadContext(thread, &context)) {
    PLOG(ERROR) << "SetThreadContext";
    return false;
  }

  return true;
}

void SomeCrashyFunction() {
  // SetLastError and NTSTATUS so that we have something to view in !gle in
  // windbg. RtlNtStatusToDosError() stores STATUS_NO_SUCH_FILE into the
  // LastStatusError of the TEB as a side-effect, and we'll be setting
  // ERROR_FILE_NOT_FOUND for GetLastError().
  SetLastError(RtlNtStatusToDosError(STATUS_NO_SUCH_FILE));

  volatile int* foo = reinterpret_cast<volatile int*>(7);
  *foo = 42;
}

void AllocateExtraMemoryToBeSaved(
    crashpad::SimpleAddressRangeBag* extra_ranges) {
  constexpr size_t kNumVals = 2000;
  auto extra_memory = new uint32_t[kNumVals];
  g_extra_memory_pointer = extra_memory;
  for (size_t i = 0; i < kNumVals; ++i)
    extra_memory[i] =
        static_cast<std::remove_reference<decltype(extra_memory[0])>::type>(
            i * 13 + 2);
  extra_ranges->Insert(extra_memory, sizeof(extra_memory[0]) * kNumVals);
  extra_ranges->Insert(&g_extra_memory_pointer, sizeof(g_extra_memory_pointer));
}

void AllocateExtraUnsavedMemory(crashpad::SimpleAddressRangeBag* extra_ranges) {
  // Allocate some extra memory, and then Insert() but also Remove() it so we
  // can confirm it doesn't get saved.
  constexpr size_t kNumVals = 2000;
  auto extra_memory = new uint32_t[kNumVals];
  g_extra_memory_not_saved = extra_memory;
  for (size_t i = 0; i < kNumVals; ++i)
    extra_memory[i] =
        static_cast<std::remove_reference<decltype(extra_memory[0])>::type>(
            i * 17 + 7);
  extra_ranges->Insert(extra_memory, sizeof(extra_memory[0]) * kNumVals);
  extra_ranges->Insert(&g_extra_memory_not_saved,
                       sizeof(g_extra_memory_not_saved));

  // We keep the pointer's memory, but remove the pointed-to memory.
  extra_ranges->Remove(extra_memory, sizeof(extra_memory[0]) * kNumVals);
}

int CrashyMain(int argc, wchar_t* argv[]) {
  CrashpadClient client;

  if (argc == 2) {
    if (!client.SetHandlerIPCPipe(argv[1])) {
      LOG(ERROR) << "SetHandler";
      return EXIT_FAILURE;
    }
  } else if (argc == 3) {
    if (!client.StartHandler(base::FilePath(argv[1]),
                             base::FilePath(argv[2]),
                             base::FilePath(),
                             std::string(),
                             std::map<std::string, std::string>(),
                             std::vector<std::string>(),
                             false,
                             true)) {
      LOG(ERROR) << "StartHandler";
      return EXIT_FAILURE;
    }
  } else {
    fprintf(stderr, "Usage: %ls <server_pipe_name>\n", argv[0]);
    fprintf(stderr, "       %ls <handler_path> <database_path>\n", argv[0]);
    return EXIT_FAILURE;
  }

  crashpad::SimpleAddressRangeBag* extra_ranges =
      new crashpad::SimpleAddressRangeBag();
  crashpad::CrashpadInfo::GetCrashpadInfo()->set_extra_memory_ranges(
      extra_ranges);
  AllocateExtraMemoryToBeSaved(extra_ranges);
  AllocateExtraUnsavedMemory(extra_ranges);

  // Load and unload some uncommonly used modules so we can see them in the list
  // reported by `lm`. At least two so that we confirm we got the element size
  // advancement of RTL_UNLOAD_EVENT_TRACE correct.
  CHECK(GetModuleHandle(L"lz32.dll") == nullptr);
  CHECK(GetModuleHandle(L"wmerror.dll") == nullptr);
  HMODULE lz32 = LoadLibrary(L"lz32.dll");
  HMODULE wmerror = LoadLibrary(L"wmerror.dll");
  FreeLibrary(lz32);
  FreeLibrary(wmerror);

  // Make sure data pointed to by the stack is captured.
  constexpr int kDataSize = 512;
  int* pointed_to_data = new int[kDataSize];
  for (int i = 0; i < kDataSize; ++i)
    pointed_to_data[i] = i | ((i % 2 == 0) ? 0x80000000 : 0);
  int* offset_pointer = &pointed_to_data[128];
  // Encourage the compiler to keep this variable around.
  printf("%p, %p\n", offset_pointer, &offset_pointer);

  crashpad::CrashpadInfo::GetCrashpadInfo()
      ->set_gather_indirectly_referenced_memory(
          TriState::kEnabled, std::numeric_limits<uint32_t>::max());

  std::vector<uint8_t> data_stream1(128, 'x');
  crashpad::CrashpadInfo::GetCrashpadInfo()->AddUserDataMinidumpStream(
      222222,
      reinterpret_cast<const void*>(data_stream1.data()),
      data_stream1.size());

  std::vector<uint8_t> data_stream2(4096, 'z');
  crashpad::CrashpadInfo::GetCrashpadInfo()->AddUserDataMinidumpStream(
      333333,
      reinterpret_cast<const void*>(data_stream2.data()),
      data_stream2.size());

  AllocateMemoryOfVariousProtections();

  if (InitializeCriticalSectionWithDebugInfoIfPossible(
          &g_test_critical_section)) {
    EnterCriticalSection(&g_test_critical_section);
  }

  if (!CreateThreadWithRegisterPointingToTestMemory())
    return EXIT_FAILURE;

  SomeCrashyFunction();

  return EXIT_SUCCESS;
}

}  // namespace
}  // namespace crashpad

int wmain(int argc, wchar_t* argv[]) {
  return crashpad::CrashyMain(argc, argv);
}
