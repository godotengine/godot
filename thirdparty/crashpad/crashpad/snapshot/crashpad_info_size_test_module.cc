// Copyright 2017 The Crashpad Authors. All rights reserved.
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

#include <stdint.h>

#include "build/build_config.h"

#if defined(OS_MACOSX)
#include <mach-o/loader.h>
#elif defined(OS_WIN)
#include <windows.h>
#endif  // OS_MACOSX

namespace crashpad {

#if defined(CRASHPAD_INFO_SIZE_TEST_MODULE_SMALL) == \
    defined(CRASHPAD_INFO_SIZE_TEST_MODULE_LARGE)
#error Define exactly one of these macros
#endif

// This module contains a CrashpadInfo structure that’s either smaller or larger
// than the one defined in the client library, depending on which macro is
// defined when it’s compiled. This tests the snapshot layer’s ability to read
// smaller structures (as might be found in modules built with older versions of
// the client library than a handler’s snapshot library) and larger ones (the
// “vice-versa” situation). This needs to be done without taking a dependency on
// the client library, which would bring with it a correct copy of the
// CrashpadInfo structure. As a result, all types have been simplified to
// fixed-size integers and void* pointers.
struct TestCrashpadInfo {
  uint32_t signature_;
  uint32_t size_;
  uint32_t version_;
  uint32_t indirectly_referenced_memory_cap_;
  uint32_t padding_0_;
  uint8_t crashpad_handler_behavior_;
  uint8_t system_crash_reporter_forwarding_;
  uint8_t gather_indirectly_referenced_memory_;
  uint8_t padding_1_;
  void* extra_memory_ranges_;
  void* simple_annotations_;
#if !defined(CRASHPAD_INFO_SIZE_TEST_MODULE_SMALL)
  void* user_data_minidump_stream_head_;
  void* annotations_list_;
#endif  // CRASHPAD_INFO_SIZE_TEST_MODULE_SMALL
#if defined(CRASHPAD_INFO_SIZE_TEST_MODULE_LARGE)
  uint8_t trailer_[64 * 1024];
#endif  // CRASHPAD_INFO_SIZE_TEST_MODULE_LARGE
};

// Put it in the correct section.
//
// The initializer also duplicates constants from the client library, sufficient
// to get this test version to be interpreted as a genuine CrashpadInfo
// structure. The size is set to the actual size of this structure (that’s kind
// of the point of this test).
#if defined(OS_POSIX)
__attribute__((
#if defined(OS_MACOSX)
    section(SEG_DATA ",crashpad_info"),
#endif
#if defined(ADDRESS_SANITIZER)
    aligned(64),
#endif  // defined(ADDRESS_SANITIZER)
    visibility("hidden"),
    used))
#elif defined(OS_WIN)
#pragma section("CPADinfo", read, write)
__declspec(allocate("CPADinfo"))
#else  // !defined(OS_POSIX) && !defined(OS_WIN)
#error Port
#endif  // !defined(OS_POSIX) && !defined(OS_WIN)
TestCrashpadInfo g_test_crashpad_info = {'CPad',
                                         sizeof(TestCrashpadInfo),
                                         1,
                                         0,
                                         0,
                                         0,
                                         0,
                                         0,
                                         0,
                                         nullptr,
                                         nullptr,
#if !defined(CRASHPAD_INFO_SIZE_TEST_MODULE_SMALL)
                                         nullptr,
                                         nullptr,
#endif  // CRASHPAD_INFO_SIZE_TEST_MODULE_SMALL
#if defined(CRASHPAD_INFO_SIZE_TEST_MODULE_LARGE)
                                         {}
#endif  // CRASHPAD_INFO_SIZE_TEST_MODULE_LARGE
};

}  // namespace crashpad

extern "C" {

#if defined(OS_POSIX)
__attribute__((visibility("default")))
#elif defined(OS_WIN)
__declspec(dllexport)
#else
#error Port
#endif  // OS_POSIX
crashpad::TestCrashpadInfo* TestModule_GetCrashpadInfo() {
  // Note that there's no need to do the back-reference here to the note on
  // POSIX like CrashpadInfo::GetCrashpadInfo() because the note .S file is
  // directly included into this test binary.
  return &crashpad::g_test_crashpad_info;
}

}  // extern "C"

#if defined(OS_WIN)
BOOL WINAPI DllMain(HINSTANCE hinstance, DWORD reason, LPVOID reserved) {
  return TRUE;
}
#endif  // OS_WIN
