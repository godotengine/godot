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

#ifndef CRASHPAD_UTIL_WIN_NT_INTERNALS_H_
#define CRASHPAD_UTIL_WIN_NT_INTERNALS_H_

#include <windows.h>
#include <winternl.h>

#include "util/win/process_structs.h"

// Copied from ntstatus.h because um/winnt.h conflicts with general inclusion of
// ntstatus.h.
#define STATUS_INFO_LENGTH_MISMATCH ((NTSTATUS)0xC0000004L)
#define STATUS_BUFFER_TOO_SMALL ((NTSTATUS)0xC0000023L)
#define STATUS_PROCESS_IS_TERMINATING ((NTSTATUS)0xC000010AL)

namespace crashpad {

NTSTATUS NtClose(HANDLE handle);

// http://processhacker.sourceforge.net/doc/ntpsapi_8h_source.html
#define THREAD_CREATE_FLAGS_SKIP_THREAD_ATTACH 0x00000002
NTSTATUS
NtCreateThreadEx(PHANDLE thread_handle,
                 ACCESS_MASK desired_access,
                 POBJECT_ATTRIBUTES object_attributes,
                 HANDLE process_handle,
                 PVOID start_routine,
                 PVOID argument,
                 ULONG create_flags,
                 SIZE_T zero_bits,
                 SIZE_T stack_size,
                 SIZE_T maximum_stack_size,
                 PVOID /*PPS_ATTRIBUTE_LIST*/ attribute_list);

// winternal.h defines THREADINFOCLASS, but not all members.
enum { ThreadBasicInformation = 0 };

// winternal.h defines SYSTEM_INFORMATION_CLASS, but not all members.
enum { SystemExtendedHandleInformation = 64 };

NTSTATUS NtQuerySystemInformation(
    SYSTEM_INFORMATION_CLASS system_information_class,
    PVOID system_information,
    ULONG system_information_length,
    PULONG return_length);

NTSTATUS NtQueryInformationThread(HANDLE thread_handle,
                                  THREADINFOCLASS thread_information_class,
                                  PVOID thread_information,
                                  ULONG thread_information_length,
                                  PULONG return_length);

template <class Traits>
NTSTATUS NtOpenThread(PHANDLE thread_handle,
                      ACCESS_MASK desired_access,
                      POBJECT_ATTRIBUTES object_attributes,
                      const process_types::CLIENT_ID<Traits>* client_id);

NTSTATUS NtQueryObject(HANDLE handle,
                       OBJECT_INFORMATION_CLASS object_information_class,
                       void* object_information,
                       ULONG object_information_length,
                       ULONG* return_length);

NTSTATUS NtSuspendProcess(HANDLE handle);

NTSTATUS NtResumeProcess(HANDLE handle);

// From https://msdn.microsoft.com/library/cc678403.aspx.
template <class Traits>
struct RTL_UNLOAD_EVENT_TRACE {
  typename Traits::Pointer BaseAddress;
  typename Traits::UnsignedIntegral SizeOfImage;
  ULONG Sequence;
  ULONG TimeDateStamp;
  ULONG CheckSum;
  WCHAR ImageName[32];
};

void RtlGetUnloadEventTraceEx(ULONG** element_size,
                              ULONG** element_count,
                              void** event_trace);

}  // namespace crashpad

#endif  // CRASHPAD_UTIL_WIN_NT_INTERNALS_H_
