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

#include "util/win/nt_internals.h"

#include "base/logging.h"
#include "util/win/get_function.h"

// Declarations that the system headers should provide but don’t.

extern "C" {

NTSTATUS NTAPI NtCreateThreadEx(PHANDLE ThreadHandle,
                                ACCESS_MASK DesiredAccess,
                                POBJECT_ATTRIBUTES ObjectAttributes,
                                HANDLE ProcessHandle,
                                PVOID StartRoutine,
                                PVOID Argument,
                                ULONG CreateFlags,
                                SIZE_T ZeroBits,
                                SIZE_T StackSize,
                                SIZE_T MaximumStackSize,
                                PVOID /*PPS_ATTRIBUTE_LIST*/ AttributeList);

NTSTATUS NTAPI NtOpenThread(HANDLE* ThreadHandle,
                            ACCESS_MASK DesiredAccess,
                            OBJECT_ATTRIBUTES* ObjectAttributes,
                            CLIENT_ID* ClientId);

NTSTATUS NTAPI NtSuspendProcess(HANDLE);

NTSTATUS NTAPI NtResumeProcess(HANDLE);

VOID NTAPI RtlGetUnloadEventTraceEx(PULONG* ElementSize,
                                    PULONG* ElementCount,
                                    PVOID* EventTrace);

}  // extern "C"

namespace crashpad {

NTSTATUS NtClose(HANDLE handle) {
  static const auto nt_close = GET_FUNCTION_REQUIRED(L"ntdll.dll", ::NtClose);
  return nt_close(handle);
}

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
                 PVOID attribute_list) {
  static const auto nt_create_thread_ex =
      GET_FUNCTION_REQUIRED(L"ntdll.dll", ::NtCreateThreadEx);
  return nt_create_thread_ex(thread_handle,
                             desired_access,
                             object_attributes,
                             process_handle,
                             start_routine,
                             argument,
                             create_flags,
                             zero_bits,
                             stack_size,
                             maximum_stack_size,
                             attribute_list);
}

NTSTATUS NtQuerySystemInformation(
    SYSTEM_INFORMATION_CLASS system_information_class,
    PVOID system_information,
    ULONG system_information_length,
    PULONG return_length) {
  static const auto nt_query_system_information =
      GET_FUNCTION_REQUIRED(L"ntdll.dll", ::NtQuerySystemInformation);
  return nt_query_system_information(system_information_class,
                                     system_information,
                                     system_information_length,
                                     return_length);
}

NTSTATUS NtQueryInformationThread(HANDLE thread_handle,
                                  THREADINFOCLASS thread_information_class,
                                  PVOID thread_information,
                                  ULONG thread_information_length,
                                  PULONG return_length) {
  static const auto nt_query_information_thread =
      GET_FUNCTION_REQUIRED(L"ntdll.dll", ::NtQueryInformationThread);
  return nt_query_information_thread(thread_handle,
                                     thread_information_class,
                                     thread_information,
                                     thread_information_length,
                                     return_length);
}

template <class Traits>
NTSTATUS NtOpenThread(PHANDLE thread_handle,
                      ACCESS_MASK desired_access,
                      POBJECT_ATTRIBUTES object_attributes,
                      const process_types::CLIENT_ID<Traits>* client_id) {
  static const auto nt_open_thread =
      GET_FUNCTION_REQUIRED(L"ntdll.dll", ::NtOpenThread);
  return nt_open_thread(
      thread_handle,
      desired_access,
      object_attributes,
      const_cast<CLIENT_ID*>(reinterpret_cast<const CLIENT_ID*>(client_id)));
}

NTSTATUS NtQueryObject(HANDLE handle,
                       OBJECT_INFORMATION_CLASS object_information_class,
                       void* object_information,
                       ULONG object_information_length,
                       ULONG* return_length) {
  static const auto nt_query_object =
      GET_FUNCTION_REQUIRED(L"ntdll.dll", ::NtQueryObject);
  return nt_query_object(handle,
                         object_information_class,
                         object_information,
                         object_information_length,
                         return_length);
}

NTSTATUS NtSuspendProcess(HANDLE handle) {
  static const auto nt_suspend_process =
      GET_FUNCTION_REQUIRED(L"ntdll.dll", ::NtSuspendProcess);
  return nt_suspend_process(handle);
}

NTSTATUS NtResumeProcess(HANDLE handle) {
  static const auto nt_resume_process =
      GET_FUNCTION_REQUIRED(L"ntdll.dll", ::NtResumeProcess);
  return nt_resume_process(handle);
}

void RtlGetUnloadEventTraceEx(ULONG** element_size,
                              ULONG** element_count,
                              void** event_trace) {
  static const auto rtl_get_unload_event_trace_ex =
      GET_FUNCTION_REQUIRED(L"ntdll.dll", ::RtlGetUnloadEventTraceEx);
  rtl_get_unload_event_trace_ex(element_size, element_count, event_trace);
}

// Explicit instantiations with the only 2 valid template arguments to avoid
// putting the body of the function in the header.
template NTSTATUS NtOpenThread<process_types::internal::Traits32>(
    PHANDLE thread_handle,
    ACCESS_MASK desired_access,
    POBJECT_ATTRIBUTES object_attributes,
    const process_types::CLIENT_ID<process_types::internal::Traits32>*
        client_id);

template NTSTATUS NtOpenThread<process_types::internal::Traits64>(
    PHANDLE thread_handle,
    ACCESS_MASK desired_access,
    POBJECT_ATTRIBUTES object_attributes,
    const process_types::CLIENT_ID<process_types::internal::Traits64>*
        client_id);

}  // namespace crashpad
