// Copyright 2008 Google LLC
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

#include "client/windows/crash_generation/minidump_generator.h"

#include <assert.h>
#include <avrfsdk.h>

#include <algorithm>
#include <iterator>
#include <list>
#include <vector>

#include "client/windows/common/auto_critical_section.h"
#include "common/scoped_ptr.h"
#include "common/windows/guid_string.h"

using std::wstring;

namespace {

// A helper class used to collect handle operations data. Unlike
// |MiniDumpWithHandleData| it records the operations for a single handle value
// only, making it possible to include this information to a minidump.
class HandleTraceData {
 public:
  HandleTraceData();
  ~HandleTraceData();

  // Collects the handle operations data and formats a user stream to be added
  // to the minidump.
  bool CollectHandleData(HANDLE process_handle,
                         EXCEPTION_POINTERS* exception_pointers);

  // Fills the user dump entry with a pointer to the collected handle operations
  // data. Returns |true| if the entry was initialized successfully, or |false|
  // if no trace data is available.
  bool GetUserStream(MINIDUMP_USER_STREAM* user_stream);

 private:
  // Reads the exception code from the client process's address space.
  // This routine assumes that the client process's pointer width matches ours.
  static bool ReadExceptionCode(HANDLE process_handle,
                                EXCEPTION_POINTERS* exception_pointers,
                                DWORD* exception_code);

  // Stores handle operations retrieved by VerifierEnumerateResource().
  static ULONG CALLBACK RecordHandleOperations(void* resource_description,
                                               void* enumeration_context,
                                               ULONG* enumeration_level);

  // Function pointer type for VerifierEnumerateResource, which is looked up
  // dynamically.
  typedef BOOL (WINAPI* VerifierEnumerateResourceType)(
      HANDLE Process,
      ULONG Flags,
      ULONG ResourceType,
      AVRF_RESOURCE_ENUMERATE_CALLBACK ResourceCallback,
      PVOID EnumerationContext);

  // Handle to dynamically loaded verifier.dll.
  HMODULE verifier_module_;

  // Pointer to the VerifierEnumerateResource function.
  VerifierEnumerateResourceType enumerate_resource_;

  // Handle value to look for.
  ULONG64 handle_;

  // List of handle operations for |handle_|.
  std::list<AVRF_HANDLE_OPERATION> operations_;

  // Minidump stream data.
  std::vector<char> stream_;
};

HandleTraceData::HandleTraceData()
    : verifier_module_(NULL),
      enumerate_resource_(NULL),
      handle_(NULL) {
}

HandleTraceData::~HandleTraceData() {
  if (verifier_module_) {
    FreeLibrary(verifier_module_);
  }
}

bool HandleTraceData::CollectHandleData(
    HANDLE process_handle,
    EXCEPTION_POINTERS* exception_pointers) {
  DWORD exception_code;
  if (!ReadExceptionCode(process_handle, exception_pointers, &exception_code)) {
    return false;
  }

  // Verify whether the execption is STATUS_INVALID_HANDLE. Do not record any
  // handle information if it is a different exception to keep the minidump
  // small.
  if (exception_code != STATUS_INVALID_HANDLE) {
    return true;
  }

  // Load verifier!VerifierEnumerateResource() dynamically.
  verifier_module_ = LoadLibrary(TEXT("verifier.dll"));
  if (!verifier_module_) {
    return false;
  }

  enumerate_resource_ = reinterpret_cast<VerifierEnumerateResourceType>(
      GetProcAddress(verifier_module_, "VerifierEnumerateResource"));
  if (!enumerate_resource_) {
    return false;
  }

  // STATUS_INVALID_HANDLE does not provide the offending handle value in
  // the exception parameters so we have to guess. At the moment we scan
  // the handle operations trace looking for the last invalid handle operation
  // and record only the operations for that handle value.
  if (enumerate_resource_(process_handle,
                          0,
                          AvrfResourceHandleTrace,
                          &RecordHandleOperations,
                          this) != ERROR_SUCCESS) {
    // The handle tracing must have not been enabled.
    return true;
  }

  // Now that |handle_| is initialized, purge all irrelevant operations.
  std::list<AVRF_HANDLE_OPERATION>::iterator i = operations_.begin();
  std::list<AVRF_HANDLE_OPERATION>::iterator i_end = operations_.end();
  while (i != i_end) {
    if (i->Handle == handle_) {
      ++i;
    } else {
      i = operations_.erase(i);
    }
  }

  // Convert the list of recorded operations to a minidump stream.
  stream_.resize(sizeof(MINIDUMP_HANDLE_OPERATION_LIST) +
      sizeof(AVRF_HANDLE_OPERATION) * operations_.size());

  MINIDUMP_HANDLE_OPERATION_LIST* stream_data =
      reinterpret_cast<MINIDUMP_HANDLE_OPERATION_LIST*>(
          &stream_.front());
  stream_data->SizeOfHeader = sizeof(MINIDUMP_HANDLE_OPERATION_LIST);
  stream_data->SizeOfEntry = sizeof(AVRF_HANDLE_OPERATION);
  stream_data->NumberOfEntries = static_cast<ULONG32>(operations_.size());
  stream_data->Reserved = 0;
  std::copy(operations_.begin(),
            operations_.end(),
#if defined(_MSC_VER) && !defined(_LIBCPP_STD_VER)
            stdext::checked_array_iterator<AVRF_HANDLE_OPERATION*>(
                reinterpret_cast<AVRF_HANDLE_OPERATION*>(stream_data + 1),
                operations_.size())
#else
            reinterpret_cast<AVRF_HANDLE_OPERATION*>(stream_data + 1)
#endif
            );

  return true;
}

bool HandleTraceData::GetUserStream(MINIDUMP_USER_STREAM* user_stream) {
  if (stream_.empty()) {
    return false;
  } else {
    user_stream->Type = HandleOperationListStream;
    user_stream->BufferSize = static_cast<ULONG>(stream_.size());
    user_stream->Buffer = &stream_.front();
    return true;
  }
}

bool HandleTraceData::ReadExceptionCode(
    HANDLE process_handle,
    EXCEPTION_POINTERS* exception_pointers,
    DWORD* exception_code) {
  EXCEPTION_POINTERS pointers;
  if (!ReadProcessMemory(process_handle,
                         exception_pointers,
                         &pointers,
                         sizeof(pointers),
                         NULL)) {
    return false;
  }

  if (!ReadProcessMemory(process_handle,
                         pointers.ExceptionRecord,
                         exception_code,
                         sizeof(*exception_code),
                         NULL)) {
    return false;
  }

  return true;
}

ULONG CALLBACK HandleTraceData::RecordHandleOperations(
    void* resource_description,
    void* enumeration_context,
    ULONG* enumeration_level) {
  AVRF_HANDLE_OPERATION* description =
      reinterpret_cast<AVRF_HANDLE_OPERATION*>(resource_description);
  HandleTraceData* self =
      reinterpret_cast<HandleTraceData*>(enumeration_context);

  // Remember the last invalid handle operation.
  if (description->OperationType == OperationDbBADREF) {
    self->handle_ = description->Handle;
  }

  // Record all handle operations.
  self->operations_.push_back(*description);

  *enumeration_level = HeapEnumerationEverything;
  return ERROR_SUCCESS;
}

}  // namespace

namespace google_breakpad {

MinidumpGenerator::MinidumpGenerator(
    const std::wstring& dump_path,
    const HANDLE process_handle,
    const DWORD process_id,
    const DWORD thread_id,
    const DWORD requesting_thread_id,
    EXCEPTION_POINTERS* exception_pointers,
    MDRawAssertionInfo* assert_info,
    const MINIDUMP_TYPE dump_type,
    const bool is_client_pointers)
    : dbghelp_module_(NULL),
      write_dump_(NULL),
      rpcrt4_module_(NULL),
      create_uuid_(NULL),
      process_handle_(process_handle),
      process_id_(process_id),
      thread_id_(thread_id),
      requesting_thread_id_(requesting_thread_id),
      exception_pointers_(exception_pointers),
      assert_info_(assert_info),
      dump_type_(dump_type),
      is_client_pointers_(is_client_pointers),
      dump_path_(dump_path),
      uuid_generated_(false),
      dump_file_(INVALID_HANDLE_VALUE),
      full_dump_file_(INVALID_HANDLE_VALUE),
      dump_file_is_internal_(false),
      full_dump_file_is_internal_(false),
      additional_streams_(NULL),
      callback_info_(NULL) {
  uuid_ = {0};
  InitializeCriticalSection(&module_load_sync_);
  InitializeCriticalSection(&get_proc_address_sync_);
}

MinidumpGenerator::~MinidumpGenerator() {
  if (dump_file_is_internal_ && dump_file_ != INVALID_HANDLE_VALUE) {
    CloseHandle(dump_file_);
  }

  if (full_dump_file_is_internal_ && full_dump_file_ != INVALID_HANDLE_VALUE) {
    CloseHandle(full_dump_file_);
  }

  if (dbghelp_module_) {
    FreeLibrary(dbghelp_module_);
  }

  if (rpcrt4_module_) {
    FreeLibrary(rpcrt4_module_);
  }

  DeleteCriticalSection(&get_proc_address_sync_);
  DeleteCriticalSection(&module_load_sync_);
}

bool MinidumpGenerator::WriteMinidump() {
  bool full_memory_dump = (dump_type_ & MiniDumpWithFullMemory) != 0;
  if (dump_file_ == INVALID_HANDLE_VALUE ||
      (full_memory_dump && full_dump_file_ == INVALID_HANDLE_VALUE)) {
    return false;
  }

  MiniDumpWriteDumpType write_dump = GetWriteDump();
  if (!write_dump) {
    return false;
  }

  MINIDUMP_EXCEPTION_INFORMATION* dump_exception_pointers = NULL;
  MINIDUMP_EXCEPTION_INFORMATION dump_exception_info;

  // Setup the exception information object only if it's a dump
  // due to an exception.
  if (exception_pointers_) {
    dump_exception_pointers = &dump_exception_info;
    dump_exception_info.ThreadId = thread_id_;
    dump_exception_info.ExceptionPointers = exception_pointers_;
    dump_exception_info.ClientPointers = is_client_pointers_;
  }

  // Add an MDRawBreakpadInfo stream to the minidump, to provide additional
  // information about the exception handler to the Breakpad processor.
  // The information will help the processor determine which threads are
  // relevant. The Breakpad processor does not require this information but
  // can function better with Breakpad-generated dumps when it is present.
  // The native debugger is not harmed by the presence of this information.
  MDRawBreakpadInfo breakpad_info = {0};
  if (!is_client_pointers_) {
    // Set the dump thread id and requesting thread id only in case of
    // in-process dump generation.
    breakpad_info.validity = MD_BREAKPAD_INFO_VALID_DUMP_THREAD_ID |
                             MD_BREAKPAD_INFO_VALID_REQUESTING_THREAD_ID;
    breakpad_info.dump_thread_id = thread_id_;
    breakpad_info.requesting_thread_id = requesting_thread_id_;
  }

  int additional_streams_count = additional_streams_ ?
      additional_streams_->UserStreamCount : 0;
  scoped_array<MINIDUMP_USER_STREAM> user_stream_array(
      new MINIDUMP_USER_STREAM[3 + additional_streams_count]);
  user_stream_array[0].Type = MD_BREAKPAD_INFO_STREAM;
  user_stream_array[0].BufferSize = sizeof(breakpad_info);
  user_stream_array[0].Buffer = &breakpad_info;

  MINIDUMP_USER_STREAM_INFORMATION user_streams;
  user_streams.UserStreamCount = 1;
  user_streams.UserStreamArray = user_stream_array.get();

  MDRawAssertionInfo* actual_assert_info = assert_info_;
  MDRawAssertionInfo client_assert_info = {{0}};

  if (assert_info_) {
    // If the assertion info object lives in the client process,
    // read the memory of the client process.
    if (is_client_pointers_) {
      SIZE_T bytes_read = 0;
      if (!ReadProcessMemory(process_handle_,
                             assert_info_,
                             &client_assert_info,
                             sizeof(client_assert_info),
                             &bytes_read)) {
        if (dump_file_is_internal_)
          CloseHandle(dump_file_);
        if (full_dump_file_is_internal_ &&
            full_dump_file_ != INVALID_HANDLE_VALUE)
          CloseHandle(full_dump_file_);
        return false;
      }

      if (bytes_read != sizeof(client_assert_info)) {
        if (dump_file_is_internal_)
          CloseHandle(dump_file_);
        if (full_dump_file_is_internal_ &&
            full_dump_file_ != INVALID_HANDLE_VALUE)
          CloseHandle(full_dump_file_);
        return false;
      }

      actual_assert_info  = &client_assert_info;
    }

    user_stream_array[1].Type = MD_ASSERTION_INFO_STREAM;
    user_stream_array[1].BufferSize = sizeof(MDRawAssertionInfo);
    user_stream_array[1].Buffer = actual_assert_info;
    ++user_streams.UserStreamCount;
  }

  if (additional_streams_) {
    for (size_t i = 0;
         i < additional_streams_->UserStreamCount;
         i++, user_streams.UserStreamCount++) {
      user_stream_array[user_streams.UserStreamCount].Type =
          additional_streams_->UserStreamArray[i].Type;
      user_stream_array[user_streams.UserStreamCount].BufferSize =
          additional_streams_->UserStreamArray[i].BufferSize;
      user_stream_array[user_streams.UserStreamCount].Buffer =
          additional_streams_->UserStreamArray[i].Buffer;
    }
  }

  // If the process is terminated by STATUS_INVALID_HANDLE exception store
  // the trace of operations for the offending handle value. Do nothing special
  // if the client already requested the handle trace to be stored in the dump.
  HandleTraceData handle_trace_data;
  if (exception_pointers_ && (dump_type_ & MiniDumpWithHandleData) == 0) {
    if (!handle_trace_data.CollectHandleData(process_handle_,
                                             exception_pointers_)) {
      if (dump_file_is_internal_)
        CloseHandle(dump_file_);
      if (full_dump_file_is_internal_ &&
          full_dump_file_ != INVALID_HANDLE_VALUE)
        CloseHandle(full_dump_file_);
      return false;
    }
  }

  bool result_full_memory = true;
  if (full_memory_dump) {
    result_full_memory = write_dump(
        process_handle_,
        process_id_,
        full_dump_file_,
        static_cast<MINIDUMP_TYPE>((dump_type_ & (~MiniDumpNormal))
                                    | MiniDumpWithHandleData),
        dump_exception_pointers,
        &user_streams,
        NULL) != FALSE;
  }

  // Add handle operations trace stream to the minidump if it was collected.
  if (handle_trace_data.GetUserStream(
          &user_stream_array[user_streams.UserStreamCount])) {
    ++user_streams.UserStreamCount;
  }

  bool result_minidump = write_dump(
      process_handle_,
      process_id_,
      dump_file_,
      static_cast<MINIDUMP_TYPE>((dump_type_ & (~MiniDumpWithFullMemory))
                                  | MiniDumpNormal),
      dump_exception_pointers,
      &user_streams,
      callback_info_) != FALSE;

  return result_minidump && result_full_memory;
}

bool MinidumpGenerator::GenerateDumpFile(wstring* dump_path) {
  // The dump file was already set by handle or this function was previously
  // called.
  if (dump_file_ != INVALID_HANDLE_VALUE) {
    return false;
  }

  wstring dump_file_path;
  if (!GenerateDumpFilePath(&dump_file_path)) {
    return false;
  }

  dump_file_ = CreateFile(dump_file_path.c_str(),
                          GENERIC_WRITE,
                          0,
                          NULL,
                          CREATE_NEW,
                          FILE_ATTRIBUTE_NORMAL,
                          NULL);
  if (dump_file_ == INVALID_HANDLE_VALUE) {
    return false;
  }

  dump_file_is_internal_ = true;
  *dump_path = dump_file_path;
  return true;
}

bool MinidumpGenerator::GenerateFullDumpFile(wstring* full_dump_path) {
  // A full minidump was not requested.
  if ((dump_type_ & MiniDumpWithFullMemory) == 0) {
    return false;
  }

  // The dump file was already set by handle or this function was previously
  // called.
  if (full_dump_file_ != INVALID_HANDLE_VALUE) {
    return false;
  }

  wstring full_dump_file_path;
  if (!GenerateDumpFilePath(&full_dump_file_path)) {
    return false;
  }
  full_dump_file_path.resize(full_dump_file_path.size() - 4);  // strip .dmp
  full_dump_file_path.append(TEXT("-full.dmp"));

  full_dump_file_ = CreateFile(full_dump_file_path.c_str(),
                               GENERIC_WRITE,
                               0,
                               NULL,
                               CREATE_NEW,
                               FILE_ATTRIBUTE_NORMAL,
                               NULL);
  if (full_dump_file_ == INVALID_HANDLE_VALUE) {
    return false;
  }

  full_dump_file_is_internal_ = true;
  *full_dump_path = full_dump_file_path;
  return true;
}

HMODULE MinidumpGenerator::GetDbghelpModule() {
  AutoCriticalSection lock(&module_load_sync_);
  if (!dbghelp_module_) {
    dbghelp_module_ = LoadLibrary(TEXT("dbghelp.dll"));
  }

  return dbghelp_module_;
}

MinidumpGenerator::MiniDumpWriteDumpType MinidumpGenerator::GetWriteDump() {
  AutoCriticalSection lock(&get_proc_address_sync_);
  if (!write_dump_) {
    HMODULE module = GetDbghelpModule();
    if (module) {
      FARPROC proc = GetProcAddress(module, "MiniDumpWriteDump");
      write_dump_ = reinterpret_cast<MiniDumpWriteDumpType>(proc);
    }
  }

  return write_dump_;
}

HMODULE MinidumpGenerator::GetRpcrt4Module() {
  AutoCriticalSection lock(&module_load_sync_);
  if (!rpcrt4_module_) {
    rpcrt4_module_ = LoadLibrary(TEXT("rpcrt4.dll"));
  }

  return rpcrt4_module_;
}

MinidumpGenerator::UuidCreateType MinidumpGenerator::GetCreateUuid() {
  AutoCriticalSection lock(&module_load_sync_);
  if (!create_uuid_) {
    HMODULE module = GetRpcrt4Module();
    if (module) {
      FARPROC proc = GetProcAddress(module, "UuidCreate");
      create_uuid_ = reinterpret_cast<UuidCreateType>(proc);
    }
  }

  return create_uuid_;
}

bool MinidumpGenerator::GenerateDumpFilePath(wstring* file_path) {
  if (!uuid_generated_) {
    UuidCreateType create_uuid = GetCreateUuid();
    if (!create_uuid) {
      return false;
    }

    create_uuid(&uuid_);
    uuid_generated_ = true;
  }

  wstring id_str = GUIDString::GUIDToWString(&uuid_);

  *file_path = dump_path_ + TEXT("\\") + id_str + TEXT(".dmp");
  return true;
}

}  // namespace google_breakpad
