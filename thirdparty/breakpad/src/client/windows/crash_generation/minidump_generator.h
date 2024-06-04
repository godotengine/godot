// Copyright (c) 2008, Google Inc.
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

#ifndef CLIENT_WINDOWS_CRASH_GENERATION_MINIDUMP_GENERATOR_H_
#define CLIENT_WINDOWS_CRASH_GENERATION_MINIDUMP_GENERATOR_H_

#include <windows.h>
#include <dbghelp.h>
#include <rpc.h>
#include <list>
#include <string>
#include "google_breakpad/common/minidump_format.h"

namespace google_breakpad {

// Abstraction for various objects and operations needed to generate
// minidump on Windows. This abstraction is useful to hide all the gory
// details for minidump generation and provide a clean interface to
// the clients to generate minidumps.
class MinidumpGenerator {
 public:
  // Creates an instance with the given parameters.
  // is_client_pointers specifies whether the exception_pointers and
  // assert_info point into the process that is being dumped.
  // Before calling WriteMinidump on the returned instance a dump file muct be
  // specified by a call to either SetDumpFile() or GenerateDumpFile().
  // If a full dump file will be requested via a subsequent call to either
  // SetFullDumpFile or GenerateFullDumpFile() dump_type must include
  // MiniDumpWithFullMemory.
  MinidumpGenerator(const std::wstring& dump_path,
                    const HANDLE process_handle,
                    const DWORD process_id,
                    const DWORD thread_id,
                    const DWORD requesting_thread_id,
                    EXCEPTION_POINTERS* exception_pointers,
                    MDRawAssertionInfo* assert_info,
                    const MINIDUMP_TYPE dump_type,
                    const bool is_client_pointers);

  ~MinidumpGenerator();

  void SetDumpFile(const HANDLE dump_file) { dump_file_ = dump_file; }
  void SetFullDumpFile(const HANDLE full_dump_file) {
    full_dump_file_ = full_dump_file;
  }

  // Generate the name for the dump file that will be written to once
  // WriteMinidump() is called. Can only be called once and cannot be called
  // if the dump file is set via SetDumpFile().
  bool GenerateDumpFile(std::wstring* dump_path);

  // Generate the name for the full dump file that will be written to once
  // WriteMinidump() is called. Cannot be called unless the minidump type
  // includes MiniDumpWithFullMemory. Can only be called once and cannot be
  // called if the dump file is set via SetFullDumpFile().
  bool GenerateFullDumpFile(std::wstring* full_dump_path);

  void SetAdditionalStreams(
      MINIDUMP_USER_STREAM_INFORMATION* additional_streams) {
    additional_streams_ = additional_streams;
  }

  void SetCallback(MINIDUMP_CALLBACK_INFORMATION* callback_info) {
    callback_info_ = callback_info;
  }

  // Writes the minidump with the given parameters. Stores the
  // dump file path in the dump_path parameter if dump generation
  // succeeds.
  bool WriteMinidump();

 private:
  // Function pointer type for MiniDumpWriteDump, which is looked up
  // dynamically.
  typedef BOOL (WINAPI* MiniDumpWriteDumpType)(
      HANDLE hProcess,
      DWORD ProcessId,
      HANDLE hFile,
      MINIDUMP_TYPE DumpType,
      CONST PMINIDUMP_EXCEPTION_INFORMATION ExceptionParam,
      CONST PMINIDUMP_USER_STREAM_INFORMATION UserStreamParam,
      CONST PMINIDUMP_CALLBACK_INFORMATION CallbackParam);

  // Function pointer type for UuidCreate, which is looked up dynamically.
  typedef RPC_STATUS (RPC_ENTRY* UuidCreateType)(UUID* Uuid);

  // Loads the appropriate DLL lazily in a thread safe way.
  HMODULE GetDbghelpModule();

  // Loads the appropriate DLL and gets a pointer to the MiniDumpWriteDump
  // function lazily and in a thread-safe manner.
  MiniDumpWriteDumpType GetWriteDump();

  // Loads the appropriate DLL lazily in a thread safe way.
  HMODULE GetRpcrt4Module();

  // Loads the appropriate DLL and gets a pointer to the UuidCreate
  // function lazily and in a thread-safe manner.
  UuidCreateType GetCreateUuid();

  // Returns the path for the file to write dump to.
  bool GenerateDumpFilePath(std::wstring* file_path);

  // Handle to dynamically loaded DbgHelp.dll.
  HMODULE dbghelp_module_;

  // Pointer to the MiniDumpWriteDump function.
  MiniDumpWriteDumpType write_dump_;

  // Handle to dynamically loaded rpcrt4.dll.
  HMODULE rpcrt4_module_;

  // Pointer to the UuidCreate function.
  UuidCreateType create_uuid_;

  // Handle for the process to dump.
  HANDLE process_handle_;

  // Process ID for the process to dump.
  DWORD process_id_;

  // The crashing thread ID.
  DWORD thread_id_;

  // The thread ID which is requesting the dump.
  DWORD requesting_thread_id_;

  // Pointer to the exception information for the crash. This may point to an
  // address in the crashing process so it should not be dereferenced.
  EXCEPTION_POINTERS* exception_pointers_;

  // Assertion info for the report.
  MDRawAssertionInfo* assert_info_;

  // Type of minidump to generate.
  MINIDUMP_TYPE dump_type_;

  // Specifies whether the exception_pointers_ reference memory in the crashing
  // process.
  bool is_client_pointers_;

  // Folder path to store dump files.
  std::wstring dump_path_;

  // UUID used to make dump file names.
  UUID uuid_;
  bool uuid_generated_;

  // The file where the dump will be written.
  HANDLE dump_file_;

  // The file where the full dump will be written.
  HANDLE full_dump_file_;

  // Tracks whether the dump file handle is managed externally.
  bool dump_file_is_internal_;

  // Tracks whether the full dump file handle is managed externally.
  bool full_dump_file_is_internal_;

  // Additional streams to be written to the dump.
  MINIDUMP_USER_STREAM_INFORMATION* additional_streams_;

  // The user defined callback for the various stages of the dump process.
  MINIDUMP_CALLBACK_INFORMATION* callback_info_;

  // Critical section to sychronize action of loading modules dynamically.
  CRITICAL_SECTION module_load_sync_;

  // Critical section to synchronize action of dynamically getting function
  // addresses from modules.
  CRITICAL_SECTION get_proc_address_sync_;
};

}  // namespace google_breakpad

#endif  // CLIENT_WINDOWS_CRASH_GENERATION_MINIDUMP_GENERATOR_H_
