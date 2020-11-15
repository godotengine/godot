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

#ifndef CRASHPAD_UTIL_LINUX_PTRACE_BROKER_H_
#define CRASHPAD_UTIL_LINUX_PTRACE_BROKER_H_

#include <errno.h>
#include <stdint.h>
#include <sys/types.h>

#include "base/macros.h"
#include "util/file/file_io.h"
#include "util/linux/exception_handler_protocol.h"
#include "util/linux/ptrace_connection.h"
#include "util/linux/ptracer.h"
#include "util/linux/scoped_ptrace_attach.h"
#include "util/linux/thread_info.h"
#include "util/misc/address_types.h"

namespace crashpad {

//! \brief Implements a PtraceConnection over a socket.
//!
//! This class is the server half of the connection. The broker should be run
//! in a process with `ptrace` capabilities for the target process and may run
//! in a compromised context.
class PtraceBroker {
 public:
#pragma pack(push, 1)
  //! \brief A request sent to a PtraceBroker from a PtraceClient.
  struct Request {
    static constexpr uint16_t kVersion = 1;

    //! \brief The version number for this Request.
    uint16_t version = kVersion;

    //! \brief The type of request to serve.
    enum Type : uint16_t {
      //! \brief `ptrace`-attach the specified thread ID. Responds with
      //!     kBoolTrue on success, otherwise kBoolFalse, followed by an Errno.
      kTypeAttach,

      //! \brief Responds with kBoolTrue if the target process is 64-bit.
      //!     Otherwise, kBoolFalse.
      kTypeIs64Bit,

      //! \brief Responds with a GetThreadInfoResponse containing a ThreadInfo
      //!     for the specified thread ID. If an error occurs,
      //!     GetThreadInfoResponse::success is set to kBoolFalse and is
      //!     followed by an Errno.
      kTypeGetThreadInfo,

      //! \brief Reads memory from the attached process. The data is returned in
      //!     a series of messages. Each message begins with an int32_t
      //!     indicating the number of bytes read, 0 for end-of-file, or -1 for
      //!     errors, followed by a ReadError. On success the bytes read follow.
      kTypeReadMemory,

      //! \brief Read a file's contents. The data is returned in a series of
      //!     messages. The first message is an OpenResult, indicating the
      //!     validity of the received file path. If the OpenResult is
      //!     kOpenResultSuccess, each subsequent message begins with an int32_t
      //!     indicating the number of bytes read, 0 for end-of-file, or -1 for
      //!     errors, followed by an Errno. On success, the bytes read follow.
      kTypeReadFile,

      //! \brief Reads the contents of a directory. The data is returned in a
      //!     series of messages. The first message is an OpenResult, indicating
      //!     the validity of the received file path. If the OpenResult is
      //!     kOpenResultSuccess, the subsequent messages return the contents of
      //!     the directory as a dirent stream, as read by `getdents64()`. Each
      //!     subsequent message begins with an int32_t indicating the number of
      //!     bytes read, 0 for end-of-file, or -1 for errors, followed by an
      //!     Errno. On success, the bytes read follow.
      kTypeListDirectory,

      //! \brief Causes the broker to return from Run(), detaching all attached
      //!     threads. Does not respond.
      kTypeExit
    } type;

    //! \brief The thread ID associated with this request. Valid for kTypeAttach,
    //!     kTypeGetThreadInfo, and kTypeReadMemory.
    pid_t tid;

    union {
      //! \brief Specifies the memory region to read for a kTypeReadMemory
      //! request.
      struct {
        //! \brief The base address of the memory region.
        VMAddress base;

        //! \brief The size of the memory region.
        VMSize size;
      } iov;

      //! \brief Specifies the file path to read for a kTypeReadFile request.
      struct {
        //! \brief The number of bytes in #path. The path should not include a
        //!     `NUL`-terminator.
        VMSize path_length;

        //! \brief The file path to read.
        char path[];
      } path;
    };
  };

  //! \brief A result used in operations that accept paths.
  //!
  //! Positive values of this enum are reserved for sending errno values.
  enum OpenResult : int32_t {
    //! \brief Access to the path is denied.
    kOpenResultAccessDenied = -2,

    //! \brief The path name is too long.
    kOpenResultTooLong = -1,

    //! \brief The file was successfully opened.
    kOpenResultSuccess = 0,
  };

  //! \brief A result used in operations that read from memory or files.
  //!
  //! Positive values of this enum are reserved for sending errno values.
  enum ReadError : int32_t {
    //! \brief Access to this data is denied.
    kReadErrorAccessDenied = -1,
  };

  //! \brief The response sent for a Request with type kTypeGetThreadInfo.
  struct GetThreadInfoResponse {
    //! \brief Information about the specified thread. Only valid if #success
    //!     is kBoolTrue.
    ThreadInfo info;

    //! \brief Specifies the success or failure of this call.
    Bool success;
  };
#pragma pack(pop)

  //! \brief Constructs this object.
  //!
  //! \param[in] sock A socket on which to read requests from a connected
  //!     PtraceClient. Does not take ownership of the socket.
  //! \param[in] pid The process ID of the process the broker is expected to
  //!     trace. Setting this value exends the default file root to
  //!     "/proc/[pid]/" and enables memory reading via /proc/[pid]/mem. The
  //!     broker will deny any requests to read memory from processes whose
  //!     processID is not \a pid. If pid is -1, the broker will serve requests
  //!     to read memory from any process it is able to via `ptrace PEEKDATA`.
  //! \param[in] is_64_bit Whether this broker should be configured to trace a
  //!     64-bit process.
  PtraceBroker(int sock, pid_t pid, bool is_64_bit);

  ~PtraceBroker();

  //! \brief Restricts the broker to serving the contents of files under \a
  //!     root.
  //!
  //! If this method is not called, the broker defaults to only serving files
  //! under "/proc/" or "/proc/[pid]/" if a pid was set.
  //!
  //! Calling this function disables reading from a memory file if one has not
  //! already been opened.
  //!
  //! \param[in] root A NUL-terminated c-string containing the path to the new
  //!     root. \a root must not be `nullptr`, must end in a '/', and the caller
  //!     should ensure that \a root remains valid for the lifetime of the
  //!     broker.
  void SetFileRoot(const char* root);

  //! \brief Begin serving requests on the configured socket.
  //!
  //! This method returns when a PtraceBrokerRequest with type kTypeExit is
  //! received or an error is encountered on the socket.
  //!
  //! This method calls `sbrk`, which may break other memory management tools,
  //! such as `malloc`.
  //!
  //! \return 0 if Run() exited due to an exit request. Otherwise an error code.
  int Run();

 private:
  bool AllocateAttachments();
  void ReleaseAttachments();
  int RunImpl();
  int SendError(Errno err);
  int SendReadError(ReadError err);
  int SendOpenResult(OpenResult result);
  int SendFileContents(FileHandle handle);
  int SendDirectory(FileHandle handle);
  void TryOpeningMemFile();
  int SendMemory(pid_t pid, VMAddress address, VMSize size);
  int ReceiveAndOpenFilePath(VMSize path_length,
                             bool is_directory,
                             ScopedFileHandle* handle);

  char file_root_buffer_[32];
  Ptracer ptracer_;
  const char* file_root_;
  ScopedPtraceAttach* attachments_;
  size_t attach_count_;
  size_t attach_capacity_;
  ScopedFileHandle memory_file_;
  int sock_;
  pid_t memory_pid_;
  bool tried_opening_mem_file_;

  DISALLOW_COPY_AND_ASSIGN(PtraceBroker);
};

}  // namespace crashpad

#endif  // CRASHPAD_UTIL_LINUX_PTRACE_BROKER_H_
