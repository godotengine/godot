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

#include "util/linux/ptrace_broker.h"

#include <fcntl.h>
#include <limits.h>
#include <string.h>
#include <syscall.h>
#include <unistd.h>

#include <algorithm>

#include "base/logging.h"
#include "base/posix/eintr_wrapper.h"

namespace crashpad {

namespace {

size_t FormatPID(char* buffer, pid_t pid) {
  DCHECK_GE(pid, 0);

  char pid_buf[16];
  size_t length = 0;
  do {
    DCHECK_LT(length, sizeof(pid_buf));

    pid_buf[length] = '0' + pid % 10;
    pid /= 10;
    ++length;
  } while (pid > 0);

  for (size_t index = 0; index < length; ++index) {
    buffer[index] = pid_buf[length - index - 1];
  }

  return length;
}

}  // namespace

PtraceBroker::PtraceBroker(int sock, pid_t pid, bool is_64_bit)
    : ptracer_(is_64_bit, /* can_log= */ false),
      file_root_(file_root_buffer_),
      attachments_(nullptr),
      attach_count_(0),
      attach_capacity_(0),
      memory_file_(),
      sock_(sock),
      memory_pid_(pid),
      tried_opening_mem_file_(false) {
  AllocateAttachments();

  static constexpr char kProc[] = "/proc/";
  size_t root_length = strlen(kProc);
  memcpy(file_root_buffer_, kProc, root_length);

  if (pid >= 0) {
    root_length += FormatPID(file_root_buffer_ + root_length, pid);
    DCHECK_LT(root_length, sizeof(file_root_buffer_));
    file_root_buffer_[root_length] = '/';
    ++root_length;
  }

  DCHECK_LT(root_length, sizeof(file_root_buffer_));
  file_root_buffer_[root_length] = '\0';
}

PtraceBroker::~PtraceBroker() = default;

void PtraceBroker::SetFileRoot(const char* new_root) {
  DCHECK_EQ(new_root[strlen(new_root) - 1], '/');
  memory_pid_ = -1;
  file_root_ = new_root;
}

int PtraceBroker::Run() {
  int result = RunImpl();
  ReleaseAttachments();
  return result;
}

bool PtraceBroker::AllocateAttachments() {
  constexpr size_t page_size = 4096;
  constexpr size_t alloc_size =
      (sizeof(ScopedPtraceAttach) + page_size - 1) & ~(page_size - 1);
  void* alloc = sbrk(alloc_size);
  if (reinterpret_cast<intptr_t>(alloc) == -1) {
    return false;
  }

  if (attachments_ == nullptr) {
    attachments_ = reinterpret_cast<ScopedPtraceAttach*>(alloc);
  }

  attach_capacity_ += alloc_size / sizeof(ScopedPtraceAttach);
  return true;
}

void PtraceBroker::ReleaseAttachments() {
  for (size_t index = 0; index < attach_count_; ++index) {
    attachments_[index].Reset();
  }
}

int PtraceBroker::RunImpl() {
  while (true) {
    Request request = {};
    if (!ReadFileExactly(sock_, &request, sizeof(request))) {
      return errno;
    }

    if (request.version != Request::kVersion) {
      return EINVAL;
    }

    switch (request.type) {
      case Request::kTypeAttach: {
        ScopedPtraceAttach* attach;
        ScopedPtraceAttach stack_attach;
        bool attach_on_stack = false;

        if (attach_capacity_ > attach_count_ || AllocateAttachments()) {
          attach = new (&attachments_[attach_count_]) ScopedPtraceAttach;
        } else {
          attach = &stack_attach;
          attach_on_stack = true;
        }

        Bool status = kBoolFalse;
        if (attach->ResetAttach(request.tid)) {
          status = kBoolTrue;
          if (!attach_on_stack) {
            ++attach_count_;
          }
        }

        if (!WriteFile(sock_, &status, sizeof(status))) {
          return errno;
        }

        if (status == kBoolFalse) {
          Errno error = errno;
          if (!WriteFile(sock_, &error, sizeof(error))) {
            return errno;
          }
        }

        if (attach_on_stack && status == kBoolTrue) {
          return RunImpl();
        }
        continue;
      }

      case Request::kTypeIs64Bit: {
        Bool is_64_bit = ptracer_.Is64Bit() ? kBoolTrue : kBoolFalse;
        if (!WriteFile(sock_, &is_64_bit, sizeof(is_64_bit))) {
          return errno;
        }
        continue;
      }

      case Request::kTypeGetThreadInfo: {
        GetThreadInfoResponse response;
        response.success = ptracer_.GetThreadInfo(request.tid, &response.info)
                               ? kBoolTrue
                               : kBoolFalse;

        if (!WriteFile(sock_, &response, sizeof(response))) {
          return errno;
        }

        if (response.success == kBoolFalse) {
          Errno error = errno;
          if (!WriteFile(sock_, &error, sizeof(error))) {
            return errno;
          }
        }
        continue;
      }

      case Request::kTypeReadFile: {
        ScopedFileHandle handle;
        int result = ReceiveAndOpenFilePath(request.path.path_length,
                                            /* is_directory= */ false,
                                            &handle);
        if (result != 0) {
          return result;
        }

        if (!handle.is_valid()) {
          continue;
        }

        result = SendFileContents(handle.get());
        if (result != 0) {
          return result;
        }
        continue;
      }

      case Request::kTypeReadMemory: {
        int result =
            SendMemory(request.tid, request.iov.base, request.iov.size);
        if (result != 0) {
          return result;
        }
        continue;
      }

      case Request::kTypeListDirectory: {
        ScopedFileHandle handle;
        int result = ReceiveAndOpenFilePath(request.path.path_length,
                                            /* is_directory= */ true,
                                            &handle);
        if (result != 0) {
          return result;
        }

        if (!handle.is_valid()) {
          continue;
        }

        result = SendDirectory(handle.get());
        if (result != 0) {
          return result;
        }
        continue;
      }

      case Request::kTypeExit:
        return 0;
    }

    DCHECK(false);
    return EINVAL;
  }
}

int PtraceBroker::SendError(Errno err) {
  return WriteFile(sock_, &err, sizeof(err)) ? 0 : errno;
}

int PtraceBroker::SendReadError(ReadError error) {
  int32_t rv = -1;
  return WriteFile(sock_, &rv, sizeof(rv)) &&
                 WriteFile(sock_, &error, sizeof(error))
             ? 0
             : errno;
}

int PtraceBroker::SendOpenResult(OpenResult result) {
  return WriteFile(sock_, &result, sizeof(result)) ? 0 : errno;
}

int PtraceBroker::SendFileContents(FileHandle handle) {
  char buffer[4096];
  int32_t rv;
  do {
    rv = ReadFile(handle, buffer, sizeof(buffer));

    if (rv < 0) {
      return SendReadError(static_cast<ReadError>(errno));
    }

    if (!WriteFile(sock_, &rv, sizeof(rv))) {
      return errno;
    }

    if (rv > 0) {
      if (!WriteFile(sock_, buffer, static_cast<size_t>(rv))) {
        return errno;
      }
    }
  } while (rv > 0);

  return 0;
}

void PtraceBroker::TryOpeningMemFile() {
  if (tried_opening_mem_file_) {
    return;
  }
  tried_opening_mem_file_ = true;

  if (memory_pid_ < 0) {
    return;
  }

  char mem_path[32];
  size_t root_length = strlen(file_root_buffer_);
  static constexpr char kMem[] = "mem";

  DCHECK_LT(root_length + strlen(kMem) + 1, sizeof(mem_path));
  memcpy(mem_path, file_root_buffer_, root_length);
  // Include the trailing NUL.
  memcpy(mem_path + root_length, kMem, strlen(kMem) + 1);
  memory_file_.reset(
      HANDLE_EINTR(open(mem_path, O_RDONLY | O_CLOEXEC | O_NOCTTY)));
}

int PtraceBroker::SendMemory(pid_t pid, VMAddress address, VMSize size) {
  if (memory_pid_ >= 0 && pid != memory_pid_) {
    return SendReadError(kReadErrorAccessDenied);
  }

  TryOpeningMemFile();
  auto read_memory = [this, pid](VMAddress address, size_t size, char* buffer) {
    return this->memory_file_.is_valid()
               ? HANDLE_EINTR(
                     pread64(this->memory_file_.get(), buffer, size, address))
               : this->ptracer_.ReadUpTo(pid, address, size, buffer);
  };

  char buffer[4096];
  while (size > 0) {
    size_t to_read = std::min(size, VMSize{sizeof(buffer)});

    int32_t bytes_read = read_memory(address, to_read, buffer);

    if (bytes_read < 0) {
      return SendReadError(static_cast<ReadError>(errno));
    }

    if (!WriteFile(sock_, &bytes_read, sizeof(bytes_read))) {
      return errno;
    }

    if (bytes_read == 0) {
      return 0;
    }

    if (!WriteFile(sock_, buffer, bytes_read)) {
      return errno;
    }

    size -= bytes_read;
    address += bytes_read;
  }
  return 0;
}

int PtraceBroker::SendDirectory(FileHandle handle) {
  char buffer[4096];
  int rv;
  do {
    rv = syscall(SYS_getdents64, handle, buffer, sizeof(buffer));

    if (rv < 0) {
      return SendReadError(static_cast<ReadError>(errno));
    }

    if (!WriteFile(sock_, &rv, sizeof(rv))) {
      return errno;
    }

    if (rv > 0) {
      if (!WriteFile(sock_, buffer, static_cast<size_t>(rv))) {
        return errno;
      }
    }
  } while (rv > 0);

  return 0;
}

int PtraceBroker::ReceiveAndOpenFilePath(VMSize path_length,
                                         bool is_directory,
                                         ScopedFileHandle* handle) {
  char path[std::max(4096, PATH_MAX)];

  if (path_length >= sizeof(path)) {
    return SendOpenResult(kOpenResultTooLong);
  }

  if (!ReadFileExactly(sock_, path, path_length)) {
    return errno;
  }
  path[path_length] = '\0';

  if (strncmp(path, file_root_, strlen(file_root_)) != 0) {
    return SendOpenResult(kOpenResultAccessDenied);
  }

  int flags = O_RDONLY | O_CLOEXEC | O_NOCTTY;
  if (is_directory) {
    flags |= O_DIRECTORY;
  }
  ScopedFileHandle local_handle(HANDLE_EINTR(open(path, flags)));
  if (!local_handle.is_valid()) {
    return SendOpenResult(static_cast<OpenResult>(errno));
  }

  handle->reset(local_handle.release());
  return SendOpenResult(kOpenResultSuccess);
}

}  // namespace crashpad
