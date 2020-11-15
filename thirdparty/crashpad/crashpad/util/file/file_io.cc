// Copyright 2014 The Crashpad Authors. All rights reserved.
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

#include "util/file/file_io.h"

#include "base/logging.h"
#include "base/macros.h"
#include "base/numerics/safe_conversions.h"

namespace crashpad {

namespace {

class FileIOReadExactly final : public internal::ReadExactlyInternal {
 public:
  explicit FileIOReadExactly(FileHandle file)
      : ReadExactlyInternal(), file_(file) {}
  ~FileIOReadExactly() {}

 private:
  // ReadExactlyInternal:
  FileOperationResult Read(void* buffer, size_t size, bool can_log) override {
    FileOperationResult rv = ReadFile(file_, buffer, size);
    if (rv < 0) {
      PLOG_IF(ERROR, can_log) << internal::kNativeReadFunctionName;
      return -1;
    }
    return rv;
  }

  FileHandle file_;

  DISALLOW_COPY_AND_ASSIGN(FileIOReadExactly);
};

class FileIOWriteAll final : public internal::WriteAllInternal {
 public:
  explicit FileIOWriteAll(FileHandle file) : WriteAllInternal(), file_(file) {}
  ~FileIOWriteAll() {}

 private:
  // WriteAllInternal:
  FileOperationResult Write(const void* buffer, size_t size) override {
    return internal::NativeWriteFile(file_, buffer, size);
  }

  FileHandle file_;

  DISALLOW_COPY_AND_ASSIGN(FileIOWriteAll);
};

}  // namespace

namespace internal {

bool ReadExactlyInternal::ReadExactly(void* buffer, size_t size, bool can_log) {
  char* buffer_c = static_cast<char*>(buffer);
  size_t total_bytes = 0;
  size_t remaining = size;
  while (remaining > 0) {
    FileOperationResult bytes_read = Read(buffer_c, remaining, can_log);
    if (bytes_read < 0) {
      return false;
    }

    DCHECK_LE(static_cast<size_t>(bytes_read), remaining);

    if (bytes_read == 0) {
      break;
    }

    buffer_c += bytes_read;
    remaining -= bytes_read;
    total_bytes += bytes_read;
  }

  if (total_bytes != size) {
    LOG_IF(ERROR, can_log) << "ReadExactly: expected " << size << ", observed "
                           << total_bytes;
    return false;
  }

  return true;
}

bool WriteAllInternal::WriteAll(const void* buffer, size_t size) {
  const char* buffer_c = static_cast<const char*>(buffer);

  while (size > 0) {
    FileOperationResult bytes_written = Write(buffer_c, size);
    if (bytes_written < 0) {
      return false;
    }

    DCHECK_NE(bytes_written, 0);

    buffer_c += bytes_written;
    size -= bytes_written;
  }

  return true;
}

}  // namespace internal

bool ReadFileExactly(FileHandle file, void* buffer, size_t size) {
  FileIOReadExactly read_exactly(file);
  return read_exactly.ReadExactly(buffer, size, false);
}

bool LoggingReadFileExactly(FileHandle file, void* buffer, size_t size) {
  FileIOReadExactly read_exactly(file);
  return read_exactly.ReadExactly(buffer, size, true);
}

bool WriteFile(FileHandle file, const void* buffer, size_t size) {
  FileIOWriteAll write_all(file);
  return write_all.WriteAll(buffer, size);
}

bool LoggingWriteFile(FileHandle file, const void* buffer, size_t size) {
  if (!WriteFile(file, buffer, size)) {
    PLOG(ERROR) << internal::kNativeWriteFunctionName;
    return false;
  }

  return true;
}

void CheckedReadFileExactly(FileHandle file, void* buffer, size_t size) {
  CHECK(LoggingReadFileExactly(file, buffer, size));
}

void CheckedWriteFile(FileHandle file, const void* buffer, size_t size) {
  CHECK(LoggingWriteFile(file, buffer, size));
}

void CheckedReadFileAtEOF(FileHandle file) {
  char c;
  FileOperationResult rv = ReadFile(file, &c, 1);
  if (rv < 0) {
    PCHECK(rv == 0) << internal::kNativeReadFunctionName;
  } else {
    CHECK_EQ(rv, 0) << internal::kNativeReadFunctionName;
  }
}

bool LoggingReadToEOF(FileHandle file, std::string* contents) {
  char buffer[4096];
  FileOperationResult rv;
  std::string local_contents;
  while ((rv = ReadFile(file, buffer, sizeof(buffer))) > 0) {
    DCHECK_LE(static_cast<size_t>(rv), sizeof(buffer));
    local_contents.append(buffer, rv);
  }
  if (rv < 0) {
    PLOG(ERROR) << internal::kNativeReadFunctionName;
    return false;
  }
  contents->swap(local_contents);
  return true;
}

bool LoggingReadEntireFile(const base::FilePath& path, std::string* contents) {
  ScopedFileHandle handle(LoggingOpenFileForRead(path));
  if (!handle.is_valid()) {
    return false;
  }

  return LoggingReadToEOF(handle.get(), contents);
}

void CheckedCloseFile(FileHandle file) {
  CHECK(LoggingCloseFile(file));
}

}  // namespace crashpad
