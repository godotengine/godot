// Copyright 2006 Google LLC
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

// minidump_file_writer.cc: Minidump file writer implementation.
//
// See minidump_file_writer.h for documentation.

#ifdef HAVE_CONFIG_H
#include <config.h>  // Must come first
#endif

#include <fcntl.h>
#include <limits.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>

#include "client/minidump_file_writer-inl.h"
#include "common/linux/linux_libc_support.h"
#include "common/string_conversion.h"
#if defined(__linux__) && __linux__
#include "third_party/lss/linux_syscall_support.h"
#endif

#if defined(__ANDROID__)
#include <errno.h>

namespace {

bool g_need_ftruncate_workaround = false;
bool g_checked_need_ftruncate_workaround = false;

void CheckNeedsFTruncateWorkAround(int file) {
  if (g_checked_need_ftruncate_workaround) {
    return;
  }
  g_checked_need_ftruncate_workaround = true;

  // Attempt an idempotent truncate that chops off nothing and see if we
  // run into any sort of errors.
  off_t offset = sys_lseek(file, 0, SEEK_END);
  if (offset == -1) {
    // lseek failed. Don't apply work around. It's unlikely that we can write
    // to a minidump with either method.
    return;
  }

  int result = ftruncate(file, offset);
  if (result == -1 && errno == EACCES) {
    // It very likely that we are running into the kernel bug in M devices.
    // We are going to deploy the workaround for writing minidump files
    // without uses of ftruncate(). This workaround should be fine even
    // for kernels without the bug.
    // See http://crbug.com/542840 for more details.
    g_need_ftruncate_workaround = true;
  }
}

bool NeedsFTruncateWorkAround() {
  return g_need_ftruncate_workaround;
}

}  // namespace
#endif  // defined(__ANDROID__)

namespace google_breakpad {

const MDRVA MinidumpFileWriter::kInvalidMDRVA = static_cast<MDRVA>(-1);

MinidumpFileWriter::MinidumpFileWriter()
    : file_(-1),
      close_file_when_destroyed_(true),
      position_(0),
      size_(0) {
}

MinidumpFileWriter::~MinidumpFileWriter() {
  if (close_file_when_destroyed_)
    Close();
}

bool MinidumpFileWriter::Open(const char* path) {
  assert(file_ == -1);
#if defined(__linux__) && __linux__
  file_ = sys_open(path, O_WRONLY | O_CREAT | O_EXCL, 0600);
#else
  file_ = open(path, O_WRONLY | O_CREAT | O_EXCL, 0600);
#endif

  return file_ != -1;
}

void MinidumpFileWriter::SetFile(const int file) {
  assert(file_ == -1);
  file_ = file;
  close_file_when_destroyed_ = false;
#if defined(__ANDROID__)
  CheckNeedsFTruncateWorkAround(file);
#endif
}

bool MinidumpFileWriter::Close() {
  bool result = true;

  if (file_ != -1) {
#if defined(__ANDROID__)
    if (!NeedsFTruncateWorkAround() && ftruncate(file_, position_)) {
       return false;
    }
#else
    if (ftruncate(file_, position_)) {
       return false;
    }
#endif
#if defined(__linux__) && __linux__
    result = (sys_close(file_) == 0);
#else
    result = (close(file_) == 0);
#endif
    file_ = -1;
  }

  return result;
}

bool MinidumpFileWriter::CopyStringToMDString(const wchar_t* str,
                                              unsigned int length,
                                              TypedMDRVA<MDString>* mdstring) {
  bool result = true;
  if (sizeof(wchar_t) == sizeof(uint16_t)) {
    // Shortcut if wchar_t is the same size as MDString's buffer
    result = mdstring->Copy(str, mdstring->get()->length);
  } else {
    uint16_t out[2];
    int out_idx = 0;

    // Copy the string character by character
    while (length && result) {
      UTF32ToUTF16Char(*str, out);
      if (!out[0])
        return false;

      // Process one character at a time
      --length;
      ++str;

      // Append the one or two UTF-16 characters.  The first one will be non-
      // zero, but the second one may be zero, depending on the conversion from
      // UTF-32.
      int out_count = out[1] ? 2 : 1;
      size_t out_size = sizeof(uint16_t) * out_count;
      result = mdstring->CopyIndexAfterObject(out_idx, out, out_size);
      out_idx += out_count;
    }
  }
  return result;
}

bool MinidumpFileWriter::CopyStringToMDString(const char* str,
                                              unsigned int length,
                                              TypedMDRVA<MDString>* mdstring) {
  bool result = true;
  uint16_t out[2];
  int out_idx = 0;

  // Copy the string character by character
  while (length && result) {
    int conversion_count = UTF8ToUTF16Char(str, length, out);
    if (!conversion_count)
      return false;

    // Move the pointer along based on the nubmer of converted characters
    length -= conversion_count;
    str += conversion_count;

    // Append the one or two UTF-16 characters
    int out_count = out[1] ? 2 : 1;
    size_t out_size = sizeof(uint16_t) * out_count;
    result = mdstring->CopyIndexAfterObject(out_idx, out, out_size);
    out_idx += out_count;
  }
  return result;
}

template <typename CharType>
bool MinidumpFileWriter::WriteStringCore(const CharType* str,
                                         unsigned int length,
                                         MDLocationDescriptor* location) {
  assert(str);
  assert(location);
  // Calculate the mdstring length by either limiting to |length| as passed in
  // or by finding the location of the NULL character.
  unsigned int mdstring_length = 0;
  if (!length)
    length = INT_MAX;
  for (; mdstring_length < length && str[mdstring_length]; ++mdstring_length)
    ;

  // Allocate the string buffer
  TypedMDRVA<MDString> mdstring(this);
  if (!mdstring.AllocateObjectAndArray(mdstring_length + 1, sizeof(uint16_t)))
    return false;

  // Set length excluding the NULL and copy the string
  mdstring.get()->length =
      static_cast<uint32_t>(mdstring_length * sizeof(uint16_t));
  bool result = CopyStringToMDString(str, mdstring_length, &mdstring);

  // NULL terminate
  if (result) {
    uint16_t ch = 0;
    result = mdstring.CopyIndexAfterObject(mdstring_length, &ch, sizeof(ch));

    if (result)
      *location = mdstring.location();
  }

  return result;
}

bool MinidumpFileWriter::WriteString(const wchar_t* str, unsigned int length,
                                     MDLocationDescriptor* location) {
  return WriteStringCore(str, length, location);
}

bool MinidumpFileWriter::WriteString(const char* str, unsigned int length,
                                     MDLocationDescriptor* location) {
  return WriteStringCore(str, length, location);
}

bool MinidumpFileWriter::WriteMemory(const void* src, size_t size,
                                     MDMemoryDescriptor* output) {
  assert(src);
  assert(output);
  UntypedMDRVA mem(this);

  if (!mem.Allocate(size))
    return false;
  if (!mem.Copy(src, mem.size()))
    return false;

  output->start_of_memory_range = reinterpret_cast<uint64_t>(src);
  output->memory = mem.location();

  return true;
}

MDRVA MinidumpFileWriter::Allocate(size_t size) {
  assert(size);
  assert(file_ != -1);
#if defined(__ANDROID__)
  if (NeedsFTruncateWorkAround()) {
    // If ftruncate() is not available. We simply increase the size beyond the
    // current file size. sys_write() will expand the file when data is written
    // to it. Because we did not over allocate to fit memory pages, we also
    // do not need to ftruncate() the file once we are done.
    size_ += size;

    // We don't need to seek since the file is unchanged.
    MDRVA current_position = position_;
    position_ += static_cast<MDRVA>(size);
    return current_position;
  }
#endif
  size_t aligned_size = (size + 7) & ~7;  // 64-bit alignment

  if (position_ + aligned_size > size_) {
    size_t growth = aligned_size;
    size_t minimal_growth = getpagesize();

    // Ensure that the file grows by at least the size of a memory page
    if (growth < minimal_growth)
      growth = minimal_growth;

    size_t new_size = size_ + growth;
    if (ftruncate(file_, new_size) != 0)
      return kInvalidMDRVA;

    size_ = new_size;
  }

  MDRVA current_position = position_;
  position_ += static_cast<MDRVA>(aligned_size);

  return current_position;
}

bool MinidumpFileWriter::Copy(MDRVA position, const void* src, ssize_t size) {
  assert(src);
  assert(size);
  assert(file_ != -1);

  // Ensure that the data will fit in the allocated space
  if (static_cast<size_t>(size + position) > size_)
    return false;

  // Seek and write the data
#if defined(__linux__) && __linux__
  if (sys_lseek(file_, position, SEEK_SET) == static_cast<off_t>(position)) {
    if (sys_write(file_, src, size) == size) {
      return true;
    }
  }
#else
  if (lseek(file_, position, SEEK_SET) == static_cast<off_t>(position)) {
    if (write(file_, src, size) == size) {
      return true;
    }
  }
#endif
  return false;
}

bool UntypedMDRVA::Allocate(size_t size) {
  assert(size_ == 0);
  size_ = size;
  position_ = writer_->Allocate(size_);
  return position_ != MinidumpFileWriter::kInvalidMDRVA;
}

bool UntypedMDRVA::Copy(MDRVA pos, const void* src, size_t size) {
  assert(src);
  assert(size);
  assert(pos + size <= position_ + size_);
  return writer_->Copy(pos, src, size);
}

}  // namespace google_breakpad
