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

// minidump_file_writer.h:  Implements file-based minidump generation.  It's
// intended to be used with the Google Breakpad open source crash handling
// project.

#ifndef CLIENT_MINIDUMP_FILE_WRITER_H__
#define CLIENT_MINIDUMP_FILE_WRITER_H__

#include <string>

#include "google_breakpad/common/minidump_format.h"

namespace google_breakpad {

class UntypedMDRVA;
template<typename MDType> class TypedMDRVA;

// The user of this class can Open() a file and add minidump streams, data, and
// strings using the definitions in minidump_format.h.  Since this class is
// expected to be used in a situation where the current process may be
// damaged, it will not allocate heap memory.
// Sample usage:
// MinidumpFileWriter writer;
// writer.Open("/tmp/minidump.dmp");
// TypedMDRVA<MDRawHeader> header(&writer_);
// header.Allocate();
// header->get()->signature = MD_HEADER_SIGNATURE;
//  :
// writer.Close();
//
// An alternative is to use SetFile and provide a file descriptor:
// MinidumpFileWriter writer;
// writer.SetFile(minidump_fd);
// TypedMDRVA<MDRawHeader> header(&writer_);
// header.Allocate();
// header->get()->signature = MD_HEADER_SIGNATURE;
//  :
// writer.Close();

class MinidumpFileWriter {
public:
  // Invalid MDRVA (Minidump Relative Virtual Address)
  // returned on failed allocation
  static const MDRVA kInvalidMDRVA;

  MinidumpFileWriter();
  ~MinidumpFileWriter();

  // Open |path| as the destination of the minidump data. If |path| already
  // exists, then Open() will fail.
  // Return true on success, or false on failure.
  bool Open(const char* path);

  // Sets the file descriptor |file| as the destination of the minidump data.
  // Can be used as an alternative to Open() when a file descriptor is
  // available.
  // Note that |fd| is not closed when the instance of MinidumpFileWriter is
  // destroyed.
  void SetFile(const int file);

  // Close the current file (that was either created when Open was called, or
  // specified with SetFile).
  // Return true on success, or false on failure.
  bool Close();

  // Copy the contents of |str| to a MDString and write it to the file.
  // |str| is expected to be either UTF-16 or UTF-32 depending on the size
  // of wchar_t.
  // Maximum |length| of characters to copy from |str|, or specify 0 to use the
  // entire NULL terminated string.  Copying will stop at the first NULL.
  // |location| the allocated location
  // Return true on success, or false on failure
  bool WriteString(const wchar_t* str, unsigned int length,
                   MDLocationDescriptor* location);

  // Same as above, except with |str| as a UTF-8 string
  bool WriteString(const char* str, unsigned int length,
                   MDLocationDescriptor* location);

  // Write |size| bytes starting at |src| into the current position.
  // Return true on success and set |output| to position, or false on failure
  bool WriteMemory(const void* src, size_t size, MDMemoryDescriptor* output);

  // Copies |size| bytes from |src| to |position|
  // Return true on success, or false on failure
  bool Copy(MDRVA position, const void* src, ssize_t size);

  // Return the current position for writing to the minidump
  inline MDRVA position() const { return position_; }

 private:
  friend class UntypedMDRVA;

  // Allocates an area of |size| bytes.
  // Returns the position of the allocation, or kInvalidMDRVA if it was
  // unable to allocate the bytes.
  MDRVA Allocate(size_t size);

  // The file descriptor for the output file.
  int file_;

  // Whether |file_| should be closed when the instance is destroyed.
  bool close_file_when_destroyed_;

  // Current position in buffer
  MDRVA position_;

  // Current allocated size
  size_t size_;

  // Copy |length| characters from |str| to |mdstring|.  These are distinct
  // because the underlying MDString is a UTF-16 based string.  The wchar_t
  // variant may need to create a MDString that has more characters than the
  // source |str|, whereas the UTF-8 variant may coalesce characters to form
  // a single UTF-16 character.
  bool CopyStringToMDString(const wchar_t* str, unsigned int length,
                            TypedMDRVA<MDString>* mdstring);
  bool CopyStringToMDString(const char* str, unsigned int length,
                            TypedMDRVA<MDString>* mdstring);

  // The common templated code for writing a string
  template <typename CharType>
  bool WriteStringCore(const CharType* str, unsigned int length,
                       MDLocationDescriptor* location);
};

// Represents an untyped allocated chunk
class UntypedMDRVA {
 public:
  explicit UntypedMDRVA(MinidumpFileWriter* writer)
      : writer_(writer),
        position_(writer->position()),
        size_(0) {}

  // Allocates |size| bytes.  Must not call more than once.
  // Return true on success, or false on failure
  bool Allocate(size_t size);

  // Returns the current position or kInvalidMDRVA if allocation failed
  inline MDRVA position() const { return position_; }

  // Number of bytes allocated
  inline size_t size() const { return size_; }

  // Return size and position
  inline MDLocationDescriptor location() const {
    MDLocationDescriptor location = { static_cast<uint32_t>(size_),
                                      position_ };
    return location;
  }

  // Copy |size| bytes starting at |src| into the minidump at |position|
  // Return true on success, or false on failure
  bool Copy(MDRVA position, const void* src, size_t size);

  // Copy |size| bytes from |src| to the current position
  inline bool Copy(const void* src, size_t size) {
    return Copy(position_, src, size);
  }

 protected:
  // Writer we associate with
  MinidumpFileWriter* writer_;

  // Position of the start of the data
  MDRVA position_;

  // Allocated size
  size_t size_;
};

// Represents a Minidump object chunk.  Additional memory can be allocated at
// the end of the object as a:
// - single allocation
// - Array of MDType objects
// - A MDType object followed by an array
template<typename MDType>
class TypedMDRVA : public UntypedMDRVA {
 public:
  // Constructs an unallocated MDRVA
  explicit TypedMDRVA(MinidumpFileWriter* writer)
      : UntypedMDRVA(writer),
        data_(),
        allocation_state_(UNALLOCATED) {}

  inline ~TypedMDRVA() {
    // Ensure that the data_ object is written out
    if (allocation_state_ != ARRAY)
      Flush();
  }

  // Address of object data_ of MDType.  This is not declared const as the
  // typical usage will be to access the underlying |data_| object as to
  // alter its contents.
  MDType* get() { return &data_; }

  // Allocates minidump_size<MDType>::size() bytes.
  // Must not call more than once.
  // Return true on success, or false on failure
  bool Allocate();

  // Allocates minidump_size<MDType>::size() + |additional| bytes.
  // Must not call more than once.
  // Return true on success, or false on failure
  bool Allocate(size_t additional);

  // Allocate an array of |count| elements of MDType.
  // Must not call more than once.
  // Return true on success, or false on failure
  bool AllocateArray(size_t count);

  // Allocate an array of |count| elements of |size| after object of MDType
  // Must not call more than once.
  // Return true on success, or false on failure
  bool AllocateObjectAndArray(size_t count, size_t size);

  // Copy |item| to |index|
  // Must have been allocated using AllocateArray().
  // Return true on success, or false on failure
  bool CopyIndex(unsigned int index, MDType* item);

  // Copy |size| bytes starting at |str| to |index|
  // Must have been allocated using AllocateObjectAndArray().
  // Return true on success, or false on failure
  bool CopyIndexAfterObject(unsigned int index, const void* src, size_t size);

  // Write data_
  bool Flush();

 private:
  enum AllocationState {
    UNALLOCATED = 0,
    SINGLE_OBJECT,
    ARRAY,
    SINGLE_OBJECT_WITH_ARRAY
  };

  MDType data_;
  AllocationState allocation_state_;
};

}  // namespace google_breakpad

#endif  // CLIENT_MINIDUMP_FILE_WRITER_H__
