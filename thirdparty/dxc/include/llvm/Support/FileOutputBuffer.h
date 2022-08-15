//=== FileOutputBuffer.h - File Output Buffer -------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Utility for creating a in-memory buffer that will be written to a file.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_SUPPORT_FILEOUTPUTBUFFER_H
#define LLVM_SUPPORT_FILEOUTPUTBUFFER_H

#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/DataTypes.h"
#include "llvm/Support/FileSystem.h"

namespace llvm {
/// FileOutputBuffer - This interface provides simple way to create an in-memory
/// buffer which will be written to a file. During the lifetime of these
/// objects, the content or existence of the specified file is undefined. That
/// is, creating an OutputBuffer for a file may immediately remove the file.
/// If the FileOutputBuffer is committed, the target file's content will become
/// the buffer content at the time of the commit.  If the FileOutputBuffer is
/// not committed, the file will be deleted in the FileOutputBuffer destructor.
class FileOutputBuffer {
public:

  enum  {
    F_executable = 1  /// set the 'x' bit on the resulting file
  };

  /// Factory method to create an OutputBuffer object which manages a read/write
  /// buffer of the specified size. When committed, the buffer will be written
  /// to the file at the specified path.
  static std::error_code create(StringRef FilePath, size_t Size,
                                std::unique_ptr<FileOutputBuffer> &Result,
                                unsigned Flags = 0);

  /// Returns a pointer to the start of the buffer.
  uint8_t *getBufferStart() {
    return (uint8_t*)Region->data();
  }

  /// Returns a pointer to the end of the buffer.
  uint8_t *getBufferEnd() {
    return (uint8_t*)Region->data() + Region->size();
  }

  /// Returns size of the buffer.
  size_t getBufferSize() const {
    return Region->size();
  }

  /// Returns path where file will show up if buffer is committed.
  StringRef getPath() const {
    return FinalPath;
  }

  /// Flushes the content of the buffer to its file and deallocates the
  /// buffer.  If commit() is not called before this object's destructor
  /// is called, the file is deleted in the destructor. The optional parameter
  /// is used if it turns out you want the file size to be smaller than
  /// initially requested.
  std::error_code commit();

  /// If this object was previously committed, the destructor just deletes
  /// this object.  If this object was not committed, the destructor
  /// deallocates the buffer and the target file is never written.
  ~FileOutputBuffer();

private:
  FileOutputBuffer(const FileOutputBuffer &) = delete;
  FileOutputBuffer &operator=(const FileOutputBuffer &) = delete;

  FileOutputBuffer(std::unique_ptr<llvm::sys::fs::mapped_file_region> R,
                   StringRef Path, StringRef TempPath);

  std::unique_ptr<llvm::sys::fs::mapped_file_region> Region;
  SmallString<128>    FinalPath;
  SmallString<128>    TempPath;
};
} // end namespace llvm

#endif
