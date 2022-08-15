//===- FileOutputBuffer.cpp - File Output Buffer ----------------*- C++ -*-===//
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

#include "llvm/Support/FileOutputBuffer.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/Support/Errc.h"
#include <system_error>

#if !defined(_MSC_VER) && !defined(__MINGW32__)
#include <unistd.h>
#else
#include <io.h>
#endif

using llvm::sys::fs::mapped_file_region;

namespace llvm {
FileOutputBuffer::FileOutputBuffer(std::unique_ptr<mapped_file_region> R,
                                   StringRef Path, StringRef TmpPath)
    : Region(std::move(R)), FinalPath(Path), TempPath(TmpPath) {}

FileOutputBuffer::~FileOutputBuffer() {
  sys::fs::remove(Twine(TempPath));
}

std::error_code
FileOutputBuffer::create(StringRef FilePath, size_t Size,
                         std::unique_ptr<FileOutputBuffer> &Result,
                         unsigned Flags) {
  // If file already exists, it must be a regular file (to be mappable).
  sys::fs::file_status Stat;
  std::error_code EC = sys::fs::status(FilePath, Stat);
  switch (Stat.type()) {
    case sys::fs::file_type::file_not_found:
      // If file does not exist, we'll create one.
      break;
    case sys::fs::file_type::regular_file: {
        // If file is not currently writable, error out.
        // FIXME: There is no sys::fs:: api for checking this.
        // FIXME: In posix, you use the access() call to check this.
      }
      break;
    default:
      if (EC)
        return EC;
      else
        return make_error_code(errc::operation_not_permitted);
  }

  // Delete target file.
  EC = sys::fs::remove(FilePath);
  if (EC)
    return EC;

  unsigned Mode = sys::fs::all_read | sys::fs::all_write;
  // If requested, make the output file executable.
  if (Flags & F_executable)
    Mode |= sys::fs::all_exe;

  // Create new file in same directory but with random name.
  SmallString<128> TempFilePath;
  int FD;
  EC = sys::fs::createUniqueFile(Twine(FilePath) + ".tmp%%%%%%%", FD,
                                 TempFilePath, Mode);
  if (EC)
    return EC;

#ifndef LLVM_ON_WIN32
  // On Windows, CreateFileMapping (the mmap function on Windows)
  // automatically extends the underlying file. We don't need to
  // extend the file beforehand. _chsize (ftruncate on Windows) is
  // pretty slow just like it writes specified amount of bytes,
  // so we should avoid calling that.
  EC = sys::fs::resize_file(FD, Size);
  if (EC)
    return EC;
#endif

  auto MappedFile = llvm::make_unique<mapped_file_region>(
      FD, mapped_file_region::readwrite, Size, 0, EC);
  int Ret = close(FD);
  if (EC)
    return EC;
  if (Ret)
    return std::error_code(errno, std::generic_category());

  Result.reset(
      new FileOutputBuffer(std::move(MappedFile), FilePath, TempFilePath));

  return std::error_code();
}

std::error_code FileOutputBuffer::commit() {
  // Unmap buffer, letting OS flush dirty pages to file on disk.
  Region.reset();


  // Rename file to final name.
  return sys::fs::rename(Twine(TempPath), Twine(FinalPath));
}
} // namespace
