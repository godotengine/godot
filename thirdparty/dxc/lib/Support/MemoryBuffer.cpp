//===--- MemoryBuffer.cpp - Memory Buffer implementation ------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file implements the MemoryBuffer interface.
//
//===----------------------------------------------------------------------===//

#include "llvm/Support/MemoryBuffer.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/Config/config.h"
#include "llvm/Support/Errc.h"
#include "llvm/Support/Errno.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Process.h"
#include "llvm/Support/Program.h"
#include <cassert>
#include <cerrno>
#include <cstring>
#include <new>
#include <sys/types.h>
#include <system_error>
#if !defined(_MSC_VER) && !defined(__MINGW32__)
#include <unistd.h>
#else
#include <io.h>
#endif
using namespace llvm;

//===----------------------------------------------------------------------===//
// MemoryBuffer implementation itself.
//===----------------------------------------------------------------------===//

MemoryBuffer::~MemoryBuffer() { }

/// init - Initialize this MemoryBuffer as a reference to externally allocated
/// memory, memory that we know is already null terminated.
void MemoryBuffer::init(const char *BufStart, const char *BufEnd,
                        bool RequiresNullTerminator) {
  assert((!RequiresNullTerminator || BufEnd[0] == 0) &&
         "Buffer is not null terminated!");
  BufferStart = BufStart;
  BufferEnd = BufEnd;
}

//===----------------------------------------------------------------------===//
// MemoryBufferMem implementation.
//===----------------------------------------------------------------------===//

/// CopyStringRef - Copies contents of a StringRef into a block of memory and
/// null-terminates it.
static void CopyStringRef(_Out_cap_x_(Data.size()+1) char *Memory, StringRef Data) {
  if (!Data.empty())
    memcpy(Memory, Data.data(), Data.size());
  Memory[Data.size()] = 0; // Null terminate string.
}

namespace {
struct NamedBufferAlloc {
  const Twine &Name;
  NamedBufferAlloc(const Twine &Name) : Name(Name) {}
};
}

void *operator new(size_t N, const NamedBufferAlloc &Alloc) {
  SmallString<256> NameBuf;
  StringRef NameRef = Alloc.Name.toStringRef(NameBuf);

  char *Mem = static_cast<char *>(operator new(N + NameRef.size() + 1));
  CopyStringRef(Mem + N, NameRef);
  return Mem;
}

namespace {
/// MemoryBufferMem - Named MemoryBuffer pointing to a block of memory.
class MemoryBufferMem : public MemoryBuffer {
public:
  MemoryBufferMem(StringRef InputData, bool RequiresNullTerminator) {
    init(InputData.begin(), InputData.end(), RequiresNullTerminator);
  }

  const char *getBufferIdentifier() const override {
     // The name is stored after the class itself.
    return reinterpret_cast<const char*>(this + 1);
  }

  BufferKind getBufferKind() const override {
    return MemoryBuffer_Malloc;
  }
};
}

static ErrorOr<std::unique_ptr<MemoryBuffer>>
getFileAux(const Twine &Filename, int64_t FileSize, uint64_t MapSize, 
           uint64_t Offset, bool RequiresNullTerminator, bool IsVolatileSize);

std::unique_ptr<MemoryBuffer>
MemoryBuffer::getMemBuffer(StringRef InputData, StringRef BufferName,
                           bool RequiresNullTerminator) {
  auto *Ret = new (NamedBufferAlloc(BufferName))
      MemoryBufferMem(InputData, RequiresNullTerminator);
  return std::unique_ptr<MemoryBuffer>(Ret);
}

std::unique_ptr<MemoryBuffer>
MemoryBuffer::getMemBuffer(MemoryBufferRef Ref, bool RequiresNullTerminator) {
  return std::unique_ptr<MemoryBuffer>(getMemBuffer(
      Ref.getBuffer(), Ref.getBufferIdentifier(), RequiresNullTerminator));
}

std::unique_ptr<MemoryBuffer>
MemoryBuffer::getMemBufferCopy(StringRef InputData, const Twine &BufferName) {
  std::unique_ptr<MemoryBuffer> Buf =
      getNewUninitMemBuffer(InputData.size(), BufferName);
  if (!Buf)
    return nullptr;
  memcpy(const_cast<char*>(Buf->getBufferStart()), InputData.data(),
         InputData.size());
  return Buf;
}

std::unique_ptr<MemoryBuffer>
MemoryBuffer::getNewUninitMemBuffer(size_t Size, const Twine &BufferName) {
  // Allocate space for the MemoryBuffer, the data and the name. It is important
  // that MemoryBuffer and data are aligned so PointerIntPair works with them.
  // TODO: Is 16-byte alignment enough?  We copy small object files with large
  // alignment expectations into this buffer.
  SmallString<256> NameBuf;
  StringRef NameRef = BufferName.toStringRef(NameBuf);
  size_t AlignedStringLen =
      RoundUpToAlignment(sizeof(MemoryBufferMem) + NameRef.size() + 1, 16);
  size_t RealLen = AlignedStringLen + Size + 1;
  char *Mem = static_cast<char*>(operator new(RealLen, std::nothrow));
  if (!Mem)
    return nullptr;

  // The name is stored after the class itself.
  CopyStringRef(Mem + sizeof(MemoryBufferMem), NameRef);

  // The buffer begins after the name and must be aligned.
  char *Buf = Mem + AlignedStringLen;
  Buf[Size] = 0; // Null terminate buffer.

  auto *Ret = new (Mem) MemoryBufferMem(StringRef(Buf, Size), true);
  return std::unique_ptr<MemoryBuffer>(Ret);
}

std::unique_ptr<MemoryBuffer>
MemoryBuffer::getNewMemBuffer(size_t Size, StringRef BufferName) {
  std::unique_ptr<MemoryBuffer> SB = getNewUninitMemBuffer(Size, BufferName);
  if (!SB)
    return nullptr;
  memset(const_cast<char*>(SB->getBufferStart()), 0, Size);
  return SB;
}

ErrorOr<std::unique_ptr<MemoryBuffer>>
MemoryBuffer::getFileOrSTDIN(const Twine &Filename, int64_t FileSize) {
  SmallString<256> NameBuf;
  StringRef NameRef = Filename.toStringRef(NameBuf);

  if (NameRef == "-")
    return getSTDIN();
  return getFile(Filename, FileSize);
}

ErrorOr<std::unique_ptr<MemoryBuffer>>
MemoryBuffer::getFileSlice(const Twine &FilePath, uint64_t MapSize, 
                           uint64_t Offset) {
  return getFileAux(FilePath, -1, MapSize, Offset, false, false);
}


//===----------------------------------------------------------------------===//
// MemoryBuffer::getFile implementation.
//===----------------------------------------------------------------------===//

namespace {
/// \brief Memory maps a file descriptor using sys::fs::mapped_file_region.
///
/// This handles converting the offset into a legal offset on the platform.
class MemoryBufferMMapFile : public MemoryBuffer {
  sys::fs::mapped_file_region MFR;

  static uint64_t getLegalMapOffset(uint64_t Offset) {
    return Offset & ~(sys::fs::mapped_file_region::alignment() - 1);
  }

  static uint64_t getLegalMapSize(uint64_t Len, uint64_t Offset) {
    return Len + (Offset - getLegalMapOffset(Offset));
  }

  const char *getStart(uint64_t Len, uint64_t Offset) {
    return MFR.const_data() + (Offset - getLegalMapOffset(Offset));
  }

public:
  MemoryBufferMMapFile(bool RequiresNullTerminator, int FD, uint64_t Len,
                       uint64_t Offset, std::error_code &EC)
      : MFR(FD, sys::fs::mapped_file_region::readonly,
            getLegalMapSize(Len, Offset), getLegalMapOffset(Offset), EC) {
    if (!EC) {
      const char *Start = getStart(Len, Offset);
      init(Start, Start + Len, RequiresNullTerminator);
    }
  }

  const char *getBufferIdentifier() const override {
    // The name is stored after the class itself.
    return reinterpret_cast<const char *>(this + 1);
  }

  BufferKind getBufferKind() const override {
    return MemoryBuffer_MMap;
  }
};
}

static ErrorOr<std::unique_ptr<MemoryBuffer>>
getMemoryBufferForStream(int FD, const Twine &BufferName) {
  const ssize_t ChunkSize = 4096*4;
  const ssize_t InitialChunkSize = 4096*2; // HLSL Change - be more conservative on how much stack space we start with
  SmallString<InitialChunkSize> Buffer;
  ssize_t ReadBytes;
  // Read into Buffer until we hit EOF.
  do {
    Buffer.reserve(Buffer.size() + ChunkSize);
    ReadBytes = llvm::sys::fs::msf_read(FD, Buffer.end(), ChunkSize);
    if (ReadBytes == -1) {
      if (errno == EINTR) continue;
      return std::error_code(errno, std::generic_category());
    }
    Buffer.set_size(Buffer.size() + ReadBytes);
  } while (ReadBytes != 0);

  return MemoryBuffer::getMemBufferCopy(Buffer, BufferName);
}


ErrorOr<std::unique_ptr<MemoryBuffer>>
MemoryBuffer::getFile(const Twine &Filename, int64_t FileSize,
                      bool RequiresNullTerminator, bool IsVolatileSize) {
  return getFileAux(Filename, FileSize, FileSize, 0,
                    RequiresNullTerminator, IsVolatileSize);
}

static ErrorOr<std::unique_ptr<MemoryBuffer>>
getOpenFileImpl(int FD, const Twine &Filename, uint64_t FileSize,
                uint64_t MapSize, int64_t Offset, bool RequiresNullTerminator,
                bool IsVolatileSize);

static ErrorOr<std::unique_ptr<MemoryBuffer>>
getFileAux(const Twine &Filename, int64_t FileSize, uint64_t MapSize,
           uint64_t Offset, bool RequiresNullTerminator, bool IsVolatileSize) {
  int FD;
  std::error_code EC = sys::fs::openFileForRead(Filename, FD);
  if (EC)
    return EC;

  ErrorOr<std::unique_ptr<MemoryBuffer>> Ret =
      getOpenFileImpl(FD, Filename, FileSize, MapSize, Offset,
                      RequiresNullTerminator, IsVolatileSize);
  llvm::sys::fs::msf_close(FD);  // HLSL Change - use msf_close
  return Ret;
}

static bool shouldUseMmap(int FD,
                          size_t FileSize,
                          size_t MapSize,
                          off_t Offset,
                          bool RequiresNullTerminator,
                          int PageSize,
                          bool IsVolatileSize) {
  // mmap may leave the buffer without null terminator if the file size changed
  // by the time the last page is mapped in, so avoid it if the file size is
  // likely to change.
  if (IsVolatileSize)
    return false;

  // We don't use mmap for small files because this can severely fragment our
  // address space.
  if (MapSize < 4 * 4096 || MapSize < (unsigned)PageSize)
    return false;

  if (!RequiresNullTerminator)
    return true;


  // If we don't know the file size, use fstat to find out.  fstat on an open
  // file descriptor is cheaper than stat on a random path.
  // FIXME: this chunk of code is duplicated, but it avoids a fstat when
  // RequiresNullTerminator = false and MapSize != -1.
  if (FileSize == size_t(-1)) {
    sys::fs::file_status Status;
    if (sys::fs::status(FD, Status))
      return false;
    FileSize = Status.getSize();
  }

  // If we need a null terminator and the end of the map is inside the file,
  // we cannot use mmap.
  size_t End = Offset + MapSize;
  assert(End <= FileSize);
  if (End != FileSize)
    return false;

  // Don't try to map files that are exactly a multiple of the system page size
  // if we need a null terminator.
  if ((FileSize & (PageSize -1)) == 0)
    return false;

#if defined(__CYGWIN__)
  // Don't try to map files that are exactly a multiple of the physical page size
  // if we need a null terminator.
  // FIXME: We should reorganize again getPageSize() on Win32.
  if ((FileSize & (4096 - 1)) == 0)
    return false;
#endif

  return true;
}

static ErrorOr<std::unique_ptr<MemoryBuffer>>
getOpenFileImpl(int FD, const Twine &Filename, uint64_t FileSize,
                uint64_t MapSize, int64_t Offset, bool RequiresNullTerminator,
                bool IsVolatileSize) {
  static int PageSize = sys::Process::getPageSize();

  // Default is to map the full file.
  if (MapSize == uint64_t(-1)) {
    // If we don't know the file size, use fstat to find out.  fstat on an open
    // file descriptor is cheaper than stat on a random path.
    if (FileSize == uint64_t(-1)) {
      sys::fs::file_status Status;
      std::error_code EC = sys::fs::status(FD, Status);
      if (EC)
        return EC;

      // If this not a file or a block device (e.g. it's a named pipe
      // or character device), we can't trust the size. Create the memory
      // buffer by copying off the stream.
      sys::fs::file_type Type = Status.type();
      if (Type != sys::fs::file_type::regular_file &&
          Type != sys::fs::file_type::block_file)
        return getMemoryBufferForStream(FD, Filename);

      FileSize = Status.getSize();
    }
    MapSize = FileSize;
  }

  if (shouldUseMmap(FD, FileSize, MapSize, Offset, RequiresNullTerminator,
                    PageSize, IsVolatileSize)) {
    std::error_code EC;
    std::unique_ptr<MemoryBuffer> Result(
        new (NamedBufferAlloc(Filename))
        MemoryBufferMMapFile(RequiresNullTerminator, FD, MapSize, Offset, EC));
    if (!EC)
      return std::move(Result);
  }

  std::unique_ptr<MemoryBuffer> Buf =
      MemoryBuffer::getNewUninitMemBuffer(MapSize, Filename);
  if (!Buf) {
    // Failed to create a buffer. The only way it can fail is if
    // new(std::nothrow) returns 0.
    return make_error_code(errc::not_enough_memory);
  }

  char *BufPtr = const_cast<char *>(Buf->getBufferStart());

  size_t BytesLeft = MapSize;
#undef HAVE_PREAD // HLSL Change - pread bypasses needed layers
#ifndef HAVE_PREAD
  if (llvm::sys::fs::msf_lseek(FD, Offset, SEEK_SET) == -1)  // HLSL Change - use msf_lseek
    return std::error_code(errno, std::generic_category());
#endif

  while (BytesLeft) {
#ifdef HAVE_PREAD
    ssize_t NumRead = ::pread(FD, BufPtr, BytesLeft, MapSize-BytesLeft+Offset);
#else
    ssize_t NumRead = ::llvm::sys::fs::msf_read(FD, BufPtr, BytesLeft);
#endif
    if (NumRead == -1) {
      if (errno == EINTR)
        continue;
      // Error while reading.
      return std::error_code(errno, std::generic_category());
    }
    if (NumRead == 0) {
      memset(BufPtr, 0, BytesLeft); // zero-initialize rest of the buffer.
      break;
    }
    BytesLeft -= NumRead;
    BufPtr += NumRead;
  }

  return std::move(Buf);
}

ErrorOr<std::unique_ptr<MemoryBuffer>>
MemoryBuffer::getOpenFile(int FD, const Twine &Filename, uint64_t FileSize,
                          bool RequiresNullTerminator, bool IsVolatileSize) {
  return getOpenFileImpl(FD, Filename, FileSize, FileSize, 0,
                         RequiresNullTerminator, IsVolatileSize);
}

ErrorOr<std::unique_ptr<MemoryBuffer>>
MemoryBuffer::getOpenFileSlice(int FD, const Twine &Filename, uint64_t MapSize,
                               int64_t Offset) {
  assert(MapSize != uint64_t(-1));
  return getOpenFileImpl(FD, Filename, -1, MapSize, Offset, false,
                         /*IsVolatileSize*/ false);
}

ErrorOr<std::unique_ptr<MemoryBuffer>> MemoryBuffer::getSTDIN() {
  // Read in all of the data from stdin, we cannot mmap stdin.
  //
  // FIXME: That isn't necessarily true, we should try to mmap stdin and
  // fallback if it fails.
  sys::ChangeStdinToBinary();

  return getMemoryBufferForStream(0, "<stdin>");
}

MemoryBufferRef MemoryBuffer::getMemBufferRef() const {
  StringRef Data = getBuffer();
  StringRef Identifier = getBufferIdentifier();
  return MemoryBufferRef(Data, Identifier);
}
