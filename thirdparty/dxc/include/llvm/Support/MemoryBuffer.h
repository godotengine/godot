//===--- MemoryBuffer.h - Memory Buffer Interface ---------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file defines the MemoryBuffer interface.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_SUPPORT_MEMORYBUFFER_H
#define LLVM_SUPPORT_MEMORYBUFFER_H

#include "llvm-c/Support.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/CBindingWrapping.h"
#include "llvm/Support/DataTypes.h"
#include "llvm/Support/ErrorOr.h"
#include <memory>

namespace llvm {
class MemoryBufferRef;

/// This interface provides simple read-only access to a block of memory, and
/// provides simple methods for reading files and standard input into a memory
/// buffer.  In addition to basic access to the characters in the file, this
/// interface guarantees you can read one character past the end of the file,
/// and that this character will read as '\0'.
///
/// The '\0' guarantee is needed to support an optimization -- it's intended to
/// be more efficient for clients which are reading all the data to stop
/// reading when they encounter a '\0' than to continually check the file
/// position to see if it has reached the end of the file.
class MemoryBuffer {
  const char *BufferStart; // Start of the buffer.
  const char *BufferEnd;   // End of the buffer.

  MemoryBuffer(const MemoryBuffer &) = delete;
  MemoryBuffer &operator=(const MemoryBuffer &) = delete;
protected:
  MemoryBuffer() {}
  void init(const char *BufStart, const char *BufEnd,
            bool RequiresNullTerminator);
public:
  virtual ~MemoryBuffer();

  const char *getBufferStart() const { return BufferStart; }
  const char *getBufferEnd() const   { return BufferEnd; }
  size_t getBufferSize() const { return BufferEnd-BufferStart; }

  StringRef getBuffer() const {
    return StringRef(BufferStart, getBufferSize());
  }

  /// Return an identifier for this buffer, typically the filename it was read
  /// from.
  virtual const char *getBufferIdentifier() const {
    return "Unknown buffer";
  }

  /// Open the specified file as a MemoryBuffer, returning a new MemoryBuffer
  /// if successful, otherwise returning null. If FileSize is specified, this
  /// means that the client knows that the file exists and that it has the
  /// specified size.
  ///
  /// \param IsVolatileSize Set to true to indicate that the file size may be
  /// changing, e.g. when libclang tries to parse while the user is
  /// editing/updating the file.
  static ErrorOr<std::unique_ptr<MemoryBuffer>>
  getFile(const Twine &Filename, int64_t FileSize = -1,
          bool RequiresNullTerminator = true, bool IsVolatileSize = false);

  /// Given an already-open file descriptor, map some slice of it into a
  /// MemoryBuffer. The slice is specified by an \p Offset and \p MapSize.
  /// Since this is in the middle of a file, the buffer is not null terminated.
  static ErrorOr<std::unique_ptr<MemoryBuffer>>
  getOpenFileSlice(int FD, const Twine &Filename, uint64_t MapSize,
                   int64_t Offset);

  /// Given an already-open file descriptor, read the file and return a
  /// MemoryBuffer.
  ///
  /// \param IsVolatileSize Set to true to indicate that the file size may be
  /// changing, e.g. when libclang tries to parse while the user is
  /// editing/updating the file.
  static ErrorOr<std::unique_ptr<MemoryBuffer>>
  getOpenFile(int FD, const Twine &Filename, uint64_t FileSize,
              bool RequiresNullTerminator = true, bool IsVolatileSize = false);

  /// Open the specified memory range as a MemoryBuffer. Note that InputData
  /// must be null terminated if RequiresNullTerminator is true.
  static std::unique_ptr<MemoryBuffer>
  getMemBuffer(StringRef InputData, StringRef BufferName = "",
               bool RequiresNullTerminator = true);

  static std::unique_ptr<MemoryBuffer>
  getMemBuffer(MemoryBufferRef Ref, bool RequiresNullTerminator = true);

  /// Open the specified memory range as a MemoryBuffer, copying the contents
  /// and taking ownership of it. InputData does not have to be null terminated.
  static std::unique_ptr<MemoryBuffer>
  getMemBufferCopy(StringRef InputData, const Twine &BufferName = "");

  /// Allocate a new zero-initialized MemoryBuffer of the specified size. Note
  /// that the caller need not initialize the memory allocated by this method.
  /// The memory is owned by the MemoryBuffer object.
  static std::unique_ptr<MemoryBuffer>
  getNewMemBuffer(size_t Size, StringRef BufferName = "");

  /// Allocate a new MemoryBuffer of the specified size that is not initialized.
  /// Note that the caller should initialize the memory allocated by this
  /// method. The memory is owned by the MemoryBuffer object.
  static std::unique_ptr<MemoryBuffer>
  getNewUninitMemBuffer(size_t Size, const Twine &BufferName = "");

  /// Read all of stdin into a file buffer, and return it.
  static ErrorOr<std::unique_ptr<MemoryBuffer>> getSTDIN();

  /// Open the specified file as a MemoryBuffer, or open stdin if the Filename
  /// is "-".
  static ErrorOr<std::unique_ptr<MemoryBuffer>>
  getFileOrSTDIN(const Twine &Filename, int64_t FileSize = -1);

  /// Map a subrange of the specified file as a MemoryBuffer.
  static ErrorOr<std::unique_ptr<MemoryBuffer>>
  getFileSlice(const Twine &Filename, uint64_t MapSize, uint64_t Offset);

  //===--------------------------------------------------------------------===//
  // Provided for performance analysis.
  //===--------------------------------------------------------------------===//

  /// The kind of memory backing used to support the MemoryBuffer.
  enum BufferKind {
    MemoryBuffer_Malloc,
    MemoryBuffer_MMap
  };

  /// Return information on the memory mechanism used to support the
  /// MemoryBuffer.
  virtual BufferKind getBufferKind() const = 0;

  MemoryBufferRef getMemBufferRef() const;
};

class MemoryBufferRef {
  StringRef Buffer;
  StringRef Identifier;

public:
  MemoryBufferRef() {}
  MemoryBufferRef(StringRef Buffer, StringRef Identifier)
      : Buffer(Buffer), Identifier(Identifier) {}

  StringRef getBuffer() const { return Buffer; }

  StringRef getBufferIdentifier() const { return Identifier; }

  const char *getBufferStart() const { return Buffer.begin(); }
  const char *getBufferEnd() const { return Buffer.end(); }
  size_t getBufferSize() const { return Buffer.size(); }
};

// Create wrappers for C Binding types (see CBindingWrapping.h).
DEFINE_SIMPLE_CONVERSION_FUNCTIONS(MemoryBuffer, LLVMMemoryBufferRef)

} // end namespace llvm

#endif
