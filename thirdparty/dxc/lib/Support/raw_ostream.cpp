//===--- raw_ostream.cpp - Implement the raw_ostream classes --------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This implements support for bulk buffered stream output.
//
//===----------------------------------------------------------------------===//

#include "llvm/Support/raw_ostream.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Config/config.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/Process.h"
#include "llvm/Support/Program.h"
#include <cctype>
#include <cerrno>
#include <ios>
#include <sys/stat.h>
#include <system_error>

// <fcntl.h> may provide O_BINARY.
#if defined(HAVE_FCNTL_H)
# include <fcntl.h>
#endif

#if defined(HAVE_UNISTD_H)
# include <unistd.h>
#endif
#if defined(HAVE_SYS_UIO_H) && defined(HAVE_WRITEV)
#  include <sys/uio.h>
#endif

#if defined(__CYGWIN__)
#include <io.h>
#endif

#if defined(_MSC_VER)
#include <io.h>
#ifndef STDIN_FILENO
# define STDIN_FILENO 0
#endif
#ifndef STDOUT_FILENO
# define STDOUT_FILENO 1
#endif
#ifndef STDERR_FILENO
# define STDERR_FILENO 2
#endif
#endif

using namespace llvm;

raw_ostream::~raw_ostream() {
  // raw_ostream's subclasses should take care to flush the buffer
  // in their destructors.
  assert(OutBufCur == OutBufStart &&
         "raw_ostream destructor called with non-empty buffer!");

  if (BufferMode == InternalBuffer)
    delete [] OutBufStart;
}

// An out of line virtual method to provide a home for the class vtable.
void raw_ostream::handle() {}

size_t raw_ostream::preferred_buffer_size() const {
  // BUFSIZ is intended to be a reasonable default.
  return BUFSIZ;
}

void raw_ostream::SetBuffered() {
  // Ask the subclass to determine an appropriate buffer size.
  if (size_t Size = preferred_buffer_size())
    SetBufferSize(Size);
  else
    // It may return 0, meaning this stream should be unbuffered.
    SetUnbuffered();
}

void raw_ostream::SetBufferAndMode(_In_opt_ char *BufferStart, size_t Size,
                                   BufferKind Mode) {
  assert(((Mode == Unbuffered && !BufferStart && Size == 0) ||
          (Mode != Unbuffered && BufferStart && Size != 0)) &&
         "stream must be unbuffered or have at least one byte");
  // Make sure the current buffer is free of content (we can't flush here; the
  // child buffer management logic will be in write_impl).
  assert(GetNumBytesInBuffer() == 0 && "Current buffer is non-empty!");

  if (BufferMode == InternalBuffer)
    delete [] OutBufStart;
  OutBufStart = BufferStart;
  OutBufEnd = OutBufStart+Size;
  OutBufCur = OutBufStart;
  BufferMode = Mode;

  assert(OutBufStart <= OutBufEnd && "Invalid size!");
}

raw_ostream &raw_ostream::operator<<(unsigned long N) {

  // HLSL Change Starts - Handle non-base10 printing
  if (writeBase != 10) {
    *this << '0';
    if (writeBase == 16)
      *this << 'x';
    return write_base((unsigned long long)N);
  }
  // HLSL Change Ends

  // Zero is a special case.
  if (N == 0)
    return *this << '0';

  char NumberBuffer[20];
  char *EndPtr = NumberBuffer+sizeof(NumberBuffer);
  char *CurPtr = EndPtr;

  while (N) {
    *--CurPtr = '0' + char(N % 10);
    N /= 10;
  }
  return write(CurPtr, EndPtr-CurPtr);
}

raw_ostream &raw_ostream::operator<<(long N) {
  if (N < 0 && writeBase == 10) {
    *this << '-';
    // Avoid undefined behavior on LONG_MIN with a cast.
    N = -(unsigned long)N;
  }

  return this->operator<<(static_cast<unsigned long>(N));
}

raw_ostream &raw_ostream::operator<<(unsigned long long N) {
  // Output using 32-bit div/mod when possible.
  if (N == static_cast<unsigned long>(N))
    return this->operator<<(static_cast<unsigned long>(N));

  // HLSL Change Starts - Handle non-base10 printing
  if (writeBase != 10) {
    *this << '0';
    if (writeBase == 16)
      *this << 'x';
    return write_base((unsigned long long)N);
  }
  // HLSL Change Ends

  char NumberBuffer[20];
  char *EndPtr = NumberBuffer+sizeof(NumberBuffer);
  char *CurPtr = EndPtr;

  while (N) {
    *--CurPtr = '0' + char(N % 10);
    N /= 10;
  }
  return write(CurPtr, EndPtr-CurPtr);
}

raw_ostream &raw_ostream::operator<<(long long N) {
  if (N < 0 && writeBase == 10) {
    *this << '-';
    // Avoid undefined behavior on INT64_MIN with a cast.
    N = -(unsigned long long)N;
  }

  return this->operator<<(static_cast<unsigned long long>(N));
}

// HLSL Change Starts - Generalize non-base10 printing.
raw_ostream &raw_ostream::write_hex(unsigned long long N) {
  int oldBase = writeBase;
  writeBase = 16;
  raw_ostream &rv = write_base(N);
  writeBase = oldBase;
  return rv;
}

raw_ostream &raw_ostream::write_base(unsigned long long N) {
  // Zero is a special case.
  if (N == 0)
    return *this << '0';

  char NumberBuffer[20];
  char *EndPtr = NumberBuffer + sizeof(NumberBuffer);
  char *CurPtr = EndPtr;

  while (N) {
    uintptr_t x = N % writeBase;
    *--CurPtr = (x < 10 ? '0' + x : 'a' + x - 10);
    N /= writeBase;
  }

  return write(CurPtr, EndPtr - CurPtr);
}
// HLSL Change Ends

raw_ostream &raw_ostream::write_escaped(StringRef Str,
                                        bool UseHexEscapes) {
  for (unsigned i = 0, e = Str.size(); i != e; ++i) {
    unsigned char c = Str[i];

    switch (c) {
    case '\\':
      *this << '\\' << '\\';
      break;
    case '\t':
      *this << '\\' << 't';
      break;
    case '\n':
      *this << '\\' << 'n';
      break;
    case '"':
      *this << '\\' << '"';
      break;
    default:
      if (std::isprint(c)) {
        *this << c;
        break;
      }

      // Write out the escaped representation.
      if (UseHexEscapes) {
        *this << '\\' << 'x';
        *this << hexdigit((c >> 4 & 0xF));
        *this << hexdigit((c >> 0) & 0xF);
      } else {
        // Always use a full 3-character octal escape.
        *this << '\\';
        *this << char('0' + ((c >> 6) & 7));
        *this << char('0' + ((c >> 3) & 7));
        *this << char('0' + ((c >> 0) & 7));
      }
    }
  }

  return *this;
}

raw_ostream &raw_ostream::operator<<(const void *P) {
  *this << '0' << 'x';

  return write_hex((uintptr_t) P);
}

raw_ostream &raw_ostream::operator<<(double N) {
#ifdef _WIN32
  // On MSVCRT and compatible, output of %e is incompatible to Posix
  // by default. Number of exponent digits should be at least 2. "%+03d"
  // FIXME: Implement our formatter to here or Support/Format.h!
#if __cplusplus >= 201103L && defined(__MINGW32__)
  // FIXME: It should be generic to C++11.
  if (N == 0.0 && std::signbit(N))
    return *this << "-0.000000e+00";
#else
  int fpcl = _fpclass(N);

  // negative zero
  if (fpcl == _FPCLASS_NZ)
    return *this << "-0.000000e+00";
#endif

  char buf[16];
  unsigned len;
  len = format("%e", N).snprint(buf, sizeof(buf));
  if (len <= sizeof(buf) - 2) {
    if (len >= 5 && buf[len - 5] == 'e' && buf[len - 3] == '0') {
      int cs = buf[len - 4];
      if (cs == '+' || cs == '-') {
        int c1 = buf[len - 2];
        int c0 = buf[len - 1];
        if (isdigit(static_cast<unsigned char>(c1)) &&
            isdigit(static_cast<unsigned char>(c0))) {
          // Trim leading '0': "...e+012" -> "...e+12\0"
          buf[len - 3] = c1;
          buf[len - 2] = c0;
          buf[--len] = 0;
        }
      }
    }
    return this->operator<<(buf);
  }
#endif
  return this->operator<<(format("%e", N));
}



void raw_ostream::flush_nonempty() {
  assert(OutBufCur > OutBufStart && "Invalid call to flush_nonempty.");
  size_t Length = OutBufCur - OutBufStart;
  OutBufCur = OutBufStart;
  write_impl(OutBufStart, Length);
}

raw_ostream &raw_ostream::write(unsigned char C) {
  // Group exceptional cases into a single branch.
  if (LLVM_UNLIKELY(OutBufCur >= OutBufEnd)) {
    if (LLVM_UNLIKELY(!OutBufStart)) {
      if (BufferMode == Unbuffered) {
        write_impl(reinterpret_cast<char*>(&C), 1);
        return *this;
      }
      // Set up a buffer and start over.
      SetBuffered();
      return write(C);
    }

    flush_nonempty();
  }

  *OutBufCur++ = C;
  return *this;
}

raw_ostream &raw_ostream::write(const char *Ptr, size_t Size) {
  // Group exceptional cases into a single branch.
  if (LLVM_UNLIKELY(size_t(OutBufEnd - OutBufCur) < Size)) {
    if (LLVM_UNLIKELY(!OutBufStart)) {
      if (BufferMode == Unbuffered) {
        write_impl(Ptr, Size);
        return *this;
      }
      // Set up a buffer and start over.
      SetBuffered();
      return write(Ptr, Size);
    }

    size_t NumBytes = OutBufEnd - OutBufCur;

    // If the buffer is empty at this point we have a string that is larger
    // than the buffer. Directly write the chunk that is a multiple of the
    // preferred buffer size and put the remainder in the buffer.
    if (LLVM_UNLIKELY(OutBufCur == OutBufStart)) {
      assert(NumBytes != 0 && "undefined behavior");
      size_t BytesToWrite = Size - (Size % NumBytes);
      write_impl(Ptr, BytesToWrite);
      size_t BytesRemaining = Size - BytesToWrite;
      if (BytesRemaining > size_t(OutBufEnd - OutBufCur)) {
        // Too much left over to copy into our buffer.
        return write(Ptr + BytesToWrite, BytesRemaining);
      }
      copy_to_buffer(Ptr + BytesToWrite, BytesRemaining);
      return *this;
    }

    // We don't have enough space in the buffer to fit the string in. Insert as
    // much as possible, flush and start over with the remainder.
    copy_to_buffer(Ptr, NumBytes);
    flush_nonempty();
    return write(Ptr + NumBytes, Size - NumBytes);
  }

  copy_to_buffer(Ptr, Size);

  return *this;
}

void raw_ostream::copy_to_buffer(const char *Ptr, size_t Size) {
  assert(Size <= size_t(OutBufEnd - OutBufCur) && "Buffer overrun!");

  // Handle short strings specially, memcpy isn't very good at very short
  // strings.
  switch (Size) {
  case 4: OutBufCur[3] = Ptr[3]; // FALL THROUGH
  case 3: OutBufCur[2] = Ptr[2]; // FALL THROUGH
  case 2: OutBufCur[1] = Ptr[1]; // FALL THROUGH
  case 1: OutBufCur[0] = Ptr[0]; // FALL THROUGH
  case 0: break;
  default:
    memcpy(OutBufCur, Ptr, Size);
    break;
  }

  OutBufCur += Size;
}

// Formatted output.
raw_ostream &raw_ostream::operator<<(const format_object_base &Fmt) {
  // If we have more than a few bytes left in our output buffer, try
  // formatting directly onto its end.
  size_t NextBufferSize = 127;
  size_t BufferBytesLeft = OutBufEnd - OutBufCur;
  if (BufferBytesLeft > 3) {
    size_t BytesUsed = Fmt.print(OutBufCur, BufferBytesLeft);

    // Common case is that we have plenty of space.
    if (BytesUsed <= BufferBytesLeft) {
      OutBufCur += BytesUsed;
      return *this;
    }

    // Otherwise, we overflowed and the return value tells us the size to try
    // again with.
    NextBufferSize = BytesUsed;
  }

  // If we got here, we didn't have enough space in the output buffer for the
  // string.  Try printing into a SmallVector that is resized to have enough
  // space.  Iterate until we win.
  SmallVector<char, 128> V;

  while (1) {
    V.resize(NextBufferSize);

    // Try formatting into the SmallVector.
    size_t BytesUsed = Fmt.print(V.data(), NextBufferSize);

    // If BytesUsed fit into the vector, we win.
    if (BytesUsed <= NextBufferSize)
      return write(V.data(), BytesUsed);

    // Otherwise, try again with a new size.
    assert(BytesUsed > NextBufferSize && "Didn't grow buffer!?");
    NextBufferSize = BytesUsed;
  }
}

raw_ostream &raw_ostream::operator<<(const FormattedString &FS) {
  unsigned Len = FS.Str.size(); 
  int PadAmount = FS.Width - Len;
  if (FS.RightJustify && (PadAmount > 0))
    this->indent(PadAmount);
  this->operator<<(FS.Str);
  if (!FS.RightJustify && (PadAmount > 0))
    this->indent(PadAmount);
  return *this;
}

raw_ostream &raw_ostream::operator<<(const FormattedNumber &FN) {
  if (FN.Hex) {
    unsigned Nibbles = (64 - countLeadingZeros(FN.HexValue)+3)/4;
    unsigned PrefixChars = FN.HexPrefix ? 2 : 0;
    unsigned Width = std::max(FN.Width, Nibbles + PrefixChars);

    char NumberBuffer[20] = "0x0000000000000000";
    if (!FN.HexPrefix)
      NumberBuffer[1] = '0';
    char *EndPtr = NumberBuffer+Width;
    char *CurPtr = EndPtr;
    const char A = FN.Upper ? 'A' : 'a';
    unsigned long long N = FN.HexValue;
    while (N) {
      uintptr_t x = N % 16;
      assert(CurPtr > NumberBuffer && "else FN values inconsistent"); // HLSL Change
      _Analysis_assume_(CurPtr > NumberBuffer); // HLSL Change
      *--CurPtr = (x < 10 ? '0' + x : A + x - 10);
      N /= 16;
    }

    return write(NumberBuffer, Width);
  } else {
    // Zero is a special case.
    if (FN.DecValue == 0) {
      this->indent(FN.Width-1);
      return *this << '0';
    }
    char NumberBuffer[32];
    char *EndPtr = NumberBuffer+sizeof(NumberBuffer);
    char *CurPtr = EndPtr;
    bool Neg = (FN.DecValue < 0);
    uint64_t N = Neg ? -static_cast<uint64_t>(FN.DecValue) : FN.DecValue;
    while (N) {
      *--CurPtr = '0' + char(N % 10);
      N /= 10;
    }
    int Len = EndPtr - CurPtr;
    int Pad = FN.Width - Len;
    if (Neg) 
      --Pad;
    if (Pad > 0)
      this->indent(Pad);
    if (Neg)
      *this << '-';
    return write(CurPtr, Len);
  }
}

// HLSL Change Starts - Add handling of numerical base IO manipulators.
raw_ostream &raw_ostream::
operator<<(std::ios_base &(__cdecl*iomanip)(std::ios_base &)) {
  if (iomanip == std::hex)
    writeBase = 16;
  else if (iomanip == std::oct)
    writeBase = 8;
  else
    writeBase = 10;

  return *this;
}
// HLSL Change Ends

/// indent - Insert 'NumSpaces' spaces.
raw_ostream &raw_ostream::indent(unsigned NumSpaces) {
  static const char Spaces[] = "                                "
                               "                                "
                               "                ";

  // Usually the indentation is small, handle it with a fastpath.
  if (NumSpaces < array_lengthof(Spaces))
    return write(Spaces, NumSpaces);

  while (NumSpaces) {
    unsigned NumToWrite = std::min(NumSpaces,
                                   (unsigned)array_lengthof(Spaces)-1);
    write(Spaces, NumToWrite);
    NumSpaces -= NumToWrite;
  }
  return *this;
}


//===----------------------------------------------------------------------===//
//  Formatted Output
//===----------------------------------------------------------------------===//

// Out of line virtual method.
void format_object_base::home() {
}

//===----------------------------------------------------------------------===//
//  raw_fd_ostream
//===----------------------------------------------------------------------===//

static int getFD(StringRef Filename, std::error_code &EC,
                 sys::fs::OpenFlags Flags) {
  // Handle "-" as stdout. Note that when we do this, we consider ourself
  // the owner of stdout. This means that we can do things like close the
  // file descriptor when we're done and set the "binary" flag globally.
  if (Filename == "-") {
    EC = std::error_code();
    // If user requested binary then put stdout into binary mode if
    // possible.
    if (!(Flags & sys::fs::F_Text))
      sys::ChangeStdoutToBinary();
    return STDOUT_FILENO;
  }

  int FD;
  EC = sys::fs::openFileForWrite(Filename, FD, Flags);
  if (EC)
    return -1;

  return FD;
}

raw_fd_ostream::raw_fd_ostream(StringRef Filename, std::error_code &EC,
                               sys::fs::OpenFlags Flags)
    : raw_fd_ostream(getFD(Filename, EC, Flags), true) {}

/// FD is the file descriptor that this writes to.  If ShouldClose is true, this
/// closes the file when the stream is destroyed.
raw_fd_ostream::raw_fd_ostream(int fd, bool shouldClose, bool unbuffered)
    : raw_pwrite_stream(unbuffered), FD(fd), ShouldClose(shouldClose),
      Error(false), UseAtomicWrites(false) {
  if (FD < 0 ) {
    ShouldClose = false;
    return;
  }

  // Get the starting position.
  off_t loc = llvm::sys::fs::msf_lseek(FD, 0, SEEK_CUR);  // HLSL Change - msf_lseek
#ifdef LLVM_ON_WIN32
  // MSVCRT's _lseek(SEEK_CUR) doesn't return -1 for pipes.
  sys::fs::file_status Status;
  std::error_code EC = status(FD, Status);
  SupportsSeeking = !EC && Status.type() == sys::fs::file_type::regular_file;
#else
  SupportsSeeking = loc != (off_t)-1;
#endif
  if (!SupportsSeeking)
    pos = 0;
  else
    pos = static_cast<uint64_t>(loc);
}

raw_fd_ostream::~raw_fd_ostream() {
  if (FD >= 0) {
    flush();
    if (ShouldClose && sys::Process::SafelyCloseFileDescriptor(FD))
      error_detected();
  }

#ifdef __MINGW32__
  // On mingw, global dtors should not call exit().
  // report_fatal_error() invokes exit(). We know report_fatal_error()
  // might not write messages to stderr when any errors were detected
  // on FD == 2.
  if (FD == 2) return;
#endif

  // If there are any pending errors, report them now. Clients wishing
  // to avoid report_fatal_error calls should check for errors with
  // has_error() and clear the error flag with clear_error() before
  // destructing raw_ostream objects which may have errors.
  if (has_error() && ShouldClose) // HLSL Change - ignore error on !ShouldClose
    report_fatal_error("IO failure on output stream.", /*GenCrashDiag=*/false);
}


void raw_fd_ostream::write_impl(const char *Ptr, size_t Size) {
  assert(FD >= 0 && "File already closed.");
  pos += Size;

  do {
    ssize_t ret;

    // Check whether we should attempt to use atomic writes.
    if (LLVM_LIKELY(!UseAtomicWrites)) {
      ret = llvm::sys::fs::msf_write(FD, Ptr, Size);
    } else {
      // Use ::writev() where available.
#if defined(HAVE_WRITEV)
      const void *Addr = static_cast<const void *>(Ptr);
      struct iovec IOV = {const_cast<void *>(Addr), Size };
      ret = ::writev(FD, &IOV, 1);
#else
      ret = llvm::sys::fs::msf_write(FD, Ptr, Size);
#endif
    }

    if (ret < 0) {
      // If it's a recoverable error, swallow it and retry the write.
      //
      // Ideally we wouldn't ever see EAGAIN or EWOULDBLOCK here, since
      // raw_ostream isn't designed to do non-blocking I/O. However, some
      // programs, such as old versions of bjam, have mistakenly used
      // O_NONBLOCK. For compatibility, emulate blocking semantics by
      // spinning until the write succeeds. If you don't want spinning,
      // don't use O_NONBLOCK file descriptors with raw_ostream.
      if (errno == EINTR || errno == EAGAIN
#ifdef EWOULDBLOCK
          || errno == EWOULDBLOCK
#endif
          )
        continue;

      // Otherwise it's a non-recoverable error. Note it and quit.
      error_detected();
      break;
    }

    // The write may have written some or all of the data. Update the
    // size and buffer pointer to reflect the remainder that needs
    // to be written. If there are no bytes left, we're done.
    Ptr += ret;
    Size -= ret;
  } while (Size > 0);
}

void raw_fd_ostream::close() {
  assert(ShouldClose);
  ShouldClose = false;
  flush();
  if (sys::Process::SafelyCloseFileDescriptor(FD))
    error_detected();
  FD = -1;
}

uint64_t raw_fd_ostream::seek(uint64_t off) {
  flush();
  pos = llvm::sys::fs::msf_lseek(FD, off, SEEK_SET);  // HLSL Change
  if (pos == (uint64_t)-1)
    error_detected();
  return pos;
}

void raw_fd_ostream::pwrite_impl(const char *Ptr, size_t Size,
                                 uint64_t Offset) {
  uint64_t Pos = tell();
  seek(Offset);
  write(Ptr, Size);
  seek(Pos);
}

size_t raw_fd_ostream::preferred_buffer_size() const {
#if !defined(_MSC_VER) && !defined(__MINGW32__) && !defined(__minix)
  // Windows and Minix have no st_blksize.
  assert(FD >= 0 && "File not yet open!");
  struct stat statbuf;
  if (fstat(FD, &statbuf) != 0)
    return 0;

  // If this is a terminal, don't use buffering. Line buffering
  // would be a more traditional thing to do, but it's not worth
  // the complexity.
  if (S_ISCHR(statbuf.st_mode) && isatty(FD))
    return 0;
  // Return the preferred block size.
  return statbuf.st_blksize;
#else
  return raw_ostream::preferred_buffer_size();
#endif
}

raw_ostream &raw_fd_ostream::changeColor(enum Colors colors, bool bold,
                                         bool bg) {
  if (sys::Process::ColorNeedsFlush())
    flush();
  const char *colorcode =
    (colors == SAVEDCOLOR) ? sys::Process::OutputBold(bg)
    : sys::Process::OutputColor(colors, bold, bg);
  if (colorcode) {
    size_t len = strlen(colorcode);
    write(colorcode, len);
    // don't account colors towards output characters
    pos -= len;
  }
  return *this;
}

raw_ostream &raw_fd_ostream::resetColor() {
  if (sys::Process::ColorNeedsFlush())
    flush();
  const char *colorcode = sys::Process::ResetColor();
  if (colorcode) {
    size_t len = strlen(colorcode);
    write(colorcode, len);
    // don't account colors towards output characters
    pos -= len;
  }
  return *this;
}

raw_ostream &raw_fd_ostream::reverseColor() {
  if (sys::Process::ColorNeedsFlush())
    flush();
  const char *colorcode = sys::Process::OutputReverse();
  if (colorcode) {
    size_t len = strlen(colorcode);
    write(colorcode, len);
    // don't account colors towards output characters
    pos -= len;
  }
  return *this;
}

bool raw_fd_ostream::is_displayed() const {
  return sys::Process::FileDescriptorIsDisplayed(FD);
}

bool raw_fd_ostream::has_colors() const {
  return sys::Process::FileDescriptorHasColors(FD);
}

//===----------------------------------------------------------------------===//
//  outs(), errs(), nulls()
//===----------------------------------------------------------------------===//

/// outs() - This returns a reference to a raw_ostream for standard output.
/// Use it like: outs() << "foo" << "bar";
raw_ostream &llvm::outs() {
  // Set buffer settings to model stdout behavior.
  // Delete the file descriptor when the program exits, forcing error
  // detection. If you don't want this behavior, don't use outs().
  std::error_code EC;
  static raw_fd_ostream S("-", EC, sys::fs::F_None);
  assert(!EC);
  return S;
}

/// errs() - This returns a reference to a raw_ostream for standard error.
/// Use it like: errs() << "foo" << "bar";
raw_ostream &llvm::errs() {
  // Set standard error to be unbuffered by default.
  static raw_fd_ostream S(STDERR_FILENO, false, true);
  return S;
}

/// nulls() - This returns a reference to a raw_ostream which discards output.
raw_ostream &llvm::nulls() {
  static raw_null_ostream S;
  return S;
}


//===----------------------------------------------------------------------===//
//  raw_string_ostream
//===----------------------------------------------------------------------===//

raw_string_ostream::~raw_string_ostream() {
#if 0 // HLSL Change Starts
  flush();
#else
  // C++ and exception in destructors don't play nice. The proper pattern
  // here is to have the raw_string_ostream's owner flush before destruction
  // and take appropriate action, like throwing or returning an error value.
  try {
    flush();
  }
  catch (const std::bad_alloc &) {
    // Don't std::terminate()
  }
#endif // HLSL Change Ends
}

void raw_string_ostream::write_impl(const char *Ptr, size_t Size) {
  OS.append(Ptr, Size);
}

//===----------------------------------------------------------------------===//
//  raw_svector_ostream
//===----------------------------------------------------------------------===//

// The raw_svector_ostream implementation uses the SmallVector itself as the
// buffer for the raw_ostream. We guarantee that the raw_ostream buffer is
// always pointing past the end of the vector, but within the vector
// capacity. This allows raw_ostream to write directly into the correct place,
// and we only need to set the vector size when the data is flushed.

raw_svector_ostream::raw_svector_ostream(SmallVectorImpl<char> &O, unsigned)
    : OS(O) {}

raw_svector_ostream::raw_svector_ostream(SmallVectorImpl<char> &O) : OS(O) {
  init();
}

void raw_svector_ostream::init() {
  // Set up the initial external buffer. We make sure that the buffer has at
  // least 128 bytes free; raw_ostream itself only requires 64, but we want to
  // make sure that we don't grow the buffer unnecessarily on destruction (when
  // the data is flushed). See the FIXME below.
  OS.reserve(OS.size() + 128);
  SetBuffer(OS.end(), OS.capacity() - OS.size());
}

raw_svector_ostream::~raw_svector_ostream() {
  // FIXME: Prevent resizing during this flush().
  flush();
}

void raw_svector_ostream::pwrite_impl(const char *Ptr, size_t Size,
                                      uint64_t Offset) {
  flush();
  memcpy(OS.begin() + Offset, Ptr, Size);
}

/// resync - This is called when the SmallVector we're appending to is changed
/// outside of the raw_svector_ostream's control.  It is only safe to do this
/// if the raw_svector_ostream has previously been flushed.
void raw_svector_ostream::resync() {
  assert(GetNumBytesInBuffer() == 0 && "Didn't flush before mutating vector");

  if (OS.capacity() - OS.size() < 64)
    OS.reserve(OS.capacity() * 2);
  SetBuffer(OS.end(), OS.capacity() - OS.size());
}

void raw_svector_ostream::write_impl(const char *Ptr, size_t Size) {
  if (Ptr == OS.end()) {
    // Grow the buffer to include the scratch area without copying.
    size_t NewSize = OS.size() + Size;
    assert(NewSize <= OS.capacity() && "Invalid write_impl() call!");
    OS.set_size(NewSize);
  } else {
    assert(!GetNumBytesInBuffer());
    OS.append(Ptr, Ptr + Size);
  }

  OS.reserve(OS.size() + 64);
  SetBuffer(OS.end(), OS.capacity() - OS.size());
}

uint64_t raw_svector_ostream::current_pos() const {
  return OS.size();
}

StringRef raw_svector_ostream::str() {
  flush();
  return StringRef(OS.begin(), OS.size());
}

//===----------------------------------------------------------------------===//
//  raw_null_ostream
//===----------------------------------------------------------------------===//

raw_null_ostream::~raw_null_ostream() {
#ifndef NDEBUG
  // ~raw_ostream asserts that the buffer is empty. This isn't necessary
  // with raw_null_ostream, but it's better to have raw_null_ostream follow
  // the rules than to change the rules just for raw_null_ostream.
  flush();
#endif
}

void raw_null_ostream::write_impl(const char *Ptr, size_t Size) {
}

uint64_t raw_null_ostream::current_pos() const {
  return 0;
}

void raw_null_ostream::pwrite_impl(const char *Ptr, size_t Size,
                                   uint64_t Offset) {}
