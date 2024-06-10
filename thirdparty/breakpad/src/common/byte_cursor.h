// -*- mode: c++ -*-

// Copyright 2010 Google LLC
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

// Original author: Jim Blandy <jimb@mozilla.com> <jimb@red-bean.com>

// byte_cursor.h: Classes for parsing values from a buffer of bytes.
// The ByteCursor class provides a convenient interface for reading
// fixed-size integers of arbitrary endianness, being thorough about
// checking for buffer overruns.

#ifndef COMMON_BYTE_CURSOR_H_
#define COMMON_BYTE_CURSOR_H_

#include <assert.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <string>

#include "common/using_std_string.h"

namespace google_breakpad {

// A buffer holding a series of bytes.
struct ByteBuffer {
  ByteBuffer() : start(0), end(0) { }
  ByteBuffer(const uint8_t* set_start, size_t set_size)
      : start(set_start), end(set_start + set_size) { }
  ~ByteBuffer() { };

  // Equality operators. Useful in unit tests, and when we're using
  // ByteBuffers to refer to regions of a larger buffer.
  bool operator==(const ByteBuffer& that) const {
    return start == that.start && end == that.end;
  }
  bool operator!=(const ByteBuffer& that) const {
    return start != that.start || end != that.end;
  }

  // Not C++ style guide compliant, but this definitely belongs here.
  size_t Size() const {
    assert(start <= end);
    return end - start;
  }

  const uint8_t* start;
  const uint8_t* end;
};

// A cursor pointing into a ByteBuffer that can parse numbers of various
// widths and representations, strings, and data blocks, advancing through
// the buffer as it goes. All ByteCursor operations check that accesses
// haven't gone beyond the end of the enclosing ByteBuffer.
class ByteCursor {
 public:
  // Create a cursor reading bytes from the start of BUFFER. By default, the
  // cursor reads multi-byte values in little-endian form.
  ByteCursor(const ByteBuffer* buffer, bool big_endian = false)
      : buffer_(buffer), here_(buffer->start),
        big_endian_(big_endian), complete_(true) { }

  // Accessor and setter for this cursor's endianness flag.
  bool big_endian() const { return big_endian_; }
  void set_big_endian(bool big_endian) { big_endian_ = big_endian; }

  // Accessor and setter for this cursor's current position. The setter
  // returns a reference to this cursor.
  const uint8_t* here() const { return here_; }
  ByteCursor& set_here(const uint8_t* here) {
    assert(buffer_->start <= here && here <= buffer_->end);
    here_ = here;
    return *this;
  }

  // Return the number of bytes available to read at the cursor.
  size_t Available() const { return size_t(buffer_->end - here_); }

  // Return true if this cursor is at the end of its buffer.
  bool AtEnd() const { return Available() == 0; }

  // When used as a boolean value this cursor converts to true if all
  // prior reads have been completed, or false if we ran off the end
  // of the buffer.
  operator bool() const { return complete_; }

  // Read a SIZE-byte integer at this cursor, signed if IS_SIGNED is true,
  // unsigned otherwise, using the cursor's established endianness, and set
  // *RESULT to the number. If we read off the end of our buffer, clear
  // this cursor's complete_ flag, and store a dummy value in *RESULT.
  // Return a reference to this cursor.
  template<typename T>
  ByteCursor& Read(size_t size, bool is_signed, T* result) {
    if (CheckAvailable(size)) {
      T v = 0;
      if (big_endian_) {
        for (size_t i = 0; i < size; i++)
          v = (v << 8) + here_[i];
      } else {
        // This loop condition looks weird, but size_t is unsigned, so
        // decrementing i after it is zero yields the largest size_t value.
        for (size_t i = size - 1; i < size; i--)
          v = (v << 8) + here_[i];
      }
      if (is_signed && size < sizeof(T)) {
        size_t sign_bit = (T)1 << (size * 8 - 1);
        v = (v ^ sign_bit) - sign_bit;
      }
      here_ += size;
      *result = v;
    } else {
      *result = (T) 0xdeadbeef;
    }
    return *this;
  }

  // Read an integer, using the cursor's established endianness and
  // *RESULT's size and signedness, and set *RESULT to the number. If we
  // read off the end of our buffer, clear this cursor's complete_ flag.
  // Return a reference to this cursor.
  template<typename T>
  ByteCursor& operator>>(T& result) {
    bool T_is_signed = (T)-1 < 0;
    return Read(sizeof(T), T_is_signed, &result); 
  }

  // Copy the SIZE bytes at the cursor to BUFFER, and advance this
  // cursor to the end of them. If we read off the end of our buffer,
  // clear this cursor's complete_ flag, and set *POINTER to NULL.
  // Return a reference to this cursor.
  ByteCursor& Read(uint8_t* buffer, size_t size) {
    if (CheckAvailable(size)) {
      memcpy(buffer, here_, size);
      here_ += size;
    }
    return *this;
  }

  // Set STR to a copy of the '\0'-terminated string at the cursor. If the
  // byte buffer does not contain a terminating zero, clear this cursor's
  // complete_ flag, and set STR to the empty string. Return a reference to
  // this cursor.
  ByteCursor& CString(string* str) {
    const uint8_t* end
      = static_cast<const uint8_t*>(memchr(here_, '\0', Available()));
    if (end) {
      str->assign(reinterpret_cast<const char*>(here_), end - here_);
      here_ = end + 1;
    } else {
      str->clear();
      here_ = buffer_->end;
      complete_ = false;
    }
    return *this;
  }

  // Like CString(STR), but extract the string from a fixed-width buffer
  // LIMIT bytes long, which may or may not contain a terminating '\0'
  // byte. Specifically:
  //
  // - If there are not LIMIT bytes available at the cursor, clear the
  //   cursor's complete_ flag and set STR to the empty string.
  //
  // - Otherwise, if the LIMIT bytes at the cursor contain any '\0'
  //   characters, set *STR to a copy of the bytes before the first '\0',
  //   and advance the cursor by LIMIT bytes.
  //   
  // - Otherwise, set *STR to a copy of those LIMIT bytes, and advance the
  //   cursor by LIMIT bytes.
  ByteCursor& CString(string* str, size_t limit) {
    if (CheckAvailable(limit)) {
      const uint8_t* end
        = static_cast<const uint8_t*>(memchr(here_, '\0', limit));
      if (end)
        str->assign(reinterpret_cast<const char*>(here_), end - here_);
      else
        str->assign(reinterpret_cast<const char*>(here_), limit);
      here_ += limit;
    } else {
      str->clear();
    }
    return *this;
  }

  // Set *POINTER to point to the SIZE bytes at the cursor, and advance
  // this cursor to the end of them. If SIZE is omitted, don't move the
  // cursor. If we read off the end of our buffer, clear this cursor's
  // complete_ flag, and set *POINTER to NULL. Return a reference to this
  // cursor.
  ByteCursor& PointTo(const uint8_t** pointer, size_t size = 0) {
    if (CheckAvailable(size)) {
      *pointer = here_;
      here_ += size;
    } else {
      *pointer = NULL;
    }
    return *this;
  }

  // Skip SIZE bytes at the cursor. If doing so would advance us off
  // the end of our buffer, clear this cursor's complete_ flag, and
  // set *POINTER to NULL. Return a reference to this cursor.
  ByteCursor& Skip(size_t size) {
    if (CheckAvailable(size))
      here_ += size;
    return *this;
  }

 private:
  // If there are at least SIZE bytes available to read from the buffer,
  // return true. Otherwise, set here_ to the end of the buffer, set
  // complete_ to false, and return false.
  bool CheckAvailable(size_t size) {
    if (Available() >= size) {
      return true;
    } else {
      here_ = buffer_->end;
      complete_ = false;
      return false;
    }
  }

  // The buffer we're reading bytes from.
  const ByteBuffer* buffer_;

  // The next byte within buffer_ that we'll read.
  const uint8_t* here_;

  // True if we should read numbers in big-endian form; false if we
  // should read in little-endian form.
  bool big_endian_;

  // True if we've been able to read all we've been asked to.
  bool complete_;
};

}  // namespace google_breakpad

#endif  // COMMON_BYTE_CURSOR_H_
