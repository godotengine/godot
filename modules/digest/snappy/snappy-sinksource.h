// Copyright 2011 Google Inc. All Rights Reserved.
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
//     * Neither the name of Google Inc. nor the names of its
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

#ifndef UTIL_SNAPPY_SNAPPY_SINKSOURCE_H_
#define UTIL_SNAPPY_SNAPPY_SINKSOURCE_H_

#include <stddef.h>


namespace snappy {

// A Sink is an interface that consumes a sequence of bytes.
class Sink {
 public:
  Sink() { }
  virtual ~Sink();

  // Append "bytes[0,n-1]" to this.
  virtual void Append(const char* bytes, size_t n) = 0;

  // Returns a writable buffer of the specified length for appending.
  // May return a pointer to the caller-owned scratch buffer which
  // must have at least the indicated length.  The returned buffer is
  // only valid until the next operation on this Sink.
  //
  // After writing at most "length" bytes, call Append() with the
  // pointer returned from this function and the number of bytes
  // written.  Many Append() implementations will avoid copying
  // bytes if this function returned an internal buffer.
  //
  // If a non-scratch buffer is returned, the caller may only pass a
  // prefix of it to Append().  That is, it is not correct to pass an
  // interior pointer of the returned array to Append().
  //
  // The default implementation always returns the scratch buffer.
  virtual char* GetAppendBuffer(size_t length, char* scratch);


 private:
  // No copying
  Sink(const Sink&);
  void operator=(const Sink&);
};

// A Source is an interface that yields a sequence of bytes
class Source {
 public:
  Source() { }
  virtual ~Source();

  // Return the number of bytes left to read from the source
  virtual size_t Available() const = 0;

  // Peek at the next flat region of the source.  Does not reposition
  // the source.  The returned region is empty iff Available()==0.
  //
  // Returns a pointer to the beginning of the region and store its
  // length in *len.
  //
  // The returned region is valid until the next call to Skip() or
  // until this object is destroyed, whichever occurs first.
  //
  // The returned region may be larger than Available() (for example
  // if this ByteSource is a view on a substring of a larger source).
  // The caller is responsible for ensuring that it only reads the
  // Available() bytes.
  virtual const char* Peek(size_t* len) = 0;

  // Skip the next n bytes.  Invalidates any buffer returned by
  // a previous call to Peek().
  // REQUIRES: Available() >= n
  virtual void Skip(size_t n) = 0;

 private:
  // No copying
  Source(const Source&);
  void operator=(const Source&);
};

// A Source implementation that yields the contents of a flat array
class ByteArraySource : public Source {
 public:
  ByteArraySource(const char* p, size_t n) : ptr_(p), left_(n) { }
  virtual ~ByteArraySource();
  virtual size_t Available() const;
  virtual const char* Peek(size_t* len);
  virtual void Skip(size_t n);
 private:
  const char* ptr_;
  size_t left_;
};

// A Sink implementation that writes to a flat array without any bound checks.
class UncheckedByteArraySink : public Sink {
 public:
  explicit UncheckedByteArraySink(char* dest) : dest_(dest) { }
  virtual ~UncheckedByteArraySink();
  virtual void Append(const char* data, size_t n);
  virtual char* GetAppendBuffer(size_t len, char* scratch);

  // Return the current output pointer so that a caller can see how
  // many bytes were produced.
  // Note: this is not a Sink method.
  char* CurrentDestination() const { return dest_; }
 private:
  char* dest_;
};


}

#endif  // UTIL_SNAPPY_SNAPPY_SINKSOURCE_H_
