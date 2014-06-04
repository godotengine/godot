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

#include <string.h>

#include "snappy-sinksource.h"

namespace snappy {

Source::~Source() { }

Sink::~Sink() { }

char* Sink::GetAppendBuffer(size_t length, char* scratch) {
  return scratch;
}

ByteArraySource::~ByteArraySource() { }

size_t ByteArraySource::Available() const { return left_; }

const char* ByteArraySource::Peek(size_t* len) {
  *len = left_;
  return ptr_;
}

void ByteArraySource::Skip(size_t n) {
  left_ -= n;
  ptr_ += n;
}

UncheckedByteArraySink::~UncheckedByteArraySink() { }

void UncheckedByteArraySink::Append(const char* data, size_t n) {
  // Do no copying if the caller filled in the result of GetAppendBuffer()
  if (data != dest_) {
    memcpy(dest_, data, n);
  }
  dest_ += n;
}

char* UncheckedByteArraySink::GetAppendBuffer(size_t len, char* scratch) {
  return dest_;
}

}
