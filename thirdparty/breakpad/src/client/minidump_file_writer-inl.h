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

// minidump_file_writer-inl.h: Minidump file writer implementation.
//
// See minidump_file_writer.h for documentation.

#ifndef CLIENT_MINIDUMP_FILE_WRITER_INL_H__
#define CLIENT_MINIDUMP_FILE_WRITER_INL_H__

#include <assert.h>

#include "client/minidump_file_writer.h"
#include "google_breakpad/common/minidump_size.h"

namespace google_breakpad {

template<typename MDType>
inline bool TypedMDRVA<MDType>::Allocate() {
  allocation_state_ = SINGLE_OBJECT;
  return UntypedMDRVA::Allocate(minidump_size<MDType>::size());
}

template<typename MDType>
inline bool TypedMDRVA<MDType>::Allocate(size_t additional) {
  allocation_state_ = SINGLE_OBJECT;
  return UntypedMDRVA::Allocate(minidump_size<MDType>::size() + additional);
}

template<typename MDType>
inline bool TypedMDRVA<MDType>::AllocateArray(size_t count) {
  assert(count);
  allocation_state_ = ARRAY;
  return UntypedMDRVA::Allocate(minidump_size<MDType>::size() * count);
}

template<typename MDType>
inline bool TypedMDRVA<MDType>::AllocateObjectAndArray(size_t count,
                                                       size_t length) {
  assert(count && length);
  allocation_state_ = SINGLE_OBJECT_WITH_ARRAY;
  return UntypedMDRVA::Allocate(minidump_size<MDType>::size() + count * length);
}

template<typename MDType>
inline bool TypedMDRVA<MDType>::CopyIndex(unsigned int index, MDType* item) {
  assert(allocation_state_ == ARRAY);
  return writer_->Copy(
      static_cast<MDRVA>(position_ + index * minidump_size<MDType>::size()), 
      item, minidump_size<MDType>::size());
}

template<typename MDType>
inline bool TypedMDRVA<MDType>::CopyIndexAfterObject(unsigned int index,
                                                     const void* src,
                                                     size_t length) {
  assert(allocation_state_ == SINGLE_OBJECT_WITH_ARRAY);
  return writer_->Copy(
      static_cast<MDRVA>(position_ + minidump_size<MDType>::size() 
                         + index * length),
      src, length);
}

template<typename MDType>
inline bool TypedMDRVA<MDType>::Flush() {
  return writer_->Copy(position_, &data_, minidump_size<MDType>::size());
}

}  // namespace google_breakpad

#endif  // CLIENT_MINIDUMP_FILE_WRITER_INL_H__
