// Copyright 2017 The Crashpad Authors. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "util/net/http_body_gzip.h"

#include <utility>

#include "base/logging.h"
#include "base/numerics/safe_conversions.h"
#include "third_party/zlib/zlib_crashpad.h"
#include "util/misc/zlib.h"

namespace crashpad {

GzipHTTPBodyStream::GzipHTTPBodyStream(std::unique_ptr<HTTPBodyStream> source)
    : input_(),
      source_(std::move(source)),
      z_stream_(new z_stream()),
      state_(State::kUninitialized) {}

GzipHTTPBodyStream::~GzipHTTPBodyStream() {
  DCHECK(state_ == State::kUninitialized ||
         state_ == State::kFinished ||
         state_ == State::kError);
}

FileOperationResult GzipHTTPBodyStream::GetBytesBuffer(uint8_t* buffer,
                                                       size_t max_len) {
  if (state_ == State::kError) {
    return -1;
  }

  if (state_ == State::kFinished) {
    return 0;
  }

  if (state_ == State::kUninitialized) {
    z_stream_->zalloc = Z_NULL;
    z_stream_->zfree = Z_NULL;
    z_stream_->opaque = Z_NULL;

    // The default values for zlib’s internal MAX_WBITS and DEF_MEM_LEVEL. These
    // are the values that deflateInit() would use, but they’re not exported
    // from zlib. deflateInit2() is used instead of deflateInit() to get the
    // gzip wrapper.
    constexpr int kZlibMaxWindowBits = 15;
    constexpr int kZlibDefaultMemoryLevel = 8;

    int zr = deflateInit2(z_stream_.get(),
                          Z_DEFAULT_COMPRESSION,
                          Z_DEFLATED,
                          ZlibWindowBitsWithGzipWrapper(kZlibMaxWindowBits),
                          kZlibDefaultMemoryLevel,
                          Z_DEFAULT_STRATEGY);
    if (zr != Z_OK) {
      LOG(ERROR) << "deflateInit2: " << ZlibErrorString(zr);
      state_ = State::kError;
      return -1;
    }

    state_ = State::kOperating;
  }

  z_stream_->next_out = buffer;
  z_stream_->avail_out = base::saturated_cast<uInt>(max_len);

  while (state_ != State::kFinished && z_stream_->avail_out > 0) {
    if (state_ != State::kInputEOF && z_stream_->avail_in == 0) {
      FileOperationResult input_bytes =
          source_->GetBytesBuffer(input_, sizeof(input_));
      if (input_bytes == -1) {
        Done(State::kError);
        return -1;
      }

      if (input_bytes == 0) {
        state_ = State::kInputEOF;
      }

      z_stream_->next_in = input_;
      z_stream_->avail_in = base::checked_cast<uInt>(input_bytes);
    }

    int zr = deflate(z_stream_.get(),
                     state_ == State::kInputEOF ? Z_FINISH : Z_NO_FLUSH);
    if (state_ == State::kInputEOF && zr == Z_STREAM_END) {
      Done(State::kFinished);
      if (state_ == State::kError) {
        return -1;
      }
    } else if (zr != Z_OK) {
      LOG(ERROR) << "deflate: " << ZlibErrorString(zr);
      Done(State::kError);
      return -1;
    }
  }

  DCHECK_LE(z_stream_->avail_out, max_len);
  return max_len - z_stream_->avail_out;
}

void GzipHTTPBodyStream::Done(State state) {
  DCHECK(state_ == State::kOperating || state_ == State::kInputEOF) << state_;
  DCHECK(state == State::kFinished || state == State::kError) << state;

  int zr = deflateEnd(z_stream_.get());
  if (zr != Z_OK) {
    LOG(ERROR) << "deflateEnd: " << ZlibErrorString(zr);
    state_ = State::kError;
  } else {
    state_ = state;
  }
}

}  // namespace crashpad
