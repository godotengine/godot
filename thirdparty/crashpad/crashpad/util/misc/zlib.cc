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

#include "util/misc/zlib.h"

#include "base/logging.h"
#include "base/strings/stringprintf.h"
#include "third_party/zlib/zlib_crashpad.h"

namespace crashpad {

int ZlibWindowBitsWithGzipWrapper(int window_bits) {
  // See the documentation for deflateInit2() and inflateInit2() in <zlib.h>. 0
  // is only valid during decompression.

  DCHECK(window_bits == 0 || (window_bits >= 8 && window_bits <= 15))
      << window_bits;

  return 16 + window_bits;
}

std::string ZlibErrorString(int zr) {
  return base::StringPrintf("%s (%d)", zError(zr), zr);
}

}  // namespace crashpad
