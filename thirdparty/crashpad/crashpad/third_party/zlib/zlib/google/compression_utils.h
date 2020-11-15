// Copyright 2014 The Chromium Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#ifndef THIRD_PARTY_ZLIB_GOOGLE_COMPRESSION_UTILS_H_
#define THIRD_PARTY_ZLIB_GOOGLE_COMPRESSION_UTILS_H_

#include <string>

namespace compression {

// Compresses the data in |input| using gzip, storing the result in |output|.
// |input| and |output| are allowed to be the same string (in-place operation).
bool GzipCompress(const std::string& input, std::string* output);

// Uncompresses the data in |input| using gzip, storing the result in |output|.
// |input| and |output| are allowed to be the same string (in-place operation).
bool GzipUncompress(const std::string& input, std::string* output);

}  // namespace compression

#endif  // THIRD_PARTY_ZLIB_GOOGLE_COMPRESSION_UTILS_H_
