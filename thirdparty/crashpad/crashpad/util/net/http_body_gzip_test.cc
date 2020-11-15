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

#include <string.h>

#include <algorithm>
#include <memory>
#include <string>
#include <utility>

#include "base/macros.h"
#include "base/rand_util.h"
#include "base/numerics/safe_conversions.h"
#include "gtest/gtest.h"
#include "third_party/zlib/zlib_crashpad.h"
#include "util/misc/zlib.h"
#include "util/net/http_body.h"

namespace crashpad {
namespace test {
namespace {

class ScopedZlibInflateStream {
 public:
  explicit ScopedZlibInflateStream(z_stream* zlib) : zlib_(zlib) {}
  ~ScopedZlibInflateStream() {
    int zr = inflateEnd(zlib_);
    EXPECT_EQ(zr, Z_OK) << "inflateEnd: " << ZlibErrorString(zr);
  }

 private:
  z_stream* zlib_;  // weak
  DISALLOW_COPY_AND_ASSIGN(ScopedZlibInflateStream);
};

void GzipInflate(const std::string& compressed,
                 std::string* decompressed,
                 size_t buf_size) {
  decompressed->clear();

  // There’s got to be at least a small buffer.
  buf_size = std::max(buf_size, static_cast<size_t>(1));

  std::unique_ptr<uint8_t[]> buf(new uint8_t[buf_size]);
  z_stream zlib = {};
  zlib.zalloc = Z_NULL;
  zlib.zfree = Z_NULL;
  zlib.opaque = Z_NULL;
  zlib.next_in = reinterpret_cast<Bytef*>(const_cast<char*>(&compressed[0]));
  zlib.avail_in = base::checked_cast<uInt>(compressed.size());
  zlib.next_out = buf.get();
  zlib.avail_out = base::checked_cast<uInt>(buf_size);

  int zr = inflateInit2(&zlib, ZlibWindowBitsWithGzipWrapper(0));
  ASSERT_EQ(zr, Z_OK) << "inflateInit2: " << ZlibErrorString(zr);
  ScopedZlibInflateStream zlib_inflate(&zlib);

  zr = inflate(&zlib, Z_FINISH);
  ASSERT_EQ(zr, Z_STREAM_END) << "inflate: " << ZlibErrorString(zr);

  ASSERT_LE(zlib.avail_out, buf_size);
  decompressed->assign(reinterpret_cast<char*>(buf.get()),
                       buf_size - zlib.avail_out);
}

void TestGzipDeflateInflate(const std::string& string) {
  std::unique_ptr<HTTPBodyStream> string_stream(
      new StringHTTPBodyStream(string));
  GzipHTTPBodyStream gzip_stream(std::move(string_stream));

  // The minimum size of a gzip wrapper per RFC 1952: a 10-byte header and an
  // 8-byte trailer.
  constexpr size_t kGzipHeaderSize = 18;

  // Per https://zlib.net/zlib_tech.html, in the worst case, zlib will store
  // uncompressed data as-is, at an overhead of 5 bytes per 16384-byte block.
  // Zero-length input will “compress” to a 2-byte zlib stream. Add the overhead
  // of the gzip wrapper, assuming no optional fields are present.
  size_t buf_size =
      string.size() + kGzipHeaderSize +
      (string.empty() ? 2 : (((string.size() + 16383) / 16384) * 5));
  std::unique_ptr<uint8_t[]> buf(new uint8_t[buf_size]);
  FileOperationResult compressed_bytes =
      gzip_stream.GetBytesBuffer(buf.get(), buf_size);
  ASSERT_NE(compressed_bytes, -1);
  ASSERT_LE(static_cast<size_t>(compressed_bytes), buf_size);

  // Make sure that the stream is really at EOF.
  uint8_t eof_buf[16];
  ASSERT_EQ(gzip_stream.GetBytesBuffer(eof_buf, sizeof(eof_buf)), 0);

  std::string compressed(reinterpret_cast<char*>(buf.get()), compressed_bytes);

  ASSERT_GE(compressed.size(), kGzipHeaderSize);
  EXPECT_EQ(compressed[0], '\37');
  EXPECT_EQ(compressed[1], '\213');
  EXPECT_EQ(compressed[2], Z_DEFLATED);

  std::string decompressed;
  ASSERT_NO_FATAL_FAILURE(
      GzipInflate(compressed, &decompressed, string.size()));

  EXPECT_EQ(decompressed, string);

  // In block mode, compression should be identical.
  string_stream.reset(new StringHTTPBodyStream(string));
  GzipHTTPBodyStream block_gzip_stream(std::move(string_stream));
  uint8_t block_buf[4096];
  std::string block_compressed;
  FileOperationResult block_compressed_bytes;
  while ((block_compressed_bytes = block_gzip_stream.GetBytesBuffer(
              block_buf, sizeof(block_buf))) > 0) {
    block_compressed.append(reinterpret_cast<char*>(block_buf),
                            block_compressed_bytes);
  }
  ASSERT_EQ(block_compressed_bytes, 0);
  EXPECT_EQ(block_compressed, compressed);
}

std::string MakeString(size_t size) {
  std::string string;
  for (size_t i = 0; i < size; ++i) {
    string.append(1, (i % 256) ^ ((i >> 8) % 256));
  }
  return string;
}

constexpr size_t kFourKBytes = 4096;
constexpr size_t kManyBytes = 375017;

TEST(GzipHTTPBodyStream, Empty) {
  TestGzipDeflateInflate(std::string());
}

TEST(GzipHTTPBodyStream, OneByte) {
  TestGzipDeflateInflate(std::string("Z"));
}

TEST(GzipHTTPBodyStream, FourKBytes_NUL) {
  TestGzipDeflateInflate(std::string(kFourKBytes, '\0'));
}

TEST(GzipHTTPBodyStream, ManyBytes_NUL) {
  TestGzipDeflateInflate(std::string(kManyBytes, '\0'));
}

TEST(GzipHTTPBodyStream, FourKBytes_Deterministic) {
  TestGzipDeflateInflate(MakeString(kFourKBytes));
}

TEST(GzipHTTPBodyStream, ManyBytes_Deterministic) {
  TestGzipDeflateInflate(MakeString(kManyBytes));
}

TEST(GzipHTTPBodyStream, FourKBytes_Random) {
  TestGzipDeflateInflate(base::RandBytesAsString(kFourKBytes));
}

TEST(GzipHTTPBodyStream, ManyBytes_Random) {
  TestGzipDeflateInflate(base::RandBytesAsString(kManyBytes));
}

}  // namespace
}  // namespace test
}  // namespace crashpad
