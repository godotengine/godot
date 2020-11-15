// Copyright 2014 The Chromium Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#include "third_party/zlib/google/compression_utils.h"

#include <stddef.h>
#include <stdint.h>
#include <string.h>

#include <vector>

#include "base/bit_cast.h"
#include "base/logging.h"
#include "base/sys_byteorder.h"
#include "third_party/zlib/zlib.h"

namespace {

// The difference in bytes between a zlib header and a gzip header.
const size_t kGzipZlibHeaderDifferenceBytes = 16;

// Pass an integer greater than the following get a gzip header instead of a
// zlib header when calling deflateInit2() and inflateInit2().
const int kWindowBitsToGetGzipHeader = 16;

// This describes the amount of memory zlib uses to compress data. It can go
// from 1 to 9, with 8 being the default. For details, see:
// http://www.zlib.net/manual.html (search for memLevel).
const int kZlibMemoryLevel = 8;

// This code is taken almost verbatim from third_party/zlib/compress.c. The only
// difference is deflateInit2() is called which sets the window bits to be > 16.
// That causes a gzip header to be emitted rather than a zlib header.
int GzipCompressHelper(Bytef* dest,
                       uLongf* dest_length,
                       const Bytef* source,
                       uLong source_length) {
  z_stream stream;

  stream.next_in = bit_cast<Bytef*>(source);
  stream.avail_in = static_cast<uInt>(source_length);
  stream.next_out = dest;
  stream.avail_out = static_cast<uInt>(*dest_length);
  if (static_cast<uLong>(stream.avail_out) != *dest_length)
    return Z_BUF_ERROR;

  stream.zalloc = static_cast<alloc_func>(0);
  stream.zfree = static_cast<free_func>(0);
  stream.opaque = static_cast<voidpf>(0);

  gz_header gzip_header;
  memset(&gzip_header, 0, sizeof(gzip_header));
  int err = deflateInit2(&stream,
                         Z_DEFAULT_COMPRESSION,
                         Z_DEFLATED,
                         MAX_WBITS + kWindowBitsToGetGzipHeader,
                         kZlibMemoryLevel,
                         Z_DEFAULT_STRATEGY);
  if (err != Z_OK)
    return err;

  err = deflateSetHeader(&stream, &gzip_header);
  if (err != Z_OK)
    return err;

  err = deflate(&stream, Z_FINISH);
  if (err != Z_STREAM_END) {
    deflateEnd(&stream);
    return err == Z_OK ? Z_BUF_ERROR : err;
  }
  *dest_length = stream.total_out;

  err = deflateEnd(&stream);
  return err;
}

// This code is taken almost verbatim from third_party/zlib/uncompr.c. The only
// difference is inflateInit2() is called which sets the window bits to be > 16.
// That causes a gzip header to be parsed rather than a zlib header.
int GzipUncompressHelper(Bytef* dest,
                         uLongf* dest_length,
                         const Bytef* source,
                         uLong source_length) {
  z_stream stream;

  stream.next_in = bit_cast<Bytef*>(source);
  stream.avail_in = static_cast<uInt>(source_length);
  if (static_cast<uLong>(stream.avail_in) != source_length)
    return Z_BUF_ERROR;

  stream.next_out = dest;
  stream.avail_out = static_cast<uInt>(*dest_length);
  if (static_cast<uLong>(stream.avail_out) != *dest_length)
    return Z_BUF_ERROR;

  stream.zalloc = static_cast<alloc_func>(0);
  stream.zfree = static_cast<free_func>(0);

  int err = inflateInit2(&stream, MAX_WBITS + kWindowBitsToGetGzipHeader);
  if (err != Z_OK)
    return err;

  err = inflate(&stream, Z_FINISH);
  if (err != Z_STREAM_END) {
    inflateEnd(&stream);
    if (err == Z_NEED_DICT || (err == Z_BUF_ERROR && stream.avail_in == 0))
      return Z_DATA_ERROR;
    return err;
  }
  *dest_length = stream.total_out;

  err = inflateEnd(&stream);
  return err;
}

// Returns the uncompressed size from GZIP-compressed |compressed_data|.
uint32_t GetUncompressedSize(const std::string& compressed_data) {
  // The uncompressed size is stored in the last 4 bytes of |input| in LE.
  uint32_t size;
  if (compressed_data.length() < sizeof(size))
    return 0;
  memcpy(&size, &compressed_data[compressed_data.length() - sizeof(size)],
         sizeof(size));
  return base::ByteSwapToLE32(size);
}

}  // namespace

namespace compression {

bool GzipCompress(const std::string& input, std::string* output) {
  const uLongf input_size = static_cast<uLongf>(input.size());
  std::vector<Bytef> compressed_data(kGzipZlibHeaderDifferenceBytes +
                                     compressBound(input_size));

  uLongf compressed_size = static_cast<uLongf>(compressed_data.size());
  if (GzipCompressHelper(&compressed_data.front(),
                         &compressed_size,
                         bit_cast<const Bytef*>(input.data()),
                         input_size) != Z_OK) {
    return false;
  }

  compressed_data.resize(compressed_size);
  output->assign(compressed_data.begin(), compressed_data.end());
  DCHECK_EQ(input_size, GetUncompressedSize(*output));
  return true;
}

bool GzipUncompress(const std::string& input, std::string* output) {
  std::string uncompressed_output;
  uLongf uncompressed_size = static_cast<uLongf>(GetUncompressedSize(input));
  uncompressed_output.resize(uncompressed_size);
  if (GzipUncompressHelper(bit_cast<Bytef*>(uncompressed_output.data()),
                           &uncompressed_size,
                           bit_cast<const Bytef*>(input.data()),
                           static_cast<uLongf>(input.length())) == Z_OK) {
    output->swap(uncompressed_output);
    return true;
  }
  return false;
}

}  // namespace compression
