// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#ifndef LIB_JXL_ENC_HUFFMAN_H_
#define LIB_JXL_ENC_HUFFMAN_H_

#include <cstddef>
#include <cstdint>

#include "lib/jxl/base/status.h"
#include "lib/jxl/enc_bit_writer.h"

namespace jxl {

// Builds a Huffman tree for the given histogram, and encodes it into writer
// in a format that can be read by HuffmanDecodingData::ReadFromBitstream.
// An allotment for `writer` must already have been created by the caller.
Status BuildAndStoreHuffmanTree(const uint32_t* histogram, size_t length,
                                uint8_t* depth, uint16_t* bits,
                                BitWriter* writer);

}  // namespace jxl

#endif  // LIB_JXL_ENC_HUFFMAN_H_
