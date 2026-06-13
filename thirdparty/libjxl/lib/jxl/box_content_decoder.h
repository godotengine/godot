// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#ifndef LIB_JXL_BOX_CONTENT_DECODER_H_
#define LIB_JXL_BOX_CONTENT_DECODER_H_

#include <brotli/decode.h>
#include <jxl/decode.h>
#include <stdint.h>
#include <stdlib.h>

namespace jxl {

/** Outputs the contents of a box in a streaming fashion, either directly, or
 * optionally decoding with Brotli, in case of a brob box. The input must be
 * the contents of a box, excluding the box header.
 */
class JxlBoxContentDecoder {
 public:
  JxlBoxContentDecoder();
  ~JxlBoxContentDecoder();

  void StartBox(bool brob_decode, bool box_until_eof, size_t contents_size);

  // Outputs decoded bytes from the box, decoding with brotli if needed.
  // box_pos is the position in the box content which next_in points to.
  // Returns success, whether more input or output bytes are needed, or error.
  JxlDecoderStatus Process(const uint8_t* next_in, size_t avail_in,
                           size_t box_pos, uint8_t** next_out,
                           size_t* avail_out);

 private:
  BrotliDecoderState* brotli_dec;

  bool header_done_;
  bool brob_decode_;
  bool box_until_eof_;
  size_t remaining_;
  size_t pos_;
};

}  // namespace jxl

#endif  // LIB_JXL_BOX_CONTENT_DECODER_H_
