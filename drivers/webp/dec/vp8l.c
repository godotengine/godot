// Copyright 2012 Google Inc. All Rights Reserved.
//
// Use of this source code is governed by a BSD-style license
// that can be found in the COPYING file in the root of the source
// tree. An additional intellectual property rights grant can be found
// in the file PATENTS. All contributing project authors may
// be found in the AUTHORS file in the root of the source tree.
// -----------------------------------------------------------------------------
//
// main entry for the decoder
//
// Authors: Vikas Arora (vikaas.arora@gmail.com)
//          Jyrki Alakuijala (jyrki@google.com)

#include <stdio.h>
#include <stdlib.h>
#include "./alphai.h"
#include "./vp8li.h"
#include "../dsp/lossless.h"
#include "../dsp/yuv.h"
#include "../utils/alpha_processing.h"
#include "../utils/huffman.h"
#include "../utils/utils.h"

#define NUM_ARGB_CACHE_ROWS          16

static const int kCodeLengthLiterals = 16;
static const int kCodeLengthRepeatCode = 16;
static const int kCodeLengthExtraBits[3] = { 2, 3, 7 };
static const int kCodeLengthRepeatOffsets[3] = { 3, 3, 11 };

// -----------------------------------------------------------------------------
//  Five Huffman codes are used at each meta code:
//  1. green + length prefix codes + color cache codes,
//  2. alpha,
//  3. red,
//  4. blue, and,
//  5. distance prefix codes.
typedef enum {
  GREEN = 0,
  RED   = 1,
  BLUE  = 2,
  ALPHA = 3,
  DIST  = 4
} HuffIndex;

static const uint16_t kAlphabetSize[HUFFMAN_CODES_PER_META_CODE] = {
  NUM_LITERAL_CODES + NUM_LENGTH_CODES,
  NUM_LITERAL_CODES, NUM_LITERAL_CODES, NUM_LITERAL_CODES,
  NUM_DISTANCE_CODES
};


#define NUM_CODE_LENGTH_CODES       19
static const uint8_t kCodeLengthCodeOrder[NUM_CODE_LENGTH_CODES] = {
  17, 18, 0, 1, 2, 3, 4, 5, 16, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15
};

#define CODE_TO_PLANE_CODES        120
static const uint8_t kCodeToPlane[CODE_TO_PLANE_CODES] = {
  0x18, 0x07, 0x17, 0x19, 0x28, 0x06, 0x27, 0x29, 0x16, 0x1a,
  0x26, 0x2a, 0x38, 0x05, 0x37, 0x39, 0x15, 0x1b, 0x36, 0x3a,
  0x25, 0x2b, 0x48, 0x04, 0x47, 0x49, 0x14, 0x1c, 0x35, 0x3b,
  0x46, 0x4a, 0x24, 0x2c, 0x58, 0x45, 0x4b, 0x34, 0x3c, 0x03,
  0x57, 0x59, 0x13, 0x1d, 0x56, 0x5a, 0x23, 0x2d, 0x44, 0x4c,
  0x55, 0x5b, 0x33, 0x3d, 0x68, 0x02, 0x67, 0x69, 0x12, 0x1e,
  0x66, 0x6a, 0x22, 0x2e, 0x54, 0x5c, 0x43, 0x4d, 0x65, 0x6b,
  0x32, 0x3e, 0x78, 0x01, 0x77, 0x79, 0x53, 0x5d, 0x11, 0x1f,
  0x64, 0x6c, 0x42, 0x4e, 0x76, 0x7a, 0x21, 0x2f, 0x75, 0x7b,
  0x31, 0x3f, 0x63, 0x6d, 0x52, 0x5e, 0x00, 0x74, 0x7c, 0x41,
  0x4f, 0x10, 0x20, 0x62, 0x6e, 0x30, 0x73, 0x7d, 0x51, 0x5f,
  0x40, 0x72, 0x7e, 0x61, 0x6f, 0x50, 0x71, 0x7f, 0x60, 0x70
};

static int DecodeImageStream(int xsize, int ysize,
                             int is_level0,
                             VP8LDecoder* const dec,
                             uint32_t** const decoded_data);

//------------------------------------------------------------------------------

int VP8LCheckSignature(const uint8_t* const data, size_t size) {
  return (size >= VP8L_FRAME_HEADER_SIZE &&
          data[0] == VP8L_MAGIC_BYTE &&
          (data[4] >> 5) == 0);  // version
}

static int ReadImageInfo(VP8LBitReader* const br,
                         int* const width, int* const height,
                         int* const has_alpha) {
  if (VP8LReadBits(br, 8) != VP8L_MAGIC_BYTE) return 0;
  *width = VP8LReadBits(br, VP8L_IMAGE_SIZE_BITS) + 1;
  *height = VP8LReadBits(br, VP8L_IMAGE_SIZE_BITS) + 1;
  *has_alpha = VP8LReadBits(br, 1);
  if (VP8LReadBits(br, VP8L_VERSION_BITS) != 0) return 0;
  return 1;
}

int VP8LGetInfo(const uint8_t* data, size_t data_size,
                int* const width, int* const height, int* const has_alpha) {
  if (data == NULL || data_size < VP8L_FRAME_HEADER_SIZE) {
    return 0;         // not enough data
  } else if (!VP8LCheckSignature(data, data_size)) {
    return 0;         // bad signature
  } else {
    int w, h, a;
    VP8LBitReader br;
    VP8LInitBitReader(&br, data, data_size);
    if (!ReadImageInfo(&br, &w, &h, &a)) {
      return 0;
    }
    if (width != NULL) *width = w;
    if (height != NULL) *height = h;
    if (has_alpha != NULL) *has_alpha = a;
    return 1;
  }
}

//------------------------------------------------------------------------------

static WEBP_INLINE int GetCopyDistance(int distance_symbol,
                                       VP8LBitReader* const br) {
  int extra_bits, offset;
  if (distance_symbol < 4) {
    return distance_symbol + 1;
  }
  extra_bits = (distance_symbol - 2) >> 1;
  offset = (2 + (distance_symbol & 1)) << extra_bits;
  return offset + VP8LReadBits(br, extra_bits) + 1;
}

static WEBP_INLINE int GetCopyLength(int length_symbol,
                                     VP8LBitReader* const br) {
  // Length and distance prefixes are encoded the same way.
  return GetCopyDistance(length_symbol, br);
}

static WEBP_INLINE int PlaneCodeToDistance(int xsize, int plane_code) {
  if (plane_code > CODE_TO_PLANE_CODES) {
    return plane_code - CODE_TO_PLANE_CODES;
  } else {
    const int dist_code = kCodeToPlane[plane_code - 1];
    const int yoffset = dist_code >> 4;
    const int xoffset = 8 - (dist_code & 0xf);
    const int dist = yoffset * xsize + xoffset;
    return (dist >= 1) ? dist : 1;  // dist<1 can happen if xsize is very small
  }
}

//------------------------------------------------------------------------------
// Decodes the next Huffman code from bit-stream.
// FillBitWindow(br) needs to be called at minimum every second call
// to ReadSymbol, in order to pre-fetch enough bits.
static WEBP_INLINE int ReadSymbol(const HuffmanTree* tree,
                                  VP8LBitReader* const br) {
  const HuffmanTreeNode* node = tree->root_;
  uint32_t bits = VP8LPrefetchBits(br);
  int bitpos = br->bit_pos_;
  // Check if we find the bit combination from the Huffman lookup table.
  const int lut_ix = bits & (HUFF_LUT - 1);
  const int lut_bits = tree->lut_bits_[lut_ix];
  if (lut_bits <= HUFF_LUT_BITS) {
    VP8LSetBitPos(br, bitpos + lut_bits);
    return tree->lut_symbol_[lut_ix];
  }
  node += tree->lut_jump_[lut_ix];
  bitpos += HUFF_LUT_BITS;
  bits >>= HUFF_LUT_BITS;

  // Decode the value from a binary tree.
  assert(node != NULL);
  do {
    node = HuffmanTreeNextNode(node, bits & 1);
    bits >>= 1;
    ++bitpos;
  } while (HuffmanTreeNodeIsNotLeaf(node));
  VP8LSetBitPos(br, bitpos);
  return node->symbol_;
}

static int ReadHuffmanCodeLengths(
    VP8LDecoder* const dec, const int* const code_length_code_lengths,
    int num_symbols, int* const code_lengths) {
  int ok = 0;
  VP8LBitReader* const br = &dec->br_;
  int symbol;
  int max_symbol;
  int prev_code_len = DEFAULT_CODE_LENGTH;
  HuffmanTree tree;

  if (!HuffmanTreeBuildImplicit(&tree, code_length_code_lengths,
                                NUM_CODE_LENGTH_CODES)) {
    dec->status_ = VP8_STATUS_BITSTREAM_ERROR;
    return 0;
  }

  if (VP8LReadBits(br, 1)) {    // use length
    const int length_nbits = 2 + 2 * VP8LReadBits(br, 3);
    max_symbol = 2 + VP8LReadBits(br, length_nbits);
    if (max_symbol > num_symbols) {
      dec->status_ = VP8_STATUS_BITSTREAM_ERROR;
      goto End;
    }
  } else {
    max_symbol = num_symbols;
  }

  symbol = 0;
  while (symbol < num_symbols) {
    int code_len;
    if (max_symbol-- == 0) break;
    VP8LFillBitWindow(br);
    code_len = ReadSymbol(&tree, br);
    if (code_len < kCodeLengthLiterals) {
      code_lengths[symbol++] = code_len;
      if (code_len != 0) prev_code_len = code_len;
    } else {
      const int use_prev = (code_len == kCodeLengthRepeatCode);
      const int slot = code_len - kCodeLengthLiterals;
      const int extra_bits = kCodeLengthExtraBits[slot];
      const int repeat_offset = kCodeLengthRepeatOffsets[slot];
      int repeat = VP8LReadBits(br, extra_bits) + repeat_offset;
      if (symbol + repeat > num_symbols) {
        dec->status_ = VP8_STATUS_BITSTREAM_ERROR;
        goto End;
      } else {
        const int length = use_prev ? prev_code_len : 0;
        while (repeat-- > 0) code_lengths[symbol++] = length;
      }
    }
  }
  ok = 1;

 End:
  HuffmanTreeRelease(&tree);
  return ok;
}

static int ReadHuffmanCode(int alphabet_size, VP8LDecoder* const dec,
                           HuffmanTree* const tree) {
  int ok = 0;
  VP8LBitReader* const br = &dec->br_;
  const int simple_code = VP8LReadBits(br, 1);

  if (simple_code) {  // Read symbols, codes & code lengths directly.
    int symbols[2];
    int codes[2];
    int code_lengths[2];
    const int num_symbols = VP8LReadBits(br, 1) + 1;
    const int first_symbol_len_code = VP8LReadBits(br, 1);
    // The first code is either 1 bit or 8 bit code.
    symbols[0] = VP8LReadBits(br, (first_symbol_len_code == 0) ? 1 : 8);
    codes[0] = 0;
    code_lengths[0] = num_symbols - 1;
    // The second code (if present), is always 8 bit long.
    if (num_symbols == 2) {
      symbols[1] = VP8LReadBits(br, 8);
      codes[1] = 1;
      code_lengths[1] = num_symbols - 1;
    }
    ok = HuffmanTreeBuildExplicit(tree, code_lengths, codes, symbols,
                                  alphabet_size, num_symbols);
  } else {  // Decode Huffman-coded code lengths.
    int* code_lengths = NULL;
    int i;
    int code_length_code_lengths[NUM_CODE_LENGTH_CODES] = { 0 };
    const int num_codes = VP8LReadBits(br, 4) + 4;
    if (num_codes > NUM_CODE_LENGTH_CODES) {
      dec->status_ = VP8_STATUS_BITSTREAM_ERROR;
      return 0;
    }

    code_lengths =
        (int*)WebPSafeCalloc((uint64_t)alphabet_size, sizeof(*code_lengths));
    if (code_lengths == NULL) {
      dec->status_ = VP8_STATUS_OUT_OF_MEMORY;
      return 0;
    }

    for (i = 0; i < num_codes; ++i) {
      code_length_code_lengths[kCodeLengthCodeOrder[i]] = VP8LReadBits(br, 3);
    }
    ok = ReadHuffmanCodeLengths(dec, code_length_code_lengths, alphabet_size,
                                code_lengths);
    if (ok) {
      ok = HuffmanTreeBuildImplicit(tree, code_lengths, alphabet_size);
    }
    free(code_lengths);
  }
  ok = ok && !br->error_;
  if (!ok) {
    dec->status_ = VP8_STATUS_BITSTREAM_ERROR;
    return 0;
  }
  return 1;
}

static void DeleteHtreeGroups(HTreeGroup* htree_groups, int num_htree_groups) {
  if (htree_groups != NULL) {
    int i, j;
    for (i = 0; i < num_htree_groups; ++i) {
      HuffmanTree* const htrees = htree_groups[i].htrees_;
      for (j = 0; j < HUFFMAN_CODES_PER_META_CODE; ++j) {
        HuffmanTreeRelease(&htrees[j]);
      }
    }
    free(htree_groups);
  }
}

static int ReadHuffmanCodes(VP8LDecoder* const dec, int xsize, int ysize,
                            int color_cache_bits, int allow_recursion) {
  int i, j;
  VP8LBitReader* const br = &dec->br_;
  VP8LMetadata* const hdr = &dec->hdr_;
  uint32_t* huffman_image = NULL;
  HTreeGroup* htree_groups = NULL;
  int num_htree_groups = 1;

  if (allow_recursion && VP8LReadBits(br, 1)) {
    // use meta Huffman codes.
    const int huffman_precision = VP8LReadBits(br, 3) + 2;
    const int huffman_xsize = VP8LSubSampleSize(xsize, huffman_precision);
    const int huffman_ysize = VP8LSubSampleSize(ysize, huffman_precision);
    const int huffman_pixs = huffman_xsize * huffman_ysize;
    if (!DecodeImageStream(huffman_xsize, huffman_ysize, 0, dec,
                           &huffman_image)) {
      dec->status_ = VP8_STATUS_BITSTREAM_ERROR;
      goto Error;
    }
    hdr->huffman_subsample_bits_ = huffman_precision;
    for (i = 0; i < huffman_pixs; ++i) {
      // The huffman data is stored in red and green bytes.
      const int group = (huffman_image[i] >> 8) & 0xffff;
      huffman_image[i] = group;
      if (group >= num_htree_groups) {
        num_htree_groups = group + 1;
      }
    }
  }

  if (br->error_) goto Error;

  assert(num_htree_groups <= 0x10000);
  htree_groups =
      (HTreeGroup*)WebPSafeCalloc((uint64_t)num_htree_groups,
                                  sizeof(*htree_groups));
  if (htree_groups == NULL) {
    dec->status_ = VP8_STATUS_OUT_OF_MEMORY;
    goto Error;
  }

  for (i = 0; i < num_htree_groups; ++i) {
    HuffmanTree* const htrees = htree_groups[i].htrees_;
    for (j = 0; j < HUFFMAN_CODES_PER_META_CODE; ++j) {
      int alphabet_size = kAlphabetSize[j];
      if (j == 0 && color_cache_bits > 0) {
        alphabet_size += 1 << color_cache_bits;
      }
      if (!ReadHuffmanCode(alphabet_size, dec, htrees + j)) goto Error;
    }
  }

  // All OK. Finalize pointers and return.
  hdr->huffman_image_ = huffman_image;
  hdr->num_htree_groups_ = num_htree_groups;
  hdr->htree_groups_ = htree_groups;
  return 1;

 Error:
  free(huffman_image);
  DeleteHtreeGroups(htree_groups, num_htree_groups);
  return 0;
}

//------------------------------------------------------------------------------
// Scaling.

static int AllocateAndInitRescaler(VP8LDecoder* const dec, VP8Io* const io) {
  const int num_channels = 4;
  const int in_width = io->mb_w;
  const int out_width = io->scaled_width;
  const int in_height = io->mb_h;
  const int out_height = io->scaled_height;
  const uint64_t work_size = 2 * num_channels * (uint64_t)out_width;
  int32_t* work;        // Rescaler work area.
  const uint64_t scaled_data_size = num_channels * (uint64_t)out_width;
  uint32_t* scaled_data;  // Temporary storage for scaled BGRA data.
  const uint64_t memory_size = sizeof(*dec->rescaler) +
                               work_size * sizeof(*work) +
                               scaled_data_size * sizeof(*scaled_data);
  uint8_t* memory = (uint8_t*)WebPSafeCalloc(memory_size, sizeof(*memory));
  if (memory == NULL) {
    dec->status_ = VP8_STATUS_OUT_OF_MEMORY;
    return 0;
  }
  assert(dec->rescaler_memory == NULL);
  dec->rescaler_memory = memory;

  dec->rescaler = (WebPRescaler*)memory;
  memory += sizeof(*dec->rescaler);
  work = (int32_t*)memory;
  memory += work_size * sizeof(*work);
  scaled_data = (uint32_t*)memory;

  WebPRescalerInit(dec->rescaler, in_width, in_height, (uint8_t*)scaled_data,
                   out_width, out_height, 0, num_channels,
                   in_width, out_width, in_height, out_height, work);
  return 1;
}

//------------------------------------------------------------------------------
// Export to ARGB

// We have special "export" function since we need to convert from BGRA
static int Export(WebPRescaler* const rescaler, WEBP_CSP_MODE colorspace,
                  int rgba_stride, uint8_t* const rgba) {
  uint32_t* const src = (uint32_t*)rescaler->dst;
  const int dst_width = rescaler->dst_width;
  int num_lines_out = 0;
  while (WebPRescalerHasPendingOutput(rescaler)) {
    uint8_t* const dst = rgba + num_lines_out * rgba_stride;
    WebPRescalerExportRow(rescaler);
    WebPMultARGBRow(src, dst_width, 1);
    VP8LConvertFromBGRA(src, dst_width, colorspace, dst);
    ++num_lines_out;
  }
  return num_lines_out;
}

// Emit scaled rows.
static int EmitRescaledRowsRGBA(const VP8LDecoder* const dec,
                                uint8_t* in, int in_stride, int mb_h,
                                uint8_t* const out, int out_stride) {
  const WEBP_CSP_MODE colorspace = dec->output_->colorspace;
  int num_lines_in = 0;
  int num_lines_out = 0;
  while (num_lines_in < mb_h) {
    uint8_t* const row_in = in + num_lines_in * in_stride;
    uint8_t* const row_out = out + num_lines_out * out_stride;
    const int lines_left = mb_h - num_lines_in;
    const int needed_lines = WebPRescaleNeededLines(dec->rescaler, lines_left);
    assert(needed_lines > 0 && needed_lines <= lines_left);
    WebPMultARGBRows(row_in, in_stride,
                     dec->rescaler->src_width, needed_lines, 0);
    WebPRescalerImport(dec->rescaler, lines_left, row_in, in_stride);
    num_lines_in += needed_lines;
    num_lines_out += Export(dec->rescaler, colorspace, out_stride, row_out);
  }
  return num_lines_out;
}

// Emit rows without any scaling.
static int EmitRows(WEBP_CSP_MODE colorspace,
                    const uint8_t* row_in, int in_stride,
                    int mb_w, int mb_h,
                    uint8_t* const out, int out_stride) {
  int lines = mb_h;
  uint8_t* row_out = out;
  while (lines-- > 0) {
    VP8LConvertFromBGRA((const uint32_t*)row_in, mb_w, colorspace, row_out);
    row_in += in_stride;
    row_out += out_stride;
  }
  return mb_h;  // Num rows out == num rows in.
}

//------------------------------------------------------------------------------
// Export to YUVA

static void ConvertToYUVA(const uint32_t* const src, int width, int y_pos,
                          const WebPDecBuffer* const output) {
  const WebPYUVABuffer* const buf = &output->u.YUVA;
  // first, the luma plane
  {
    int i;
    uint8_t* const y = buf->y + y_pos * buf->y_stride;
    for (i = 0; i < width; ++i) {
      const uint32_t p = src[i];
      y[i] = VP8RGBToY((p >> 16) & 0xff, (p >> 8) & 0xff, (p >> 0) & 0xff,
                       YUV_HALF);
    }
  }

  // then U/V planes
  {
    uint8_t* const u = buf->u + (y_pos >> 1) * buf->u_stride;
    uint8_t* const v = buf->v + (y_pos >> 1) * buf->v_stride;
    const int uv_width = width >> 1;
    int i;
    for (i = 0; i < uv_width; ++i) {
      const uint32_t v0 = src[2 * i + 0];
      const uint32_t v1 = src[2 * i + 1];
      // VP8RGBToU/V expects four accumulated pixels. Hence we need to
      // scale r/g/b value by a factor 2. We just shift v0/v1 one bit less.
      const int r = ((v0 >> 15) & 0x1fe) + ((v1 >> 15) & 0x1fe);
      const int g = ((v0 >>  7) & 0x1fe) + ((v1 >>  7) & 0x1fe);
      const int b = ((v0 <<  1) & 0x1fe) + ((v1 <<  1) & 0x1fe);
      if (!(y_pos & 1)) {  // even lines: store values
        u[i] = VP8RGBToU(r, g, b, YUV_HALF << 2);
        v[i] = VP8RGBToV(r, g, b, YUV_HALF << 2);
      } else {             // odd lines: average with previous values
        const int tmp_u = VP8RGBToU(r, g, b, YUV_HALF << 2);
        const int tmp_v = VP8RGBToV(r, g, b, YUV_HALF << 2);
        // Approximated average-of-four. But it's an acceptable diff.
        u[i] = (u[i] + tmp_u + 1) >> 1;
        v[i] = (v[i] + tmp_v + 1) >> 1;
      }
    }
    if (width & 1) {       // last pixel
      const uint32_t v0 = src[2 * i + 0];
      const int r = (v0 >> 14) & 0x3fc;
      const int g = (v0 >>  6) & 0x3fc;
      const int b = (v0 <<  2) & 0x3fc;
      if (!(y_pos & 1)) {  // even lines
        u[i] = VP8RGBToU(r, g, b, YUV_HALF << 2);
        v[i] = VP8RGBToV(r, g, b, YUV_HALF << 2);
      } else {             // odd lines (note: we could just skip this)
        const int tmp_u = VP8RGBToU(r, g, b, YUV_HALF << 2);
        const int tmp_v = VP8RGBToV(r, g, b, YUV_HALF << 2);
        u[i] = (u[i] + tmp_u + 1) >> 1;
        v[i] = (v[i] + tmp_v + 1) >> 1;
      }
    }
  }
  // Lastly, store alpha if needed.
  if (buf->a != NULL) {
    int i;
    uint8_t* const a = buf->a + y_pos * buf->a_stride;
    for (i = 0; i < width; ++i) a[i] = (src[i] >> 24);
  }
}

static int ExportYUVA(const VP8LDecoder* const dec, int y_pos) {
  WebPRescaler* const rescaler = dec->rescaler;
  uint32_t* const src = (uint32_t*)rescaler->dst;
  const int dst_width = rescaler->dst_width;
  int num_lines_out = 0;
  while (WebPRescalerHasPendingOutput(rescaler)) {
    WebPRescalerExportRow(rescaler);
    WebPMultARGBRow(src, dst_width, 1);
    ConvertToYUVA(src, dst_width, y_pos, dec->output_);
    ++y_pos;
    ++num_lines_out;
  }
  return num_lines_out;
}

static int EmitRescaledRowsYUVA(const VP8LDecoder* const dec,
                                uint8_t* in, int in_stride, int mb_h) {
  int num_lines_in = 0;
  int y_pos = dec->last_out_row_;
  while (num_lines_in < mb_h) {
    const int lines_left = mb_h - num_lines_in;
    const int needed_lines = WebPRescaleNeededLines(dec->rescaler, lines_left);
    WebPMultARGBRows(in, in_stride, dec->rescaler->src_width, needed_lines, 0);
    WebPRescalerImport(dec->rescaler, lines_left, in, in_stride);
    num_lines_in += needed_lines;
    in += needed_lines * in_stride;
    y_pos += ExportYUVA(dec, y_pos);
  }
  return y_pos;
}

static int EmitRowsYUVA(const VP8LDecoder* const dec,
                        const uint8_t* in, int in_stride,
                        int mb_w, int num_rows) {
  int y_pos = dec->last_out_row_;
  while (num_rows-- > 0) {
    ConvertToYUVA((const uint32_t*)in, mb_w, y_pos, dec->output_);
    in += in_stride;
    ++y_pos;
  }
  return y_pos;
}

//------------------------------------------------------------------------------
// Cropping.

// Sets io->mb_y, io->mb_h & io->mb_w according to start row, end row and
// crop options. Also updates the input data pointer, so that it points to the
// start of the cropped window. Note that pixels are in ARGB format even if
// 'in_data' is uint8_t*.
// Returns true if the crop window is not empty.
static int SetCropWindow(VP8Io* const io, int y_start, int y_end,
                         uint8_t** const in_data, int pixel_stride) {
  assert(y_start < y_end);
  assert(io->crop_left < io->crop_right);
  if (y_end > io->crop_bottom) {
    y_end = io->crop_bottom;  // make sure we don't overflow on last row.
  }
  if (y_start < io->crop_top) {
    const int delta = io->crop_top - y_start;
    y_start = io->crop_top;
    *in_data += delta * pixel_stride;
  }
  if (y_start >= y_end) return 0;  // Crop window is empty.

  *in_data += io->crop_left * sizeof(uint32_t);

  io->mb_y = y_start - io->crop_top;
  io->mb_w = io->crop_right - io->crop_left;
  io->mb_h = y_end - y_start;
  return 1;  // Non-empty crop window.
}

//------------------------------------------------------------------------------

static WEBP_INLINE int GetMetaIndex(
    const uint32_t* const image, int xsize, int bits, int x, int y) {
  if (bits == 0) return 0;
  return image[xsize * (y >> bits) + (x >> bits)];
}

static WEBP_INLINE HTreeGroup* GetHtreeGroupForPos(VP8LMetadata* const hdr,
                                                   int x, int y) {
  const int meta_index = GetMetaIndex(hdr->huffman_image_, hdr->huffman_xsize_,
                                      hdr->huffman_subsample_bits_, x, y);
  assert(meta_index < hdr->num_htree_groups_);
  return hdr->htree_groups_ + meta_index;
}

//------------------------------------------------------------------------------
// Main loop, with custom row-processing function

typedef void (*ProcessRowsFunc)(VP8LDecoder* const dec, int row);

static void ApplyInverseTransforms(VP8LDecoder* const dec, int num_rows,
                                   const uint32_t* const rows) {
  int n = dec->next_transform_;
  const int cache_pixs = dec->width_ * num_rows;
  const int start_row = dec->last_row_;
  const int end_row = start_row + num_rows;
  const uint32_t* rows_in = rows;
  uint32_t* const rows_out = dec->argb_cache_;

  // Inverse transforms.
  // TODO: most transforms only need to operate on the cropped region only.
  memcpy(rows_out, rows_in, cache_pixs * sizeof(*rows_out));
  while (n-- > 0) {
    VP8LTransform* const transform = &dec->transforms_[n];
    VP8LInverseTransform(transform, start_row, end_row, rows_in, rows_out);
    rows_in = rows_out;
  }
}

// Special method for paletted alpha data.
static void ApplyInverseTransformsAlpha(VP8LDecoder* const dec, int num_rows,
                                        const uint8_t* const rows) {
  const int start_row = dec->last_row_;
  const int end_row = start_row + num_rows;
  const uint8_t* rows_in = rows;
  uint8_t* rows_out = (uint8_t*)dec->io_->opaque + dec->io_->width * start_row;
  VP8LTransform* const transform = &dec->transforms_[0];
  assert(dec->next_transform_ == 1);
  assert(transform->type_ == COLOR_INDEXING_TRANSFORM);
  VP8LColorIndexInverseTransformAlpha(transform, start_row, end_row, rows_in,
                                      rows_out);
}

// Processes (transforms, scales & color-converts) the rows decoded after the
// last call.
static void ProcessRows(VP8LDecoder* const dec, int row) {
  const uint32_t* const rows = dec->pixels_ + dec->width_ * dec->last_row_;
  const int num_rows = row - dec->last_row_;

  if (num_rows <= 0) return;  // Nothing to be done.
  ApplyInverseTransforms(dec, num_rows, rows);

  // Emit output.
  {
    VP8Io* const io = dec->io_;
    uint8_t* rows_data = (uint8_t*)dec->argb_cache_;
    const int in_stride = io->width * sizeof(uint32_t);  // in unit of RGBA
    if (!SetCropWindow(io, dec->last_row_, row, &rows_data, in_stride)) {
      // Nothing to output (this time).
    } else {
      const WebPDecBuffer* const output = dec->output_;
      if (output->colorspace < MODE_YUV) {  // convert to RGBA
        const WebPRGBABuffer* const buf = &output->u.RGBA;
        uint8_t* const rgba = buf->rgba + dec->last_out_row_ * buf->stride;
        const int num_rows_out = io->use_scaling ?
            EmitRescaledRowsRGBA(dec, rows_data, in_stride, io->mb_h,
                                 rgba, buf->stride) :
            EmitRows(output->colorspace, rows_data, in_stride,
                     io->mb_w, io->mb_h, rgba, buf->stride);
        // Update 'last_out_row_'.
        dec->last_out_row_ += num_rows_out;
      } else {                              // convert to YUVA
        dec->last_out_row_ = io->use_scaling ?
            EmitRescaledRowsYUVA(dec, rows_data, in_stride, io->mb_h) :
            EmitRowsYUVA(dec, rows_data, in_stride, io->mb_w, io->mb_h);
      }
      assert(dec->last_out_row_ <= output->height);
    }
  }

  // Update 'last_row_'.
  dec->last_row_ = row;
  assert(dec->last_row_ <= dec->height_);
}

// Row-processing for the special case when alpha data contains only one
// transform (color indexing), and trivial non-green literals.
static int Is8bOptimizable(const VP8LMetadata* const hdr) {
  int i;
  if (hdr->color_cache_size_ > 0) return 0;
  // When the Huffman tree contains only one symbol, we can skip the
  // call to ReadSymbol() for red/blue/alpha channels.
  for (i = 0; i < hdr->num_htree_groups_; ++i) {
    const HuffmanTree* const htrees = hdr->htree_groups_[i].htrees_;
    if (htrees[RED].num_nodes_ > 1) return 0;
    if (htrees[BLUE].num_nodes_ > 1) return 0;
    if (htrees[ALPHA].num_nodes_ > 1) return 0;
  }
  return 1;
}

static void ExtractPalettedAlphaRows(VP8LDecoder* const dec, int row) {
  const int num_rows = row - dec->last_row_;
  const uint8_t* const in =
      (uint8_t*)dec->pixels_ + dec->width_ * dec->last_row_;
  if (num_rows > 0) {
    ApplyInverseTransformsAlpha(dec, num_rows, in);
  }
  dec->last_row_ = dec->last_out_row_ = row;
}

static int DecodeAlphaData(VP8LDecoder* const dec, uint8_t* const data,
                           int width, int height, int last_row) {
  int ok = 1;
  int row = dec->last_pixel_ / width;
  int col = dec->last_pixel_ % width;
  VP8LBitReader* const br = &dec->br_;
  VP8LMetadata* const hdr = &dec->hdr_;
  const HTreeGroup* htree_group = GetHtreeGroupForPos(hdr, col, row);
  int pos = dec->last_pixel_;         // current position
  const int end = width * height;     // End of data
  const int last = width * last_row;  // Last pixel to decode
  const int len_code_limit = NUM_LITERAL_CODES + NUM_LENGTH_CODES;
  const int mask = hdr->huffman_mask_;
  assert(htree_group != NULL);
  assert(last_row <= height);
  assert(Is8bOptimizable(hdr));

  while (!br->eos_ && pos < last) {
    int code;
    // Only update when changing tile.
    if ((col & mask) == 0) {
      htree_group = GetHtreeGroupForPos(hdr, col, row);
    }
    VP8LFillBitWindow(br);
    code = ReadSymbol(&htree_group->htrees_[GREEN], br);
    if (code < NUM_LITERAL_CODES) {  // Literal
      data[pos] = code;
      ++pos;
      ++col;
      if (col >= width) {
        col = 0;
        ++row;
        if (row % NUM_ARGB_CACHE_ROWS == 0) {
          ExtractPalettedAlphaRows(dec, row);
        }
      }
    } else if (code < len_code_limit) {  // Backward reference
      int dist_code, dist;
      const int length_sym = code - NUM_LITERAL_CODES;
      const int length = GetCopyLength(length_sym, br);
      const int dist_symbol = ReadSymbol(&htree_group->htrees_[DIST], br);
      VP8LFillBitWindow(br);
      dist_code = GetCopyDistance(dist_symbol, br);
      dist = PlaneCodeToDistance(width, dist_code);
      if (pos >= dist && end - pos >= length) {
        int i;
        for (i = 0; i < length; ++i) data[pos + i] = data[pos + i - dist];
      } else {
        ok = 0;
        goto End;
      }
      pos += length;
      col += length;
      while (col >= width) {
        col -= width;
        ++row;
        if (row % NUM_ARGB_CACHE_ROWS == 0) {
          ExtractPalettedAlphaRows(dec, row);
        }
      }
      if (pos < last && (col & mask)) {
        htree_group = GetHtreeGroupForPos(hdr, col, row);
      }
    } else {  // Not reached
      ok = 0;
      goto End;
    }
    ok = !br->error_;
    if (!ok) goto End;
  }
  // Process the remaining rows corresponding to last row-block.
  ExtractPalettedAlphaRows(dec, row);

 End:
  if (br->error_ || !ok || (br->eos_ && pos < end)) {
    ok = 0;
    dec->status_ = br->eos_ ? VP8_STATUS_SUSPENDED
                            : VP8_STATUS_BITSTREAM_ERROR;
  } else {
    dec->last_pixel_ = (int)pos;
    if (pos == end) dec->state_ = READ_DATA;
  }
  return ok;
}

static int DecodeImageData(VP8LDecoder* const dec, uint32_t* const data,
                           int width, int height, int last_row,
                           ProcessRowsFunc process_func) {
  int ok = 1;
  int row = dec->last_pixel_ / width;
  int col = dec->last_pixel_ % width;
  VP8LBitReader* const br = &dec->br_;
  VP8LMetadata* const hdr = &dec->hdr_;
  HTreeGroup* htree_group = GetHtreeGroupForPos(hdr, col, row);
  uint32_t* src = data + dec->last_pixel_;
  uint32_t* last_cached = src;
  uint32_t* const src_end = data + width * height;     // End of data
  uint32_t* const src_last = data + width * last_row;  // Last pixel to decode
  const int len_code_limit = NUM_LITERAL_CODES + NUM_LENGTH_CODES;
  const int color_cache_limit = len_code_limit + hdr->color_cache_size_;
  VP8LColorCache* const color_cache =
      (hdr->color_cache_size_ > 0) ? &hdr->color_cache_ : NULL;
  const int mask = hdr->huffman_mask_;
  assert(htree_group != NULL);
  assert(src_last <= src_end);

  while (!br->eos_ && src < src_last) {
    int code;
    // Only update when changing tile. Note we could use this test:
    // if "((((prev_col ^ col) | prev_row ^ row)) > mask)" -> tile changed
    // but that's actually slower and needs storing the previous col/row.
    if ((col & mask) == 0) {
      htree_group = GetHtreeGroupForPos(hdr, col, row);
    }
    VP8LFillBitWindow(br);
    code = ReadSymbol(&htree_group->htrees_[GREEN], br);
    if (code < NUM_LITERAL_CODES) {  // Literal
      int red, green, blue, alpha;
      red = ReadSymbol(&htree_group->htrees_[RED], br);
      green = code;
      VP8LFillBitWindow(br);
      blue = ReadSymbol(&htree_group->htrees_[BLUE], br);
      alpha = ReadSymbol(&htree_group->htrees_[ALPHA], br);
      *src = (alpha << 24) | (red << 16) | (green << 8) | blue;
    AdvanceByOne:
      ++src;
      ++col;
      if (col >= width) {
        col = 0;
        ++row;
        if ((row % NUM_ARGB_CACHE_ROWS == 0) && (process_func != NULL)) {
          process_func(dec, row);
        }
        if (color_cache != NULL) {
          while (last_cached < src) {
            VP8LColorCacheInsert(color_cache, *last_cached++);
          }
        }
      }
    } else if (code < len_code_limit) {  // Backward reference
      int dist_code, dist;
      const int length_sym = code - NUM_LITERAL_CODES;
      const int length = GetCopyLength(length_sym, br);
      const int dist_symbol = ReadSymbol(&htree_group->htrees_[DIST], br);
      VP8LFillBitWindow(br);
      dist_code = GetCopyDistance(dist_symbol, br);
      dist = PlaneCodeToDistance(width, dist_code);
      if (src - data < (ptrdiff_t)dist || src_end - src < (ptrdiff_t)length) {
        ok = 0;
        goto End;
      } else {
        int i;
        for (i = 0; i < length; ++i) src[i] = src[i - dist];
        src += length;
      }
      col += length;
      while (col >= width) {
        col -= width;
        ++row;
        if ((row % NUM_ARGB_CACHE_ROWS == 0) && (process_func != NULL)) {
          process_func(dec, row);
        }
      }
      if (src < src_last) {
        if (col & mask) htree_group = GetHtreeGroupForPos(hdr, col, row);
        if (color_cache != NULL) {
          while (last_cached < src) {
            VP8LColorCacheInsert(color_cache, *last_cached++);
          }
        }
      }
    } else if (code < color_cache_limit) {  // Color cache
      const int key = code - len_code_limit;
      assert(color_cache != NULL);
      while (last_cached < src) {
        VP8LColorCacheInsert(color_cache, *last_cached++);
      }
      *src = VP8LColorCacheLookup(color_cache, key);
      goto AdvanceByOne;
    } else {  // Not reached
      ok = 0;
      goto End;
    }
    ok = !br->error_;
    if (!ok) goto End;
  }
  // Process the remaining rows corresponding to last row-block.
  if (process_func != NULL) process_func(dec, row);

 End:
  if (br->error_ || !ok || (br->eos_ && src < src_end)) {
    ok = 0;
    dec->status_ = br->eos_ ? VP8_STATUS_SUSPENDED
                            : VP8_STATUS_BITSTREAM_ERROR;
  } else {
    dec->last_pixel_ = (int)(src - data);
    if (src == src_end) dec->state_ = READ_DATA;
  }
  return ok;
}

// -----------------------------------------------------------------------------
// VP8LTransform

static void ClearTransform(VP8LTransform* const transform) {
  free(transform->data_);
  transform->data_ = NULL;
}

// For security reason, we need to remap the color map to span
// the total possible bundled values, and not just the num_colors.
static int ExpandColorMap(int num_colors, VP8LTransform* const transform) {
  int i;
  const int final_num_colors = 1 << (8 >> transform->bits_);
  uint32_t* const new_color_map =
      (uint32_t*)WebPSafeMalloc((uint64_t)final_num_colors,
                                sizeof(*new_color_map));
  if (new_color_map == NULL) {
    return 0;
  } else {
    uint8_t* const data = (uint8_t*)transform->data_;
    uint8_t* const new_data = (uint8_t*)new_color_map;
    new_color_map[0] = transform->data_[0];
    for (i = 4; i < 4 * num_colors; ++i) {
      // Equivalent to AddPixelEq(), on a byte-basis.
      new_data[i] = (data[i] + new_data[i - 4]) & 0xff;
    }
    for (; i < 4 * final_num_colors; ++i)
      new_data[i] = 0;  // black tail.
    free(transform->data_);
    transform->data_ = new_color_map;
  }
  return 1;
}

static int ReadTransform(int* const xsize, int const* ysize,
                         VP8LDecoder* const dec) {
  int ok = 1;
  VP8LBitReader* const br = &dec->br_;
  VP8LTransform* transform = &dec->transforms_[dec->next_transform_];
  const VP8LImageTransformType type =
      (VP8LImageTransformType)VP8LReadBits(br, 2);

  // Each transform type can only be present once in the stream.
  if (dec->transforms_seen_ & (1U << type)) {
    return 0;  // Already there, let's not accept the second same transform.
  }
  dec->transforms_seen_ |= (1U << type);

  transform->type_ = type;
  transform->xsize_ = *xsize;
  transform->ysize_ = *ysize;
  transform->data_ = NULL;
  ++dec->next_transform_;
  assert(dec->next_transform_ <= NUM_TRANSFORMS);

  switch (type) {
    case PREDICTOR_TRANSFORM:
    case CROSS_COLOR_TRANSFORM:
      transform->bits_ = VP8LReadBits(br, 3) + 2;
      ok = DecodeImageStream(VP8LSubSampleSize(transform->xsize_,
                                               transform->bits_),
                             VP8LSubSampleSize(transform->ysize_,
                                               transform->bits_),
                             0, dec, &transform->data_);
      break;
    case COLOR_INDEXING_TRANSFORM: {
       const int num_colors = VP8LReadBits(br, 8) + 1;
       const int bits = (num_colors > 16) ? 0
                      : (num_colors > 4) ? 1
                      : (num_colors > 2) ? 2
                      : 3;
       *xsize = VP8LSubSampleSize(transform->xsize_, bits);
       transform->bits_ = bits;
       ok = DecodeImageStream(num_colors, 1, 0, dec, &transform->data_);
       ok = ok && ExpandColorMap(num_colors, transform);
      break;
    }
    case SUBTRACT_GREEN:
      break;
    default:
      assert(0);    // can't happen
      break;
  }

  return ok;
}

// -----------------------------------------------------------------------------
// VP8LMetadata

static void InitMetadata(VP8LMetadata* const hdr) {
  assert(hdr);
  memset(hdr, 0, sizeof(*hdr));
}

static void ClearMetadata(VP8LMetadata* const hdr) {
  assert(hdr);

  free(hdr->huffman_image_);
  DeleteHtreeGroups(hdr->htree_groups_, hdr->num_htree_groups_);
  VP8LColorCacheClear(&hdr->color_cache_);
  InitMetadata(hdr);
}

// -----------------------------------------------------------------------------
// VP8LDecoder

VP8LDecoder* VP8LNew(void) {
  VP8LDecoder* const dec = (VP8LDecoder*)calloc(1, sizeof(*dec));
  if (dec == NULL) return NULL;
  dec->status_ = VP8_STATUS_OK;
  dec->action_ = READ_DIM;
  dec->state_ = READ_DIM;

  VP8LDspInit();  // Init critical function pointers.

  return dec;
}

void VP8LClear(VP8LDecoder* const dec) {
  int i;
  if (dec == NULL) return;
  ClearMetadata(&dec->hdr_);

  free(dec->pixels_);
  dec->pixels_ = NULL;
  for (i = 0; i < dec->next_transform_; ++i) {
    ClearTransform(&dec->transforms_[i]);
  }
  dec->next_transform_ = 0;
  dec->transforms_seen_ = 0;

  free(dec->rescaler_memory);
  dec->rescaler_memory = NULL;

  dec->output_ = NULL;   // leave no trace behind
}

void VP8LDelete(VP8LDecoder* const dec) {
  if (dec != NULL) {
    VP8LClear(dec);
    free(dec);
  }
}

static void UpdateDecoder(VP8LDecoder* const dec, int width, int height) {
  VP8LMetadata* const hdr = &dec->hdr_;
  const int num_bits = hdr->huffman_subsample_bits_;
  dec->width_ = width;
  dec->height_ = height;

  hdr->huffman_xsize_ = VP8LSubSampleSize(width, num_bits);
  hdr->huffman_mask_ = (num_bits == 0) ? ~0 : (1 << num_bits) - 1;
}

static int DecodeImageStream(int xsize, int ysize,
                             int is_level0,
                             VP8LDecoder* const dec,
                             uint32_t** const decoded_data) {
  int ok = 1;
  int transform_xsize = xsize;
  int transform_ysize = ysize;
  VP8LBitReader* const br = &dec->br_;
  VP8LMetadata* const hdr = &dec->hdr_;
  uint32_t* data = NULL;
  int color_cache_bits = 0;

  // Read the transforms (may recurse).
  if (is_level0) {
    while (ok && VP8LReadBits(br, 1)) {
      ok = ReadTransform(&transform_xsize, &transform_ysize, dec);
    }
  }

  // Color cache
  if (ok && VP8LReadBits(br, 1)) {
    color_cache_bits = VP8LReadBits(br, 4);
    ok = (color_cache_bits >= 1 && color_cache_bits <= MAX_CACHE_BITS);
    if (!ok) {
      dec->status_ = VP8_STATUS_BITSTREAM_ERROR;
      goto End;
    }
  }

  // Read the Huffman codes (may recurse).
  ok = ok && ReadHuffmanCodes(dec, transform_xsize, transform_ysize,
                              color_cache_bits, is_level0);
  if (!ok) {
    dec->status_ = VP8_STATUS_BITSTREAM_ERROR;
    goto End;
  }

  // Finish setting up the color-cache
  if (color_cache_bits > 0) {
    hdr->color_cache_size_ = 1 << color_cache_bits;
    if (!VP8LColorCacheInit(&hdr->color_cache_, color_cache_bits)) {
      dec->status_ = VP8_STATUS_OUT_OF_MEMORY;
      ok = 0;
      goto End;
    }
  } else {
    hdr->color_cache_size_ = 0;
  }
  UpdateDecoder(dec, transform_xsize, transform_ysize);

  if (is_level0) {   // level 0 complete
    dec->state_ = READ_HDR;
    goto End;
  }

  {
    const uint64_t total_size = (uint64_t)transform_xsize * transform_ysize;
    data = (uint32_t*)WebPSafeMalloc(total_size, sizeof(*data));
    if (data == NULL) {
      dec->status_ = VP8_STATUS_OUT_OF_MEMORY;
      ok = 0;
      goto End;
    }
  }

  // Use the Huffman trees to decode the LZ77 encoded data.
  ok = DecodeImageData(dec, data, transform_xsize, transform_ysize,
                       transform_ysize, NULL);
  ok = ok && !br->error_;

 End:

  if (!ok) {
    free(data);
    ClearMetadata(hdr);
    // If not enough data (br.eos_) resulted in BIT_STREAM_ERROR, update the
    // status appropriately.
    if (dec->status_ == VP8_STATUS_BITSTREAM_ERROR && dec->br_.eos_) {
      dec->status_ = VP8_STATUS_SUSPENDED;
    }
  } else {
    if (decoded_data != NULL) {
      *decoded_data = data;
    } else {
      // We allocate image data in this function only for transforms. At level 0
      // (that is: not the transforms), we shouldn't have allocated anything.
      assert(data == NULL);
      assert(is_level0);
    }
    dec->last_pixel_ = 0;  // Reset for future DECODE_DATA_FUNC() calls.
    if (!is_level0) ClearMetadata(hdr);  // Clean up temporary data behind.
  }
  return ok;
}

//------------------------------------------------------------------------------
// Allocate internal buffers dec->pixels_ and dec->argb_cache_.
static int AllocateInternalBuffers32b(VP8LDecoder* const dec, int final_width) {
  const uint64_t num_pixels = (uint64_t)dec->width_ * dec->height_;
  // Scratch buffer corresponding to top-prediction row for transforming the
  // first row in the row-blocks. Not needed for paletted alpha.
  const uint64_t cache_top_pixels = (uint16_t)final_width;
  // Scratch buffer for temporary BGRA storage. Not needed for paletted alpha.
  const uint64_t cache_pixels = (uint64_t)final_width * NUM_ARGB_CACHE_ROWS;
  const uint64_t total_num_pixels =
      num_pixels + cache_top_pixels + cache_pixels;

  assert(dec->width_ <= final_width);
  dec->pixels_ = (uint32_t*)WebPSafeMalloc(total_num_pixels, sizeof(uint32_t));
  if (dec->pixels_ == NULL) {
    dec->argb_cache_ = NULL;    // for sanity check
    dec->status_ = VP8_STATUS_OUT_OF_MEMORY;
    return 0;
  }
  dec->argb_cache_ = dec->pixels_ + num_pixels + cache_top_pixels;
  return 1;
}

static int AllocateInternalBuffers8b(VP8LDecoder* const dec) {
  const uint64_t total_num_pixels = (uint64_t)dec->width_ * dec->height_;
  dec->argb_cache_ = NULL;    // for sanity check
  dec->pixels_ = (uint32_t*)WebPSafeMalloc(total_num_pixels, sizeof(uint8_t));
  if (dec->pixels_ == NULL) {
    dec->status_ = VP8_STATUS_OUT_OF_MEMORY;
    return 0;
  }
  return 1;
}

//------------------------------------------------------------------------------

// Special row-processing that only stores the alpha data.
static void ExtractAlphaRows(VP8LDecoder* const dec, int row) {
  const int num_rows = row - dec->last_row_;
  const uint32_t* const in = dec->pixels_ + dec->width_ * dec->last_row_;

  if (num_rows <= 0) return;  // Nothing to be done.
  ApplyInverseTransforms(dec, num_rows, in);

  // Extract alpha (which is stored in the green plane).
  {
    const int width = dec->io_->width;      // the final width (!= dec->width_)
    const int cache_pixs = width * num_rows;
    uint8_t* const dst = (uint8_t*)dec->io_->opaque + width * dec->last_row_;
    const uint32_t* const src = dec->argb_cache_;
    int i;
    for (i = 0; i < cache_pixs; ++i) dst[i] = (src[i] >> 8) & 0xff;
  }
  dec->last_row_ = dec->last_out_row_ = row;
}

int VP8LDecodeAlphaHeader(ALPHDecoder* const alph_dec,
                          const uint8_t* const data, size_t data_size,
                          uint8_t* const output) {
  int ok = 0;
  VP8LDecoder* dec;
  VP8Io* io;
  assert(alph_dec != NULL);
  alph_dec->vp8l_dec_ = VP8LNew();
  if (alph_dec->vp8l_dec_ == NULL) return 0;
  dec = alph_dec->vp8l_dec_;

  dec->width_ = alph_dec->width_;
  dec->height_ = alph_dec->height_;
  dec->io_ = &alph_dec->io_;
  io = dec->io_;

  VP8InitIo(io);
  WebPInitCustomIo(NULL, io);  // Just a sanity Init. io won't be used.
  io->opaque = output;
  io->width = alph_dec->width_;
  io->height = alph_dec->height_;

  dec->status_ = VP8_STATUS_OK;
  VP8LInitBitReader(&dec->br_, data, data_size);

  dec->action_ = READ_HDR;
  if (!DecodeImageStream(alph_dec->width_, alph_dec->height_, 1, dec, NULL)) {
    goto Err;
  }

  // Special case: if alpha data uses only the color indexing transform and
  // doesn't use color cache (a frequent case), we will use DecodeAlphaData()
  // method that only needs allocation of 1 byte per pixel (alpha channel).
  if (dec->next_transform_ == 1 &&
      dec->transforms_[0].type_ == COLOR_INDEXING_TRANSFORM &&
      Is8bOptimizable(&dec->hdr_)) {
    alph_dec->use_8b_decode = 1;
    ok = AllocateInternalBuffers8b(dec);
  } else {
    // Allocate internal buffers (note that dec->width_ may have changed here).
    alph_dec->use_8b_decode = 0;
    ok = AllocateInternalBuffers32b(dec, alph_dec->width_);
  }

  if (!ok) goto Err;

  dec->action_ = READ_DATA;
  return 1;

 Err:
  VP8LDelete(alph_dec->vp8l_dec_);
  alph_dec->vp8l_dec_ = NULL;
  return 0;
}

int VP8LDecodeAlphaImageStream(ALPHDecoder* const alph_dec, int last_row) {
  VP8LDecoder* const dec = alph_dec->vp8l_dec_;
  assert(dec != NULL);
  assert(dec->action_ == READ_DATA);
  assert(last_row <= dec->height_);

  // Decode (with special row processing).
  return alph_dec->use_8b_decode ?
      DecodeAlphaData(dec, (uint8_t*)dec->pixels_, dec->width_, dec->height_,
                      last_row) :
      DecodeImageData(dec, dec->pixels_, dec->width_, dec->height_,
                      last_row, ExtractAlphaRows);
}

//------------------------------------------------------------------------------

int VP8LDecodeHeader(VP8LDecoder* const dec, VP8Io* const io) {
  int width, height, has_alpha;

  if (dec == NULL) return 0;
  if (io == NULL) {
    dec->status_ = VP8_STATUS_INVALID_PARAM;
    return 0;
  }

  dec->io_ = io;
  dec->status_ = VP8_STATUS_OK;
  VP8LInitBitReader(&dec->br_, io->data, io->data_size);
  if (!ReadImageInfo(&dec->br_, &width, &height, &has_alpha)) {
    dec->status_ = VP8_STATUS_BITSTREAM_ERROR;
    goto Error;
  }
  dec->state_ = READ_DIM;
  io->width = width;
  io->height = height;

  dec->action_ = READ_HDR;
  if (!DecodeImageStream(width, height, 1, dec, NULL)) goto Error;
  return 1;

 Error:
  VP8LClear(dec);
  assert(dec->status_ != VP8_STATUS_OK);
  return 0;
}

int VP8LDecodeImage(VP8LDecoder* const dec) {
  VP8Io* io = NULL;
  WebPDecParams* params = NULL;

  // Sanity checks.
  if (dec == NULL) return 0;

  io = dec->io_;
  assert(io != NULL);
  params = (WebPDecParams*)io->opaque;
  assert(params != NULL);
  dec->output_ = params->output;
  assert(dec->output_ != NULL);

  // Initialization.
  if (!WebPIoInitFromOptions(params->options, io, MODE_BGRA)) {
    dec->status_ = VP8_STATUS_INVALID_PARAM;
    goto Err;
  }

  if (!AllocateInternalBuffers32b(dec, io->width)) goto Err;

  if (io->use_scaling && !AllocateAndInitRescaler(dec, io)) goto Err;

  // Decode.
  dec->action_ = READ_DATA;
  if (!DecodeImageData(dec, dec->pixels_, dec->width_, dec->height_,
                       dec->height_, ProcessRows)) {
    goto Err;
  }

  // Cleanup.
  params->last_y = dec->last_out_row_;
  VP8LClear(dec);
  return 1;

 Err:
  VP8LClear(dec);
  assert(dec->status_ != VP8_STATUS_OK);
  return 0;
}

//------------------------------------------------------------------------------

