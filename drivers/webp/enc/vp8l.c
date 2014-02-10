// Copyright 2012 Google Inc. All Rights Reserved.
//
// Use of this source code is governed by a BSD-style license
// that can be found in the COPYING file in the root of the source
// tree. An additional intellectual property rights grant can be found
// in the file PATENTS. All contributing project authors may
// be found in the AUTHORS file in the root of the source tree.
// -----------------------------------------------------------------------------
//
// main entry for the lossless encoder.
//
// Author: Vikas Arora (vikaas.arora@gmail.com)
//

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>

#include "./backward_references.h"
#include "./vp8enci.h"
#include "./vp8li.h"
#include "../dsp/lossless.h"
#include "../utils/bit_writer.h"
#include "../utils/huffman_encode.h"
#include "../utils/utils.h"
#include "../webp/format_constants.h"

#define PALETTE_KEY_RIGHT_SHIFT   22  // Key for 1K buffer.
#define MAX_HUFF_IMAGE_SIZE       (16 * 1024 * 1024)
#define MAX_COLORS_FOR_GRAPH      64

// -----------------------------------------------------------------------------
// Palette

static int CompareColors(const void* p1, const void* p2) {
  const uint32_t a = *(const uint32_t*)p1;
  const uint32_t b = *(const uint32_t*)p2;
  assert(a != b);
  return (a < b) ? -1 : 1;
}

// If number of colors in the image is less than or equal to MAX_PALETTE_SIZE,
// creates a palette and returns true, else returns false.
static int AnalyzeAndCreatePalette(const WebPPicture* const pic,
                                   uint32_t palette[MAX_PALETTE_SIZE],
                                   int* const palette_size) {
  int i, x, y, key;
  int num_colors = 0;
  uint8_t in_use[MAX_PALETTE_SIZE * 4] = { 0 };
  uint32_t colors[MAX_PALETTE_SIZE * 4];
  static const uint32_t kHashMul = 0x1e35a7bd;
  const uint32_t* argb = pic->argb;
  const int width = pic->width;
  const int height = pic->height;
  uint32_t last_pix = ~argb[0];   // so we're sure that last_pix != argb[0]

  for (y = 0; y < height; ++y) {
    for (x = 0; x < width; ++x) {
      if (argb[x] == last_pix) {
        continue;
      }
      last_pix = argb[x];
      key = (kHashMul * last_pix) >> PALETTE_KEY_RIGHT_SHIFT;
      while (1) {
        if (!in_use[key]) {
          colors[key] = last_pix;
          in_use[key] = 1;
          ++num_colors;
          if (num_colors > MAX_PALETTE_SIZE) {
            return 0;
          }
          break;
        } else if (colors[key] == last_pix) {
          // The color is already there.
          break;
        } else {
          // Some other color sits there.
          // Do linear conflict resolution.
          ++key;
          key &= (MAX_PALETTE_SIZE * 4 - 1);  // key mask for 1K buffer.
        }
      }
    }
    argb += pic->argb_stride;
  }

  // TODO(skal): could we reuse in_use[] to speed up EncodePalette()?
  num_colors = 0;
  for (i = 0; i < (int)(sizeof(in_use) / sizeof(in_use[0])); ++i) {
    if (in_use[i]) {
      palette[num_colors] = colors[i];
      ++num_colors;
    }
  }

  qsort(palette, num_colors, sizeof(*palette), CompareColors);
  *palette_size = num_colors;
  return 1;
}

static int AnalyzeEntropy(const uint32_t* argb,
                          int width, int height, int argb_stride,
                          double* const nonpredicted_bits,
                          double* const predicted_bits) {
  int x, y;
  const uint32_t* last_line = NULL;
  uint32_t last_pix = argb[0];    // so we're sure that pix_diff == 0

  VP8LHistogram* nonpredicted = NULL;
  VP8LHistogram* predicted =
      (VP8LHistogram*)malloc(2 * sizeof(*predicted));
  if (predicted == NULL) return 0;
  nonpredicted = predicted + 1;

  VP8LHistogramInit(predicted, 0);
  VP8LHistogramInit(nonpredicted, 0);
  for (y = 0; y < height; ++y) {
    for (x = 0; x < width; ++x) {
      const uint32_t pix = argb[x];
      const uint32_t pix_diff = VP8LSubPixels(pix, last_pix);
      if (pix_diff == 0) continue;
      if (last_line != NULL && pix == last_line[x]) {
        continue;
      }
      last_pix = pix;
      {
        const PixOrCopy pix_token = PixOrCopyCreateLiteral(pix);
        const PixOrCopy pix_diff_token = PixOrCopyCreateLiteral(pix_diff);
        VP8LHistogramAddSinglePixOrCopy(nonpredicted, &pix_token);
        VP8LHistogramAddSinglePixOrCopy(predicted, &pix_diff_token);
      }
    }
    last_line = argb;
    argb += argb_stride;
  }
  *nonpredicted_bits = VP8LHistogramEstimateBitsBulk(nonpredicted);
  *predicted_bits = VP8LHistogramEstimateBitsBulk(predicted);
  free(predicted);
  return 1;
}

static int VP8LEncAnalyze(VP8LEncoder* const enc, WebPImageHint image_hint) {
  const WebPPicture* const pic = enc->pic_;
  assert(pic != NULL && pic->argb != NULL);

  enc->use_palette_ =
      AnalyzeAndCreatePalette(pic, enc->palette_, &enc->palette_size_);

  if (image_hint == WEBP_HINT_GRAPH) {
    if (enc->use_palette_ && enc->palette_size_ < MAX_COLORS_FOR_GRAPH) {
      enc->use_palette_ = 0;
    }
  }

  if (!enc->use_palette_) {
    if (image_hint == WEBP_HINT_PHOTO) {
      enc->use_predict_ = 1;
      enc->use_cross_color_ = 1;
    } else {
      double non_pred_entropy, pred_entropy;
      if (!AnalyzeEntropy(pic->argb, pic->width, pic->height, pic->argb_stride,
                          &non_pred_entropy, &pred_entropy)) {
        return 0;
      }
      if (pred_entropy < 0.95 * non_pred_entropy) {
        enc->use_predict_ = 1;
        enc->use_cross_color_ = 1;
      }
    }
  }

  return 1;
}

static int GetHuffBitLengthsAndCodes(
    const VP8LHistogramSet* const histogram_image,
    HuffmanTreeCode* const huffman_codes) {
  int i, k;
  int ok = 1;
  uint64_t total_length_size = 0;
  uint8_t* mem_buf = NULL;
  const int histogram_image_size = histogram_image->size;

  // Iterate over all histograms and get the aggregate number of codes used.
  for (i = 0; i < histogram_image_size; ++i) {
    const VP8LHistogram* const histo = histogram_image->histograms[i];
    HuffmanTreeCode* const codes = &huffman_codes[5 * i];
    for (k = 0; k < 5; ++k) {
      const int num_symbols = (k == 0) ? VP8LHistogramNumCodes(histo)
                            : (k == 4) ? NUM_DISTANCE_CODES
                            : 256;
      codes[k].num_symbols = num_symbols;
      total_length_size += num_symbols;
    }
  }

  // Allocate and Set Huffman codes.
  {
    uint16_t* codes;
    uint8_t* lengths;
    mem_buf = (uint8_t*)WebPSafeCalloc(total_length_size,
                                       sizeof(*lengths) + sizeof(*codes));
    if (mem_buf == NULL) {
      ok = 0;
      goto End;
    }
    codes = (uint16_t*)mem_buf;
    lengths = (uint8_t*)&codes[total_length_size];
    for (i = 0; i < 5 * histogram_image_size; ++i) {
      const int bit_length = huffman_codes[i].num_symbols;
      huffman_codes[i].codes = codes;
      huffman_codes[i].code_lengths = lengths;
      codes += bit_length;
      lengths += bit_length;
    }
  }

  // Create Huffman trees.
  for (i = 0; ok && (i < histogram_image_size); ++i) {
    HuffmanTreeCode* const codes = &huffman_codes[5 * i];
    VP8LHistogram* const histo = histogram_image->histograms[i];
    ok = ok && VP8LCreateHuffmanTree(histo->literal_, 15, codes + 0);
    ok = ok && VP8LCreateHuffmanTree(histo->red_, 15, codes + 1);
    ok = ok && VP8LCreateHuffmanTree(histo->blue_, 15, codes + 2);
    ok = ok && VP8LCreateHuffmanTree(histo->alpha_, 15, codes + 3);
    ok = ok && VP8LCreateHuffmanTree(histo->distance_, 15, codes + 4);
  }

 End:
  if (!ok) {
    free(mem_buf);
    // If one VP8LCreateHuffmanTree() above fails, we need to clean up behind.
    memset(huffman_codes, 0, 5 * histogram_image_size * sizeof(*huffman_codes));
  }
  return ok;
}

static void StoreHuffmanTreeOfHuffmanTreeToBitMask(
    VP8LBitWriter* const bw, const uint8_t* code_length_bitdepth) {
  // RFC 1951 will calm you down if you are worried about this funny sequence.
  // This sequence is tuned from that, but more weighted for lower symbol count,
  // and more spiking histograms.
  static const uint8_t kStorageOrder[CODE_LENGTH_CODES] = {
    17, 18, 0, 1, 2, 3, 4, 5, 16, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15
  };
  int i;
  // Throw away trailing zeros:
  int codes_to_store = CODE_LENGTH_CODES;
  for (; codes_to_store > 4; --codes_to_store) {
    if (code_length_bitdepth[kStorageOrder[codes_to_store - 1]] != 0) {
      break;
    }
  }
  VP8LWriteBits(bw, 4, codes_to_store - 4);
  for (i = 0; i < codes_to_store; ++i) {
    VP8LWriteBits(bw, 3, code_length_bitdepth[kStorageOrder[i]]);
  }
}

static void ClearHuffmanTreeIfOnlyOneSymbol(
    HuffmanTreeCode* const huffman_code) {
  int k;
  int count = 0;
  for (k = 0; k < huffman_code->num_symbols; ++k) {
    if (huffman_code->code_lengths[k] != 0) {
      ++count;
      if (count > 1) return;
    }
  }
  for (k = 0; k < huffman_code->num_symbols; ++k) {
    huffman_code->code_lengths[k] = 0;
    huffman_code->codes[k] = 0;
  }
}

static void StoreHuffmanTreeToBitMask(
    VP8LBitWriter* const bw,
    const HuffmanTreeToken* const tokens, const int num_tokens,
    const HuffmanTreeCode* const huffman_code) {
  int i;
  for (i = 0; i < num_tokens; ++i) {
    const int ix = tokens[i].code;
    const int extra_bits = tokens[i].extra_bits;
    VP8LWriteBits(bw, huffman_code->code_lengths[ix], huffman_code->codes[ix]);
    switch (ix) {
      case 16:
        VP8LWriteBits(bw, 2, extra_bits);
        break;
      case 17:
        VP8LWriteBits(bw, 3, extra_bits);
        break;
      case 18:
        VP8LWriteBits(bw, 7, extra_bits);
        break;
    }
  }
}

static int StoreFullHuffmanCode(VP8LBitWriter* const bw,
                                const HuffmanTreeCode* const tree) {
  int ok = 0;
  uint8_t code_length_bitdepth[CODE_LENGTH_CODES] = { 0 };
  uint16_t code_length_bitdepth_symbols[CODE_LENGTH_CODES] = { 0 };
  const int max_tokens = tree->num_symbols;
  int num_tokens;
  HuffmanTreeCode huffman_code;
  HuffmanTreeToken* const tokens =
      (HuffmanTreeToken*)WebPSafeMalloc((uint64_t)max_tokens, sizeof(*tokens));
  if (tokens == NULL) return 0;

  huffman_code.num_symbols = CODE_LENGTH_CODES;
  huffman_code.code_lengths = code_length_bitdepth;
  huffman_code.codes = code_length_bitdepth_symbols;

  VP8LWriteBits(bw, 1, 0);
  num_tokens = VP8LCreateCompressedHuffmanTree(tree, tokens, max_tokens);
  {
    int histogram[CODE_LENGTH_CODES] = { 0 };
    int i;
    for (i = 0; i < num_tokens; ++i) {
      ++histogram[tokens[i].code];
    }

    if (!VP8LCreateHuffmanTree(histogram, 7, &huffman_code)) {
      goto End;
    }
  }

  StoreHuffmanTreeOfHuffmanTreeToBitMask(bw, code_length_bitdepth);
  ClearHuffmanTreeIfOnlyOneSymbol(&huffman_code);
  {
    int trailing_zero_bits = 0;
    int trimmed_length = num_tokens;
    int write_trimmed_length;
    int length;
    int i = num_tokens;
    while (i-- > 0) {
      const int ix = tokens[i].code;
      if (ix == 0 || ix == 17 || ix == 18) {
        --trimmed_length;   // discount trailing zeros
        trailing_zero_bits += code_length_bitdepth[ix];
        if (ix == 17) {
          trailing_zero_bits += 3;
        } else if (ix == 18) {
          trailing_zero_bits += 7;
        }
      } else {
        break;
      }
    }
    write_trimmed_length = (trimmed_length > 1 && trailing_zero_bits > 12);
    length = write_trimmed_length ? trimmed_length : num_tokens;
    VP8LWriteBits(bw, 1, write_trimmed_length);
    if (write_trimmed_length) {
      const int nbits = VP8LBitsLog2Ceiling(trimmed_length - 1);
      const int nbitpairs = (nbits == 0) ? 1 : (nbits + 1) / 2;
      VP8LWriteBits(bw, 3, nbitpairs - 1);
      assert(trimmed_length >= 2);
      VP8LWriteBits(bw, nbitpairs * 2, trimmed_length - 2);
    }
    StoreHuffmanTreeToBitMask(bw, tokens, length, &huffman_code);
  }
  ok = 1;
 End:
  free(tokens);
  return ok;
}

static int StoreHuffmanCode(VP8LBitWriter* const bw,
                            const HuffmanTreeCode* const huffman_code) {
  int i;
  int count = 0;
  int symbols[2] = { 0, 0 };
  const int kMaxBits = 8;
  const int kMaxSymbol = 1 << kMaxBits;

  // Check whether it's a small tree.
  for (i = 0; i < huffman_code->num_symbols && count < 3; ++i) {
    if (huffman_code->code_lengths[i] != 0) {
      if (count < 2) symbols[count] = i;
      ++count;
    }
  }

  if (count == 0) {   // emit minimal tree for empty cases
    // bits: small tree marker: 1, count-1: 0, large 8-bit code: 0, code: 0
    VP8LWriteBits(bw, 4, 0x01);
    return 1;
  } else if (count <= 2 && symbols[0] < kMaxSymbol && symbols[1] < kMaxSymbol) {
    VP8LWriteBits(bw, 1, 1);  // Small tree marker to encode 1 or 2 symbols.
    VP8LWriteBits(bw, 1, count - 1);
    if (symbols[0] <= 1) {
      VP8LWriteBits(bw, 1, 0);  // Code bit for small (1 bit) symbol value.
      VP8LWriteBits(bw, 1, symbols[0]);
    } else {
      VP8LWriteBits(bw, 1, 1);
      VP8LWriteBits(bw, 8, symbols[0]);
    }
    if (count == 2) {
      VP8LWriteBits(bw, 8, symbols[1]);
    }
    return 1;
  } else {
    return StoreFullHuffmanCode(bw, huffman_code);
  }
}

static void WriteHuffmanCode(VP8LBitWriter* const bw,
                             const HuffmanTreeCode* const code,
                             int code_index) {
  const int depth = code->code_lengths[code_index];
  const int symbol = code->codes[code_index];
  VP8LWriteBits(bw, depth, symbol);
}

static void StoreImageToBitMask(
    VP8LBitWriter* const bw, int width, int histo_bits,
    const VP8LBackwardRefs* const refs,
    const uint16_t* histogram_symbols,
    const HuffmanTreeCode* const huffman_codes) {
  // x and y trace the position in the image.
  int x = 0;
  int y = 0;
  const int histo_xsize = histo_bits ? VP8LSubSampleSize(width, histo_bits) : 1;
  int i;
  for (i = 0; i < refs->size; ++i) {
    const PixOrCopy* const v = &refs->refs[i];
    const int histogram_ix = histogram_symbols[histo_bits ?
                                               (y >> histo_bits) * histo_xsize +
                                               (x >> histo_bits) : 0];
    const HuffmanTreeCode* const codes = huffman_codes + 5 * histogram_ix;
    if (PixOrCopyIsCacheIdx(v)) {
      const int code = PixOrCopyCacheIdx(v);
      const int literal_ix = 256 + NUM_LENGTH_CODES + code;
      WriteHuffmanCode(bw, codes, literal_ix);
    } else if (PixOrCopyIsLiteral(v)) {
      static const int order[] = { 1, 2, 0, 3 };
      int k;
      for (k = 0; k < 4; ++k) {
        const int code = PixOrCopyLiteral(v, order[k]);
        WriteHuffmanCode(bw, codes + k, code);
      }
    } else {
      int bits, n_bits;
      int code, distance;

      VP8LPrefixEncode(v->len, &code, &n_bits, &bits);
      WriteHuffmanCode(bw, codes, 256 + code);
      VP8LWriteBits(bw, n_bits, bits);

      distance = PixOrCopyDistance(v);
      VP8LPrefixEncode(distance, &code, &n_bits, &bits);
      WriteHuffmanCode(bw, codes + 4, code);
      VP8LWriteBits(bw, n_bits, bits);
    }
    x += PixOrCopyLength(v);
    while (x >= width) {
      x -= width;
      ++y;
    }
  }
}

// Special case of EncodeImageInternal() for cache-bits=0, histo_bits=31
static int EncodeImageNoHuffman(VP8LBitWriter* const bw,
                                const uint32_t* const argb,
                                int width, int height, int quality) {
  int i;
  int ok = 0;
  VP8LBackwardRefs refs;
  HuffmanTreeCode huffman_codes[5] = { { 0, NULL, NULL } };
  const uint16_t histogram_symbols[1] = { 0 };    // only one tree, one symbol
  VP8LHistogramSet* const histogram_image = VP8LAllocateHistogramSet(1, 0);
  if (histogram_image == NULL) return 0;

  // Calculate backward references from ARGB image.
  if (!VP8LGetBackwardReferences(width, height, argb, quality, 0, 1, &refs)) {
    goto Error;
  }
  // Build histogram image and symbols from backward references.
  VP8LHistogramStoreRefs(&refs, histogram_image->histograms[0]);

  // Create Huffman bit lengths and codes for each histogram image.
  assert(histogram_image->size == 1);
  if (!GetHuffBitLengthsAndCodes(histogram_image, huffman_codes)) {
    goto Error;
  }

  // No color cache, no Huffman image.
  VP8LWriteBits(bw, 1, 0);

  // Store Huffman codes.
  for (i = 0; i < 5; ++i) {
    HuffmanTreeCode* const codes = &huffman_codes[i];
    if (!StoreHuffmanCode(bw, codes)) {
      goto Error;
    }
    ClearHuffmanTreeIfOnlyOneSymbol(codes);
  }

  // Store actual literals.
  StoreImageToBitMask(bw, width, 0, &refs, histogram_symbols, huffman_codes);
  ok = 1;

 Error:
  free(histogram_image);
  VP8LClearBackwardRefs(&refs);
  free(huffman_codes[0].codes);
  return ok;
}

static int EncodeImageInternal(VP8LBitWriter* const bw,
                               const uint32_t* const argb,
                               int width, int height, int quality,
                               int cache_bits, int histogram_bits) {
  int ok = 0;
  const int use_2d_locality = 1;
  const int use_color_cache = (cache_bits > 0);
  const uint32_t histogram_image_xysize =
      VP8LSubSampleSize(width, histogram_bits) *
      VP8LSubSampleSize(height, histogram_bits);
  VP8LHistogramSet* histogram_image =
      VP8LAllocateHistogramSet(histogram_image_xysize, 0);
  int histogram_image_size = 0;
  size_t bit_array_size = 0;
  HuffmanTreeCode* huffman_codes = NULL;
  VP8LBackwardRefs refs;
  uint16_t* const histogram_symbols =
      (uint16_t*)WebPSafeMalloc((uint64_t)histogram_image_xysize,
                                sizeof(*histogram_symbols));
  assert(histogram_bits >= MIN_HUFFMAN_BITS);
  assert(histogram_bits <= MAX_HUFFMAN_BITS);

  if (histogram_image == NULL || histogram_symbols == NULL) {
    free(histogram_image);
    free(histogram_symbols);
    return 0;
  }

  // Calculate backward references from ARGB image.
  if (!VP8LGetBackwardReferences(width, height, argb, quality, cache_bits,
                                 use_2d_locality, &refs)) {
    goto Error;
  }
  // Build histogram image and symbols from backward references.
  if (!VP8LGetHistoImageSymbols(width, height, &refs,
                                quality, histogram_bits, cache_bits,
                                histogram_image,
                                histogram_symbols)) {
    goto Error;
  }
  // Create Huffman bit lengths and codes for each histogram image.
  histogram_image_size = histogram_image->size;
  bit_array_size = 5 * histogram_image_size;
  huffman_codes = (HuffmanTreeCode*)WebPSafeCalloc(bit_array_size,
                                                   sizeof(*huffman_codes));
  if (huffman_codes == NULL ||
      !GetHuffBitLengthsAndCodes(histogram_image, huffman_codes)) {
    goto Error;
  }
  // Free combined histograms.
  free(histogram_image);
  histogram_image = NULL;

  // Color Cache parameters.
  VP8LWriteBits(bw, 1, use_color_cache);
  if (use_color_cache) {
    VP8LWriteBits(bw, 4, cache_bits);
  }

  // Huffman image + meta huffman.
  {
    const int write_histogram_image = (histogram_image_size > 1);
    VP8LWriteBits(bw, 1, write_histogram_image);
    if (write_histogram_image) {
      uint32_t* const histogram_argb =
          (uint32_t*)WebPSafeMalloc((uint64_t)histogram_image_xysize,
                                    sizeof(*histogram_argb));
      int max_index = 0;
      uint32_t i;
      if (histogram_argb == NULL) goto Error;
      for (i = 0; i < histogram_image_xysize; ++i) {
        const int symbol_index = histogram_symbols[i] & 0xffff;
        histogram_argb[i] = 0xff000000 | (symbol_index << 8);
        if (symbol_index >= max_index) {
          max_index = symbol_index + 1;
        }
      }
      histogram_image_size = max_index;

      VP8LWriteBits(bw, 3, histogram_bits - 2);
      ok = EncodeImageNoHuffman(bw, histogram_argb,
                                VP8LSubSampleSize(width, histogram_bits),
                                VP8LSubSampleSize(height, histogram_bits),
                                quality);
      free(histogram_argb);
      if (!ok) goto Error;
    }
  }

  // Store Huffman codes.
  {
    int i;
    for (i = 0; i < 5 * histogram_image_size; ++i) {
      HuffmanTreeCode* const codes = &huffman_codes[i];
      if (!StoreHuffmanCode(bw, codes)) goto Error;
      ClearHuffmanTreeIfOnlyOneSymbol(codes);
    }
  }

  // Store actual literals.
  StoreImageToBitMask(bw, width, histogram_bits, &refs,
                      histogram_symbols, huffman_codes);
  ok = 1;

 Error:
  free(histogram_image);

  VP8LClearBackwardRefs(&refs);
  if (huffman_codes != NULL) {
    free(huffman_codes->codes);
    free(huffman_codes);
  }
  free(histogram_symbols);
  return ok;
}

// -----------------------------------------------------------------------------
// Transforms

// Check if it would be a good idea to subtract green from red and blue. We
// only impact entropy in red/blue components, don't bother to look at others.
static int EvalAndApplySubtractGreen(VP8LEncoder* const enc,
                                     int width, int height,
                                     VP8LBitWriter* const bw) {
  if (!enc->use_palette_) {
    int i;
    const uint32_t* const argb = enc->argb_;
    double bit_cost_before, bit_cost_after;
    VP8LHistogram* const histo = (VP8LHistogram*)malloc(sizeof(*histo));
    if (histo == NULL) return 0;

    VP8LHistogramInit(histo, 1);
    for (i = 0; i < width * height; ++i) {
      const uint32_t c = argb[i];
      ++histo->red_[(c >> 16) & 0xff];
      ++histo->blue_[(c >> 0) & 0xff];
    }
    bit_cost_before = VP8LHistogramEstimateBits(histo);

    VP8LHistogramInit(histo, 1);
    for (i = 0; i < width * height; ++i) {
      const uint32_t c = argb[i];
      const int green = (c >> 8) & 0xff;
      ++histo->red_[((c >> 16) - green) & 0xff];
      ++histo->blue_[((c >> 0) - green) & 0xff];
    }
    bit_cost_after = VP8LHistogramEstimateBits(histo);
    free(histo);

    // Check if subtracting green yields low entropy.
    enc->use_subtract_green_ = (bit_cost_after < bit_cost_before);
    if (enc->use_subtract_green_) {
      VP8LWriteBits(bw, 1, TRANSFORM_PRESENT);
      VP8LWriteBits(bw, 2, SUBTRACT_GREEN);
      VP8LSubtractGreenFromBlueAndRed(enc->argb_, width * height);
    }
  }
  return 1;
}

static int ApplyPredictFilter(const VP8LEncoder* const enc,
                              int width, int height, int quality,
                              VP8LBitWriter* const bw) {
  const int pred_bits = enc->transform_bits_;
  const int transform_width = VP8LSubSampleSize(width, pred_bits);
  const int transform_height = VP8LSubSampleSize(height, pred_bits);

  VP8LResidualImage(width, height, pred_bits, enc->argb_, enc->argb_scratch_,
                    enc->transform_data_);
  VP8LWriteBits(bw, 1, TRANSFORM_PRESENT);
  VP8LWriteBits(bw, 2, PREDICTOR_TRANSFORM);
  assert(pred_bits >= 2);
  VP8LWriteBits(bw, 3, pred_bits - 2);
  if (!EncodeImageNoHuffman(bw, enc->transform_data_,
                            transform_width, transform_height, quality)) {
    return 0;
  }
  return 1;
}

static int ApplyCrossColorFilter(const VP8LEncoder* const enc,
                                 int width, int height, int quality,
                                 VP8LBitWriter* const bw) {
  const int ccolor_transform_bits = enc->transform_bits_;
  const int transform_width = VP8LSubSampleSize(width, ccolor_transform_bits);
  const int transform_height = VP8LSubSampleSize(height, ccolor_transform_bits);
  const int step = (quality < 25) ? 32 : (quality > 50) ? 8 : 16;

  VP8LColorSpaceTransform(width, height, ccolor_transform_bits, step,
                          enc->argb_, enc->transform_data_);
  VP8LWriteBits(bw, 1, TRANSFORM_PRESENT);
  VP8LWriteBits(bw, 2, CROSS_COLOR_TRANSFORM);
  assert(ccolor_transform_bits >= 2);
  VP8LWriteBits(bw, 3, ccolor_transform_bits - 2);
  if (!EncodeImageNoHuffman(bw, enc->transform_data_,
                            transform_width, transform_height, quality)) {
    return 0;
  }
  return 1;
}

// -----------------------------------------------------------------------------

static WebPEncodingError WriteRiffHeader(const WebPPicture* const pic,
                                         size_t riff_size, size_t vp8l_size) {
  uint8_t riff[RIFF_HEADER_SIZE + CHUNK_HEADER_SIZE + VP8L_SIGNATURE_SIZE] = {
    'R', 'I', 'F', 'F', 0, 0, 0, 0, 'W', 'E', 'B', 'P',
    'V', 'P', '8', 'L', 0, 0, 0, 0, VP8L_MAGIC_BYTE,
  };
  PutLE32(riff + TAG_SIZE, (uint32_t)riff_size);
  PutLE32(riff + RIFF_HEADER_SIZE + TAG_SIZE, (uint32_t)vp8l_size);
  if (!pic->writer(riff, sizeof(riff), pic)) {
    return VP8_ENC_ERROR_BAD_WRITE;
  }
  return VP8_ENC_OK;
}

static int WriteImageSize(const WebPPicture* const pic,
                          VP8LBitWriter* const bw) {
  const int width = pic->width - 1;
  const int height = pic->height - 1;
  assert(width < WEBP_MAX_DIMENSION && height < WEBP_MAX_DIMENSION);

  VP8LWriteBits(bw, VP8L_IMAGE_SIZE_BITS, width);
  VP8LWriteBits(bw, VP8L_IMAGE_SIZE_BITS, height);
  return !bw->error_;
}

static int WriteRealAlphaAndVersion(VP8LBitWriter* const bw, int has_alpha) {
  VP8LWriteBits(bw, 1, has_alpha);
  VP8LWriteBits(bw, VP8L_VERSION_BITS, VP8L_VERSION);
  return !bw->error_;
}

static WebPEncodingError WriteImage(const WebPPicture* const pic,
                                    VP8LBitWriter* const bw,
                                    size_t* const coded_size) {
  WebPEncodingError err = VP8_ENC_OK;
  const uint8_t* const webpll_data = VP8LBitWriterFinish(bw);
  const size_t webpll_size = VP8LBitWriterNumBytes(bw);
  const size_t vp8l_size = VP8L_SIGNATURE_SIZE + webpll_size;
  const size_t pad = vp8l_size & 1;
  const size_t riff_size = TAG_SIZE + CHUNK_HEADER_SIZE + vp8l_size + pad;

  err = WriteRiffHeader(pic, riff_size, vp8l_size);
  if (err != VP8_ENC_OK) goto Error;

  if (!pic->writer(webpll_data, webpll_size, pic)) {
    err = VP8_ENC_ERROR_BAD_WRITE;
    goto Error;
  }

  if (pad) {
    const uint8_t pad_byte[1] = { 0 };
    if (!pic->writer(pad_byte, 1, pic)) {
      err = VP8_ENC_ERROR_BAD_WRITE;
      goto Error;
    }
  }
  *coded_size = CHUNK_HEADER_SIZE + riff_size;
  return VP8_ENC_OK;

 Error:
  return err;
}

// -----------------------------------------------------------------------------

// Allocates the memory for argb (W x H) buffer, 2 rows of context for
// prediction and transform data.
static WebPEncodingError AllocateTransformBuffer(VP8LEncoder* const enc,
                                                 int width, int height) {
  WebPEncodingError err = VP8_ENC_OK;
  const int tile_size = 1 << enc->transform_bits_;
  const uint64_t image_size = width * height;
  const uint64_t argb_scratch_size = tile_size * width + width;
  const uint64_t transform_data_size =
      (uint64_t)VP8LSubSampleSize(width, enc->transform_bits_) *
      (uint64_t)VP8LSubSampleSize(height, enc->transform_bits_);
  const uint64_t total_size =
      image_size + argb_scratch_size + transform_data_size;
  uint32_t* mem = (uint32_t*)WebPSafeMalloc(total_size, sizeof(*mem));
  if (mem == NULL) {
    err = VP8_ENC_ERROR_OUT_OF_MEMORY;
    goto Error;
  }
  enc->argb_ = mem;
  mem += image_size;
  enc->argb_scratch_ = mem;
  mem += argb_scratch_size;
  enc->transform_data_ = mem;
  enc->current_width_ = width;

 Error:
  return err;
}

static void ApplyPalette(uint32_t* src, uint32_t* dst,
                         uint32_t src_stride, uint32_t dst_stride,
                         const uint32_t* palette, int palette_size,
                         int width, int height, int xbits, uint8_t* row) {
  int i, x, y;
  int use_LUT = 1;
  for (i = 0; i < palette_size; ++i) {
    if ((palette[i] & 0xffff00ffu) != 0) {
      use_LUT = 0;
      break;
    }
  }

  if (use_LUT) {
    uint8_t inv_palette[MAX_PALETTE_SIZE] = { 0 };
    for (i = 0; i < palette_size; ++i) {
      const int color = (palette[i] >> 8) & 0xff;
      inv_palette[color] = i;
    }
    for (y = 0; y < height; ++y) {
      for (x = 0; x < width; ++x) {
        const int color = (src[x] >> 8) & 0xff;
        row[x] = inv_palette[color];
      }
      VP8LBundleColorMap(row, width, xbits, dst);
      src += src_stride;
      dst += dst_stride;
    }
  } else {
    // Use 1 pixel cache for ARGB pixels.
    uint32_t last_pix = palette[0];
    int last_idx = 0;
    for (y = 0; y < height; ++y) {
      for (x = 0; x < width; ++x) {
        const uint32_t pix = src[x];
        if (pix != last_pix) {
          for (i = 0; i < palette_size; ++i) {
            if (pix == palette[i]) {
              last_idx = i;
              last_pix = pix;
              break;
            }
          }
        }
        row[x] = last_idx;
      }
      VP8LBundleColorMap(row, width, xbits, dst);
      src += src_stride;
      dst += dst_stride;
    }
  }
}

// Note: Expects "enc->palette_" to be set properly.
// Also, "enc->palette_" will be modified after this call and should not be used
// later.
static WebPEncodingError EncodePalette(VP8LBitWriter* const bw,
                                       VP8LEncoder* const enc, int quality) {
  WebPEncodingError err = VP8_ENC_OK;
  int i;
  const WebPPicture* const pic = enc->pic_;
  uint32_t* src = pic->argb;
  uint32_t* dst;
  const int width = pic->width;
  const int height = pic->height;
  uint32_t* const palette = enc->palette_;
  const int palette_size = enc->palette_size_;
  uint8_t* row = NULL;
  int xbits;

  // Replace each input pixel by corresponding palette index.
  // This is done line by line.
  if (palette_size <= 4) {
    xbits = (palette_size <= 2) ? 3 : 2;
  } else {
    xbits = (palette_size <= 16) ? 1 : 0;
  }

  err = AllocateTransformBuffer(enc, VP8LSubSampleSize(width, xbits), height);
  if (err != VP8_ENC_OK) goto Error;
  dst = enc->argb_;

  row = (uint8_t*)WebPSafeMalloc((uint64_t)width, sizeof(*row));
  if (row == NULL) return VP8_ENC_ERROR_OUT_OF_MEMORY;

  ApplyPalette(src, dst, pic->argb_stride, enc->current_width_,
               palette, palette_size, width, height, xbits, row);

  // Save palette to bitstream.
  VP8LWriteBits(bw, 1, TRANSFORM_PRESENT);
  VP8LWriteBits(bw, 2, COLOR_INDEXING_TRANSFORM);
  assert(palette_size >= 1);
  VP8LWriteBits(bw, 8, palette_size - 1);
  for (i = palette_size - 1; i >= 1; --i) {
    palette[i] = VP8LSubPixels(palette[i], palette[i - 1]);
  }
  if (!EncodeImageNoHuffman(bw, palette, palette_size, 1, quality)) {
    err = VP8_ENC_ERROR_INVALID_CONFIGURATION;
    goto Error;
  }

 Error:
  free(row);
  return err;
}

// -----------------------------------------------------------------------------

static int GetHistoBits(int method, int use_palette, int width, int height) {
  const uint64_t hist_size = sizeof(VP8LHistogram);
  // Make tile size a function of encoding method (Range: 0 to 6).
  int histo_bits = (use_palette ? 9 : 7) - method;
  while (1) {
    const uint64_t huff_image_size = VP8LSubSampleSize(width, histo_bits) *
                                     VP8LSubSampleSize(height, histo_bits) *
                                     hist_size;
    if (huff_image_size <= MAX_HUFF_IMAGE_SIZE) break;
    ++histo_bits;
  }
  return (histo_bits < MIN_HUFFMAN_BITS) ? MIN_HUFFMAN_BITS :
         (histo_bits > MAX_HUFFMAN_BITS) ? MAX_HUFFMAN_BITS : histo_bits;
}

static void FinishEncParams(VP8LEncoder* const enc) {
  const WebPConfig* const config = enc->config_;
  const WebPPicture* const pic = enc->pic_;
  const int method = config->method;
  const float quality = config->quality;
  const int use_palette = enc->use_palette_;
  enc->transform_bits_ = (method < 4) ? 5 : (method > 4) ? 3 : 4;
  enc->histo_bits_ = GetHistoBits(method, use_palette, pic->width, pic->height);
  enc->cache_bits_ = (quality <= 25.f) ? 0 : 7;
}

// -----------------------------------------------------------------------------
// VP8LEncoder

static VP8LEncoder* VP8LEncoderNew(const WebPConfig* const config,
                                   const WebPPicture* const picture) {
  VP8LEncoder* const enc = (VP8LEncoder*)calloc(1, sizeof(*enc));
  if (enc == NULL) {
    WebPEncodingSetError(picture, VP8_ENC_ERROR_OUT_OF_MEMORY);
    return NULL;
  }
  enc->config_ = config;
  enc->pic_ = picture;

  VP8LDspInit();

  return enc;
}

static void VP8LEncoderDelete(VP8LEncoder* enc) {
  free(enc->argb_);
  free(enc);
}

// -----------------------------------------------------------------------------
// Main call

WebPEncodingError VP8LEncodeStream(const WebPConfig* const config,
                                   const WebPPicture* const picture,
                                   VP8LBitWriter* const bw) {
  WebPEncodingError err = VP8_ENC_OK;
  const int quality = (int)config->quality;
  const int width = picture->width;
  const int height = picture->height;
  VP8LEncoder* const enc = VP8LEncoderNew(config, picture);
  const size_t byte_position = VP8LBitWriterNumBytes(bw);

  if (enc == NULL) {
    err = VP8_ENC_ERROR_OUT_OF_MEMORY;
    goto Error;
  }

  // ---------------------------------------------------------------------------
  // Analyze image (entropy, num_palettes etc)

  if (!VP8LEncAnalyze(enc, config->image_hint)) {
    err = VP8_ENC_ERROR_OUT_OF_MEMORY;
    goto Error;
  }

  FinishEncParams(enc);

  if (enc->use_palette_) {
    err = EncodePalette(bw, enc, quality);
    if (err != VP8_ENC_OK) goto Error;
    // Color cache is disabled for palette.
    enc->cache_bits_ = 0;
  }

  // In case image is not packed.
  if (enc->argb_ == NULL) {
    int y;
    err = AllocateTransformBuffer(enc, width, height);
    if (err != VP8_ENC_OK) goto Error;
    for (y = 0; y < height; ++y) {
      memcpy(enc->argb_ + y * width,
             picture->argb + y * picture->argb_stride,
             width * sizeof(*enc->argb_));
    }
    enc->current_width_ = width;
  }

  // ---------------------------------------------------------------------------
  // Apply transforms and write transform data.

  if (!EvalAndApplySubtractGreen(enc, enc->current_width_, height, bw)) {
    err = VP8_ENC_ERROR_OUT_OF_MEMORY;
    goto Error;
  }

  if (enc->use_predict_) {
    if (!ApplyPredictFilter(enc, enc->current_width_, height, quality, bw)) {
      err = VP8_ENC_ERROR_INVALID_CONFIGURATION;
      goto Error;
    }
  }

  if (enc->use_cross_color_) {
    if (!ApplyCrossColorFilter(enc, enc->current_width_, height, quality, bw)) {
      err = VP8_ENC_ERROR_INVALID_CONFIGURATION;
      goto Error;
    }
  }

  VP8LWriteBits(bw, 1, !TRANSFORM_PRESENT);  // No more transforms.

  // ---------------------------------------------------------------------------
  // Estimate the color cache size.

  if (enc->cache_bits_ > 0) {
    if (!VP8LCalculateEstimateForCacheSize(enc->argb_, enc->current_width_,
                                           height, &enc->cache_bits_)) {
      err = VP8_ENC_ERROR_INVALID_CONFIGURATION;
      goto Error;
    }
  }

  // ---------------------------------------------------------------------------
  // Encode and write the transformed image.

  if (!EncodeImageInternal(bw, enc->argb_, enc->current_width_, height,
                           quality, enc->cache_bits_, enc->histo_bits_)) {
    err = VP8_ENC_ERROR_OUT_OF_MEMORY;
    goto Error;
  }

  if (picture->stats != NULL) {
    WebPAuxStats* const stats = picture->stats;
    stats->lossless_features = 0;
    if (enc->use_predict_) stats->lossless_features |= 1;
    if (enc->use_cross_color_) stats->lossless_features |= 2;
    if (enc->use_subtract_green_) stats->lossless_features |= 4;
    if (enc->use_palette_) stats->lossless_features |= 8;
    stats->histogram_bits = enc->histo_bits_;
    stats->transform_bits = enc->transform_bits_;
    stats->cache_bits = enc->cache_bits_;
    stats->palette_size = enc->palette_size_;
    stats->lossless_size = (int)(VP8LBitWriterNumBytes(bw) - byte_position);
  }

 Error:
  VP8LEncoderDelete(enc);
  return err;
}

int VP8LEncodeImage(const WebPConfig* const config,
                    const WebPPicture* const picture) {
  int width, height;
  int has_alpha;
  size_t coded_size;
  int percent = 0;
  WebPEncodingError err = VP8_ENC_OK;
  VP8LBitWriter bw;

  if (picture == NULL) return 0;

  if (config == NULL || picture->argb == NULL) {
    err = VP8_ENC_ERROR_NULL_PARAMETER;
    WebPEncodingSetError(picture, err);
    return 0;
  }

  width = picture->width;
  height = picture->height;
  if (!VP8LBitWriterInit(&bw, (width * height) >> 1)) {
    err = VP8_ENC_ERROR_OUT_OF_MEMORY;
    goto Error;
  }

  if (!WebPReportProgress(picture, 1, &percent)) {
 UserAbort:
    err = VP8_ENC_ERROR_USER_ABORT;
    goto Error;
  }
  // Reset stats (for pure lossless coding)
  if (picture->stats != NULL) {
    WebPAuxStats* const stats = picture->stats;
    memset(stats, 0, sizeof(*stats));
    stats->PSNR[0] = 99.f;
    stats->PSNR[1] = 99.f;
    stats->PSNR[2] = 99.f;
    stats->PSNR[3] = 99.f;
    stats->PSNR[4] = 99.f;
  }

  // Write image size.
  if (!WriteImageSize(picture, &bw)) {
    err = VP8_ENC_ERROR_OUT_OF_MEMORY;
    goto Error;
  }

  has_alpha = WebPPictureHasTransparency(picture);
  // Write the non-trivial Alpha flag and lossless version.
  if (!WriteRealAlphaAndVersion(&bw, has_alpha)) {
    err = VP8_ENC_ERROR_OUT_OF_MEMORY;
    goto Error;
  }

  if (!WebPReportProgress(picture, 5, &percent)) goto UserAbort;

  // Encode main image stream.
  err = VP8LEncodeStream(config, picture, &bw);
  if (err != VP8_ENC_OK) goto Error;

  // TODO(skal): have a fine-grained progress report in VP8LEncodeStream().
  if (!WebPReportProgress(picture, 90, &percent)) goto UserAbort;

  // Finish the RIFF chunk.
  err = WriteImage(picture, &bw, &coded_size);
  if (err != VP8_ENC_OK) goto Error;

  if (!WebPReportProgress(picture, 100, &percent)) goto UserAbort;

  // Save size.
  if (picture->stats != NULL) {
    picture->stats->coded_size += (int)coded_size;
    picture->stats->lossless_size = (int)coded_size;
  }

  if (picture->extra_info != NULL) {
    const int mb_w = (width + 15) >> 4;
    const int mb_h = (height + 15) >> 4;
    memset(picture->extra_info, 0, mb_w * mb_h * sizeof(*picture->extra_info));
  }

 Error:
  if (bw.error_) err = VP8_ENC_ERROR_OUT_OF_MEMORY;
  VP8LBitWriterDestroy(&bw);
  if (err != VP8_ENC_OK) {
    WebPEncodingSetError(picture, err);
    return 0;
  }
  return 1;
}

//------------------------------------------------------------------------------

