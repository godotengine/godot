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
#include <stdlib.h>

#include "./backward_references.h"
#include "./histogram.h"
#include "./vp8enci.h"
#include "./vp8li.h"
#include "../dsp/lossless.h"
#include "../utils/bit_writer.h"
#include "../utils/huffman_encode.h"
#include "../utils/utils.h"
#include "../webp/format_constants.h"

#include "./delta_palettization.h"

#define PALETTE_KEY_RIGHT_SHIFT   22  // Key for 1K buffer.
// Maximum number of histogram images (sub-blocks).
#define MAX_HUFF_IMAGE_SIZE       2600

// Palette reordering for smaller sum of deltas (and for smaller storage).

static int PaletteCompareColorsForQsort(const void* p1, const void* p2) {
  const uint32_t a = WebPMemToUint32((uint8_t*)p1);
  const uint32_t b = WebPMemToUint32((uint8_t*)p2);
  assert(a != b);
  return (a < b) ? -1 : 1;
}

static WEBP_INLINE uint32_t PaletteComponentDistance(uint32_t v) {
  return (v <= 128) ? v : (256 - v);
}

// Computes a value that is related to the entropy created by the
// palette entry diff.
//
// Note that the last & 0xff is a no-operation in the next statement, but
// removed by most compilers and is here only for regularity of the code.
static WEBP_INLINE uint32_t PaletteColorDistance(uint32_t col1, uint32_t col2) {
  const uint32_t diff = VP8LSubPixels(col1, col2);
  const int kMoreWeightForRGBThanForAlpha = 9;
  uint32_t score;
  score =  PaletteComponentDistance((diff >>  0) & 0xff);
  score += PaletteComponentDistance((diff >>  8) & 0xff);
  score += PaletteComponentDistance((diff >> 16) & 0xff);
  score *= kMoreWeightForRGBThanForAlpha;
  score += PaletteComponentDistance((diff >> 24) & 0xff);
  return score;
}

static WEBP_INLINE void SwapColor(uint32_t* const col1, uint32_t* const col2) {
  const uint32_t tmp = *col1;
  *col1 = *col2;
  *col2 = tmp;
}

static void GreedyMinimizeDeltas(uint32_t palette[], int num_colors) {
  // Find greedily always the closest color of the predicted color to minimize
  // deltas in the palette. This reduces storage needs since the
  // palette is stored with delta encoding.
  uint32_t predict = 0x00000000;
  int i, k;
  for (i = 0; i < num_colors; ++i) {
    int best_ix = i;
    uint32_t best_score = ~0U;
    for (k = i; k < num_colors; ++k) {
      const uint32_t cur_score = PaletteColorDistance(palette[k], predict);
      if (best_score > cur_score) {
        best_score = cur_score;
        best_ix = k;
      }
    }
    SwapColor(&palette[best_ix], &palette[i]);
    predict = palette[i];
  }
}

// The palette has been sorted by alpha. This function checks if the other
// components of the palette have a monotonic development with regards to
// position in the palette. If all have monotonic development, there is
// no benefit to re-organize them greedily. A monotonic development
// would be spotted in green-only situations (like lossy alpha) or gray-scale
// images.
static int PaletteHasNonMonotonousDeltas(uint32_t palette[], int num_colors) {
  uint32_t predict = 0x000000;
  int i;
  uint8_t sign_found = 0x00;
  for (i = 0; i < num_colors; ++i) {
    const uint32_t diff = VP8LSubPixels(palette[i], predict);
    const uint8_t rd = (diff >> 16) & 0xff;
    const uint8_t gd = (diff >>  8) & 0xff;
    const uint8_t bd = (diff >>  0) & 0xff;
    if (rd != 0x00) {
      sign_found |= (rd < 0x80) ? 1 : 2;
    }
    if (gd != 0x00) {
      sign_found |= (gd < 0x80) ? 8 : 16;
    }
    if (bd != 0x00) {
      sign_found |= (bd < 0x80) ? 64 : 128;
    }
    predict = palette[i];
  }
  return (sign_found & (sign_found << 1)) != 0;  // two consequent signs.
}

// -----------------------------------------------------------------------------
// Palette

// If number of colors in the image is less than or equal to MAX_PALETTE_SIZE,
// creates a palette and returns true, else returns false.
static int AnalyzeAndCreatePalette(const WebPPicture* const pic,
                                   int low_effort,
                                   uint32_t palette[MAX_PALETTE_SIZE],
                                   int* const palette_size) {
  const int num_colors = WebPGetColorPalette(pic, palette);
  if (num_colors > MAX_PALETTE_SIZE) return 0;
  *palette_size = num_colors;
  qsort(palette, num_colors, sizeof(*palette), PaletteCompareColorsForQsort);
  if (!low_effort && PaletteHasNonMonotonousDeltas(palette, num_colors)) {
    GreedyMinimizeDeltas(palette, num_colors);
  }
  return 1;
}

// These five modes are evaluated and their respective entropy is computed.
typedef enum {
  kDirect = 0,
  kSpatial = 1,
  kSubGreen = 2,
  kSpatialSubGreen = 3,
  kPalette = 4,
  kNumEntropyIx = 5
} EntropyIx;

typedef enum {
  kHistoAlpha = 0,
  kHistoAlphaPred,
  kHistoGreen,
  kHistoGreenPred,
  kHistoRed,
  kHistoRedPred,
  kHistoBlue,
  kHistoBluePred,
  kHistoRedSubGreen,
  kHistoRedPredSubGreen,
  kHistoBlueSubGreen,
  kHistoBluePredSubGreen,
  kHistoPalette,
  kHistoTotal  // Must be last.
} HistoIx;

static void AddSingleSubGreen(uint32_t p, uint32_t* r, uint32_t* b) {
  const uint32_t green = p >> 8;  // The upper bits are masked away later.
  ++r[((p >> 16) - green) & 0xff];
  ++b[(p - green) & 0xff];
}

static void AddSingle(uint32_t p,
                      uint32_t* a, uint32_t* r, uint32_t* g, uint32_t* b) {
  ++a[p >> 24];
  ++r[(p >> 16) & 0xff];
  ++g[(p >> 8) & 0xff];
  ++b[(p & 0xff)];
}

static int AnalyzeEntropy(const uint32_t* argb,
                          int width, int height, int argb_stride,
                          int use_palette,
                          EntropyIx* const min_entropy_ix,
                          int* const red_and_blue_always_zero) {
  // Allocate histogram set with cache_bits = 0.
  uint32_t* const histo =
      (uint32_t*)WebPSafeCalloc(kHistoTotal, sizeof(*histo) * 256);
  if (histo != NULL) {
    int i, x, y;
    const uint32_t* prev_row = argb;
    const uint32_t* curr_row = argb + argb_stride;
    for (y = 1; y < height; ++y) {
      uint32_t prev_pix = curr_row[0];
      for (x = 1; x < width; ++x) {
        const uint32_t pix = curr_row[x];
        const uint32_t pix_diff = VP8LSubPixels(pix, prev_pix);
        if ((pix_diff == 0) || (pix == prev_row[x])) continue;
        prev_pix = pix;
        AddSingle(pix,
                  &histo[kHistoAlpha * 256],
                  &histo[kHistoRed * 256],
                  &histo[kHistoGreen * 256],
                  &histo[kHistoBlue * 256]);
        AddSingle(pix_diff,
                  &histo[kHistoAlphaPred * 256],
                  &histo[kHistoRedPred * 256],
                  &histo[kHistoGreenPred * 256],
                  &histo[kHistoBluePred * 256]);
        AddSingleSubGreen(pix,
                          &histo[kHistoRedSubGreen * 256],
                          &histo[kHistoBlueSubGreen * 256]);
        AddSingleSubGreen(pix_diff,
                          &histo[kHistoRedPredSubGreen * 256],
                          &histo[kHistoBluePredSubGreen * 256]);
        {
          // Approximate the palette by the entropy of the multiplicative hash.
          const int hash = ((pix + (pix >> 19)) * 0x39c5fba7) >> 24;
          ++histo[kHistoPalette * 256 + (hash & 0xff)];
        }
      }
      prev_row = curr_row;
      curr_row += argb_stride;
    }
    {
      double entropy_comp[kHistoTotal];
      double entropy[kNumEntropyIx];
      int k;
      int last_mode_to_analyze = use_palette ? kPalette : kSpatialSubGreen;
      int j;
      // Let's add one zero to the predicted histograms. The zeros are removed
      // too efficiently by the pix_diff == 0 comparison, at least one of the
      // zeros is likely to exist.
      ++histo[kHistoRedPredSubGreen * 256];
      ++histo[kHistoBluePredSubGreen * 256];
      ++histo[kHistoRedPred * 256];
      ++histo[kHistoGreenPred * 256];
      ++histo[kHistoBluePred * 256];
      ++histo[kHistoAlphaPred * 256];

      for (j = 0; j < kHistoTotal; ++j) {
        entropy_comp[j] = VP8LBitsEntropy(&histo[j * 256], 256, NULL);
      }
      entropy[kDirect] = entropy_comp[kHistoAlpha] +
          entropy_comp[kHistoRed] +
          entropy_comp[kHistoGreen] +
          entropy_comp[kHistoBlue];
      entropy[kSpatial] = entropy_comp[kHistoAlphaPred] +
          entropy_comp[kHistoRedPred] +
          entropy_comp[kHistoGreenPred] +
          entropy_comp[kHistoBluePred];
      entropy[kSubGreen] = entropy_comp[kHistoAlpha] +
          entropy_comp[kHistoRedSubGreen] +
          entropy_comp[kHistoGreen] +
          entropy_comp[kHistoBlueSubGreen];
      entropy[kSpatialSubGreen] = entropy_comp[kHistoAlphaPred] +
          entropy_comp[kHistoRedPredSubGreen] +
          entropy_comp[kHistoGreenPred] +
          entropy_comp[kHistoBluePredSubGreen];
      // Palette mode seems more efficient in a breakeven case. Bias with 1.0.
      entropy[kPalette] = entropy_comp[kHistoPalette] - 1.0;

      *min_entropy_ix = kDirect;
      for (k = kDirect + 1; k <= last_mode_to_analyze; ++k) {
        if (entropy[*min_entropy_ix] > entropy[k]) {
          *min_entropy_ix = (EntropyIx)k;
        }
      }
      *red_and_blue_always_zero = 1;
      // Let's check if the histogram of the chosen entropy mode has
      // non-zero red and blue values. If all are zero, we can later skip
      // the cross color optimization.
      {
        static const uint8_t kHistoPairs[5][2] = {
          { kHistoRed, kHistoBlue },
          { kHistoRedPred, kHistoBluePred },
          { kHistoRedSubGreen, kHistoBlueSubGreen },
          { kHistoRedPredSubGreen, kHistoBluePredSubGreen },
          { kHistoRed, kHistoBlue }
        };
        const uint32_t* const red_histo =
            &histo[256 * kHistoPairs[*min_entropy_ix][0]];
        const uint32_t* const blue_histo =
            &histo[256 * kHistoPairs[*min_entropy_ix][1]];
        for (i = 1; i < 256; ++i) {
          if ((red_histo[i] | blue_histo[i]) != 0) {
            *red_and_blue_always_zero = 0;
            break;
          }
        }
      }
    }
    WebPSafeFree(histo);
    return 1;
  } else {
    return 0;
  }
}

static int GetHistoBits(int method, int use_palette, int width, int height) {
  // Make tile size a function of encoding method (Range: 0 to 6).
  int histo_bits = (use_palette ? 9 : 7) - method;
  while (1) {
    const int huff_image_size = VP8LSubSampleSize(width, histo_bits) *
                                VP8LSubSampleSize(height, histo_bits);
    if (huff_image_size <= MAX_HUFF_IMAGE_SIZE) break;
    ++histo_bits;
  }
  return (histo_bits < MIN_HUFFMAN_BITS) ? MIN_HUFFMAN_BITS :
         (histo_bits > MAX_HUFFMAN_BITS) ? MAX_HUFFMAN_BITS : histo_bits;
}

static int GetTransformBits(int method, int histo_bits) {
  const int max_transform_bits = (method < 4) ? 6 : (method > 4) ? 4 : 5;
  return (histo_bits > max_transform_bits) ? max_transform_bits : histo_bits;
}

static int AnalyzeAndInit(VP8LEncoder* const enc) {
  const WebPPicture* const pic = enc->pic_;
  const int width = pic->width;
  const int height = pic->height;
  const int pix_cnt = width * height;
  const WebPConfig* const config = enc->config_;
  const int method = config->method;
  const int low_effort = (config->method == 0);
  // we round the block size up, so we're guaranteed to have
  // at max MAX_REFS_BLOCK_PER_IMAGE blocks used:
  int refs_block_size = (pix_cnt - 1) / MAX_REFS_BLOCK_PER_IMAGE + 1;
  assert(pic != NULL && pic->argb != NULL);

  enc->use_cross_color_ = 0;
  enc->use_predict_ = 0;
  enc->use_subtract_green_ = 0;
  enc->use_palette_ =
      AnalyzeAndCreatePalette(pic, low_effort,
                              enc->palette_, &enc->palette_size_);

  // TODO(jyrki): replace the decision to be based on an actual estimate
  // of entropy, or even spatial variance of entropy.
  enc->histo_bits_ = GetHistoBits(method, enc->use_palette_,
                                  pic->width, pic->height);
  enc->transform_bits_ = GetTransformBits(method, enc->histo_bits_);

  if (low_effort) {
    // AnalyzeEntropy is somewhat slow.
    enc->use_predict_ = !enc->use_palette_;
    enc->use_subtract_green_ = !enc->use_palette_;
    enc->use_cross_color_ = 0;
  } else {
    int red_and_blue_always_zero;
    EntropyIx min_entropy_ix;
    if (!AnalyzeEntropy(pic->argb, width, height, pic->argb_stride,
                        enc->use_palette_, &min_entropy_ix,
                        &red_and_blue_always_zero)) {
      return 0;
    }
    enc->use_palette_ = (min_entropy_ix == kPalette);
    enc->use_subtract_green_ =
        (min_entropy_ix == kSubGreen) || (min_entropy_ix == kSpatialSubGreen);
    enc->use_predict_ =
        (min_entropy_ix == kSpatial) || (min_entropy_ix == kSpatialSubGreen);
    enc->use_cross_color_ = red_and_blue_always_zero ? 0 : enc->use_predict_;
  }

  if (!VP8LHashChainInit(&enc->hash_chain_, pix_cnt)) return 0;

  // palette-friendly input typically uses less literals
  //  -> reduce block size a bit
  if (enc->use_palette_) refs_block_size /= 2;
  VP8LBackwardRefsInit(&enc->refs_[0], refs_block_size);
  VP8LBackwardRefsInit(&enc->refs_[1], refs_block_size);

  return 1;
}

// Returns false in case of memory error.
static int GetHuffBitLengthsAndCodes(
    const VP8LHistogramSet* const histogram_image,
    HuffmanTreeCode* const huffman_codes) {
  int i, k;
  int ok = 0;
  uint64_t total_length_size = 0;
  uint8_t* mem_buf = NULL;
  const int histogram_image_size = histogram_image->size;
  int max_num_symbols = 0;
  uint8_t* buf_rle = NULL;
  HuffmanTree* huff_tree = NULL;

  // Iterate over all histograms and get the aggregate number of codes used.
  for (i = 0; i < histogram_image_size; ++i) {
    const VP8LHistogram* const histo = histogram_image->histograms[i];
    HuffmanTreeCode* const codes = &huffman_codes[5 * i];
    for (k = 0; k < 5; ++k) {
      const int num_symbols =
          (k == 0) ? VP8LHistogramNumCodes(histo->palette_code_bits_) :
          (k == 4) ? NUM_DISTANCE_CODES : 256;
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
    if (mem_buf == NULL) goto End;

    codes = (uint16_t*)mem_buf;
    lengths = (uint8_t*)&codes[total_length_size];
    for (i = 0; i < 5 * histogram_image_size; ++i) {
      const int bit_length = huffman_codes[i].num_symbols;
      huffman_codes[i].codes = codes;
      huffman_codes[i].code_lengths = lengths;
      codes += bit_length;
      lengths += bit_length;
      if (max_num_symbols < bit_length) {
        max_num_symbols = bit_length;
      }
    }
  }

  buf_rle = (uint8_t*)WebPSafeMalloc(1ULL, max_num_symbols);
  huff_tree = (HuffmanTree*)WebPSafeMalloc(3ULL * max_num_symbols,
                                           sizeof(*huff_tree));
  if (buf_rle == NULL || huff_tree == NULL) goto End;

  // Create Huffman trees.
  for (i = 0; i < histogram_image_size; ++i) {
    HuffmanTreeCode* const codes = &huffman_codes[5 * i];
    VP8LHistogram* const histo = histogram_image->histograms[i];
    VP8LCreateHuffmanTree(histo->literal_, 15, buf_rle, huff_tree, codes + 0);
    VP8LCreateHuffmanTree(histo->red_, 15, buf_rle, huff_tree, codes + 1);
    VP8LCreateHuffmanTree(histo->blue_, 15, buf_rle, huff_tree, codes + 2);
    VP8LCreateHuffmanTree(histo->alpha_, 15, buf_rle, huff_tree, codes + 3);
    VP8LCreateHuffmanTree(histo->distance_, 15, buf_rle, huff_tree, codes + 4);
  }
  ok = 1;
 End:
  WebPSafeFree(huff_tree);
  WebPSafeFree(buf_rle);
  if (!ok) {
    WebPSafeFree(mem_buf);
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
  VP8LPutBits(bw, codes_to_store - 4, 4);
  for (i = 0; i < codes_to_store; ++i) {
    VP8LPutBits(bw, code_length_bitdepth[kStorageOrder[i]], 3);
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
    VP8LPutBits(bw, huffman_code->codes[ix], huffman_code->code_lengths[ix]);
    switch (ix) {
      case 16:
        VP8LPutBits(bw, extra_bits, 2);
        break;
      case 17:
        VP8LPutBits(bw, extra_bits, 3);
        break;
      case 18:
        VP8LPutBits(bw, extra_bits, 7);
        break;
    }
  }
}

// 'huff_tree' and 'tokens' are pre-alloacted buffers.
static void StoreFullHuffmanCode(VP8LBitWriter* const bw,
                                 HuffmanTree* const huff_tree,
                                 HuffmanTreeToken* const tokens,
                                 const HuffmanTreeCode* const tree) {
  uint8_t code_length_bitdepth[CODE_LENGTH_CODES] = { 0 };
  uint16_t code_length_bitdepth_symbols[CODE_LENGTH_CODES] = { 0 };
  const int max_tokens = tree->num_symbols;
  int num_tokens;
  HuffmanTreeCode huffman_code;
  huffman_code.num_symbols = CODE_LENGTH_CODES;
  huffman_code.code_lengths = code_length_bitdepth;
  huffman_code.codes = code_length_bitdepth_symbols;

  VP8LPutBits(bw, 0, 1);
  num_tokens = VP8LCreateCompressedHuffmanTree(tree, tokens, max_tokens);
  {
    uint32_t histogram[CODE_LENGTH_CODES] = { 0 };
    uint8_t buf_rle[CODE_LENGTH_CODES] = { 0 };
    int i;
    for (i = 0; i < num_tokens; ++i) {
      ++histogram[tokens[i].code];
    }

    VP8LCreateHuffmanTree(histogram, 7, buf_rle, huff_tree, &huffman_code);
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
    VP8LPutBits(bw, write_trimmed_length, 1);
    if (write_trimmed_length) {
      const int nbits = VP8LBitsLog2Ceiling(trimmed_length - 1);
      const int nbitpairs = (nbits == 0) ? 1 : (nbits + 1) / 2;
      VP8LPutBits(bw, nbitpairs - 1, 3);
      assert(trimmed_length >= 2);
      VP8LPutBits(bw, trimmed_length - 2, nbitpairs * 2);
    }
    StoreHuffmanTreeToBitMask(bw, tokens, length, &huffman_code);
  }
}

// 'huff_tree' and 'tokens' are pre-alloacted buffers.
static void StoreHuffmanCode(VP8LBitWriter* const bw,
                             HuffmanTree* const huff_tree,
                             HuffmanTreeToken* const tokens,
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
    VP8LPutBits(bw, 0x01, 4);
  } else if (count <= 2 && symbols[0] < kMaxSymbol && symbols[1] < kMaxSymbol) {
    VP8LPutBits(bw, 1, 1);  // Small tree marker to encode 1 or 2 symbols.
    VP8LPutBits(bw, count - 1, 1);
    if (symbols[0] <= 1) {
      VP8LPutBits(bw, 0, 1);  // Code bit for small (1 bit) symbol value.
      VP8LPutBits(bw, symbols[0], 1);
    } else {
      VP8LPutBits(bw, 1, 1);
      VP8LPutBits(bw, symbols[0], 8);
    }
    if (count == 2) {
      VP8LPutBits(bw, symbols[1], 8);
    }
  } else {
    StoreFullHuffmanCode(bw, huff_tree, tokens, huffman_code);
  }
}

static WEBP_INLINE void WriteHuffmanCode(VP8LBitWriter* const bw,
                             const HuffmanTreeCode* const code,
                             int code_index) {
  const int depth = code->code_lengths[code_index];
  const int symbol = code->codes[code_index];
  VP8LPutBits(bw, symbol, depth);
}

static WEBP_INLINE void WriteHuffmanCodeWithExtraBits(
    VP8LBitWriter* const bw,
    const HuffmanTreeCode* const code,
    int code_index,
    int bits,
    int n_bits) {
  const int depth = code->code_lengths[code_index];
  const int symbol = code->codes[code_index];
  VP8LPutBits(bw, (bits << depth) | symbol, depth + n_bits);
}

static WebPEncodingError StoreImageToBitMask(
    VP8LBitWriter* const bw, int width, int histo_bits,
    VP8LBackwardRefs* const refs,
    const uint16_t* histogram_symbols,
    const HuffmanTreeCode* const huffman_codes) {
  const int histo_xsize = histo_bits ? VP8LSubSampleSize(width, histo_bits) : 1;
  const int tile_mask = (histo_bits == 0) ? 0 : -(1 << histo_bits);
  // x and y trace the position in the image.
  int x = 0;
  int y = 0;
  int tile_x = x & tile_mask;
  int tile_y = y & tile_mask;
  int histogram_ix = histogram_symbols[0];
  const HuffmanTreeCode* codes = huffman_codes + 5 * histogram_ix;
  VP8LRefsCursor c = VP8LRefsCursorInit(refs);
  while (VP8LRefsCursorOk(&c)) {
    const PixOrCopy* const v = c.cur_pos;
    if ((tile_x != (x & tile_mask)) || (tile_y != (y & tile_mask))) {
      tile_x = x & tile_mask;
      tile_y = y & tile_mask;
      histogram_ix = histogram_symbols[(y >> histo_bits) * histo_xsize +
                                       (x >> histo_bits)];
      codes = huffman_codes + 5 * histogram_ix;
    }
    if (PixOrCopyIsLiteral(v)) {
      static const int order[] = { 1, 2, 0, 3 };
      int k;
      for (k = 0; k < 4; ++k) {
        const int code = PixOrCopyLiteral(v, order[k]);
        WriteHuffmanCode(bw, codes + k, code);
      }
    } else if (PixOrCopyIsCacheIdx(v)) {
      const int code = PixOrCopyCacheIdx(v);
      const int literal_ix = 256 + NUM_LENGTH_CODES + code;
      WriteHuffmanCode(bw, codes, literal_ix);
    } else {
      int bits, n_bits;
      int code;

      const int distance = PixOrCopyDistance(v);
      VP8LPrefixEncode(v->len, &code, &n_bits, &bits);
      WriteHuffmanCodeWithExtraBits(bw, codes, 256 + code, bits, n_bits);

      // Don't write the distance with the extra bits code since
      // the distance can be up to 18 bits of extra bits, and the prefix
      // 15 bits, totaling to 33, and our PutBits only supports up to 32 bits.
      // TODO(jyrki): optimize this further.
      VP8LPrefixEncode(distance, &code, &n_bits, &bits);
      WriteHuffmanCode(bw, codes + 4, code);
      VP8LPutBits(bw, bits, n_bits);
    }
    x += PixOrCopyLength(v);
    while (x >= width) {
      x -= width;
      ++y;
    }
    VP8LRefsCursorNext(&c);
  }
  return bw->error_ ? VP8_ENC_ERROR_OUT_OF_MEMORY : VP8_ENC_OK;
}

// Special case of EncodeImageInternal() for cache-bits=0, histo_bits=31
static WebPEncodingError EncodeImageNoHuffman(VP8LBitWriter* const bw,
                                              const uint32_t* const argb,
                                              VP8LHashChain* const hash_chain,
                                              VP8LBackwardRefs refs_array[2],
                                              int width, int height,
                                              int quality) {
  int i;
  int max_tokens = 0;
  WebPEncodingError err = VP8_ENC_OK;
  VP8LBackwardRefs* refs;
  HuffmanTreeToken* tokens = NULL;
  HuffmanTreeCode huffman_codes[5] = { { 0, NULL, NULL } };
  const uint16_t histogram_symbols[1] = { 0 };    // only one tree, one symbol
  int cache_bits = 0;
  VP8LHistogramSet* histogram_image = NULL;
  HuffmanTree* const huff_tree = (HuffmanTree*)WebPSafeMalloc(
        3ULL * CODE_LENGTH_CODES, sizeof(*huff_tree));
  if (huff_tree == NULL) {
    err = VP8_ENC_ERROR_OUT_OF_MEMORY;
    goto Error;
  }

  // Calculate backward references from ARGB image.
  if (VP8LHashChainFill(hash_chain, quality, argb, width, height) == 0) {
    err = VP8_ENC_ERROR_OUT_OF_MEMORY;
    goto Error;
  }
  refs = VP8LGetBackwardReferences(width, height, argb, quality, 0, &cache_bits,
                                   hash_chain, refs_array);
  if (refs == NULL) {
    err = VP8_ENC_ERROR_OUT_OF_MEMORY;
    goto Error;
  }
  histogram_image = VP8LAllocateHistogramSet(1, cache_bits);
  if (histogram_image == NULL) {
    err = VP8_ENC_ERROR_OUT_OF_MEMORY;
    goto Error;
  }

  // Build histogram image and symbols from backward references.
  VP8LHistogramStoreRefs(refs, histogram_image->histograms[0]);

  // Create Huffman bit lengths and codes for each histogram image.
  assert(histogram_image->size == 1);
  if (!GetHuffBitLengthsAndCodes(histogram_image, huffman_codes)) {
    err = VP8_ENC_ERROR_OUT_OF_MEMORY;
    goto Error;
  }

  // No color cache, no Huffman image.
  VP8LPutBits(bw, 0, 1);

  // Find maximum number of symbols for the huffman tree-set.
  for (i = 0; i < 5; ++i) {
    HuffmanTreeCode* const codes = &huffman_codes[i];
    if (max_tokens < codes->num_symbols) {
      max_tokens = codes->num_symbols;
    }
  }

  tokens = (HuffmanTreeToken*)WebPSafeMalloc(max_tokens, sizeof(*tokens));
  if (tokens == NULL) {
    err = VP8_ENC_ERROR_OUT_OF_MEMORY;
    goto Error;
  }

  // Store Huffman codes.
  for (i = 0; i < 5; ++i) {
    HuffmanTreeCode* const codes = &huffman_codes[i];
    StoreHuffmanCode(bw, huff_tree, tokens, codes);
    ClearHuffmanTreeIfOnlyOneSymbol(codes);
  }

  // Store actual literals.
  err = StoreImageToBitMask(bw, width, 0, refs, histogram_symbols,
                            huffman_codes);

 Error:
  WebPSafeFree(tokens);
  WebPSafeFree(huff_tree);
  VP8LFreeHistogramSet(histogram_image);
  WebPSafeFree(huffman_codes[0].codes);
  return err;
}

static WebPEncodingError EncodeImageInternal(VP8LBitWriter* const bw,
                                             const uint32_t* const argb,
                                             VP8LHashChain* const hash_chain,
                                             VP8LBackwardRefs refs_array[2],
                                             int width, int height, int quality,
                                             int low_effort,
                                             int use_cache, int* cache_bits,
                                             int histogram_bits,
                                             size_t init_byte_position,
                                             int* const hdr_size,
                                             int* const data_size) {
  WebPEncodingError err = VP8_ENC_OK;
  const uint32_t histogram_image_xysize =
      VP8LSubSampleSize(width, histogram_bits) *
      VP8LSubSampleSize(height, histogram_bits);
  VP8LHistogramSet* histogram_image = NULL;
  VP8LHistogramSet* tmp_histos = NULL;
  int histogram_image_size = 0;
  size_t bit_array_size = 0;
  HuffmanTree* huff_tree = NULL;
  HuffmanTreeToken* tokens = NULL;
  HuffmanTreeCode* huffman_codes = NULL;
  VP8LBackwardRefs refs;
  VP8LBackwardRefs* best_refs;
  uint16_t* const histogram_symbols =
      (uint16_t*)WebPSafeMalloc(histogram_image_xysize,
                                sizeof(*histogram_symbols));
  assert(histogram_bits >= MIN_HUFFMAN_BITS);
  assert(histogram_bits <= MAX_HUFFMAN_BITS);
  assert(hdr_size != NULL);
  assert(data_size != NULL);

  VP8LBackwardRefsInit(&refs, refs_array[0].block_size_);
  if (histogram_symbols == NULL) {
    err = VP8_ENC_ERROR_OUT_OF_MEMORY;
    goto Error;
  }

  *cache_bits = use_cache ? MAX_COLOR_CACHE_BITS : 0;
  // 'best_refs' is the reference to the best backward refs and points to one
  // of refs_array[0] or refs_array[1].
  // Calculate backward references from ARGB image.
  if (VP8LHashChainFill(hash_chain, quality, argb, width, height) == 0) {
    err = VP8_ENC_ERROR_OUT_OF_MEMORY;
    goto Error;
  }
  best_refs = VP8LGetBackwardReferences(width, height, argb, quality,
                                        low_effort, cache_bits, hash_chain,
                                        refs_array);
  if (best_refs == NULL || !VP8LBackwardRefsCopy(best_refs, &refs)) {
    err = VP8_ENC_ERROR_OUT_OF_MEMORY;
    goto Error;
  }
  histogram_image =
      VP8LAllocateHistogramSet(histogram_image_xysize, *cache_bits);
  tmp_histos = VP8LAllocateHistogramSet(2, *cache_bits);
  if (histogram_image == NULL || tmp_histos == NULL) {
    err = VP8_ENC_ERROR_OUT_OF_MEMORY;
    goto Error;
  }

  // Build histogram image and symbols from backward references.
  if (!VP8LGetHistoImageSymbols(width, height, &refs, quality, low_effort,
                                histogram_bits, *cache_bits, histogram_image,
                                tmp_histos, histogram_symbols)) {
    err = VP8_ENC_ERROR_OUT_OF_MEMORY;
    goto Error;
  }
  // Create Huffman bit lengths and codes for each histogram image.
  histogram_image_size = histogram_image->size;
  bit_array_size = 5 * histogram_image_size;
  huffman_codes = (HuffmanTreeCode*)WebPSafeCalloc(bit_array_size,
                                                   sizeof(*huffman_codes));
  // Note: some histogram_image entries may point to tmp_histos[], so the latter
  // need to outlive the following call to GetHuffBitLengthsAndCodes().
  if (huffman_codes == NULL ||
      !GetHuffBitLengthsAndCodes(histogram_image, huffman_codes)) {
    err = VP8_ENC_ERROR_OUT_OF_MEMORY;
    goto Error;
  }
  // Free combined histograms.
  VP8LFreeHistogramSet(histogram_image);
  histogram_image = NULL;

  // Free scratch histograms.
  VP8LFreeHistogramSet(tmp_histos);
  tmp_histos = NULL;

  // Color Cache parameters.
  if (*cache_bits > 0) {
    VP8LPutBits(bw, 1, 1);
    VP8LPutBits(bw, *cache_bits, 4);
  } else {
    VP8LPutBits(bw, 0, 1);
  }

  // Huffman image + meta huffman.
  {
    const int write_histogram_image = (histogram_image_size > 1);
    VP8LPutBits(bw, write_histogram_image, 1);
    if (write_histogram_image) {
      uint32_t* const histogram_argb =
          (uint32_t*)WebPSafeMalloc(histogram_image_xysize,
                                    sizeof(*histogram_argb));
      int max_index = 0;
      uint32_t i;
      if (histogram_argb == NULL) {
        err = VP8_ENC_ERROR_OUT_OF_MEMORY;
        goto Error;
      }
      for (i = 0; i < histogram_image_xysize; ++i) {
        const int symbol_index = histogram_symbols[i] & 0xffff;
        histogram_argb[i] = (symbol_index << 8);
        if (symbol_index >= max_index) {
          max_index = symbol_index + 1;
        }
      }
      histogram_image_size = max_index;

      VP8LPutBits(bw, histogram_bits - 2, 3);
      err = EncodeImageNoHuffman(bw, histogram_argb, hash_chain, refs_array,
                                 VP8LSubSampleSize(width, histogram_bits),
                                 VP8LSubSampleSize(height, histogram_bits),
                                 quality);
      WebPSafeFree(histogram_argb);
      if (err != VP8_ENC_OK) goto Error;
    }
  }

  // Store Huffman codes.
  {
    int i;
    int max_tokens = 0;
    huff_tree = (HuffmanTree*)WebPSafeMalloc(3ULL * CODE_LENGTH_CODES,
                                             sizeof(*huff_tree));
    if (huff_tree == NULL) {
      err = VP8_ENC_ERROR_OUT_OF_MEMORY;
      goto Error;
    }
    // Find maximum number of symbols for the huffman tree-set.
    for (i = 0; i < 5 * histogram_image_size; ++i) {
      HuffmanTreeCode* const codes = &huffman_codes[i];
      if (max_tokens < codes->num_symbols) {
        max_tokens = codes->num_symbols;
      }
    }
    tokens = (HuffmanTreeToken*)WebPSafeMalloc(max_tokens,
                                               sizeof(*tokens));
    if (tokens == NULL) {
      err = VP8_ENC_ERROR_OUT_OF_MEMORY;
      goto Error;
    }
    for (i = 0; i < 5 * histogram_image_size; ++i) {
      HuffmanTreeCode* const codes = &huffman_codes[i];
      StoreHuffmanCode(bw, huff_tree, tokens, codes);
      ClearHuffmanTreeIfOnlyOneSymbol(codes);
    }
  }

  *hdr_size = (int)(VP8LBitWriterNumBytes(bw) - init_byte_position);
  // Store actual literals.
  err = StoreImageToBitMask(bw, width, histogram_bits, &refs,
                            histogram_symbols, huffman_codes);
  *data_size =
        (int)(VP8LBitWriterNumBytes(bw) - init_byte_position - *hdr_size);

 Error:
  WebPSafeFree(tokens);
  WebPSafeFree(huff_tree);
  VP8LFreeHistogramSet(histogram_image);
  VP8LFreeHistogramSet(tmp_histos);
  VP8LBackwardRefsClear(&refs);
  if (huffman_codes != NULL) {
    WebPSafeFree(huffman_codes->codes);
    WebPSafeFree(huffman_codes);
  }
  WebPSafeFree(histogram_symbols);
  return err;
}

// -----------------------------------------------------------------------------
// Transforms

static void ApplySubtractGreen(VP8LEncoder* const enc, int width, int height,
                               VP8LBitWriter* const bw) {
  VP8LPutBits(bw, TRANSFORM_PRESENT, 1);
  VP8LPutBits(bw, SUBTRACT_GREEN, 2);
  VP8LSubtractGreenFromBlueAndRed(enc->argb_, width * height);
}

static WebPEncodingError ApplyPredictFilter(const VP8LEncoder* const enc,
                                            int width, int height,
                                            int quality, int low_effort,
                                            int used_subtract_green,
                                            VP8LBitWriter* const bw) {
  const int pred_bits = enc->transform_bits_;
  const int transform_width = VP8LSubSampleSize(width, pred_bits);
  const int transform_height = VP8LSubSampleSize(height, pred_bits);
  // we disable near-lossless quantization if palette is used.
  const int near_lossless_strength = enc->use_palette_ ? 100
                                   : enc->config_->near_lossless;

  VP8LResidualImage(width, height, pred_bits, low_effort, enc->argb_,
                    enc->argb_scratch_, enc->transform_data_,
                    near_lossless_strength, enc->config_->exact,
                    used_subtract_green);
  VP8LPutBits(bw, TRANSFORM_PRESENT, 1);
  VP8LPutBits(bw, PREDICTOR_TRANSFORM, 2);
  assert(pred_bits >= 2);
  VP8LPutBits(bw, pred_bits - 2, 3);
  return EncodeImageNoHuffman(bw, enc->transform_data_,
                              (VP8LHashChain*)&enc->hash_chain_,
                              (VP8LBackwardRefs*)enc->refs_,  // cast const away
                              transform_width, transform_height,
                              quality);
}

static WebPEncodingError ApplyCrossColorFilter(const VP8LEncoder* const enc,
                                               int width, int height,
                                               int quality,
                                               VP8LBitWriter* const bw) {
  const int ccolor_transform_bits = enc->transform_bits_;
  const int transform_width = VP8LSubSampleSize(width, ccolor_transform_bits);
  const int transform_height = VP8LSubSampleSize(height, ccolor_transform_bits);

  VP8LColorSpaceTransform(width, height, ccolor_transform_bits, quality,
                          enc->argb_, enc->transform_data_);
  VP8LPutBits(bw, TRANSFORM_PRESENT, 1);
  VP8LPutBits(bw, CROSS_COLOR_TRANSFORM, 2);
  assert(ccolor_transform_bits >= 2);
  VP8LPutBits(bw, ccolor_transform_bits - 2, 3);
  return EncodeImageNoHuffman(bw, enc->transform_data_,
                              (VP8LHashChain*)&enc->hash_chain_,
                              (VP8LBackwardRefs*)enc->refs_,  // cast const away
                              transform_width, transform_height,
                              quality);
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

  VP8LPutBits(bw, width, VP8L_IMAGE_SIZE_BITS);
  VP8LPutBits(bw, height, VP8L_IMAGE_SIZE_BITS);
  return !bw->error_;
}

static int WriteRealAlphaAndVersion(VP8LBitWriter* const bw, int has_alpha) {
  VP8LPutBits(bw, has_alpha, 1);
  VP8LPutBits(bw, VP8L_VERSION, VP8L_VERSION_BITS);
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

static void ClearTransformBuffer(VP8LEncoder* const enc) {
  WebPSafeFree(enc->transform_mem_);
  enc->transform_mem_ = NULL;
  enc->transform_mem_size_ = 0;
}

// Allocates the memory for argb (W x H) buffer, 2 rows of context for
// prediction and transform data.
// Flags influencing the memory allocated:
//  enc->transform_bits_
//  enc->use_predict_, enc->use_cross_color_
static WebPEncodingError AllocateTransformBuffer(VP8LEncoder* const enc,
                                                 int width, int height) {
  WebPEncodingError err = VP8_ENC_OK;
  const uint64_t image_size = width * height;
  // VP8LResidualImage needs room for 2 scanlines of uint32 pixels with an extra
  // pixel in each, plus 2 regular scanlines of bytes.
  // TODO(skal): Clean up by using arithmetic in bytes instead of words.
  const uint64_t argb_scratch_size =
      enc->use_predict_
          ? (width + 1) * 2 +
            (width * 2 + sizeof(uint32_t) - 1) / sizeof(uint32_t)
          : 0;
  const uint64_t transform_data_size =
      (enc->use_predict_ || enc->use_cross_color_)
          ? VP8LSubSampleSize(width, enc->transform_bits_) *
                VP8LSubSampleSize(height, enc->transform_bits_)
          : 0;
  const uint64_t max_alignment_in_words =
      (WEBP_ALIGN_CST + sizeof(uint32_t) - 1) / sizeof(uint32_t);
  const uint64_t mem_size =
      image_size + max_alignment_in_words +
      argb_scratch_size + max_alignment_in_words +
      transform_data_size;
  uint32_t* mem = enc->transform_mem_;
  if (mem == NULL || mem_size > enc->transform_mem_size_) {
    ClearTransformBuffer(enc);
    mem = (uint32_t*)WebPSafeMalloc(mem_size, sizeof(*mem));
    if (mem == NULL) {
      err = VP8_ENC_ERROR_OUT_OF_MEMORY;
      goto Error;
    }
    enc->transform_mem_ = mem;
    enc->transform_mem_size_ = (size_t)mem_size;
  }
  enc->argb_ = mem;
  mem = (uint32_t*)WEBP_ALIGN(mem + image_size);
  enc->argb_scratch_ = mem;
  mem = (uint32_t*)WEBP_ALIGN(mem + argb_scratch_size);
  enc->transform_data_ = mem;

  enc->current_width_ = width;
 Error:
  return err;
}

static WebPEncodingError MakeInputImageCopy(VP8LEncoder* const enc) {
  WebPEncodingError err = VP8_ENC_OK;
  const WebPPicture* const picture = enc->pic_;
  const int width = picture->width;
  const int height = picture->height;
  int y;
  err = AllocateTransformBuffer(enc, width, height);
  if (err != VP8_ENC_OK) return err;
  for (y = 0; y < height; ++y) {
    memcpy(enc->argb_ + y * width,
           picture->argb + y * picture->argb_stride,
           width * sizeof(*enc->argb_));
  }
  assert(enc->current_width_ == width);
  return VP8_ENC_OK;
}

// -----------------------------------------------------------------------------

static int SearchColor(const uint32_t sorted[], uint32_t color, int hi) {
  int low = 0;
  if (sorted[low] == color) return low;  // loop invariant: sorted[low] != color
  while (1) {
    const int mid = (low + hi) >> 1;
    if (sorted[mid] == color) {
      return mid;
    } else if (sorted[mid] < color) {
      low = mid;
    } else {
      hi = mid;
    }
  }
}

// Sort palette in increasing order and prepare an inverse mapping array.
static void PrepareMapToPalette(const uint32_t palette[], int num_colors,
                                uint32_t sorted[], int idx_map[]) {
  int i;
  memcpy(sorted, palette, num_colors * sizeof(*sorted));
  qsort(sorted, num_colors, sizeof(*sorted), PaletteCompareColorsForQsort);
  for (i = 0; i < num_colors; ++i) {
    idx_map[SearchColor(sorted, palette[i], num_colors)] = i;
  }
}

static void MapToPalette(const uint32_t sorted_palette[], int num_colors,
                         uint32_t* const last_pix, int* const last_idx,
                         const int idx_map[],
                         const uint32_t* src, uint8_t* dst, int width) {
  int x;
  int prev_idx = *last_idx;
  uint32_t prev_pix = *last_pix;
  for (x = 0; x < width; ++x) {
    const uint32_t pix = src[x];
    if (pix != prev_pix) {
      prev_idx = idx_map[SearchColor(sorted_palette, pix, num_colors)];
      prev_pix = pix;
    }
    dst[x] = prev_idx;
  }
  *last_idx = prev_idx;
  *last_pix = prev_pix;
}

// Remap argb values in src[] to packed palettes entries in dst[]
// using 'row' as a temporary buffer of size 'width'.
// We assume that all src[] values have a corresponding entry in the palette.
// Note: src[] can be the same as dst[]
static WebPEncodingError ApplyPalette(const uint32_t* src, uint32_t src_stride,
                                      uint32_t* dst, uint32_t dst_stride,
                                      const uint32_t* palette, int palette_size,
                                      int width, int height, int xbits) {
  // TODO(skal): this tmp buffer is not needed if VP8LBundleColorMap() can be
  // made to work in-place.
  uint8_t* const tmp_row = (uint8_t*)WebPSafeMalloc(width, sizeof(*tmp_row));
  int i, x, y;
  int use_LUT = 1;

  if (tmp_row == NULL) return VP8_ENC_ERROR_OUT_OF_MEMORY;
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
        tmp_row[x] = inv_palette[color];
      }
      VP8LBundleColorMap(tmp_row, width, xbits, dst);
      src += src_stride;
      dst += dst_stride;
    }
  } else {
    // Use 1 pixel cache for ARGB pixels.
    uint32_t last_pix;
    int last_idx;
    uint32_t sorted[MAX_PALETTE_SIZE];
    int idx_map[MAX_PALETTE_SIZE];
    PrepareMapToPalette(palette, palette_size, sorted, idx_map);
    last_pix = palette[0];
    last_idx = 0;
    for (y = 0; y < height; ++y) {
      MapToPalette(sorted, palette_size, &last_pix, &last_idx,
                   idx_map, src, tmp_row, width);
      VP8LBundleColorMap(tmp_row, width, xbits, dst);
      src += src_stride;
      dst += dst_stride;
    }
  }
  WebPSafeFree(tmp_row);
  return VP8_ENC_OK;
}

// Note: Expects "enc->palette_" to be set properly.
static WebPEncodingError MapImageFromPalette(VP8LEncoder* const enc,
                                             int in_place) {
  WebPEncodingError err = VP8_ENC_OK;
  const WebPPicture* const pic = enc->pic_;
  const int width = pic->width;
  const int height = pic->height;
  const uint32_t* const palette = enc->palette_;
  const uint32_t* src = in_place ? enc->argb_ : pic->argb;
  const int src_stride = in_place ? enc->current_width_ : pic->argb_stride;
  const int palette_size = enc->palette_size_;
  int xbits;

  // Replace each input pixel by corresponding palette index.
  // This is done line by line.
  if (palette_size <= 4) {
    xbits = (palette_size <= 2) ? 3 : 2;
  } else {
    xbits = (palette_size <= 16) ? 1 : 0;
  }

  err = AllocateTransformBuffer(enc, VP8LSubSampleSize(width, xbits), height);
  if (err != VP8_ENC_OK) return err;

  err = ApplyPalette(src, src_stride,
                     enc->argb_, enc->current_width_,
                     palette, palette_size, width, height, xbits);
  return err;
}

// Save palette_[] to bitstream.
static WebPEncodingError EncodePalette(VP8LBitWriter* const bw,
                                       VP8LEncoder* const enc) {
  int i;
  uint32_t tmp_palette[MAX_PALETTE_SIZE];
  const int palette_size = enc->palette_size_;
  const uint32_t* const palette = enc->palette_;
  VP8LPutBits(bw, TRANSFORM_PRESENT, 1);
  VP8LPutBits(bw, COLOR_INDEXING_TRANSFORM, 2);
  assert(palette_size >= 1 && palette_size <= MAX_PALETTE_SIZE);
  VP8LPutBits(bw, palette_size - 1, 8);
  for (i = palette_size - 1; i >= 1; --i) {
    tmp_palette[i] = VP8LSubPixels(palette[i], palette[i - 1]);
  }
  tmp_palette[0] = palette[0];
  return EncodeImageNoHuffman(bw, tmp_palette, &enc->hash_chain_, enc->refs_,
                              palette_size, 1, 20 /* quality */);
}

#ifdef WEBP_EXPERIMENTAL_FEATURES

static WebPEncodingError EncodeDeltaPalettePredictorImage(
    VP8LBitWriter* const bw, VP8LEncoder* const enc, int quality) {
  const WebPPicture* const pic = enc->pic_;
  const int width = pic->width;
  const int height = pic->height;

  const int pred_bits = 5;
  const int transform_width = VP8LSubSampleSize(width, pred_bits);
  const int transform_height = VP8LSubSampleSize(height, pred_bits);
  const int pred = 7;   // default is Predictor7 (Top/Left Average)
  const int tiles_per_row = VP8LSubSampleSize(width, pred_bits);
  const int tiles_per_col = VP8LSubSampleSize(height, pred_bits);
  uint32_t* predictors;
  int tile_x, tile_y;
  WebPEncodingError err = VP8_ENC_OK;

  predictors = (uint32_t*)WebPSafeMalloc(tiles_per_col * tiles_per_row,
                                         sizeof(*predictors));
  if (predictors == NULL) return VP8_ENC_ERROR_OUT_OF_MEMORY;

  for (tile_y = 0; tile_y < tiles_per_col; ++tile_y) {
    for (tile_x = 0; tile_x < tiles_per_row; ++tile_x) {
      predictors[tile_y * tiles_per_row + tile_x] = 0xff000000u | (pred << 8);
    }
  }

  VP8LPutBits(bw, TRANSFORM_PRESENT, 1);
  VP8LPutBits(bw, PREDICTOR_TRANSFORM, 2);
  VP8LPutBits(bw, pred_bits - 2, 3);
  err = EncodeImageNoHuffman(bw, predictors, &enc->hash_chain_,
                             (VP8LBackwardRefs*)enc->refs_,  // cast const away
                             transform_width, transform_height,
                             quality);
  WebPSafeFree(predictors);
  return err;
}

#endif // WEBP_EXPERIMENTAL_FEATURES

// -----------------------------------------------------------------------------
// VP8LEncoder

static VP8LEncoder* VP8LEncoderNew(const WebPConfig* const config,
                                   const WebPPicture* const picture) {
  VP8LEncoder* const enc = (VP8LEncoder*)WebPSafeCalloc(1ULL, sizeof(*enc));
  if (enc == NULL) {
    WebPEncodingSetError(picture, VP8_ENC_ERROR_OUT_OF_MEMORY);
    return NULL;
  }
  enc->config_ = config;
  enc->pic_ = picture;

  VP8LEncDspInit();

  return enc;
}

static void VP8LEncoderDelete(VP8LEncoder* enc) {
  if (enc != NULL) {
    VP8LHashChainClear(&enc->hash_chain_);
    VP8LBackwardRefsClear(&enc->refs_[0]);
    VP8LBackwardRefsClear(&enc->refs_[1]);
    ClearTransformBuffer(enc);
    WebPSafeFree(enc);
  }
}

// -----------------------------------------------------------------------------
// Main call

WebPEncodingError VP8LEncodeStream(const WebPConfig* const config,
                                   const WebPPicture* const picture,
                                   VP8LBitWriter* const bw, int use_cache) {
  WebPEncodingError err = VP8_ENC_OK;
  const int quality = (int)config->quality;
  const int low_effort = (config->method == 0);
  const int width = picture->width;
  const int height = picture->height;
  VP8LEncoder* const enc = VP8LEncoderNew(config, picture);
  const size_t byte_position = VP8LBitWriterNumBytes(bw);
  int use_near_lossless = 0;
  int hdr_size = 0;
  int data_size = 0;
  int use_delta_palettization = 0;

  if (enc == NULL) {
    err = VP8_ENC_ERROR_OUT_OF_MEMORY;
    goto Error;
  }

  // ---------------------------------------------------------------------------
  // Analyze image (entropy, num_palettes etc)

  if (!AnalyzeAndInit(enc)) {
    err = VP8_ENC_ERROR_OUT_OF_MEMORY;
    goto Error;
  }

  // Apply near-lossless preprocessing.
  use_near_lossless =
      (config->near_lossless < 100) && !enc->use_palette_ && !enc->use_predict_;
  if (use_near_lossless) {
    if (!VP8ApplyNearLossless(width, height, picture->argb,
                              config->near_lossless)) {
      err = VP8_ENC_ERROR_OUT_OF_MEMORY;
      goto Error;
    }
  }

#ifdef WEBP_EXPERIMENTAL_FEATURES
  if (config->delta_palettization) {
    enc->use_predict_ = 1;
    enc->use_cross_color_ = 0;
    enc->use_subtract_green_ = 0;
    enc->use_palette_ = 1;
    err = MakeInputImageCopy(enc);
    if (err != VP8_ENC_OK) goto Error;
    err = WebPSearchOptimalDeltaPalette(enc);
    if (err != VP8_ENC_OK) goto Error;
    if (enc->use_palette_) {
      err = AllocateTransformBuffer(enc, width, height);
      if (err != VP8_ENC_OK) goto Error;
      err = EncodeDeltaPalettePredictorImage(bw, enc, quality);
      if (err != VP8_ENC_OK) goto Error;
      use_delta_palettization = 1;
    }
  }
#endif  // WEBP_EXPERIMENTAL_FEATURES

  // Encode palette
  if (enc->use_palette_) {
    err = EncodePalette(bw, enc);
    if (err != VP8_ENC_OK) goto Error;
    err = MapImageFromPalette(enc, use_delta_palettization);
    if (err != VP8_ENC_OK) goto Error;
  }
  if (!use_delta_palettization) {
    // In case image is not packed.
    if (enc->argb_ == NULL) {
      err = MakeInputImageCopy(enc);
      if (err != VP8_ENC_OK) goto Error;
    }

    // -------------------------------------------------------------------------
    // Apply transforms and write transform data.

    if (enc->use_subtract_green_) {
      ApplySubtractGreen(enc, enc->current_width_, height, bw);
    }

    if (enc->use_predict_) {
      err = ApplyPredictFilter(enc, enc->current_width_, height, quality,
                               low_effort, enc->use_subtract_green_, bw);
      if (err != VP8_ENC_OK) goto Error;
    }

    if (enc->use_cross_color_) {
      err = ApplyCrossColorFilter(enc, enc->current_width_,
                                  height, quality, bw);
      if (err != VP8_ENC_OK) goto Error;
    }
  }

  VP8LPutBits(bw, !TRANSFORM_PRESENT, 1);  // No more transforms.

  // ---------------------------------------------------------------------------
  // Encode and write the transformed image.
  err = EncodeImageInternal(bw, enc->argb_, &enc->hash_chain_, enc->refs_,
                            enc->current_width_, height, quality, low_effort,
                            use_cache, &enc->cache_bits_, enc->histo_bits_,
                            byte_position, &hdr_size, &data_size);
  if (err != VP8_ENC_OK) goto Error;

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
    stats->lossless_hdr_size = hdr_size;
    stats->lossless_data_size = data_size;
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
  int initial_size;
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
  // Initialize BitWriter with size corresponding to 16 bpp to photo images and
  // 8 bpp for graphical images.
  initial_size = (config->image_hint == WEBP_HINT_GRAPH) ?
      width * height : width * height * 2;
  if (!VP8LBitWriterInit(&bw, initial_size)) {
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
  err = VP8LEncodeStream(config, picture, &bw, 1 /*use_cache*/);
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
  VP8LBitWriterWipeOut(&bw);
  if (err != VP8_ENC_OK) {
    WebPEncodingSetError(picture, err);
    return 0;
  }
  return 1;
}

//------------------------------------------------------------------------------
