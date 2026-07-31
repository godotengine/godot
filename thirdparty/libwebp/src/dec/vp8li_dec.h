// Copyright 2012 Google Inc. All Rights Reserved.
//
// Use of this source code is governed by a BSD-style license
// that can be found in the COPYING file in the root of the source
// tree. An additional intellectual property rights grant can be found
// in the file PATENTS. All contributing project authors may
// be found in the AUTHORS file in the root of the source tree.
// -----------------------------------------------------------------------------
//
// Lossless decoder: internal header.
//
// Author: Skal (pascal.massimino@gmail.com)
//         Vikas Arora(vikaas.arora@gmail.com)

#ifndef WEBP_DEC_VP8LI_DEC_H_
#define WEBP_DEC_VP8LI_DEC_H_

#include <string.h>     // for memcpy()

#include "src/dec/vp8_dec.h"
#include "src/dec/webpi_dec.h"
#include "src/utils/bit_reader_utils.h"
#include "src/utils/color_cache_utils.h"
#include "src/utils/huffman_utils.h"
#include "src/utils/rescaler_utils.h"
#include "src/webp/decode.h"
#include "src/webp/format_constants.h"
#include "src/webp/types.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
  READ_DATA = 0,
  READ_HDR = 1,
  READ_DIM = 2
} VP8LDecodeState;

typedef struct VP8LTransform VP8LTransform;
struct VP8LTransform {
  VP8LImageTransformType type;   // transform type.
  int                    bits;   // subsampling bits defining transform window.
  int                    xsize;  // transform window X index.
  int                    ysize;  // transform window Y index.
  uint32_t*              data;   // transform data.
};

typedef struct {
  int             color_cache_size;
  VP8LColorCache  color_cache;
  VP8LColorCache  saved_color_cache;  // for incremental

  int             huffman_mask;
  int             huffman_subsample_bits;
  int             huffman_xsize;
  uint32_t*       huffman_image;
  int             num_htree_groups;
  HTreeGroup*     htree_groups;
  HuffmanTables   huffman_tables;
} VP8LMetadata;

typedef struct VP8LDecoder VP8LDecoder;
struct VP8LDecoder {
  VP8StatusCode    status;
  VP8LDecodeState  state;
  VP8Io*           io;

  const WebPDecBuffer* output;    // shortcut to io->opaque->output

  uint32_t*        pixels;        // Internal data: either uint8_t* for alpha
                                  // or uint32_t* for BGRA.
  uint32_t*        argb_cache;    // Scratch buffer for temporary BGRA storage.

  VP8LBitReader    br;
  int              incremental;   // if true, incremental decoding is expected
  VP8LBitReader    saved_br;      // note: could be local variables too
  int              saved_last_pixel;

  int              width;
  int              height;
  int              last_row;      // last input row decoded so far.
  int              last_pixel;    // last pixel decoded so far. However, it may
                                  // not be transformed, scaled and
                                  // color-converted yet.
  int              last_out_row;  // last row output so far.

  VP8LMetadata     hdr;

  int              next_transform;
  VP8LTransform    transforms[NUM_TRANSFORMS];
  // or'd bitset storing the transforms types.
  uint32_t         transforms_seen;

  uint8_t*         rescaler_memory;  // Working memory for rescaling work.
  WebPRescaler*    rescaler;         // Common rescaler for all channels.
};

//------------------------------------------------------------------------------
// internal functions. Not public.

struct ALPHDecoder;  // Defined in dec/alphai.h.

// in vp8l.c

// Decodes image header for alpha data stored using lossless compression.
// Returns false in case of error.
WEBP_NODISCARD int VP8LDecodeAlphaHeader(struct ALPHDecoder* const alph_dec,
                                         const uint8_t* const data,
                                         size_t data_size);

// Decodes *at least* 'last_row' rows of alpha. If some of the initial rows are
// already decoded in previous call(s), it will resume decoding from where it
// was paused.
// Returns false in case of bitstream error.
WEBP_NODISCARD int VP8LDecodeAlphaImageStream(
    struct ALPHDecoder* const alph_dec, int last_row);

// Allocates and initialize a new lossless decoder instance.
WEBP_NODISCARD VP8LDecoder* VP8LNew(void);

// Decodes the image header. Returns false in case of error.
WEBP_NODISCARD int VP8LDecodeHeader(VP8LDecoder* const dec, VP8Io* const io);

// Decodes an image. It's required to decode the lossless header before calling
// this function. Returns false in case of error, with updated dec->status.
WEBP_NODISCARD int VP8LDecodeImage(VP8LDecoder* const dec);

// Clears and deallocate a lossless decoder instance.
void VP8LDelete(VP8LDecoder* const dec);

// Helper function for reading the different Huffman codes and storing them in
// 'huffman_tables' and 'htree_groups'.
// If mapping is NULL 'num_htree_groups_max' must equal 'num_htree_groups'.
// If it is not NULL, it maps 'num_htree_groups_max' indices to the
// 'num_htree_groups' groups. If 'num_htree_groups_max' > 'num_htree_groups',
// some of those indices map to -1. This is used for non-balanced codes to
// limit memory usage.
WEBP_NODISCARD int ReadHuffmanCodesHelper(
    int color_cache_bits, int num_htree_groups, int num_htree_groups_max,
    const int* const mapping, VP8LDecoder* const dec,
    HuffmanTables* const huffman_tables, HTreeGroup** const htree_groups);

//------------------------------------------------------------------------------

#ifdef __cplusplus
}    // extern "C"
#endif

#endif  // WEBP_DEC_VP8LI_DEC_H_
