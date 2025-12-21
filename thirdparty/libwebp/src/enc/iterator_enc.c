// Copyright 2011 Google Inc. All Rights Reserved.
//
// Use of this source code is governed by a BSD-style license
// that can be found in the COPYING file in the root of the source
// tree. An additional intellectual property rights grant can be found
// in the file PATENTS. All contributing project authors may
// be found in the AUTHORS file in the root of the source tree.
// -----------------------------------------------------------------------------
//
// VP8Iterator: block iterator
//
// Author: Skal (pascal.massimino@gmail.com)

#include <string.h>

#include "src/dsp/cpu.h"
#include "src/dsp/dsp.h"
#include "src/enc/vp8i_enc.h"
#include "src/utils/utils.h"
#include "src/webp/types.h"

//------------------------------------------------------------------------------
// VP8Iterator
//------------------------------------------------------------------------------

static void InitLeft(VP8EncIterator* const it) {
  it->y_left[-1] = it->u_left[-1] = it->v_left[-1] =
      (it->y > 0) ? 129 : 127;
  memset(it->y_left, 129, 16);
  memset(it->u_left, 129, 8);
  memset(it->v_left, 129, 8);
  it->left_nz[8] = 0;
  if (it->top_derr != NULL) {
    memset(&it->left_derr, 0, sizeof(it->left_derr));
  }
}

static void InitTop(VP8EncIterator* const it) {
  const VP8Encoder* const enc = it->enc;
  const size_t top_size = enc->mb_w * 16;
  memset(enc->y_top, 127, 2 * top_size);
  memset(enc->nz, 0, enc->mb_w * sizeof(*enc->nz));
  if (enc->top_derr != NULL) {
    memset(enc->top_derr, 0, enc->mb_w * sizeof(*enc->top_derr));
  }
}

void VP8IteratorSetRow(VP8EncIterator* const it, int y) {
  VP8Encoder* const enc = it->enc;
  it->x = 0;
  it->y = y;
  it->bw = &enc->parts[y & (enc->num_parts - 1)];
  it->preds = enc->preds + y * 4 * enc->preds_w;
  it->nz = enc->nz;
  it->mb = enc->mb_info + y * enc->mb_w;
  it->y_top = enc->y_top;
  it->uv_top = enc->uv_top;
  InitLeft(it);
}

// restart a scan
static void VP8IteratorReset(VP8EncIterator* const it) {
  VP8Encoder* const enc = it->enc;
  VP8IteratorSetRow(it, 0);
  VP8IteratorSetCountDown(it, enc->mb_w * enc->mb_h);  // default
  InitTop(it);
  memset(it->bit_count, 0, sizeof(it->bit_count));
  it->do_trellis = 0;
}

void VP8IteratorSetCountDown(VP8EncIterator* const it, int count_down) {
  it->count_down = it->count_down0 = count_down;
}

int VP8IteratorIsDone(const VP8EncIterator* const it) {
  return (it->count_down <= 0);
}

void VP8IteratorInit(VP8Encoder* const enc, VP8EncIterator* const it) {
  it->enc = enc;
  it->yuv_in   = (uint8_t*)WEBP_ALIGN(it->yuv_mem);
  it->yuv_out  = it->yuv_in + YUV_SIZE_ENC;
  it->yuv_out2 = it->yuv_out + YUV_SIZE_ENC;
  it->yuv_p    = it->yuv_out2 + YUV_SIZE_ENC;
  it->lf_stats = enc->lf_stats;
  it->percent0 = enc->percent;
  it->y_left = (uint8_t*)WEBP_ALIGN(it->yuv_left_mem + 1);
  it->u_left = it->y_left + 16 + 16;
  it->v_left = it->u_left + 16;
  it->top_derr = enc->top_derr;
  VP8IteratorReset(it);
}

int VP8IteratorProgress(const VP8EncIterator* const it, int delta) {
  VP8Encoder* const enc = it->enc;
  if (delta && enc->pic->progress_hook != NULL) {
    const int done = it->count_down0 - it->count_down;
    const int percent = (it->count_down0 <= 0)
                      ? it->percent0
                      : it->percent0 + delta * done / it->count_down0;
    return WebPReportProgress(enc->pic, percent, &enc->percent);
  }
  return 1;
}

//------------------------------------------------------------------------------
// Import the source samples into the cache. Takes care of replicating
// boundary pixels if necessary.

static WEBP_INLINE int MinSize(int a, int b) { return (a < b) ? a : b; }

static void ImportBlock(const uint8_t* src, int src_stride,
                        uint8_t* dst, int w, int h, int size) {
  int i;
  for (i = 0; i < h; ++i) {
    memcpy(dst, src, w);
    if (w < size) {
      memset(dst + w, dst[w - 1], size - w);
    }
    dst += BPS;
    src += src_stride;
  }
  for (i = h; i < size; ++i) {
    memcpy(dst, dst - BPS, size);
    dst += BPS;
  }
}

static void ImportLine(const uint8_t* src, int src_stride,
                       uint8_t* dst, int len, int total_len) {
  int i;
  for (i = 0; i < len; ++i, src += src_stride) dst[i] = *src;
  for (; i < total_len; ++i) dst[i] = dst[len - 1];
}

void VP8IteratorImport(VP8EncIterator* const it, uint8_t* const tmp_32) {
  const VP8Encoder* const enc = it->enc;
  const int x = it->x, y = it->y;
  const WebPPicture* const pic = enc->pic;
  const uint8_t* const ysrc = pic->y + (y * pic->y_stride  + x) * 16;
  const uint8_t* const usrc = pic->u + (y * pic->uv_stride + x) * 8;
  const uint8_t* const vsrc = pic->v + (y * pic->uv_stride + x) * 8;
  const int w = MinSize(pic->width - x * 16, 16);
  const int h = MinSize(pic->height - y * 16, 16);
  const int uv_w = (w + 1) >> 1;
  const int uv_h = (h + 1) >> 1;

  ImportBlock(ysrc, pic->y_stride,  it->yuv_in + Y_OFF_ENC, w, h, 16);
  ImportBlock(usrc, pic->uv_stride, it->yuv_in + U_OFF_ENC, uv_w, uv_h, 8);
  ImportBlock(vsrc, pic->uv_stride, it->yuv_in + V_OFF_ENC, uv_w, uv_h, 8);

  if (tmp_32 == NULL) return;

  // Import source (uncompressed) samples into boundary.
  if (x == 0) {
    InitLeft(it);
  } else {
    if (y == 0) {
      it->y_left[-1] = it->u_left[-1] = it->v_left[-1] = 127;
    } else {
      it->y_left[-1] = ysrc[- 1 - pic->y_stride];
      it->u_left[-1] = usrc[- 1 - pic->uv_stride];
      it->v_left[-1] = vsrc[- 1 - pic->uv_stride];
    }
    ImportLine(ysrc - 1, pic->y_stride,  it->y_left, h,   16);
    ImportLine(usrc - 1, pic->uv_stride, it->u_left, uv_h, 8);
    ImportLine(vsrc - 1, pic->uv_stride, it->v_left, uv_h, 8);
  }

  it->y_top  = tmp_32 + 0;
  it->uv_top = tmp_32 + 16;
  if (y == 0) {
    memset(tmp_32, 127, 32 * sizeof(*tmp_32));
  } else {
    ImportLine(ysrc - pic->y_stride,  1, tmp_32,          w,   16);
    ImportLine(usrc - pic->uv_stride, 1, tmp_32 + 16,     uv_w, 8);
    ImportLine(vsrc - pic->uv_stride, 1, tmp_32 + 16 + 8, uv_w, 8);
  }
}

//------------------------------------------------------------------------------
// Copy back the compressed samples into user space if requested.

static void ExportBlock(const uint8_t* src, uint8_t* dst, int dst_stride,
                        int w, int h) {
  while (h-- > 0) {
    memcpy(dst, src, w);
    dst += dst_stride;
    src += BPS;
  }
}

void VP8IteratorExport(const VP8EncIterator* const it) {
  const VP8Encoder* const enc = it->enc;
  if (enc->config->show_compressed) {
    const int x = it->x, y = it->y;
    const uint8_t* const ysrc = it->yuv_out + Y_OFF_ENC;
    const uint8_t* const usrc = it->yuv_out + U_OFF_ENC;
    const uint8_t* const vsrc = it->yuv_out + V_OFF_ENC;
    const WebPPicture* const pic = enc->pic;
    uint8_t* const ydst = pic->y + (y * pic->y_stride + x) * 16;
    uint8_t* const udst = pic->u + (y * pic->uv_stride + x) * 8;
    uint8_t* const vdst = pic->v + (y * pic->uv_stride + x) * 8;
    int w = (pic->width - x * 16);
    int h = (pic->height - y * 16);

    if (w > 16) w = 16;
    if (h > 16) h = 16;

    // Luma plane
    ExportBlock(ysrc, ydst, pic->y_stride, w, h);

    {   // U/V planes
      const int uv_w = (w + 1) >> 1;
      const int uv_h = (h + 1) >> 1;
      ExportBlock(usrc, udst, pic->uv_stride, uv_w, uv_h);
      ExportBlock(vsrc, vdst, pic->uv_stride, uv_w, uv_h);
    }
  }
}

//------------------------------------------------------------------------------
// Non-zero contexts setup/teardown

// Nz bits:
//  0  1  2  3  Y
//  4  5  6  7
//  8  9 10 11
// 12 13 14 15
// 16 17        U
// 18 19
// 20 21        V
// 22 23
// 24           DC-intra16

// Convert packed context to byte array
#define BIT(nz, n) (!!((nz) & (1 << (n))))

void VP8IteratorNzToBytes(VP8EncIterator* const it) {
  const int tnz = it->nz[0], lnz = it->nz[-1];
  int* const top_nz = it->top_nz;
  int* const left_nz = it->left_nz;

  // Top-Y
  top_nz[0] = BIT(tnz, 12);
  top_nz[1] = BIT(tnz, 13);
  top_nz[2] = BIT(tnz, 14);
  top_nz[3] = BIT(tnz, 15);
  // Top-U
  top_nz[4] = BIT(tnz, 18);
  top_nz[5] = BIT(tnz, 19);
  // Top-V
  top_nz[6] = BIT(tnz, 22);
  top_nz[7] = BIT(tnz, 23);
  // DC
  top_nz[8] = BIT(tnz, 24);

  // left-Y
  left_nz[0] = BIT(lnz,  3);
  left_nz[1] = BIT(lnz,  7);
  left_nz[2] = BIT(lnz, 11);
  left_nz[3] = BIT(lnz, 15);
  // left-U
  left_nz[4] = BIT(lnz, 17);
  left_nz[5] = BIT(lnz, 19);
  // left-V
  left_nz[6] = BIT(lnz, 21);
  left_nz[7] = BIT(lnz, 23);
  // left-DC is special, iterated separately
}

void VP8IteratorBytesToNz(VP8EncIterator* const it) {
  uint32_t nz = 0;
  const int* const top_nz = it->top_nz;
  const int* const left_nz = it->left_nz;
  // top
  nz |= (top_nz[0] << 12) | (top_nz[1] << 13);
  nz |= (top_nz[2] << 14) | (top_nz[3] << 15);
  nz |= (top_nz[4] << 18) | (top_nz[5] << 19);
  nz |= (top_nz[6] << 22) | (top_nz[7] << 23);
  nz |= (top_nz[8] << 24);  // we propagate the _top_ bit, esp. for intra4
  // left
  nz |= (left_nz[0] << 3) | (left_nz[1] << 7);
  nz |= (left_nz[2] << 11);
  nz |= (left_nz[4] << 17) | (left_nz[6] << 21);

  *it->nz = nz;
}

#undef BIT

//------------------------------------------------------------------------------
// Advance to the next position, doing the bookkeeping.

void VP8IteratorSaveBoundary(VP8EncIterator* const it) {
  VP8Encoder* const enc = it->enc;
  const int x = it->x, y = it->y;
  const uint8_t* const ysrc = it->yuv_out + Y_OFF_ENC;
  const uint8_t* const uvsrc = it->yuv_out + U_OFF_ENC;
  if (x < enc->mb_w - 1) {   // left
    int i;
    for (i = 0; i < 16; ++i) {
      it->y_left[i] = ysrc[15 + i * BPS];
    }
    for (i = 0; i < 8; ++i) {
      it->u_left[i] = uvsrc[7 + i * BPS];
      it->v_left[i] = uvsrc[15 + i * BPS];
    }
    // top-left (before 'top'!)
    it->y_left[-1] = it->y_top[15];
    it->u_left[-1] = it->uv_top[0 + 7];
    it->v_left[-1] = it->uv_top[8 + 7];
  }
  if (y < enc->mb_h - 1) {  // top
    memcpy(it->y_top, ysrc + 15 * BPS, 16);
    memcpy(it->uv_top, uvsrc + 7 * BPS, 8 + 8);
  }
}

int VP8IteratorNext(VP8EncIterator* const it) {
  if (++it->x == it->enc->mb_w) {
    VP8IteratorSetRow(it, ++it->y);
  } else {
    it->preds += 4;
    it->mb += 1;
    it->nz += 1;
    it->y_top += 16;
    it->uv_top += 16;
  }
  return (0 < --it->count_down);
}

//------------------------------------------------------------------------------
// Helper function to set mode properties

void VP8SetIntra16Mode(const VP8EncIterator* const it, int mode) {
  uint8_t* preds = it->preds;
  int y;
  for (y = 0; y < 4; ++y) {
    memset(preds, mode, 4);
    preds += it->enc->preds_w;
  }
  it->mb->type = 1;
}

void VP8SetIntra4Mode(const VP8EncIterator* const it, const uint8_t* modes) {
  uint8_t* preds = it->preds;
  int y;
  for (y = 4; y > 0; --y) {
    memcpy(preds, modes, 4 * sizeof(*modes));
    preds += it->enc->preds_w;
    modes += 4;
  }
  it->mb->type = 0;
}

void VP8SetIntraUVMode(const VP8EncIterator* const it, int mode) {
  it->mb->uv_mode = mode;
}

void VP8SetSkip(const VP8EncIterator* const it, int skip) {
  it->mb->skip = skip;
}

void VP8SetSegment(const VP8EncIterator* const it, int segment) {
  it->mb->segment = segment;
}

//------------------------------------------------------------------------------
// Intra4x4 sub-blocks iteration
//
//  We store and update the boundary samples into an array of 37 pixels. They
//  are updated as we iterate and reconstructs each intra4x4 blocks in turn.
//  The position of the samples has the following snake pattern:
//
// 16|17 18 19 20|21 22 23 24|25 26 27 28|29 30 31 32|33 34 35 36  <- Top-right
// --+-----------+-----------+-----------+-----------+
// 15|         19|         23|         27|         31|
// 14|         18|         22|         26|         30|
// 13|         17|         21|         25|         29|
// 12|13 14 15 16|17 18 19 20|21 22 23 24|25 26 27 28|
// --+-----------+-----------+-----------+-----------+
// 11|         15|         19|         23|         27|
// 10|         14|         18|         22|         26|
//  9|         13|         17|         21|         25|
//  8| 9 10 11 12|13 14 15 16|17 18 19 20|21 22 23 24|
// --+-----------+-----------+-----------+-----------+
//  7|         11|         15|         19|         23|
//  6|         10|         14|         18|         22|
//  5|          9|         13|         17|         21|
//  4| 5  6  7  8| 9 10 11 12|13 14 15 16|17 18 19 20|
// --+-----------+-----------+-----------+-----------+
//  3|          7|         11|         15|         19|
//  2|          6|         10|         14|         18|
//  1|          5|          9|         13|         17|
//  0| 1  2  3  4| 5  6  7  8| 9 10 11 12|13 14 15 16|
// --+-----------+-----------+-----------+-----------+

// Array to record the position of the top sample to pass to the prediction
// functions in dsp.c.
static const uint8_t VP8TopLeftI4[16] = {
  17, 21, 25, 29,
  13, 17, 21, 25,
  9,  13, 17, 21,
  5,   9, 13, 17
};

void VP8IteratorStartI4(VP8EncIterator* const it) {
  const VP8Encoder* const enc = it->enc;
  int i;

  it->i4 = 0;    // first 4x4 sub-block
  it->i4_top = it->i4_boundary + VP8TopLeftI4[0];

  // Import the boundary samples
  for (i = 0; i < 17; ++i) {    // left
    it->i4_boundary[i] = it->y_left[15 - i];
  }
  for (i = 0; i < 16; ++i) {    // top
    it->i4_boundary[17 + i] = it->y_top[i];
  }
  // top-right samples have a special case on the far right of the picture
  if (it->x < enc->mb_w - 1) {
    for (i = 16; i < 16 + 4; ++i) {
      it->i4_boundary[17 + i] = it->y_top[i];
    }
  } else {    // else, replicate the last valid pixel four times
    for (i = 16; i < 16 + 4; ++i) {
      it->i4_boundary[17 + i] = it->i4_boundary[17 + 15];
    }
  }
#if WEBP_AARCH64 && BPS == 32 && defined(WEBP_MSAN)
  // Intra4Preds_NEON() reads 3 uninitialized bytes from 'i4_boundary' when top
  // is positioned at offset 29 (VP8TopLeftI4[3]). The values are not used
  // meaningfully, but due to limitations in MemorySanitizer related to
  // modeling of tbl instructions, a warning will be issued. This can be
  // removed if MSan is updated to support the instructions. See
  // https://issues.webmproject.org/372109644.
  memset(it->i4_boundary + sizeof(it->i4_boundary) - 3, 0xaa, 3);
#endif
  VP8IteratorNzToBytes(it);  // import the non-zero context
}

int VP8IteratorRotateI4(VP8EncIterator* const it,
                        const uint8_t* const yuv_out) {
  const uint8_t* const blk = yuv_out + VP8Scan[it->i4];
  uint8_t* const top = it->i4_top;
  int i;

  // Update the cache with 7 fresh samples
  for (i = 0; i <= 3; ++i) {
    top[-4 + i] = blk[i + 3 * BPS];   // store future top samples
  }
  if ((it->i4 & 3) != 3) {  // if not on the right sub-blocks #3, #7, #11, #15
    for (i = 0; i <= 2; ++i) {        // store future left samples
      top[i] = blk[3 + (2 - i) * BPS];
    }
  } else {  // else replicate top-right samples, as says the specs.
    for (i = 0; i <= 3; ++i) {
      top[i] = top[i + 4];
    }
  }
  // move pointers to next sub-block
  ++it->i4;
  if (it->i4 == 16) {    // we're done
    return 0;
  }

  it->i4_top = it->i4_boundary + VP8TopLeftI4[it->i4];
  return 1;
}

//------------------------------------------------------------------------------
