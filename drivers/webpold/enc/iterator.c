// Copyright 2011 Google Inc. All Rights Reserved.
//
// This code is licensed under the same terms as WebM:
//  Software License Agreement:  http://www.webmproject.org/license/software/
//  Additional IP Rights Grant:  http://www.webmproject.org/license/additional/
// -----------------------------------------------------------------------------
//
// VP8Iterator: block iterator
//
// Author: Skal (pascal.massimino@gmail.com)

#include <string.h>

#include "./vp8enci.h"

#if defined(__cplusplus) || defined(c_plusplus)
extern "C" {
#endif

//------------------------------------------------------------------------------
// VP8Iterator
//------------------------------------------------------------------------------

static void InitLeft(VP8EncIterator* const it) {
  const VP8Encoder* const enc = it->enc_;
  enc->y_left_[-1] = enc->u_left_[-1] = enc->v_left_[-1] =
      (it->y_ > 0) ? 129 : 127;
  memset(enc->y_left_, 129, 16);
  memset(enc->u_left_, 129, 8);
  memset(enc->v_left_, 129, 8);
  it->left_nz_[8] = 0;
}

static void InitTop(VP8EncIterator* const it) {
  const VP8Encoder* const enc = it->enc_;
  const size_t top_size = enc->mb_w_ * 16;
  memset(enc->y_top_, 127, 2 * top_size);
  memset(enc->nz_, 0, enc->mb_w_ * sizeof(*enc->nz_));
}

void VP8IteratorReset(VP8EncIterator* const it) {
  VP8Encoder* const enc = it->enc_;
  it->x_ = 0;
  it->y_ = 0;
  it->y_offset_ = 0;
  it->uv_offset_ = 0;
  it->mb_ = enc->mb_info_;
  it->preds_ = enc->preds_;
  it->nz_ = enc->nz_;
  it->bw_ = &enc->parts_[0];
  it->done_ = enc->mb_w_* enc->mb_h_;
  InitTop(it);
  InitLeft(it);
  memset(it->bit_count_, 0, sizeof(it->bit_count_));
  it->do_trellis_ = 0;
}

void VP8IteratorInit(VP8Encoder* const enc, VP8EncIterator* const it) {
  it->enc_ = enc;
  it->y_stride_  = enc->pic_->y_stride;
  it->uv_stride_ = enc->pic_->uv_stride;
  // TODO(later): for multithreading, these should be owned by 'it'.
  it->yuv_in_   = enc->yuv_in_;
  it->yuv_out_  = enc->yuv_out_;
  it->yuv_out2_ = enc->yuv_out2_;
  it->yuv_p_    = enc->yuv_p_;
  it->lf_stats_ = enc->lf_stats_;
  it->percent0_ = enc->percent_;
  VP8IteratorReset(it);
}

int VP8IteratorProgress(const VP8EncIterator* const it, int delta) {
  VP8Encoder* const enc = it->enc_;
  if (delta && enc->pic_->progress_hook) {
    const int percent = (enc->mb_h_ <= 1)
                      ? it->percent0_
                      : it->percent0_ + delta * it->y_ / (enc->mb_h_ - 1);
    return WebPReportProgress(enc->pic_, percent, &enc->percent_);
  }
  return 1;
}

//------------------------------------------------------------------------------
// Import the source samples into the cache. Takes care of replicating
// boundary pixels if necessary.

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

void VP8IteratorImport(const VP8EncIterator* const it) {
  const VP8Encoder* const enc = it->enc_;
  const int x = it->x_, y = it->y_;
  const WebPPicture* const pic = enc->pic_;
  const uint8_t* const ysrc = pic->y + (y * pic->y_stride + x) * 16;
  const uint8_t* const usrc = pic->u + (y * pic->uv_stride + x) * 8;
  const uint8_t* const vsrc = pic->v + (y * pic->uv_stride + x) * 8;
  uint8_t* const ydst = it->yuv_in_ + Y_OFF;
  uint8_t* const udst = it->yuv_in_ + U_OFF;
  uint8_t* const vdst = it->yuv_in_ + V_OFF;
  int w = (pic->width - x * 16);
  int h = (pic->height - y * 16);

  if (w > 16) w = 16;
  if (h > 16) h = 16;

  // Luma plane
  ImportBlock(ysrc, pic->y_stride, ydst, w, h, 16);

  {   // U/V planes
    const int uv_w = (w + 1) >> 1;
    const int uv_h = (h + 1) >> 1;
    ImportBlock(usrc, pic->uv_stride, udst, uv_w, uv_h, 8);
    ImportBlock(vsrc, pic->uv_stride, vdst, uv_w, uv_h, 8);
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
  const VP8Encoder* const enc = it->enc_;
  if (enc->config_->show_compressed) {
    const int x = it->x_, y = it->y_;
    const uint8_t* const ysrc = it->yuv_out_ + Y_OFF;
    const uint8_t* const usrc = it->yuv_out_ + U_OFF;
    const uint8_t* const vsrc = it->yuv_out_ + V_OFF;
    const WebPPicture* const pic = enc->pic_;
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
  const int tnz = it->nz_[0], lnz = it->nz_[-1];
  int* const top_nz = it->top_nz_;
  int* const left_nz = it->left_nz_;

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
  const int* const top_nz = it->top_nz_;
  const int* const left_nz = it->left_nz_;
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

  *it->nz_ = nz;
}

#undef BIT

//------------------------------------------------------------------------------
// Advance to the next position, doing the bookeeping.

int VP8IteratorNext(VP8EncIterator* const it,
                    const uint8_t* const block_to_save) {
  VP8Encoder* const enc = it->enc_;
  if (block_to_save) {
    const int x = it->x_, y = it->y_;
    const uint8_t* const ysrc = block_to_save + Y_OFF;
    const uint8_t* const usrc = block_to_save + U_OFF;
    if (x < enc->mb_w_ - 1) {   // left
      int i;
      for (i = 0; i < 16; ++i) {
        enc->y_left_[i] = ysrc[15 + i * BPS];
      }
      for (i = 0; i < 8; ++i) {
        enc->u_left_[i] = usrc[7 + i * BPS];
        enc->v_left_[i] = usrc[15 + i * BPS];
      }
      // top-left (before 'top'!)
      enc->y_left_[-1] = enc->y_top_[x * 16 + 15];
      enc->u_left_[-1] = enc->uv_top_[x * 16 + 0 + 7];
      enc->v_left_[-1] = enc->uv_top_[x * 16 + 8 + 7];
    }
    if (y < enc->mb_h_ - 1) {  // top
      memcpy(enc->y_top_ + x * 16, ysrc + 15 * BPS, 16);
      memcpy(enc->uv_top_ + x * 16, usrc + 7 * BPS, 8 + 8);
    }
  }

  it->mb_++;
  it->preds_ += 4;
  it->nz_++;
  it->x_++;
  if (it->x_ == enc->mb_w_) {
    it->x_ = 0;
    it->y_++;
    it->bw_ = &enc->parts_[it->y_ & (enc->num_parts_ - 1)];
    it->preds_ = enc->preds_ + it->y_ * 4 * enc->preds_w_;
    it->nz_ = enc->nz_;
    InitLeft(it);
  }
  return (0 < --it->done_);
}

//------------------------------------------------------------------------------
// Helper function to set mode properties

void VP8SetIntra16Mode(const VP8EncIterator* const it, int mode) {
  uint8_t* preds = it->preds_;
  int y;
  for (y = 0; y < 4; ++y) {
    memset(preds, mode, 4);
    preds += it->enc_->preds_w_;
  }
  it->mb_->type_ = 1;
}

void VP8SetIntra4Mode(const VP8EncIterator* const it, const uint8_t* modes) {
  uint8_t* preds = it->preds_;
  int y;
  for (y = 4; y > 0; --y) {
    memcpy(preds, modes, 4 * sizeof(*modes));
    preds += it->enc_->preds_w_;
    modes += 4;
  }
  it->mb_->type_ = 0;
}

void VP8SetIntraUVMode(const VP8EncIterator* const it, int mode) {
  it->mb_->uv_mode_ = mode;
}

void VP8SetSkip(const VP8EncIterator* const it, int skip) {
  it->mb_->skip_ = skip;
}

void VP8SetSegment(const VP8EncIterator* const it, int segment) {
  it->mb_->segment_ = segment;
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
  const VP8Encoder* const enc = it->enc_;
  int i;

  it->i4_ = 0;    // first 4x4 sub-block
  it->i4_top_ = it->i4_boundary_ + VP8TopLeftI4[0];

  // Import the boundary samples
  for (i = 0; i < 17; ++i) {    // left
    it->i4_boundary_[i] = enc->y_left_[15 - i];
  }
  for (i = 0; i < 16; ++i) {    // top
    it->i4_boundary_[17 + i] = enc->y_top_[it->x_ * 16 + i];
  }
  // top-right samples have a special case on the far right of the picture
  if (it->x_ < enc->mb_w_ - 1) {
    for (i = 16; i < 16 + 4; ++i) {
      it->i4_boundary_[17 + i] = enc->y_top_[it->x_ * 16 + i];
    }
  } else {    // else, replicate the last valid pixel four times
    for (i = 16; i < 16 + 4; ++i) {
      it->i4_boundary_[17 + i] = it->i4_boundary_[17 + 15];
    }
  }
  VP8IteratorNzToBytes(it);  // import the non-zero context
}

int VP8IteratorRotateI4(VP8EncIterator* const it,
                        const uint8_t* const yuv_out) {
  const uint8_t* const blk = yuv_out + VP8Scan[it->i4_];
  uint8_t* const top = it->i4_top_;
  int i;

  // Update the cache with 7 fresh samples
  for (i = 0; i <= 3; ++i) {
    top[-4 + i] = blk[i + 3 * BPS];   // store future top samples
  }
  if ((it->i4_ & 3) != 3) {  // if not on the right sub-blocks #3, #7, #11, #15
    for (i = 0; i <= 2; ++i) {        // store future left samples
      top[i] = blk[3 + (2 - i) * BPS];
    }
  } else {  // else replicate top-right samples, as says the specs.
    for (i = 0; i <= 3; ++i) {
      top[i] = top[i + 4];
    }
  }
  // move pointers to next sub-block
  ++it->i4_;
  if (it->i4_ == 16) {    // we're done
    return 0;
  }

  it->i4_top_ = it->i4_boundary_ + VP8TopLeftI4[it->i4_];
  return 1;
}

//------------------------------------------------------------------------------

#if defined(__cplusplus) || defined(c_plusplus)
}    // extern "C"
#endif
