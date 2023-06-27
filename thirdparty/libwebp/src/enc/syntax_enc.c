// Copyright 2011 Google Inc. All Rights Reserved.
//
// Use of this source code is governed by a BSD-style license
// that can be found in the COPYING file in the root of the source
// tree. An additional intellectual property rights grant can be found
// in the file PATENTS. All contributing project authors may
// be found in the AUTHORS file in the root of the source tree.
// -----------------------------------------------------------------------------
//
// Header syntax writing
//
// Author: Skal (pascal.massimino@gmail.com)

#include <assert.h>

#include "src/utils/utils.h"
#include "src/webp/format_constants.h"  // RIFF constants
#include "src/webp/mux_types.h"         // ALPHA_FLAG
#include "src/enc/vp8i_enc.h"

//------------------------------------------------------------------------------
// Helper functions

static int IsVP8XNeeded(const VP8Encoder* const enc) {
  return !!enc->has_alpha_;  // Currently the only case when VP8X is needed.
                             // This could change in the future.
}

static int PutPaddingByte(const WebPPicture* const pic) {
  const uint8_t pad_byte[1] = { 0 };
  return !!pic->writer(pad_byte, 1, pic);
}

//------------------------------------------------------------------------------
// Writers for header's various pieces (in order of appearance)

static WebPEncodingError PutRIFFHeader(const VP8Encoder* const enc,
                                       size_t riff_size) {
  const WebPPicture* const pic = enc->pic_;
  uint8_t riff[RIFF_HEADER_SIZE] = {
    'R', 'I', 'F', 'F', 0, 0, 0, 0, 'W', 'E', 'B', 'P'
  };
  assert(riff_size == (uint32_t)riff_size);
  PutLE32(riff + TAG_SIZE, (uint32_t)riff_size);
  if (!pic->writer(riff, sizeof(riff), pic)) {
    return VP8_ENC_ERROR_BAD_WRITE;
  }
  return VP8_ENC_OK;
}

static WebPEncodingError PutVP8XHeader(const VP8Encoder* const enc) {
  const WebPPicture* const pic = enc->pic_;
  uint8_t vp8x[CHUNK_HEADER_SIZE + VP8X_CHUNK_SIZE] = {
    'V', 'P', '8', 'X'
  };
  uint32_t flags = 0;

  assert(IsVP8XNeeded(enc));
  assert(pic->width >= 1 && pic->height >= 1);
  assert(pic->width <= MAX_CANVAS_SIZE && pic->height <= MAX_CANVAS_SIZE);

  if (enc->has_alpha_) {
    flags |= ALPHA_FLAG;
  }

  PutLE32(vp8x + TAG_SIZE,              VP8X_CHUNK_SIZE);
  PutLE32(vp8x + CHUNK_HEADER_SIZE,     flags);
  PutLE24(vp8x + CHUNK_HEADER_SIZE + 4, pic->width - 1);
  PutLE24(vp8x + CHUNK_HEADER_SIZE + 7, pic->height - 1);
  if (!pic->writer(vp8x, sizeof(vp8x), pic)) {
    return VP8_ENC_ERROR_BAD_WRITE;
  }
  return VP8_ENC_OK;
}

static WebPEncodingError PutAlphaChunk(const VP8Encoder* const enc) {
  const WebPPicture* const pic = enc->pic_;
  uint8_t alpha_chunk_hdr[CHUNK_HEADER_SIZE] = {
    'A', 'L', 'P', 'H'
  };

  assert(enc->has_alpha_);

  // Alpha chunk header.
  PutLE32(alpha_chunk_hdr + TAG_SIZE, enc->alpha_data_size_);
  if (!pic->writer(alpha_chunk_hdr, sizeof(alpha_chunk_hdr), pic)) {
    return VP8_ENC_ERROR_BAD_WRITE;
  }

  // Alpha chunk data.
  if (!pic->writer(enc->alpha_data_, enc->alpha_data_size_, pic)) {
    return VP8_ENC_ERROR_BAD_WRITE;
  }

  // Padding.
  if ((enc->alpha_data_size_ & 1) && !PutPaddingByte(pic)) {
    return VP8_ENC_ERROR_BAD_WRITE;
  }
  return VP8_ENC_OK;
}

static WebPEncodingError PutVP8Header(const WebPPicture* const pic,
                                      size_t vp8_size) {
  uint8_t vp8_chunk_hdr[CHUNK_HEADER_SIZE] = {
    'V', 'P', '8', ' '
  };
  assert(vp8_size == (uint32_t)vp8_size);
  PutLE32(vp8_chunk_hdr + TAG_SIZE, (uint32_t)vp8_size);
  if (!pic->writer(vp8_chunk_hdr, sizeof(vp8_chunk_hdr), pic)) {
    return VP8_ENC_ERROR_BAD_WRITE;
  }
  return VP8_ENC_OK;
}

static WebPEncodingError PutVP8FrameHeader(const WebPPicture* const pic,
                                           int profile, size_t size0) {
  uint8_t vp8_frm_hdr[VP8_FRAME_HEADER_SIZE];
  uint32_t bits;

  if (size0 >= VP8_MAX_PARTITION0_SIZE) {  // partition #0 is too big to fit
    return VP8_ENC_ERROR_PARTITION0_OVERFLOW;
  }

  // Paragraph 9.1.
  bits = 0                         // keyframe (1b)
       | (profile << 1)            // profile (3b)
       | (1 << 4)                  // visible (1b)
       | ((uint32_t)size0 << 5);   // partition length (19b)
  vp8_frm_hdr[0] = (bits >>  0) & 0xff;
  vp8_frm_hdr[1] = (bits >>  8) & 0xff;
  vp8_frm_hdr[2] = (bits >> 16) & 0xff;
  // signature
  vp8_frm_hdr[3] = (VP8_SIGNATURE >> 16) & 0xff;
  vp8_frm_hdr[4] = (VP8_SIGNATURE >>  8) & 0xff;
  vp8_frm_hdr[5] = (VP8_SIGNATURE >>  0) & 0xff;
  // dimensions
  vp8_frm_hdr[6] = pic->width & 0xff;
  vp8_frm_hdr[7] = pic->width >> 8;
  vp8_frm_hdr[8] = pic->height & 0xff;
  vp8_frm_hdr[9] = pic->height >> 8;

  if (!pic->writer(vp8_frm_hdr, sizeof(vp8_frm_hdr), pic)) {
    return VP8_ENC_ERROR_BAD_WRITE;
  }
  return VP8_ENC_OK;
}

// WebP Headers.
static int PutWebPHeaders(const VP8Encoder* const enc, size_t size0,
                          size_t vp8_size, size_t riff_size) {
  WebPPicture* const pic = enc->pic_;
  WebPEncodingError err = VP8_ENC_OK;

  // RIFF header.
  err = PutRIFFHeader(enc, riff_size);
  if (err != VP8_ENC_OK) goto Error;

  // VP8X.
  if (IsVP8XNeeded(enc)) {
    err = PutVP8XHeader(enc);
    if (err != VP8_ENC_OK) goto Error;
  }

  // Alpha.
  if (enc->has_alpha_) {
    err = PutAlphaChunk(enc);
    if (err != VP8_ENC_OK) goto Error;
  }

  // VP8 header.
  err = PutVP8Header(pic, vp8_size);
  if (err != VP8_ENC_OK) goto Error;

  // VP8 frame header.
  err = PutVP8FrameHeader(pic, enc->profile_, size0);
  if (err != VP8_ENC_OK) goto Error;

  // All OK.
  return 1;

  // Error.
 Error:
  return WebPEncodingSetError(pic, err);
}

// Segmentation header
static void PutSegmentHeader(VP8BitWriter* const bw,
                             const VP8Encoder* const enc) {
  const VP8EncSegmentHeader* const hdr = &enc->segment_hdr_;
  const VP8EncProba* const proba = &enc->proba_;
  if (VP8PutBitUniform(bw, (hdr->num_segments_ > 1))) {
    // We always 'update' the quant and filter strength values
    const int update_data = 1;
    int s;
    VP8PutBitUniform(bw, hdr->update_map_);
    if (VP8PutBitUniform(bw, update_data)) {
      // we always use absolute values, not relative ones
      VP8PutBitUniform(bw, 1);   // (segment_feature_mode = 1. Paragraph 9.3.)
      for (s = 0; s < NUM_MB_SEGMENTS; ++s) {
        VP8PutSignedBits(bw, enc->dqm_[s].quant_, 7);
      }
      for (s = 0; s < NUM_MB_SEGMENTS; ++s) {
        VP8PutSignedBits(bw, enc->dqm_[s].fstrength_, 6);
      }
    }
    if (hdr->update_map_) {
      for (s = 0; s < 3; ++s) {
        if (VP8PutBitUniform(bw, (proba->segments_[s] != 255u))) {
          VP8PutBits(bw, proba->segments_[s], 8);
        }
      }
    }
  }
}

// Filtering parameters header
static void PutFilterHeader(VP8BitWriter* const bw,
                            const VP8EncFilterHeader* const hdr) {
  const int use_lf_delta = (hdr->i4x4_lf_delta_ != 0);
  VP8PutBitUniform(bw, hdr->simple_);
  VP8PutBits(bw, hdr->level_, 6);
  VP8PutBits(bw, hdr->sharpness_, 3);
  if (VP8PutBitUniform(bw, use_lf_delta)) {
    // '0' is the default value for i4x4_lf_delta_ at frame #0.
    const int need_update = (hdr->i4x4_lf_delta_ != 0);
    if (VP8PutBitUniform(bw, need_update)) {
      // we don't use ref_lf_delta => emit four 0 bits
      VP8PutBits(bw, 0, 4);
      // we use mode_lf_delta for i4x4
      VP8PutSignedBits(bw, hdr->i4x4_lf_delta_, 6);
      VP8PutBits(bw, 0, 3);    // all others unused
    }
  }
}

// Nominal quantization parameters
static void PutQuant(VP8BitWriter* const bw,
                     const VP8Encoder* const enc) {
  VP8PutBits(bw, enc->base_quant_, 7);
  VP8PutSignedBits(bw, enc->dq_y1_dc_, 4);
  VP8PutSignedBits(bw, enc->dq_y2_dc_, 4);
  VP8PutSignedBits(bw, enc->dq_y2_ac_, 4);
  VP8PutSignedBits(bw, enc->dq_uv_dc_, 4);
  VP8PutSignedBits(bw, enc->dq_uv_ac_, 4);
}

// Partition sizes
static int EmitPartitionsSize(const VP8Encoder* const enc,
                              WebPPicture* const pic) {
  uint8_t buf[3 * (MAX_NUM_PARTITIONS - 1)];
  int p;
  for (p = 0; p < enc->num_parts_ - 1; ++p) {
    const size_t part_size = VP8BitWriterSize(enc->parts_ + p);
    if (part_size >= VP8_MAX_PARTITION_SIZE) {
      return WebPEncodingSetError(pic, VP8_ENC_ERROR_PARTITION_OVERFLOW);
    }
    buf[3 * p + 0] = (part_size >>  0) & 0xff;
    buf[3 * p + 1] = (part_size >>  8) & 0xff;
    buf[3 * p + 2] = (part_size >> 16) & 0xff;
  }
  return p ? pic->writer(buf, 3 * p, pic) : 1;
}

//------------------------------------------------------------------------------

static int GeneratePartition0(VP8Encoder* const enc) {
  VP8BitWriter* const bw = &enc->bw_;
  const int mb_size = enc->mb_w_ * enc->mb_h_;
  uint64_t pos1, pos2, pos3;

  pos1 = VP8BitWriterPos(bw);
  if (!VP8BitWriterInit(bw, mb_size * 7 / 8)) {        // ~7 bits per macroblock
    return WebPEncodingSetError(enc->pic_, VP8_ENC_ERROR_OUT_OF_MEMORY);
  }
  VP8PutBitUniform(bw, 0);   // colorspace
  VP8PutBitUniform(bw, 0);   // clamp type

  PutSegmentHeader(bw, enc);
  PutFilterHeader(bw, &enc->filter_hdr_);
  VP8PutBits(bw, enc->num_parts_ == 8 ? 3 :
                 enc->num_parts_ == 4 ? 2 :
                 enc->num_parts_ == 2 ? 1 : 0, 2);
  PutQuant(bw, enc);
  VP8PutBitUniform(bw, 0);   // no proba update
  VP8WriteProbas(bw, &enc->proba_);
  pos2 = VP8BitWriterPos(bw);
  VP8CodeIntraModes(enc);
  VP8BitWriterFinish(bw);

  pos3 = VP8BitWriterPos(bw);

#if !defined(WEBP_DISABLE_STATS)
  if (enc->pic_->stats) {
    enc->pic_->stats->header_bytes[0] = (int)((pos2 - pos1 + 7) >> 3);
    enc->pic_->stats->header_bytes[1] = (int)((pos3 - pos2 + 7) >> 3);
    enc->pic_->stats->alpha_data_size = (int)enc->alpha_data_size_;
  }
#else
  (void)pos1;
  (void)pos2;
  (void)pos3;
#endif
  if (bw->error_) {
    return WebPEncodingSetError(enc->pic_, VP8_ENC_ERROR_OUT_OF_MEMORY);
  }
  return 1;
}

void VP8EncFreeBitWriters(VP8Encoder* const enc) {
  int p;
  VP8BitWriterWipeOut(&enc->bw_);
  for (p = 0; p < enc->num_parts_; ++p) {
    VP8BitWriterWipeOut(enc->parts_ + p);
  }
}

int VP8EncWrite(VP8Encoder* const enc) {
  WebPPicture* const pic = enc->pic_;
  VP8BitWriter* const bw = &enc->bw_;
  const int task_percent = 19;
  const int percent_per_part = task_percent / enc->num_parts_;
  const int final_percent = enc->percent_ + task_percent;
  int ok = 0;
  size_t vp8_size, pad, riff_size;
  int p;

  // Partition #0 with header and partition sizes
  ok = GeneratePartition0(enc);
  if (!ok) return 0;

  // Compute VP8 size
  vp8_size = VP8_FRAME_HEADER_SIZE +
             VP8BitWriterSize(bw) +
             3 * (enc->num_parts_ - 1);
  for (p = 0; p < enc->num_parts_; ++p) {
    vp8_size += VP8BitWriterSize(enc->parts_ + p);
  }
  pad = vp8_size & 1;
  vp8_size += pad;

  // Compute RIFF size
  // At the minimum it is: "WEBPVP8 nnnn" + VP8 data size.
  riff_size = TAG_SIZE + CHUNK_HEADER_SIZE + vp8_size;
  if (IsVP8XNeeded(enc)) {  // Add size for: VP8X header + data.
    riff_size += CHUNK_HEADER_SIZE + VP8X_CHUNK_SIZE;
  }
  if (enc->has_alpha_) {  // Add size for: ALPH header + data.
    const uint32_t padded_alpha_size = enc->alpha_data_size_ +
                                       (enc->alpha_data_size_ & 1);
    riff_size += CHUNK_HEADER_SIZE + padded_alpha_size;
  }
  // RIFF size should fit in 32-bits.
  if (riff_size > 0xfffffffeU) {
    return WebPEncodingSetError(pic, VP8_ENC_ERROR_FILE_TOO_BIG);
  }

  // Emit headers and partition #0
  {
    const uint8_t* const part0 = VP8BitWriterBuf(bw);
    const size_t size0 = VP8BitWriterSize(bw);
    ok = ok && PutWebPHeaders(enc, size0, vp8_size, riff_size)
            && pic->writer(part0, size0, pic)
            && EmitPartitionsSize(enc, pic);
    VP8BitWriterWipeOut(bw);    // will free the internal buffer.
  }

  // Token partitions
  for (p = 0; p < enc->num_parts_; ++p) {
    const uint8_t* const buf = VP8BitWriterBuf(enc->parts_ + p);
    const size_t size = VP8BitWriterSize(enc->parts_ + p);
    if (size) ok = ok && pic->writer(buf, size, pic);
    VP8BitWriterWipeOut(enc->parts_ + p);    // will free the internal buffer.
    ok = ok && WebPReportProgress(pic, enc->percent_ + percent_per_part,
                                  &enc->percent_);
  }

  // Padding byte
  if (ok && pad) {
    ok = PutPaddingByte(pic);
  }

  enc->coded_size_ = (int)(CHUNK_HEADER_SIZE + riff_size);
  ok = ok && WebPReportProgress(pic, final_percent, &enc->percent_);
  return ok;
}

//------------------------------------------------------------------------------

