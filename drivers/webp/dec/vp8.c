// Copyright 2010 Google Inc. All Rights Reserved.
//
// This code is licensed under the same terms as WebM:
//  Software License Agreement:  http://www.webmproject.org/license/software/
//  Additional IP Rights Grant:  http://www.webmproject.org/license/additional/
// -----------------------------------------------------------------------------
//
// main entry for the decoder
//
// Author: Skal (pascal.massimino@gmail.com)

#include <stdlib.h>

#include "./vp8i.h"
#include "./vp8li.h"
#include "./webpi.h"
#include "../utils/bit_reader.h"

#if defined(__cplusplus) || defined(c_plusplus)
extern "C" {
#endif

//------------------------------------------------------------------------------

int WebPGetDecoderVersion(void) {
  return (DEC_MAJ_VERSION << 16) | (DEC_MIN_VERSION << 8) | DEC_REV_VERSION;
}

//------------------------------------------------------------------------------
// VP8Decoder

static void SetOk(VP8Decoder* const dec) {
  dec->status_ = VP8_STATUS_OK;
  dec->error_msg_ = "OK";
}

int VP8InitIoInternal(VP8Io* const io, int version) {
  if (WEBP_ABI_IS_INCOMPATIBLE(version, WEBP_DECODER_ABI_VERSION)) {
    return 0;  // mismatch error
  }
  if (io != NULL) {
    memset(io, 0, sizeof(*io));
  }
  return 1;
}

VP8Decoder* VP8New(void) {
  VP8Decoder* const dec = (VP8Decoder*)calloc(1, sizeof(*dec));
  if (dec != NULL) {
    SetOk(dec);
    WebPWorkerInit(&dec->worker_);
    dec->ready_ = 0;
    dec->num_parts_ = 1;
  }
  return dec;
}

VP8StatusCode VP8Status(VP8Decoder* const dec) {
  if (!dec) return VP8_STATUS_INVALID_PARAM;
  return dec->status_;
}

const char* VP8StatusMessage(VP8Decoder* const dec) {
  if (dec == NULL) return "no object";
  if (!dec->error_msg_) return "OK";
  return dec->error_msg_;
}

void VP8Delete(VP8Decoder* const dec) {
  if (dec != NULL) {
    VP8Clear(dec);
    free(dec);
  }
}

int VP8SetError(VP8Decoder* const dec,
                VP8StatusCode error, const char* const msg) {
  // TODO This check would be unnecessary if alpha decompression was separated
  // from VP8ProcessRow/FinishRow. This avoids setting 'dec->status_' to
  // something other than VP8_STATUS_BITSTREAM_ERROR on alpha decompression
  // failure.
  if (dec->status_ == VP8_STATUS_OK) {
    dec->status_ = error;
    dec->error_msg_ = msg;
    dec->ready_ = 0;
  }
  return 0;
}

//------------------------------------------------------------------------------

int VP8CheckSignature(const uint8_t* const data, size_t data_size) {
  return (data_size >= 3 &&
          data[0] == 0x9d && data[1] == 0x01 && data[2] == 0x2a);
}

int VP8GetInfo(const uint8_t* data, size_t data_size, size_t chunk_size,
               int* const width, int* const height) {
  if (data == NULL || data_size < VP8_FRAME_HEADER_SIZE) {
    return 0;         // not enough data
  }
  // check signature
  if (!VP8CheckSignature(data + 3, data_size - 3)) {
    return 0;         // Wrong signature.
  } else {
    const uint32_t bits = data[0] | (data[1] << 8) | (data[2] << 16);
    const int key_frame = !(bits & 1);
    const int w = ((data[7] << 8) | data[6]) & 0x3fff;
    const int h = ((data[9] << 8) | data[8]) & 0x3fff;

    if (!key_frame) {   // Not a keyframe.
      return 0;
    }

    if (((bits >> 1) & 7) > 3) {
      return 0;         // unknown profile
    }
    if (!((bits >> 4) & 1)) {
      return 0;         // first frame is invisible!
    }
    if (((bits >> 5)) >= chunk_size) {  // partition_length
      return 0;         // inconsistent size information.
    }

    if (width) {
      *width = w;
    }
    if (height) {
      *height = h;
    }

    return 1;
  }
}

//------------------------------------------------------------------------------
// Header parsing

static void ResetSegmentHeader(VP8SegmentHeader* const hdr) {
  assert(hdr != NULL);
  hdr->use_segment_ = 0;
  hdr->update_map_ = 0;
  hdr->absolute_delta_ = 1;
  memset(hdr->quantizer_, 0, sizeof(hdr->quantizer_));
  memset(hdr->filter_strength_, 0, sizeof(hdr->filter_strength_));
}

// Paragraph 9.3
static int ParseSegmentHeader(VP8BitReader* br,
                              VP8SegmentHeader* hdr, VP8Proba* proba) {
  assert(br != NULL);
  assert(hdr != NULL);
  hdr->use_segment_ = VP8Get(br);
  if (hdr->use_segment_) {
    hdr->update_map_ = VP8Get(br);
    if (VP8Get(br)) {   // update data
      int s;
      hdr->absolute_delta_ = VP8Get(br);
      for (s = 0; s < NUM_MB_SEGMENTS; ++s) {
        hdr->quantizer_[s] = VP8Get(br) ? VP8GetSignedValue(br, 7) : 0;
      }
      for (s = 0; s < NUM_MB_SEGMENTS; ++s) {
        hdr->filter_strength_[s] = VP8Get(br) ? VP8GetSignedValue(br, 6) : 0;
      }
    }
    if (hdr->update_map_) {
      int s;
      for (s = 0; s < MB_FEATURE_TREE_PROBS; ++s) {
        proba->segments_[s] = VP8Get(br) ? VP8GetValue(br, 8) : 255u;
      }
    }
  } else {
    hdr->update_map_ = 0;
  }
  return !br->eof_;
}

// Paragraph 9.5
// This function returns VP8_STATUS_SUSPENDED if we don't have all the
// necessary data in 'buf'.
// This case is not necessarily an error (for incremental decoding).
// Still, no bitreader is ever initialized to make it possible to read
// unavailable memory.
// If we don't even have the partitions' sizes, than VP8_STATUS_NOT_ENOUGH_DATA
// is returned, and this is an unrecoverable error.
// If the partitions were positioned ok, VP8_STATUS_OK is returned.
static VP8StatusCode ParsePartitions(VP8Decoder* const dec,
                                     const uint8_t* buf, size_t size) {
  VP8BitReader* const br = &dec->br_;
  const uint8_t* sz = buf;
  const uint8_t* buf_end = buf + size;
  const uint8_t* part_start;
  int last_part;
  int p;

  dec->num_parts_ = 1 << VP8GetValue(br, 2);
  last_part = dec->num_parts_ - 1;
  part_start = buf + last_part * 3;
  if (buf_end < part_start) {
    // we can't even read the sizes with sz[]! That's a failure.
    return VP8_STATUS_NOT_ENOUGH_DATA;
  }
  for (p = 0; p < last_part; ++p) {
    const uint32_t psize = sz[0] | (sz[1] << 8) | (sz[2] << 16);
    const uint8_t* part_end = part_start + psize;
    if (part_end > buf_end) part_end = buf_end;
    VP8InitBitReader(dec->parts_ + p, part_start, part_end);
    part_start = part_end;
    sz += 3;
  }
  VP8InitBitReader(dec->parts_ + last_part, part_start, buf_end);
  return (part_start < buf_end) ? VP8_STATUS_OK :
           VP8_STATUS_SUSPENDED;   // Init is ok, but there's not enough data
}

// Paragraph 9.4
static int ParseFilterHeader(VP8BitReader* br, VP8Decoder* const dec) {
  VP8FilterHeader* const hdr = &dec->filter_hdr_;
  hdr->simple_    = VP8Get(br);
  hdr->level_     = VP8GetValue(br, 6);
  hdr->sharpness_ = VP8GetValue(br, 3);
  hdr->use_lf_delta_ = VP8Get(br);
  if (hdr->use_lf_delta_) {
    if (VP8Get(br)) {   // update lf-delta?
      int i;
      for (i = 0; i < NUM_REF_LF_DELTAS; ++i) {
        if (VP8Get(br)) {
          hdr->ref_lf_delta_[i] = VP8GetSignedValue(br, 6);
        }
      }
      for (i = 0; i < NUM_MODE_LF_DELTAS; ++i) {
        if (VP8Get(br)) {
          hdr->mode_lf_delta_[i] = VP8GetSignedValue(br, 6);
        }
      }
    }
  }
  dec->filter_type_ = (hdr->level_ == 0) ? 0 : hdr->simple_ ? 1 : 2;
  if (dec->filter_type_ > 0) {    // precompute filter levels per segment
    if (dec->segment_hdr_.use_segment_) {
      int s;
      for (s = 0; s < NUM_MB_SEGMENTS; ++s) {
        int strength = dec->segment_hdr_.filter_strength_[s];
        if (!dec->segment_hdr_.absolute_delta_) {
          strength += hdr->level_;
        }
        dec->filter_levels_[s] = strength;
      }
    } else {
      dec->filter_levels_[0] = hdr->level_;
    }
  }
  return !br->eof_;
}

// Topmost call
int VP8GetHeaders(VP8Decoder* const dec, VP8Io* const io) {
  const uint8_t* buf;
  size_t buf_size;
  VP8FrameHeader* frm_hdr;
  VP8PictureHeader* pic_hdr;
  VP8BitReader* br;
  VP8StatusCode status;
  WebPHeaderStructure headers;

  if (dec == NULL) {
    return 0;
  }
  SetOk(dec);
  if (io == NULL) {
    return VP8SetError(dec, VP8_STATUS_INVALID_PARAM,
                       "null VP8Io passed to VP8GetHeaders()");
  }

  // Process Pre-VP8 chunks.
  headers.data = io->data;
  headers.data_size = io->data_size;
  status = WebPParseHeaders(&headers);
  if (status != VP8_STATUS_OK) {
    return VP8SetError(dec, status, "Incorrect/incomplete header.");
  }
  if (headers.is_lossless) {
    return VP8SetError(dec, VP8_STATUS_BITSTREAM_ERROR,
                       "Unexpected lossless format encountered.");
  }

  if (dec->alpha_data_ == NULL) {
    assert(dec->alpha_data_size_ == 0);
    // We have NOT set alpha data yet. Set it now.
    // (This is to ensure that dec->alpha_data_ is NOT reset to NULL if
    // WebPParseHeaders() is called more than once, as in incremental decoding
    // case.)
    dec->alpha_data_ = headers.alpha_data;
    dec->alpha_data_size_ = headers.alpha_data_size;
  }

  // Process the VP8 frame header.
  buf = headers.data + headers.offset;
  buf_size = headers.data_size - headers.offset;
  assert(headers.data_size >= headers.offset);  // WebPParseHeaders' guarantee
  if (buf_size < 4) {
    return VP8SetError(dec, VP8_STATUS_NOT_ENOUGH_DATA,
                       "Truncated header.");
  }

  // Paragraph 9.1
  {
    const uint32_t bits = buf[0] | (buf[1] << 8) | (buf[2] << 16);
    frm_hdr = &dec->frm_hdr_;
    frm_hdr->key_frame_ = !(bits & 1);
    frm_hdr->profile_ = (bits >> 1) & 7;
    frm_hdr->show_ = (bits >> 4) & 1;
    frm_hdr->partition_length_ = (bits >> 5);
    if (frm_hdr->profile_ > 3)
      return VP8SetError(dec, VP8_STATUS_BITSTREAM_ERROR,
                         "Incorrect keyframe parameters.");
    if (!frm_hdr->show_)
      return VP8SetError(dec, VP8_STATUS_UNSUPPORTED_FEATURE,
                         "Frame not displayable.");
    buf += 3;
    buf_size -= 3;
  }

  pic_hdr = &dec->pic_hdr_;
  if (frm_hdr->key_frame_) {
    // Paragraph 9.2
    if (buf_size < 7) {
      return VP8SetError(dec, VP8_STATUS_NOT_ENOUGH_DATA,
                         "cannot parse picture header");
    }
    if (!VP8CheckSignature(buf, buf_size)) {
      return VP8SetError(dec, VP8_STATUS_BITSTREAM_ERROR,
                         "Bad code word");
    }
    pic_hdr->width_ = ((buf[4] << 8) | buf[3]) & 0x3fff;
    pic_hdr->xscale_ = buf[4] >> 6;   // ratio: 1, 5/4 5/3 or 2
    pic_hdr->height_ = ((buf[6] << 8) | buf[5]) & 0x3fff;
    pic_hdr->yscale_ = buf[6] >> 6;
    buf += 7;
    buf_size -= 7;

    dec->mb_w_ = (pic_hdr->width_ + 15) >> 4;
    dec->mb_h_ = (pic_hdr->height_ + 15) >> 4;
    // Setup default output area (can be later modified during io->setup())
    io->width = pic_hdr->width_;
    io->height = pic_hdr->height_;
    io->use_scaling  = 0;
    io->use_cropping = 0;
    io->crop_top  = 0;
    io->crop_left = 0;
    io->crop_right  = io->width;
    io->crop_bottom = io->height;
    io->mb_w = io->width;   // sanity check
    io->mb_h = io->height;  // ditto

    VP8ResetProba(&dec->proba_);
    ResetSegmentHeader(&dec->segment_hdr_);
    dec->segment_ = 0;    // default for intra
  }

  // Check if we have all the partition #0 available, and initialize dec->br_
  // to read this partition (and this partition only).
  if (frm_hdr->partition_length_ > buf_size) {
    return VP8SetError(dec, VP8_STATUS_NOT_ENOUGH_DATA,
                       "bad partition length");
  }

  br = &dec->br_;
  VP8InitBitReader(br, buf, buf + frm_hdr->partition_length_);
  buf += frm_hdr->partition_length_;
  buf_size -= frm_hdr->partition_length_;

  if (frm_hdr->key_frame_) {
    pic_hdr->colorspace_ = VP8Get(br);
    pic_hdr->clamp_type_ = VP8Get(br);
  }
  if (!ParseSegmentHeader(br, &dec->segment_hdr_, &dec->proba_)) {
    return VP8SetError(dec, VP8_STATUS_BITSTREAM_ERROR,
                       "cannot parse segment header");
  }
  // Filter specs
  if (!ParseFilterHeader(br, dec)) {
    return VP8SetError(dec, VP8_STATUS_BITSTREAM_ERROR,
                       "cannot parse filter header");
  }
  status = ParsePartitions(dec, buf, buf_size);
  if (status != VP8_STATUS_OK) {
    return VP8SetError(dec, status, "cannot parse partitions");
  }

  // quantizer change
  VP8ParseQuant(dec);

  // Frame buffer marking
  if (!frm_hdr->key_frame_) {
    // Paragraph 9.7
#ifndef ONLY_KEYFRAME_CODE
    dec->buffer_flags_ = VP8Get(br) << 0;   // update golden
    dec->buffer_flags_ |= VP8Get(br) << 1;  // update alt ref
    if (!(dec->buffer_flags_ & 1)) {
      dec->buffer_flags_ |= VP8GetValue(br, 2) << 2;
    }
    if (!(dec->buffer_flags_ & 2)) {
      dec->buffer_flags_ |= VP8GetValue(br, 2) << 4;
    }
    dec->buffer_flags_ |= VP8Get(br) << 6;    // sign bias golden
    dec->buffer_flags_ |= VP8Get(br) << 7;    // sign bias alt ref
#else
    return VP8SetError(dec, VP8_STATUS_UNSUPPORTED_FEATURE,
                       "Not a key frame.");
#endif
  } else {
    dec->buffer_flags_ = 0x003 | 0x100;
  }

  // Paragraph 9.8
#ifndef ONLY_KEYFRAME_CODE
  dec->update_proba_ = VP8Get(br);
  if (!dec->update_proba_) {    // save for later restore
    dec->proba_saved_ = dec->proba_;
  }
  dec->buffer_flags_ &= 1 << 8;
  dec->buffer_flags_ |=
      (frm_hdr->key_frame_ || VP8Get(br)) << 8;    // refresh last frame
#else
  VP8Get(br);   // just ignore the value of update_proba_
#endif

  VP8ParseProba(br, dec);

#ifdef WEBP_EXPERIMENTAL_FEATURES
  // Extensions
  if (dec->pic_hdr_.colorspace_) {
    const size_t kTrailerSize = 8;
    const uint8_t kTrailerMarker = 0x01;
    const uint8_t* ext_buf = buf - kTrailerSize;
    size_t size;

    if (frm_hdr->partition_length_ < kTrailerSize ||
        ext_buf[kTrailerSize - 1] != kTrailerMarker) {
      return VP8SetError(dec, VP8_STATUS_BITSTREAM_ERROR,
                         "RIFF: Inconsistent extra information.");
    }

    // Layer
    size = (ext_buf[0] << 0) | (ext_buf[1] << 8) | (ext_buf[2] << 16);
    dec->layer_data_size_ = size;
    dec->layer_data_ = NULL;  // will be set later
    dec->layer_colorspace_ = ext_buf[3];
  }
#endif

  // sanitized state
  dec->ready_ = 1;
  return 1;
}

//------------------------------------------------------------------------------
// Residual decoding (Paragraph 13.2 / 13.3)

static const uint8_t kBands[16 + 1] = {
  0, 1, 2, 3, 6, 4, 5, 6, 6, 6, 6, 6, 6, 6, 6, 7,
  0  // extra entry as sentinel
};

static const uint8_t kCat3[] = { 173, 148, 140, 0 };
static const uint8_t kCat4[] = { 176, 155, 140, 135, 0 };
static const uint8_t kCat5[] = { 180, 157, 141, 134, 130, 0 };
static const uint8_t kCat6[] =
  { 254, 254, 243, 230, 196, 177, 153, 140, 133, 130, 129, 0 };
static const uint8_t* const kCat3456[] = { kCat3, kCat4, kCat5, kCat6 };
static const uint8_t kZigzag[16] = {
  0, 1, 4, 8,  5, 2, 3, 6,  9, 12, 13, 10,  7, 11, 14, 15
};

typedef const uint8_t (*ProbaArray)[NUM_CTX][NUM_PROBAS];  // for const-casting

// Returns the position of the last non-zero coeff plus one
// (and 0 if there's no coeff at all)
static int GetCoeffs(VP8BitReader* const br, ProbaArray prob,
                     int ctx, const quant_t dq, int n, int16_t* out) {
  // n is either 0 or 1 here. kBands[n] is not necessary for extracting '*p'.
  const uint8_t* p = prob[n][ctx];
  if (!VP8GetBit(br, p[0])) {   // first EOB is more a 'CBP' bit.
    return 0;
  }
  while (1) {
    ++n;
    if (!VP8GetBit(br, p[1])) {
      p = prob[kBands[n]][0];
    } else {  // non zero coeff
      int v, j;
      if (!VP8GetBit(br, p[2])) {
        p = prob[kBands[n]][1];
        v = 1;
      } else {
        if (!VP8GetBit(br, p[3])) {
          if (!VP8GetBit(br, p[4])) {
            v = 2;
          } else {
            v = 3 + VP8GetBit(br, p[5]);
          }
        } else {
          if (!VP8GetBit(br, p[6])) {
            if (!VP8GetBit(br, p[7])) {
              v = 5 + VP8GetBit(br, 159);
            } else {
              v = 7 + 2 * VP8GetBit(br, 165);
              v += VP8GetBit(br, 145);
            }
          } else {
            const uint8_t* tab;
            const int bit1 = VP8GetBit(br, p[8]);
            const int bit0 = VP8GetBit(br, p[9 + bit1]);
            const int cat = 2 * bit1 + bit0;
            v = 0;
            for (tab = kCat3456[cat]; *tab; ++tab) {
              v += v + VP8GetBit(br, *tab);
            }
            v += 3 + (8 << cat);
          }
        }
        p = prob[kBands[n]][2];
      }
      j = kZigzag[n - 1];
      out[j] = VP8GetSigned(br, v) * dq[j > 0];
      if (n == 16 || !VP8GetBit(br, p[0])) {   // EOB
        return n;
      }
    }
    if (n == 16) {
      return 16;
    }
  }
}

// Alias-safe way of converting 4bytes to 32bits.
typedef union {
  uint8_t  i8[4];
  uint32_t i32;
} PackedNz;

// Table to unpack four bits into four bytes
static const PackedNz kUnpackTab[16] = {
  {{0, 0, 0, 0}},  {{1, 0, 0, 0}},  {{0, 1, 0, 0}},  {{1, 1, 0, 0}},
  {{0, 0, 1, 0}},  {{1, 0, 1, 0}},  {{0, 1, 1, 0}},  {{1, 1, 1, 0}},
  {{0, 0, 0, 1}},  {{1, 0, 0, 1}},  {{0, 1, 0, 1}},  {{1, 1, 0, 1}},
  {{0, 0, 1, 1}},  {{1, 0, 1, 1}},  {{0, 1, 1, 1}},  {{1, 1, 1, 1}} };

// Macro to pack four LSB of four bytes into four bits.
#if defined(__PPC__) || defined(_M_PPC) || defined(_ARCH_PPC) || \
    defined(__BIG_ENDIAN__)
#define PACK_CST 0x08040201U
#else
#define PACK_CST 0x01020408U
#endif
#define PACK(X, S) ((((X).i32 * PACK_CST) & 0xff000000) >> (S))

static void ParseResiduals(VP8Decoder* const dec,
                           VP8MB* const mb, VP8BitReader* const token_br) {
  int out_t_nz, out_l_nz, first;
  ProbaArray ac_prob;
  const VP8QuantMatrix* q = &dec->dqm_[dec->segment_];
  int16_t* dst = dec->coeffs_;
  VP8MB* const left_mb = dec->mb_info_ - 1;
  PackedNz nz_ac, nz_dc;
  PackedNz tnz, lnz;
  uint32_t non_zero_ac = 0;
  uint32_t non_zero_dc = 0;
  int x, y, ch;

  nz_dc.i32 = nz_ac.i32 = 0;
  memset(dst, 0, 384 * sizeof(*dst));
  if (!dec->is_i4x4_) {    // parse DC
    int16_t dc[16] = { 0 };
    const int ctx = mb->dc_nz_ + left_mb->dc_nz_;
    mb->dc_nz_ = left_mb->dc_nz_ =
        (GetCoeffs(token_br, (ProbaArray)dec->proba_.coeffs_[1],
                   ctx, q->y2_mat_, 0, dc) > 0);
    first = 1;
    ac_prob = (ProbaArray)dec->proba_.coeffs_[0];
    VP8TransformWHT(dc, dst);
  } else {
    first = 0;
    ac_prob = (ProbaArray)dec->proba_.coeffs_[3];
  }

  tnz = kUnpackTab[mb->nz_ & 0xf];
  lnz = kUnpackTab[left_mb->nz_ & 0xf];
  for (y = 0; y < 4; ++y) {
    int l = lnz.i8[y];
    for (x = 0; x < 4; ++x) {
      const int ctx = l + tnz.i8[x];
      const int nz = GetCoeffs(token_br, ac_prob, ctx,
                               q->y1_mat_, first, dst);
      tnz.i8[x] = l = (nz > 0);
      nz_dc.i8[x] = (dst[0] != 0);
      nz_ac.i8[x] = (nz > 1);
      dst += 16;
    }
    lnz.i8[y] = l;
    non_zero_dc |= PACK(nz_dc, 24 - y * 4);
    non_zero_ac |= PACK(nz_ac, 24 - y * 4);
  }
  out_t_nz = PACK(tnz, 24);
  out_l_nz = PACK(lnz, 24);

  tnz = kUnpackTab[mb->nz_ >> 4];
  lnz = kUnpackTab[left_mb->nz_ >> 4];
  for (ch = 0; ch < 4; ch += 2) {
    for (y = 0; y < 2; ++y) {
      int l = lnz.i8[ch + y];
      for (x = 0; x < 2; ++x) {
        const int ctx = l + tnz.i8[ch + x];
        const int nz =
            GetCoeffs(token_br, (ProbaArray)dec->proba_.coeffs_[2],
                      ctx, q->uv_mat_, 0, dst);
        tnz.i8[ch + x] = l = (nz > 0);
        nz_dc.i8[y * 2 + x] = (dst[0] != 0);
        nz_ac.i8[y * 2 + x] = (nz > 1);
        dst += 16;
      }
      lnz.i8[ch + y] = l;
    }
    non_zero_dc |= PACK(nz_dc, 8 - ch * 2);
    non_zero_ac |= PACK(nz_ac, 8 - ch * 2);
  }
  out_t_nz |= PACK(tnz, 20);
  out_l_nz |= PACK(lnz, 20);
  mb->nz_ = out_t_nz;
  left_mb->nz_ = out_l_nz;

  dec->non_zero_ac_ = non_zero_ac;
  dec->non_zero_ = non_zero_ac | non_zero_dc;
  mb->skip_ = !dec->non_zero_;
}
#undef PACK

//------------------------------------------------------------------------------
// Main loop

int VP8DecodeMB(VP8Decoder* const dec, VP8BitReader* const token_br) {
  VP8BitReader* const br = &dec->br_;
  VP8MB* const left = dec->mb_info_ - 1;
  VP8MB* const info = dec->mb_info_ + dec->mb_x_;

  // Note: we don't save segment map (yet), as we don't expect
  // to decode more than 1 keyframe.
  if (dec->segment_hdr_.update_map_) {
    // Hardcoded tree parsing
    dec->segment_ = !VP8GetBit(br, dec->proba_.segments_[0]) ?
        VP8GetBit(br, dec->proba_.segments_[1]) :
        2 + VP8GetBit(br, dec->proba_.segments_[2]);
  }
  info->skip_ = dec->use_skip_proba_ ? VP8GetBit(br, dec->skip_p_) : 0;

  VP8ParseIntraMode(br, dec);
  if (br->eof_) {
    return 0;
  }

  if (!info->skip_) {
    ParseResiduals(dec, info, token_br);
  } else {
    left->nz_ = info->nz_ = 0;
    if (!dec->is_i4x4_) {
      left->dc_nz_ = info->dc_nz_ = 0;
    }
    dec->non_zero_ = 0;
    dec->non_zero_ac_ = 0;
  }

  return (!token_br->eof_);
}

void VP8InitScanline(VP8Decoder* const dec) {
  VP8MB* const left = dec->mb_info_ - 1;
  left->nz_ = 0;
  left->dc_nz_ = 0;
  memset(dec->intra_l_, B_DC_PRED, sizeof(dec->intra_l_));
  dec->filter_row_ =
    (dec->filter_type_ > 0) &&
    (dec->mb_y_ >= dec->tl_mb_y_) && (dec->mb_y_ <= dec->br_mb_y_);
}

static int ParseFrame(VP8Decoder* const dec, VP8Io* io) {
  for (dec->mb_y_ = 0; dec->mb_y_ < dec->br_mb_y_; ++dec->mb_y_) {
    VP8BitReader* const token_br =
        &dec->parts_[dec->mb_y_ & (dec->num_parts_ - 1)];
    VP8InitScanline(dec);
    for (dec->mb_x_ = 0; dec->mb_x_ < dec->mb_w_;  dec->mb_x_++) {
      if (!VP8DecodeMB(dec, token_br)) {
        return VP8SetError(dec, VP8_STATUS_NOT_ENOUGH_DATA,
                           "Premature end-of-file encountered.");
      }
      VP8ReconstructBlock(dec);

      // Store data and save block's filtering params
      VP8StoreBlock(dec);
    }
    if (!VP8ProcessRow(dec, io)) {
      return VP8SetError(dec, VP8_STATUS_USER_ABORT, "Output aborted.");
    }
  }
  if (dec->use_threads_ && !WebPWorkerSync(&dec->worker_)) {
    return 0;
  }

  // Finish
#ifndef ONLY_KEYFRAME_CODE
  if (!dec->update_proba_) {
    dec->proba_ = dec->proba_saved_;
  }
#endif

#ifdef WEBP_EXPERIMENTAL_FEATURES
  if (dec->layer_data_size_ > 0) {
    if (!VP8DecodeLayer(dec)) {
      return 0;
    }
  }
#endif

  return 1;
}

// Main entry point
int VP8Decode(VP8Decoder* const dec, VP8Io* const io) {
  int ok = 0;
  if (dec == NULL) {
    return 0;
  }
  if (io == NULL) {
    return VP8SetError(dec, VP8_STATUS_INVALID_PARAM,
                       "NULL VP8Io parameter in VP8Decode().");
  }

  if (!dec->ready_) {
    if (!VP8GetHeaders(dec, io)) {
      return 0;
    }
  }
  assert(dec->ready_);

  // Finish setting up the decoding parameter. Will call io->setup().
  ok = (VP8EnterCritical(dec, io) == VP8_STATUS_OK);
  if (ok) {   // good to go.
    // Will allocate memory and prepare everything.
    if (ok) ok = VP8InitFrame(dec, io);

    // Main decoding loop
    if (ok) ok = ParseFrame(dec, io);

    // Exit.
    ok &= VP8ExitCritical(dec, io);
  }

  if (!ok) {
    VP8Clear(dec);
    return 0;
  }

  dec->ready_ = 0;
  return ok;
}

void VP8Clear(VP8Decoder* const dec) {
  if (dec == NULL) {
    return;
  }
  if (dec->use_threads_) {
    WebPWorkerEnd(&dec->worker_);
  }
  if (dec->mem_) {
    free(dec->mem_);
  }
  dec->mem_ = NULL;
  dec->mem_size_ = 0;
  memset(&dec->br_, 0, sizeof(dec->br_));
  dec->ready_ = 0;
}

//------------------------------------------------------------------------------

#if defined(__cplusplus) || defined(c_plusplus)
}    // extern "C"
#endif
