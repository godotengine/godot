// Copyright 2010 Google Inc. All Rights Reserved.
//
// This code is licensed under the same terms as WebM:
//  Software License Agreement:  http://www.webmproject.org/license/software/
//  Additional IP Rights Grant:  http://www.webmproject.org/license/additional/
// -----------------------------------------------------------------------------
//
// Frame-reconstruction function. Memory allocation.
//
// Author: Skal (pascal.massimino@gmail.com)

#include <stdlib.h>
#include "./vp8i.h"
#include "../utils/utils.h"

#if defined(__cplusplus) || defined(c_plusplus)
extern "C" {
#endif

#define ALIGN_MASK (32 - 1)

//------------------------------------------------------------------------------
// Filtering

// kFilterExtraRows[] = How many extra lines are needed on the MB boundary
// for caching, given a filtering level.
// Simple filter:  up to 2 luma samples are read and 1 is written.
// Complex filter: up to 4 luma samples are read and 3 are written. Same for
//                 U/V, so it's 8 samples total (because of the 2x upsampling).
static const uint8_t kFilterExtraRows[3] = { 0, 2, 8 };

static WEBP_INLINE int hev_thresh_from_level(int level, int keyframe) {
  if (keyframe) {
    return (level >= 40) ? 2 : (level >= 15) ? 1 : 0;
  } else {
    return (level >= 40) ? 3 : (level >= 20) ? 2 : (level >= 15) ? 1 : 0;
  }
}

static void DoFilter(const VP8Decoder* const dec, int mb_x, int mb_y) {
  const VP8ThreadContext* const ctx = &dec->thread_ctx_;
  const int y_bps = dec->cache_y_stride_;
  VP8FInfo* const f_info = ctx->f_info_ + mb_x;
  uint8_t* const y_dst = dec->cache_y_ + ctx->id_ * 16 * y_bps + mb_x * 16;
  const int level = f_info->f_level_;
  const int ilevel = f_info->f_ilevel_;
  const int limit = 2 * level + ilevel;
  if (level == 0) {
    return;
  }
  if (dec->filter_type_ == 1) {   // simple
    if (mb_x > 0) {
      VP8SimpleHFilter16(y_dst, y_bps, limit + 4);
    }
    if (f_info->f_inner_) {
      VP8SimpleHFilter16i(y_dst, y_bps, limit);
    }
    if (mb_y > 0) {
      VP8SimpleVFilter16(y_dst, y_bps, limit + 4);
    }
    if (f_info->f_inner_) {
      VP8SimpleVFilter16i(y_dst, y_bps, limit);
    }
  } else {    // complex
    const int uv_bps = dec->cache_uv_stride_;
    uint8_t* const u_dst = dec->cache_u_ + ctx->id_ * 8 * uv_bps + mb_x * 8;
    uint8_t* const v_dst = dec->cache_v_ + ctx->id_ * 8 * uv_bps + mb_x * 8;
    const int hev_thresh =
        hev_thresh_from_level(level, dec->frm_hdr_.key_frame_);
    if (mb_x > 0) {
      VP8HFilter16(y_dst, y_bps, limit + 4, ilevel, hev_thresh);
      VP8HFilter8(u_dst, v_dst, uv_bps, limit + 4, ilevel, hev_thresh);
    }
    if (f_info->f_inner_) {
      VP8HFilter16i(y_dst, y_bps, limit, ilevel, hev_thresh);
      VP8HFilter8i(u_dst, v_dst, uv_bps, limit, ilevel, hev_thresh);
    }
    if (mb_y > 0) {
      VP8VFilter16(y_dst, y_bps, limit + 4, ilevel, hev_thresh);
      VP8VFilter8(u_dst, v_dst, uv_bps, limit + 4, ilevel, hev_thresh);
    }
    if (f_info->f_inner_) {
      VP8VFilter16i(y_dst, y_bps, limit, ilevel, hev_thresh);
      VP8VFilter8i(u_dst, v_dst, uv_bps, limit, ilevel, hev_thresh);
    }
  }
}

// Filter the decoded macroblock row (if needed)
static void FilterRow(const VP8Decoder* const dec) {
  int mb_x;
  const int mb_y = dec->thread_ctx_.mb_y_;
  assert(dec->thread_ctx_.filter_row_);
  for (mb_x = dec->tl_mb_x_; mb_x < dec->br_mb_x_; ++mb_x) {
    DoFilter(dec, mb_x, mb_y);
  }
}

//------------------------------------------------------------------------------

void VP8StoreBlock(VP8Decoder* const dec) {
  if (dec->filter_type_ > 0) {
    VP8FInfo* const info = dec->f_info_ + dec->mb_x_;
    const int skip = dec->mb_info_[dec->mb_x_].skip_;
    int level = dec->filter_levels_[dec->segment_];
    if (dec->filter_hdr_.use_lf_delta_) {
      // TODO(skal): only CURRENT is handled for now.
      level += dec->filter_hdr_.ref_lf_delta_[0];
      if (dec->is_i4x4_) {
        level += dec->filter_hdr_.mode_lf_delta_[0];
      }
    }
    level = (level < 0) ? 0 : (level > 63) ? 63 : level;
    info->f_level_ = level;

    if (dec->filter_hdr_.sharpness_ > 0) {
      if (dec->filter_hdr_.sharpness_ > 4) {
        level >>= 2;
      } else {
        level >>= 1;
      }
      if (level > 9 - dec->filter_hdr_.sharpness_) {
        level = 9 - dec->filter_hdr_.sharpness_;
      }
    }

    info->f_ilevel_ = (level < 1) ? 1 : level;
    info->f_inner_ = (!skip || dec->is_i4x4_);
  }
  {
    // Transfer samples to row cache
    int y;
    const int y_offset = dec->cache_id_ * 16 * dec->cache_y_stride_;
    const int uv_offset = dec->cache_id_ * 8 * dec->cache_uv_stride_;
    uint8_t* const ydst = dec->cache_y_ + dec->mb_x_ * 16 + y_offset;
    uint8_t* const udst = dec->cache_u_ + dec->mb_x_ * 8 + uv_offset;
    uint8_t* const vdst = dec->cache_v_ + dec->mb_x_ * 8 + uv_offset;
    for (y = 0; y < 16; ++y) {
      memcpy(ydst + y * dec->cache_y_stride_,
             dec->yuv_b_ + Y_OFF + y * BPS, 16);
    }
    for (y = 0; y < 8; ++y) {
      memcpy(udst + y * dec->cache_uv_stride_,
           dec->yuv_b_ + U_OFF + y * BPS, 8);
      memcpy(vdst + y * dec->cache_uv_stride_,
           dec->yuv_b_ + V_OFF + y * BPS, 8);
    }
  }
}

//------------------------------------------------------------------------------
// This function is called after a row of macroblocks is finished decoding.
// It also takes into account the following restrictions:
//  * In case of in-loop filtering, we must hold off sending some of the bottom
//    pixels as they are yet unfiltered. They will be when the next macroblock
//    row is decoded. Meanwhile, we must preserve them by rotating them in the
//    cache area. This doesn't hold for the very bottom row of the uncropped
//    picture of course.
//  * we must clip the remaining pixels against the cropping area. The VP8Io
//    struct must have the following fields set correctly before calling put():

#define MACROBLOCK_VPOS(mb_y)  ((mb_y) * 16)    // vertical position of a MB

// Finalize and transmit a complete row. Return false in case of user-abort.
static int FinishRow(VP8Decoder* const dec, VP8Io* const io) {
  int ok = 1;
  const VP8ThreadContext* const ctx = &dec->thread_ctx_;
  const int extra_y_rows = kFilterExtraRows[dec->filter_type_];
  const int ysize = extra_y_rows * dec->cache_y_stride_;
  const int uvsize = (extra_y_rows / 2) * dec->cache_uv_stride_;
  const int y_offset = ctx->id_ * 16 * dec->cache_y_stride_;
  const int uv_offset = ctx->id_ * 8 * dec->cache_uv_stride_;
  uint8_t* const ydst = dec->cache_y_ - ysize + y_offset;
  uint8_t* const udst = dec->cache_u_ - uvsize + uv_offset;
  uint8_t* const vdst = dec->cache_v_ - uvsize + uv_offset;
  const int first_row = (ctx->mb_y_ == 0);
  const int last_row = (ctx->mb_y_ >= dec->br_mb_y_ - 1);
  int y_start = MACROBLOCK_VPOS(ctx->mb_y_);
  int y_end = MACROBLOCK_VPOS(ctx->mb_y_ + 1);

  if (ctx->filter_row_) {
    FilterRow(dec);
  }

  if (io->put) {
    if (!first_row) {
      y_start -= extra_y_rows;
      io->y = ydst;
      io->u = udst;
      io->v = vdst;
    } else {
      io->y = dec->cache_y_ + y_offset;
      io->u = dec->cache_u_ + uv_offset;
      io->v = dec->cache_v_ + uv_offset;
    }

    if (!last_row) {
      y_end -= extra_y_rows;
    }
    if (y_end > io->crop_bottom) {
      y_end = io->crop_bottom;    // make sure we don't overflow on last row.
    }
    io->a = NULL;
    if (dec->alpha_data_ != NULL && y_start < y_end) {
      // TODO(skal): several things to correct here:
      // * testing presence of alpha with dec->alpha_data_ is not a good idea
      // * we're actually decompressing the full plane only once. It should be
      //   more obvious from signature.
      // * we could free alpha_data_ right after this call, but we don't own.
      io->a = VP8DecompressAlphaRows(dec, y_start, y_end - y_start);
      if (io->a == NULL) {
        return VP8SetError(dec, VP8_STATUS_BITSTREAM_ERROR,
                           "Could not decode alpha data.");
      }
    }
    if (y_start < io->crop_top) {
      const int delta_y = io->crop_top - y_start;
      y_start = io->crop_top;
      assert(!(delta_y & 1));
      io->y += dec->cache_y_stride_ * delta_y;
      io->u += dec->cache_uv_stride_ * (delta_y >> 1);
      io->v += dec->cache_uv_stride_ * (delta_y >> 1);
      if (io->a != NULL) {
        io->a += io->width * delta_y;
      }
    }
    if (y_start < y_end) {
      io->y += io->crop_left;
      io->u += io->crop_left >> 1;
      io->v += io->crop_left >> 1;
      if (io->a != NULL) {
        io->a += io->crop_left;
      }
      io->mb_y = y_start - io->crop_top;
      io->mb_w = io->crop_right - io->crop_left;
      io->mb_h = y_end - y_start;
      ok = io->put(io);
    }
  }
  // rotate top samples if needed
  if (ctx->id_ + 1 == dec->num_caches_) {
    if (!last_row) {
      memcpy(dec->cache_y_ - ysize, ydst + 16 * dec->cache_y_stride_, ysize);
      memcpy(dec->cache_u_ - uvsize, udst + 8 * dec->cache_uv_stride_, uvsize);
      memcpy(dec->cache_v_ - uvsize, vdst + 8 * dec->cache_uv_stride_, uvsize);
    }
  }

  return ok;
}

#undef MACROBLOCK_VPOS

//------------------------------------------------------------------------------

int VP8ProcessRow(VP8Decoder* const dec, VP8Io* const io) {
  int ok = 1;
  VP8ThreadContext* const ctx = &dec->thread_ctx_;
  if (!dec->use_threads_) {
    // ctx->id_ and ctx->f_info_ are already set
    ctx->mb_y_ = dec->mb_y_;
    ctx->filter_row_ = dec->filter_row_;
    ok = FinishRow(dec, io);
  } else {
    WebPWorker* const worker = &dec->worker_;
    // Finish previous job *before* updating context
    ok &= WebPWorkerSync(worker);
    assert(worker->status_ == OK);
    if (ok) {   // spawn a new deblocking/output job
      ctx->io_ = *io;
      ctx->id_ = dec->cache_id_;
      ctx->mb_y_ = dec->mb_y_;
      ctx->filter_row_ = dec->filter_row_;
      if (ctx->filter_row_) {    // just swap filter info
        VP8FInfo* const tmp = ctx->f_info_;
        ctx->f_info_ = dec->f_info_;
        dec->f_info_ = tmp;
      }
      WebPWorkerLaunch(worker);
      if (++dec->cache_id_ == dec->num_caches_) {
        dec->cache_id_ = 0;
      }
    }
  }
  return ok;
}

//------------------------------------------------------------------------------
// Finish setting up the decoding parameter once user's setup() is called.

VP8StatusCode VP8EnterCritical(VP8Decoder* const dec, VP8Io* const io) {
  // Call setup() first. This may trigger additional decoding features on 'io'.
  // Note: Afterward, we must call teardown() not matter what.
  if (io->setup && !io->setup(io)) {
    VP8SetError(dec, VP8_STATUS_USER_ABORT, "Frame setup failed");
    return dec->status_;
  }

  // Disable filtering per user request
  if (io->bypass_filtering) {
    dec->filter_type_ = 0;
  }
  // TODO(skal): filter type / strength / sharpness forcing

  // Define the area where we can skip in-loop filtering, in case of cropping.
  //
  // 'Simple' filter reads two luma samples outside of the macroblock and
  // and filters one. It doesn't filter the chroma samples. Hence, we can
  // avoid doing the in-loop filtering before crop_top/crop_left position.
  // For the 'Complex' filter, 3 samples are read and up to 3 are filtered.
  // Means: there's a dependency chain that goes all the way up to the
  // top-left corner of the picture (MB #0). We must filter all the previous
  // macroblocks.
  // TODO(skal): add an 'approximate_decoding' option, that won't produce
  // a 1:1 bit-exactness for complex filtering?
  {
    const int extra_pixels = kFilterExtraRows[dec->filter_type_];
    if (dec->filter_type_ == 2) {
      // For complex filter, we need to preserve the dependency chain.
      dec->tl_mb_x_ = 0;
      dec->tl_mb_y_ = 0;
    } else {
      // For simple filter, we can filter only the cropped region.
      // We include 'extra_pixels' on the other side of the boundary, since
      // vertical or horizontal filtering of the previous macroblock can
      // modify some abutting pixels.
      dec->tl_mb_x_ = (io->crop_left - extra_pixels) >> 4;
      dec->tl_mb_y_ = (io->crop_top - extra_pixels) >> 4;
      if (dec->tl_mb_x_ < 0) dec->tl_mb_x_ = 0;
      if (dec->tl_mb_y_ < 0) dec->tl_mb_y_ = 0;
    }
    // We need some 'extra' pixels on the right/bottom.
    dec->br_mb_y_ = (io->crop_bottom + 15 + extra_pixels) >> 4;
    dec->br_mb_x_ = (io->crop_right + 15 + extra_pixels) >> 4;
    if (dec->br_mb_x_ > dec->mb_w_) {
      dec->br_mb_x_ = dec->mb_w_;
    }
    if (dec->br_mb_y_ > dec->mb_h_) {
      dec->br_mb_y_ = dec->mb_h_;
    }
  }
  return VP8_STATUS_OK;
}

int VP8ExitCritical(VP8Decoder* const dec, VP8Io* const io) {
  int ok = 1;
  if (dec->use_threads_) {
    ok = WebPWorkerSync(&dec->worker_);
  }

  if (io->teardown) {
    io->teardown(io);
  }
  return ok;
}

//------------------------------------------------------------------------------
// For multi-threaded decoding we need to use 3 rows of 16 pixels as delay line.
//
// Reason is: the deblocking filter cannot deblock the bottom horizontal edges
// immediately, and needs to wait for first few rows of the next macroblock to
// be decoded. Hence, deblocking is lagging behind by 4 or 8 pixels (depending
// on strength).
// With two threads, the vertical positions of the rows being decoded are:
// Decode:  [ 0..15][16..31][32..47][48..63][64..79][...
// Deblock:         [ 0..11][12..27][28..43][44..59][...
// If we use two threads and two caches of 16 pixels, the sequence would be:
// Decode:  [ 0..15][16..31][ 0..15!!][16..31][ 0..15][...
// Deblock:         [ 0..11][12..27!!][-4..11][12..27][...
// The problem occurs during row [12..15!!] that both the decoding and
// deblocking threads are writing simultaneously.
// With 3 cache lines, one get a safe write pattern:
// Decode:  [ 0..15][16..31][32..47][ 0..15][16..31][32..47][0..
// Deblock:         [ 0..11][12..27][28..43][-4..11][12..27][28...
// Note that multi-threaded output _without_ deblocking can make use of two
// cache lines of 16 pixels only, since there's no lagging behind. The decoding
// and output process have non-concurrent writing:
// Decode:  [ 0..15][16..31][ 0..15][16..31][...
// io->put:         [ 0..15][16..31][ 0..15][...

#define MT_CACHE_LINES 3
#define ST_CACHE_LINES 1   // 1 cache row only for single-threaded case

// Initialize multi/single-thread worker
static int InitThreadContext(VP8Decoder* const dec) {
  dec->cache_id_ = 0;
  if (dec->use_threads_) {
    WebPWorker* const worker = &dec->worker_;
    if (!WebPWorkerReset(worker)) {
      return VP8SetError(dec, VP8_STATUS_OUT_OF_MEMORY,
                         "thread initialization failed.");
    }
    worker->data1 = dec;
    worker->data2 = (void*)&dec->thread_ctx_.io_;
    worker->hook = (WebPWorkerHook)FinishRow;
    dec->num_caches_ =
      (dec->filter_type_ > 0) ? MT_CACHE_LINES : MT_CACHE_LINES - 1;
  } else {
    dec->num_caches_ = ST_CACHE_LINES;
  }
  return 1;
}

#undef MT_CACHE_LINES
#undef ST_CACHE_LINES

//------------------------------------------------------------------------------
// Memory setup

static int AllocateMemory(VP8Decoder* const dec) {
  const int num_caches = dec->num_caches_;
  const int mb_w = dec->mb_w_;
  // Note: we use 'size_t' when there's no overflow risk, uint64_t otherwise.
  const size_t intra_pred_mode_size = 4 * mb_w * sizeof(uint8_t);
  const size_t top_size = (16 + 8 + 8) * mb_w;
  const size_t mb_info_size = (mb_w + 1) * sizeof(VP8MB);
  const size_t f_info_size =
      (dec->filter_type_ > 0) ?
          mb_w * (dec->use_threads_ ? 2 : 1) * sizeof(VP8FInfo)
        : 0;
  const size_t yuv_size = YUV_SIZE * sizeof(*dec->yuv_b_);
  const size_t coeffs_size = 384 * sizeof(*dec->coeffs_);
  const size_t cache_height = (16 * num_caches
                            + kFilterExtraRows[dec->filter_type_]) * 3 / 2;
  const size_t cache_size = top_size * cache_height;
  // alpha_size is the only one that scales as width x height.
  const uint64_t alpha_size = (dec->alpha_data_ != NULL) ?
      (uint64_t)dec->pic_hdr_.width_ * dec->pic_hdr_.height_ : 0ULL;
  const uint64_t needed = (uint64_t)intra_pred_mode_size
                        + top_size + mb_info_size + f_info_size
                        + yuv_size + coeffs_size
                        + cache_size + alpha_size + ALIGN_MASK;
  uint8_t* mem;

  if (needed != (size_t)needed) return 0;  // check for overflow
  if (needed > dec->mem_size_) {
    free(dec->mem_);
    dec->mem_size_ = 0;
    dec->mem_ = WebPSafeMalloc(needed, sizeof(uint8_t));
    if (dec->mem_ == NULL) {
      return VP8SetError(dec, VP8_STATUS_OUT_OF_MEMORY,
                         "no memory during frame initialization.");
    }
    // down-cast is ok, thanks to WebPSafeAlloc() above.
    dec->mem_size_ = (size_t)needed;
  }

  mem = (uint8_t*)dec->mem_;
  dec->intra_t_ = (uint8_t*)mem;
  mem += intra_pred_mode_size;

  dec->y_t_ = (uint8_t*)mem;
  mem += 16 * mb_w;
  dec->u_t_ = (uint8_t*)mem;
  mem += 8 * mb_w;
  dec->v_t_ = (uint8_t*)mem;
  mem += 8 * mb_w;

  dec->mb_info_ = ((VP8MB*)mem) + 1;
  mem += mb_info_size;

  dec->f_info_ = f_info_size ? (VP8FInfo*)mem : NULL;
  mem += f_info_size;
  dec->thread_ctx_.id_ = 0;
  dec->thread_ctx_.f_info_ = dec->f_info_;
  if (dec->use_threads_) {
    // secondary cache line. The deblocking process need to make use of the
    // filtering strength from previous macroblock row, while the new ones
    // are being decoded in parallel. We'll just swap the pointers.
    dec->thread_ctx_.f_info_ += mb_w;
  }

  mem = (uint8_t*)((uintptr_t)(mem + ALIGN_MASK) & ~ALIGN_MASK);
  assert((yuv_size & ALIGN_MASK) == 0);
  dec->yuv_b_ = (uint8_t*)mem;
  mem += yuv_size;

  dec->coeffs_ = (int16_t*)mem;
  mem += coeffs_size;

  dec->cache_y_stride_ = 16 * mb_w;
  dec->cache_uv_stride_ = 8 * mb_w;
  {
    const int extra_rows = kFilterExtraRows[dec->filter_type_];
    const int extra_y = extra_rows * dec->cache_y_stride_;
    const int extra_uv = (extra_rows / 2) * dec->cache_uv_stride_;
    dec->cache_y_ = ((uint8_t*)mem) + extra_y;
    dec->cache_u_ = dec->cache_y_
                  + 16 * num_caches * dec->cache_y_stride_ + extra_uv;
    dec->cache_v_ = dec->cache_u_
                  + 8 * num_caches * dec->cache_uv_stride_ + extra_uv;
    dec->cache_id_ = 0;
  }
  mem += cache_size;

  // alpha plane
  dec->alpha_plane_ = alpha_size ? (uint8_t*)mem : NULL;
  mem += alpha_size;

  // note: left-info is initialized once for all.
  memset(dec->mb_info_ - 1, 0, mb_info_size);

  // initialize top
  memset(dec->intra_t_, B_DC_PRED, intra_pred_mode_size);

  return 1;
}

static void InitIo(VP8Decoder* const dec, VP8Io* io) {
  // prepare 'io'
  io->mb_y = 0;
  io->y = dec->cache_y_;
  io->u = dec->cache_u_;
  io->v = dec->cache_v_;
  io->y_stride = dec->cache_y_stride_;
  io->uv_stride = dec->cache_uv_stride_;
  io->a = NULL;
}

int VP8InitFrame(VP8Decoder* const dec, VP8Io* io) {
  if (!InitThreadContext(dec)) return 0;  // call first. Sets dec->num_caches_.
  if (!AllocateMemory(dec)) return 0;
  InitIo(dec, io);
  VP8DspInit();  // Init critical function pointers and look-up tables.
  return 1;
}

//------------------------------------------------------------------------------
// Main reconstruction function.

static const int kScan[16] = {
  0 +  0 * BPS,  4 +  0 * BPS, 8 +  0 * BPS, 12 +  0 * BPS,
  0 +  4 * BPS,  4 +  4 * BPS, 8 +  4 * BPS, 12 +  4 * BPS,
  0 +  8 * BPS,  4 +  8 * BPS, 8 +  8 * BPS, 12 +  8 * BPS,
  0 + 12 * BPS,  4 + 12 * BPS, 8 + 12 * BPS, 12 + 12 * BPS
};

static WEBP_INLINE int CheckMode(VP8Decoder* const dec, int mode) {
  if (mode == B_DC_PRED) {
    if (dec->mb_x_ == 0) {
      return (dec->mb_y_ == 0) ? B_DC_PRED_NOTOPLEFT : B_DC_PRED_NOLEFT;
    } else {
      return (dec->mb_y_ == 0) ? B_DC_PRED_NOTOP : B_DC_PRED;
    }
  }
  return mode;
}

static WEBP_INLINE void Copy32b(uint8_t* dst, uint8_t* src) {
  *(uint32_t*)dst = *(uint32_t*)src;
}

void VP8ReconstructBlock(VP8Decoder* const dec) {
  uint8_t* const y_dst = dec->yuv_b_ + Y_OFF;
  uint8_t* const u_dst = dec->yuv_b_ + U_OFF;
  uint8_t* const v_dst = dec->yuv_b_ + V_OFF;

  // Rotate in the left samples from previously decoded block. We move four
  // pixels at a time for alignment reason, and because of in-loop filter.
  if (dec->mb_x_ > 0) {
    int j;
    for (j = -1; j < 16; ++j) {
      Copy32b(&y_dst[j * BPS - 4], &y_dst[j * BPS + 12]);
    }
    for (j = -1; j < 8; ++j) {
      Copy32b(&u_dst[j * BPS - 4], &u_dst[j * BPS + 4]);
      Copy32b(&v_dst[j * BPS - 4], &v_dst[j * BPS + 4]);
    }
  } else {
    int j;
    for (j = 0; j < 16; ++j) {
      y_dst[j * BPS - 1] = 129;
    }
    for (j = 0; j < 8; ++j) {
      u_dst[j * BPS - 1] = 129;
      v_dst[j * BPS - 1] = 129;
    }
    // Init top-left sample on left column too
    if (dec->mb_y_ > 0) {
      y_dst[-1 - BPS] = u_dst[-1 - BPS] = v_dst[-1 - BPS] = 129;
    }
  }
  {
    // bring top samples into the cache
    uint8_t* const top_y = dec->y_t_ + dec->mb_x_ * 16;
    uint8_t* const top_u = dec->u_t_ + dec->mb_x_ * 8;
    uint8_t* const top_v = dec->v_t_ + dec->mb_x_ * 8;
    const int16_t* coeffs = dec->coeffs_;
    int n;

    if (dec->mb_y_ > 0) {
      memcpy(y_dst - BPS, top_y, 16);
      memcpy(u_dst - BPS, top_u, 8);
      memcpy(v_dst - BPS, top_v, 8);
    } else if (dec->mb_x_ == 0) {
      // we only need to do this init once at block (0,0).
      // Afterward, it remains valid for the whole topmost row.
      memset(y_dst - BPS - 1, 127, 16 + 4 + 1);
      memset(u_dst - BPS - 1, 127, 8 + 1);
      memset(v_dst - BPS - 1, 127, 8 + 1);
    }

    // predict and add residuals

    if (dec->is_i4x4_) {   // 4x4
      uint32_t* const top_right = (uint32_t*)(y_dst - BPS + 16);

      if (dec->mb_y_ > 0) {
        if (dec->mb_x_ >= dec->mb_w_ - 1) {    // on rightmost border
          top_right[0] = top_y[15] * 0x01010101u;
        } else {
          memcpy(top_right, top_y + 16, sizeof(*top_right));
        }
      }
      // replicate the top-right pixels below
      top_right[BPS] = top_right[2 * BPS] = top_right[3 * BPS] = top_right[0];

      // predict and add residues for all 4x4 blocks in turn.
      for (n = 0; n < 16; n++) {
        uint8_t* const dst = y_dst + kScan[n];
        VP8PredLuma4[dec->imodes_[n]](dst);
        if (dec->non_zero_ac_ & (1 << n)) {
          VP8Transform(coeffs + n * 16, dst, 0);
        } else if (dec->non_zero_ & (1 << n)) {  // only DC is present
          VP8TransformDC(coeffs + n * 16, dst);
        }
      }
    } else {    // 16x16
      const int pred_func = CheckMode(dec, dec->imodes_[0]);
      VP8PredLuma16[pred_func](y_dst);
      if (dec->non_zero_) {
        for (n = 0; n < 16; n++) {
          uint8_t* const dst = y_dst + kScan[n];
          if (dec->non_zero_ac_ & (1 << n)) {
            VP8Transform(coeffs + n * 16, dst, 0);
          } else if (dec->non_zero_ & (1 << n)) {  // only DC is present
            VP8TransformDC(coeffs + n * 16, dst);
          }
        }
      }
    }
    {
      // Chroma
      const int pred_func = CheckMode(dec, dec->uvmode_);
      VP8PredChroma8[pred_func](u_dst);
      VP8PredChroma8[pred_func](v_dst);

      if (dec->non_zero_ & 0x0f0000) {   // chroma-U
        const int16_t* const u_coeffs = dec->coeffs_ + 16 * 16;
        if (dec->non_zero_ac_ & 0x0f0000) {
          VP8TransformUV(u_coeffs, u_dst);
        } else {
          VP8TransformDCUV(u_coeffs, u_dst);
        }
      }
      if (dec->non_zero_ & 0xf00000) {   // chroma-V
        const int16_t* const v_coeffs = dec->coeffs_ + 20 * 16;
        if (dec->non_zero_ac_ & 0xf00000) {
          VP8TransformUV(v_coeffs, v_dst);
        } else {
          VP8TransformDCUV(v_coeffs, v_dst);
        }
      }

      // stash away top samples for next block
      if (dec->mb_y_ < dec->mb_h_ - 1) {
        memcpy(top_y, y_dst + 15 * BPS, 16);
        memcpy(top_u, u_dst +  7 * BPS,  8);
        memcpy(top_v, v_dst +  7 * BPS,  8);
      }
    }
  }
}

//------------------------------------------------------------------------------

#if defined(__cplusplus) || defined(c_plusplus)
}    // extern "C"
#endif
