// Copyright 2011 Google Inc. All Rights Reserved.
//
// Use of this source code is governed by a BSD-style license
// that can be found in the COPYING file in the root of the source
// tree. An additional intellectual property rights grant can be found
// in the file PATENTS. All contributing project authors may
// be found in the AUTHORS file in the root of the source tree.
// -----------------------------------------------------------------------------
//
// WebP encoder: main entry point
//
// Author: Skal (pascal.massimino@gmail.com)

#include <assert.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "./vp8enci.h"
#include "./vp8li.h"
#include "../utils/utils.h"

// #define PRINT_MEMORY_INFO

#ifdef PRINT_MEMORY_INFO
#include <stdio.h>
#endif

//------------------------------------------------------------------------------

int WebPGetEncoderVersion(void) {
  return (ENC_MAJ_VERSION << 16) | (ENC_MIN_VERSION << 8) | ENC_REV_VERSION;
}

//------------------------------------------------------------------------------
// WebPPicture
//------------------------------------------------------------------------------

static int DummyWriter(const uint8_t* data, size_t data_size,
                       const WebPPicture* const picture) {
  // The following are to prevent 'unused variable' error message.
  (void)data;
  (void)data_size;
  (void)picture;
  return 1;
}

int WebPPictureInitInternal(WebPPicture* picture, int version) {
  if (WEBP_ABI_IS_INCOMPATIBLE(version, WEBP_ENCODER_ABI_VERSION)) {
    return 0;   // caller/system version mismatch!
  }
  if (picture != NULL) {
    memset(picture, 0, sizeof(*picture));
    picture->writer = DummyWriter;
    WebPEncodingSetError(picture, VP8_ENC_OK);
  }
  return 1;
}

//------------------------------------------------------------------------------
// VP8Encoder
//------------------------------------------------------------------------------

static void ResetSegmentHeader(VP8Encoder* const enc) {
  VP8SegmentHeader* const hdr = &enc->segment_hdr_;
  hdr->num_segments_ = enc->config_->segments;
  hdr->update_map_  = (hdr->num_segments_ > 1);
  hdr->size_ = 0;
}

static void ResetFilterHeader(VP8Encoder* const enc) {
  VP8FilterHeader* const hdr = &enc->filter_hdr_;
  hdr->simple_ = 1;
  hdr->level_ = 0;
  hdr->sharpness_ = 0;
  hdr->i4x4_lf_delta_ = 0;
}

static void ResetBoundaryPredictions(VP8Encoder* const enc) {
  // init boundary values once for all
  // Note: actually, initializing the preds_[] is only needed for intra4.
  int i;
  uint8_t* const top = enc->preds_ - enc->preds_w_;
  uint8_t* const left = enc->preds_ - 1;
  for (i = -1; i < 4 * enc->mb_w_; ++i) {
    top[i] = B_DC_PRED;
  }
  for (i = 0; i < 4 * enc->mb_h_; ++i) {
    left[i * enc->preds_w_] = B_DC_PRED;
  }
  enc->nz_[-1] = 0;   // constant
}

// Mapping from config->method_ to coding tools used.
//-------------------+---+---+---+---+---+---+---+
//   Method          | 0 | 1 | 2 | 3 |(4)| 5 | 6 |
//-------------------+---+---+---+---+---+---+---+
// fast probe        | x |   |   | x |   |   |   |
//-------------------+---+---+---+---+---+---+---+
// dynamic proba     | ~ | x | x | x | x | x | x |
//-------------------+---+---+---+---+---+---+---+
// fast mode analysis|   |   |   |   | x | x | x |
//-------------------+---+---+---+---+---+---+---+
// basic rd-opt      |   |   |   | x | x | x | x |
//-------------------+---+---+---+---+---+---+---+
// disto-score i4/16 |   |   | x |   |   |   |   |
//-------------------+---+---+---+---+---+---+---+
// rd-opt i4/16      |   |   | ~ | x | x | x | x |
//-------------------+---+---+---+---+---+---+---+
// token buffer (opt)|   |   |   | x | x | x | x |
//-------------------+---+---+---+---+---+---+---+
// Trellis           |   |   |   |   |   | x |Ful|
//-------------------+---+---+---+---+---+---+---+
// full-SNS          |   |   |   |   | x | x | x |
//-------------------+---+---+---+---+---+---+---+

static void MapConfigToTools(VP8Encoder* const enc) {
  const WebPConfig* const config = enc->config_;
  const int method = config->method;
  const int limit = 100 - config->partition_limit;
  enc->method_ = method;
  enc->rd_opt_level_ = (method >= 6) ? RD_OPT_TRELLIS_ALL
                     : (method >= 5) ? RD_OPT_TRELLIS
                     : (method >= 3) ? RD_OPT_BASIC
                     : RD_OPT_NONE;
  enc->max_i4_header_bits_ =
      256 * 16 * 16 *                 // upper bound: up to 16bit per 4x4 block
      (limit * limit) / (100 * 100);  // ... modulated with a quadratic curve.

  enc->thread_level_ = config->thread_level;

  enc->do_search_ = (config->target_size > 0 || config->target_PSNR > 0);
  if (!config->low_memory) {
#if !defined(DISABLE_TOKEN_BUFFER)
    enc->use_tokens_ = (enc->rd_opt_level_ >= RD_OPT_BASIC);  // need rd stats
#endif
    if (enc->use_tokens_) {
      enc->num_parts_ = 1;   // doesn't work with multi-partition
    }
  }
}

// Memory scaling with dimensions:
//  memory (bytes) ~= 2.25 * w + 0.0625 * w * h
//
// Typical memory footprint (768x510 picture)
// Memory used:
//              encoder: 33919
//          block cache: 2880
//                 info: 3072
//                preds: 24897
//          top samples: 1623
//             non-zero: 196
//             lf-stats: 2048
//                total: 68635
// Transient object sizes:
//       VP8EncIterator: 352
//         VP8ModeScore: 912
//       VP8SegmentInfo: 532
//             VP8Proba: 31032
//              LFStats: 2048
// Picture size (yuv): 589824

static VP8Encoder* InitVP8Encoder(const WebPConfig* const config,
                                  WebPPicture* const picture) {
  const int use_filter =
      (config->filter_strength > 0) || (config->autofilter > 0);
  const int mb_w = (picture->width + 15) >> 4;
  const int mb_h = (picture->height + 15) >> 4;
  const int preds_w = 4 * mb_w + 1;
  const int preds_h = 4 * mb_h + 1;
  const size_t preds_size = preds_w * preds_h * sizeof(uint8_t);
  const int top_stride = mb_w * 16;
  const size_t nz_size = (mb_w + 1) * sizeof(uint32_t) + ALIGN_CST;
  const size_t info_size = mb_w * mb_h * sizeof(VP8MBInfo);
  const size_t samples_size = 2 * top_stride * sizeof(uint8_t)  // top-luma/u/v
                            + ALIGN_CST;                        // align all
  const size_t lf_stats_size =
      config->autofilter ? sizeof(LFStats) + ALIGN_CST : 0;
  VP8Encoder* enc;
  uint8_t* mem;
  const uint64_t size = (uint64_t)sizeof(VP8Encoder)   // main struct
                      + ALIGN_CST                      // cache alignment
                      + info_size                      // modes info
                      + preds_size                     // prediction modes
                      + samples_size                   // top/left samples
                      + nz_size                        // coeff context bits
                      + lf_stats_size;                 // autofilter stats

#ifdef PRINT_MEMORY_INFO
  printf("===================================\n");
  printf("Memory used:\n"
         "             encoder: %ld\n"
         "                info: %ld\n"
         "               preds: %ld\n"
         "         top samples: %ld\n"
         "            non-zero: %ld\n"
         "            lf-stats: %ld\n"
         "               total: %ld\n",
         sizeof(VP8Encoder) + ALIGN_CST, info_size,
         preds_size, samples_size, nz_size, lf_stats_size, size);
  printf("Transient object sizes:\n"
         "      VP8EncIterator: %ld\n"
         "        VP8ModeScore: %ld\n"
         "      VP8SegmentInfo: %ld\n"
         "            VP8Proba: %ld\n"
         "             LFStats: %ld\n",
         sizeof(VP8EncIterator), sizeof(VP8ModeScore),
         sizeof(VP8SegmentInfo), sizeof(VP8Proba),
         sizeof(LFStats));
  printf("Picture size (yuv): %ld\n",
         mb_w * mb_h * 384 * sizeof(uint8_t));
  printf("===================================\n");
#endif
  mem = (uint8_t*)WebPSafeMalloc(size, sizeof(*mem));
  if (mem == NULL) {
    WebPEncodingSetError(picture, VP8_ENC_ERROR_OUT_OF_MEMORY);
    return NULL;
  }
  enc = (VP8Encoder*)mem;
  mem = (uint8_t*)DO_ALIGN(mem + sizeof(*enc));
  memset(enc, 0, sizeof(*enc));
  enc->num_parts_ = 1 << config->partitions;
  enc->mb_w_ = mb_w;
  enc->mb_h_ = mb_h;
  enc->preds_w_ = preds_w;
  enc->mb_info_ = (VP8MBInfo*)mem;
  mem += info_size;
  enc->preds_ = ((uint8_t*)mem) + 1 + enc->preds_w_;
  mem += preds_w * preds_h * sizeof(uint8_t);
  enc->nz_ = 1 + (uint32_t*)DO_ALIGN(mem);
  mem += nz_size;
  enc->lf_stats_ = lf_stats_size ? (LFStats*)DO_ALIGN(mem) : NULL;
  mem += lf_stats_size;

  // top samples (all 16-aligned)
  mem = (uint8_t*)DO_ALIGN(mem);
  enc->y_top_ = (uint8_t*)mem;
  enc->uv_top_ = enc->y_top_ + top_stride;
  mem += 2 * top_stride;
  assert(mem <= (uint8_t*)enc + size);

  enc->config_ = config;
  enc->profile_ = use_filter ? ((config->filter_type == 1) ? 0 : 1) : 2;
  enc->pic_ = picture;
  enc->percent_ = 0;

  MapConfigToTools(enc);
  VP8EncDspInit();
  VP8DefaultProbas(enc);
  ResetSegmentHeader(enc);
  ResetFilterHeader(enc);
  ResetBoundaryPredictions(enc);

  VP8EncInitAlpha(enc);
#ifdef WEBP_EXPERIMENTAL_FEATURES
  VP8EncInitLayer(enc);
#endif

  VP8TBufferInit(&enc->tokens_);
  return enc;
}

static int DeleteVP8Encoder(VP8Encoder* enc) {
  int ok = 1;
  if (enc != NULL) {
    ok = VP8EncDeleteAlpha(enc);
#ifdef WEBP_EXPERIMENTAL_FEATURES
    VP8EncDeleteLayer(enc);
#endif
    VP8TBufferClear(&enc->tokens_);
    free(enc);
  }
  return ok;
}

//------------------------------------------------------------------------------

static double GetPSNR(uint64_t err, uint64_t size) {
  return (err > 0 && size > 0) ? 10. * log10(255. * 255. * size / err) : 99.;
}

static void FinalizePSNR(const VP8Encoder* const enc) {
  WebPAuxStats* stats = enc->pic_->stats;
  const uint64_t size = enc->sse_count_;
  const uint64_t* const sse = enc->sse_;
  stats->PSNR[0] = (float)GetPSNR(sse[0], size);
  stats->PSNR[1] = (float)GetPSNR(sse[1], size / 4);
  stats->PSNR[2] = (float)GetPSNR(sse[2], size / 4);
  stats->PSNR[3] = (float)GetPSNR(sse[0] + sse[1] + sse[2], size * 3 / 2);
  stats->PSNR[4] = (float)GetPSNR(sse[3], size);
}

static void StoreStats(VP8Encoder* const enc) {
  WebPAuxStats* const stats = enc->pic_->stats;
  if (stats != NULL) {
    int i, s;
    for (i = 0; i < NUM_MB_SEGMENTS; ++i) {
      stats->segment_level[i] = enc->dqm_[i].fstrength_;
      stats->segment_quant[i] = enc->dqm_[i].quant_;
      for (s = 0; s <= 2; ++s) {
        stats->residual_bytes[s][i] = enc->residual_bytes_[s][i];
      }
    }
    FinalizePSNR(enc);
    stats->coded_size = enc->coded_size_;
    for (i = 0; i < 3; ++i) {
      stats->block_count[i] = enc->block_count_[i];
    }
  }
  WebPReportProgress(enc->pic_, 100, &enc->percent_);  // done!
}

int WebPEncodingSetError(const WebPPicture* const pic,
                         WebPEncodingError error) {
  assert((int)error < VP8_ENC_ERROR_LAST);
  assert((int)error >= VP8_ENC_OK);
  ((WebPPicture*)pic)->error_code = error;
  return 0;
}

int WebPReportProgress(const WebPPicture* const pic,
                       int percent, int* const percent_store) {
  if (percent_store != NULL && percent != *percent_store) {
    *percent_store = percent;
    if (pic->progress_hook && !pic->progress_hook(percent, pic)) {
      // user abort requested
      WebPEncodingSetError(pic, VP8_ENC_ERROR_USER_ABORT);
      return 0;
    }
  }
  return 1;  // ok
}
//------------------------------------------------------------------------------

int WebPEncode(const WebPConfig* config, WebPPicture* pic) {
  int ok = 0;

  if (pic == NULL)
    return 0;
  WebPEncodingSetError(pic, VP8_ENC_OK);  // all ok so far
  if (config == NULL)  // bad params
    return WebPEncodingSetError(pic, VP8_ENC_ERROR_NULL_PARAMETER);
  if (!WebPValidateConfig(config))
    return WebPEncodingSetError(pic, VP8_ENC_ERROR_INVALID_CONFIGURATION);
  if (pic->width <= 0 || pic->height <= 0)
    return WebPEncodingSetError(pic, VP8_ENC_ERROR_BAD_DIMENSION);
  if (pic->width > WEBP_MAX_DIMENSION || pic->height > WEBP_MAX_DIMENSION)
    return WebPEncodingSetError(pic, VP8_ENC_ERROR_BAD_DIMENSION);

  if (pic->stats != NULL) memset(pic->stats, 0, sizeof(*pic->stats));

  if (!config->lossless) {
    VP8Encoder* enc = NULL;
    if (pic->y == NULL || pic->u == NULL || pic->v == NULL) {
      // Make sure we have YUVA samples.
      float dithering = 0.f;
      if (config->preprocessing & 2) {
        const float x = config->quality / 100.f;
        const float x2 = x * x;
        // slowly decreasing from max dithering at low quality (q->0)
        // to 0.5 dithering amplitude at high quality (q->100)
        dithering = 1.0f + (0.5f - 1.0f) * x2 * x2;
      }
      if (!WebPPictureARGBToYUVADithered(pic, WEBP_YUV420, dithering)) {
        return 0;
      }
    }

    enc = InitVP8Encoder(config, pic);
    if (enc == NULL) return 0;  // pic->error is already set.
    // Note: each of the tasks below account for 20% in the progress report.
    ok = VP8EncAnalyze(enc);

    // Analysis is done, proceed to actual coding.
    ok = ok && VP8EncStartAlpha(enc);   // possibly done in parallel
    if (!enc->use_tokens_) {
      ok = ok && VP8EncLoop(enc);
    } else {
      ok = ok && VP8EncTokenLoop(enc);
    }
    ok = ok && VP8EncFinishAlpha(enc);
#ifdef WEBP_EXPERIMENTAL_FEATURES
    ok = ok && VP8EncFinishLayer(enc);
#endif

    ok = ok && VP8EncWrite(enc);
    StoreStats(enc);
    if (!ok) {
      VP8EncFreeBitWriters(enc);
    }
    ok &= DeleteVP8Encoder(enc);  // must always be called, even if !ok
  } else {
    // Make sure we have ARGB samples.
    if (pic->argb == NULL && !WebPPictureYUVAToARGB(pic)) {
      return 0;
    }

    ok = VP8LEncodeImage(config, pic);  // Sets pic->error in case of problem.
  }

  return ok;
}

