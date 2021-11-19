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

#include "src/enc/cost_enc.h"
#include "src/enc/vp8i_enc.h"
#include "src/enc/vp8li_enc.h"
#include "src/utils/utils.h"

// #define PRINT_MEMORY_INFO

#ifdef PRINT_MEMORY_INFO
#include <stdio.h>
#endif

//------------------------------------------------------------------------------

int WebPGetEncoderVersion(void) {
  return (ENC_MAJ_VERSION << 16) | (ENC_MIN_VERSION << 8) | ENC_REV_VERSION;
}

//------------------------------------------------------------------------------
// VP8Encoder
//------------------------------------------------------------------------------

static void ResetSegmentHeader(VP8Encoder* const enc) {
  VP8EncSegmentHeader* const hdr = &enc->segment_hdr_;
  hdr->num_segments_ = enc->config_->segments;
  hdr->update_map_  = (hdr->num_segments_ > 1);
  hdr->size_ = 0;
}

static void ResetFilterHeader(VP8Encoder* const enc) {
  VP8EncFilterHeader* const hdr = &enc->filter_hdr_;
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
// fast mode analysis|[x]|[x]|   |   | x | x | x |
//-------------------+---+---+---+---+---+---+---+
// basic rd-opt      |   |   |   | x | x | x | x |
//-------------------+---+---+---+---+---+---+---+
// disto-refine i4/16| x | x | x |   |   |   |   |
//-------------------+---+---+---+---+---+---+---+
// disto-refine uv   |   | x | x |   |   |   |   |
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

  // partition0 = 512k max.
  enc->mb_header_limit_ =
      (score_t)256 * 510 * 8 * 1024 / (enc->mb_w_ * enc->mb_h_);

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
// Typical memory footprint (614x440 picture)
//              encoder: 22111
//                 info: 4368
//                preds: 17741
//          top samples: 1263
//             non-zero: 175
//             lf-stats: 0
//                total: 45658
// Transient object sizes:
//       VP8EncIterator: 3360
//         VP8ModeScore: 872
//       VP8SegmentInfo: 732
//          VP8EncProba: 18352
//              LFStats: 2048
// Picture size (yuv): 419328

static VP8Encoder* InitVP8Encoder(const WebPConfig* const config,
                                  WebPPicture* const picture) {
  VP8Encoder* enc;
  const int use_filter =
      (config->filter_strength > 0) || (config->autofilter > 0);
  const int mb_w = (picture->width + 15) >> 4;
  const int mb_h = (picture->height + 15) >> 4;
  const int preds_w = 4 * mb_w + 1;
  const int preds_h = 4 * mb_h + 1;
  const size_t preds_size = preds_w * preds_h * sizeof(*enc->preds_);
  const int top_stride = mb_w * 16;
  const size_t nz_size = (mb_w + 1) * sizeof(*enc->nz_) + WEBP_ALIGN_CST;
  const size_t info_size = mb_w * mb_h * sizeof(*enc->mb_info_);
  const size_t samples_size =
      2 * top_stride * sizeof(*enc->y_top_)  // top-luma/u/v
      + WEBP_ALIGN_CST;                      // align all
  const size_t lf_stats_size =
      config->autofilter ? sizeof(*enc->lf_stats_) + WEBP_ALIGN_CST : 0;
  const size_t top_derr_size =
      (config->quality <= ERROR_DIFFUSION_QUALITY || config->pass > 1) ?
          mb_w * sizeof(*enc->top_derr_) : 0;
  uint8_t* mem;
  const uint64_t size = (uint64_t)sizeof(*enc)   // main struct
                      + WEBP_ALIGN_CST           // cache alignment
                      + info_size                // modes info
                      + preds_size               // prediction modes
                      + samples_size             // top/left samples
                      + top_derr_size            // top diffusion error
                      + nz_size                  // coeff context bits
                      + lf_stats_size;           // autofilter stats

#ifdef PRINT_MEMORY_INFO
  printf("===================================\n");
  printf("Memory used:\n"
         "             encoder: %ld\n"
         "                info: %ld\n"
         "               preds: %ld\n"
         "         top samples: %ld\n"
         "       top diffusion: %ld\n"
         "            non-zero: %ld\n"
         "            lf-stats: %ld\n"
         "               total: %ld\n",
         sizeof(*enc) + WEBP_ALIGN_CST, info_size,
         preds_size, samples_size, top_derr_size, nz_size, lf_stats_size, size);
  printf("Transient object sizes:\n"
         "      VP8EncIterator: %ld\n"
         "        VP8ModeScore: %ld\n"
         "      VP8SegmentInfo: %ld\n"
         "         VP8EncProba: %ld\n"
         "             LFStats: %ld\n",
         sizeof(VP8EncIterator), sizeof(VP8ModeScore),
         sizeof(VP8SegmentInfo), sizeof(VP8EncProba),
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
  mem = (uint8_t*)WEBP_ALIGN(mem + sizeof(*enc));
  memset(enc, 0, sizeof(*enc));
  enc->num_parts_ = 1 << config->partitions;
  enc->mb_w_ = mb_w;
  enc->mb_h_ = mb_h;
  enc->preds_w_ = preds_w;
  enc->mb_info_ = (VP8MBInfo*)mem;
  mem += info_size;
  enc->preds_ = mem + 1 + enc->preds_w_;
  mem += preds_size;
  enc->nz_ = 1 + (uint32_t*)WEBP_ALIGN(mem);
  mem += nz_size;
  enc->lf_stats_ = lf_stats_size ? (LFStats*)WEBP_ALIGN(mem) : NULL;
  mem += lf_stats_size;

  // top samples (all 16-aligned)
  mem = (uint8_t*)WEBP_ALIGN(mem);
  enc->y_top_ = mem;
  enc->uv_top_ = enc->y_top_ + top_stride;
  mem += 2 * top_stride;
  enc->top_derr_ = top_derr_size ? (DError*)mem : NULL;
  mem += top_derr_size;
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
  VP8EncDspCostInit();
  VP8EncInitAlpha(enc);

  // lower quality means smaller output -> we modulate a little the page
  // size based on quality. This is just a crude 1rst-order prediction.
  {
    const float scale = 1.f + config->quality * 5.f / 100.f;  // in [1,6]
    VP8TBufferInit(&enc->tokens_, (int)(mb_w * mb_h * 4 * scale));
  }
  return enc;
}

static int DeleteVP8Encoder(VP8Encoder* enc) {
  int ok = 1;
  if (enc != NULL) {
    ok = VP8EncDeleteAlpha(enc);
    VP8TBufferClear(&enc->tokens_);
    WebPSafeFree(enc);
  }
  return ok;
}

//------------------------------------------------------------------------------

#if !defined(WEBP_DISABLE_STATS)
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
#endif  // !defined(WEBP_DISABLE_STATS)

static void StoreStats(VP8Encoder* const enc) {
#if !defined(WEBP_DISABLE_STATS)
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
#else  // defined(WEBP_DISABLE_STATS)
  WebPReportProgress(enc->pic_, 100, &enc->percent_);  // done!
#endif  // !defined(WEBP_DISABLE_STATS)
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
  if (pic == NULL) return 0;

  WebPEncodingSetError(pic, VP8_ENC_OK);  // all ok so far
  if (config == NULL) {  // bad params
    return WebPEncodingSetError(pic, VP8_ENC_ERROR_NULL_PARAMETER);
  }
  if (!WebPValidateConfig(config)) {
    return WebPEncodingSetError(pic, VP8_ENC_ERROR_INVALID_CONFIGURATION);
  }
  if (pic->width <= 0 || pic->height <= 0) {
    return WebPEncodingSetError(pic, VP8_ENC_ERROR_BAD_DIMENSION);
  }
  if (pic->width > WEBP_MAX_DIMENSION || pic->height > WEBP_MAX_DIMENSION) {
    return WebPEncodingSetError(pic, VP8_ENC_ERROR_BAD_DIMENSION);
  }

  if (pic->stats != NULL) memset(pic->stats, 0, sizeof(*pic->stats));

  if (!config->lossless) {
    VP8Encoder* enc = NULL;

    if (pic->use_argb || pic->y == NULL || pic->u == NULL || pic->v == NULL) {
      // Make sure we have YUVA samples.
      if (config->use_sharp_yuv || (config->preprocessing & 4)) {
        if (!WebPPictureSharpARGBToYUVA(pic)) {
          return 0;
        }
      } else {
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
    }

    if (!config->exact) {
      WebPCleanupTransparentArea(pic);
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

    if (!config->exact) {
      WebPReplaceTransparentPixels(pic, 0x000000);
    }

    ok = VP8LEncodeImage(config, pic);  // Sets pic->error in case of problem.
  }

  return ok;
}
