// Copyright 2011 Google Inc. All Rights Reserved.
//
// This code is licensed under the same terms as WebM:
//  Software License Agreement:  http://www.webmproject.org/license/software/
//  Additional IP Rights Grant:  http://www.webmproject.org/license/additional/
// -----------------------------------------------------------------------------
//
// Macroblock analysis
//
// Author: Skal (pascal.massimino@gmail.com)

#include <stdlib.h>
#include <string.h>
#include <assert.h>

#include "./vp8enci.h"
#include "./cost.h"
#include "../utils/utils.h"

#if defined(__cplusplus) || defined(c_plusplus)
extern "C" {
#endif

#define MAX_ITERS_K_MEANS  6

static int ClipAlpha(int alpha) {
  return alpha < 0 ? 0 : alpha > 255 ? 255 : alpha;
}

//------------------------------------------------------------------------------
// Smooth the segment map by replacing isolated block by the majority of its
// neighbours.

static void SmoothSegmentMap(VP8Encoder* const enc) {
  int n, x, y;
  const int w = enc->mb_w_;
  const int h = enc->mb_h_;
  const int majority_cnt_3_x_3_grid = 5;
  uint8_t* const tmp = (uint8_t*)WebPSafeMalloc((uint64_t)w * h, sizeof(*tmp));
  assert((uint64_t)(w * h) == (uint64_t)w * h);   // no overflow, as per spec

  if (tmp == NULL) return;
  for (y = 1; y < h - 1; ++y) {
    for (x = 1; x < w - 1; ++x) {
      int cnt[NUM_MB_SEGMENTS] = { 0 };
      const VP8MBInfo* const mb = &enc->mb_info_[x + w * y];
      int majority_seg = mb->segment_;
      // Check the 8 neighbouring segment values.
      cnt[mb[-w - 1].segment_]++;  // top-left
      cnt[mb[-w + 0].segment_]++;  // top
      cnt[mb[-w + 1].segment_]++;  // top-right
      cnt[mb[   - 1].segment_]++;  // left
      cnt[mb[   + 1].segment_]++;  // right
      cnt[mb[ w - 1].segment_]++;  // bottom-left
      cnt[mb[ w + 0].segment_]++;  // bottom
      cnt[mb[ w + 1].segment_]++;  // bottom-right
      for (n = 0; n < NUM_MB_SEGMENTS; ++n) {
        if (cnt[n] >= majority_cnt_3_x_3_grid) {
          majority_seg = n;
        }
      }
      tmp[x + y * w] = majority_seg;
    }
  }
  for (y = 1; y < h - 1; ++y) {
    for (x = 1; x < w - 1; ++x) {
      VP8MBInfo* const mb = &enc->mb_info_[x + w * y];
      mb->segment_ = tmp[x + y * w];
    }
  }
  free(tmp);
}

//------------------------------------------------------------------------------
// Finalize Segment probability based on the coding tree

static int GetProba(int a, int b) {
  int proba;
  const int total = a + b;
  if (total == 0) return 255;  // that's the default probability.
  proba = (255 * a + total / 2) / total;
  return proba;
}

static void SetSegmentProbas(VP8Encoder* const enc) {
  int p[NUM_MB_SEGMENTS] = { 0 };
  int n;

  for (n = 0; n < enc->mb_w_ * enc->mb_h_; ++n) {
    const VP8MBInfo* const mb = &enc->mb_info_[n];
    p[mb->segment_]++;
  }
  if (enc->pic_->stats) {
    for (n = 0; n < NUM_MB_SEGMENTS; ++n) {
      enc->pic_->stats->segment_size[n] = p[n];
    }
  }
  if (enc->segment_hdr_.num_segments_ > 1) {
    uint8_t* const probas = enc->proba_.segments_;
    probas[0] = GetProba(p[0] + p[1], p[2] + p[3]);
    probas[1] = GetProba(p[0], p[1]);
    probas[2] = GetProba(p[2], p[3]);

    enc->segment_hdr_.update_map_ =
        (probas[0] != 255) || (probas[1] != 255) || (probas[2] != 255);
    enc->segment_hdr_.size_ =
      p[0] * (VP8BitCost(0, probas[0]) + VP8BitCost(0, probas[1])) +
      p[1] * (VP8BitCost(0, probas[0]) + VP8BitCost(1, probas[1])) +
      p[2] * (VP8BitCost(1, probas[0]) + VP8BitCost(0, probas[2])) +
      p[3] * (VP8BitCost(1, probas[0]) + VP8BitCost(1, probas[2]));
  } else {
    enc->segment_hdr_.update_map_ = 0;
    enc->segment_hdr_.size_ = 0;
  }
}

static WEBP_INLINE int clip(int v, int m, int M) {
  return v < m ? m : v > M ? M : v;
}

static void SetSegmentAlphas(VP8Encoder* const enc,
                             const int centers[NUM_MB_SEGMENTS],
                             int mid) {
  const int nb = enc->segment_hdr_.num_segments_;
  int min = centers[0], max = centers[0];
  int n;

  if (nb > 1) {
    for (n = 0; n < nb; ++n) {
      if (min > centers[n]) min = centers[n];
      if (max < centers[n]) max = centers[n];
    }
  }
  if (max == min) max = min + 1;
  assert(mid <= max && mid >= min);
  for (n = 0; n < nb; ++n) {
    const int alpha = 255 * (centers[n] - mid) / (max - min);
    const int beta = 255 * (centers[n] - min) / (max - min);
    enc->dqm_[n].alpha_ = clip(alpha, -127, 127);
    enc->dqm_[n].beta_ = clip(beta, 0, 255);
  }
}

//------------------------------------------------------------------------------
// Simplified k-Means, to assign Nb segments based on alpha-histogram

static void AssignSegments(VP8Encoder* const enc, const int alphas[256]) {
  const int nb = enc->segment_hdr_.num_segments_;
  int centers[NUM_MB_SEGMENTS];
  int weighted_average = 0;
  int map[256];
  int a, n, k;
  int min_a = 0, max_a = 255, range_a;
  // 'int' type is ok for histo, and won't overflow
  int accum[NUM_MB_SEGMENTS], dist_accum[NUM_MB_SEGMENTS];

  // bracket the input
  for (n = 0; n < 256 && alphas[n] == 0; ++n) {}
  min_a = n;
  for (n = 255; n > min_a && alphas[n] == 0; --n) {}
  max_a = n;
  range_a = max_a - min_a;

  // Spread initial centers evenly
  for (n = 1, k = 0; n < 2 * nb; n += 2) {
    centers[k++] = min_a + (n * range_a) / (2 * nb);
  }

  for (k = 0; k < MAX_ITERS_K_MEANS; ++k) {     // few iters are enough
    int total_weight;
    int displaced;
    // Reset stats
    for (n = 0; n < nb; ++n) {
      accum[n] = 0;
      dist_accum[n] = 0;
    }
    // Assign nearest center for each 'a'
    n = 0;    // track the nearest center for current 'a'
    for (a = min_a; a <= max_a; ++a) {
      if (alphas[a]) {
        while (n < nb - 1 && abs(a - centers[n + 1]) < abs(a - centers[n])) {
          n++;
        }
        map[a] = n;
        // accumulate contribution into best centroid
        dist_accum[n] += a * alphas[a];
        accum[n] += alphas[a];
      }
    }
    // All point are classified. Move the centroids to the
    // center of their respective cloud.
    displaced = 0;
    weighted_average = 0;
    total_weight = 0;
    for (n = 0; n < nb; ++n) {
      if (accum[n]) {
        const int new_center = (dist_accum[n] + accum[n] / 2) / accum[n];
        displaced += abs(centers[n] - new_center);
        centers[n] = new_center;
        weighted_average += new_center * accum[n];
        total_weight += accum[n];
      }
    }
    weighted_average = (weighted_average + total_weight / 2) / total_weight;
    if (displaced < 5) break;   // no need to keep on looping...
  }

  // Map each original value to the closest centroid
  for (n = 0; n < enc->mb_w_ * enc->mb_h_; ++n) {
    VP8MBInfo* const mb = &enc->mb_info_[n];
    const int alpha = mb->alpha_;
    mb->segment_ = map[alpha];
    mb->alpha_ = centers[map[alpha]];     // just for the record.
  }

  if (nb > 1) {
    const int smooth = (enc->config_->preprocessing & 1);
    if (smooth) SmoothSegmentMap(enc);
  }

  SetSegmentProbas(enc);                             // Assign final proba
  SetSegmentAlphas(enc, centers, weighted_average);  // pick some alphas.
}

//------------------------------------------------------------------------------
// Macroblock analysis: collect histogram for each mode, deduce the maximal
// susceptibility and set best modes for this macroblock.
// Segment assignment is done later.

// Number of modes to inspect for alpha_ evaluation. For high-quality settings,
// we don't need to test all the possible modes during the analysis phase.
#define MAX_INTRA16_MODE 2
#define MAX_INTRA4_MODE  2
#define MAX_UV_MODE      2

static int MBAnalyzeBestIntra16Mode(VP8EncIterator* const it) {
  const int max_mode = (it->enc_->method_ >= 3) ? MAX_INTRA16_MODE : 4;
  int mode;
  int best_alpha = -1;
  int best_mode = 0;

  VP8MakeLuma16Preds(it);
  for (mode = 0; mode < max_mode; ++mode) {
    const int alpha = VP8CollectHistogram(it->yuv_in_ + Y_OFF,
                                          it->yuv_p_ + VP8I16ModeOffsets[mode],
                                          0, 16);
    if (alpha > best_alpha) {
      best_alpha = alpha;
      best_mode = mode;
    }
  }
  VP8SetIntra16Mode(it, best_mode);
  return best_alpha;
}

static int MBAnalyzeBestIntra4Mode(VP8EncIterator* const it,
                                   int best_alpha) {
  uint8_t modes[16];
  const int max_mode = (it->enc_->method_ >= 3) ? MAX_INTRA4_MODE : NUM_BMODES;
  int i4_alpha = 0;
  VP8IteratorStartI4(it);
  do {
    int mode;
    int best_mode_alpha = -1;
    const uint8_t* const src = it->yuv_in_ + Y_OFF + VP8Scan[it->i4_];

    VP8MakeIntra4Preds(it);
    for (mode = 0; mode < max_mode; ++mode) {
      const int alpha = VP8CollectHistogram(src,
                                            it->yuv_p_ + VP8I4ModeOffsets[mode],
                                            0, 1);
      if (alpha > best_mode_alpha) {
        best_mode_alpha = alpha;
        modes[it->i4_] = mode;
      }
    }
    i4_alpha += best_mode_alpha;
    // Note: we reuse the original samples for predictors
  } while (VP8IteratorRotateI4(it, it->yuv_in_ + Y_OFF));

  if (i4_alpha > best_alpha) {
    VP8SetIntra4Mode(it, modes);
    best_alpha = ClipAlpha(i4_alpha);
  }
  return best_alpha;
}

static int MBAnalyzeBestUVMode(VP8EncIterator* const it) {
  int best_alpha = -1;
  int best_mode = 0;
  const int max_mode = (it->enc_->method_ >= 3) ? MAX_UV_MODE : 4;
  int mode;
  VP8MakeChroma8Preds(it);
  for (mode = 0; mode < max_mode; ++mode) {
    const int alpha = VP8CollectHistogram(it->yuv_in_ + U_OFF,
                                          it->yuv_p_ + VP8UVModeOffsets[mode],
                                          16, 16 + 4 + 4);
    if (alpha > best_alpha) {
      best_alpha = alpha;
      best_mode = mode;
    }
  }
  VP8SetIntraUVMode(it, best_mode);
  return best_alpha;
}

static void MBAnalyze(VP8EncIterator* const it,
                      int alphas[256], int* const uv_alpha) {
  const VP8Encoder* const enc = it->enc_;
  int best_alpha, best_uv_alpha;

  VP8SetIntra16Mode(it, 0);  // default: Intra16, DC_PRED
  VP8SetSkip(it, 0);         // not skipped
  VP8SetSegment(it, 0);      // default segment, spec-wise.

  best_alpha = MBAnalyzeBestIntra16Mode(it);
  if (enc->method_ != 3) {
    // We go and make a fast decision for intra4/intra16.
    // It's usually not a good and definitive pick, but helps seeding the stats
    // about level bit-cost.
    // TODO(skal): improve criterion.
    best_alpha = MBAnalyzeBestIntra4Mode(it, best_alpha);
  }
  best_uv_alpha = MBAnalyzeBestUVMode(it);

  // Final susceptibility mix
  best_alpha = (best_alpha + best_uv_alpha + 1) / 2;
  alphas[best_alpha]++;
  *uv_alpha += best_uv_alpha;
  it->mb_->alpha_ = best_alpha;   // Informative only.
}

//------------------------------------------------------------------------------
// Main analysis loop:
// Collect all susceptibilities for each macroblock and record their
// distribution in alphas[]. Segments is assigned a-posteriori, based on
// this histogram.
// We also pick an intra16 prediction mode, which shouldn't be considered
// final except for fast-encode settings. We can also pick some intra4 modes
// and decide intra4/intra16, but that's usually almost always a bad choice at
// this stage.

int VP8EncAnalyze(VP8Encoder* const enc) {
  int ok = 1;
  int alphas[256] = { 0 };
  VP8EncIterator it;

  VP8IteratorInit(enc, &it);
  enc->uv_alpha_ = 0;
  do {
    VP8IteratorImport(&it);
    MBAnalyze(&it, alphas, &enc->uv_alpha_);
    ok = VP8IteratorProgress(&it, 20);
    // Let's pretend we have perfect lossless reconstruction.
  } while (ok && VP8IteratorNext(&it, it.yuv_in_));
  enc->uv_alpha_ /= enc->mb_w_ * enc->mb_h_;
  if (ok) AssignSegments(enc, alphas);

  return ok;
}

#if defined(__cplusplus) || defined(c_plusplus)
}    // extern "C"
#endif
