// Copyright 2011 Google Inc. All Rights Reserved.
//
// Use of this source code is governed by a BSD-style license
// that can be found in the COPYING file in the root of the source
// tree. An additional intellectual property rights grant can be found
// in the file PATENTS. All contributing project authors may
// be found in the AUTHORS file in the root of the source tree.
// -----------------------------------------------------------------------------
//
// Macroblock analysis
//
// Author: Skal (pascal.massimino@gmail.com)

#include <stdlib.h>
#include <string.h>
#include <assert.h>

#include "src/enc/vp8i_enc.h"
#include "src/enc/cost_enc.h"
#include "src/utils/utils.h"

#define MAX_ITERS_K_MEANS  6

//------------------------------------------------------------------------------
// Smooth the segment map by replacing isolated block by the majority of its
// neighbours.

static void SmoothSegmentMap(VP8Encoder* const enc) {
  int n, x, y;
  const int w = enc->mb_w_;
  const int h = enc->mb_h_;
  const int majority_cnt_3_x_3_grid = 5;
  uint8_t* const tmp = (uint8_t*)WebPSafeMalloc(w * h, sizeof(*tmp));
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
          break;
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
  WebPSafeFree(tmp);
}

//------------------------------------------------------------------------------
// set segment susceptibility alpha_ / beta_

static WEBP_INLINE int clip(int v, int m, int M) {
  return (v < m) ? m : (v > M) ? M : v;
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
// Compute susceptibility based on DCT-coeff histograms:
// the higher, the "easier" the macroblock is to compress.

#define MAX_ALPHA 255                // 8b of precision for susceptibilities.
#define ALPHA_SCALE (2 * MAX_ALPHA)  // scaling factor for alpha.
#define DEFAULT_ALPHA (-1)
#define IS_BETTER_ALPHA(alpha, best_alpha) ((alpha) > (best_alpha))

static int FinalAlphaValue(int alpha) {
  alpha = MAX_ALPHA - alpha;
  return clip(alpha, 0, MAX_ALPHA);
}

static int GetAlpha(const VP8Histogram* const histo) {
  // 'alpha' will later be clipped to [0..MAX_ALPHA] range, clamping outer
  // values which happen to be mostly noise. This leaves the maximum precision
  // for handling the useful small values which contribute most.
  const int max_value = histo->max_value;
  const int last_non_zero = histo->last_non_zero;
  const int alpha =
      (max_value > 1) ? ALPHA_SCALE * last_non_zero / max_value : 0;
  return alpha;
}

static void InitHistogram(VP8Histogram* const histo) {
  histo->max_value = 0;
  histo->last_non_zero = 1;
}

static void MergeHistograms(const VP8Histogram* const in,
                            VP8Histogram* const out) {
  if (in->max_value > out->max_value) {
    out->max_value = in->max_value;
  }
  if (in->last_non_zero > out->last_non_zero) {
    out->last_non_zero = in->last_non_zero;
  }
}

//------------------------------------------------------------------------------
// Simplified k-Means, to assign Nb segments based on alpha-histogram

static void AssignSegments(VP8Encoder* const enc,
                           const int alphas[MAX_ALPHA + 1]) {
  // 'num_segments_' is previously validated and <= NUM_MB_SEGMENTS, but an
  // explicit check is needed to avoid spurious warning about 'n + 1' exceeding
  // array bounds of 'centers' with some compilers (noticed with gcc-4.9).
  const int nb = (enc->segment_hdr_.num_segments_ < NUM_MB_SEGMENTS) ?
                 enc->segment_hdr_.num_segments_ : NUM_MB_SEGMENTS;
  int centers[NUM_MB_SEGMENTS];
  int weighted_average = 0;
  int map[MAX_ALPHA + 1];
  int a, n, k;
  int min_a = 0, max_a = MAX_ALPHA, range_a;
  // 'int' type is ok for histo, and won't overflow
  int accum[NUM_MB_SEGMENTS], dist_accum[NUM_MB_SEGMENTS];

  assert(nb >= 1);
  assert(nb <= NUM_MB_SEGMENTS);

  // bracket the input
  for (n = 0; n <= MAX_ALPHA && alphas[n] == 0; ++n) {}
  min_a = n;
  for (n = MAX_ALPHA; n > min_a && alphas[n] == 0; --n) {}
  max_a = n;
  range_a = max_a - min_a;

  // Spread initial centers evenly
  for (k = 0, n = 1; k < nb; ++k, n += 2) {
    assert(n < 2 * nb);
    centers[k] = min_a + (n * range_a) / (2 * nb);
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
        while (n + 1 < nb && abs(a - centers[n + 1]) < abs(a - centers[n])) {
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
    mb->alpha_ = centers[map[alpha]];  // for the record.
  }

  if (nb > 1) {
    const int smooth = (enc->config_->preprocessing & 1);
    if (smooth) SmoothSegmentMap(enc);
  }

  SetSegmentAlphas(enc, centers, weighted_average);  // pick some alphas.
}

//------------------------------------------------------------------------------
// Macroblock analysis: collect histogram for each mode, deduce the maximal
// susceptibility and set best modes for this macroblock.
// Segment assignment is done later.

// Number of modes to inspect for alpha_ evaluation. We don't need to test all
// the possible modes during the analysis phase: we risk falling into a local
// optimum, or be subject to boundary effect
#define MAX_INTRA16_MODE 2
#define MAX_INTRA4_MODE  2
#define MAX_UV_MODE      2

static int MBAnalyzeBestIntra16Mode(VP8EncIterator* const it) {
  const int max_mode = MAX_INTRA16_MODE;
  int mode;
  int best_alpha = DEFAULT_ALPHA;
  int best_mode = 0;

  VP8MakeLuma16Preds(it);
  for (mode = 0; mode < max_mode; ++mode) {
    VP8Histogram histo;
    int alpha;

    InitHistogram(&histo);
    VP8CollectHistogram(it->yuv_in_ + Y_OFF_ENC,
                        it->yuv_p_ + VP8I16ModeOffsets[mode],
                        0, 16, &histo);
    alpha = GetAlpha(&histo);
    if (IS_BETTER_ALPHA(alpha, best_alpha)) {
      best_alpha = alpha;
      best_mode = mode;
    }
  }
  VP8SetIntra16Mode(it, best_mode);
  return best_alpha;
}

static int FastMBAnalyze(VP8EncIterator* const it) {
  // Empirical cut-off value, should be around 16 (~=block size). We use the
  // [8-17] range and favor intra4 at high quality, intra16 for low quality.
  const int q = (int)it->enc_->config_->quality;
  const uint32_t kThreshold = 8 + (17 - 8) * q / 100;
  int k;
  uint32_t dc[16], m, m2;
  for (k = 0; k < 16; k += 4) {
    VP8Mean16x4(it->yuv_in_ + Y_OFF_ENC + k * BPS, &dc[k]);
  }
  for (m = 0, m2 = 0, k = 0; k < 16; ++k) {
    m += dc[k];
    m2 += dc[k] * dc[k];
  }
  if (kThreshold * m2 < m * m) {
    VP8SetIntra16Mode(it, 0);   // DC16
  } else {
    const uint8_t modes[16] = { 0 };  // DC4
    VP8SetIntra4Mode(it, modes);
  }
  return 0;
}

static int MBAnalyzeBestIntra4Mode(VP8EncIterator* const it,
                                   int best_alpha) {
  uint8_t modes[16];
  const int max_mode = MAX_INTRA4_MODE;
  int i4_alpha;
  VP8Histogram total_histo;
  int cur_histo = 0;
  InitHistogram(&total_histo);

  VP8IteratorStartI4(it);
  do {
    int mode;
    int best_mode_alpha = DEFAULT_ALPHA;
    VP8Histogram histos[2];
    const uint8_t* const src = it->yuv_in_ + Y_OFF_ENC + VP8Scan[it->i4_];

    VP8MakeIntra4Preds(it);
    for (mode = 0; mode < max_mode; ++mode) {
      int alpha;

      InitHistogram(&histos[cur_histo]);
      VP8CollectHistogram(src, it->yuv_p_ + VP8I4ModeOffsets[mode],
                          0, 1, &histos[cur_histo]);
      alpha = GetAlpha(&histos[cur_histo]);
      if (IS_BETTER_ALPHA(alpha, best_mode_alpha)) {
        best_mode_alpha = alpha;
        modes[it->i4_] = mode;
        cur_histo ^= 1;   // keep track of best histo so far.
      }
    }
    // accumulate best histogram
    MergeHistograms(&histos[cur_histo ^ 1], &total_histo);
    // Note: we reuse the original samples for predictors
  } while (VP8IteratorRotateI4(it, it->yuv_in_ + Y_OFF_ENC));

  i4_alpha = GetAlpha(&total_histo);
  if (IS_BETTER_ALPHA(i4_alpha, best_alpha)) {
    VP8SetIntra4Mode(it, modes);
    best_alpha = i4_alpha;
  }
  return best_alpha;
}

static int MBAnalyzeBestUVMode(VP8EncIterator* const it) {
  int best_alpha = DEFAULT_ALPHA;
  int smallest_alpha = 0;
  int best_mode = 0;
  const int max_mode = MAX_UV_MODE;
  int mode;

  VP8MakeChroma8Preds(it);
  for (mode = 0; mode < max_mode; ++mode) {
    VP8Histogram histo;
    int alpha;
    InitHistogram(&histo);
    VP8CollectHistogram(it->yuv_in_ + U_OFF_ENC,
                        it->yuv_p_ + VP8UVModeOffsets[mode],
                        16, 16 + 4 + 4, &histo);
    alpha = GetAlpha(&histo);
    if (IS_BETTER_ALPHA(alpha, best_alpha)) {
      best_alpha = alpha;
    }
    // The best prediction mode tends to be the one with the smallest alpha.
    if (mode == 0 || alpha < smallest_alpha) {
      smallest_alpha = alpha;
      best_mode = mode;
    }
  }
  VP8SetIntraUVMode(it, best_mode);
  return best_alpha;
}

static void MBAnalyze(VP8EncIterator* const it,
                      int alphas[MAX_ALPHA + 1],
                      int* const alpha, int* const uv_alpha) {
  const VP8Encoder* const enc = it->enc_;
  int best_alpha, best_uv_alpha;

  VP8SetIntra16Mode(it, 0);  // default: Intra16, DC_PRED
  VP8SetSkip(it, 0);         // not skipped
  VP8SetSegment(it, 0);      // default segment, spec-wise.

  if (enc->method_ <= 1) {
    best_alpha = FastMBAnalyze(it);
  } else {
    best_alpha = MBAnalyzeBestIntra16Mode(it);
    if (enc->method_ >= 5) {
      // We go and make a fast decision for intra4/intra16.
      // It's usually not a good and definitive pick, but helps seeding the
      // stats about level bit-cost.
      // TODO(skal): improve criterion.
      best_alpha = MBAnalyzeBestIntra4Mode(it, best_alpha);
    }
  }
  best_uv_alpha = MBAnalyzeBestUVMode(it);

  // Final susceptibility mix
  best_alpha = (3 * best_alpha + best_uv_alpha + 2) >> 2;
  best_alpha = FinalAlphaValue(best_alpha);
  alphas[best_alpha]++;
  it->mb_->alpha_ = best_alpha;   // for later remapping.

  // Accumulate for later complexity analysis.
  *alpha += best_alpha;   // mixed susceptibility (not just luma)
  *uv_alpha += best_uv_alpha;
}

static void DefaultMBInfo(VP8MBInfo* const mb) {
  mb->type_ = 1;     // I16x16
  mb->uv_mode_ = 0;
  mb->skip_ = 0;     // not skipped
  mb->segment_ = 0;  // default segment
  mb->alpha_ = 0;
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

static void ResetAllMBInfo(VP8Encoder* const enc) {
  int n;
  for (n = 0; n < enc->mb_w_ * enc->mb_h_; ++n) {
    DefaultMBInfo(&enc->mb_info_[n]);
  }
  // Default susceptibilities.
  enc->dqm_[0].alpha_ = 0;
  enc->dqm_[0].beta_ = 0;
  // Note: we can't compute this alpha_ / uv_alpha_ -> set to default value.
  enc->alpha_ = 0;
  enc->uv_alpha_ = 0;
  WebPReportProgress(enc->pic_, enc->percent_ + 20, &enc->percent_);
}

// struct used to collect job result
typedef struct {
  WebPWorker worker;
  int alphas[MAX_ALPHA + 1];
  int alpha, uv_alpha;
  VP8EncIterator it;
  int delta_progress;
} SegmentJob;

// main work call
static int DoSegmentsJob(void* arg1, void* arg2) {
  SegmentJob* const job = (SegmentJob*)arg1;
  VP8EncIterator* const it = (VP8EncIterator*)arg2;
  int ok = 1;
  if (!VP8IteratorIsDone(it)) {
    uint8_t tmp[32 + WEBP_ALIGN_CST];
    uint8_t* const scratch = (uint8_t*)WEBP_ALIGN(tmp);
    do {
      // Let's pretend we have perfect lossless reconstruction.
      VP8IteratorImport(it, scratch);
      MBAnalyze(it, job->alphas, &job->alpha, &job->uv_alpha);
      ok = VP8IteratorProgress(it, job->delta_progress);
    } while (ok && VP8IteratorNext(it));
  }
  return ok;
}

static void MergeJobs(const SegmentJob* const src, SegmentJob* const dst) {
  int i;
  for (i = 0; i <= MAX_ALPHA; ++i) dst->alphas[i] += src->alphas[i];
  dst->alpha += src->alpha;
  dst->uv_alpha += src->uv_alpha;
}

// initialize the job struct with some tasks to perform
static void InitSegmentJob(VP8Encoder* const enc, SegmentJob* const job,
                           int start_row, int end_row) {
  WebPGetWorkerInterface()->Init(&job->worker);
  job->worker.data1 = job;
  job->worker.data2 = &job->it;
  job->worker.hook = DoSegmentsJob;
  VP8IteratorInit(enc, &job->it);
  VP8IteratorSetRow(&job->it, start_row);
  VP8IteratorSetCountDown(&job->it, (end_row - start_row) * enc->mb_w_);
  memset(job->alphas, 0, sizeof(job->alphas));
  job->alpha = 0;
  job->uv_alpha = 0;
  // only one of both jobs can record the progress, since we don't
  // expect the user's hook to be multi-thread safe
  job->delta_progress = (start_row == 0) ? 20 : 0;
}

// main entry point
int VP8EncAnalyze(VP8Encoder* const enc) {
  int ok = 1;
  const int do_segments =
      enc->config_->emulate_jpeg_size ||   // We need the complexity evaluation.
      (enc->segment_hdr_.num_segments_ > 1) ||
      (enc->method_ <= 1);  // for method 0 - 1, we need preds_[] to be filled.
  if (do_segments) {
    const int last_row = enc->mb_h_;
    // We give a little more than a half work to the main thread.
    const int split_row = (9 * last_row + 15) >> 4;
    const int total_mb = last_row * enc->mb_w_;
#ifdef WEBP_USE_THREAD
    const int kMinSplitRow = 2;  // minimal rows needed for mt to be worth it
    const int do_mt = (enc->thread_level_ > 0) && (split_row >= kMinSplitRow);
#else
    const int do_mt = 0;
#endif
    const WebPWorkerInterface* const worker_interface =
        WebPGetWorkerInterface();
    SegmentJob main_job;
    if (do_mt) {
      SegmentJob side_job;
      // Note the use of '&' instead of '&&' because we must call the functions
      // no matter what.
      InitSegmentJob(enc, &main_job, 0, split_row);
      InitSegmentJob(enc, &side_job, split_row, last_row);
      // we don't need to call Reset() on main_job.worker, since we're calling
      // WebPWorkerExecute() on it
      ok &= worker_interface->Reset(&side_job.worker);
      // launch the two jobs in parallel
      if (ok) {
        worker_interface->Launch(&side_job.worker);
        worker_interface->Execute(&main_job.worker);
        ok &= worker_interface->Sync(&side_job.worker);
        ok &= worker_interface->Sync(&main_job.worker);
      }
      worker_interface->End(&side_job.worker);
      if (ok) MergeJobs(&side_job, &main_job);  // merge results together
    } else {
      // Even for single-thread case, we use the generic Worker tools.
      InitSegmentJob(enc, &main_job, 0, last_row);
      worker_interface->Execute(&main_job.worker);
      ok &= worker_interface->Sync(&main_job.worker);
    }
    worker_interface->End(&main_job.worker);
    if (ok) {
      enc->alpha_ = main_job.alpha / total_mb;
      enc->uv_alpha_ = main_job.uv_alpha / total_mb;
      AssignSegments(enc, main_job.alphas);
    }
  } else {   // Use only one default segment.
    ResetAllMBInfo(enc);
  }
  return ok;
}

