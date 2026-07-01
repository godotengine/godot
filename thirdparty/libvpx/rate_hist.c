/*
 *  Copyright (c) 2014 The WebM project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */

#include <assert.h>
#include <limits.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include "./rate_hist.h"

#define RATE_BINS 100
#define HIST_BAR_MAX 40

struct hist_bucket {
  int low;
  int high;
  int count;
};

struct rate_hist {
  int64_t *pts;
  int *sz;
  int samples;
  int frames;
  struct hist_bucket bucket[RATE_BINS];
  int total;
};

struct rate_hist *init_rate_histogram(const vpx_codec_enc_cfg_t *cfg,
                                      const vpx_rational_t *fps) {
  int i;
  struct rate_hist *hist = calloc(1, sizeof(*hist));

  if (hist == NULL || cfg == NULL || fps == NULL || fps->num == 0 ||
      fps->den == 0) {
    destroy_rate_histogram(hist);
    return NULL;
  }

  // Determine the number of samples in the buffer. Use the file's framerate
  // to determine the number of frames in rc_buf_sz milliseconds, with an
  // adjustment (5/4) to account for alt-refs
  hist->samples =
      (int)((int64_t)cfg->rc_buf_sz * 5 / 4 * fps->num / fps->den / 1000);

  // prevent division by zero
  if (hist->samples == 0) hist->samples = 1;

  hist->frames = 0;
  hist->total = 0;

  hist->pts = calloc(hist->samples, sizeof(*hist->pts));
  hist->sz = calloc(hist->samples, sizeof(*hist->sz));
  for (i = 0; i < RATE_BINS; i++) {
    hist->bucket[i].low = INT_MAX;
    hist->bucket[i].high = 0;
    hist->bucket[i].count = 0;
  }

  return hist;
}

void destroy_rate_histogram(struct rate_hist *hist) {
  if (hist) {
    free(hist->pts);
    free(hist->sz);
    free(hist);
  }
}

void update_rate_histogram(struct rate_hist *hist,
                           const vpx_codec_enc_cfg_t *cfg,
                           const vpx_codec_cx_pkt_t *pkt) {
  int i;
  int64_t then = 0;
  int64_t avg_bitrate = 0;
  int64_t sum_sz = 0;
  const int64_t now = pkt->data.frame.pts * 1000 *
                      (uint64_t)cfg->g_timebase.num /
                      (uint64_t)cfg->g_timebase.den;

  int idx;

  if (hist == NULL || cfg == NULL || pkt == NULL) return;

  idx = hist->frames++ % hist->samples;
  hist->pts[idx] = now;
  hist->sz[idx] = (int)pkt->data.frame.sz;

  if (now < cfg->rc_buf_initial_sz) return;

  if (!cfg->rc_target_bitrate) return;

  then = now;

  /* Sum the size over the past rc_buf_sz ms */
  for (i = hist->frames; i > 0 && hist->frames - i < hist->samples; i--) {
    const int i_idx = (i - 1) % hist->samples;

    then = hist->pts[i_idx];
    if (now - then > cfg->rc_buf_sz) break;
    sum_sz += hist->sz[i_idx];
  }

  if (now == then) return;

  avg_bitrate = sum_sz * 8 * 1000 / (now - then);
  idx = (int)(avg_bitrate * (RATE_BINS / 2) / (cfg->rc_target_bitrate * 1000));
  if (idx < 0) idx = 0;
  if (idx > RATE_BINS - 1) idx = RATE_BINS - 1;
  if (hist->bucket[idx].low > avg_bitrate)
    hist->bucket[idx].low = (int)avg_bitrate;
  if (hist->bucket[idx].high < avg_bitrate)
    hist->bucket[idx].high = (int)avg_bitrate;
  hist->bucket[idx].count++;
  hist->total++;
}

static int merge_hist_buckets(struct hist_bucket *bucket, int max_buckets,
                              int *num_buckets) {
  int small_bucket = 0, merge_bucket = INT_MAX, big_bucket = 0;
  int buckets;
  int i;

  assert(bucket != NULL);
  assert(num_buckets != NULL);

  buckets = *num_buckets;

  /* Find the extrema for this list of buckets */
  big_bucket = small_bucket = 0;
  for (i = 0; i < buckets; i++) {
    if (bucket[i].count < bucket[small_bucket].count) small_bucket = i;
    if (bucket[i].count > bucket[big_bucket].count) big_bucket = i;
  }

  /* If we have too many buckets, merge the smallest with an adjacent
   * bucket.
   */
  while (buckets > max_buckets) {
    int last_bucket = buckets - 1;

    /* merge the small bucket with an adjacent one. */
    if (small_bucket == 0)
      merge_bucket = 1;
    else if (small_bucket == last_bucket)
      merge_bucket = last_bucket - 1;
    else if (bucket[small_bucket - 1].count < bucket[small_bucket + 1].count)
      merge_bucket = small_bucket - 1;
    else
      merge_bucket = small_bucket + 1;

    assert(abs(merge_bucket - small_bucket) <= 1);
    assert(small_bucket < buckets);
    assert(big_bucket < buckets);
    assert(merge_bucket < buckets);

    if (merge_bucket < small_bucket) {
      bucket[merge_bucket].high = bucket[small_bucket].high;
      bucket[merge_bucket].count += bucket[small_bucket].count;
    } else {
      bucket[small_bucket].high = bucket[merge_bucket].high;
      bucket[small_bucket].count += bucket[merge_bucket].count;
      merge_bucket = small_bucket;
    }

    assert(bucket[merge_bucket].low != bucket[merge_bucket].high);

    buckets--;

    /* Remove the merge_bucket from the list, and find the new small
     * and big buckets while we're at it
     */
    big_bucket = small_bucket = 0;
    for (i = 0; i < buckets; i++) {
      if (i > merge_bucket) bucket[i] = bucket[i + 1];

      if (bucket[i].count < bucket[small_bucket].count) small_bucket = i;
      if (bucket[i].count > bucket[big_bucket].count) big_bucket = i;
    }
  }

  *num_buckets = buckets;
  return bucket[big_bucket].count;
}

static void show_histogram(const struct hist_bucket *bucket, int buckets,
                           int total, int scale) {
  int width1, width2;
  int i;

  if (!buckets) return;
  assert(bucket != NULL);
  assert(buckets > 0);

  switch ((int)(log(bucket[buckets - 1].high) / log(10)) + 1) {
    case 1:
    case 2:
      width1 = 4;
      width2 = 2;
      break;
    case 3:
      width1 = 5;
      width2 = 3;
      break;
    case 4:
      width1 = 6;
      width2 = 4;
      break;
    case 5:
      width1 = 7;
      width2 = 5;
      break;
    case 6:
      width1 = 8;
      width2 = 6;
      break;
    case 7:
      width1 = 9;
      width2 = 7;
      break;
    default:
      width1 = 12;
      width2 = 10;
      break;
  }

  for (i = 0; i < buckets; i++) {
    int len;
    int j;
    float pct;

    pct = (float)(100.0 * bucket[i].count / total);
    len = HIST_BAR_MAX * bucket[i].count / scale;
    if (len < 1) len = 1;
    assert(len <= HIST_BAR_MAX);

    if (bucket[i].low == bucket[i].high)
      fprintf(stderr, "%*d %*s: ", width1, bucket[i].low, width2, "");
    else
      fprintf(stderr, "%*d-%*d: ", width1, bucket[i].low, width2,
              bucket[i].high);

    for (j = 0; j < HIST_BAR_MAX; j++) fprintf(stderr, j < len ? "=" : " ");
    fprintf(stderr, "\t%5d (%6.2f%%)\n", bucket[i].count, pct);
  }
}

void show_q_histogram(const int counts[64], int max_buckets) {
  struct hist_bucket bucket[64];
  int buckets = 0;
  int total = 0;
  int scale;
  int i;

  for (i = 0; i < 64; i++) {
    if (counts[i]) {
      bucket[buckets].low = bucket[buckets].high = i;
      bucket[buckets].count = counts[i];
      buckets++;
      total += counts[i];
    }
  }

  fprintf(stderr, "\nQuantizer Selection:\n");
  scale = merge_hist_buckets(bucket, max_buckets, &buckets);
  show_histogram(bucket, buckets, total, scale);
}

void show_rate_histogram(struct rate_hist *hist, const vpx_codec_enc_cfg_t *cfg,
                         int max_buckets) {
  int i, scale;
  int buckets = 0;

  if (hist == NULL || cfg == NULL) return;

  for (i = 0; i < RATE_BINS; i++) {
    if (hist->bucket[i].low == INT_MAX) continue;
    hist->bucket[buckets++] = hist->bucket[i];
  }

  fprintf(stderr, "\nRate (over %dms window):\n", cfg->rc_buf_sz);
  scale = merge_hist_buckets(hist->bucket, max_buckets, &buckets);
  show_histogram(hist->bucket, buckets, hist->total, scale);
}
