// Copyright 2011 Google Inc. All Rights Reserved.
//
// Use of this source code is governed by a BSD-style license
// that can be found in the COPYING file in the root of the source
// tree. An additional intellectual property rights grant can be found
// in the file PATENTS. All contributing project authors may
// be found in the AUTHORS file in the root of the source tree.
// -----------------------------------------------------------------------------
//
//   frame coding and analysis
//
// Author: Skal (pascal.massimino@gmail.com)

#include <string.h>
#include <math.h>

#include "./cost.h"
#include "./vp8enci.h"
#include "../dsp/dsp.h"
#include "../webp/format_constants.h"  // RIFF constants

#define SEGMENT_VISU 0
#define DEBUG_SEARCH 0    // useful to track search convergence

//------------------------------------------------------------------------------
// multi-pass convergence

#define HEADER_SIZE_ESTIMATE (RIFF_HEADER_SIZE + CHUNK_HEADER_SIZE +  \
                              VP8_FRAME_HEADER_SIZE)
#define DQ_LIMIT 0.4  // convergence is considered reached if dq < DQ_LIMIT
// we allow 2k of extra head-room in PARTITION0 limit.
#define PARTITION0_SIZE_LIMIT ((VP8_MAX_PARTITION0_SIZE - 2048ULL) << 11)

typedef struct {  // struct for organizing convergence in either size or PSNR
  int is_first;
  float dq;
  float q, last_q;
  double value, last_value;   // PSNR or size
  double target;
  int do_size_search;
} PassStats;

static int InitPassStats(const VP8Encoder* const enc, PassStats* const s) {
  const uint64_t target_size = (uint64_t)enc->config_->target_size;
  const int do_size_search = (target_size != 0);
  const float target_PSNR = enc->config_->target_PSNR;

  s->is_first = 1;
  s->dq = 10.f;
  s->q = s->last_q = enc->config_->quality;
  s->target = do_size_search ? (double)target_size
            : (target_PSNR > 0.) ? target_PSNR
            : 40.;   // default, just in case
  s->value = s->last_value = 0.;
  s->do_size_search = do_size_search;
  return do_size_search;
}

static float Clamp(float v, float min, float max) {
  return (v < min) ? min : (v > max) ? max : v;
}

static float ComputeNextQ(PassStats* const s) {
  float dq;
  if (s->is_first) {
    dq = (s->value > s->target) ? -s->dq : s->dq;
    s->is_first = 0;
  } else if (s->value != s->last_value) {
    const double slope = (s->target - s->value) / (s->last_value - s->value);
    dq = (float)(slope * (s->last_q - s->q));
  } else {
    dq = 0.;  // we're done?!
  }
  // Limit variable to avoid large swings.
  s->dq = Clamp(dq, -30.f, 30.f);
  s->last_q = s->q;
  s->last_value = s->value;
  s->q = Clamp(s->q + s->dq, 0.f, 100.f);
  return s->q;
}

//------------------------------------------------------------------------------
// Tables for level coding

const uint8_t VP8Cat3[] = { 173, 148, 140 };
const uint8_t VP8Cat4[] = { 176, 155, 140, 135 };
const uint8_t VP8Cat5[] = { 180, 157, 141, 134, 130 };
const uint8_t VP8Cat6[] =
    { 254, 254, 243, 230, 196, 177, 153, 140, 133, 130, 129 };

//------------------------------------------------------------------------------
// Reset the statistics about: number of skips, token proba, level cost,...

static void ResetStats(VP8Encoder* const enc) {
  VP8EncProba* const proba = &enc->proba_;
  VP8CalculateLevelCosts(proba);
  proba->nb_skip_ = 0;
}

//------------------------------------------------------------------------------
// Skip decision probability

#define SKIP_PROBA_THRESHOLD 250  // value below which using skip_proba is OK.

static int CalcSkipProba(uint64_t nb, uint64_t total) {
  return (int)(total ? (total - nb) * 255 / total : 255);
}

// Returns the bit-cost for coding the skip probability.
static int FinalizeSkipProba(VP8Encoder* const enc) {
  VP8EncProba* const proba = &enc->proba_;
  const int nb_mbs = enc->mb_w_ * enc->mb_h_;
  const int nb_events = proba->nb_skip_;
  int size;
  proba->skip_proba_ = CalcSkipProba(nb_events, nb_mbs);
  proba->use_skip_proba_ = (proba->skip_proba_ < SKIP_PROBA_THRESHOLD);
  size = 256;   // 'use_skip_proba' bit
  if (proba->use_skip_proba_) {
    size +=  nb_events * VP8BitCost(1, proba->skip_proba_)
         + (nb_mbs - nb_events) * VP8BitCost(0, proba->skip_proba_);
    size += 8 * 256;   // cost of signaling the skip_proba_ itself.
  }
  return size;
}

// Collect statistics and deduce probabilities for next coding pass.
// Return the total bit-cost for coding the probability updates.
static int CalcTokenProba(int nb, int total) {
  assert(nb <= total);
  return nb ? (255 - nb * 255 / total) : 255;
}

// Cost of coding 'nb' 1's and 'total-nb' 0's using 'proba' probability.
static int BranchCost(int nb, int total, int proba) {
  return nb * VP8BitCost(1, proba) + (total - nb) * VP8BitCost(0, proba);
}

static void ResetTokenStats(VP8Encoder* const enc) {
  VP8EncProba* const proba = &enc->proba_;
  memset(proba->stats_, 0, sizeof(proba->stats_));
}

static int FinalizeTokenProbas(VP8EncProba* const proba) {
  int has_changed = 0;
  int size = 0;
  int t, b, c, p;
  for (t = 0; t < NUM_TYPES; ++t) {
    for (b = 0; b < NUM_BANDS; ++b) {
      for (c = 0; c < NUM_CTX; ++c) {
        for (p = 0; p < NUM_PROBAS; ++p) {
          const proba_t stats = proba->stats_[t][b][c][p];
          const int nb = (stats >> 0) & 0xffff;
          const int total = (stats >> 16) & 0xffff;
          const int update_proba = VP8CoeffsUpdateProba[t][b][c][p];
          const int old_p = VP8CoeffsProba0[t][b][c][p];
          const int new_p = CalcTokenProba(nb, total);
          const int old_cost = BranchCost(nb, total, old_p)
                             + VP8BitCost(0, update_proba);
          const int new_cost = BranchCost(nb, total, new_p)
                             + VP8BitCost(1, update_proba)
                             + 8 * 256;
          const int use_new_p = (old_cost > new_cost);
          size += VP8BitCost(use_new_p, update_proba);
          if (use_new_p) {  // only use proba that seem meaningful enough.
            proba->coeffs_[t][b][c][p] = new_p;
            has_changed |= (new_p != old_p);
            size += 8 * 256;
          } else {
            proba->coeffs_[t][b][c][p] = old_p;
          }
        }
      }
    }
  }
  proba->dirty_ = has_changed;
  return size;
}

//------------------------------------------------------------------------------
// Finalize Segment probability based on the coding tree

static int GetProba(int a, int b) {
  const int total = a + b;
  return (total == 0) ? 255     // that's the default probability.
                      : (255 * a + total / 2) / total;  // rounded proba
}

static void SetSegmentProbas(VP8Encoder* const enc) {
  int p[NUM_MB_SEGMENTS] = { 0 };
  int n;

  for (n = 0; n < enc->mb_w_ * enc->mb_h_; ++n) {
    const VP8MBInfo* const mb = &enc->mb_info_[n];
    p[mb->segment_]++;
  }
  if (enc->pic_->stats != NULL) {
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

//------------------------------------------------------------------------------
// Coefficient coding

static int PutCoeffs(VP8BitWriter* const bw, int ctx, const VP8Residual* res) {
  int n = res->first;
  // should be prob[VP8EncBands[n]], but it's equivalent for n=0 or 1
  const uint8_t* p = res->prob[n][ctx];
  if (!VP8PutBit(bw, res->last >= 0, p[0])) {
    return 0;
  }

  while (n < 16) {
    const int c = res->coeffs[n++];
    const int sign = c < 0;
    int v = sign ? -c : c;
    if (!VP8PutBit(bw, v != 0, p[1])) {
      p = res->prob[VP8EncBands[n]][0];
      continue;
    }
    if (!VP8PutBit(bw, v > 1, p[2])) {
      p = res->prob[VP8EncBands[n]][1];
    } else {
      if (!VP8PutBit(bw, v > 4, p[3])) {
        if (VP8PutBit(bw, v != 2, p[4]))
          VP8PutBit(bw, v == 4, p[5]);
      } else if (!VP8PutBit(bw, v > 10, p[6])) {
        if (!VP8PutBit(bw, v > 6, p[7])) {
          VP8PutBit(bw, v == 6, 159);
        } else {
          VP8PutBit(bw, v >= 9, 165);
          VP8PutBit(bw, !(v & 1), 145);
        }
      } else {
        int mask;
        const uint8_t* tab;
        if (v < 3 + (8 << 1)) {          // VP8Cat3  (3b)
          VP8PutBit(bw, 0, p[8]);
          VP8PutBit(bw, 0, p[9]);
          v -= 3 + (8 << 0);
          mask = 1 << 2;
          tab = VP8Cat3;
        } else if (v < 3 + (8 << 2)) {   // VP8Cat4  (4b)
          VP8PutBit(bw, 0, p[8]);
          VP8PutBit(bw, 1, p[9]);
          v -= 3 + (8 << 1);
          mask = 1 << 3;
          tab = VP8Cat4;
        } else if (v < 3 + (8 << 3)) {   // VP8Cat5  (5b)
          VP8PutBit(bw, 1, p[8]);
          VP8PutBit(bw, 0, p[10]);
          v -= 3 + (8 << 2);
          mask = 1 << 4;
          tab = VP8Cat5;
        } else {                         // VP8Cat6 (11b)
          VP8PutBit(bw, 1, p[8]);
          VP8PutBit(bw, 1, p[10]);
          v -= 3 + (8 << 3);
          mask = 1 << 10;
          tab = VP8Cat6;
        }
        while (mask) {
          VP8PutBit(bw, !!(v & mask), *tab++);
          mask >>= 1;
        }
      }
      p = res->prob[VP8EncBands[n]][2];
    }
    VP8PutBitUniform(bw, sign);
    if (n == 16 || !VP8PutBit(bw, n <= res->last, p[0])) {
      return 1;   // EOB
    }
  }
  return 1;
}

static void CodeResiduals(VP8BitWriter* const bw, VP8EncIterator* const it,
                          const VP8ModeScore* const rd) {
  int x, y, ch;
  VP8Residual res;
  uint64_t pos1, pos2, pos3;
  const int i16 = (it->mb_->type_ == 1);
  const int segment = it->mb_->segment_;
  VP8Encoder* const enc = it->enc_;

  VP8IteratorNzToBytes(it);

  pos1 = VP8BitWriterPos(bw);
  if (i16) {
    VP8InitResidual(0, 1, enc, &res);
    VP8SetResidualCoeffs(rd->y_dc_levels, &res);
    it->top_nz_[8] = it->left_nz_[8] =
      PutCoeffs(bw, it->top_nz_[8] + it->left_nz_[8], &res);
    VP8InitResidual(1, 0, enc, &res);
  } else {
    VP8InitResidual(0, 3, enc, &res);
  }

  // luma-AC
  for (y = 0; y < 4; ++y) {
    for (x = 0; x < 4; ++x) {
      const int ctx = it->top_nz_[x] + it->left_nz_[y];
      VP8SetResidualCoeffs(rd->y_ac_levels[x + y * 4], &res);
      it->top_nz_[x] = it->left_nz_[y] = PutCoeffs(bw, ctx, &res);
    }
  }
  pos2 = VP8BitWriterPos(bw);

  // U/V
  VP8InitResidual(0, 2, enc, &res);
  for (ch = 0; ch <= 2; ch += 2) {
    for (y = 0; y < 2; ++y) {
      for (x = 0; x < 2; ++x) {
        const int ctx = it->top_nz_[4 + ch + x] + it->left_nz_[4 + ch + y];
        VP8SetResidualCoeffs(rd->uv_levels[ch * 2 + x + y * 2], &res);
        it->top_nz_[4 + ch + x] = it->left_nz_[4 + ch + y] =
            PutCoeffs(bw, ctx, &res);
      }
    }
  }
  pos3 = VP8BitWriterPos(bw);
  it->luma_bits_ = pos2 - pos1;
  it->uv_bits_ = pos3 - pos2;
  it->bit_count_[segment][i16] += it->luma_bits_;
  it->bit_count_[segment][2] += it->uv_bits_;
  VP8IteratorBytesToNz(it);
}

// Same as CodeResiduals, but doesn't actually write anything.
// Instead, it just records the event distribution.
static void RecordResiduals(VP8EncIterator* const it,
                            const VP8ModeScore* const rd) {
  int x, y, ch;
  VP8Residual res;
  VP8Encoder* const enc = it->enc_;

  VP8IteratorNzToBytes(it);

  if (it->mb_->type_ == 1) {   // i16x16
    VP8InitResidual(0, 1, enc, &res);
    VP8SetResidualCoeffs(rd->y_dc_levels, &res);
    it->top_nz_[8] = it->left_nz_[8] =
      VP8RecordCoeffs(it->top_nz_[8] + it->left_nz_[8], &res);
    VP8InitResidual(1, 0, enc, &res);
  } else {
    VP8InitResidual(0, 3, enc, &res);
  }

  // luma-AC
  for (y = 0; y < 4; ++y) {
    for (x = 0; x < 4; ++x) {
      const int ctx = it->top_nz_[x] + it->left_nz_[y];
      VP8SetResidualCoeffs(rd->y_ac_levels[x + y * 4], &res);
      it->top_nz_[x] = it->left_nz_[y] = VP8RecordCoeffs(ctx, &res);
    }
  }

  // U/V
  VP8InitResidual(0, 2, enc, &res);
  for (ch = 0; ch <= 2; ch += 2) {
    for (y = 0; y < 2; ++y) {
      for (x = 0; x < 2; ++x) {
        const int ctx = it->top_nz_[4 + ch + x] + it->left_nz_[4 + ch + y];
        VP8SetResidualCoeffs(rd->uv_levels[ch * 2 + x + y * 2], &res);
        it->top_nz_[4 + ch + x] = it->left_nz_[4 + ch + y] =
            VP8RecordCoeffs(ctx, &res);
      }
    }
  }

  VP8IteratorBytesToNz(it);
}

//------------------------------------------------------------------------------
// Token buffer

#if !defined(DISABLE_TOKEN_BUFFER)

static int RecordTokens(VP8EncIterator* const it, const VP8ModeScore* const rd,
                        VP8TBuffer* const tokens) {
  int x, y, ch;
  VP8Residual res;
  VP8Encoder* const enc = it->enc_;

  VP8IteratorNzToBytes(it);
  if (it->mb_->type_ == 1) {   // i16x16
    const int ctx = it->top_nz_[8] + it->left_nz_[8];
    VP8InitResidual(0, 1, enc, &res);
    VP8SetResidualCoeffs(rd->y_dc_levels, &res);
    it->top_nz_[8] = it->left_nz_[8] =
        VP8RecordCoeffTokens(ctx, 1,
                             res.first, res.last, res.coeffs, tokens);
    VP8RecordCoeffs(ctx, &res);
    VP8InitResidual(1, 0, enc, &res);
  } else {
    VP8InitResidual(0, 3, enc, &res);
  }

  // luma-AC
  for (y = 0; y < 4; ++y) {
    for (x = 0; x < 4; ++x) {
      const int ctx = it->top_nz_[x] + it->left_nz_[y];
      VP8SetResidualCoeffs(rd->y_ac_levels[x + y * 4], &res);
      it->top_nz_[x] = it->left_nz_[y] =
          VP8RecordCoeffTokens(ctx, res.coeff_type,
                               res.first, res.last, res.coeffs, tokens);
      VP8RecordCoeffs(ctx, &res);
    }
  }

  // U/V
  VP8InitResidual(0, 2, enc, &res);
  for (ch = 0; ch <= 2; ch += 2) {
    for (y = 0; y < 2; ++y) {
      for (x = 0; x < 2; ++x) {
        const int ctx = it->top_nz_[4 + ch + x] + it->left_nz_[4 + ch + y];
        VP8SetResidualCoeffs(rd->uv_levels[ch * 2 + x + y * 2], &res);
        it->top_nz_[4 + ch + x] = it->left_nz_[4 + ch + y] =
            VP8RecordCoeffTokens(ctx, 2,
                                 res.first, res.last, res.coeffs, tokens);
        VP8RecordCoeffs(ctx, &res);
      }
    }
  }
  VP8IteratorBytesToNz(it);
  return !tokens->error_;
}

#endif    // !DISABLE_TOKEN_BUFFER

//------------------------------------------------------------------------------
// ExtraInfo map / Debug function

#if SEGMENT_VISU
static void SetBlock(uint8_t* p, int value, int size) {
  int y;
  for (y = 0; y < size; ++y) {
    memset(p, value, size);
    p += BPS;
  }
}
#endif

static void ResetSSE(VP8Encoder* const enc) {
  enc->sse_[0] = 0;
  enc->sse_[1] = 0;
  enc->sse_[2] = 0;
  // Note: enc->sse_[3] is managed by alpha.c
  enc->sse_count_ = 0;
}

static void StoreSSE(const VP8EncIterator* const it) {
  VP8Encoder* const enc = it->enc_;
  const uint8_t* const in = it->yuv_in_;
  const uint8_t* const out = it->yuv_out_;
  // Note: not totally accurate at boundary. And doesn't include in-loop filter.
  enc->sse_[0] += VP8SSE16x16(in + Y_OFF_ENC, out + Y_OFF_ENC);
  enc->sse_[1] += VP8SSE8x8(in + U_OFF_ENC, out + U_OFF_ENC);
  enc->sse_[2] += VP8SSE8x8(in + V_OFF_ENC, out + V_OFF_ENC);
  enc->sse_count_ += 16 * 16;
}

static void StoreSideInfo(const VP8EncIterator* const it) {
  VP8Encoder* const enc = it->enc_;
  const VP8MBInfo* const mb = it->mb_;
  WebPPicture* const pic = enc->pic_;

  if (pic->stats != NULL) {
    StoreSSE(it);
    enc->block_count_[0] += (mb->type_ == 0);
    enc->block_count_[1] += (mb->type_ == 1);
    enc->block_count_[2] += (mb->skip_ != 0);
  }

  if (pic->extra_info != NULL) {
    uint8_t* const info = &pic->extra_info[it->x_ + it->y_ * enc->mb_w_];
    switch (pic->extra_info_type) {
      case 1: *info = mb->type_; break;
      case 2: *info = mb->segment_; break;
      case 3: *info = enc->dqm_[mb->segment_].quant_; break;
      case 4: *info = (mb->type_ == 1) ? it->preds_[0] : 0xff; break;
      case 5: *info = mb->uv_mode_; break;
      case 6: {
        const int b = (int)((it->luma_bits_ + it->uv_bits_ + 7) >> 3);
        *info = (b > 255) ? 255 : b; break;
      }
      case 7: *info = mb->alpha_; break;
      default: *info = 0; break;
    }
  }
#if SEGMENT_VISU  // visualize segments and prediction modes
  SetBlock(it->yuv_out_ + Y_OFF_ENC, mb->segment_ * 64, 16);
  SetBlock(it->yuv_out_ + U_OFF_ENC, it->preds_[0] * 64, 8);
  SetBlock(it->yuv_out_ + V_OFF_ENC, mb->uv_mode_ * 64, 8);
#endif
}

static double GetPSNR(uint64_t mse, uint64_t size) {
  return (mse > 0 && size > 0) ? 10. * log10(255. * 255. * size / mse) : 99;
}

//------------------------------------------------------------------------------
//  StatLoop(): only collect statistics (number of skips, token usage, ...).
//  This is used for deciding optimal probabilities. It also modifies the
//  quantizer value if some target (size, PSNR) was specified.

static void SetLoopParams(VP8Encoder* const enc, float q) {
  // Make sure the quality parameter is inside valid bounds
  q = Clamp(q, 0.f, 100.f);

  VP8SetSegmentParams(enc, q);      // setup segment quantizations and filters
  SetSegmentProbas(enc);            // compute segment probabilities

  ResetStats(enc);
  ResetSSE(enc);
}

static uint64_t OneStatPass(VP8Encoder* const enc, VP8RDLevel rd_opt,
                            int nb_mbs, int percent_delta,
                            PassStats* const s) {
  VP8EncIterator it;
  uint64_t size = 0;
  uint64_t size_p0 = 0;
  uint64_t distortion = 0;
  const uint64_t pixel_count = nb_mbs * 384;

  VP8IteratorInit(enc, &it);
  SetLoopParams(enc, s->q);
  do {
    VP8ModeScore info;
    VP8IteratorImport(&it, NULL);
    if (VP8Decimate(&it, &info, rd_opt)) {
      // Just record the number of skips and act like skip_proba is not used.
      enc->proba_.nb_skip_++;
    }
    RecordResiduals(&it, &info);
    size += info.R + info.H;
    size_p0 += info.H;
    distortion += info.D;
    if (percent_delta && !VP8IteratorProgress(&it, percent_delta))
      return 0;
    VP8IteratorSaveBoundary(&it);
  } while (VP8IteratorNext(&it) && --nb_mbs > 0);

  size_p0 += enc->segment_hdr_.size_;
  if (s->do_size_search) {
    size += FinalizeSkipProba(enc);
    size += FinalizeTokenProbas(&enc->proba_);
    size = ((size + size_p0 + 1024) >> 11) + HEADER_SIZE_ESTIMATE;
    s->value = (double)size;
  } else {
    s->value = GetPSNR(distortion, pixel_count);
  }
  return size_p0;
}

static int StatLoop(VP8Encoder* const enc) {
  const int method = enc->method_;
  const int do_search = enc->do_search_;
  const int fast_probe = ((method == 0 || method == 3) && !do_search);
  int num_pass_left = enc->config_->pass;
  const int task_percent = 20;
  const int percent_per_pass =
      (task_percent + num_pass_left / 2) / num_pass_left;
  const int final_percent = enc->percent_ + task_percent;
  const VP8RDLevel rd_opt =
      (method >= 3 || do_search) ? RD_OPT_BASIC : RD_OPT_NONE;
  int nb_mbs = enc->mb_w_ * enc->mb_h_;
  PassStats stats;

  InitPassStats(enc, &stats);
  ResetTokenStats(enc);

  // Fast mode: quick analysis pass over few mbs. Better than nothing.
  if (fast_probe) {
    if (method == 3) {  // we need more stats for method 3 to be reliable.
      nb_mbs = (nb_mbs > 200) ? nb_mbs >> 1 : 100;
    } else {
      nb_mbs = (nb_mbs > 200) ? nb_mbs >> 2 : 50;
    }
  }

  while (num_pass_left-- > 0) {
    const int is_last_pass = (fabs(stats.dq) <= DQ_LIMIT) ||
                             (num_pass_left == 0) ||
                             (enc->max_i4_header_bits_ == 0);
    const uint64_t size_p0 =
        OneStatPass(enc, rd_opt, nb_mbs, percent_per_pass, &stats);
    if (size_p0 == 0) return 0;
#if (DEBUG_SEARCH > 0)
    printf("#%d value:%.1lf -> %.1lf   q:%.2f -> %.2f\n",
           num_pass_left, stats.last_value, stats.value, stats.last_q, stats.q);
#endif
    if (enc->max_i4_header_bits_ > 0 && size_p0 > PARTITION0_SIZE_LIMIT) {
      ++num_pass_left;
      enc->max_i4_header_bits_ >>= 1;  // strengthen header bit limitation...
      continue;                        // ...and start over
    }
    if (is_last_pass) {
      break;
    }
    // If no target size: just do several pass without changing 'q'
    if (do_search) {
      ComputeNextQ(&stats);
      if (fabs(stats.dq) <= DQ_LIMIT) break;
    }
  }
  if (!do_search || !stats.do_size_search) {
    // Need to finalize probas now, since it wasn't done during the search.
    FinalizeSkipProba(enc);
    FinalizeTokenProbas(&enc->proba_);
  }
  VP8CalculateLevelCosts(&enc->proba_);  // finalize costs
  return WebPReportProgress(enc->pic_, final_percent, &enc->percent_);
}

//------------------------------------------------------------------------------
// Main loops
//

static const int kAverageBytesPerMB[8] = { 50, 24, 16, 9, 7, 5, 3, 2 };

static int PreLoopInitialize(VP8Encoder* const enc) {
  int p;
  int ok = 1;
  const int average_bytes_per_MB = kAverageBytesPerMB[enc->base_quant_ >> 4];
  const int bytes_per_parts =
      enc->mb_w_ * enc->mb_h_ * average_bytes_per_MB / enc->num_parts_;
  // Initialize the bit-writers
  for (p = 0; ok && p < enc->num_parts_; ++p) {
    ok = VP8BitWriterInit(enc->parts_ + p, bytes_per_parts);
  }
  if (!ok) {
    VP8EncFreeBitWriters(enc);  // malloc error occurred
    WebPEncodingSetError(enc->pic_, VP8_ENC_ERROR_OUT_OF_MEMORY);
  }
  return ok;
}

static int PostLoopFinalize(VP8EncIterator* const it, int ok) {
  VP8Encoder* const enc = it->enc_;
  if (ok) {      // Finalize the partitions, check for extra errors.
    int p;
    for (p = 0; p < enc->num_parts_; ++p) {
      VP8BitWriterFinish(enc->parts_ + p);
      ok &= !enc->parts_[p].error_;
    }
  }

  if (ok) {      // All good. Finish up.
    if (enc->pic_->stats != NULL) {  // finalize byte counters...
      int i, s;
      for (i = 0; i <= 2; ++i) {
        for (s = 0; s < NUM_MB_SEGMENTS; ++s) {
          enc->residual_bytes_[i][s] = (int)((it->bit_count_[s][i] + 7) >> 3);
        }
      }
    }
    VP8AdjustFilterStrength(it);     // ...and store filter stats.
  } else {
    // Something bad happened -> need to do some memory cleanup.
    VP8EncFreeBitWriters(enc);
  }
  return ok;
}

//------------------------------------------------------------------------------
//  VP8EncLoop(): does the final bitstream coding.

static void ResetAfterSkip(VP8EncIterator* const it) {
  if (it->mb_->type_ == 1) {
    *it->nz_ = 0;  // reset all predictors
    it->left_nz_[8] = 0;
  } else {
    *it->nz_ &= (1 << 24);  // preserve the dc_nz bit
  }
}

int VP8EncLoop(VP8Encoder* const enc) {
  VP8EncIterator it;
  int ok = PreLoopInitialize(enc);
  if (!ok) return 0;

  StatLoop(enc);  // stats-collection loop

  VP8IteratorInit(enc, &it);
  VP8InitFilter(&it);
  do {
    VP8ModeScore info;
    const int dont_use_skip = !enc->proba_.use_skip_proba_;
    const VP8RDLevel rd_opt = enc->rd_opt_level_;

    VP8IteratorImport(&it, NULL);
    // Warning! order is important: first call VP8Decimate() and
    // *then* decide how to code the skip decision if there's one.
    if (!VP8Decimate(&it, &info, rd_opt) || dont_use_skip) {
      CodeResiduals(it.bw_, &it, &info);
    } else {   // reset predictors after a skip
      ResetAfterSkip(&it);
    }
    StoreSideInfo(&it);
    VP8StoreFilterStats(&it);
    VP8IteratorExport(&it);
    ok = VP8IteratorProgress(&it, 20);
    VP8IteratorSaveBoundary(&it);
  } while (ok && VP8IteratorNext(&it));

  return PostLoopFinalize(&it, ok);
}

//------------------------------------------------------------------------------
// Single pass using Token Buffer.

#if !defined(DISABLE_TOKEN_BUFFER)

#define MIN_COUNT 96  // minimum number of macroblocks before updating stats

int VP8EncTokenLoop(VP8Encoder* const enc) {
  // Roughly refresh the proba eight times per pass
  int max_count = (enc->mb_w_ * enc->mb_h_) >> 3;
  int num_pass_left = enc->config_->pass;
  const int do_search = enc->do_search_;
  VP8EncIterator it;
  VP8EncProba* const proba = &enc->proba_;
  const VP8RDLevel rd_opt = enc->rd_opt_level_;
  const uint64_t pixel_count = enc->mb_w_ * enc->mb_h_ * 384;
  PassStats stats;
  int ok;

  InitPassStats(enc, &stats);
  ok = PreLoopInitialize(enc);
  if (!ok) return 0;

  if (max_count < MIN_COUNT) max_count = MIN_COUNT;

  assert(enc->num_parts_ == 1);
  assert(enc->use_tokens_);
  assert(proba->use_skip_proba_ == 0);
  assert(rd_opt >= RD_OPT_BASIC);   // otherwise, token-buffer won't be useful
  assert(num_pass_left > 0);

  while (ok && num_pass_left-- > 0) {
    const int is_last_pass = (fabs(stats.dq) <= DQ_LIMIT) ||
                             (num_pass_left == 0) ||
                             (enc->max_i4_header_bits_ == 0);
    uint64_t size_p0 = 0;
    uint64_t distortion = 0;
    int cnt = max_count;
    VP8IteratorInit(enc, &it);
    SetLoopParams(enc, stats.q);
    if (is_last_pass) {
      ResetTokenStats(enc);
      VP8InitFilter(&it);  // don't collect stats until last pass (too costly)
    }
    VP8TBufferClear(&enc->tokens_);
    do {
      VP8ModeScore info;
      VP8IteratorImport(&it, NULL);
      if (--cnt < 0) {
        FinalizeTokenProbas(proba);
        VP8CalculateLevelCosts(proba);  // refresh cost tables for rd-opt
        cnt = max_count;
      }
      VP8Decimate(&it, &info, rd_opt);
      ok = RecordTokens(&it, &info, &enc->tokens_);
      if (!ok) {
        WebPEncodingSetError(enc->pic_, VP8_ENC_ERROR_OUT_OF_MEMORY);
        break;
      }
      size_p0 += info.H;
      distortion += info.D;
      if (is_last_pass) {
        StoreSideInfo(&it);
        VP8StoreFilterStats(&it);
        VP8IteratorExport(&it);
        ok = VP8IteratorProgress(&it, 20);
      }
      VP8IteratorSaveBoundary(&it);
    } while (ok && VP8IteratorNext(&it));
    if (!ok) break;

    size_p0 += enc->segment_hdr_.size_;
    if (stats.do_size_search) {
      uint64_t size = FinalizeTokenProbas(&enc->proba_);
      size += VP8EstimateTokenSize(&enc->tokens_,
                                   (const uint8_t*)proba->coeffs_);
      size = (size + size_p0 + 1024) >> 11;  // -> size in bytes
      size += HEADER_SIZE_ESTIMATE;
      stats.value = (double)size;
    } else {  // compute and store PSNR
      stats.value = GetPSNR(distortion, pixel_count);
    }

#if (DEBUG_SEARCH > 0)
    printf("#%2d metric:%.1lf -> %.1lf   last_q=%.2lf q=%.2lf dq=%.2lf\n",
           num_pass_left, stats.last_value, stats.value,
           stats.last_q, stats.q, stats.dq);
#endif
    if (size_p0 > PARTITION0_SIZE_LIMIT) {
      ++num_pass_left;
      enc->max_i4_header_bits_ >>= 1;  // strengthen header bit limitation...
      continue;                        // ...and start over
    }
    if (is_last_pass) {
      break;   // done
    }
    if (do_search) {
      ComputeNextQ(&stats);  // Adjust q
    }
  }
  if (ok) {
    if (!stats.do_size_search) {
      FinalizeTokenProbas(&enc->proba_);
    }
    ok = VP8EmitTokens(&enc->tokens_, enc->parts_ + 0,
                       (const uint8_t*)proba->coeffs_, 1);
  }
  ok = ok && WebPReportProgress(enc->pic_, enc->percent_ + 20, &enc->percent_);
  return PostLoopFinalize(&it, ok);
}

#else

int VP8EncTokenLoop(VP8Encoder* const enc) {
  (void)enc;
  return 0;   // we shouldn't be here.
}

#endif    // DISABLE_TOKEN_BUFFER

//------------------------------------------------------------------------------

