// Copyright 2011 Google Inc. All Rights Reserved.
//
// This code is licensed under the same terms as WebM:
//  Software License Agreement:  http://www.webmproject.org/license/software/
//  Additional IP Rights Grant:  http://www.webmproject.org/license/additional/
// -----------------------------------------------------------------------------
//
//   frame coding and analysis
//
// Author: Skal (pascal.massimino@gmail.com)

#include <assert.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "./vp8enci.h"
#include "./cost.h"

#if defined(__cplusplus) || defined(c_plusplus)
extern "C" {
#endif

#define SEGMENT_VISU 0
#define DEBUG_SEARCH 0    // useful to track search convergence

// On-the-fly info about the current set of residuals. Handy to avoid
// passing zillions of params.
typedef struct {
  int first;
  int last;
  const int16_t* coeffs;

  int coeff_type;
  ProbaArray* prob;
  StatsArray* stats;
  CostArray*  cost;
} VP8Residual;

//------------------------------------------------------------------------------
// Tables for level coding

const uint8_t VP8EncBands[16 + 1] = {
  0, 1, 2, 3, 6, 4, 5, 6, 6, 6, 6, 6, 6, 6, 6, 7,
  0  // sentinel
};

static const uint8_t kCat3[] = { 173, 148, 140 };
static const uint8_t kCat4[] = { 176, 155, 140, 135 };
static const uint8_t kCat5[] = { 180, 157, 141, 134, 130 };
static const uint8_t kCat6[] =
    { 254, 254, 243, 230, 196, 177, 153, 140, 133, 130, 129 };

//------------------------------------------------------------------------------
// Reset the statistics about: number of skips, token proba, level cost,...

static void ResetStats(VP8Encoder* const enc) {
  VP8Proba* const proba = &enc->proba_;
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
  VP8Proba* const proba = &enc->proba_;
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

//------------------------------------------------------------------------------
// Recording of token probabilities.

static void ResetTokenStats(VP8Encoder* const enc) {
  VP8Proba* const proba = &enc->proba_;
  memset(proba->stats_, 0, sizeof(proba->stats_));
}

// Record proba context used
static int Record(int bit, proba_t* const stats) {
  proba_t p = *stats;
  if (p >= 0xffff0000u) {               // an overflow is inbound.
    p = ((p + 1u) >> 1) & 0x7fff7fffu;  // -> divide the stats by 2.
  }
  // record bit count (lower 16 bits) and increment total count (upper 16 bits).
  p += 0x00010000u + bit;
  *stats = p;
  return bit;
}

// We keep the table free variant around for reference, in case.
#define USE_LEVEL_CODE_TABLE

// Simulate block coding, but only record statistics.
// Note: no need to record the fixed probas.
static int RecordCoeffs(int ctx, const VP8Residual* const res) {
  int n = res->first;
  proba_t* s = res->stats[VP8EncBands[n]][ctx];
  if (res->last  < 0) {
    Record(0, s + 0);
    return 0;
  }
  while (n <= res->last) {
    int v;
    Record(1, s + 0);
    while ((v = res->coeffs[n++]) == 0) {
      Record(0, s + 1);
      s = res->stats[VP8EncBands[n]][0];
    }
    Record(1, s + 1);
    if (!Record(2u < (unsigned int)(v + 1), s + 2)) {  // v = -1 or 1
      s = res->stats[VP8EncBands[n]][1];
    } else {
      v = abs(v);
#if !defined(USE_LEVEL_CODE_TABLE)
      if (!Record(v > 4, s + 3)) {
        if (Record(v != 2, s + 4))
          Record(v == 4, s + 5);
      } else if (!Record(v > 10, s + 6)) {
        Record(v > 6, s + 7);
      } else if (!Record((v >= 3 + (8 << 2)), s + 8)) {
        Record((v >= 3 + (8 << 1)), s + 9);
      } else {
        Record((v >= 3 + (8 << 3)), s + 10);
      }
#else
      if (v > MAX_VARIABLE_LEVEL)
        v = MAX_VARIABLE_LEVEL;

      {
        const int bits = VP8LevelCodes[v - 1][1];
        int pattern = VP8LevelCodes[v - 1][0];
        int i;
        for (i = 0; (pattern >>= 1) != 0; ++i) {
          const int mask = 2 << i;
          if (pattern & 1) Record(!!(bits & mask), s + 3 + i);
        }
      }
#endif
      s = res->stats[VP8EncBands[n]][2];
    }
  }
  if (n < 16) Record(0, s + 0);
  return 1;
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

static int FinalizeTokenProbas(VP8Encoder* const enc) {
  VP8Proba* const proba = &enc->proba_;
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
// helper functions for residuals struct VP8Residual.

static void InitResidual(int first, int coeff_type,
                         VP8Encoder* const enc, VP8Residual* const res) {
  res->coeff_type = coeff_type;
  res->prob  = enc->proba_.coeffs_[coeff_type];
  res->stats = enc->proba_.stats_[coeff_type];
  res->cost  = enc->proba_.level_cost_[coeff_type];
  res->first = first;
}

static void SetResidualCoeffs(const int16_t* const coeffs,
                              VP8Residual* const res) {
  int n;
  res->last = -1;
  for (n = 15; n >= res->first; --n) {
    if (coeffs[n]) {
      res->last = n;
      break;
    }
  }
  res->coeffs = coeffs;
}

//------------------------------------------------------------------------------
// Mode costs

static int GetResidualCost(int ctx, const VP8Residual* const res) {
  int n = res->first;
  int p0 = res->prob[VP8EncBands[n]][ctx][0];
  const uint16_t* t = res->cost[VP8EncBands[n]][ctx];
  int cost;

  if (res->last < 0) {
    return VP8BitCost(0, p0);
  }
  cost = 0;
  while (n <= res->last) {
    const int v = res->coeffs[n];
    const int b = VP8EncBands[n + 1];
    ++n;
    if (v == 0) {
      // short-case for VP8LevelCost(t, 0) (note: VP8LevelFixedCosts[0] == 0):
      cost += t[0];
      t = res->cost[b][0];
      continue;
    }
    cost += VP8BitCost(1, p0);
    if (2u >= (unsigned int)(v + 1)) {   // v = -1 or 1
      // short-case for "VP8LevelCost(t, 1)" (256 is VP8LevelFixedCosts[1]):
      cost += 256 + t[1];
      p0 = res->prob[b][1][0];
      t = res->cost[b][1];
    } else {
      cost += VP8LevelCost(t, abs(v));
      p0 = res->prob[b][2][0];
      t = res->cost[b][2];
    }
  }
  if (n < 16) cost += VP8BitCost(0, p0);
  return cost;
}

int VP8GetCostLuma4(VP8EncIterator* const it, const int16_t levels[16]) {
  const int x = (it->i4_ & 3), y = (it->i4_ >> 2);
  VP8Residual res;
  VP8Encoder* const enc = it->enc_;
  int R = 0;
  int ctx;

  InitResidual(0, 3, enc, &res);
  ctx = it->top_nz_[x] + it->left_nz_[y];
  SetResidualCoeffs(levels, &res);
  R += GetResidualCost(ctx, &res);
  return R;
}

int VP8GetCostLuma16(VP8EncIterator* const it, const VP8ModeScore* const rd) {
  VP8Residual res;
  VP8Encoder* const enc = it->enc_;
  int x, y;
  int R = 0;

  VP8IteratorNzToBytes(it);   // re-import the non-zero context

  // DC
  InitResidual(0, 1, enc, &res);
  SetResidualCoeffs(rd->y_dc_levels, &res);
  R += GetResidualCost(it->top_nz_[8] + it->left_nz_[8], &res);

  // AC
  InitResidual(1, 0, enc, &res);
  for (y = 0; y < 4; ++y) {
    for (x = 0; x < 4; ++x) {
      const int ctx = it->top_nz_[x] + it->left_nz_[y];
      SetResidualCoeffs(rd->y_ac_levels[x + y * 4], &res);
      R += GetResidualCost(ctx, &res);
      it->top_nz_[x] = it->left_nz_[y] = (res.last >= 0);
    }
  }
  return R;
}

int VP8GetCostUV(VP8EncIterator* const it, const VP8ModeScore* const rd) {
  VP8Residual res;
  VP8Encoder* const enc = it->enc_;
  int ch, x, y;
  int R = 0;

  VP8IteratorNzToBytes(it);  // re-import the non-zero context

  InitResidual(0, 2, enc, &res);
  for (ch = 0; ch <= 2; ch += 2) {
    for (y = 0; y < 2; ++y) {
      for (x = 0; x < 2; ++x) {
        const int ctx = it->top_nz_[4 + ch + x] + it->left_nz_[4 + ch + y];
        SetResidualCoeffs(rd->uv_levels[ch * 2 + x + y * 2], &res);
        R += GetResidualCost(ctx, &res);
        it->top_nz_[4 + ch + x] = it->left_nz_[4 + ch + y] = (res.last >= 0);
      }
    }
  }
  return R;
}

//------------------------------------------------------------------------------
// Coefficient coding

static int PutCoeffs(VP8BitWriter* const bw, int ctx, const VP8Residual* res) {
  int n = res->first;
  const uint8_t* p = res->prob[VP8EncBands[n]][ctx];
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
        if (v < 3 + (8 << 1)) {          // kCat3  (3b)
          VP8PutBit(bw, 0, p[8]);
          VP8PutBit(bw, 0, p[9]);
          v -= 3 + (8 << 0);
          mask = 1 << 2;
          tab = kCat3;
        } else if (v < 3 + (8 << 2)) {   // kCat4  (4b)
          VP8PutBit(bw, 0, p[8]);
          VP8PutBit(bw, 1, p[9]);
          v -= 3 + (8 << 1);
          mask = 1 << 3;
          tab = kCat4;
        } else if (v < 3 + (8 << 3)) {   // kCat5  (5b)
          VP8PutBit(bw, 1, p[8]);
          VP8PutBit(bw, 0, p[10]);
          v -= 3 + (8 << 2);
          mask = 1 << 4;
          tab = kCat5;
        } else {                         // kCat6 (11b)
          VP8PutBit(bw, 1, p[8]);
          VP8PutBit(bw, 1, p[10]);
          v -= 3 + (8 << 3);
          mask = 1 << 10;
          tab = kCat6;
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

static void CodeResiduals(VP8BitWriter* const bw,
                          VP8EncIterator* const it,
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
    InitResidual(0, 1, enc, &res);
    SetResidualCoeffs(rd->y_dc_levels, &res);
    it->top_nz_[8] = it->left_nz_[8] =
      PutCoeffs(bw, it->top_nz_[8] + it->left_nz_[8], &res);
    InitResidual(1, 0, enc, &res);
  } else {
    InitResidual(0, 3, enc, &res);
  }

  // luma-AC
  for (y = 0; y < 4; ++y) {
    for (x = 0; x < 4; ++x) {
      const int ctx = it->top_nz_[x] + it->left_nz_[y];
      SetResidualCoeffs(rd->y_ac_levels[x + y * 4], &res);
      it->top_nz_[x] = it->left_nz_[y] = PutCoeffs(bw, ctx, &res);
    }
  }
  pos2 = VP8BitWriterPos(bw);

  // U/V
  InitResidual(0, 2, enc, &res);
  for (ch = 0; ch <= 2; ch += 2) {
    for (y = 0; y < 2; ++y) {
      for (x = 0; x < 2; ++x) {
        const int ctx = it->top_nz_[4 + ch + x] + it->left_nz_[4 + ch + y];
        SetResidualCoeffs(rd->uv_levels[ch * 2 + x + y * 2], &res);
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
    InitResidual(0, 1, enc, &res);
    SetResidualCoeffs(rd->y_dc_levels, &res);
    it->top_nz_[8] = it->left_nz_[8] =
      RecordCoeffs(it->top_nz_[8] + it->left_nz_[8], &res);
    InitResidual(1, 0, enc, &res);
  } else {
    InitResidual(0, 3, enc, &res);
  }

  // luma-AC
  for (y = 0; y < 4; ++y) {
    for (x = 0; x < 4; ++x) {
      const int ctx = it->top_nz_[x] + it->left_nz_[y];
      SetResidualCoeffs(rd->y_ac_levels[x + y * 4], &res);
      it->top_nz_[x] = it->left_nz_[y] = RecordCoeffs(ctx, &res);
    }
  }

  // U/V
  InitResidual(0, 2, enc, &res);
  for (ch = 0; ch <= 2; ch += 2) {
    for (y = 0; y < 2; ++y) {
      for (x = 0; x < 2; ++x) {
        const int ctx = it->top_nz_[4 + ch + x] + it->left_nz_[4 + ch + y];
        SetResidualCoeffs(rd->uv_levels[ch * 2 + x + y * 2], &res);
        it->top_nz_[4 + ch + x] = it->left_nz_[4 + ch + y] =
            RecordCoeffs(ctx, &res);
      }
    }
  }

  VP8IteratorBytesToNz(it);
}

//------------------------------------------------------------------------------
// Token buffer

#ifdef USE_TOKEN_BUFFER

void VP8TBufferInit(VP8TBuffer* const b) {
  b->rows_ = NULL;
  b->tokens_ = NULL;
  b->last_ = &b->rows_;
  b->left_ = 0;
  b->error_ = 0;
}

int VP8TBufferNewPage(VP8TBuffer* const b) {
  VP8Tokens* const page = b->error_ ? NULL : (VP8Tokens*)malloc(sizeof(*page));
  if (page == NULL) {
    b->error_ = 1;
    return 0;
  }
  *b->last_ = page;
  b->last_ = &page->next_;
  b->left_ = MAX_NUM_TOKEN;
  b->tokens_ = page->tokens_;
  return 1;
}

void VP8TBufferClear(VP8TBuffer* const b) {
  if (b != NULL) {
    const VP8Tokens* p = b->rows_;
    while (p != NULL) {
      const VP8Tokens* const next = p->next_;
      free((void*)p);
      p = next;
    }
    VP8TBufferInit(b);
  }
}

int VP8EmitTokens(const VP8TBuffer* const b, VP8BitWriter* const bw,
                  const uint8_t* const probas) {
  VP8Tokens* p = b->rows_;
  if (b->error_) return 0;
  while (p != NULL) {
    const int N = (p->next_ == NULL) ? b->left_ : 0;
    int n = MAX_NUM_TOKEN;
    while (n-- > N) {
      VP8PutBit(bw, (p->tokens_[n] >> 15) & 1, probas[p->tokens_[n] & 0x7fff]);
    }
    p = p->next_;
  }
  return 1;
}

#define TOKEN_ID(b, ctx, p) ((p) + NUM_PROBAS * ((ctx) + (b) * NUM_CTX))

static int RecordCoeffTokens(int ctx, const VP8Residual* const res,
                             VP8TBuffer* tokens) {
  int n = res->first;
  int b = VP8EncBands[n];
  if (!VP8AddToken(tokens, res->last >= 0, TOKEN_ID(b, ctx, 0))) {
    return 0;
  }

  while (n < 16) {
    const int c = res->coeffs[n++];
    const int sign = c < 0;
    int v = sign ? -c : c;
    const int base_id = TOKEN_ID(b, ctx, 0);
    if (!VP8AddToken(tokens, v != 0, base_id + 1)) {
      b = VP8EncBands[n];
      ctx = 0;
      continue;
    }
    if (!VP8AddToken(tokens, v > 1, base_id + 2)) {
      b = VP8EncBands[n];
      ctx = 1;
    } else {
      if (!VP8AddToken(tokens, v > 4, base_id + 3)) {
        if (VP8AddToken(tokens, v != 2, base_id + 4))
          VP8AddToken(tokens, v == 4, base_id + 5);
      } else if (!VP8AddToken(tokens, v > 10, base_id + 6)) {
        if (!VP8AddToken(tokens, v > 6, base_id + 7)) {
//          VP8AddToken(tokens, v == 6, 159);
        } else {
//          VP8AddToken(tokens, v >= 9, 165);
//          VP8AddToken(tokens, !(v & 1), 145);
        }
      } else {
        int mask;
        const uint8_t* tab;
        if (v < 3 + (8 << 1)) {          // kCat3  (3b)
          VP8AddToken(tokens, 0, base_id + 8);
          VP8AddToken(tokens, 0, base_id + 9);
          v -= 3 + (8 << 0);
          mask = 1 << 2;
          tab = kCat3;
        } else if (v < 3 + (8 << 2)) {   // kCat4  (4b)
          VP8AddToken(tokens, 0, base_id + 8);
          VP8AddToken(tokens, 1, base_id + 9);
          v -= 3 + (8 << 1);
          mask = 1 << 3;
          tab = kCat4;
        } else if (v < 3 + (8 << 3)) {   // kCat5  (5b)
          VP8AddToken(tokens, 1, base_id + 8);
          VP8AddToken(tokens, 0, base_id + 10);
          v -= 3 + (8 << 2);
          mask = 1 << 4;
          tab = kCat5;
        } else {                         // kCat6 (11b)
          VP8AddToken(tokens, 1, base_id + 8);
          VP8AddToken(tokens, 1, base_id + 10);
          v -= 3 + (8 << 3);
          mask = 1 << 10;
          tab = kCat6;
        }
        while (mask) {
          // VP8AddToken(tokens, !!(v & mask), *tab++);
          mask >>= 1;
        }
      }
      ctx = 2;
    }
    b = VP8EncBands[n];
    // VP8PutBitUniform(bw, sign);
    if (n == 16 || !VP8AddToken(tokens, n <= res->last, TOKEN_ID(b, ctx, 0))) {
      return 1;   // EOB
    }
  }
  return 1;
}

static void RecordTokens(VP8EncIterator* const it,
                         const VP8ModeScore* const rd, VP8TBuffer tokens[2]) {
  int x, y, ch;
  VP8Residual res;
  VP8Encoder* const enc = it->enc_;

  VP8IteratorNzToBytes(it);
  if (it->mb_->type_ == 1) {   // i16x16
    InitResidual(0, 1, enc, &res);
    SetResidualCoeffs(rd->y_dc_levels, &res);
// TODO(skal): FIX ->    it->top_nz_[8] = it->left_nz_[8] =
      RecordCoeffTokens(it->top_nz_[8] + it->left_nz_[8], &res, &tokens[0]);
    InitResidual(1, 0, enc, &res);
  } else {
    InitResidual(0, 3, enc, &res);
  }

  // luma-AC
  for (y = 0; y < 4; ++y) {
    for (x = 0; x < 4; ++x) {
      const int ctx = it->top_nz_[x] + it->left_nz_[y];
      SetResidualCoeffs(rd->y_ac_levels[x + y * 4], &res);
      it->top_nz_[x] = it->left_nz_[y] =
          RecordCoeffTokens(ctx, &res, &tokens[0]);
    }
  }

  // U/V
  InitResidual(0, 2, enc, &res);
  for (ch = 0; ch <= 2; ch += 2) {
    for (y = 0; y < 2; ++y) {
      for (x = 0; x < 2; ++x) {
        const int ctx = it->top_nz_[4 + ch + x] + it->left_nz_[4 + ch + y];
        SetResidualCoeffs(rd->uv_levels[ch * 2 + x + y * 2], &res);
        it->top_nz_[4 + ch + x] = it->left_nz_[4 + ch + y] =
            RecordCoeffTokens(ctx, &res, &tokens[1]);
      }
    }
  }
}

#endif    // USE_TOKEN_BUFFER

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
  memset(enc->sse_, 0, sizeof(enc->sse_));
  enc->sse_count_ = 0;
}

static void StoreSSE(const VP8EncIterator* const it) {
  VP8Encoder* const enc = it->enc_;
  const uint8_t* const in = it->yuv_in_;
  const uint8_t* const out = it->yuv_out_;
  // Note: not totally accurate at boundary. And doesn't include in-loop filter.
  enc->sse_[0] += VP8SSE16x16(in + Y_OFF, out + Y_OFF);
  enc->sse_[1] += VP8SSE8x8(in + U_OFF, out + U_OFF);
  enc->sse_[2] += VP8SSE8x8(in + V_OFF, out + V_OFF);
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
      default: *info = 0; break;
    };
  }
#if SEGMENT_VISU  // visualize segments and prediction modes
  SetBlock(it->yuv_out_ + Y_OFF, mb->segment_ * 64, 16);
  SetBlock(it->yuv_out_ + U_OFF, it->preds_[0] * 64, 8);
  SetBlock(it->yuv_out_ + V_OFF, mb->uv_mode_ * 64, 8);
#endif
}

//------------------------------------------------------------------------------
// Main loops
//
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
  int i, s, p;
  int ok = 1;
  VP8EncIterator it;
  VP8ModeScore info;
  const int dont_use_skip = !enc->proba_.use_skip_proba_;
  const int rd_opt = enc->rd_opt_level_;
  const int kAverageBytesPerMB = 5;     // TODO: have a kTable[quality/10]
  const int bytes_per_parts =
    enc->mb_w_ * enc->mb_h_ * kAverageBytesPerMB / enc->num_parts_;

  // Initialize the bit-writers
  for (p = 0; p < enc->num_parts_; ++p) {
    VP8BitWriterInit(enc->parts_ + p, bytes_per_parts);
  }

  ResetStats(enc);
  ResetSSE(enc);

  VP8IteratorInit(enc, &it);
  VP8InitFilter(&it);
  do {
    VP8IteratorImport(&it);
    // Warning! order is important: first call VP8Decimate() and
    // *then* decide how to code the skip decision if there's one.
    if (!VP8Decimate(&it, &info, rd_opt) || dont_use_skip) {
      CodeResiduals(it.bw_, &it, &info);
    } else {   // reset predictors after a skip
      ResetAfterSkip(&it);
    }
#ifdef WEBP_EXPERIMENTAL_FEATURES
    if (enc->use_layer_) {
      VP8EncCodeLayerBlock(&it);
    }
#endif
    StoreSideInfo(&it);
    VP8StoreFilterStats(&it);
    VP8IteratorExport(&it);
    ok = VP8IteratorProgress(&it, 20);
  } while (ok && VP8IteratorNext(&it, it.yuv_out_));

  if (ok) {      // Finalize the partitions, check for extra errors.
    for (p = 0; p < enc->num_parts_; ++p) {
      VP8BitWriterFinish(enc->parts_ + p);
      ok &= !enc->parts_[p].error_;
    }
  }

  if (ok) {      // All good. Finish up.
    if (enc->pic_->stats) {           // finalize byte counters...
      for (i = 0; i <= 2; ++i) {
        for (s = 0; s < NUM_MB_SEGMENTS; ++s) {
          enc->residual_bytes_[i][s] = (int)((it.bit_count_[s][i] + 7) >> 3);
        }
      }
    }
    VP8AdjustFilterStrength(&it);     // ...and store filter stats.
  } else {
    // Something bad happened -> need to do some memory cleanup.
    VP8EncFreeBitWriters(enc);
  }

  return ok;
}

//------------------------------------------------------------------------------
//  VP8StatLoop(): only collect statistics (number of skips, token usage, ...)
//                 This is used for deciding optimal probabilities. It also
//                 modifies the quantizer value if some target (size, PNSR)
//                 was specified.

#define kHeaderSizeEstimate (15 + 20 + 10)      // TODO: fix better

static int OneStatPass(VP8Encoder* const enc, float q, int rd_opt, int nb_mbs,
                       float* const PSNR, int percent_delta) {
  VP8EncIterator it;
  uint64_t size = 0;
  uint64_t distortion = 0;
  const uint64_t pixel_count = nb_mbs * 384;

  // Make sure the quality parameter is inside valid bounds
  if (q < 0.) {
    q = 0;
  } else if (q > 100.) {
    q = 100;
  }

  VP8SetSegmentParams(enc, q);      // setup segment quantizations and filters

  ResetStats(enc);
  ResetTokenStats(enc);

  VP8IteratorInit(enc, &it);
  do {
    VP8ModeScore info;
    VP8IteratorImport(&it);
    if (VP8Decimate(&it, &info, rd_opt)) {
      // Just record the number of skips and act like skip_proba is not used.
      enc->proba_.nb_skip_++;
    }
    RecordResiduals(&it, &info);
    size += info.R;
    distortion += info.D;
    if (percent_delta && !VP8IteratorProgress(&it, percent_delta))
      return 0;
  } while (VP8IteratorNext(&it, it.yuv_out_) && --nb_mbs > 0);
  size += FinalizeSkipProba(enc);
  size += FinalizeTokenProbas(enc);
  size += enc->segment_hdr_.size_;
  size = ((size + 1024) >> 11) + kHeaderSizeEstimate;

  if (PSNR) {
    *PSNR = (float)(10.* log10(255. * 255. * pixel_count / distortion));
  }
  return (int)size;
}

// successive refinement increments.
static const int dqs[] = { 20, 15, 10, 8, 6, 4, 2, 1, 0 };

int VP8StatLoop(VP8Encoder* const enc) {
  const int do_search =
    (enc->config_->target_size > 0 || enc->config_->target_PSNR > 0);
  const int fast_probe = (enc->method_ < 2 && !do_search);
  float q = enc->config_->quality;
  const int max_passes = enc->config_->pass;
  const int task_percent = 20;
  const int percent_per_pass = (task_percent + max_passes / 2) / max_passes;
  const int final_percent = enc->percent_ + task_percent;
  int pass;
  int nb_mbs;

  // Fast mode: quick analysis pass over few mbs. Better than nothing.
  nb_mbs = enc->mb_w_ * enc->mb_h_;
  if (fast_probe && nb_mbs > 100) nb_mbs = 100;

  // No target size: just do several pass without changing 'q'
  if (!do_search) {
    for (pass = 0; pass < max_passes; ++pass) {
      const int rd_opt = (enc->method_ > 2);
      if (!OneStatPass(enc, q, rd_opt, nb_mbs, NULL, percent_per_pass)) {
        return 0;
      }
    }
  } else {
    // binary search for a size close to target
    for (pass = 0; pass < max_passes && (dqs[pass] > 0); ++pass) {
      const int rd_opt = 1;
      float PSNR;
      int criterion;
      const int size = OneStatPass(enc, q, rd_opt, nb_mbs, &PSNR,
                                   percent_per_pass);
#if DEBUG_SEARCH
      printf("#%d size=%d PSNR=%.2f q=%.2f\n", pass, size, PSNR, q);
#endif
      if (!size) return 0;
      if (enc->config_->target_PSNR > 0) {
        criterion = (PSNR < enc->config_->target_PSNR);
      } else {
        criterion = (size < enc->config_->target_size);
      }
      // dichotomize
      if (criterion) {
        q += dqs[pass];
      } else {
        q -= dqs[pass];
      }
    }
  }
  return WebPReportProgress(enc->pic_, final_percent, &enc->percent_);
}

//------------------------------------------------------------------------------

#if defined(__cplusplus) || defined(c_plusplus)
}    // extern "C"
#endif
