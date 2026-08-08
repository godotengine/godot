// Copyright 2011 Google Inc. All Rights Reserved.
//
// Use of this source code is governed by a BSD-style license
// that can be found in the COPYING file in the root of the source
// tree. An additional intellectual property rights grant can be found
// in the file PATENTS. All contributing project authors may
// be found in the AUTHORS file in the root of the source tree.
// -----------------------------------------------------------------------------
//
// Paginated token buffer
//
//  A 'token' is a bit value associated with a probability, either fixed
// or a later-to-be-determined after statistics have been collected.
// For dynamic probability, we just record the slot id (idx) for the probability
// value in the final probability array (uint8_t* probas in VP8EmitTokens).
//
// Author: Skal (pascal.massimino@gmail.com)

#include <assert.h>
#include <stdlib.h>
#include <string.h>

#include "src/dec/common_dec.h"
#include "src/dsp/dsp.h"
#include "src/enc/cost_enc.h"
#include "src/enc/vp8i_enc.h"
#include "src/utils/bit_writer_utils.h"
#include "src/utils/utils.h"
#include "src/webp/types.h"

#if !defined(DISABLE_TOKEN_BUFFER)

// we use pages to reduce the number of memcpy()
#define MIN_PAGE_SIZE 8192          // minimum number of token per page
#define FIXED_PROBA_BIT (1u << 14)

typedef uint16_t token_t;  // bit #15: bit value
                           // bit #14: flags for constant proba or idx
                           // bits #0..13: slot or constant proba
struct VP8Tokens {
  VP8Tokens* next;         // pointer to next page
};
// Token data is located in memory just after the 'next' field.
// This macro is used to return their address and hide the trick.
#define TOKEN_DATA(p) ((const token_t*)&(p)[1])

//------------------------------------------------------------------------------

void VP8TBufferInit(VP8TBuffer* const b, int page_size) {
  b->tokens = NULL;
  b->pages = NULL;
  b->last_page = &b->pages;
  b->left = 0;
  b->page_size = (page_size < MIN_PAGE_SIZE) ? MIN_PAGE_SIZE : page_size;
  b->error = 0;
}

void VP8TBufferClear(VP8TBuffer* const b) {
  if (b != NULL) {
    VP8Tokens* p = b->pages;
    while (p != NULL) {
      VP8Tokens* const next = p->next;
      WebPSafeFree(p);
      p = next;
    }
    VP8TBufferInit(b, b->page_size);
  }
}

static int TBufferNewPage(VP8TBuffer* const b) {
  VP8Tokens* page = NULL;
  if (!b->error) {
    const size_t size = sizeof(*page) + b->page_size * sizeof(token_t);
    page = (VP8Tokens*)WebPSafeMalloc(1ULL, size);
  }
  if (page == NULL) {
    b->error = 1;
    return 0;
  }
  page->next = NULL;

  *b->last_page = page;
  b->last_page = &page->next;
  b->left = b->page_size;
  b->tokens = (token_t*)TOKEN_DATA(page);
  return 1;
}

//------------------------------------------------------------------------------

#define TOKEN_ID(t, b, ctx) \
    (NUM_PROBAS * ((ctx) + NUM_CTX * ((b) + NUM_BANDS * (t))))

static WEBP_INLINE uint32_t AddToken(VP8TBuffer* const b, uint32_t bit,
                                     uint32_t proba_idx,
                                     proba_t* const stats) {
  assert(proba_idx < FIXED_PROBA_BIT);
  assert(bit <= 1);
  if (b->left > 0 || TBufferNewPage(b)) {
    const int slot = --b->left;
    b->tokens[slot] = (bit << 15) | proba_idx;
  }
  VP8RecordStats(bit, stats);
  return bit;
}

static WEBP_INLINE void AddConstantToken(VP8TBuffer* const b,
                                         uint32_t bit, uint32_t proba) {
  assert(proba < 256);
  assert(bit <= 1);
  if (b->left > 0 || TBufferNewPage(b)) {
    const int slot = --b->left;
    b->tokens[slot] = (bit << 15) | FIXED_PROBA_BIT | proba;
  }
}

int VP8RecordCoeffTokens(int ctx, const struct VP8Residual* const res,
                         VP8TBuffer* const tokens) {
  const int16_t* const coeffs = res->coeffs;
  const int coeff_type = res->coeff_type;
  const int last = res->last;
  int n = res->first;
  uint32_t base_id = TOKEN_ID(coeff_type, n, ctx);
  // should be stats[VP8EncBands[n]], but it's equivalent for n=0 or 1
  proba_t* s = res->stats[n][ctx];
  if (!AddToken(tokens, last >= 0, base_id + 0, s + 0)) {
    return 0;
  }

  while (n < 16) {
    const int c = coeffs[n++];
    const int sign = c < 0;
    const uint32_t v = sign ? -c : c;
    if (!AddToken(tokens, v != 0, base_id + 1, s + 1)) {
      base_id = TOKEN_ID(coeff_type, VP8EncBands[n], 0);  // ctx=0
      s = res->stats[VP8EncBands[n]][0];
      continue;
    }
    if (!AddToken(tokens, v > 1, base_id + 2, s + 2)) {
      base_id = TOKEN_ID(coeff_type, VP8EncBands[n], 1);  // ctx=1
      s = res->stats[VP8EncBands[n]][1];
    } else {
      if (!AddToken(tokens, v > 4, base_id + 3, s + 3)) {
        if (AddToken(tokens, v != 2, base_id + 4, s + 4)) {
          AddToken(tokens, v == 4, base_id + 5, s + 5);
        }
      } else if (!AddToken(tokens, v > 10, base_id + 6, s + 6)) {
        if (!AddToken(tokens, v > 6, base_id + 7, s + 7)) {
          AddConstantToken(tokens, v == 6, 159);
        } else {
          AddConstantToken(tokens, v >= 9, 165);
          AddConstantToken(tokens, !(v & 1), 145);
        }
      } else {
        int mask;
        const uint8_t* tab;
        uint32_t residue = v - 3;
        if (residue < (8 << 1)) {          // VP8Cat3  (3b)
          AddToken(tokens, 0, base_id + 8, s + 8);
          AddToken(tokens, 0, base_id + 9, s + 9);
          residue -= (8 << 0);
          mask = 1 << 2;
          tab = VP8Cat3;
        } else if (residue < (8 << 2)) {   // VP8Cat4  (4b)
          AddToken(tokens, 0, base_id + 8, s + 8);
          AddToken(tokens, 1, base_id + 9, s + 9);
          residue -= (8 << 1);
          mask = 1 << 3;
          tab = VP8Cat4;
        } else if (residue < (8 << 3)) {   // VP8Cat5  (5b)
          AddToken(tokens, 1, base_id + 8, s + 8);
          AddToken(tokens, 0, base_id + 10, s + 9);
          residue -= (8 << 2);
          mask = 1 << 4;
          tab = VP8Cat5;
        } else {                         // VP8Cat6 (11b)
          AddToken(tokens, 1, base_id + 8, s + 8);
          AddToken(tokens, 1, base_id + 10, s + 9);
          residue -= (8 << 3);
          mask = 1 << 10;
          tab = VP8Cat6;
        }
        while (mask) {
          AddConstantToken(tokens, !!(residue & mask), *tab++);
          mask >>= 1;
        }
      }
      base_id = TOKEN_ID(coeff_type, VP8EncBands[n], 2);  // ctx=2
      s = res->stats[VP8EncBands[n]][2];
    }
    AddConstantToken(tokens, sign, 128);
    if (n == 16 || !AddToken(tokens, n <= last, base_id + 0, s + 0)) {
      return 1;   // EOB
    }
  }
  return 1;
}

#undef TOKEN_ID

//------------------------------------------------------------------------------
// Final coding pass, with known probabilities

int VP8EmitTokens(VP8TBuffer* const b, VP8BitWriter* const bw,
                  const uint8_t* const probas, int final_pass) {
  const VP8Tokens* p = b->pages;
  assert(!b->error);
  while (p != NULL) {
    const VP8Tokens* const next = p->next;
    const int N = (next == NULL) ? b->left : 0;
    int n = b->page_size;
    const token_t* const tokens = TOKEN_DATA(p);
    while (n-- > N) {
      const token_t token = tokens[n];
      const int bit = (token >> 15) & 1;
      if (token & FIXED_PROBA_BIT) {
        VP8PutBit(bw, bit, token & 0xffu);  // constant proba
      } else {
        VP8PutBit(bw, bit, probas[token & 0x3fffu]);
      }
    }
    if (final_pass) WebPSafeFree((void*)p);
    p = next;
  }
  if (final_pass) b->pages = NULL;
  return 1;
}

// Size estimation
size_t VP8EstimateTokenSize(VP8TBuffer* const b, const uint8_t* const probas) {
  size_t size = 0;
  const VP8Tokens* p = b->pages;
  assert(!b->error);
  while (p != NULL) {
    const VP8Tokens* const next = p->next;
    const int N = (next == NULL) ? b->left : 0;
    int n = b->page_size;
    const token_t* const tokens = TOKEN_DATA(p);
    while (n-- > N) {
      const token_t token = tokens[n];
      const int bit = token & (1 << 15);
      if (token & FIXED_PROBA_BIT) {
        size += VP8BitCost(bit, token & 0xffu);
      } else {
        size += VP8BitCost(bit, probas[token & 0x3fffu]);
      }
    }
    p = next;
  }
  return size;
}

//------------------------------------------------------------------------------

#else     // DISABLE_TOKEN_BUFFER

void VP8TBufferInit(VP8TBuffer* const b, int page_size) {
  (void)b;
  (void)page_size;
}
void VP8TBufferClear(VP8TBuffer* const b) {
  (void)b;
}

#endif    // !DISABLE_TOKEN_BUFFER
