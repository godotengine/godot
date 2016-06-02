// Copyright 2012 Google Inc. All Rights Reserved.
//
// Use of this source code is governed by a BSD-style license
// that can be found in the COPYING file in the root of the source
// tree. An additional intellectual property rights grant can be found
// in the file PATENTS. All contributing project authors may
// be found in the AUTHORS file in the root of the source tree.
// -----------------------------------------------------------------------------
//
// Color Cache for WebP Lossless
//
// Author: Jyrki Alakuijala (jyrki@google.com)

#include <assert.h>
#include <stdlib.h>
#include <string.h>
#include "./color_cache.h"
#include "../utils/utils.h"

//------------------------------------------------------------------------------
// VP8LColorCache.

int VP8LColorCacheInit(VP8LColorCache* const cc, int hash_bits) {
  const int hash_size = 1 << hash_bits;
  assert(cc != NULL);
  assert(hash_bits > 0);
  cc->colors_ = (uint32_t*)WebPSafeCalloc((uint64_t)hash_size,
                                          sizeof(*cc->colors_));
  if (cc->colors_ == NULL) return 0;
  cc->hash_shift_ = 32 - hash_bits;
  cc->hash_bits_ = hash_bits;
  return 1;
}

void VP8LColorCacheClear(VP8LColorCache* const cc) {
  if (cc != NULL) {
    WebPSafeFree(cc->colors_);
    cc->colors_ = NULL;
  }
}

void VP8LColorCacheCopy(const VP8LColorCache* const src,
                        VP8LColorCache* const dst) {
  assert(src != NULL);
  assert(dst != NULL);
  assert(src->hash_bits_ == dst->hash_bits_);
  memcpy(dst->colors_, src->colors_,
         ((size_t)1u << dst->hash_bits_) * sizeof(*dst->colors_));
}
