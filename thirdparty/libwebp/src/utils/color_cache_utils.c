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
#include "src/utils/color_cache_utils.h"
#include "src/utils/utils.h"

//------------------------------------------------------------------------------
// VP8LColorCache.

int VP8LColorCacheInit(VP8LColorCache* const color_cache, int hash_bits) {
  const int hash_size = 1 << hash_bits;
  assert(color_cache != NULL);
  assert(hash_bits > 0);
  color_cache->colors_ = (uint32_t*)WebPSafeCalloc(
      (uint64_t)hash_size, sizeof(*color_cache->colors_));
  if (color_cache->colors_ == NULL) return 0;
  color_cache->hash_shift_ = 32 - hash_bits;
  color_cache->hash_bits_ = hash_bits;
  return 1;
}

void VP8LColorCacheClear(VP8LColorCache* const color_cache) {
  if (color_cache != NULL) {
    WebPSafeFree(color_cache->colors_);
    color_cache->colors_ = NULL;
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
