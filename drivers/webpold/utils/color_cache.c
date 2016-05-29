// Copyright 2012 Google Inc. All Rights Reserved.
//
// This code is licensed under the same terms as WebM:
//  Software License Agreement:  http://www.webmproject.org/license/software/
//  Additional IP Rights Grant:  http://www.webmproject.org/license/additional/
// -----------------------------------------------------------------------------
//
// Color Cache for WebP Lossless
//
// Author: Jyrki Alakuijala (jyrki@google.com)

#include <assert.h>
#include <stdlib.h>
#include "./color_cache.h"
#include "../utils/utils.h"

#if defined(__cplusplus) || defined(c_plusplus)
extern "C" {
#endif

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
  return 1;
}

void VP8LColorCacheClear(VP8LColorCache* const cc) {
  if (cc != NULL) {
    free(cc->colors_);
    cc->colors_ = NULL;
  }
}

#if defined(__cplusplus) || defined(c_plusplus)
}
#endif
