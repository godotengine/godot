// Copyright 2012 Google Inc. All Rights Reserved.
//
// This code is licensed under the same terms as WebM:
//  Software License Agreement:  http://www.webmproject.org/license/software/
//  Additional IP Rights Grant:  http://www.webmproject.org/license/additional/
// -----------------------------------------------------------------------------
//
// Misc. common utility functions
//
// Author: Skal (pascal.massimino@gmail.com)

#include <stdlib.h>
#include "./utils.h"

#if defined(__cplusplus) || defined(c_plusplus)
extern "C" {
#endif

//------------------------------------------------------------------------------
// Checked memory allocation

static int CheckSizeArguments(uint64_t nmemb, size_t size) {
  const uint64_t total_size = nmemb * size;
  if (nmemb == 0) return 1;
  if ((uint64_t)size > WEBP_MAX_ALLOCABLE_MEMORY / nmemb) return 0;
  if (total_size != (size_t)total_size) return 0;
  return 1;
}

void* WebPSafeMalloc(uint64_t nmemb, size_t size) {
  if (!CheckSizeArguments(nmemb, size)) return NULL;
  return malloc((size_t)(nmemb * size));
}

void* WebPSafeCalloc(uint64_t nmemb, size_t size) {
  if (!CheckSizeArguments(nmemb, size)) return NULL;
  return calloc((size_t)nmemb, size);
}

//------------------------------------------------------------------------------

#if defined(__cplusplus) || defined(c_plusplus)
}    // extern "C"
#endif
