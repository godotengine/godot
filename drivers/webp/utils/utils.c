// Copyright 2012 Google Inc. All Rights Reserved.
//
// Use of this source code is governed by a BSD-style license
// that can be found in the COPYING file in the root of the source
// tree. An additional intellectual property rights grant can be found
// in the file PATENTS. All contributing project authors may
// be found in the AUTHORS file in the root of the source tree.
// -----------------------------------------------------------------------------
//
// Misc. common utility functions
//
// Author: Skal (pascal.massimino@gmail.com)

#include <stdlib.h>
#include "./utils.h"

//------------------------------------------------------------------------------
// Checked memory allocation

// Returns 0 in case of overflow of nmemb * size.
static int CheckSizeArgumentsOverflow(uint64_t nmemb, size_t size) {
  const uint64_t total_size = nmemb * size;
  if (nmemb == 0) return 1;
  if ((uint64_t)size > WEBP_MAX_ALLOCABLE_MEMORY / nmemb) return 0;
  if (total_size != (size_t)total_size) return 0;
  return 1;
}

void* WebPSafeMalloc(uint64_t nmemb, size_t size) {
  if (!CheckSizeArgumentsOverflow(nmemb, size)) return NULL;
  assert(nmemb * size > 0);
  return malloc((size_t)(nmemb * size));
}

void* WebPSafeCalloc(uint64_t nmemb, size_t size) {
  if (!CheckSizeArgumentsOverflow(nmemb, size)) return NULL;
  assert(nmemb * size > 0);
  return calloc((size_t)nmemb, size);
}

//------------------------------------------------------------------------------

