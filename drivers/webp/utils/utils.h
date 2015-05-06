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

#ifndef WEBP_UTILS_UTILS_H_
#define WEBP_UTILS_UTILS_H_

#include "../webp/types.h"

#if defined(__cplusplus) || defined(c_plusplus)
extern "C" {
#endif

//------------------------------------------------------------------------------
// Memory allocation

// This is the maximum memory amount that libwebp will ever try to allocate.
#define WEBP_MAX_ALLOCABLE_MEMORY (1ULL << 40)

// size-checking safe malloc/calloc: verify that the requested size is not too
// large, or return NULL. You don't need to call these for constructs like
// malloc(sizeof(foo)), but only if there's picture-dependent size involved
// somewhere (like: malloc(num_pixels * sizeof(*something))). That's why this
// safe malloc() borrows the signature from calloc(), pointing at the dangerous
// underlying multiply involved.
void* WebPSafeMalloc(uint64_t nmemb, size_t size);
// Note that WebPSafeCalloc() expects the second argument type to be 'size_t'
// in order to favor the "calloc(num_foo, sizeof(foo))" pattern.
void* WebPSafeCalloc(uint64_t nmemb, size_t size);

//------------------------------------------------------------------------------

#if defined(__cplusplus) || defined(c_plusplus)
}    // extern "C"
#endif

#endif  /* WEBP_UTILS_UTILS_H_ */
