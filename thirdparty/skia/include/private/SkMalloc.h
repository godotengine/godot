/*
 * Copyright 2017 Google Inc.
 *
 * Use of this source code is governed by a BSD-style license that can be
 * found in the LICENSE file.
 */

#ifndef SkMalloc_DEFINED
#define SkMalloc_DEFINED

#include <cstddef>
#include <cstring>

#include "include/core/SkTypes.h"

/*
    memory wrappers to be implemented by the porting layer (platform)
*/


/** Free memory returned by sk_malloc(). It is safe to pass null. */
SK_API extern void sk_free(void*);

/**
 *  Called internally if we run out of memory. The platform implementation must
 *  not return, but should either throw an exception or otherwise exit.
 */
SK_API extern void sk_out_of_memory(void);

enum {
    /**
     *  If this bit is set, the returned buffer must be zero-initialized. If this bit is not set
     *  the buffer can be uninitialized.
     */
    SK_MALLOC_ZERO_INITIALIZE   = 1 << 0,

    /**
     *  If this bit is set, the implementation must throw/crash/quit if the request cannot
     *  be fulfilled. If this bit is not set, then it should return nullptr on failure.
     */
    SK_MALLOC_THROW             = 1 << 1,
};
/**
 *  Return a block of memory (at least 4-byte aligned) of at least the specified size.
 *  If the requested memory cannot be returned, either return nullptr or throw/exit, depending
 *  on the SK_MALLOC_THROW bit. If the allocation succeeds, the memory will be zero-initialized
 *  if the SK_MALLOC_ZERO_INITIALIZE bit was set.
 *
 *  To free the memory, call sk_free()
 */
SK_API extern void* sk_malloc_flags(size_t size, unsigned flags);

/** Same as standard realloc(), but this one never returns null on failure. It will throw
 *  an exception if it fails.
 */
SK_API extern void* sk_realloc_throw(void* buffer, size_t size);

static inline void* sk_malloc_throw(size_t size) {
    return sk_malloc_flags(size, SK_MALLOC_THROW);
}

static inline void* sk_calloc_throw(size_t size) {
    return sk_malloc_flags(size, SK_MALLOC_THROW | SK_MALLOC_ZERO_INITIALIZE);
}

static inline void* sk_calloc_canfail(size_t size) {
#if defined(SK_BUILD_FOR_FUZZER)
    // To reduce the chance of OOM, pretend we can't allocate more than 200kb.
    if (size > 200000) {
        return nullptr;
    }
#endif
    return sk_malloc_flags(size, SK_MALLOC_ZERO_INITIALIZE);
}

// Performs a safe multiply count * elemSize, checking for overflow
SK_API extern void* sk_calloc_throw(size_t count, size_t elemSize);
SK_API extern void* sk_malloc_throw(size_t count, size_t elemSize);
SK_API extern void* sk_realloc_throw(void* buffer, size_t count, size_t elemSize);

/**
 *  These variants return nullptr on failure
 */
static inline void* sk_malloc_canfail(size_t size) {
#if defined(SK_BUILD_FOR_FUZZER)
    // To reduce the chance of OOM, pretend we can't allocate more than 200kb.
    if (size > 200000) {
        return nullptr;
    }
#endif
    return sk_malloc_flags(size, 0);
}
SK_API extern void* sk_malloc_canfail(size_t count, size_t elemSize);

// bzero is safer than memset, but we can't rely on it, so... sk_bzero()
static inline void sk_bzero(void* buffer, size_t size) {
    // Please c.f. sk_careful_memcpy.  It's undefined behavior to call memset(null, 0, 0).
    if (size) {
        memset(buffer, 0, size);
    }
}

/**
 *  sk_careful_memcpy() is just like memcpy(), but guards against undefined behavior.
 *
 * It is undefined behavior to call memcpy() with null dst or src, even if len is 0.
 * If an optimizer is "smart" enough, it can exploit this to do unexpected things.
 *     memcpy(dst, src, 0);
 *     if (src) {
 *         printf("%x\n", *src);
 *     }
 * In this code the compiler can assume src is not null and omit the if (src) {...} check,
 * unconditionally running the printf, crashing the program if src really is null.
 * Of the compilers we pay attention to only GCC performs this optimization in practice.
 */
static inline void* sk_careful_memcpy(void* dst, const void* src, size_t len) {
    // When we pass >0 len we had better already be passing valid pointers.
    // So we just need to skip calling memcpy when len == 0.
    if (len) {
        memcpy(dst,src,len);
    }
    return dst;
}

static inline void* sk_careful_memmove(void* dst, const void* src, size_t len) {
    // When we pass >0 len we had better already be passing valid pointers.
    // So we just need to skip calling memcpy when len == 0.
    if (len) {
        memmove(dst,src,len);
    }
    return dst;
}

static inline int sk_careful_memcmp(const void* a, const void* b, size_t len) {
    // When we pass >0 len we had better already be passing valid pointers.
    // So we just need to skip calling memcmp when len == 0.
    if (len == 0) {
        return 0;   // we treat zero-length buffers as "equal"
    }
    return memcmp(a, b, len);
}

#endif  // SkMalloc_DEFINED
