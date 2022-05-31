/*
 * Copyright 2011 Google Inc.
 *
 * Use of this source code is governed by a BSD-style license that can be
 * found in the LICENSE file.
 */

#include "include/private/SkMalloc.h"

#include <cstdlib>

#if defined(SK_DEBUG) && defined(SK_BUILD_FOR_WIN)
#include <intrin.h>
// This is a super stable value and setting it here avoids pulling in all of windows.h.
#ifndef FAST_FAIL_FATAL_APP_EXIT
#define FAST_FAIL_FATAL_APP_EXIT              7
#endif
#endif

#define SK_DEBUGFAILF(fmt, ...) \
    SkASSERT((SkDebugf(fmt"\n", __VA_ARGS__), false))

static inline void sk_out_of_memory(size_t size) {
    SK_DEBUGFAILF("sk_out_of_memory (asked for %zu bytes)",
                  size);
#if defined(SK_BUILD_FOR_AFL_FUZZ)
    exit(1);
#else
    abort();
#endif
}

static inline void* throw_on_failure(size_t size, void* p) {
    if (size > 0 && p == nullptr) {
        // If we've got a nullptr here, the only reason we should have failed is running out of RAM.
        sk_out_of_memory(size);
    }
    return p;
}

void sk_abort_no_print() {
#if defined(SK_BUILD_FOR_WIN) && defined(SK_IS_BOT)
    // do not display a system dialog before aborting the process
    _set_abort_behavior(0, _WRITE_ABORT_MSG);
#endif
#if defined(SK_DEBUG) && defined(SK_BUILD_FOR_WIN)
    __fastfail(FAST_FAIL_FATAL_APP_EXIT);
#elif defined(__clang__)
    __builtin_trap();
#else
    abort();
#endif
}

void sk_out_of_memory(void) {
    SkDEBUGFAIL("sk_out_of_memory");
#if defined(SK_BUILD_FOR_AFL_FUZZ)
    exit(1);
#else
    abort();
#endif
}

void* sk_realloc_throw(void* addr, size_t size) {
    return throw_on_failure(size, realloc(addr, size));
}

void sk_free(void* p) {
    if (p) {
        free(p);
    }
}

void* sk_malloc_flags(size_t size, unsigned flags) {
    void* p;
    if (flags & SK_MALLOC_ZERO_INITIALIZE) {
        p = calloc(size, 1);
    } else {
#if defined(SK_BUILD_FOR_ANDROID_FRAMEWORK) && defined(__BIONIC__)
        /* TODO: After b/169449588 is fixed, we will want to change this to restore
         *       original behavior instead of always disabling the flag.
         * TODO: After b/158870657 is fixed and scudo is used globally, we can assert when an
         *       an error is returned.
         */
        // malloc() generally doesn't initialize its memory and that's a huge security hole,
        // so Android has replaced its malloc() with one that zeros memory,
        // but that's a huge performance hit for HWUI, so turn it back off again.
        (void)mallopt(M_THREAD_DISABLE_MEM_INIT, 1);
#endif
        p = malloc(size);
#if defined(SK_BUILD_FOR_ANDROID_FRAMEWORK) && defined(__BIONIC__)
        (void)mallopt(M_THREAD_DISABLE_MEM_INIT, 0);
#endif
    }
    if (flags & SK_MALLOC_THROW) {
        return throw_on_failure(size, p);
    } else {
        return p;
    }
}
