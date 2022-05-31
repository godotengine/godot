/*
 * Copyright 2019 Google LLC
 *
 * Use of this source code is governed by a BSD-style license that can be
 * found in the LICENSE file.
 */

#ifndef SkThreadAnnotations_DEFINED
#define SkThreadAnnotations_DEFINED

// The bulk of this code is cribbed from:
// http://clang.llvm.org/docs/ThreadSafetyAnalysis.html

#if defined(__clang__) && (!defined(SWIG))
#define SK_THREAD_ANNOTATION_ATTRIBUTE(x)   __attribute__((x))
#else
#define SK_THREAD_ANNOTATION_ATTRIBUTE(x)   // no-op
#endif

#define SK_CAPABILITY(x) \
  SK_THREAD_ANNOTATION_ATTRIBUTE(capability(x))

#define SK_SCOPED_CAPABILITY \
  SK_THREAD_ANNOTATION_ATTRIBUTE(scoped_lockable)

#define SK_GUARDED_BY(x) \
  SK_THREAD_ANNOTATION_ATTRIBUTE(guarded_by(x))

#define SK_PT_GUARDED_BY(x) \
  SK_THREAD_ANNOTATION_ATTRIBUTE(pt_guarded_by(x))

#define SK_ACQUIRED_BEFORE(...) \
  SK_THREAD_ANNOTATION_ATTRIBUTE(acquired_before(__VA_ARGS__))

#define SK_ACQUIRED_AFTER(...) \
  SK_THREAD_ANNOTATION_ATTRIBUTE(acquired_after(__VA_ARGS__))

#define SK_REQUIRES(...) \
  SK_THREAD_ANNOTATION_ATTRIBUTE(requires_capability(__VA_ARGS__))

#define SK_REQUIRES_SHARED(...) \
  SK_THREAD_ANNOTATION_ATTRIBUTE(requires_shared_capability(__VA_ARGS__))

#define SK_ACQUIRE(...) \
  SK_THREAD_ANNOTATION_ATTRIBUTE(acquire_capability(__VA_ARGS__))

#define SK_ACQUIRE_SHARED(...) \
  SK_THREAD_ANNOTATION_ATTRIBUTE(acquire_shared_capability(__VA_ARGS__))

// Would be SK_RELEASE, but that is already in use as SK_DEBUG vs. SK_RELEASE.
#define SK_RELEASE_CAPABILITY(...) \
  SK_THREAD_ANNOTATION_ATTRIBUTE(release_capability(__VA_ARGS__))

// For symmetry with SK_RELEASE_CAPABILITY.
#define SK_RELEASE_SHARED_CAPABILITY(...) \
  SK_THREAD_ANNOTATION_ATTRIBUTE(release_shared_capability(__VA_ARGS__))

#define SK_TRY_ACQUIRE(...) \
  SK_THREAD_ANNOTATION_ATTRIBUTE(try_acquire_capability(__VA_ARGS__))

#define SK_TRY_ACQUIRE_SHARED(...) \
  SK_THREAD_ANNOTATION_ATTRIBUTE(try_acquire_shared_capability(__VA_ARGS__))

#define SK_EXCLUDES(...) \
  SK_THREAD_ANNOTATION_ATTRIBUTE(locks_excluded(__VA_ARGS__))

#define SK_ASSERT_CAPABILITY(x) \
  SK_THREAD_ANNOTATION_ATTRIBUTE(assert_capability(x))

#define SK_ASSERT_SHARED_CAPABILITY(x) \
  SK_THREAD_ANNOTATION_ATTRIBUTE(assert_shared_capability(x))

#define SK_RETURN_CAPABILITY(x) \
  SK_THREAD_ANNOTATION_ATTRIBUTE(lock_returned(x))

#define SK_NO_THREAD_SAFETY_ANALYSIS \
  SK_THREAD_ANNOTATION_ATTRIBUTE(no_thread_safety_analysis)

#if defined(SK_BUILD_FOR_GOOGLE3) && !defined(SK_BUILD_FOR_WASM_IN_GOOGLE3)
    extern "C" {
        void __google_cxa_guard_acquire_begin(void);
        void __google_cxa_guard_acquire_end  (void);
    }
    #define SK_POTENTIALLY_BLOCKING_REGION_BEGIN __google_cxa_guard_acquire_begin()
    #define SK_POTENTIALLY_BLOCKING_REGION_END   __google_cxa_guard_acquire_end()
#else
    #define SK_POTENTIALLY_BLOCKING_REGION_BEGIN
    #define SK_POTENTIALLY_BLOCKING_REGION_END
#endif

#endif  // SkThreadAnnotations_DEFINED
