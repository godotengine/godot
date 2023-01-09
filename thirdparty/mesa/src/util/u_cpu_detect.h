/**************************************************************************
 *
 * Copyright 2008 Dennis Smit
 * All Rights Reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * on the rights to use, copy, modify, merge, publish, distribute, sub
 * license, and/or sell copies of the Software, and to permit persons to whom
 * the Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice (including the next
 * paragraph) shall be included in all copies or substantial portions of the
 * Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NON-INFRINGEMENT.  IN NO EVENT SHALL
 * AUTHORS, COPYRIGHT HOLDERS, AND/OR THEIR SUPPLIERS BE LIABLE FOR ANY CLAIM,
 * DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
 * OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE
 * USE OR OTHER DEALINGS IN THE SOFTWARE.
 *
 ***************************************************************************/

/**
 * @file
 * CPU feature detection.
 *
 * @author Dennis Smit
 * @author Based on the work of Eric Anholt <anholt@FreeBSD.org>
 */

#ifndef _UTIL_CPU_DETECT_H
#define _UTIL_CPU_DETECT_H

#include <stdbool.h>

#include "util/macros.h"
#include "util/u_atomic.h"
#include "util/u_thread.h"


/* Maximal cpu count for update affinity */
#define UTIL_MAX_CPUS               1024  /* this should be enough */

#ifdef __cplusplus
extern "C" {
#endif

enum cpu_family {
   CPU_UNKNOWN,

   CPU_AMD_ZEN1_ZEN2,
   CPU_AMD_ZEN_HYGON,
   CPU_AMD_ZEN3,
   CPU_AMD_ZEN_NEXT,
   CPU_AMD_LAST,

   CPU_S390X,
};

typedef uint32_t util_affinity_mask[UTIL_MAX_CPUS / 32];

struct util_cpu_caps_t {
   /**
    * Number of CPUs available to the process.
    *
    * This will be less than or equal to \c max_cpus.  This is the number of
    * CPUs that are online and available to the process.
    */
   int16_t nr_cpus;

   /**
    * Maximum number of CPUs that can be online in the system.
    *
    * This will be greater than or equal to \c nr_cpus.  This is the number of
    * CPUs installed in the system.  \c nr_cpus will be less if some CPUs are
    * offline.
    */
   int16_t max_cpus;

   enum cpu_family family;

   /* Feature flags */
   int x86_cpu_type;
   unsigned cacheline;

   unsigned has_intel:1;
   unsigned has_tsc:1;
   unsigned has_mmx:1;
   unsigned has_mmx2:1;
   unsigned has_sse:1;
   unsigned has_sse2:1;
   unsigned has_sse3:1;
   unsigned has_ssse3:1;
   unsigned has_sse4_1:1;
   unsigned has_sse4_2:1;
   unsigned has_popcnt:1;
   unsigned has_avx:1;
   unsigned has_avx2:1;
   unsigned has_f16c:1;
   unsigned has_fma:1;
   unsigned has_3dnow:1;
   unsigned has_3dnow_ext:1;
   unsigned has_xop:1;
   unsigned has_altivec:1;
   unsigned has_vsx:1;
   unsigned has_daz:1;
   unsigned has_neon:1;
   unsigned has_msa:1;

   unsigned has_avx512f:1;
   unsigned has_avx512dq:1;
   unsigned has_avx512ifma:1;
   unsigned has_avx512pf:1;
   unsigned has_avx512er:1;
   unsigned has_avx512cd:1;
   unsigned has_avx512bw:1;
   unsigned has_avx512vl:1;
   unsigned has_avx512vbmi:1;

   unsigned num_L3_caches;
   unsigned num_cpu_mask_bits;
   unsigned max_vector_bits;

   uint16_t cpu_to_L3[UTIL_MAX_CPUS];
   /* Affinity masks for each L3 cache. */
   util_affinity_mask *L3_affinity_mask;
};

struct _util_cpu_caps_state_t {
   once_flag once_flag;
   /**
    * Initialized to 0 and set to non-zero with an atomic after the entire
    * struct has been initialized.
    */
   uint32_t detect_done;
   struct util_cpu_caps_t caps;
};

#define U_CPU_INVALID_L3 0xffff

static inline ATTRIBUTE_CONST const struct util_cpu_caps_t *
util_get_cpu_caps(void)
{
   extern void _util_cpu_detect_once(void);
   extern struct _util_cpu_caps_state_t _util_cpu_caps_state;

   /* On most CPU architectures, an atomic read is simply a regular memory
    * load instruction with some extra compiler magic to prevent code
    * re-ordering around it.  The perf impact of doing this check should be
    * negligible in most cases.
    *
    * Also, even though it looks like  a bit of a lie, we've declared this
    * function with ATTRIBUTE_CONST.  The GCC docs say:
    *
    *    "Calls to functions whose return value is not affected by changes to
    *    the observable state of the program and that have no observable
    *    effects on such state other than to return a value may lend
    *    themselves to optimizations such as common subexpression elimination.
    *    Declaring such functions with the const attribute allows GCC to avoid
    *    emitting some calls in repeated invocations of the function with the
    *    same argument values."
    *
    * The word "observable" is important here.  With the exception of a
    * llvmpipe debug flag behind an environment variable and a few unit tests,
    * all of which emulate worse CPUs, this function neither affects nor is
    * affected by any "observable" state.  It has its own internal state for
    * sure, but that state is such that it appears to return exactly the same
    * value with the same internal data every time.
    */
   if (unlikely(!p_atomic_read(&_util_cpu_caps_state.detect_done)))
      call_once(&_util_cpu_caps_state.once_flag, _util_cpu_detect_once);

   return &_util_cpu_caps_state.caps;
}

#ifdef __cplusplus
}
#endif


#endif /* _UTIL_CPU_DETECT_H */
