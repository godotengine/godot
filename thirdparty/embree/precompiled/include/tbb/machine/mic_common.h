/*
    Copyright (c) 2005-2020 Intel Corporation

    Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at

        http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.
*/

#ifndef __TBB_mic_common_H
#define __TBB_mic_common_H

#ifndef __TBB_machine_H
#error Do not #include this internal file directly; use public TBB headers instead.
#endif

#if ! __TBB_DEFINE_MIC
    #error mic_common.h should be included only when building for Intel(R) Many Integrated Core Architecture
#endif

#ifndef __TBB_PREFETCHING
#define __TBB_PREFETCHING 1
#endif
#if __TBB_PREFETCHING
#include <immintrin.h>
#define __TBB_cl_prefetch(p) _mm_prefetch((const char*)p, _MM_HINT_T1)
#define __TBB_cl_evict(p) _mm_clevict(p, _MM_HINT_T1)
#endif

/** Intel(R) Many Integrated Core Architecture does not support mfence and pause instructions **/
#define __TBB_full_memory_fence() __asm__ __volatile__("lock; addl $0,(%%rsp)":::"memory")
#define __TBB_Pause(x) _mm_delay_32(16*(x))
#define __TBB_STEALING_PAUSE 1500/16
#include <sched.h>
#define __TBB_Yield() sched_yield()

/** Specifics **/
#define __TBB_STEALING_ABORT_ON_CONTENTION 1
#define __TBB_YIELD2P 1
#define __TBB_HOARD_NONLOCAL_TASKS 1

#if ! ( __FreeBSD__ || __linux__ )
    #error Intel(R) Many Integrated Core Compiler does not define __FreeBSD__ or __linux__ anymore. Check for the __TBB_XXX_BROKEN defined under __FreeBSD__ or __linux__.
#endif /* ! ( __FreeBSD__ || __linux__ ) */

#endif /* __TBB_mic_common_H */
