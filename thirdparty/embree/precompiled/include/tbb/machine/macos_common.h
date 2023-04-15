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

#if !defined(__TBB_machine_H) || defined(__TBB_machine_macos_common_H)
#error Do not #include this internal file directly; use public TBB headers instead.
#endif

#define __TBB_machine_macos_common_H

#include <sched.h>
#define __TBB_Yield()  sched_yield()

// __TBB_HardwareConcurrency

#include <sys/types.h>
#include <sys/sysctl.h>

static inline int __TBB_macos_available_cpu() {
    int name[2] = {CTL_HW, HW_AVAILCPU};
    int ncpu;
    size_t size = sizeof(ncpu);
    sysctl( name, 2, &ncpu, &size, NULL, 0 );
    return ncpu;
}

#define __TBB_HardwareConcurrency() __TBB_macos_available_cpu()

#ifndef __TBB_full_memory_fence
    // TBB has not recognized the architecture (none of the architecture abstraction
    // headers was included).
    #define __TBB_UnknownArchitecture 1
#endif

#if __TBB_UnknownArchitecture
// Implementation of atomic operations based on OS provided primitives
#include <libkern/OSAtomic.h>

static inline int64_t __TBB_machine_cmpswp8_OsX(volatile void *ptr, int64_t value, int64_t comparand)
{
    __TBB_ASSERT( tbb::internal::is_aligned(ptr,8), "address not properly aligned for macOS* atomics");
    int64_t* address = (int64_t*)ptr;
    while( !OSAtomicCompareAndSwap64Barrier(comparand, value, address) ){
#if __TBB_WORDSIZE==8
        int64_t snapshot = *address;
#else
        int64_t snapshot = OSAtomicAdd64( 0, address );
#endif
        if( snapshot!=comparand ) return snapshot;
    }
    return comparand;
}

#define __TBB_machine_cmpswp8 __TBB_machine_cmpswp8_OsX

#endif /* __TBB_UnknownArchitecture */

#if __TBB_UnknownArchitecture

#ifndef __TBB_WORDSIZE
#define __TBB_WORDSIZE __SIZEOF_POINTER__
#endif

#ifdef __TBB_ENDIANNESS
    // Already determined based on hardware architecture.
#elif __BIG_ENDIAN__
    #define __TBB_ENDIANNESS __TBB_ENDIAN_BIG
#elif __LITTLE_ENDIAN__
    #define __TBB_ENDIANNESS __TBB_ENDIAN_LITTLE
#else
    #define __TBB_ENDIANNESS __TBB_ENDIAN_UNSUPPORTED
#endif

/** As this generic implementation has absolutely no information about underlying
    hardware, its performance most likely will be sub-optimal because of full memory
    fence usages where a more lightweight synchronization means (or none at all)
    could suffice. Thus if you use this header to enable TBB on a new platform,
    consider forking it and relaxing below helpers as appropriate. **/
#define __TBB_control_consistency_helper() OSMemoryBarrier()
#define __TBB_acquire_consistency_helper() OSMemoryBarrier()
#define __TBB_release_consistency_helper() OSMemoryBarrier()
#define __TBB_full_memory_fence()          OSMemoryBarrier()

static inline int32_t __TBB_machine_cmpswp4(volatile void *ptr, int32_t value, int32_t comparand)
{
    __TBB_ASSERT( tbb::internal::is_aligned(ptr,4), "address not properly aligned for macOS atomics");
    int32_t* address = (int32_t*)ptr;
    while( !OSAtomicCompareAndSwap32Barrier(comparand, value, address) ){
        int32_t snapshot = *address;
        if( snapshot!=comparand ) return snapshot;
    }
    return comparand;
}

static inline int32_t __TBB_machine_fetchadd4(volatile void *ptr, int32_t addend)
{
    __TBB_ASSERT( tbb::internal::is_aligned(ptr,4), "address not properly aligned for macOS atomics");
    return OSAtomicAdd32Barrier(addend, (int32_t*)ptr) - addend;
}

static inline int64_t __TBB_machine_fetchadd8(volatile void *ptr, int64_t addend)
{
    __TBB_ASSERT( tbb::internal::is_aligned(ptr,8), "address not properly aligned for macOS atomics");
    return OSAtomicAdd64Barrier(addend, (int64_t*)ptr) - addend;
}

#define __TBB_USE_GENERIC_PART_WORD_CAS                     1
#define __TBB_USE_GENERIC_PART_WORD_FETCH_ADD               1
#define __TBB_USE_GENERIC_FETCH_STORE                       1
#define __TBB_USE_GENERIC_HALF_FENCED_LOAD_STORE            1
#define __TBB_USE_GENERIC_RELAXED_LOAD_STORE                1
#if __TBB_WORDSIZE == 4
    #define __TBB_USE_GENERIC_DWORD_LOAD_STORE              1
#endif
#define __TBB_USE_GENERIC_SEQUENTIAL_CONSISTENCY_LOAD_STORE 1

#endif /* __TBB_UnknownArchitecture */
