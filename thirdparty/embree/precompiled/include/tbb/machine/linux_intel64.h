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

#if !defined(__TBB_machine_H) || defined(__TBB_machine_linux_intel64_H)
#error Do not #include this internal file directly; use public TBB headers instead.
#endif

#define __TBB_machine_linux_intel64_H

#include <stdint.h>
#include "gcc_ia32_common.h"

#define __TBB_WORDSIZE 8
#define __TBB_ENDIANNESS __TBB_ENDIAN_LITTLE

#define __TBB_compiler_fence() __asm__ __volatile__("": : :"memory")
#define __TBB_control_consistency_helper() __TBB_compiler_fence()
#define __TBB_acquire_consistency_helper() __TBB_compiler_fence()
#define __TBB_release_consistency_helper() __TBB_compiler_fence()

#ifndef __TBB_full_memory_fence
#define __TBB_full_memory_fence() __asm__ __volatile__("mfence": : :"memory")
#endif

#define __TBB_MACHINE_DEFINE_ATOMICS(S,T,X)                                          \
static inline T __TBB_machine_cmpswp##S (volatile void *ptr, T value, T comparand )  \
{                                                                                    \
    T result;                                                                        \
                                                                                     \
    __asm__ __volatile__("lock\ncmpxchg" X " %2,%1"                                  \
                          : "=a"(result), "=m"(*(volatile T*)ptr)                    \
                          : "q"(value), "0"(comparand), "m"(*(volatile T*)ptr)       \
                          : "memory");                                               \
    return result;                                                                   \
}                                                                                    \
                                                                                     \
static inline T __TBB_machine_fetchadd##S(volatile void *ptr, T addend)              \
{                                                                                    \
    T result;                                                                        \
    __asm__ __volatile__("lock\nxadd" X " %0,%1"                                     \
                          : "=r"(result),"=m"(*(volatile T*)ptr)                     \
                          : "0"(addend), "m"(*(volatile T*)ptr)                      \
                          : "memory");                                               \
    return result;                                                                   \
}                                                                                    \
                                                                                     \
static inline  T __TBB_machine_fetchstore##S(volatile void *ptr, T value)            \
{                                                                                    \
    T result;                                                                        \
    __asm__ __volatile__("lock\nxchg" X " %0,%1"                                     \
                          : "=r"(result),"=m"(*(volatile T*)ptr)                     \
                          : "0"(value), "m"(*(volatile T*)ptr)                       \
                          : "memory");                                               \
    return result;                                                                   \
}                                                                                    \

__TBB_MACHINE_DEFINE_ATOMICS(1,int8_t,"")
__TBB_MACHINE_DEFINE_ATOMICS(2,int16_t,"")
__TBB_MACHINE_DEFINE_ATOMICS(4,int32_t,"")
__TBB_MACHINE_DEFINE_ATOMICS(8,int64_t,"q")

#undef __TBB_MACHINE_DEFINE_ATOMICS

static inline void __TBB_machine_or( volatile void *ptr, uint64_t value ) {
    __asm__ __volatile__("lock\norq %1,%0" : "=m"(*(volatile uint64_t*)ptr) : "r"(value), "m"(*(volatile uint64_t*)ptr) : "memory");
}

static inline void __TBB_machine_and( volatile void *ptr, uint64_t value ) {
    __asm__ __volatile__("lock\nandq %1,%0" : "=m"(*(volatile uint64_t*)ptr) : "r"(value), "m"(*(volatile uint64_t*)ptr) : "memory");
}

#define __TBB_AtomicOR(P,V) __TBB_machine_or(P,V)
#define __TBB_AtomicAND(P,V) __TBB_machine_and(P,V)

#define __TBB_USE_FETCHSTORE_AS_FULL_FENCED_STORE           1
#define __TBB_USE_GENERIC_HALF_FENCED_LOAD_STORE            1
#define __TBB_USE_GENERIC_RELAXED_LOAD_STORE                1
#define __TBB_USE_GENERIC_SEQUENTIAL_CONSISTENCY_LOAD_STORE 1

