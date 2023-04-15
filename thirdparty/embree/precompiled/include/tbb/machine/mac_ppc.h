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

#if !defined(__TBB_machine_H) || defined(__TBB_machine_gcc_power_H)
#error Do not #include this internal file directly; use public TBB headers instead.
#endif

#define __TBB_machine_gcc_power_H

#include <stdint.h>
#include <unistd.h>

// TODO: rename to gcc_power.h?
// This file is for Power Architecture with compilers supporting GNU inline-assembler syntax (currently GNU g++ and IBM XL).
// Note that XL V9.0 (sometimes?) has trouble dealing with empty input and/or clobber lists, so they should be avoided.

#if __powerpc64__ || __ppc64__
    // IBM XL documents __powerpc64__ (and __PPC64__).
    // Apple documents __ppc64__ (with __ppc__ only on 32-bit).
    #define __TBB_WORDSIZE 8
#else
    #define __TBB_WORDSIZE 4
#endif

// Traditionally Power Architecture is big-endian.
// Little-endian could be just an address manipulation (compatibility with TBB not verified),
// or normal little-endian (on more recent systems). Embedded PowerPC systems may support
// page-specific endianness, but then one endianness must be hidden from TBB so that it still sees only one.
#if __BIG_ENDIAN__ || (defined(__BYTE_ORDER__) && __BYTE_ORDER__==__ORDER_BIG_ENDIAN__)
    #define __TBB_ENDIANNESS __TBB_ENDIAN_BIG
#elif __LITTLE_ENDIAN__ || (defined(__BYTE_ORDER__) && __BYTE_ORDER__==__ORDER_LITTLE_ENDIAN__)
    #define __TBB_ENDIANNESS __TBB_ENDIAN_LITTLE
#elif defined(__BYTE_ORDER__)
    #define __TBB_ENDIANNESS __TBB_ENDIAN_UNSUPPORTED
#else
    #define __TBB_ENDIANNESS __TBB_ENDIAN_DETECT
#endif

// On Power Architecture, (lock-free) 64-bit atomics require 64-bit hardware:
#if __TBB_WORDSIZE==8
    // Do not change the following definition, because TBB itself will use 64-bit atomics in 64-bit builds.
    #define __TBB_64BIT_ATOMICS 1
#elif __bgp__
    // Do not change the following definition, because this is known 32-bit hardware.
    #define __TBB_64BIT_ATOMICS 0
#else
    // To enable 64-bit atomics in 32-bit builds, set the value below to 1 instead of 0.
    // You must make certain that the program will only use them on actual 64-bit hardware
    // (which typically means that the entire program is only executed on such hardware),
    // because their implementation involves machine instructions that are illegal elsewhere.
    // The setting can be chosen independently per compilation unit,
    // which also means that TBB itself does not need to be rebuilt.
    // Alternatively (but only for the current architecture and TBB version),
    // override the default as a predefined macro when invoking the compiler.
    #ifndef __TBB_64BIT_ATOMICS
    #define __TBB_64BIT_ATOMICS 0
    #endif
#endif

inline int32_t __TBB_machine_cmpswp4 (volatile void *ptr, int32_t value, int32_t comparand )
{
    int32_t result;

    __asm__ __volatile__("sync\n"
                         "0:\n\t"
                         "lwarx %[res],0,%[ptr]\n\t"     /* load w/ reservation */
                         "cmpw %[res],%[cmp]\n\t"        /* compare against comparand */
                         "bne- 1f\n\t"                   /* exit if not same */
                         "stwcx. %[val],0,%[ptr]\n\t"    /* store new value */
                         "bne- 0b\n"                     /* retry if reservation lost */
                         "1:\n\t"                        /* the exit */
                         "isync"
                         : [res]"=&r"(result)
                         , "+m"(* (int32_t*) ptr)        /* redundant with "memory" */
                         : [ptr]"r"(ptr)
                         , [val]"r"(value)
                         , [cmp]"r"(comparand)
                         : "memory"                      /* compiler full fence */
                         , "cr0"                         /* clobbered by cmp and/or stwcx. */
                         );
    return result;
}

#if __TBB_WORDSIZE==8

inline int64_t __TBB_machine_cmpswp8 (volatile void *ptr, int64_t value, int64_t comparand )
{
    int64_t result;
    __asm__ __volatile__("sync\n"
                         "0:\n\t"
                         "ldarx %[res],0,%[ptr]\n\t"     /* load w/ reservation */
                         "cmpd %[res],%[cmp]\n\t"        /* compare against comparand */
                         "bne- 1f\n\t"                   /* exit if not same */
                         "stdcx. %[val],0,%[ptr]\n\t"    /* store new value */
                         "bne- 0b\n"                     /* retry if reservation lost */
                         "1:\n\t"                        /* the exit */
                         "isync"
                         : [res]"=&r"(result)
                         , "+m"(* (int64_t*) ptr)        /* redundant with "memory" */
                         : [ptr]"r"(ptr)
                         , [val]"r"(value)
                         , [cmp]"r"(comparand)
                         : "memory"                      /* compiler full fence */
                         , "cr0"                         /* clobbered by cmp and/or stdcx. */
                         );
    return result;
}

#elif __TBB_64BIT_ATOMICS /* && __TBB_WORDSIZE==4 */

inline int64_t __TBB_machine_cmpswp8 (volatile void *ptr, int64_t value, int64_t comparand )
{
    int64_t result;
    int64_t value_register, comparand_register, result_register; // dummy variables to allocate registers
    __asm__ __volatile__("sync\n\t"
                         "ld %[val],%[valm]\n\t"
                         "ld %[cmp],%[cmpm]\n"
                         "0:\n\t"
                         "ldarx %[res],0,%[ptr]\n\t"     /* load w/ reservation */
                         "cmpd %[res],%[cmp]\n\t"        /* compare against comparand */
                         "bne- 1f\n\t"                   /* exit if not same */
                         "stdcx. %[val],0,%[ptr]\n\t"    /* store new value */
                         "bne- 0b\n"                     /* retry if reservation lost */
                         "1:\n\t"                        /* the exit */
                         "std %[res],%[resm]\n\t"
                         "isync"
                         : [resm]"=m"(result)
                         , [res] "=&r"(   result_register)
                         , [val] "=&r"(    value_register)
                         , [cmp] "=&r"(comparand_register)
                         , "+m"(* (int64_t*) ptr)        /* redundant with "memory" */
                         : [ptr] "r"(ptr)
                         , [valm]"m"(value)
                         , [cmpm]"m"(comparand)
                         : "memory"                      /* compiler full fence */
                         , "cr0"                         /* clobbered by cmpd and/or stdcx. */
                         );
    return result;
}

#endif /* __TBB_WORDSIZE==4 && __TBB_64BIT_ATOMICS */

#define __TBB_MACHINE_DEFINE_LOAD_STORE(S,ldx,stx,cmpx)                                                       \
    template <typename T>                                                                                     \
    struct machine_load_store<T,S> {                                                                          \
        static inline T load_with_acquire(const volatile T& location) {                                       \
            T result;                                                                                         \
            __asm__ __volatile__(ldx " %[res],0(%[ptr])\n"                                                    \
                                 "0:\n\t"                                                                     \
                                 cmpx " %[res],%[res]\n\t"                                                    \
                                 "bne- 0b\n\t"                                                                \
                                 "isync"                                                                      \
                                 : [res]"=r"(result)                                                          \
                                 : [ptr]"b"(&location) /* cannot use register 0 here */                       \
                                 , "m"(location)       /* redundant with "memory" */                          \
                                 : "memory"            /* compiler acquire fence */                           \
                                 , "cr0"               /* clobbered by cmpw/cmpd */);                         \
            return result;                                                                                    \
        }                                                                                                     \
        static inline void store_with_release(volatile T &location, T value) {                                \
            __asm__ __volatile__("lwsync\n\t"                                                                 \
                                 stx " %[val],0(%[ptr])"                                                      \
                                 : "=m"(location)      /* redundant with "memory" */                          \
                                 : [ptr]"b"(&location) /* cannot use register 0 here */                       \
                                 , [val]"r"(value)                                                            \
                                 : "memory"/*compiler release fence*/ /*(cr0 not affected)*/);                \
        }                                                                                                     \
    };                                                                                                        \
                                                                                                              \
    template <typename T>                                                                                     \
    struct machine_load_store_relaxed<T,S> {                                                                  \
        static inline T load (const __TBB_atomic T& location) {                                               \
            T result;                                                                                         \
            __asm__ __volatile__(ldx " %[res],0(%[ptr])"                                                      \
                                 : [res]"=r"(result)                                                          \
                                 : [ptr]"b"(&location) /* cannot use register 0 here */                       \
                                 , "m"(location)                                                              \
                                 ); /*(no compiler fence)*/ /*(cr0 not affected)*/                            \
            return result;                                                                                    \
        }                                                                                                     \
        static inline void store (__TBB_atomic T &location, T value) {                                        \
            __asm__ __volatile__(stx " %[val],0(%[ptr])"                                                      \
                                 : "=m"(location)                                                             \
                                 : [ptr]"b"(&location) /* cannot use register 0 here */                       \
                                 , [val]"r"(value)                                                            \
                                 ); /*(no compiler fence)*/ /*(cr0 not affected)*/                            \
        }                                                                                                     \
    };

namespace tbb {
namespace internal {
    __TBB_MACHINE_DEFINE_LOAD_STORE(1,"lbz","stb","cmpw")
    __TBB_MACHINE_DEFINE_LOAD_STORE(2,"lhz","sth","cmpw")
    __TBB_MACHINE_DEFINE_LOAD_STORE(4,"lwz","stw","cmpw")

#if __TBB_WORDSIZE==8

    __TBB_MACHINE_DEFINE_LOAD_STORE(8,"ld" ,"std","cmpd")

#elif __TBB_64BIT_ATOMICS /* && __TBB_WORDSIZE==4 */

    template <typename T>
    struct machine_load_store<T,8> {
        static inline T load_with_acquire(const volatile T& location) {
            T result;
            T result_register; // dummy variable to allocate a register
            __asm__ __volatile__("ld %[res],0(%[ptr])\n\t"
                                 "std %[res],%[resm]\n"
                                 "0:\n\t"
                                 "cmpd %[res],%[res]\n\t"
                                 "bne- 0b\n\t"
                                 "isync"
                                 : [resm]"=m"(result)
                                 , [res]"=&r"(result_register)
                                 : [ptr]"b"(&location) /* cannot use register 0 here */
                                 , "m"(location)       /* redundant with "memory" */
                                 : "memory"            /* compiler acquire fence */
                                 , "cr0"               /* clobbered by cmpd */);
            return result;
        }

        static inline void store_with_release(volatile T &location, T value) {
            T value_register; // dummy variable to allocate a register
            __asm__ __volatile__("lwsync\n\t"
                                 "ld %[val],%[valm]\n\t"
                                 "std %[val],0(%[ptr])"
                                 : "=m"(location)      /* redundant with "memory" */
                                 , [val]"=&r"(value_register)
                                 : [ptr]"b"(&location) /* cannot use register 0 here */
                                 , [valm]"m"(value)
                                 : "memory"/*compiler release fence*/ /*(cr0 not affected)*/);
        }
    };

    struct machine_load_store_relaxed<T,8> {
        static inline T load (const volatile T& location) {
            T result;
            T result_register; // dummy variable to allocate a register
            __asm__ __volatile__("ld %[res],0(%[ptr])\n\t"
                                 "std %[res],%[resm]"
                                 : [resm]"=m"(result)
                                 , [res]"=&r"(result_register)
                                 : [ptr]"b"(&location) /* cannot use register 0 here */
                                 , "m"(location)
                                 ); /*(no compiler fence)*/ /*(cr0 not affected)*/
            return result;
        }

        static inline void store (volatile T &location, T value) {
            T value_register; // dummy variable to allocate a register
            __asm__ __volatile__("ld %[val],%[valm]\n\t"
                                 "std %[val],0(%[ptr])"
                                 : "=m"(location)
                                 , [val]"=&r"(value_register)
                                 : [ptr]"b"(&location) /* cannot use register 0 here */
                                 , [valm]"m"(value)
                                 ); /*(no compiler fence)*/ /*(cr0 not affected)*/
        }
    };
    #define __TBB_machine_load_store_relaxed_8

#endif /* __TBB_WORDSIZE==4 && __TBB_64BIT_ATOMICS */

}} // namespaces internal, tbb

#undef __TBB_MACHINE_DEFINE_LOAD_STORE

#define __TBB_USE_GENERIC_PART_WORD_CAS                     1
#define __TBB_USE_GENERIC_FETCH_ADD                         1
#define __TBB_USE_GENERIC_FETCH_STORE                       1
#define __TBB_USE_GENERIC_SEQUENTIAL_CONSISTENCY_LOAD_STORE 1

#define __TBB_control_consistency_helper() __asm__ __volatile__("isync": : :"memory")
#define __TBB_full_memory_fence()          __asm__ __volatile__( "sync": : :"memory")

static inline intptr_t __TBB_machine_lg( uintptr_t x ) {
    __TBB_ASSERT(x, "__TBB_Log2(0) undefined");
    // cntlzd/cntlzw starts counting at 2^63/2^31 (ignoring any higher-order bits), and does not affect cr0
#if __TBB_WORDSIZE==8
    __asm__ __volatile__ ("cntlzd %0,%0" : "+r"(x));
    return 63-static_cast<intptr_t>(x);
#else
    __asm__ __volatile__ ("cntlzw %0,%0" : "+r"(x));
    return 31-static_cast<intptr_t>(x);
#endif
}
#define __TBB_Log2(V) __TBB_machine_lg(V)

// Assumes implicit alignment for any 32-bit value
typedef uint32_t __TBB_Flag;
#define __TBB_Flag __TBB_Flag

inline bool __TBB_machine_trylockbyte( __TBB_atomic __TBB_Flag &flag ) {
    return __TBB_machine_cmpswp4(&flag,1,0)==0;
}
#define __TBB_TryLockByte(P) __TBB_machine_trylockbyte(P)
