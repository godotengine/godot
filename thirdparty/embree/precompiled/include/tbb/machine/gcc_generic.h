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

#if !defined(__TBB_machine_H) || defined(__TBB_machine_gcc_generic_H)
#error Do not #include this internal file directly; use public TBB headers instead.
#endif

#define __TBB_machine_gcc_generic_H

#include <stdint.h>
#include <unistd.h>

#define __TBB_WORDSIZE __SIZEOF_POINTER__

#if __TBB_GCC_64BIT_ATOMIC_BUILTINS_BROKEN
    #define __TBB_64BIT_ATOMICS 0
#endif

/** FPU control setting not available for non-Intel architectures on Android **/
#if __ANDROID__ && __TBB_generic_arch
    #define __TBB_CPU_CTL_ENV_PRESENT 0
#endif

// __BYTE_ORDER__ is used in accordance with http://gcc.gnu.org/onlinedocs/cpp/Common-Predefined-Macros.html,
// but __BIG_ENDIAN__ or __LITTLE_ENDIAN__ may be more commonly found instead.
#if __BIG_ENDIAN__ || (defined(__BYTE_ORDER__) && __BYTE_ORDER__==__ORDER_BIG_ENDIAN__)
    #define __TBB_ENDIANNESS __TBB_ENDIAN_BIG
#elif __LITTLE_ENDIAN__ || (defined(__BYTE_ORDER__) && __BYTE_ORDER__==__ORDER_LITTLE_ENDIAN__)
    #define __TBB_ENDIANNESS __TBB_ENDIAN_LITTLE
#elif defined(__BYTE_ORDER__)
    #define __TBB_ENDIANNESS __TBB_ENDIAN_UNSUPPORTED
#else
    #define __TBB_ENDIANNESS __TBB_ENDIAN_DETECT
#endif

#if __TBB_GCC_VERSION < 40700
// Use __sync_* builtins

/** As this generic implementation has absolutely no information about underlying
    hardware, its performance most likely will be sub-optimal because of full memory
    fence usages where a more lightweight synchronization means (or none at all)
    could suffice. Thus if you use this header to enable TBB on a new platform,
    consider forking it and relaxing below helpers as appropriate. **/
#define __TBB_acquire_consistency_helper()  __sync_synchronize()
#define __TBB_release_consistency_helper()  __sync_synchronize()
#define __TBB_full_memory_fence()           __sync_synchronize()
#define __TBB_control_consistency_helper()  __sync_synchronize()

#define __TBB_MACHINE_DEFINE_ATOMICS(S,T)                                                         \
inline T __TBB_machine_cmpswp##S( volatile void *ptr, T value, T comparand ) {                    \
    return __sync_val_compare_and_swap(reinterpret_cast<volatile T *>(ptr),comparand,value);      \
}                                                                                                 \
inline T __TBB_machine_fetchadd##S( volatile void *ptr, T value ) {                               \
    return __sync_fetch_and_add(reinterpret_cast<volatile T *>(ptr),value);                       \
}

#define __TBB_USE_GENERIC_FETCH_STORE 1

#else
// __TBB_GCC_VERSION >= 40700; use __atomic_* builtins available since gcc 4.7

#define __TBB_compiler_fence()              __asm__ __volatile__("": : :"memory")
// Acquire and release fence intrinsics in GCC might miss compiler fence.
// Adding it at both sides of an intrinsic, as we do not know what reordering can be made.
#define __TBB_acquire_consistency_helper()  __TBB_compiler_fence(); __atomic_thread_fence(__ATOMIC_ACQUIRE); __TBB_compiler_fence()
#define __TBB_release_consistency_helper()  __TBB_compiler_fence(); __atomic_thread_fence(__ATOMIC_RELEASE); __TBB_compiler_fence()
#define __TBB_full_memory_fence()           __atomic_thread_fence(__ATOMIC_SEQ_CST)
#define __TBB_control_consistency_helper()  __TBB_acquire_consistency_helper()

#define __TBB_MACHINE_DEFINE_ATOMICS(S,T)                                                         \
inline T __TBB_machine_cmpswp##S( volatile void *ptr, T value, T comparand ) {                    \
    (void)__atomic_compare_exchange_n(reinterpret_cast<volatile T *>(ptr), &comparand, value,     \
                                      false, __ATOMIC_SEQ_CST, __ATOMIC_SEQ_CST);                 \
    return comparand;                                                                             \
}                                                                                                 \
inline T __TBB_machine_fetchadd##S( volatile void *ptr, T value ) {                               \
    return __atomic_fetch_add(reinterpret_cast<volatile T *>(ptr), value, __ATOMIC_SEQ_CST);      \
}                                                                                                 \
inline T __TBB_machine_fetchstore##S( volatile void *ptr, T value ) {                             \
    return __atomic_exchange_n(reinterpret_cast<volatile T *>(ptr), value, __ATOMIC_SEQ_CST);     \
}

#endif // __TBB_GCC_VERSION < 40700

__TBB_MACHINE_DEFINE_ATOMICS(1,int8_t)
__TBB_MACHINE_DEFINE_ATOMICS(2,int16_t)
__TBB_MACHINE_DEFINE_ATOMICS(4,int32_t)
__TBB_MACHINE_DEFINE_ATOMICS(8,int64_t)

#undef __TBB_MACHINE_DEFINE_ATOMICS

typedef unsigned char __TBB_Flag;
typedef __TBB_atomic __TBB_Flag __TBB_atomic_flag;

#if __TBB_GCC_VERSION < 40700
// Use __sync_* builtins

// Use generic machine_load_store functions if there are no builtin atomics
#define __TBB_USE_GENERIC_HALF_FENCED_LOAD_STORE            1
#define __TBB_USE_GENERIC_RELAXED_LOAD_STORE                1
#define __TBB_USE_GENERIC_SEQUENTIAL_CONSISTENCY_LOAD_STORE 1

static inline void __TBB_machine_or( volatile void *ptr, uintptr_t addend ) {
    __sync_fetch_and_or(reinterpret_cast<volatile uintptr_t *>(ptr),addend);
}

static inline void __TBB_machine_and( volatile void *ptr, uintptr_t addend ) {
    __sync_fetch_and_and(reinterpret_cast<volatile uintptr_t *>(ptr),addend);
}

inline bool __TBB_machine_try_lock_byte( __TBB_atomic_flag &flag ) {
    return __sync_lock_test_and_set(&flag,1)==0;
}

inline void __TBB_machine_unlock_byte( __TBB_atomic_flag &flag ) {
    __sync_lock_release(&flag);
}

#else
// __TBB_GCC_VERSION >= 40700; use __atomic_* builtins available since gcc 4.7

static inline void __TBB_machine_or( volatile void *ptr, uintptr_t addend ) {
    __atomic_fetch_or(reinterpret_cast<volatile uintptr_t *>(ptr),addend,__ATOMIC_SEQ_CST);
}

static inline void __TBB_machine_and( volatile void *ptr, uintptr_t addend ) {
    __atomic_fetch_and(reinterpret_cast<volatile uintptr_t *>(ptr),addend,__ATOMIC_SEQ_CST);
}

inline bool __TBB_machine_try_lock_byte( __TBB_atomic_flag &flag ) {
    return !__atomic_test_and_set(&flag,__ATOMIC_ACQUIRE);
}

inline void __TBB_machine_unlock_byte( __TBB_atomic_flag &flag ) {
    __atomic_clear(&flag,__ATOMIC_RELEASE);
}

namespace tbb { namespace internal {

/** GCC atomic operation intrinsics might miss compiler fence.
    Adding it after load-with-acquire, before store-with-release, and
    on both sides of sequentially consistent operations is sufficient for correctness. **/

template <typename T, int MemOrder>
inline T __TBB_machine_atomic_load( const volatile T& location) {
    if (MemOrder == __ATOMIC_SEQ_CST) __TBB_compiler_fence();
    T value = __atomic_load_n(&location, MemOrder);
    if (MemOrder != __ATOMIC_RELAXED) __TBB_compiler_fence();
    return value;
}

template <typename T, int MemOrder>
inline void __TBB_machine_atomic_store( volatile T& location, T value) {
    if (MemOrder != __ATOMIC_RELAXED) __TBB_compiler_fence();
    __atomic_store_n(&location, value, MemOrder);
    if (MemOrder == __ATOMIC_SEQ_CST) __TBB_compiler_fence();
}

template <typename T, size_t S>
struct machine_load_store {
    static T load_with_acquire ( const volatile T& location ) {
        return __TBB_machine_atomic_load<T, __ATOMIC_ACQUIRE>(location);
    }
    static void store_with_release ( volatile T &location, T value ) {
        __TBB_machine_atomic_store<T, __ATOMIC_RELEASE>(location, value);
    }
};

template <typename T, size_t S>
struct machine_load_store_relaxed {
    static inline T load ( const volatile T& location ) {
        return __TBB_machine_atomic_load<T, __ATOMIC_RELAXED>(location);
    }
    static inline void store ( volatile T& location, T value ) {
        __TBB_machine_atomic_store<T, __ATOMIC_RELAXED>(location, value);
    }
};

template <typename T, size_t S>
struct machine_load_store_seq_cst {
    static T load ( const volatile T& location ) {
        return __TBB_machine_atomic_load<T, __ATOMIC_SEQ_CST>(location);
    }
    static void store ( volatile T &location, T value ) {
        __TBB_machine_atomic_store<T, __ATOMIC_SEQ_CST>(location, value);
    }
};

}} // namespace tbb::internal

#endif // __TBB_GCC_VERSION < 40700

// Machine specific atomic operations
#define __TBB_AtomicOR(P,V)     __TBB_machine_or(P,V)
#define __TBB_AtomicAND(P,V)    __TBB_machine_and(P,V)

#define __TBB_TryLockByte   __TBB_machine_try_lock_byte
#define __TBB_UnlockByte    __TBB_machine_unlock_byte

// __builtin_clz counts the number of leading zeroes
namespace tbb{ namespace internal { namespace gcc_builtins {
    inline int clz(unsigned int x){ return __builtin_clz(x); }
    inline int clz(unsigned long int x){ return __builtin_clzl(x); }
    inline int clz(unsigned long long int x){ return __builtin_clzll(x); }
}}}
// logarithm is the index of the most significant non-zero bit
static inline intptr_t __TBB_machine_lg( uintptr_t x ) {
    // If P is a power of 2 and x<P, then (P-1)-x == (P-1) XOR x
    return (sizeof(x)*8 - 1) ^ tbb::internal::gcc_builtins::clz(x);
}

#define __TBB_Log2(V)  __TBB_machine_lg(V)

#if __TBB_WORDSIZE==4
    #define __TBB_USE_GENERIC_DWORD_LOAD_STORE              1
#endif

#if __TBB_x86_32 || __TBB_x86_64
#include "gcc_ia32_common.h"
#endif
