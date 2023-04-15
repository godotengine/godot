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

#if !defined(__TBB_machine_H) || defined(__TBB_machine_windows_ia32_H)
#error Do not #include this internal file directly; use public TBB headers instead.
#endif

#define __TBB_machine_windows_ia32_H

#if defined(_MSC_VER) && !defined(__INTEL_COMPILER)
    // Workaround for overzealous compiler warnings in /Wp64 mode
    #pragma warning (push)
    #pragma warning (disable: 4244 4267)
#endif

#include "msvc_ia32_common.h"

#define __TBB_WORDSIZE 4
#define __TBB_ENDIANNESS __TBB_ENDIAN_LITTLE

extern "C" {
    __int64 __TBB_EXPORTED_FUNC __TBB_machine_cmpswp8 (volatile void *ptr, __int64 value, __int64 comparand );
    __int64 __TBB_EXPORTED_FUNC __TBB_machine_fetchadd8 (volatile void *ptr, __int64 addend );
    __int64 __TBB_EXPORTED_FUNC __TBB_machine_fetchstore8 (volatile void *ptr, __int64 value );
    void __TBB_EXPORTED_FUNC __TBB_machine_store8 (volatile void *ptr, __int64 value );
    __int64 __TBB_EXPORTED_FUNC __TBB_machine_load8 (const volatile void *ptr);
}

#if !__TBB_MSVC_PART_WORD_INTERLOCKED_INTRINSICS_PRESENT

#define __TBB_MACHINE_DEFINE_ATOMICS(S,T,U,A,C) \
static inline T __TBB_machine_cmpswp##S ( volatile void * ptr, U value, U comparand ) { \
    T result; \
    volatile T *p = (T *)ptr; \
    __asm \
    { \
       __asm mov edx, p \
       __asm mov C , value \
       __asm mov A , comparand \
       __asm lock cmpxchg [edx], C \
       __asm mov result, A \
    } \
    return result; \
} \
\
static inline T __TBB_machine_fetchadd##S ( volatile void * ptr, U addend ) { \
    T result; \
    volatile T *p = (T *)ptr; \
    __asm \
    { \
        __asm mov edx, p \
        __asm mov A, addend \
        __asm lock xadd [edx], A \
        __asm mov result, A \
    } \
    return result; \
}\
\
static inline T __TBB_machine_fetchstore##S ( volatile void * ptr, U value ) { \
    T result; \
    volatile T *p = (T *)ptr; \
    __asm \
    { \
        __asm mov edx, p \
        __asm mov A, value \
        __asm lock xchg [edx], A \
        __asm mov result, A \
    } \
    return result; \
}


__TBB_MACHINE_DEFINE_ATOMICS(1, __int8, __int8, al, cl)
__TBB_MACHINE_DEFINE_ATOMICS(2, __int16, __int16, ax, cx)
__TBB_MACHINE_DEFINE_ATOMICS(4, ptrdiff_t, ptrdiff_t, eax, ecx)

#undef __TBB_MACHINE_DEFINE_ATOMICS

#endif /* __TBB_MSVC_PART_WORD_INTERLOCKED_INTRINSICS_PRESENT */

//TODO: Check if it possible and profitable for IA-32 architecture on (Linux and Windows)
//to use of 64-bit load/store via floating point registers together with full fence
//for sequentially consistent load/store, instead of CAS.
#define __TBB_USE_FETCHSTORE_AS_FULL_FENCED_STORE           1
#define __TBB_USE_GENERIC_HALF_FENCED_LOAD_STORE            1
#define __TBB_USE_GENERIC_RELAXED_LOAD_STORE                1
#define __TBB_USE_GENERIC_SEQUENTIAL_CONSISTENCY_LOAD_STORE 1


#if defined(_MSC_VER) && !defined(__INTEL_COMPILER)
    #pragma warning (pop)
#endif // warnings 4244, 4267 are back
