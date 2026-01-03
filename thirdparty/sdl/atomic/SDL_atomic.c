/*
  Simple DirectMedia Layer
  Copyright (C) 1997-2025 Sam Lantinga <slouken@libsdl.org>

  This software is provided 'as-is', without any express or implied
  warranty.  In no event will the authors be held liable for any damages
  arising from the use of this software.

  Permission is granted to anyone to use this software for any purpose,
  including commercial applications, and to alter it and redistribute it
  freely, subject to the following restrictions:

  1. The origin of this software must not be misrepresented; you must not
     claim that you wrote the original software. If you use this software
     in a product, an acknowledgment in the product documentation would be
     appreciated but is not required.
  2. Altered source versions must be plainly marked as such, and must not be
     misrepresented as being the original software.
  3. This notice may not be removed or altered from any source distribution.
*/
#include "SDL_internal.h"

#if defined(_MSC_VER) && (_MSC_VER >= 1900)
#include <intrin.h>
#define HAVE_MSC_ATOMICS 1
#endif

#ifdef SDL_PLATFORM_MACOS // !!! FIXME: should we favor gcc atomics?
#include <libkern/OSAtomic.h>
#endif

#if !defined(HAVE_GCC_ATOMICS) && defined(SDL_PLATFORM_SOLARIS)
#include <atomic.h>
#endif

// The __atomic_load_n() intrinsic showed up in different times for different compilers.
#ifdef __clang__
#if __has_builtin(__atomic_load_n) || defined(HAVE_GCC_ATOMICS)
/* !!! FIXME: this advertises as available in the NDK but uses an external symbol we don't have.
   It might be in a later NDK or we might need an extra library? --ryan. */
#ifndef SDL_PLATFORM_ANDROID
#define HAVE_ATOMIC_LOAD_N 1
#endif
#endif
#elif defined(__GNUC__)
#if (__GNUC__ >= 5)
#define HAVE_ATOMIC_LOAD_N 1
#endif
#endif

/* *INDENT-OFF* */ // clang-format off
#if defined(__WATCOMC__) && defined(__386__)
SDL_COMPILE_TIME_ASSERT(intsize, 4==sizeof(int));
#define HAVE_WATCOM_ATOMICS
extern __inline int _SDL_xchg_watcom(volatile int *a, int v);
#pragma aux _SDL_xchg_watcom = \
  "lock xchg [ecx], eax" \
  parm [ecx] [eax] \
  value [eax] \
  modify exact [eax];

extern __inline unsigned char _SDL_cmpxchg_watcom(volatile int *a, int newval, int oldval);
#pragma aux _SDL_cmpxchg_watcom = \
  "lock cmpxchg [edx], ecx" \
  "setz al" \
  parm [edx] [ecx] [eax] \
  value [al] \
  modify exact [eax];

extern __inline int _SDL_xadd_watcom(volatile int *a, int v);
#pragma aux _SDL_xadd_watcom = \
  "lock xadd [ecx], eax" \
  parm [ecx] [eax] \
  value [eax] \
  modify exact [eax];

#endif // __WATCOMC__ && __386__
/* *INDENT-ON* */ // clang-format on

/*
  If any of the operations are not provided then we must emulate some
  of them. That means we need a nice implementation of spin locks
  that avoids the "one big lock" problem. We use a vector of spin
  locks and pick which one to use based on the address of the operand
  of the function.

  To generate the index of the lock we first shift by 3 bits to get
  rid on the zero bits that result from 32 and 64 bit alignment of
  data. We then mask off all but 5 bits and use those 5 bits as an
  index into the table.

  Picking the lock this way insures that accesses to the same data at
  the same time will go to the same lock. OTOH, accesses to different
  data have only a 1/32 chance of hitting the same lock. That should
  pretty much eliminate the chances of several atomic operations on
  different data from waiting on the same "big lock". If it isn't
  then the table of locks can be expanded to a new size so long as
  the new size is a power of two.

  Contributed by Bob Pendleton, bob@pendleton.com
*/

#if !defined(HAVE_MSC_ATOMICS) && !defined(HAVE_GCC_ATOMICS) && !defined(SDL_PLATFORM_MACOS) && !defined(SDL_PLATFORM_SOLARIS) && !defined(HAVE_WATCOM_ATOMICS)
#define EMULATE_CAS 1
#endif

#ifdef EMULATE_CAS
static SDL_SpinLock locks[32];

static SDL_INLINE void enterLock(void *a)
{
    uintptr_t index = ((((uintptr_t)a) >> 3) & 0x1f);

    SDL_LockSpinlock(&locks[index]);
}

static SDL_INLINE void leaveLock(void *a)
{
    uintptr_t index = ((((uintptr_t)a) >> 3) & 0x1f);

    SDL_UnlockSpinlock(&locks[index]);
}
#endif

bool SDL_CompareAndSwapAtomicInt(SDL_AtomicInt *a, int oldval, int newval)
{
#ifdef HAVE_MSC_ATOMICS
    SDL_COMPILE_TIME_ASSERT(atomic_cas, sizeof(long) == sizeof(a->value));
    return _InterlockedCompareExchange((long *)&a->value, (long)newval, (long)oldval) == (long)oldval;
#elif defined(HAVE_WATCOM_ATOMICS)
    return _SDL_cmpxchg_watcom((volatile int *)&a->value, newval, oldval);
#elif defined(HAVE_GCC_ATOMICS)
    return __sync_bool_compare_and_swap(&a->value, oldval, newval);
#elif defined(SDL_PLATFORM_MACOS) // this is deprecated in 10.12 sdk; favor gcc atomics.
    return OSAtomicCompareAndSwap32Barrier(oldval, newval, &a->value);
#elif defined(SDL_PLATFORM_SOLARIS)
    SDL_COMPILE_TIME_ASSERT(atomic_cas, sizeof(uint_t) == sizeof(a->value));
    return ((int)atomic_cas_uint((volatile uint_t *)&a->value, (uint_t)oldval, (uint_t)newval) == oldval);
#elif defined(EMULATE_CAS)
    bool result = false;

    enterLock(a);
    if (a->value == oldval) {
        a->value = newval;
        result = true;
    }
    leaveLock(a);

    return result;
#else
#error Please define your platform.
#endif
}

bool SDL_CompareAndSwapAtomicU32(SDL_AtomicU32 *a, Uint32 oldval, Uint32 newval)
{
#ifdef HAVE_MSC_ATOMICS
    SDL_COMPILE_TIME_ASSERT(atomic_cas, sizeof(long) == sizeof(a->value));
    return _InterlockedCompareExchange((long *)&a->value, (long)newval, (long)oldval) == (long)oldval;
#elif defined(HAVE_WATCOM_ATOMICS)
    SDL_COMPILE_TIME_ASSERT(atomic_cas, sizeof(int) == sizeof(a->value));
    return _SDL_cmpxchg_watcom((volatile int *)&a->value, (int)newval, (int)oldval);
#elif defined(HAVE_GCC_ATOMICS)
    return __sync_bool_compare_and_swap(&a->value, oldval, newval);
#elif defined(SDL_PLATFORM_MACOS) // this is deprecated in 10.12 sdk; favor gcc atomics.
    return OSAtomicCompareAndSwap32Barrier((int32_t)oldval, (int32_t)newval, (int32_t*)&a->value);
#elif defined(SDL_PLATFORM_SOLARIS)
    SDL_COMPILE_TIME_ASSERT(atomic_cas, sizeof(uint_t) == sizeof(a->value));
    return ((Uint32)atomic_cas_uint((volatile uint_t *)&a->value, (uint_t)oldval, (uint_t)newval) == oldval);
#elif defined(EMULATE_CAS)
    bool result = false;

    enterLock(a);
    if (a->value == oldval) {
        a->value = newval;
        result = true;
    }
    leaveLock(a);

    return result;
#else
#error Please define your platform.
#endif
}

bool SDL_CompareAndSwapAtomicPointer(void **a, void *oldval, void *newval)
{
#ifdef HAVE_MSC_ATOMICS
    return _InterlockedCompareExchangePointer(a, newval, oldval) == oldval;
#elif defined(HAVE_WATCOM_ATOMICS)
    return _SDL_cmpxchg_watcom((int *)a, (long)newval, (long)oldval);
#elif defined(HAVE_GCC_ATOMICS)
    return __sync_bool_compare_and_swap(a, oldval, newval);
#elif defined(SDL_PLATFORM_MACOS) && defined(__LP64__)  // this is deprecated in 10.12 sdk; favor gcc atomics.
    return OSAtomicCompareAndSwap64Barrier((int64_t)oldval, (int64_t)newval, (int64_t *)a);
#elif defined(SDL_PLATFORM_MACOS) && !defined(__LP64__) // this is deprecated in 10.12 sdk; favor gcc atomics.
    return OSAtomicCompareAndSwap32Barrier((int32_t)oldval, (int32_t)newval, (int32_t *)a);
#elif defined(SDL_PLATFORM_SOLARIS)
    return (atomic_cas_ptr(a, oldval, newval) == oldval);
#elif defined(EMULATE_CAS)
    bool result = false;

    enterLock(a);
    if (*a == oldval) {
        *a = newval;
        result = true;
    }
    leaveLock(a);

    return result;
#else
#error Please define your platform.
#endif
}

int SDL_SetAtomicInt(SDL_AtomicInt *a, int v)
{
#ifdef HAVE_MSC_ATOMICS
    SDL_COMPILE_TIME_ASSERT(atomic_set, sizeof(long) == sizeof(a->value));
    return _InterlockedExchange((long *)&a->value, v);
#elif defined(HAVE_WATCOM_ATOMICS)
    return _SDL_xchg_watcom(&a->value, v);
#elif defined(HAVE_GCC_ATOMICS)
    return __sync_lock_test_and_set(&a->value, v);
#elif defined(SDL_PLATFORM_SOLARIS)
    SDL_COMPILE_TIME_ASSERT(atomic_set, sizeof(uint_t) == sizeof(a->value));
    return (int)atomic_swap_uint((volatile uint_t *)&a->value, v);
#else
    int value;
    do {
        value = a->value;
    } while (!SDL_CompareAndSwapAtomicInt(a, value, v));
    return value;
#endif
}

Uint32 SDL_SetAtomicU32(SDL_AtomicU32 *a, Uint32 v)
{
#ifdef HAVE_MSC_ATOMICS
    SDL_COMPILE_TIME_ASSERT(atomic_set, sizeof(long) == sizeof(a->value));
    return _InterlockedExchange((long *)&a->value, v);
#elif defined(HAVE_WATCOM_ATOMICS)
    return _SDL_xchg_watcom(&a->value, v);
#elif defined(HAVE_GCC_ATOMICS)
    return __sync_lock_test_and_set(&a->value, v);
#elif defined(SDL_PLATFORM_SOLARIS)
    SDL_COMPILE_TIME_ASSERT(atomic_set, sizeof(uint_t) == sizeof(a->value));
    return (Uint32)atomic_swap_uint((volatile uint_t *)&a->value, v);
#else
    Uint32 value;
    do {
        value = a->value;
    } while (!SDL_CompareAndSwapAtomicU32(a, value, v));
    return value;
#endif
}

void *SDL_SetAtomicPointer(void **a, void *v)
{
#ifdef HAVE_MSC_ATOMICS
    return _InterlockedExchangePointer(a, v);
#elif defined(HAVE_WATCOM_ATOMICS)
    return (void *)_SDL_xchg_watcom((int *)a, (long)v);
#elif defined(HAVE_GCC_ATOMICS)
    return __sync_lock_test_and_set(a, v);
#elif defined(SDL_PLATFORM_SOLARIS)
    return atomic_swap_ptr(a, v);
#else
    void *value;
    do {
        value = *a;
    } while (!SDL_CompareAndSwapAtomicPointer(a, value, v));
    return value;
#endif
}

int SDL_AddAtomicInt(SDL_AtomicInt *a, int v)
{
#ifdef HAVE_MSC_ATOMICS
    SDL_COMPILE_TIME_ASSERT(atomic_add, sizeof(long) == sizeof(a->value));
    return _InterlockedExchangeAdd((long *)&a->value, v);
#elif defined(HAVE_WATCOM_ATOMICS)
    SDL_COMPILE_TIME_ASSERT(atomic_add, sizeof(int) == sizeof(a->value));
    return _SDL_xadd_watcom((volatile int *)&a->value, v);
#elif defined(HAVE_GCC_ATOMICS)
    return __sync_fetch_and_add(&a->value, v);
#elif defined(SDL_PLATFORM_SOLARIS)
    int pv = a->value;
    membar_consumer();
    atomic_add_int((volatile uint_t *)&a->value, v);
    return pv;
#else
    int value;
    do {
        value = a->value;
    } while (!SDL_CompareAndSwapAtomicInt(a, value, (value + v)));
    return value;
#endif
}

int SDL_GetAtomicInt(SDL_AtomicInt *a)
{
#ifdef HAVE_ATOMIC_LOAD_N
    return __atomic_load_n(&a->value, __ATOMIC_SEQ_CST);
#elif defined(HAVE_MSC_ATOMICS)
    SDL_COMPILE_TIME_ASSERT(atomic_get, sizeof(long) == sizeof(a->value));
    return _InterlockedOr((long *)&a->value, 0);
#elif defined(HAVE_WATCOM_ATOMICS)
    return _SDL_xadd_watcom(&a->value, 0);
#elif defined(HAVE_GCC_ATOMICS)
    return __sync_or_and_fetch(&a->value, 0);
#elif defined(SDL_PLATFORM_MACOS) // this is deprecated in 10.12 sdk; favor gcc atomics.
    return sizeof(a->value) == sizeof(uint32_t) ? OSAtomicOr32Barrier(0, (volatile uint32_t *)&a->value) : OSAtomicAdd64Barrier(0, (volatile int64_t *)&a->value);
#elif defined(SDL_PLATFORM_SOLARIS)
    return atomic_or_uint_nv((volatile uint_t *)&a->value, 0);
#else
    int value;
    do {
        value = a->value;
    } while (!SDL_CompareAndSwapAtomicInt(a, value, value));
    return value;
#endif
}

Uint32 SDL_GetAtomicU32(SDL_AtomicU32 *a)
{
#ifdef HAVE_ATOMIC_LOAD_N
    return __atomic_load_n(&a->value, __ATOMIC_SEQ_CST);
#elif defined(HAVE_MSC_ATOMICS)
    SDL_COMPILE_TIME_ASSERT(atomic_get, sizeof(long) == sizeof(a->value));
    return (Uint32)_InterlockedOr((long *)&a->value, 0);
#elif defined(HAVE_WATCOM_ATOMICS)
    SDL_COMPILE_TIME_ASSERT(atomic_get, sizeof(int) == sizeof(a->value));
    return (Uint32)_SDL_xadd_watcom((volatile int *)&a->value, 0);
#elif defined(HAVE_GCC_ATOMICS)
    return __sync_or_and_fetch(&a->value, 0);
#elif defined(SDL_PLATFORM_MACOS) // this is deprecated in 10.12 sdk; favor gcc atomics.
    return OSAtomicOr32Barrier(0, (volatile uint32_t *)&a->value);
#elif defined(SDL_PLATFORM_SOLARIS)
    SDL_COMPILE_TIME_ASSERT(atomic_get, sizeof(uint_t) == sizeof(a->value));
    return (Uint32)atomic_or_uint_nv((volatile uint_t *)&a->value, 0);
#else
    Uint32 value;
    do {
        value = a->value;
    } while (!SDL_CompareAndSwapAtomicU32(a, value, value));
    return value;
#endif
}

void *SDL_GetAtomicPointer(void **a)
{
#ifdef HAVE_ATOMIC_LOAD_N
    return __atomic_load_n(a, __ATOMIC_SEQ_CST);
#elif defined(HAVE_MSC_ATOMICS)
    return _InterlockedCompareExchangePointer(a, NULL, NULL);
#elif defined(HAVE_GCC_ATOMICS)
    return __sync_val_compare_and_swap(a, (void *)0, (void *)0);
#elif defined(SDL_PLATFORM_SOLARIS)
    return atomic_cas_ptr(a, (void *)0, (void *)0);
#else
    void *value;
    do {
        value = *a;
    } while (!SDL_CompareAndSwapAtomicPointer(a, value, value));
    return value;
#endif
}

#ifdef SDL_MEMORY_BARRIER_USES_FUNCTION
#error This file should be built in arm mode so the mcr instruction is available for memory barriers
#endif

void SDL_MemoryBarrierReleaseFunction(void)
{
    SDL_MemoryBarrierRelease();
}

void SDL_MemoryBarrierAcquireFunction(void)
{
    SDL_MemoryBarrierAcquire();
}
