/**
 * Many similar implementations exist. See for example libwsbm
 * or the linux kernel include/atomic.h
 *
 * No copyright claimed on this file.
 *
 */

#include "no_extern_c.h"

#ifndef U_ATOMIC_H
#define U_ATOMIC_H

#include <stdbool.h>
#include <stdint.h>

/* Favor OS-provided implementations.
 *
 * Where no OS-provided implementation is available, fall back to
 * locally coded assembly, compiler intrinsic or ultimately a
 * mutex-based implementation.
 */
#if defined(__sun)
#define PIPE_ATOMIC_OS_SOLARIS
#elif defined(_MSC_VER)
#define PIPE_ATOMIC_MSVC_INTRINSIC
#elif defined(__GNUC__)
#define PIPE_ATOMIC_GCC_INTRINSIC
#else
#error "Unsupported platform"
#endif


/* Implementation using GCC-provided synchronization intrinsics
 */
#if defined(PIPE_ATOMIC_GCC_INTRINSIC)

#define PIPE_ATOMIC "GCC Sync Intrinsics"

#if defined(USE_GCC_ATOMIC_BUILTINS)

/* The builtins with explicit memory model are available since GCC 4.7. */
#define p_atomic_set(_v, _i) __atomic_store_n((_v), (_i), __ATOMIC_RELEASE)
#define p_atomic_read(_v) __atomic_load_n((_v), __ATOMIC_ACQUIRE)
#define p_atomic_read_relaxed(_v) __atomic_load_n((_v), __ATOMIC_RELAXED)
#define p_atomic_dec_zero(v) (__atomic_sub_fetch((v), 1, __ATOMIC_ACQ_REL) == 0)
#define p_atomic_inc(v) (void) __atomic_add_fetch((v), 1, __ATOMIC_ACQ_REL)
#define p_atomic_dec(v) (void) __atomic_sub_fetch((v), 1, __ATOMIC_ACQ_REL)
#define p_atomic_add(v, i) (void) __atomic_add_fetch((v), (i), __ATOMIC_ACQ_REL)
#define p_atomic_inc_return(v) __atomic_add_fetch((v), 1, __ATOMIC_ACQ_REL)
#define p_atomic_dec_return(v) __atomic_sub_fetch((v), 1, __ATOMIC_ACQ_REL)
#define p_atomic_add_return(v, i) __atomic_add_fetch((v), (i), __ATOMIC_ACQ_REL)
#define p_atomic_fetch_add(v, i) __atomic_fetch_add((v), (i), __ATOMIC_ACQ_REL)
#define p_atomic_xchg(v, i) __atomic_exchange_n((v), (i), __ATOMIC_ACQ_REL)
#define PIPE_NATIVE_ATOMIC_XCHG

#else

#define p_atomic_set(_v, _i) (*(_v) = (_i))
#define p_atomic_read(_v) (*(_v))
#define p_atomic_read_relaxed(_v) (*(_v))
#define p_atomic_dec_zero(v) (__sync_sub_and_fetch((v), 1) == 0)
#define p_atomic_inc(v) (void) __sync_add_and_fetch((v), 1)
#define p_atomic_dec(v) (void) __sync_sub_and_fetch((v), 1)
#define p_atomic_add(v, i) (void) __sync_add_and_fetch((v), (i))
#define p_atomic_inc_return(v) __sync_add_and_fetch((v), 1)
#define p_atomic_dec_return(v) __sync_sub_and_fetch((v), 1)
#define p_atomic_add_return(v, i) __sync_add_and_fetch((v), (i))
#define p_atomic_fetch_add(v, i) __sync_fetch_and_add((v), (i))

#endif

/* There is no __atomic_* compare and exchange that returns the current value.
 * Also, GCC 5.4 seems unable to optimize a compound statement expression that
 * uses an additional stack variable with __atomic_compare_exchange[_n].
 */
#define p_atomic_cmpxchg(v, old, _new) \
   __sync_val_compare_and_swap((v), (old), (_new))
#define p_atomic_cmpxchg_ptr(v, old, _new) p_atomic_cmpxchg(v, old, _new)

#endif



/* Unlocked version for single threaded environments, such as some
 * windows kernel modules.
 */
#if defined(PIPE_ATOMIC_OS_UNLOCKED)

#define PIPE_ATOMIC "Unlocked"

#define p_atomic_set(_v, _i) (*(_v) = (_i))
#define p_atomic_read(_v) (*(_v))
#define p_atomic_read_relaxed(_v) (*(_v))
#define p_atomic_dec_zero(_v) (p_atomic_dec_return(_v) == 0)
#define p_atomic_inc(_v) ((void) p_atomic_inc_return(_v))
#define p_atomic_dec(_v) ((void) p_atomic_dec_return(_v))
#define p_atomic_add(_v, _i) ((void) p_atomic_add_return((_v), (_i)))
#define p_atomic_inc_return(_v) (++(*(_v)))
#define p_atomic_dec_return(_v) (--(*(_v)))
#define p_atomic_add_return(_v, _i) (*(_v) = *(_v) + (_i))
#define p_atomic_fetch_add(_v, _i) (*(_v) = *(_v) + (_i), *(_v) - (_i))
#define p_atomic_cmpxchg(_v, _old, _new) (*(_v) == (_old) ? (*(_v) = (_new), (_old)) : *(_v))
#define p_atomic_cmpxchg_ptr(_v, _old, _new) p_atomic_cmpxchg(_v, _old, _new)

#endif


#if defined(PIPE_ATOMIC_MSVC_INTRINSIC)

#define PIPE_ATOMIC "MSVC Intrinsics"

/* We use the Windows header's Interlocked*64 functions instead of the
 * _Interlocked*64 intrinsics wherever we can, as support for the latter varies
 * with target CPU, whereas Windows headers take care of all portability
 * issues: using intrinsics where available, falling back to library
 * implementations where not.
 */
#include <intrin.h>
#include <assert.h>

__forceinline char _interlockedadd8(char volatile * _Addend, char _Value)
{
   return _InterlockedExchangeAdd8(_Addend, _Value) + _Value;
}

__forceinline short _interlockedadd16(short volatile * _Addend, short _Value)
{
   return _InterlockedExchangeAdd16(_Addend, _Value) + _Value;
}

/* MSVC supports decltype keyword, but it's only supported on C++ and doesn't
 * quite work here; and if a C++-only solution is worthwhile, then it would be
 * better to use templates / function overloading, instead of decltype magic.
 * Therefore, we rely on implicit casting to LONGLONG for the functions that return
 */

#define p_atomic_set(_v, _i) (*(_v) = (_i))
#define p_atomic_read(_v) (*(_v))
#define p_atomic_read_relaxed(_v) (*(_v))

#define p_atomic_dec_zero(_v) \
   (p_atomic_dec_return(_v) == 0)

#define p_atomic_inc(_v) \
   ((void) p_atomic_inc_return(_v))

#define p_atomic_inc_return(_v) (\
   sizeof *(_v) == sizeof(char)    ? p_atomic_add_return((_v), 1) : \
   sizeof *(_v) == sizeof(short)   ? _InterlockedIncrement16((short *)  (_v)) : \
   sizeof *(_v) == sizeof(long)    ? _InterlockedIncrement  ((long *)   (_v)) : \
   sizeof *(_v) == sizeof(__int64) ? _interlockedincrement64((__int64 *)(_v)) : \
                                     (assert(!"should not get here"), 0))

#define p_atomic_dec(_v) \
   ((void) p_atomic_dec_return(_v))

#define p_atomic_dec_return(_v) (\
   sizeof *(_v) == sizeof(char)    ? p_atomic_add_return((_v), -1) : \
   sizeof *(_v) == sizeof(short)   ? _InterlockedDecrement16((short *)  (_v)) : \
   sizeof *(_v) == sizeof(long)    ? _InterlockedDecrement  ((long *)   (_v)) : \
   sizeof *(_v) == sizeof(__int64) ? _interlockeddecrement64((__int64 *)(_v)) : \
                                     (assert(!"should not get here"), 0))

#define p_atomic_add(_v, _i) \
   ((void) p_atomic_fetch_add((_v), (_i)))

#define p_atomic_add_return(_v, _i) (\
   sizeof *(_v) == sizeof(char)    ? _interlockedadd8 ((char *)   (_v), (_i)) : \
   sizeof *(_v) == sizeof(short)   ? _interlockedadd16((short *)  (_v), (_i)) : \
   sizeof *(_v) == sizeof(long)    ? _interlockedadd  ((long *)   (_v), (_i)) : \
   sizeof *(_v) == sizeof(__int64) ? _interlockedadd64((__int64 *)(_v), (_i)) : \
                                     (assert(!"should not get here"), 0))

#define p_atomic_fetch_add(_v, _i) (\
   sizeof *(_v) == sizeof(char)    ? _InterlockedExchangeAdd8 ((char *)   (_v), (_i)) : \
   sizeof *(_v) == sizeof(short)   ? _InterlockedExchangeAdd16((short *)  (_v), (_i)) : \
   sizeof *(_v) == sizeof(long)    ? _InterlockedExchangeAdd  ((long *)   (_v), (_i)) : \
   sizeof *(_v) == sizeof(__int64) ? _interlockedexchangeadd64((__int64 *)(_v), (_i)) : \
                                     (assert(!"should not get here"), 0))

#define p_atomic_cmpxchg(_v, _old, _new) (\
   sizeof *(_v) == sizeof(char)    ? _InterlockedCompareExchange8 ((char *)   (_v), (char)   (_new), (char)   (_old)) : \
   sizeof *(_v) == sizeof(short)   ? _InterlockedCompareExchange16((short *)  (_v), (short)  (_new), (short)  (_old)) : \
   sizeof *(_v) == sizeof(long)    ? _InterlockedCompareExchange  ((long *)   (_v), (long)   (_new), (long)   (_old)) : \
   sizeof *(_v) == sizeof(__int64) ? _InterlockedCompareExchange64((__int64 *)(_v), (__int64)(_new), (__int64)(_old)) : \
                                     (assert(!"should not get here"), 0))

#if defined(_WIN64)
#define p_atomic_cmpxchg_ptr(_v, _old, _new) (void *)_InterlockedCompareExchange64((__int64 *)(_v), (__int64)(_new), (__int64)(_old))
#else
#define p_atomic_cmpxchg_ptr(_v, _old, _new) (void *)_InterlockedCompareExchange((long *)(_v), (long)(_new), (long)(_old))
#endif

#define PIPE_NATIVE_ATOMIC_XCHG
#define p_atomic_xchg(_v, _new) (\
   sizeof *(_v) == sizeof(char)    ? _InterlockedExchange8 ((char *)   (_v), (char)   (_new)) : \
   sizeof *(_v) == sizeof(short)   ? _InterlockedExchange16((short *)  (_v), (short)  (_new)) : \
   sizeof *(_v) == sizeof(long)    ? _InterlockedExchange  ((long *)   (_v), (long)   (_new)) : \
   sizeof *(_v) == sizeof(__int64) ? _interlockedexchange64((__int64 *)(_v), (__int64)(_new)) : \
                                     (assert(!"should not get here"), 0))

#endif

#if defined(PIPE_ATOMIC_OS_SOLARIS)

#define PIPE_ATOMIC "Solaris OS atomic functions"

#include <atomic.h>
#include <assert.h>

#define p_atomic_set(_v, _i) (*(_v) = (_i))
#define p_atomic_read(_v) (*(_v))

#define p_atomic_dec_zero(v) (\
   sizeof(*v) == sizeof(uint8_t)  ? atomic_dec_8_nv ((uint8_t  *)(v)) == 0 : \
   sizeof(*v) == sizeof(uint16_t) ? atomic_dec_16_nv((uint16_t *)(v)) == 0 : \
   sizeof(*v) == sizeof(uint32_t) ? atomic_dec_32_nv((uint32_t *)(v)) == 0 : \
   sizeof(*v) == sizeof(uint64_t) ? atomic_dec_64_nv((uint64_t *)(v)) == 0 : \
                                    (assert(!"should not get here"), 0))

#define p_atomic_inc(v) (void) (\
   sizeof(*v) == sizeof(uint8_t)  ? atomic_inc_8 ((uint8_t  *)(v)) : \
   sizeof(*v) == sizeof(uint16_t) ? atomic_inc_16((uint16_t *)(v)) : \
   sizeof(*v) == sizeof(uint32_t) ? atomic_inc_32((uint32_t *)(v)) : \
   sizeof(*v) == sizeof(uint64_t) ? atomic_inc_64((uint64_t *)(v)) : \
                                    (assert(!"should not get here"), 0))

#define p_atomic_inc_return(v) (__typeof(*v))( \
   sizeof(*v) == sizeof(uint8_t)  ? atomic_inc_8_nv ((uint8_t  *)(v)) : \
   sizeof(*v) == sizeof(uint16_t) ? atomic_inc_16_nv((uint16_t *)(v)) : \
   sizeof(*v) == sizeof(uint32_t) ? atomic_inc_32_nv((uint32_t *)(v)) : \
   sizeof(*v) == sizeof(uint64_t) ? atomic_inc_64_nv((uint64_t *)(v)) : \
                                    (assert(!"should not get here"), 0))

#define p_atomic_dec(v) (void) ( \
   sizeof(*v) == sizeof(uint8_t)  ? atomic_dec_8 ((uint8_t  *)(v)) : \
   sizeof(*v) == sizeof(uint16_t) ? atomic_dec_16((uint16_t *)(v)) : \
   sizeof(*v) == sizeof(uint32_t) ? atomic_dec_32((uint32_t *)(v)) : \
   sizeof(*v) == sizeof(uint64_t) ? atomic_dec_64((uint64_t *)(v)) : \
                                    (assert(!"should not get here"), 0))

#define p_atomic_dec_return(v) (__typeof(*v))( \
   sizeof(*v) == sizeof(uint8_t)  ? atomic_dec_8_nv ((uint8_t  *)(v)) : \
   sizeof(*v) == sizeof(uint16_t) ? atomic_dec_16_nv((uint16_t *)(v)) : \
   sizeof(*v) == sizeof(uint32_t) ? atomic_dec_32_nv((uint32_t *)(v)) : \
   sizeof(*v) == sizeof(uint64_t) ? atomic_dec_64_nv((uint64_t *)(v)) : \
                                    (assert(!"should not get here"), 0))

#define p_atomic_add(v, i) (void) ( \
   sizeof(*v) == sizeof(uint8_t)  ? atomic_add_8 ((uint8_t  *)(v), (i)) : \
   sizeof(*v) == sizeof(uint16_t) ? atomic_add_16((uint16_t *)(v), (i)) : \
   sizeof(*v) == sizeof(uint32_t) ? atomic_add_32((uint32_t *)(v), (i)) : \
   sizeof(*v) == sizeof(uint64_t) ? atomic_add_64((uint64_t *)(v), (i)) : \
                                    (assert(!"should not get here"), 0))

#define p_atomic_add_return(v, i) (__typeof(*v)) ( \
   sizeof(*v) == sizeof(uint8_t)  ? atomic_add_8_nv ((uint8_t  *)(v), (i)) : \
   sizeof(*v) == sizeof(uint16_t) ? atomic_add_16_nv((uint16_t *)(v), (i)) : \
   sizeof(*v) == sizeof(uint32_t) ? atomic_add_32_nv((uint32_t *)(v), (i)) : \
   sizeof(*v) == sizeof(uint64_t) ? atomic_add_64_nv((uint64_t *)(v), (i)) : \
                                    (assert(!"should not get here"), 0))

#define p_atomic_fetch_add(v, i) (__typeof(*v)) ( \
   sizeof(*v) == sizeof(uint8_t)  ? atomic_add_8_nv ((uint8_t  *)(v), (i)) - (i) : \
   sizeof(*v) == sizeof(uint16_t) ? atomic_add_16_nv((uint16_t *)(v), (i)) - (i) : \
   sizeof(*v) == sizeof(uint32_t) ? atomic_add_32_nv((uint32_t *)(v), (i)) - (i) : \
   sizeof(*v) == sizeof(uint64_t) ? atomic_add_64_nv((uint64_t *)(v), (i)) - (i) : \
                                    (assert(!"should not get here"), 0))

#define p_atomic_cmpxchg(v, old, _new) (__typeof(*v))( \
   sizeof(*v) == sizeof(uint8_t)  ? atomic_cas_8 ((uint8_t  *)(v), (uint8_t )(old), (uint8_t )(_new)) : \
   sizeof(*v) == sizeof(uint16_t) ? atomic_cas_16((uint16_t *)(v), (uint16_t)(old), (uint16_t)(_new)) : \
   sizeof(*v) == sizeof(uint32_t) ? atomic_cas_32((uint32_t *)(v), (uint32_t)(old), (uint32_t)(_new)) : \
   sizeof(*v) == sizeof(uint64_t) ? atomic_cas_64((uint64_t *)(v), (uint64_t)(old), (uint64_t)(_new)) : \
                                    (assert(!"should not get here"), 0))

#if INTPTR_MAX == INT32_MAX
#define p_atomic_cmpxchg_ptr(v, old, _new) (__typeof(*v))(atomic_cas_32((uint32_t *)(v), (uint32_t)(old), (uint32_t)(_new)))
#else
#define p_atomic_cmpxchg_ptr(v, old, _new) (__typeof(*v))(atomic_cas_64((uint64_t *)(v), (uint64_t)(old), (uint64_t)(_new)))
#endif

#endif

#ifndef PIPE_ATOMIC
#error "No pipe_atomic implementation selected"
#endif

#ifndef PIPE_NATIVE_ATOMIC_XCHG
static inline uint8_t p_atomic_xchg_8(uint8_t *v, uint8_t i)
{
   uint8_t actual = p_atomic_read(v);
   uint8_t expected;
   do {
      expected = actual;
      actual = p_atomic_cmpxchg(v, expected, i);
   } while (expected != actual);
   return actual;
}

static inline uint16_t p_atomic_xchg_16(uint16_t *v, uint16_t i)
{
   uint16_t actual = p_atomic_read(v);
   uint16_t expected;
   do {
      expected = actual;
      actual = p_atomic_cmpxchg(v, expected, i);
   } while (expected != actual);
   return actual;
}

static inline uint32_t p_atomic_xchg_32(uint32_t *v, uint32_t i)
{
   uint32_t actual = p_atomic_read(v);
   uint32_t expected;
   do {
      expected = actual;
      actual = p_atomic_cmpxchg(v, expected, i);
   } while (expected != actual);
   return actual;
}

static inline uint64_t p_atomic_xchg_64(uint64_t *v, uint64_t i)
{
   uint64_t actual = p_atomic_read(v);
   uint64_t expected;
   do {
      expected = actual;
      actual = p_atomic_cmpxchg(v, expected, i);
   } while (expected != actual);
   return actual;
}

#define p_atomic_xchg(v, i) (__typeof(*(v)))( \
   sizeof(*(v)) == sizeof(uint8_t)  ? p_atomic_xchg_8 ((uint8_t  *)(v), (uint8_t )(i)) : \
   sizeof(*(v)) == sizeof(uint16_t) ? p_atomic_xchg_16((uint16_t *)(v), (uint16_t)(i)) : \
   sizeof(*(v)) == sizeof(uint32_t) ? p_atomic_xchg_32((uint32_t *)(v), (uint32_t)(i)) : \
   sizeof(*(v)) == sizeof(uint64_t) ? p_atomic_xchg_64((uint64_t *)(v), (uint64_t)(i)) : \
                                      (assert(!"should not get here"), 0))
#endif

#endif /* U_ATOMIC_H */
