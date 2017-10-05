/*
 * Copyright © 2007  Chris Wilson
 * Copyright © 2009,2010  Red Hat, Inc.
 * Copyright © 2011,2012  Google, Inc.
 *
 *  This is part of HarfBuzz, a text shaping library.
 *
 * Permission is hereby granted, without written agreement and without
 * license or royalty fees, to use, copy, modify, and distribute this
 * software and its documentation for any purpose, provided that the
 * above copyright notice and the following two paragraphs appear in
 * all copies of this software.
 *
 * IN NO EVENT SHALL THE COPYRIGHT HOLDER BE LIABLE TO ANY PARTY FOR
 * DIRECT, INDIRECT, SPECIAL, INCIDENTAL, OR CONSEQUENTIAL DAMAGES
 * ARISING OUT OF THE USE OF THIS SOFTWARE AND ITS DOCUMENTATION, EVEN
 * IF THE COPYRIGHT HOLDER HAS BEEN ADVISED OF THE POSSIBILITY OF SUCH
 * DAMAGE.
 *
 * THE COPYRIGHT HOLDER SPECIFICALLY DISCLAIMS ANY WARRANTIES, INCLUDING,
 * BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND
 * FITNESS FOR A PARTICULAR PURPOSE.  THE SOFTWARE PROVIDED HEREUNDER IS
 * ON AN "AS IS" BASIS, AND THE COPYRIGHT HOLDER HAS NO OBLIGATION TO
 * PROVIDE MAINTENANCE, SUPPORT, UPDATES, ENHANCEMENTS, OR MODIFICATIONS.
 *
 * Contributor(s):
 *	Chris Wilson <chris@chris-wilson.co.uk>
 * Red Hat Author(s): Behdad Esfahbod
 * Google Author(s): Behdad Esfahbod
 */

#ifndef HB_ATOMIC_PRIVATE_HH
#define HB_ATOMIC_PRIVATE_HH

#include "hb-private.hh"


/*
 * Atomic integers and pointers.
 */


/* We need external help for these */

#if defined(hb_atomic_int_impl_add) \
 && defined(hb_atomic_ptr_impl_get) \
 && defined(hb_atomic_ptr_impl_cmpexch)

/* Defined externally, i.e. in config.h. */


#elif !defined(HB_NO_MT) && defined(__ATOMIC_CONSUME)

/* C++11-style GCC primitives. */

#define hb_atomic_int_impl_add(AI, V)		__atomic_fetch_add ((AI), (V), __ATOMIC_ACQ_REL)
#define hb_atomic_int_impl_set_relaxed(AI, V)	__atomic_store_n ((AI), (V), __ATOMIC_RELAXED)
#define hb_atomic_int_impl_get_relaxed(AI)	__atomic_load_n ((AI), __ATOMIC_RELAXED)

#define hb_atomic_ptr_impl_set_relaxed(P, V)	__atomic_store_n ((P), (V), __ATOMIC_RELAXED)
#define hb_atomic_ptr_impl_get_relaxed(P)	__atomic_load_n ((P), __ATOMIC_RELAXED)
#define hb_atomic_ptr_impl_get(P)		__atomic_load_n ((P), __ATOMIC_CONSUME)
static inline bool
_hb_atomic_ptr_impl_cmplexch (const void **P, const void *O_, const void *N)
{
  const void *O = O_; // Need lvalue
  return __atomic_compare_exchange_n ((void **) P, (void **) &O, (void *) N, true, __ATOMIC_ACQ_REL, __ATOMIC_RELAXED);
}
#define hb_atomic_ptr_impl_cmpexch(P,O,N)	_hb_atomic_ptr_impl_cmplexch ((const void **) (P), (O), (N))

#elif !defined(HB_NO_MT) && __cplusplus >= 201103L

/* C++11 atomics. */

#include <atomic>

#define hb_atomic_int_impl_add(AI, V)		(reinterpret_cast<std::atomic<int> *> (AI)->fetch_add ((V), std::memory_order_acq_rel))
#define hb_atomic_int_impl_set_relaxed(AI, V)	(reinterpret_cast<std::atomic<int> *> (AI)->store ((V), std::memory_order_relaxed))
#define hb_atomic_int_impl_get_relaxed(AI)	(reinterpret_cast<std::atomic<int> *> (AI)->load (std::memory_order_relaxed))

#define hb_atomic_ptr_impl_set_relaxed(P, V)	(reinterpret_cast<std::atomic<void*> *> (P)->store ((V), std::memory_order_relaxed))
#define hb_atomic_ptr_impl_get_relaxed(P)	(reinterpret_cast<std::atomic<void*> *> (P)->load (std::memory_order_relaxed))
#define hb_atomic_ptr_impl_get(P)		(reinterpret_cast<std::atomic<void*> *> (P)->load (std::memory_order_consume))
static inline bool
_hb_atomic_ptr_impl_cmplexch (const void **P, const void *O_, const void *N)
{
  const void *O = O_; // Need lvalue
  return reinterpret_cast<std::atomic<const void*> *> (P)->compare_exchange_weak (O, N, std::memory_order_acq_rel, std::memory_order_relaxed);
}
#define hb_atomic_ptr_impl_cmpexch(P,O,N)	_hb_atomic_ptr_impl_cmplexch ((const void **) (P), (O), (N))


#elif !defined(HB_NO_MT) && (defined(_WIN32) || defined(__CYGWIN__))

#include <windows.h>

static inline void _hb_memory_barrier (void)
{
#if !defined(MemoryBarrier)
  /* MinGW has a convoluted history of supporting MemoryBarrier. */
  long dummy = 0;
  InterlockedExchange (&dummy, 1);
#else
  MemoryBarrier ();
#endif
}
#define _hb_memory_barrier()			_hb_memory_barrier ()

#define hb_atomic_int_impl_add(AI, V)		InterlockedExchangeAdd ((unsigned *) (AI), (V))

#define hb_atomic_ptr_impl_cmpexch(P,O,N)	(InterlockedCompareExchangePointer ((void **) (P), (void *) (N), (void *) (O)) == (void *) (O))


#elif !defined(HB_NO_MT) && defined(HAVE_INTEL_ATOMIC_PRIMITIVES)

#define _hb_memory_barrier()			__sync_synchronize ()

#define hb_atomic_int_impl_add(AI, V)		__sync_fetch_and_add ((AI), (V))

#define hb_atomic_ptr_impl_cmpexch(P,O,N)	__sync_bool_compare_and_swap ((P), (O), (N))


#elif !defined(HB_NO_MT) && defined(HAVE_SOLARIS_ATOMIC_OPS)

#include <atomic.h>
#include <mbarrier.h>

#define _hb_memory_r_barrier()			__machine_r_barrier ()
#define _hb_memory_w_barrier()			__machine_w_barrier ()
#define _hb_memory_barrier()			__machine_rw_barrier ()

static inline int _hb_fetch_and_add (int *AI, int V)
{
  _hb_memory_w_barrier ();
  int result = atomic_add_int_nv ((uint_t *) AI, V);
  _hb_memory_r_barrier ();
  return result;
}
static inline bool _hb_compare_and_swap_ptr (const void **P, const void *O, const void *N)
{
  _hb_memory_w_barrier ();
  int result = atomic_cas_ptr ((void **) P, (void *) O, (void *) N) == (void *) O;
  _hb_memory_r_barrier ();
  return result;
}

#define hb_atomic_int_impl_add(AI, V)           _hb_fetch_and_add ((AI), (V))

#define hb_atomic_ptr_impl_cmpexch(P,O,N)       _hb_compare_and_swap_ptr ((const void **) (P), (O), (N))


#elif !defined(HB_NO_MT) && defined(__APPLE__)

#include <libkern/OSAtomic.h>
#ifdef __MAC_OS_X_MIN_REQUIRED
#include <AvailabilityMacros.h>
#elif defined(__IPHONE_OS_MIN_REQUIRED)
#include <Availability.h>
#endif

#define _hb_memory_barrier()			OSMemoryBarrier ()

#define hb_atomic_int_impl_add(AI, V)		(OSAtomicAdd32Barrier ((V), (AI)) - (V))

#if (MAC_OS_X_VERSION_MIN_REQUIRED > MAC_OS_X_VERSION_10_4 || __IPHONE_VERSION_MIN_REQUIRED >= 20100)
#define hb_atomic_ptr_impl_cmpexch(P,O,N)	OSAtomicCompareAndSwapPtrBarrier ((void *) (O), (void *) (N), (void **) (P))
#else
#if __ppc64__ || __x86_64__ || __aarch64__
#define hb_atomic_ptr_impl_cmpexch(P,O,N)	OSAtomicCompareAndSwap64Barrier ((int64_t) (void *) (O), (int64_t) (void *) (N), (int64_t*) (P))
#else
#define hb_atomic_ptr_impl_cmpexch(P,O,N)	OSAtomicCompareAndSwap32Barrier ((int32_t) (void *) (O), (int32_t) (void *) (N), (int32_t*) (P))
#endif
#endif


#elif !defined(HB_NO_MT) && defined(_AIX) && defined(__IBMCPP__)

#include <builtins.h>

#define _hb_memory_barrier()			__lwsync ()

static inline int _hb_fetch_and_add (int *AI, int V)
{
  _hb_memory_barrier ();
  int result = __fetch_and_add (AI, V);
  _hb_memory_barrier ();
  return result;
}
static inline bool _hb_compare_and_swaplp (long *P, long O, long N)
{
  _hb_memory_barrier ();
  bool result = __compare_and_swaplp (P, &O, N);
  _hb_memory_barrier ();
  return result;
}

#define hb_atomic_int_impl_add(AI, V)           _hb_fetch_and_add ((AI), (V))

#define hb_atomic_ptr_impl_cmpexch(P,O,N)       _hb_compare_and_swaplp ((long *) (P), (long) (O), (long) (N))
static_assert ((sizeof (long) == sizeof (void *)), "");


#elif !defined(HB_NO_MT)

#define HB_ATOMIC_INT_NIL 1 /* Warn that fallback implementation is in use. */

#define _hb_memory_barrier()

#define hb_atomic_int_impl_add(AI, V)		((*(AI) += (V)) - (V))

#define hb_atomic_ptr_impl_cmpexch(P,O,N)	(* (void **) (P) == (void *) (O) ? (* (void **) (P) = (void *) (N), true) : false)


#else /* HB_NO_MT */

#define hb_atomic_int_impl_add(AI, V)		((*(AI) += (V)) - (V))

#define _hb_memory_barrier()

#define hb_atomic_ptr_impl_cmpexch(P,O,N)	(* (void **) (P) == (void *) (O) ? (* (void **) (P) = (void *) (N), true) : false)


#endif


#ifndef _hb_memory_r_barrier
#define _hb_memory_r_barrier()			_hb_memory_barrier ()
#endif
#ifndef _hb_memory_w_barrier
#define _hb_memory_w_barrier()			_hb_memory_barrier ()
#endif
#ifndef hb_atomic_int_impl_set_relaxed
#define hb_atomic_int_impl_set_relaxed(AI, V)	(*(AI) = (V))
#endif
#ifndef hb_atomic_int_impl_get_relaxed
#define hb_atomic_int_impl_get_relaxed(AI)	(*(AI))
#endif

#ifndef hb_atomic_ptr_impl_set_relaxed
#define hb_atomic_ptr_impl_set_relaxed(P, V)	(*(P) = (V))
#endif
#ifndef hb_atomic_ptr_impl_get_relaxed
#define hb_atomic_ptr_impl_get_relaxed(P)	(*(P))
#endif
#ifndef hb_atomic_ptr_impl_get
inline void *hb_atomic_ptr_impl_get (void **P)	{ void *v = *P; _hb_memory_r_barrier (); return v; }
#endif


#define HB_ATOMIC_INT_INIT(V)          {V}
struct hb_atomic_int_t
{
  inline void set_relaxed (int v_) const { hb_atomic_int_impl_set_relaxed (&v, v_); }
  inline int get_relaxed (void) const { return hb_atomic_int_impl_get_relaxed (&v); }
  inline int inc (void) { return hb_atomic_int_impl_add (&v,  1); }
  inline int dec (void) { return hb_atomic_int_impl_add (&v, -1); }

  mutable int v;
};


template <typename T> struct hb_remove_ptr_t { typedef T value; };
template <typename T> struct hb_remove_ptr_t<T *> { typedef T value; };

#define HB_ATOMIC_PTR_INIT(V)          {V}
template <typename P>
struct hb_atomic_ptr_t
{
  typedef typename hb_remove_ptr_t<P>::value T;

  inline void init (T* v_ = nullptr) { set_relaxed (v_); }
  inline void set_relaxed (T* v_) const { hb_atomic_ptr_impl_set_relaxed (&v, v_); }
  inline T *get_relaxed (void) const { return hb_atomic_ptr_impl_get_relaxed (&v); }
  inline T *get (void) const { return (T *) hb_atomic_ptr_impl_get ((void **) &v); }
  inline bool cmpexch (const T *old, T *new_) const{ return hb_atomic_ptr_impl_cmpexch (&v, old, new_); }

  mutable T *v;
};


#endif /* HB_ATOMIC_PRIVATE_HH */
