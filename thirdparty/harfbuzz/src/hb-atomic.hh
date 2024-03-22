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

#ifndef HB_ATOMIC_HH
#define HB_ATOMIC_HH

#include "hb.hh"
#include "hb-meta.hh"


/*
 * Atomic integers and pointers.
 */


/* We need external help for these */

#if defined(hb_atomic_int_impl_add) \
 && defined(hb_atomic_ptr_impl_get) \
 && defined(hb_atomic_ptr_impl_cmpexch)

/* Defined externally, i.e. in config.h. */


#elif !defined(HB_NO_MT) && defined(__ATOMIC_ACQUIRE)

/* C++11-style GCC primitives. We prefer these as they don't require linking to libstdc++ / libc++. */

#define _hb_memory_barrier()			__sync_synchronize ()

#define hb_atomic_int_impl_add(AI, V)		__atomic_fetch_add ((AI), (V), __ATOMIC_ACQ_REL)
#define hb_atomic_int_impl_set_relaxed(AI, V)	__atomic_store_n ((AI), (V), __ATOMIC_RELAXED)
#define hb_atomic_int_impl_set(AI, V)		__atomic_store_n ((AI), (V), __ATOMIC_RELEASE)
#define hb_atomic_int_impl_get_relaxed(AI)	__atomic_load_n ((AI), __ATOMIC_RELAXED)
#define hb_atomic_int_impl_get(AI)		__atomic_load_n ((AI), __ATOMIC_ACQUIRE)

#define hb_atomic_ptr_impl_set_relaxed(P, V)	__atomic_store_n ((P), (V), __ATOMIC_RELAXED)
#define hb_atomic_ptr_impl_get_relaxed(P)	__atomic_load_n ((P), __ATOMIC_RELAXED)
#define hb_atomic_ptr_impl_get(P)		__atomic_load_n ((P), __ATOMIC_ACQUIRE)
static inline bool
_hb_atomic_ptr_impl_cmplexch (const void **P, const void *O_, const void *N)
{
  const void *O = O_; // Need lvalue
  return __atomic_compare_exchange_n ((void **) P, (void **) &O, (void *) N, true, __ATOMIC_ACQ_REL, __ATOMIC_RELAXED);
}
#define hb_atomic_ptr_impl_cmpexch(P,O,N)	_hb_atomic_ptr_impl_cmplexch ((const void **) (P), (O), (N))


#elif !defined(HB_NO_MT)

/* C++11 atomics. */

#include <atomic>

#define _hb_memory_barrier()			std::atomic_thread_fence(std::memory_order_ack_rel)
#define _hb_memory_r_barrier()			std::atomic_thread_fence(std::memory_order_acquire)
#define _hb_memory_w_barrier()			std::atomic_thread_fence(std::memory_order_release)

#define hb_atomic_int_impl_add(AI, V)		(reinterpret_cast<std::atomic<std::decay<decltype (*(AI))>::type> *> (AI)->fetch_add ((V), std::memory_order_acq_rel))
#define hb_atomic_int_impl_set_relaxed(AI, V)	(reinterpret_cast<std::atomic<std::decay<decltype (*(AI))>::type> *> (AI)->store ((V), std::memory_order_relaxed))
#define hb_atomic_int_impl_set(AI, V)		(reinterpret_cast<std::atomic<std::decay<decltype (*(AI))>::type> *> (AI)->store ((V), std::memory_order_release))
#define hb_atomic_int_impl_get_relaxed(AI)	(reinterpret_cast<std::atomic<std::decay<decltype (*(AI))>::type> const *> (AI)->load (std::memory_order_relaxed))
#define hb_atomic_int_impl_get(AI)		(reinterpret_cast<std::atomic<std::decay<decltype (*(AI))>::type> const *> (AI)->load (std::memory_order_acquire))

#define hb_atomic_ptr_impl_set_relaxed(P, V)	(reinterpret_cast<std::atomic<void*> *> (P)->store ((V), std::memory_order_relaxed))
#define hb_atomic_ptr_impl_get_relaxed(P)	(reinterpret_cast<std::atomic<void*> const *> (P)->load (std::memory_order_relaxed))
#define hb_atomic_ptr_impl_get(P)		(reinterpret_cast<std::atomic<void*> *> (P)->load (std::memory_order_acquire))
static inline bool
_hb_atomic_ptr_impl_cmplexch (const void **P, const void *O_, const void *N)
{
  const void *O = O_; // Need lvalue
  return reinterpret_cast<std::atomic<const void*> *> (P)->compare_exchange_weak (O, N, std::memory_order_acq_rel, std::memory_order_relaxed);
}
#define hb_atomic_ptr_impl_cmpexch(P,O,N)	_hb_atomic_ptr_impl_cmplexch ((const void **) (P), (O), (N))


#else /* defined(HB_NO_MT) */

#define hb_atomic_int_impl_add(AI, V)		((*(AI) += (V)) - (V))
#define _hb_memory_barrier()			do {} while (0)
#define hb_atomic_ptr_impl_cmpexch(P,O,N)	(* (void **) (P) == (void *) (O) ? (* (void **) (P) = (void *) (N), true) : false)

#endif


/* This should never be disabled, even under HB_NO_MT.
 * except that MSVC gives me an internal compiler error, so disabled there.
 *
 * https://github.com/harfbuzz/harfbuzz/pull/4119
 */
#ifndef _hb_compiler_memory_r_barrier
#if defined(__ATOMIC_ACQUIRE) // gcc-like
static inline void _hb_compiler_memory_r_barrier () { asm volatile("": : :"memory"); }
#elif !defined(_MSC_VER)
#include <atomic>
#define _hb_compiler_memory_r_barrier() std::atomic_signal_fence (std::memory_order_acquire)
#else
static inline void _hb_compiler_memory_r_barrier () {}
#endif
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
#ifndef hb_atomic_int_impl_set
inline void hb_atomic_int_impl_set (int *AI, int v)	{ _hb_memory_w_barrier (); *AI = v; }
inline void hb_atomic_int_impl_set (short *AI, short v)	{ _hb_memory_w_barrier (); *AI = v; }
#endif
#ifndef hb_atomic_int_impl_get
inline int hb_atomic_int_impl_get (const int *AI)	{ int v = *AI; _hb_memory_r_barrier (); return v; }
inline short hb_atomic_int_impl_get (const short *AI)	{ short v = *AI; _hb_memory_r_barrier (); return v; }
#endif
#ifndef hb_atomic_ptr_impl_get
inline void *hb_atomic_ptr_impl_get (void ** const P)	{ void *v = *P; _hb_memory_r_barrier (); return v; }
#endif


struct hb_atomic_short_t
{
  hb_atomic_short_t () = default;
  constexpr hb_atomic_short_t (short v) : v (v) {}

  hb_atomic_short_t& operator = (short v_) { set_relaxed (v_); return *this; }
  operator short () const { return get_relaxed (); }

  void set_relaxed (short v_) { hb_atomic_int_impl_set_relaxed (&v, v_); }
  void set_release (short v_) { hb_atomic_int_impl_set (&v, v_); }
  short get_relaxed () const { return hb_atomic_int_impl_get_relaxed (&v); }
  short get_acquire () const { return hb_atomic_int_impl_get (&v); }
  short inc () { return hb_atomic_int_impl_add (&v,  1); }
  short dec () { return hb_atomic_int_impl_add (&v, -1); }

  short v = 0;
};

struct hb_atomic_int_t
{
  hb_atomic_int_t () = default;
  constexpr hb_atomic_int_t (int v) : v (v) {}

  hb_atomic_int_t& operator = (int v_) { set_relaxed (v_); return *this; }
  operator int () const { return get_relaxed (); }

  void set_relaxed (int v_) { hb_atomic_int_impl_set_relaxed (&v, v_); }
  void set_release (int v_) { hb_atomic_int_impl_set (&v, v_); }
  int get_relaxed () const { return hb_atomic_int_impl_get_relaxed (&v); }
  int get_acquire () const { return hb_atomic_int_impl_get (&v); }
  int inc () { return hb_atomic_int_impl_add (&v,  1); }
  int dec () { return hb_atomic_int_impl_add (&v, -1); }

  int v = 0;
};

template <typename P>
struct hb_atomic_ptr_t
{
  typedef hb_remove_pointer<P> T;

  hb_atomic_ptr_t () = default;
  constexpr hb_atomic_ptr_t (T* v) : v (v) {}
  hb_atomic_ptr_t (const hb_atomic_ptr_t &other) = delete;

  void init (T* v_ = nullptr) { set_relaxed (v_); }
  void set_relaxed (T* v_) { hb_atomic_ptr_impl_set_relaxed (&v, v_); }
  T *get_relaxed () const { return (T *) hb_atomic_ptr_impl_get_relaxed (&v); }
  T *get_acquire () const { return (T *) hb_atomic_ptr_impl_get ((void **) &v); }
  bool cmpexch (const T *old, T *new_) const { return hb_atomic_ptr_impl_cmpexch ((void **) &v, (void *) old, (void *) new_); }

  T * operator -> () const                    { return get_acquire (); }
  template <typename C> operator C * () const { return get_acquire (); }

  T *v = nullptr;
};

static inline bool hb_barrier ()
{
  _hb_compiler_memory_r_barrier ();
  return true;
}


#endif /* HB_ATOMIC_HH */
