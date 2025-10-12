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

#define HB_STL_ATOMIC_IMPL

#define _hb_memory_r_barrier()			std::atomic_thread_fence(std::memory_order_acquire)
#define _hb_memory_w_barrier()			std::atomic_thread_fence(std::memory_order_release)

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
template <typename T>
inline void hb_atomic_int_impl_set (T *AI, T v)	{ _hb_memory_w_barrier (); *AI = v; }
#endif
#ifndef hb_atomic_int_impl_get
template <typename T>
inline T hb_atomic_int_impl_get (const T *AI)	{ T v = *AI; _hb_memory_r_barrier (); return v; }
#endif
#ifndef hb_atomic_ptr_impl_get
inline void *hb_atomic_ptr_impl_get (void ** const P)	{ void *v = *P; _hb_memory_r_barrier (); return v; }
#endif

#ifdef HB_STL_ATOMIC_IMPL
template <typename T>
struct hb_atomic_t
{
  hb_atomic_t () = default;
  constexpr hb_atomic_t (T v) : v (v) {}
  constexpr hb_atomic_t (const hb_atomic_t& o) : v (o.get_relaxed ()) {}
  constexpr hb_atomic_t (hb_atomic_t&& o) : v (o.get_relaxed ()) { o.set_relaxed ({}); }

  hb_atomic_t &operator= (const hb_atomic_t& o) { set_relaxed (o.get_relaxed ()); return *this; }
  hb_atomic_t &operator= (hb_atomic_t&& o){ set_relaxed (o.get_relaxed ()); o.set_relaxed ({}); return *this; }
  hb_atomic_t &operator= (T v_)
  {
    set_relaxed (v_);
    return *this;
  }
  operator T () const { return get_relaxed (); }

  void set_relaxed (T v_) { v.store (v_, std::memory_order_relaxed); }
  void set_release (T v_) { v.store (v_, std::memory_order_release); }
  T get_relaxed () const { return v.load (std::memory_order_relaxed); }
  T get_acquire () const { return v.load (std::memory_order_acquire); }
  T inc () { return v.fetch_add (1, std::memory_order_acq_rel); }
  T dec () { return v.fetch_add (-1, std::memory_order_acq_rel); }

  int operator++ (int) { return inc (); }
  int operator-- (int) { return dec (); }
  long operator|= (long v_)
  {
    set_relaxed (get_relaxed () | v_);
    return *this;
  }

  friend void swap (hb_atomic_t &a, hb_atomic_t &b) noexcept
  {
    T v = a.get_acquire ();
    a.set_relaxed (b.get_acquire ());
    b.set_relaxed (v);
  }

  std::atomic<T> v = 0;
};

template <typename T>
struct hb_atomic_t<T *>
{
  hb_atomic_t () = default;
  constexpr hb_atomic_t (T *v) : v (v) {}
  hb_atomic_t (const hb_atomic_t &other) = delete;

  void init (T *v_ = nullptr) { set_relaxed (v_); }
  void set_relaxed (T *v_) { v.store (v_, std::memory_order_relaxed); }
  T *get_relaxed () const { return v.load (std::memory_order_relaxed); }
  T *get_acquire () const { return v.load (std::memory_order_acquire); }
  bool cmpexch (T *old, T *new_) { return v.compare_exchange_weak (old, new_, std::memory_order_acq_rel, std::memory_order_relaxed); }

  operator bool () const { return get_acquire () != nullptr; }
  T *operator->() const { return get_acquire (); }
  template <typename C>
  operator C * () const
  {
    return get_acquire ();
  }

  friend void swap (hb_atomic_t &a, hb_atomic_t &b) noexcept
  {
    T *p = a.get_acquire ();
    a.set_relaxed (b.get_acquire ());
    b.set_relaxed (p);
  }

  std::atomic<T *> v = nullptr;
};

#else

template <typename T>
struct hb_atomic_t
{
  hb_atomic_t () = default;
  constexpr hb_atomic_t (T v) : v (v) {}

  hb_atomic_t& operator = (T v_) { set_relaxed (v_); return *this; }
  operator T () const { return get_relaxed (); }

  void set_relaxed (T v_) { hb_atomic_int_impl_set_relaxed (&v, v_); }
  void set_release (T v_) { hb_atomic_int_impl_set (&v, v_); }
  T get_relaxed () const { return hb_atomic_int_impl_get_relaxed (&v); }
  T get_acquire () const { return hb_atomic_int_impl_get (&v); }
  T inc () { return hb_atomic_int_impl_add (&v,  1); }
  T dec () { return hb_atomic_int_impl_add (&v, -1); }

  int operator ++ (int) { return inc (); }
  int operator -- (int) { return dec (); }
  long operator |= (long v_) { set_relaxed (get_relaxed () | v_); return *this; }

  T v = 0;
};

template <typename T>
struct hb_atomic_t<T*>
{
  hb_atomic_t () = default;
  constexpr hb_atomic_t (T* v) : v (v) {}
  hb_atomic_t (const hb_atomic_t &other) = delete;

  void init (T* v_ = nullptr) { set_relaxed (v_); }
  void set_relaxed (T* v_) { hb_atomic_ptr_impl_set_relaxed (&v, v_); }
  T *get_relaxed () const { return (T *) hb_atomic_ptr_impl_get_relaxed (&v); }
  T *get_acquire () const { return (T *) hb_atomic_ptr_impl_get ((void **) &v); }
  bool cmpexch (T *old, T *new_) { return hb_atomic_ptr_impl_cmpexch ((void **) &v, (void *) old, (void *) new_); }

  operator bool () const { return get_acquire () != nullptr; }
  T * operator -> () const                    { return get_acquire (); }
  template <typename C> operator C * () const { return get_acquire (); }

  T *v = nullptr;
};

#endif

static inline bool hb_barrier ()
{
  _hb_compiler_memory_r_barrier ();
  return true;
}


#endif /* HB_ATOMIC_HH */
