/*
 * Copyright © 2018  Google, Inc.
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
 * Google Author(s): Behdad Esfahbod
 */

#ifndef HB_NULL_HH
#define HB_NULL_HH

#include "hb.hh"
#include "hb-meta.hh"


/*
 * Static pools
 */

/* Global nul-content Null pool.  Enlarge as necessary. */

#define HB_NULL_POOL_SIZE 448

template <typename T, typename>
struct _hb_has_min_size : hb_false_type {};
template <typename T>
struct _hb_has_min_size<T, hb_void_t<decltype (T::min_size)>>
	: hb_true_type {};
template <typename T>
using hb_has_min_size = _hb_has_min_size<T, void>;
#define hb_has_min_size(T) hb_has_min_size<T>::value

template <typename T, typename>
struct _hb_has_null_size : hb_false_type {};
template <typename T>
struct _hb_has_null_size<T, hb_void_t<decltype (T::null_size)>>
	: hb_true_type {};
template <typename T>
using hb_has_null_size = _hb_has_null_size<T, void>;
#define hb_has_null_size(T) hb_has_null_size<T>::value

/* Use SFINAE to sniff whether T has min_size; in which case return the larger
 * of sizeof(T) and T::null_size, otherwise return sizeof(T).
 *
 * The main purpose of this is to let structs communicate that they are not nullable,
 * by defining min_size but *not* null_size. */

/* The hard way...
 * https://stackoverflow.com/questions/7776448/sfinae-tried-with-bool-gives-compiler-error-template-argument-tvalue-invol
 */

template <typename T, typename>
struct _hb_null_size : hb_integral_constant<unsigned, sizeof (T)> {};
template <typename T>
struct _hb_null_size<T, hb_void_t<decltype (T::min_size)>>
	: hb_integral_constant<unsigned,
			       (sizeof (T) > T::null_size ? sizeof (T) : T::null_size)> {};
template <typename T>
using hb_null_size = _hb_null_size<T, void>;
#define hb_null_size(T) hb_null_size<T>::value

/* These doesn't belong here, but since is copy/paste from above, put it here. */

/* hb_static_size (T)
 * Returns T::static_size if T::min_size is defined, or sizeof (T) otherwise. */

template <typename T, typename>
struct _hb_static_size : hb_integral_constant<unsigned, sizeof (T)> {};
template <typename T>
struct _hb_static_size<T, hb_void_t<decltype (T::min_size)>> : hb_integral_constant<unsigned, T::static_size> {};
template <typename T>
using hb_static_size = _hb_static_size<T, void>;
#define hb_static_size(T) hb_static_size<T>::value

template <typename T, typename>
struct _hb_min_size : hb_integral_constant<unsigned, sizeof (T)> {};
template <typename T>
struct _hb_min_size<T, hb_void_t<decltype (T::min_size)>> : hb_integral_constant<unsigned, T::min_size> {};
template <typename T>
using hb_min_size = _hb_min_size<T, void>;
#define hb_min_size(T) hb_min_size<T>::value


/*
 * Null()
 */

extern HB_INTERNAL
uint64_t const _hb_NullPool[(HB_NULL_POOL_SIZE + sizeof (uint64_t) - 1) / sizeof (uint64_t)];

/* Generic nul-content Null objects. */
template <typename Type>
struct Null {
  static Type const & get_null ()
  {
    static_assert (hb_null_size (Type) <= HB_NULL_POOL_SIZE, "Increase HB_NULL_POOL_SIZE.");
    return *reinterpret_cast<Type const *> (_hb_NullPool);
  }
};
template <typename QType>
struct NullHelper
{
  typedef hb_remove_const<hb_remove_reference<QType>> Type;
  static const Type & get_null () { return Null<Type>::get_null (); }
};
#define Null(Type) NullHelper<Type>::get_null ()

/* Specializations for arbitrary-content Null objects expressed in bytes. */
#define DECLARE_NULL_NAMESPACE_BYTES(Namespace, Type) \
	} /* Close namespace. */ \
	extern HB_INTERNAL const unsigned char _hb_Null_##Namespace##_##Type[hb_null_size (Namespace::Type)]; \
	template <> \
	struct Null<Namespace::Type> { \
	  static Namespace::Type const & get_null () { \
	    return *reinterpret_cast<const Namespace::Type *> (_hb_Null_##Namespace##_##Type); \
	  } \
	}; \
	namespace Namespace { \
	static_assert (true, "") /* Require semicolon after. */
#define DECLARE_NULL_NAMESPACE_BYTES_TEMPLATE1(Namespace, Type, Size) \
	} /* Close namespace. */ \
	extern HB_INTERNAL const unsigned char _hb_Null_##Namespace##_##Type[Size]; \
	template <typename Spec> \
	struct Null<Namespace::Type<Spec>> { \
	  static Namespace::Type<Spec> const & get_null () { \
	    return *reinterpret_cast<const Namespace::Type<Spec> *> (_hb_Null_##Namespace##_##Type); \
	  } \
	}; \
	namespace Namespace { \
	static_assert (true, "") /* Require semicolon after. */
#define DEFINE_NULL_NAMESPACE_BYTES(Namespace, Type) \
	const unsigned char _hb_Null_##Namespace##_##Type[sizeof (_hb_Null_##Namespace##_##Type)]

/* Specializations for arbitrary-content Null objects expressed as struct initializer. */
#define DECLARE_NULL_INSTANCE(Type) \
	extern HB_INTERNAL const Type _hb_Null_##Type; \
	template <> \
	struct Null<Type> { \
	  static Type const & get_null () { \
	    return _hb_Null_##Type; \
	  } \
	}; \
	static_assert (true, "") /* Require semicolon after. */
#define DEFINE_NULL_INSTANCE(Type) \
	const Type _hb_Null_##Type

/* Global writable pool.  Enlarge as necessary. */

/* To be fully correct, CrapPool must be thread_local. However, we do not rely on CrapPool
 * for correct operation. It only exist to catch and divert program logic bugs instead of
 * causing bad memory access. So, races there are not actually introducing incorrectness
 * in the code. Has ~12kb binary size overhead to have it, also clang build fails with it. */
extern HB_INTERNAL
/*thread_local*/ uint64_t _hb_CrapPool[(HB_NULL_POOL_SIZE + sizeof (uint64_t) - 1) / sizeof (uint64_t)];

/* CRAP pool: Common Region for Access Protection. */
template <typename Type>
static inline Type& Crap () {
  static_assert (hb_null_size (Type) <= HB_NULL_POOL_SIZE, "Increase HB_NULL_POOL_SIZE.");
  Type *obj = reinterpret_cast<Type *> (_hb_CrapPool);
  memcpy (obj, &Null (Type), sizeof (*obj));
  return *obj;
}
template <typename QType>
struct CrapHelper
{
  typedef hb_remove_const<hb_remove_reference<QType>> Type;
  static Type & get_crap () { return Crap<Type> (); }
};
#define Crap(Type) CrapHelper<Type>::get_crap ()

template <typename Type>
struct CrapOrNullHelper {
  static Type & get () { return Crap (Type); }
};
template <typename Type>
struct CrapOrNullHelper<const Type> {
  static const Type & get () { return Null (Type); }
};
#define CrapOrNull(Type) CrapOrNullHelper<Type>::get ()


/*
 * hb_nonnull_ptr_t
 */

template <typename P>
struct hb_nonnull_ptr_t
{
  typedef hb_remove_pointer<P> T;

  hb_nonnull_ptr_t (T *v_ = nullptr) : v (v_) {}
  T * operator = (T *v_)   { return v = v_; }
  T * operator -> () const { return get (); }
  T & operator * () const  { return *get (); }
  T ** operator & () const { return &v; }
  /* Only auto-cast to const types. */
  template <typename C> operator const C * () const { return get (); }
  operator const char * () const { return (const char *) get (); }
  T * get () const { return v ? v : const_cast<T *> (&Null (T)); }
  T * get_raw () const { return v; }

  private:
  T *v;
};


#endif /* HB_NULL_HH */
