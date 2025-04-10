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

#ifndef HB_META_HH
#define HB_META_HH

#include "hb.hh"

#include <memory>
#include <type_traits>
#include <utility>


/*
 * C++ template meta-programming & fundamentals used with them.
 */

/* Void!  For when we need a expression-type of void. */
struct hb_empty_t {};

/* https://en.cppreference.com/w/cpp/types/void_t */
template<typename... Ts> struct _hb_void_t { typedef void type; };
template<typename... Ts> using hb_void_t = typename _hb_void_t<Ts...>::type;

template<typename Head, typename... Ts> struct _hb_head_t { typedef Head type; };
template<typename... Ts> using hb_head_t = typename _hb_head_t<Ts...>::type;

template <typename T, T v> struct hb_integral_constant { static constexpr T value = v; };
template <bool b> using hb_bool_constant = hb_integral_constant<bool, b>;
using hb_true_type = hb_bool_constant<true>;
using hb_false_type = hb_bool_constant<false>;

/* Static-assert as expression. */
template <bool cond> struct static_assert_expr;
template <> struct static_assert_expr<true> : hb_false_type {};
#define static_assert_expr(C) static_assert_expr<C>::value

/* Basic type SFINAE. */

template <bool B, typename T = void> struct hb_enable_if {};
template <typename T>                struct hb_enable_if<true, T> { typedef T type; };
#define hb_enable_if(Cond) typename hb_enable_if<(Cond)>::type* = nullptr
/* Concepts/Requires alias: */
#define hb_requires(Cond) hb_enable_if((Cond))

template <typename T, typename T2> struct hb_is_same : hb_false_type {};
template <typename T>              struct hb_is_same<T, T> : hb_true_type {};
#define hb_is_same(T, T2) hb_is_same<T, T2>::value

/* Function overloading SFINAE and priority. */

#define HB_RETURN(Ret, E) -> hb_head_t<Ret, decltype ((E))> { return (E); }
#define HB_AUTO_RETURN(E) -> decltype ((E)) { return (E); }
#define HB_VOID_RETURN(E) -> hb_void_t<decltype ((E))> { (E); }

template <unsigned Pri> struct hb_priority : hb_priority<Pri - 1> {};
template <>             struct hb_priority<0> {};
#define hb_prioritize hb_priority<16> ()

#define HB_FUNCOBJ(x) static_const x HB_UNUSED


template <typename T> struct hb_type_identity_t { typedef T type; };
template <typename T> using hb_type_identity = typename hb_type_identity_t<T>::type;

template <typename T> static inline T hb_declval ();
#define hb_declval(T) (hb_declval<T> ())

template <typename T> struct hb_match_const		: hb_type_identity_t<T>, hb_false_type	{};
template <typename T> struct hb_match_const<const T>	: hb_type_identity_t<T>, hb_true_type	{};
template <typename T> using hb_remove_const = typename hb_match_const<T>::type;

template <typename T> struct hb_match_reference		: hb_type_identity_t<T>, hb_false_type	{};
template <typename T> struct hb_match_reference<T &>	: hb_type_identity_t<T>, hb_true_type	{};
template <typename T> struct hb_match_reference<T &&>	: hb_type_identity_t<T>, hb_true_type	{};
template <typename T> using hb_remove_reference = typename hb_match_reference<T>::type;
template <typename T> auto _hb_try_add_lvalue_reference (hb_priority<1>) -> hb_type_identity<T&>;
template <typename T> auto _hb_try_add_lvalue_reference (hb_priority<0>) -> hb_type_identity<T>;
template <typename T> using hb_add_lvalue_reference = decltype (_hb_try_add_lvalue_reference<T> (hb_prioritize));
template <typename T> auto _hb_try_add_rvalue_reference (hb_priority<1>) -> hb_type_identity<T&&>;
template <typename T> auto _hb_try_add_rvalue_reference (hb_priority<0>) -> hb_type_identity<T>;
template <typename T> using hb_add_rvalue_reference = decltype (_hb_try_add_rvalue_reference<T> (hb_prioritize));

template <typename T> struct hb_match_pointer		: hb_type_identity_t<T>, hb_false_type	{};
template <typename T> struct hb_match_pointer<T *>	: hb_type_identity_t<T>, hb_true_type	{};
template <typename T> using hb_remove_pointer = typename hb_match_pointer<T>::type;
template <typename T> auto _hb_try_add_pointer (hb_priority<1>) -> hb_type_identity<hb_remove_reference<T>*>;
template <typename T> auto _hb_try_add_pointer (hb_priority<1>) -> hb_type_identity<T>;
template <typename T> using hb_add_pointer = decltype (_hb_try_add_pointer<T> (hb_prioritize));


template <typename T> using hb_decay = typename std::decay<T>::type;

#define hb_is_convertible(From,To) std::is_convertible<From, To>::value

template <typename From, typename To>
using hb_is_cr_convertible = hb_bool_constant<
  hb_is_same (hb_decay<From>, hb_decay<To>) &&
  (!std::is_const<From>::value || std::is_const<To>::value) &&
  (!std::is_reference<To>::value || std::is_const<To>::value || std::is_reference<To>::value)
>;
#define hb_is_cr_convertible(From,To) hb_is_cr_convertible<From, To>::value


struct
{
  template <typename T> constexpr auto
  operator () (T&& v) const HB_AUTO_RETURN (std::forward<T> (v))

  template <typename T> constexpr auto
  operator () (T *v) const HB_AUTO_RETURN (*v)

  template <typename T> constexpr auto
  operator () (const hb::shared_ptr<T>& v) const HB_AUTO_RETURN (*v)

  template <typename T> constexpr auto
  operator () (hb::shared_ptr<T>& v) const HB_AUTO_RETURN (*v)
  
  template <typename T> constexpr auto
  operator () (const hb::unique_ptr<T>& v) const HB_AUTO_RETURN (*v)

  template <typename T> constexpr auto
  operator () (hb::unique_ptr<T>& v) const HB_AUTO_RETURN (*v)
}
HB_FUNCOBJ (hb_deref);

template <typename T>
struct hb_reference_wrapper
{
  hb_reference_wrapper (T v) : v (v) {}
  bool operator == (const hb_reference_wrapper& o) const { return v == o.v; }
  bool operator != (const hb_reference_wrapper& o) const { return v != o.v; }
  operator T& () { return v; }
  T& get () { return v; }
  T v;
};
template <typename T>
struct hb_reference_wrapper<T&>
{
  hb_reference_wrapper (T& v) : v (std::addressof (v)) {}
  bool operator == (const hb_reference_wrapper& o) const { return v == o.v; }
  bool operator != (const hb_reference_wrapper& o) const { return v != o.v; }
  operator T& () { return *v; }
  T& get () { return *v; }
  T* v;
};


/* Type traits */

template <typename T> struct hb_int_min;
template <> struct hb_int_min<char>			: hb_integral_constant<char,			CHAR_MIN>	{};
template <> struct hb_int_min<signed char>		: hb_integral_constant<signed char,		SCHAR_MIN>	{};
template <> struct hb_int_min<unsigned char>		: hb_integral_constant<unsigned char,		0>		{};
template <> struct hb_int_min<signed short>		: hb_integral_constant<signed short,		SHRT_MIN>	{};
template <> struct hb_int_min<unsigned short>		: hb_integral_constant<unsigned short,		0>		{};
template <> struct hb_int_min<signed int>		: hb_integral_constant<signed int,		INT_MIN>	{};
template <> struct hb_int_min<unsigned int>		: hb_integral_constant<unsigned int,		0>		{};
template <> struct hb_int_min<signed long>		: hb_integral_constant<signed long,		LONG_MIN>	{};
template <> struct hb_int_min<unsigned long>		: hb_integral_constant<unsigned long,		0>		{};
template <> struct hb_int_min<signed long long>		: hb_integral_constant<signed long long,	LLONG_MIN>	{};
template <> struct hb_int_min<unsigned long long>	: hb_integral_constant<unsigned long long,	0>		{};
template <typename T> struct hb_int_min<T *>		: hb_integral_constant<T *,			nullptr>	{};
#define hb_int_min(T) hb_int_min<T>::value
template <typename T> struct hb_int_max;
template <> struct hb_int_max<char>			: hb_integral_constant<char,			CHAR_MAX>	{};
template <> struct hb_int_max<signed char>		: hb_integral_constant<signed char,		SCHAR_MAX>	{};
template <> struct hb_int_max<unsigned char>		: hb_integral_constant<unsigned char,		UCHAR_MAX>	{};
template <> struct hb_int_max<signed short>		: hb_integral_constant<signed short,		SHRT_MAX>	{};
template <> struct hb_int_max<unsigned short>		: hb_integral_constant<unsigned short,		USHRT_MAX>	{};
template <> struct hb_int_max<signed int>		: hb_integral_constant<signed int,		INT_MAX>	{};
template <> struct hb_int_max<unsigned int>		: hb_integral_constant<unsigned int,		UINT_MAX>	{};
template <> struct hb_int_max<signed long>		: hb_integral_constant<signed long,		LONG_MAX>	{};
template <> struct hb_int_max<unsigned long>		: hb_integral_constant<unsigned long,		ULONG_MAX>	{};
template <> struct hb_int_max<signed long long>		: hb_integral_constant<signed long long,	LLONG_MAX>	{};
template <> struct hb_int_max<unsigned long long>	: hb_integral_constant<unsigned long long,	ULLONG_MAX>	{};
#define hb_int_max(T) hb_int_max<T>::value

#if defined(__GNUC__) && __GNUC__ < 5 && !defined(__clang__)
#define hb_is_trivially_copyable(T) __has_trivial_copy(T)
#define hb_is_trivially_copy_assignable(T) __has_trivial_assign(T)
#define hb_is_trivially_constructible(T) __has_trivial_constructor(T)
#define hb_is_trivially_copy_constructible(T) __has_trivial_copy_constructor(T)
#define hb_is_trivially_destructible(T) __has_trivial_destructor(T)
#else
#define hb_is_trivially_copyable(T) std::is_trivially_copyable<T>::value
#define hb_is_trivially_copy_assignable(T) std::is_trivially_copy_assignable<T>::value
#define hb_is_trivially_constructible(T) std::is_trivially_constructible<T>::value
#define hb_is_trivially_copy_constructible(T) std::is_trivially_copy_constructible<T>::value
#define hb_is_trivially_destructible(T) std::is_trivially_destructible<T>::value
#endif

/* Class traits. */

#define HB_DELETE_COPY_ASSIGN(TypeName) \
  TypeName(const TypeName&) = delete; \
  void operator=(const TypeName&) = delete
#define HB_DELETE_CREATE_COPY_ASSIGN(TypeName) \
  TypeName() = delete; \
  TypeName(const TypeName&) = delete; \
  void operator=(const TypeName&) = delete

/* hb_unwrap_type (T)
 * If T has no T::type, returns T. Otherwise calls itself on T::type recursively.
 */

template <typename T, typename>
struct _hb_unwrap_type : hb_type_identity_t<T> {};
template <typename T>
struct _hb_unwrap_type<T, hb_void_t<typename T::type>> : _hb_unwrap_type<typename T::type, void> {};
template <typename T>
using hb_unwrap_type = _hb_unwrap_type<T, void>;
#define hb_unwrap_type(T) typename hb_unwrap_type<T>::type

#endif /* HB_META_HH */
