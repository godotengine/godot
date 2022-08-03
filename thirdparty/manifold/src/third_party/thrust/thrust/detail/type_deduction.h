// Copyright (c)      2018 NVIDIA Corporation
//                         (Bryce Adelstein Lelbach <brycelelbach@gmail.com>)
// Copyright (c) 2013-2018 Eric Niebler (`THRUST_RETURNS`, etc)
// Copyright (c) 2016-2018 Casey Carter (`THRUST_RETURNS`, etc)
//
// Distributed under the Boost Software License v1.0 (boost.org/LICENSE_1_0.txt)

#pragma once

#include <thrust/detail/config.h>
#include <thrust/detail/cpp11_required.h>

#if THRUST_CPP_DIALECT >= 2011

#include <thrust/detail/preprocessor.h>

#include <utility>
#include <type_traits>

///////////////////////////////////////////////////////////////////////////////

/// \def THRUST_FWD(x)
/// \brief Performs universal forwarding of a universal reference.
///
#define THRUST_FWD(x) ::std::forward<decltype(x)>(x)

/// \def THRUST_MVCAP(x)
/// \brief Capture `x` into a lambda by moving.
///
#define THRUST_MVCAP(x) x = ::std::move(x)

/// \def THRUST_RETOF(invocable, ...)
/// \brief Expands to the type returned by invoking an instance of the invocable
///        type \a invocable with parameters of type \c __VA_ARGS__. Must
///        be called with 1 or fewer parameters to the invocable.
///
#define THRUST_RETOF(...)   THRUST_PP_DISPATCH(THRUST_RETOF, __VA_ARGS__)
#define THRUST_RETOF1(C)    decltype(::std::declval<C>()())
#define THRUST_RETOF2(C, V) decltype(::std::declval<C>()(::std::declval<V>()))

/// \def THRUST_RETURNS(...)
/// \brief Expands to a function definition that returns the expression
///        \c __VA_ARGS__.
///
#define THRUST_RETURNS(...)                                                   \
  noexcept(noexcept(__VA_ARGS__))                                             \
  { return (__VA_ARGS__); }                                                   \
  /**/

/// \def THRUST_DECLTYPE_RETURNS(...)
/// \brief Expands to a function definition, including a trailing returning
///        type, that returns the expression \c __VA_ARGS__.
///
// Trailing return types seem to confuse Doxygen, and cause it to interpret
// parts of the function's body as new function signatures.
#if defined(THRUST_DOXYGEN)
  #define THRUST_DECLTYPE_RETURNS(...)                                        \
  { return (__VA_ARGS__); }                                                   \
  /**/
#else
  #define THRUST_DECLTYPE_RETURNS(...)                                        \
    noexcept(noexcept(__VA_ARGS__))                                           \
    -> decltype(__VA_ARGS__)                                                  \
    { return (__VA_ARGS__); }                                                 \
    /**/
#endif

/// \def THRUST_DECLTYPE_RETURNS_WITH_SFINAE_CONDITION(condition, ...)
/// \brief Expands to a function definition, including a trailing returning
///        type, that returns the expression \c __VA_ARGS__. It shall only
///        participate in overload resolution if \c condition is \c true.
///
// Trailing return types seem to confuse Doxygen, and cause it to interpret
// parts of the function's body as new function signatures.
#if defined(THRUST_DOXYGEN)
  #define THRUST_DECLTYPE_RETURNS(...)                                        \
  { return (__VA_ARGS__); }                                                   \
  /**/
#else
  #define THRUST_DECLTYPE_RETURNS_WITH_SFINAE_CONDITION(condition, ...)       \
    noexcept(noexcept(__VA_ARGS__))                                           \
    -> typename std::enable_if<condition, decltype(__VA_ARGS__)>::type        \
    { return (__VA_ARGS__); }                                                 \
    /**/
#endif

///////////////////////////////////////////////////////////////////////////////

#endif // THRUST_CPP_DIALECT >= 2011

