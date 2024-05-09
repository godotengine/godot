/*
 *  Copyright 2008-2018 NVIDIA Corporation
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 */

/*
 * (C) Copyright John Maddock 2000.
 * 
 * Distributed under the Boost Software License, Version 1.0.
 * (See accompanying NOTICE file for the complete license)
 *
 * For more information, see http://www.boost.org
 */

#pragma once

#include <thrust/detail/config.h>
#include <thrust/detail/type_traits.h>
#include <thrust/detail/preprocessor.h>

THRUST_NAMESPACE_BEGIN

namespace detail
{

template <typename, bool x>
struct depend_on_instantiation
{
  THRUST_INLINE_INTEGRAL_MEMBER_CONSTANT bool value = x;
};

#if THRUST_CPP_DIALECT >= 2011

#  if THRUST_CPP_DIALECT >= 2017
#    define THRUST_STATIC_ASSERT(B)        static_assert(B)
#  else
#    define THRUST_STATIC_ASSERT(B)        static_assert(B, "static assertion failed")
#  endif
#  define THRUST_STATIC_ASSERT_MSG(B, msg) static_assert(B, msg)

#else // Older than C++11.

// HP aCC cannot deal with missing names for template value parameters.
template <bool x> struct STATIC_ASSERTION_FAILURE;

template <> struct STATIC_ASSERTION_FAILURE<true> {};

// HP aCC cannot deal with missing names for template value parameters.
template <int x> struct static_assert_test {};

#if    (  (THRUST_HOST_COMPILER == THRUST_HOST_COMPILER_GCC)                  \
       && (THRUST_GCC_VERSION >= 40800))                                      \
    || (THRUST_HOST_COMPILER == THRUST_HOST_COMPILER_CLANG)
  // Clang and GCC 4.8+ will complain about this typedef being unused unless we
  // annotate it as such.
#  define THRUST_STATIC_ASSERT(B)                                             \
    typedef THRUST_NS_QUALIFIER::detail::static_assert_test<                  \
      sizeof(THRUST_NS_QUALIFIER::detail::STATIC_ASSERTION_FAILURE<(bool)(B)>)\
    >                                                                         \
      THRUST_PP_CAT2(thrust_static_assert_typedef_, __LINE__)                 \
      __attribute__((unused))                                                 \
    /**/      
#else
#  define THRUST_STATIC_ASSERT(B)                                             \
    typedef THRUST_NS_QUALIFIER::detail::static_assert_test<                  \
      sizeof(THRUST_NS_QUALIFIER::detail::STATIC_ASSERTION_FAILURE<(bool)(B)>)\
    >                                                                         \
      THRUST_PP_CAT2(thrust_static_assert_typedef_, __LINE__)                 \
    /**/      
#endif

#define THRUST_STATIC_ASSERT_MSG(B, msg) THRUST_STATIC_ASSERT(B)

#endif // THRUST_CPP_DIALECT >= 2011

} // namespace detail

THRUST_NAMESPACE_END


