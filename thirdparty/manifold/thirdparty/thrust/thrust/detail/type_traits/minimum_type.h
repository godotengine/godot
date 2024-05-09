/*
 *  Copyright 2008-2013 NVIDIA Corporation
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

#pragma once

#include <thrust/detail/config.h>

#include <thrust/detail/type_traits.h>

THRUST_NAMESPACE_BEGIN

namespace detail
{ 

namespace minimum_type_detail
{

//
// Returns the minimum type or is empty
// if T1 and T2 are unrelated.
//
template <typename T1, typename T2, bool GreaterEqual, bool LessEqual> struct minimum_type_impl {};
  
template <typename T1, typename T2>
struct minimum_type_impl<T1,T2,true,false>
{
  typedef T2 type;
}; // end minimum_type_impl

template <typename T1, typename T2>
struct minimum_type_impl<T1,T2,false,true>
{
  typedef T1 type;
}; // end minimum_type_impl

template <typename T1, typename T2>
struct minimum_type_impl<T1,T2,true,true>
{
  typedef T1 type;
}; // end minimum_type_impl

template <typename T1, typename T2>
struct primitive_minimum_type
  : minimum_type_detail::minimum_type_impl<
      T1,
      T2,
      THRUST_NS_QUALIFIER::detail::is_convertible<T1,T2>::value,
      THRUST_NS_QUALIFIER::detail::is_convertible<T2,T1>::value
    >
{
}; // end primitive_minimum_type

// because some types are not convertible (even to themselves)
// specialize primitive_minimum_type for when both types are identical
template <typename T>
struct primitive_minimum_type<T,T>
{
  typedef T type;
}; // end primitive_minimum_type

// XXX this belongs somewhere more general
struct any_conversion
{
  template<typename T> operator T (void);
};

} // end minimum_type_detail

template<typename T1,
         typename T2  = minimum_type_detail::any_conversion,
         typename T3  = minimum_type_detail::any_conversion,
         typename T4  = minimum_type_detail::any_conversion,
         typename T5  = minimum_type_detail::any_conversion,
         typename T6  = minimum_type_detail::any_conversion,
         typename T7  = minimum_type_detail::any_conversion,
         typename T8  = minimum_type_detail::any_conversion,
         typename T9  = minimum_type_detail::any_conversion,
         typename T10 = minimum_type_detail::any_conversion,
         typename T11 = minimum_type_detail::any_conversion,
         typename T12 = minimum_type_detail::any_conversion,
         typename T13 = minimum_type_detail::any_conversion,
         typename T14 = minimum_type_detail::any_conversion,
         typename T15 = minimum_type_detail::any_conversion,
         typename T16 = minimum_type_detail::any_conversion>
  struct minimum_type;

// base case
template<typename T1, typename T2>
  struct minimum_type<T1,T2>
    : minimum_type_detail::primitive_minimum_type<T1,T2>
{};

template<typename T1, typename T2>
  struct lazy_minimum_type
    : minimum_type<
        typename T1::type,
        typename T2::type
      >
{};

// carefully avoid referring to a nested ::type which may not exist
template<typename T1,  typename T2,  typename T3,  typename T4,
         typename T5,  typename T6,  typename T7,  typename T8,
         typename T9,  typename T10, typename T11, typename T12,
         typename T13, typename T14, typename T15, typename T16>
  struct minimum_type
    : lazy_minimum_type<
        lazy_minimum_type<
          lazy_minimum_type<
            minimum_type<
              T1,T2
            >,
            minimum_type<
              T3,T4
            >
          >,
          lazy_minimum_type<
            minimum_type<
              T5,T6
            >,
            minimum_type<
              T7,T8
            >
          >
        >,
        lazy_minimum_type<
          lazy_minimum_type<
            minimum_type<
              T9,T10
            >,
            minimum_type<
              T11,T12
            >
          >,
          lazy_minimum_type<
            minimum_type<
              T13,T14
            >,
            minimum_type<
              T15,T16
            >
          >
        >
      >
{};

} // end detail

THRUST_NAMESPACE_END

