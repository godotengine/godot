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
#include <thrust/detail/type_traits/is_metafunction_defined.h>
#include <thrust/detail/type_traits/minimum_type.h>

THRUST_NAMESPACE_BEGIN
namespace detail
{ 


template<typename T1,
         typename T2  = void,
         typename T3  = void,
         typename T4  = void,
         typename T5  = void,
         typename T6  = void,
         typename T7  = void,
         typename T8  = void,
         typename T9  = void,
         typename T10 = void,
         typename T11 = void,
         typename T12 = void,
         typename T13 = void,
         typename T14 = void,
         typename T15 = void,
         typename T16 = void>
  struct unrelated_systems {};


// if a minimum system exists for these arguments, return it
// otherwise, collect the arguments and report them as unrelated
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
  struct minimum_system
    : thrust::detail::eval_if<
        is_metafunction_defined<
          minimum_type<T1,T2,T3,T4,T5,T6,T7,T8,T9,T10,T11,T12,T13,T14,T15,T16>
        >::value,
        minimum_type<T1,T2,T3,T4,T5,T6,T7,T8,T9,T10,T11,T12,T13,T14,T15,T16>,
        thrust::detail::identity_<
          unrelated_systems<T1,T2,T3,T4,T5,T6,T7,T8,T9,T10,T11,T12,T13,T14,T15,T16>
        >
      >
{}; // end minimum_system


} // end detail
THRUST_NAMESPACE_END

