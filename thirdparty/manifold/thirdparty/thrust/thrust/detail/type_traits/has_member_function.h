/*
 *  Copyright 2008-2021 NVIDIA Corporation
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

#include <thrust/detail/type_traits.h>

#include <utility> // for std::declval

#define __THRUST_DEFINE_HAS_MEMBER_FUNCTION(trait_name, member_function_name)  \
  template <typename T, typename Signature, typename = void>                   \
  struct trait_name : thrust::false_type                                       \
  {};                                                                          \
                                                                               \
  template <typename T, typename ResultT, typename... Args>                    \
  struct trait_name<T,                                                         \
                    ResultT(Args...),                                          \
                    typename thrust::detail::enable_if<                        \
                      thrust::detail::is_same<ResultT, void>::value ||         \
                      thrust::detail::is_convertible<                          \
                        ResultT,                                               \
                        decltype(std::declval<T>().member_function_name(       \
                          std::declval<Args>()...))>::value>::type>            \
      : thrust::true_type                                                      \
  {};
