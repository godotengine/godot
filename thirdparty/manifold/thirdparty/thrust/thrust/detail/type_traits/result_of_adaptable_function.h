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
#include <thrust/detail/type_traits/function_traits.h>

#include <type_traits>

THRUST_NAMESPACE_BEGIN
namespace detail
{

// Sets `type` to the result of the specified Signature invocation. If the
// callable defines a `result_type` alias member, that type is used instead.
// Use invoke_result / result_of when FuncType::result_type is not defined.
template <typename Signature, typename Enable = void>
struct result_of_adaptable_function
{
private:
  template <typename Sig> struct impl;

  template <typename F, typename... Args>
  struct impl<F(Args...)>
  {
    using type = invoke_result_t<F, Args...>;
  };

public:
  using type = typename impl<Signature>::type;
};

// specialization for invocations which define result_type
template <typename Functor, typename... ArgTypes>
struct result_of_adaptable_function<
  Functor(ArgTypes...),
  typename thrust::detail::enable_if<
    thrust::detail::has_result_type<Functor>::value>::type>
{
  using type = typename Functor::result_type;
};

} // namespace detail
THRUST_NAMESPACE_END
