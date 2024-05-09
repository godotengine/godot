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
#include <thrust/detail/functional/argument.h>
#include <thrust/detail/type_deduction.h>
#include <thrust/tuple.h>
#include <thrust/detail/type_traits.h>
#include <thrust/type_traits/void_t.h>

#include <type_traits>

THRUST_NAMESPACE_BEGIN
namespace detail
{
namespace functional
{

// Adapts a transparent unary functor from functional.h (e.g. thrust::negate<>)
// into the Eval interface.
template <typename UnaryFunctor>
struct transparent_unary_operator
{
  template <typename>
  using operator_type = UnaryFunctor;

  template <typename Env>
  using argument =
  typename thrust::detail::eval_if<
    thrust::tuple_size<Env>::value != 1,
    thrust::detail::identity_<thrust::null_type>,
    thrust::detail::functional::argument_helper<0, Env>
  >::type;

  template <typename Env>
  struct result_type_impl
  {
    using type = decltype(
      std::declval<UnaryFunctor>()(std::declval<argument<Env>>()));
  };

  template <typename Env>
  using result_type =
  typename thrust::detail::eval_if<
    std::is_same<thrust::null_type, argument<Env>>::value,
    thrust::detail::identity_<thrust::null_type>,
    result_type_impl<Env>
  >::type;

  template <typename Env>
  struct result
  {
    using op_type = UnaryFunctor;
    using type = result_type<Env>;
  };

  template <typename Env>
  __host__ __device__
  result_type<Env> eval(Env&& e) const
  THRUST_RETURNS(UnaryFunctor{}(thrust::get<0>(THRUST_FWD(e))))
};


// Adapts a transparent binary functor from functional.h (e.g. thrust::less<>)
// into the Eval interface.
template <typename BinaryFunctor>
struct transparent_binary_operator
{
  template <typename>
  using operator_type = BinaryFunctor;

  template <typename Env>
  using first_argument =
    typename thrust::detail::eval_if<
      thrust::tuple_size<Env>::value != 2,
      thrust::detail::identity_<thrust::null_type>,
      thrust::detail::functional::argument_helper<0, Env>
    >::type;

  template <typename Env>
  using second_argument =
    typename thrust::detail::eval_if<
      thrust::tuple_size<Env>::value != 2,
      thrust::detail::identity_<thrust::null_type>,
      thrust::detail::functional::argument_helper<1, Env>
    >::type;

  template <typename Env>
  struct result_type_impl
  {
    using type = decltype(
      std::declval<BinaryFunctor>()(std::declval<first_argument<Env>>(),
                                    std::declval<second_argument<Env>>()));
  };

  template <typename Env>
  using result_type =
    typename thrust::detail::eval_if<
      (std::is_same<thrust::null_type, first_argument<Env>>::value ||
       std::is_same<thrust::null_type, second_argument<Env>>::value),
      thrust::detail::identity_<thrust::null_type>,
      result_type_impl<Env>
    >::type;

  template <typename Env>
  struct result
  {
    using op_type = BinaryFunctor;
    using type = result_type<Env>;
  };

  template <typename Env>
  __host__ __device__
  result_type<Env> eval(Env&& e) const
  THRUST_RETURNS(BinaryFunctor{}(thrust::get<0>(e), thrust::get<1>(e)))
};

} // end functional
} // end detail
THRUST_NAMESPACE_END

