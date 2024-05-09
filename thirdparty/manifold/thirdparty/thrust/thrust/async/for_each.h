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

/*! \file
 *  \brief Algorithms for asynchronously iterating over the elements of a range.
 */

#pragma once

#include <thrust/detail/config.h>
#include <thrust/detail/cpp14_required.h>

#if THRUST_CPP_DIALECT >= 2014

#include <thrust/detail/static_assert.h>
#include <thrust/detail/select_system.h>
#include <thrust/type_traits/remove_cvref.h>
#include <thrust/system/detail/adl/async/for_each.h>

#include <thrust/event.h>

THRUST_NAMESPACE_BEGIN

namespace async
{

/*! \cond
 */

namespace unimplemented
{

template <
  typename DerivedPolicy
, typename ForwardIt, typename Sentinel, typename UnaryFunction
>
__host__
event<DerivedPolicy>
async_for_each(
  thrust::execution_policy<DerivedPolicy>&, ForwardIt, Sentinel, UnaryFunction
)
{
  THRUST_STATIC_ASSERT_MSG(
    (thrust::detail::depend_on_instantiation<ForwardIt, false>::value)
  , "this algorithm is not implemented for the specified system"
  );
  return {};
}

} // namespace unimplemented

namespace for_each_detail
{

using thrust::async::unimplemented::async_for_each;

struct for_each_fn final
{
  template <
    typename DerivedPolicy
  , typename ForwardIt, typename Sentinel, typename UnaryFunction
  >
  __host__
  static auto call(
    thrust::detail::execution_policy_base<DerivedPolicy> const& exec
  , ForwardIt&& first, Sentinel&& last
  , UnaryFunction&& f
  )
  // ADL dispatch.
  THRUST_RETURNS(
    async_for_each(
      thrust::detail::derived_cast(thrust::detail::strip_const(exec))
    , THRUST_FWD(first), THRUST_FWD(last)
    , THRUST_FWD(f)
    )
  )

  template <typename ForwardIt, typename Sentinel, typename UnaryFunction>
  __host__
  static auto call(ForwardIt&& first, Sentinel&& last, UnaryFunction&& f)
  THRUST_RETURNS(
    for_each_fn::call(
      thrust::detail::select_system(
        typename iterator_system<remove_cvref_t<ForwardIt>>::type{}
      )
    , THRUST_FWD(first), THRUST_FWD(last)
    , THRUST_FWD(f)
    )
  )

  template <typename... Args>
  THRUST_NODISCARD __host__
  auto operator()(Args&&... args) const
  THRUST_RETURNS(
    call(THRUST_FWD(args)...)
  )
};

} // namespace for_each_detail

THRUST_INLINE_CONSTANT for_each_detail::for_each_fn for_each{};

/*! \endcond
 */

} // namespace async

THRUST_NAMESPACE_END

#endif
