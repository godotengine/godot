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
 *  \brief Algorithms for asynchronously copying a range.
 */

#pragma once

#include <thrust/detail/config.h>
#include <thrust/detail/cpp14_required.h>

#if THRUST_CPP_DIALECT >= 2014

#include <thrust/detail/static_assert.h>
#include <thrust/detail/select_system.h>
#include <thrust/type_traits/remove_cvref.h>
#include <thrust/system/detail/adl/async/copy.h>

#include <thrust/event.h>

THRUST_NAMESPACE_BEGIN

namespace async
{

/*! \cond
 */

namespace unimplemented
{

template <
  typename FromPolicy, typename ToPolicy
, typename ForwardIt, typename Sentinel, typename OutputIt
>
__host__
event<FromPolicy>
async_copy(
  thrust::execution_policy<FromPolicy>& from_exec
, thrust::execution_policy<ToPolicy>&   to_exec
, ForwardIt first, Sentinel last, OutputIt output
)
{
  THRUST_STATIC_ASSERT_MSG(
    (thrust::detail::depend_on_instantiation<ForwardIt, false>::value)
  , "this algorithm is not implemented for the specified system"
  );
  return {};
}

} // namespace unimplemented

namespace copy_detail
{

using thrust::async::unimplemented::async_copy;

struct copy_fn final
{
  template <
    typename FromPolicy, typename ToPolicy
  , typename ForwardIt, typename Sentinel, typename OutputIt
  >
  __host__
  static auto call(
    thrust::detail::execution_policy_base<FromPolicy> const& from_exec
  , thrust::detail::execution_policy_base<ToPolicy> const&   to_exec
  , ForwardIt&& first, Sentinel&& last
  , OutputIt&& output
  )
  // ADL dispatch.
  THRUST_RETURNS(
    async_copy(
      thrust::detail::derived_cast(thrust::detail::strip_const(from_exec))
    , thrust::detail::derived_cast(thrust::detail::strip_const(to_exec))
    , THRUST_FWD(first), THRUST_FWD(last)
    , THRUST_FWD(output)
    )
  )

  template <
    typename DerivedPolicy
  , typename ForwardIt, typename Sentinel, typename OutputIt
  >
  __host__
  static auto call(
    thrust::detail::execution_policy_base<DerivedPolicy> const& exec
  , ForwardIt&& first, Sentinel&& last
  , OutputIt&& output
  )
  THRUST_RETURNS(
    copy_fn::call(
      thrust::detail::derived_cast(thrust::detail::strip_const(exec))
      // Synthesize a suitable new execution policy, because we don't want to
      // try and extract twice from the one we were passed.
    , typename remove_cvref_t<
        decltype(thrust::detail::derived_cast(thrust::detail::strip_const(exec)))
      >::tag_type{}
    , THRUST_FWD(first), THRUST_FWD(last)
    , THRUST_FWD(output)
    )
  )

  template <typename ForwardIt, typename Sentinel, typename OutputIt>
  __host__
  static auto call(ForwardIt&& first, Sentinel&& last, OutputIt&& output)
  THRUST_RETURNS(
    copy_fn::call(
      thrust::detail::select_system(
        typename thrust::iterator_system<remove_cvref_t<ForwardIt>>::type{}
      )
    , thrust::detail::select_system(
        typename thrust::iterator_system<remove_cvref_t<OutputIt>>::type{}
      )
    , THRUST_FWD(first), THRUST_FWD(last)
    , THRUST_FWD(output)
    )
  )

  template <typename... Args>
  THRUST_NODISCARD __host__
  auto operator()(Args&&... args) const
  THRUST_RETURNS(
    call(THRUST_FWD(args)...)
  )
};

} // namespace copy_detail

THRUST_INLINE_CONSTANT copy_detail::copy_fn copy{};

/*! \endcond
 */

} // namespace async

THRUST_NAMESPACE_END

#endif

