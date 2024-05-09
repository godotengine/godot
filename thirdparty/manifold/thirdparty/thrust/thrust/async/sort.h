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
 *  \brief Algorithms for asynchronously sorting a range.
 */

#pragma once

#include <thrust/detail/config.h>
#include <thrust/detail/cpp14_required.h>

#if THRUST_CPP_DIALECT >= 2014

#include <thrust/detail/static_assert.h>
#include <thrust/detail/select_system.h>
#include <thrust/type_traits/logical_metafunctions.h>
#include <thrust/type_traits/remove_cvref.h>
#include <thrust/type_traits/is_execution_policy.h>
#include <thrust/system/detail/adl/async/sort.h>

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
, typename ForwardIt, typename Sentinel, typename StrictWeakOrdering
>
__host__
event<DerivedPolicy>
async_stable_sort(
  thrust::execution_policy<DerivedPolicy>&
, ForwardIt, Sentinel, StrictWeakOrdering
)
{
  THRUST_STATIC_ASSERT_MSG(
    (thrust::detail::depend_on_instantiation<ForwardIt, false>::value)
  , "this algorithm is not implemented for the specified system"
  );
  return {};
}

} // namespace unimplemented

namespace stable_sort_detail
{

using thrust::async::unimplemented::async_stable_sort;

struct stable_sort_fn final
{
  template <
    typename DerivedPolicy
  , typename ForwardIt, typename Sentinel, typename StrictWeakOrdering
  >
  __host__
  static auto call(
    thrust::detail::execution_policy_base<DerivedPolicy> const& exec
  , ForwardIt&& first, Sentinel&& last
  , StrictWeakOrdering&& comp
  )
  // ADL dispatch.
  THRUST_RETURNS(
    async_stable_sort(
      thrust::detail::derived_cast(thrust::detail::strip_const(exec))
    , THRUST_FWD(first), THRUST_FWD(last)
    , THRUST_FWD(comp)
    )
  )

  template <
    typename DerivedPolicy
  , typename ForwardIt, typename Sentinel
  >
  __host__
  static auto call(
    thrust::detail::execution_policy_base<DerivedPolicy> const& exec
  , ForwardIt&& first, Sentinel&& last
  )
  // ADL dispatch.
  THRUST_RETURNS(
    async_stable_sort(
      thrust::detail::derived_cast(thrust::detail::strip_const(exec))
    , THRUST_FWD(first), THRUST_FWD(last)
    , thrust::less<
        typename iterator_traits<remove_cvref_t<ForwardIt>>::value_type
      >{}
    )
  )

  template <typename ForwardIt, typename Sentinel, typename StrictWeakOrdering>
  __host__
  static auto call(ForwardIt&& first, Sentinel&& last, StrictWeakOrdering&& comp)
  THRUST_RETURNS(
    stable_sort_fn::call(
      thrust::detail::select_system(
        typename iterator_system<remove_cvref_t<ForwardIt>>::type{}
      )
    , THRUST_FWD(first), THRUST_FWD(last)
    , THRUST_FWD(comp)
    )
  )

  template <typename ForwardIt, typename Sentinel>
  __host__
  static auto call(ForwardIt&& first, Sentinel&& last)
  THRUST_RETURNS(
    stable_sort_fn::call(
      THRUST_FWD(first), THRUST_FWD(last)
    , thrust::less<
        typename iterator_traits<remove_cvref_t<ForwardIt>>::value_type
      >{}
    )
  )

  template <typename... Args>
  THRUST_NODISCARD __host__
  auto operator()(Args&&... args) const
  THRUST_RETURNS(
    call(THRUST_FWD(args)...)
  )
};

} // namespace stable_sort_detail

THRUST_INLINE_CONSTANT stable_sort_detail::stable_sort_fn stable_sort{};

namespace fallback
{

template <
  typename DerivedPolicy
, typename ForwardIt, typename Sentinel, typename StrictWeakOrdering
>
__host__
event<DerivedPolicy>
async_sort(
  thrust::execution_policy<DerivedPolicy>& exec
, ForwardIt&& first, Sentinel&& last, StrictWeakOrdering&& comp
)
{
  return async_stable_sort(
    thrust::detail::derived_cast(exec)
  , THRUST_FWD(first), THRUST_FWD(last), THRUST_FWD(comp)
  );
}

} // namespace fallback

namespace sort_detail
{

using thrust::async::fallback::async_sort;

struct sort_fn final
{
  template <
    typename DerivedPolicy
  , typename ForwardIt, typename Sentinel, typename StrictWeakOrdering
  >
  __host__
  static auto call(
    thrust::detail::execution_policy_base<DerivedPolicy> const& exec
  , ForwardIt&& first, Sentinel&& last
  , StrictWeakOrdering&& comp
  )
  // ADL dispatch.
  THRUST_RETURNS(
    async_sort(
      thrust::detail::derived_cast(thrust::detail::strip_const(exec))
    , THRUST_FWD(first), THRUST_FWD(last)
    , THRUST_FWD(comp)
    )
  )

  template <
    typename DerivedPolicy
  , typename ForwardIt, typename Sentinel
  >
  __host__
  static auto call3(
    thrust::detail::execution_policy_base<DerivedPolicy> const& exec
  , ForwardIt&& first, Sentinel&& last
  , thrust::true_type
  )
  THRUST_RETURNS(
    sort_fn::call(
      exec
    , THRUST_FWD(first), THRUST_FWD(last)
    , thrust::less<
        typename iterator_traits<remove_cvref_t<ForwardIt>>::value_type
      >{}
    )
  )

  template <typename ForwardIt, typename Sentinel, typename StrictWeakOrdering>
  __host__
  static auto call3(ForwardIt&& first, Sentinel&& last,
                    StrictWeakOrdering&& comp,
                    thrust::false_type)
  THRUST_RETURNS(
    sort_fn::call(
      thrust::detail::select_system(
        typename iterator_system<remove_cvref_t<ForwardIt>>::type{}
      )
    , THRUST_FWD(first), THRUST_FWD(last)
    , THRUST_FWD(comp)
    )
  )

  // MSVC WAR: MSVC gets angsty and eats all available RAM when we try to detect
  // if T1 is an execution_policy by using SFINAE. Switching to a static
  // dispatch pattern to prevent this.
  template <typename T1, typename T2, typename T3>
  __host__
  static auto call(T1&& t1, T2&& t2, T3&& t3)
  THRUST_RETURNS(
    sort_fn::call3(THRUST_FWD(t1), THRUST_FWD(t2), THRUST_FWD(t3),
                   thrust::is_execution_policy<thrust::remove_cvref_t<T1>>{})
  )

  template <typename ForwardIt, typename Sentinel>
  __host__
  static auto call(ForwardIt&& first, Sentinel&& last)
  THRUST_RETURNS(
    sort_fn::call(
      thrust::detail::select_system(
        typename iterator_system<remove_cvref_t<ForwardIt>>::type{}
      )
    , THRUST_FWD(first), THRUST_FWD(last)
    , thrust::less<
        typename iterator_traits<remove_cvref_t<ForwardIt>>::value_type
      >{}
    )
  )

  template <typename... Args>
  THRUST_NODISCARD __host__
  auto operator()(Args&&... args) const
  THRUST_RETURNS(
    call(THRUST_FWD(args)...)
  )
};

} // namespace sort_detail

THRUST_INLINE_CONSTANT sort_detail::sort_fn sort{};

/*! \endcond
 */

} // namespace async

THRUST_NAMESPACE_END

#endif

