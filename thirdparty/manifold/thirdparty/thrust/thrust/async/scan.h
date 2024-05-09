/*
 *  Copyright 2008-2020 NVIDIA Corporation
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

/*! \file async/scan.h
 *  \brief Functions for asynchronously computing prefix scans.
 */

#pragma once

#include <thrust/detail/config.h>
#include <thrust/detail/cpp14_required.h>

#if THRUST_CPP_DIALECT >= 2014

#include <thrust/detail/execution_policy.h>
#include <thrust/detail/select_system.h>
#include <thrust/detail/static_assert.h>

#include <thrust/system/detail/adl/async/scan.h>

#include <thrust/type_traits/is_execution_policy.h>
#include <thrust/type_traits/logical_metafunctions.h>
#include <thrust/type_traits/remove_cvref.h>

#include <thrust/future.h>

THRUST_NAMESPACE_BEGIN

namespace async
{

// Fallback implementations used when no overloads are found via ADL:
namespace unimplemented
{

template <typename DerivedPolicy,
          typename ForwardIt,
          typename Sentinel,
          typename OutputIt,
          typename BinaryOp>
event<DerivedPolicy>
async_inclusive_scan(thrust::execution_policy<DerivedPolicy>&,
                     ForwardIt,
                     Sentinel,
                     OutputIt,
                     BinaryOp)
{
  THRUST_STATIC_ASSERT_MSG(
    (thrust::detail::depend_on_instantiation<ForwardIt, false>::value),
    "this algorithm is not implemented for the specified system"
  );
  return {};
}

template <typename DerivedPolicy,
          typename ForwardIt,
          typename Sentinel,
          typename OutputIt,
          typename InitialValueType,
          typename BinaryOp>
event<DerivedPolicy>
async_exclusive_scan(thrust::execution_policy<DerivedPolicy>&,
                     ForwardIt,
                     Sentinel,
                     OutputIt,
                     InitialValueType,
                     BinaryOp)
{
  THRUST_STATIC_ASSERT_MSG(
    (thrust::detail::depend_on_instantiation<ForwardIt, false>::value),
    "this algorithm is not implemented for the specified system"
  );
  return {};
}

} // namespace unimplemented

namespace inclusive_scan_detail
{

// Include fallback implementation for ADL failures
using thrust::async::unimplemented::async_inclusive_scan;

// Implementation of the thrust::async::inclusive_scan CPO.
struct inclusive_scan_fn final
{
  template <typename DerivedPolicy,
            typename ForwardIt,
            typename Sentinel,
            typename OutputIt,
            typename BinaryOp>
  auto
  operator()(thrust::detail::execution_policy_base<DerivedPolicy> const& exec,
             ForwardIt&& first,
             Sentinel&& last,
             OutputIt&& out,
             BinaryOp&& op) const
  // ADL dispatch.
  THRUST_RETURNS(
    async_inclusive_scan(
      thrust::detail::derived_cast(thrust::detail::strip_const(exec)),
      THRUST_FWD(first),
      THRUST_FWD(last),
      THRUST_FWD(out),
      THRUST_FWD(op)
    )
  )

  template <typename DerivedPolicy,
            typename ForwardIt,
            typename Sentinel,
            typename OutputIt>
  auto
  operator()(thrust::detail::execution_policy_base<DerivedPolicy> const& exec,
             ForwardIt&& first,
             Sentinel&& last,
             OutputIt&& out) const
  // ADL dispatch.
  THRUST_RETURNS(
    async_inclusive_scan(
      thrust::detail::derived_cast(thrust::detail::strip_const(exec)),
      THRUST_FWD(first),
      THRUST_FWD(last),
      THRUST_FWD(out),
      thrust::plus<>{}
    )
  )

  template <typename ForwardIt,
            typename Sentinel,
            typename OutputIt,
            typename BinaryOp,
            typename = std::enable_if_t<!is_execution_policy_v<remove_cvref_t<ForwardIt>>>>
  auto operator()(ForwardIt&& first,
                  Sentinel&& last,
                  OutputIt&& out,
                  BinaryOp&& op) const
  // ADL dispatch.
  THRUST_RETURNS(
    async_inclusive_scan(
      thrust::detail::select_system(
        iterator_system_t<remove_cvref_t<ForwardIt>>{},
        iterator_system_t<remove_cvref_t<OutputIt>>{}
      ),
      THRUST_FWD(first),
      THRUST_FWD(last),
      THRUST_FWD(out),
      THRUST_FWD(op)
    )
  )

  template <typename ForwardIt, typename Sentinel, typename OutputIt>
  auto operator()(ForwardIt&& first, Sentinel&& last, OutputIt&& out) const
  // ADL dispatch.
  THRUST_RETURNS(
    async_inclusive_scan(
      thrust::detail::select_system(
        iterator_system_t<remove_cvref_t<ForwardIt>>{},
        iterator_system_t<remove_cvref_t<OutputIt>>{}
      ),
      THRUST_FWD(first),
      THRUST_FWD(last),
      THRUST_FWD(out),
      thrust::plus<>{}
    )
  )
};

} // namespace inclusive_scan_detail

THRUST_INLINE_CONSTANT inclusive_scan_detail::inclusive_scan_fn inclusive_scan{};

namespace exclusive_scan_detail
{

// Include fallback implementation for ADL failures
using thrust::async::unimplemented::async_exclusive_scan;

// Implementation of the thrust::async::exclusive_scan CPO.
struct exclusive_scan_fn final
{
  template <typename DerivedPolicy,
            typename ForwardIt,
            typename Sentinel,
            typename OutputIt,
            typename InitialValueType,
            typename BinaryOp>
  auto
  operator()(thrust::detail::execution_policy_base<DerivedPolicy> const& exec,
             ForwardIt&& first,
             Sentinel&& last,
             OutputIt&& out,
             InitialValueType&& init,
             BinaryOp&& op) const
  // ADL dispatch.
  THRUST_RETURNS(
    async_exclusive_scan(
      thrust::detail::derived_cast(thrust::detail::strip_const(exec)),
      THRUST_FWD(first),
      THRUST_FWD(last),
      THRUST_FWD(out),
      THRUST_FWD(init),
      THRUST_FWD(op)
    )
  )

  template <typename DerivedPolicy,
            typename ForwardIt,
            typename Sentinel,
            typename OutputIt,
            typename InitialValueType>
  auto
  operator()(thrust::detail::execution_policy_base<DerivedPolicy> const& exec,
             ForwardIt&& first,
             Sentinel&& last,
             OutputIt&& out,
             InitialValueType&& init) const
  // ADL dispatch.
  THRUST_RETURNS(
    async_exclusive_scan(
      thrust::detail::derived_cast(thrust::detail::strip_const(exec)),
      THRUST_FWD(first),
      THRUST_FWD(last),
      THRUST_FWD(out),
      THRUST_FWD(init),
      thrust::plus<>{}
    )
  )

  template <typename DerivedPolicy,
            typename ForwardIt,
            typename Sentinel,
            typename OutputIt>
  auto
  operator()(thrust::detail::execution_policy_base<DerivedPolicy> const& exec,
             ForwardIt&& first,
             Sentinel&& last,
             OutputIt&& out) const
  // ADL dispatch.
  THRUST_RETURNS(
    async_exclusive_scan(
      thrust::detail::derived_cast(thrust::detail::strip_const(exec)),
      THRUST_FWD(first),
      THRUST_FWD(last),
      THRUST_FWD(out),
      iterator_value_t<remove_cvref_t<ForwardIt>>{},
      thrust::plus<>{}
    )
  )

  template <typename ForwardIt,
            typename Sentinel,
            typename OutputIt,
            typename InitialValueType,
            typename BinaryOp,
            typename = std::enable_if_t<!is_execution_policy_v<remove_cvref_t<ForwardIt>>>>
  auto
  operator()(ForwardIt&& first,
             Sentinel&& last,
             OutputIt&& out,
             InitialValueType&& init,
             BinaryOp&& op) const
  // ADL dispatch.
  THRUST_RETURNS(
    async_exclusive_scan(
      thrust::detail::select_system(
        iterator_system_t<remove_cvref_t<ForwardIt>>{},
        iterator_system_t<remove_cvref_t<OutputIt>>{}
      ),
      THRUST_FWD(first),
      THRUST_FWD(last),
      THRUST_FWD(out),
      THRUST_FWD(init),
      THRUST_FWD(op)
    )
  )

  template <typename ForwardIt,
            typename Sentinel,
            typename OutputIt,
            typename InitialValueType,
            typename = std::enable_if_t<!is_execution_policy_v<remove_cvref_t<ForwardIt>>>>
  auto
  operator()(ForwardIt&& first,
             Sentinel&& last,
             OutputIt&& out,
             InitialValueType&& init) const
  // ADL dispatch.
  THRUST_RETURNS(
    async_exclusive_scan(
      thrust::detail::select_system(
        iterator_system_t<remove_cvref_t<ForwardIt>>{},
        iterator_system_t<remove_cvref_t<OutputIt>>{}
      ),
      THRUST_FWD(first),
      THRUST_FWD(last),
      THRUST_FWD(out),
      THRUST_FWD(init),
      thrust::plus<>{}
    )
  )

  template <typename ForwardIt, typename Sentinel, typename OutputIt>
  auto operator()(ForwardIt&& first,
                  Sentinel&& last,
                  OutputIt&& out) const
  // ADL dispatch.
  THRUST_RETURNS(
    async_exclusive_scan(
      thrust::detail::select_system(
        iterator_system_t<remove_cvref_t<ForwardIt>>{},
        iterator_system_t<remove_cvref_t<OutputIt>>{}
      ),
      THRUST_FWD(first),
      THRUST_FWD(last),
      THRUST_FWD(out),
      iterator_value_t<remove_cvref_t<ForwardIt>>{},
      thrust::plus<>{}
    )
  )
};

} // namespace exclusive_scan_detail

THRUST_INLINE_CONSTANT exclusive_scan_detail::exclusive_scan_fn exclusive_scan{};

} // namespace async

THRUST_NAMESPACE_END

#endif
