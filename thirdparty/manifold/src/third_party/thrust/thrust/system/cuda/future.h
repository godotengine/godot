// Copyright (c) 2018 NVIDIA Corporation
// Author: Bryce Adelstein Lelbach <brycelelbach@gmail.com>
//
// Distributed under the Boost Software License v1.0 (boost.org/LICENSE_1_0.txt)

#pragma once

#include <thrust/detail/config.h>
#include <thrust/detail/cpp14_required.h>

#if THRUST_CPP_DIALECT >= 2014

#include <thrust/system/cuda/pointer.h>
#include <thrust/system/cuda/detail/execution_policy.h>

THRUST_NAMESPACE_BEGIN

namespace system { namespace cuda
{

struct ready_event;

template <typename T>
struct ready_future;

struct unique_eager_event;

template <typename T>
struct unique_eager_future;

template <typename... Events>
__host__
unique_eager_event when_all(Events&&... evs);

}} // namespace system::cuda

namespace cuda
{

using thrust::system::cuda::ready_event;

using thrust::system::cuda::ready_future;

using thrust::system::cuda::unique_eager_event;
using event = unique_eager_event;

using thrust::system::cuda::unique_eager_future;
template <typename T> using future = unique_eager_future<T>;

using thrust::system::cuda::when_all;

} // namespace cuda

template <typename DerivedPolicy>
__host__ 
thrust::cuda::unique_eager_event
unique_eager_event_type(
  thrust::cuda::execution_policy<DerivedPolicy> const&
) noexcept;

template <typename T, typename DerivedPolicy>
__host__ 
thrust::cuda::unique_eager_future<T>
unique_eager_future_type(
  thrust::cuda::execution_policy<DerivedPolicy> const&
) noexcept;

THRUST_NAMESPACE_END

#include <thrust/system/cuda/detail/future.inl>

#endif // C++14

