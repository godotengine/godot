/******************************************************************************
 * Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *     * Neither the name of the NVIDIA CORPORATION nor the
 *       names of its contributors may be used to endorse or promote products
 *       derived from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 ******************************************************************************/

#pragma once

#include <thrust/detail/config.h>

#include <thrust/version.h>
#include <thrust/detail/execution_policy.h>
#include <thrust/iterator/detail/any_system_tag.h>
#include <thrust/system/cuda/config.h>

#include <thrust/detail/allocator_aware_execution_policy.h>

#if THRUST_CPP_DIALECT >= 2011
  #include <thrust/detail/dependencies_aware_execution_policy.h>
#endif

THRUST_NAMESPACE_BEGIN

namespace cuda_cub
{

struct tag;

template <class>
struct execution_policy;

template <>
struct execution_policy<tag> : thrust::execution_policy<tag>
{
  typedef tag tag_type;
};

struct tag : execution_policy<tag>
, thrust::detail::allocator_aware_execution_policy<cuda_cub::execution_policy>
#if THRUST_CPP_DIALECT >= 2011
, thrust::detail::dependencies_aware_execution_policy<cuda_cub::execution_policy>
#endif
{};

template <class Derived>
struct execution_policy : thrust::execution_policy<Derived>
{
  typedef tag tag_type; 
  operator tag() const { return tag(); }
};

} // namespace cuda_cub

namespace system { namespace cuda { namespace detail
{

using thrust::cuda_cub::tag;
using thrust::cuda_cub::execution_policy;

}}} // namespace system::cuda::detail

namespace system { namespace cuda
{

using thrust::cuda_cub::tag;
using thrust::cuda_cub::execution_policy;

}} // namespace system::cuda

namespace cuda
{

using thrust::cuda_cub::tag;
using thrust::cuda_cub::execution_policy;

} // namespace cuda

THRUST_NAMESPACE_END

