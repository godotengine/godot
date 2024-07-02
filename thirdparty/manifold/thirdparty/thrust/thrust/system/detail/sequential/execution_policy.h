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
#include <thrust/detail/execution_policy.h>

THRUST_NAMESPACE_BEGIN
namespace system
{
namespace detail
{
namespace sequential
{


// this awkward sequence of definitions arises
// from the desire both for tag to derive
// from execution_policy and for execution_policy
// to convert to tag (when execution_policy is not
// an ancestor of tag)

// forward declaration of tag
struct tag;

// forward declaration of execution_policy
template<typename> struct execution_policy;

// specialize execution_policy for tag
template<>
  struct execution_policy<tag>
    : thrust::execution_policy<tag>
{};

// tag's definition comes before the generic definition of execution_policy
struct tag : execution_policy<tag>
{
  __host__ __device__ constexpr tag() {}
};

// allow conversion to tag when it is not a successor
template<typename Derived>
  struct execution_policy
    : thrust::execution_policy<Derived>
{
  // allow conversion to tag
  inline operator tag () const
  {
    return tag();
  }
};


THRUST_INLINE_CONSTANT tag seq;


} // end sequential
} // end detail
} // end system
THRUST_NAMESPACE_END

