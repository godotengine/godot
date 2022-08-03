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
#include <thrust/system/omp/detail/default_decomposition.h>

// don't attempt to #include this file without omp support
#if (THRUST_DEVICE_COMPILER_IS_OMP_CAPABLE == THRUST_TRUE)
#include <omp.h>
#endif // omp support

THRUST_NAMESPACE_BEGIN
namespace system
{
namespace omp
{
namespace detail
{

template <typename IndexType>
thrust::system::detail::internal::uniform_decomposition<IndexType> default_decomposition(IndexType n)
{
  // we're attempting to launch an omp kernel, assert we're compiling with omp support
  // ========================================================================
  // X Note to the user: If you've found this line due to a compiler error, X
  // X you need to OpenMP support in your compiler.                         X
  // ========================================================================
  THRUST_STATIC_ASSERT_MSG(
    (thrust::detail::depend_on_instantiation<
      IndexType, (THRUST_DEVICE_COMPILER_IS_OMP_CAPABLE == THRUST_TRUE)
    >::value)
  , "OpenMP compiler support is not enabled"
  );

#if (THRUST_DEVICE_COMPILER_IS_OMP_CAPABLE == THRUST_TRUE)
  return thrust::system::detail::internal::uniform_decomposition<IndexType>(n, 1, omp_get_num_procs());
#else
  return thrust::system::detail::internal::uniform_decomposition<IndexType>(n, 1, 1);
#endif
}

} // end namespace detail
} // end namespace omp
} // end namespace system
THRUST_NAMESPACE_END

