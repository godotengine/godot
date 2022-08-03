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

/*! \file thrust/system/cpp/execution_policy.h
 *  \brief Execution policies for Thrust's Standard C++ system.
 */

#pragma once

#include <thrust/detail/config.h>

// get the execution policies definitions first
#include <thrust/system/cpp/detail/execution_policy.h>

// get the definition of par
#include <thrust/system/cpp/detail/par.h>

// now get all the algorithm definitions

#include <thrust/system/cpp/detail/adjacent_difference.h>
#include <thrust/system/cpp/detail/assign_value.h>
#include <thrust/system/cpp/detail/binary_search.h>
#include <thrust/system/cpp/detail/copy.h>
#include <thrust/system/cpp/detail/copy_if.h>
#include <thrust/system/cpp/detail/count.h>
#include <thrust/system/cpp/detail/equal.h>
#include <thrust/system/cpp/detail/extrema.h>
#include <thrust/system/cpp/detail/fill.h>
#include <thrust/system/cpp/detail/find.h>
#include <thrust/system/cpp/detail/for_each.h>
#include <thrust/system/cpp/detail/gather.h>
#include <thrust/system/cpp/detail/generate.h>
#include <thrust/system/cpp/detail/get_value.h>
#include <thrust/system/cpp/detail/inner_product.h>
#include <thrust/system/cpp/detail/iter_swap.h>
#include <thrust/system/cpp/detail/logical.h>
#include <thrust/system/cpp/detail/malloc_and_free.h>
#include <thrust/system/cpp/detail/merge.h>
#include <thrust/system/cpp/detail/mismatch.h>
#include <thrust/system/cpp/detail/partition.h>
#include <thrust/system/cpp/detail/reduce.h>
#include <thrust/system/cpp/detail/reduce_by_key.h>
#include <thrust/system/cpp/detail/remove.h>
#include <thrust/system/cpp/detail/replace.h>
#include <thrust/system/cpp/detail/reverse.h>
#include <thrust/system/cpp/detail/scan.h>
#include <thrust/system/cpp/detail/scan_by_key.h>
#include <thrust/system/cpp/detail/scatter.h>
#include <thrust/system/cpp/detail/sequence.h>
#include <thrust/system/cpp/detail/set_operations.h>
#include <thrust/system/cpp/detail/sort.h>
#include <thrust/system/cpp/detail/swap_ranges.h>
#include <thrust/system/cpp/detail/tabulate.h>
#include <thrust/system/cpp/detail/transform.h>
#include <thrust/system/cpp/detail/transform_reduce.h>
#include <thrust/system/cpp/detail/transform_scan.h>
#include <thrust/system/cpp/detail/uninitialized_copy.h>
#include <thrust/system/cpp/detail/uninitialized_fill.h>
#include <thrust/system/cpp/detail/unique.h>
#include <thrust/system/cpp/detail/unique_by_key.h>


// define these entities here for the purpose of Doxygenating them
// they are actually defined elsewhere
#if 0
THRUST_NAMESPACE_BEGIN
namespace system
{
namespace cpp
{


/*! \addtogroup execution_policies
 *  \{
 */


/*! \p thrust::system::cpp::execution_policy is the base class for all Thrust parallel execution
 *  policies which are derived from Thrust's standard C++ backend system.
 */
template<typename DerivedPolicy>
struct execution_policy : thrust::execution_policy<DerivedPolicy>
{};


/*! \p thrust::system::cpp::tag is a type representing Thrust's standard C++ backend system in C++'s type system.
 *  Iterators "tagged" with a type which is convertible to \p cpp::tag assert that they may be
 *  "dispatched" to algorithm implementations in the \p cpp system.
 */
struct tag : thrust::system::cpp::execution_policy<tag> { unspecified };


/*!
 *  \p thrust::system::cpp::par is the parallel execution policy associated with Thrust's standard
 *  C++ backend system.
 *
 *  Instead of relying on implicit algorithm dispatch through iterator system tags, users may
 *  directly target Thrust's C++ backend system by providing \p thrust::cpp::par as an algorithm
 *  parameter.
 *
 *  Explicit dispatch can be useful in avoiding the introduction of data copies into containers such
 *  as \p thrust::cpp::vector.
 *
 *  The type of \p thrust::cpp::par is implementation-defined.
 *
 *  The following code snippet demonstrates how to use \p thrust::cpp::par to explicitly dispatch an
 *  invocation of \p thrust::for_each to the standard C++ backend system:
 *
 *  \code
 *  #include <thrust/for_each.h>
 *  #include <thrust/system/cpp/execution_policy.h>
 *  #include <cstdio>
 *
 *  struct printf_functor
 *  {
 *    __host__ __device__
 *    void operator()(int x)
 *    {
 *      printf("%d\n", x);
 *    }
 *  };
 *  ...
 *  int vec[3];
 *  vec[0] = 0; vec[1] = 1; vec[2] = 2;
 *
 *  thrust::for_each(thrust::cpp::par, vec.begin(), vec.end(), printf_functor());
 *
 *  // 0 1 2 is printed to standard output in some unspecified order
 *  \endcode
 */
static const unspecified par;


/*! \}
 */


} // end cpp
} // end system
THRUST_NAMESPACE_END
#endif


