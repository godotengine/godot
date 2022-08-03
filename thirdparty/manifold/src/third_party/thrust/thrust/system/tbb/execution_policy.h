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

/*! \file thrust/system/tbb/execution_policy.h
 *  \brief Execution policies for Thrust's TBB system.
 */

#include <thrust/detail/config.h>

// get the execution policies definitions first
#include <thrust/system/tbb/detail/execution_policy.h>

// get the definition of par
#include <thrust/system/tbb/detail/par.h>

// now get all the algorithm definitions

#include <thrust/system/tbb/detail/adjacent_difference.h>
#include <thrust/system/tbb/detail/assign_value.h>
#include <thrust/system/tbb/detail/binary_search.h>
#include <thrust/system/tbb/detail/copy.h>
#include <thrust/system/tbb/detail/copy_if.h>
#include <thrust/system/tbb/detail/count.h>
#include <thrust/system/tbb/detail/equal.h>
#include <thrust/system/tbb/detail/extrema.h>
#include <thrust/system/tbb/detail/fill.h>
#include <thrust/system/tbb/detail/find.h>
#include <thrust/system/tbb/detail/for_each.h>
#include <thrust/system/tbb/detail/gather.h>
#include <thrust/system/tbb/detail/generate.h>
#include <thrust/system/tbb/detail/get_value.h>
#include <thrust/system/tbb/detail/inner_product.h>
#include <thrust/system/tbb/detail/iter_swap.h>
#include <thrust/system/tbb/detail/logical.h>
#include <thrust/system/tbb/detail/malloc_and_free.h>
#include <thrust/system/tbb/detail/merge.h>
#include <thrust/system/tbb/detail/mismatch.h>
#include <thrust/system/tbb/detail/partition.h>
#include <thrust/system/tbb/detail/reduce.h>
#include <thrust/system/tbb/detail/reduce_by_key.h>
#include <thrust/system/tbb/detail/remove.h>
#include <thrust/system/tbb/detail/replace.h>
#include <thrust/system/tbb/detail/reverse.h>
#include <thrust/system/tbb/detail/scan.h>
#include <thrust/system/tbb/detail/scan_by_key.h>
#include <thrust/system/tbb/detail/scatter.h>
#include <thrust/system/tbb/detail/sequence.h>
#include <thrust/system/tbb/detail/set_operations.h>
#include <thrust/system/tbb/detail/sort.h>
#include <thrust/system/tbb/detail/swap_ranges.h>
#include <thrust/system/tbb/detail/tabulate.h>
#include <thrust/system/tbb/detail/transform.h>
#include <thrust/system/tbb/detail/transform_reduce.h>
#include <thrust/system/tbb/detail/transform_scan.h>
#include <thrust/system/tbb/detail/uninitialized_copy.h>
#include <thrust/system/tbb/detail/uninitialized_fill.h>
#include <thrust/system/tbb/detail/unique.h>
#include <thrust/system/tbb/detail/unique_by_key.h>


// define these entities here for the purpose of Doxygenating them
// they are actually defined elsewhere
#if 0
THRUST_NAMESPACE_BEGIN
namespace system
{
namespace tbb
{


/*! \addtogroup execution_policies
 *  \{
 */


/*! \p thrust::tbb::execution_policy is the base class for all Thrust parallel execution
 *  policies which are derived from Thrust's TBB backend system.
 */
template<typename DerivedPolicy>
struct execution_policy : thrust::execution_policy<DerivedPolicy>
{};


/*! \p tbb::tag is a type representing Thrust's TBB backend system in C++'s type system.
 *  Iterators "tagged" with a type which is convertible to \p tbb::tag assert that they may be
 *  "dispatched" to algorithm implementations in the \p tbb system.
 */
struct tag : thrust::system::tbb::execution_policy<tag> { unspecified };


/*! \p thrust::tbb::par is the parallel execution policy associated with Thrust's TBB
 *  backend system.
 *
 *  Instead of relying on implicit algorithm dispatch through iterator system tags, users may
 *  directly target Thrust's TBB backend system by providing \p thrust::tbb::par as an algorithm
 *  parameter.
 *
 *  Explicit dispatch can be useful in avoiding the introduction of data copies into containers such
 *  as \p thrust::tbb::vector.
 *
 *  The type of \p thrust::tbb::par is implementation-defined.
 *
 *  The following code snippet demonstrates how to use \p thrust::tbb::par to explicitly dispatch an
 *  invocation of \p thrust::for_each to the TBB backend system:
 *
 *  \code
 *  #include <thrust/for_each.h>
 *  #include <thrust/system/tbb/execution_policy.h>
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
 *  thrust::for_each(thrust::tbb::par, vec.begin(), vec.end(), printf_functor());
 *
 *  // 0 1 2 is printed to standard output in some unspecified order
 *  \endcode
 */
static const unspecified par;


/*! \}
 */


} // end tbb
} // end system
THRUST_NAMESPACE_END
#endif


