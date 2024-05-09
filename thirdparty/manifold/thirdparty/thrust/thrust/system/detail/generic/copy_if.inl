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
#include <thrust/system/detail/generic/copy_if.h>
#include <thrust/detail/copy_if.h>
#include <thrust/iterator/iterator_traits.h>
#include <thrust/iterator/detail/minimum_system.h>
#include <thrust/functional.h>
#include <thrust/distance.h>
#include <thrust/transform.h>
#include <thrust/detail/internal_functional.h>
#include <thrust/detail/integer_traits.h>
#include <thrust/detail/temporary_array.h>
#include <thrust/detail/type_traits.h>
#include <thrust/scan.h>
#include <thrust/scatter.h>
#include <limits>

THRUST_NAMESPACE_BEGIN
namespace system
{
namespace detail
{
namespace generic
{
namespace detail
{


template<typename IndexType,
         typename DerivedPolicy,
         typename InputIterator1,
         typename InputIterator2,
         typename OutputIterator,
         typename Predicate>
__host__ __device__
OutputIterator copy_if(thrust::execution_policy<DerivedPolicy> &exec,
                       InputIterator1 first,
                       InputIterator1 last,
                       InputIterator2 stencil,
                       OutputIterator result,
                       Predicate pred)
{
  THRUST_DISABLE_MSVC_POSSIBLE_LOSS_OF_DATA_WARNING(IndexType n = thrust::distance(first, last));
  
  // compute {0,1} predicates
  thrust::detail::temporary_array<IndexType, DerivedPolicy> predicates(exec, n);
  thrust::transform(exec,
                    stencil,
                    stencil + n,
                    predicates.begin(),
                    thrust::detail::predicate_to_integral<Predicate,IndexType>(pred));
  
  // scan {0,1} predicates
  thrust::detail::temporary_array<IndexType, DerivedPolicy> scatter_indices(exec, n);
  thrust::exclusive_scan(exec,
                         predicates.begin(),
                         predicates.end(),
                         scatter_indices.begin(),
                         static_cast<IndexType>(0),
                         thrust::plus<IndexType>());
  
  // scatter the true elements
  thrust::scatter_if(exec,
                     first,
                     last,
                     scatter_indices.begin(),
                     predicates.begin(),
                     result,
                     thrust::identity<IndexType>());
  
  // find the end of the new sequence
  IndexType output_size = scatter_indices[n - 1] + predicates[n - 1];
  
  return result + output_size;
}


} // end namespace detail


template<typename DerivedPolicy,
         typename InputIterator,
         typename OutputIterator,
         typename Predicate>
__host__ __device__
  OutputIterator copy_if(thrust::execution_policy<DerivedPolicy> &exec,
                         InputIterator first,
                         InputIterator last,
                         OutputIterator result,
                         Predicate pred)
{
  // XXX it's potentially expensive to send [first,last) twice
  //     we should probably specialize this case for POD
  //     since we can safely keep the input in a temporary instead
  //     of doing two loads
  return thrust::copy_if(exec, first, last, first, result, pred);
} // end copy_if()


template<typename DerivedPolicy,
         typename InputIterator1,
         typename InputIterator2,
         typename OutputIterator,
         typename Predicate>
__host__ __device__
   OutputIterator copy_if(thrust::execution_policy<DerivedPolicy> &exec,
                          InputIterator1 first,
                          InputIterator1 last,
                          InputIterator2 stencil,
                          OutputIterator result,
                          Predicate pred)
{
  typedef typename thrust::iterator_traits<InputIterator1>::difference_type difference_type;
  
  // empty sequence
  if(first == last)
    return result;
  
  difference_type n = thrust::distance(first, last);
  
  // create an unsigned version of n (we know n is positive from the comparison above)
  // to avoid a warning in the compare below
  typename thrust::detail::make_unsigned<difference_type>::type unsigned_n(n);
  
  // use 32-bit indices when possible (almost always)
  if(sizeof(difference_type) > sizeof(unsigned int) && unsigned_n > thrust::detail::integer_traits<unsigned int>::const_max)
  {
    result = detail::copy_if<difference_type>(exec, first, last, stencil, result, pred);
  } // end if
  else
  {
    result = detail::copy_if<unsigned int>(exec, first, last, stencil, result, pred);
  } // end else

  return result;
} // end copy_if()


} // end namespace generic
} // end namespace detail
} // end namespace system
THRUST_NAMESPACE_END

