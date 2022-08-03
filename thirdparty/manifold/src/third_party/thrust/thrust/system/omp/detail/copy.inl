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
#include <thrust/system/omp/detail/copy.h>
#include <thrust/system/detail/generic/copy.h>
#include <thrust/system/detail/sequential/copy.h>
#include <thrust/detail/type_traits/minimum_type.h>


THRUST_NAMESPACE_BEGIN
namespace system
{
namespace omp
{
namespace detail
{
namespace dispatch
{


template<typename DerivedPolicy,
         typename InputIterator,
         typename OutputIterator>
  OutputIterator copy(execution_policy<DerivedPolicy> &exec,
                      InputIterator first,
                      InputIterator last,
                      OutputIterator result,
                      thrust::incrementable_traversal_tag)
{
  return thrust::system::detail::sequential::copy(exec, first, last, result);
} // end copy()


template<typename DerivedPolicy,
         typename InputIterator,
         typename OutputIterator>
  OutputIterator copy(execution_policy<DerivedPolicy> &exec,
                      InputIterator first,
                      InputIterator last,
                      OutputIterator result,
                      thrust::random_access_traversal_tag)
{
  return thrust::system::detail::generic::copy(exec, first, last, result);
} // end copy()


template<typename DerivedPolicy,
         typename InputIterator,
         typename Size,
         typename OutputIterator>
  OutputIterator copy_n(execution_policy<DerivedPolicy> &exec,
                        InputIterator first,
                        Size n,
                        OutputIterator result,
                        thrust::incrementable_traversal_tag)
{
  return thrust::system::detail::sequential::copy_n(exec, first, n, result);
} // end copy_n()


template<typename DerivedPolicy,
         typename InputIterator,
         typename Size,
         typename OutputIterator>
  OutputIterator copy_n(execution_policy<DerivedPolicy> &exec,
                        InputIterator first,
                        Size n,
                        OutputIterator result,
                        thrust::random_access_traversal_tag)
{
  return thrust::system::detail::generic::copy_n(exec, first, n, result);
} // end copy_n()


} // end dispatch


template<typename DerivedPolicy,
         typename InputIterator,
         typename OutputIterator>
OutputIterator copy(execution_policy<DerivedPolicy> &exec,
                    InputIterator first,
                    InputIterator last,
                    OutputIterator result)
{
  typedef typename thrust::iterator_traversal<InputIterator>::type  traversal1;
  typedef typename thrust::iterator_traversal<OutputIterator>::type traversal2;
  
  typedef typename thrust::detail::minimum_type<traversal1,traversal2>::type traversal;

  // dispatch on minimum traversal
  return thrust::system::omp::detail::dispatch::copy(exec, first, last, result, traversal());
} // end copy()



template<typename DerivedPolicy,
         typename InputIterator,
         typename Size,
         typename OutputIterator>
OutputIterator copy_n(execution_policy<DerivedPolicy> &exec,
                      InputIterator first,
                      Size n,
                      OutputIterator result)
{
  typedef typename thrust::iterator_traversal<InputIterator>::type  traversal1;
  typedef typename thrust::iterator_traversal<OutputIterator>::type traversal2;
  
  typedef typename thrust::detail::minimum_type<traversal1,traversal2>::type traversal;

  // dispatch on minimum traversal
  return thrust::system::omp::detail::dispatch::copy_n(exec, first, n, result, traversal());
} // end copy_n()


} // end namespace detail
} // end namespace omp
} // end namespace system
THRUST_NAMESPACE_END

