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

// don't attempt to #include this file without omp support
#if (THRUST_DEVICE_COMPILER_IS_OMP_CAPABLE == THRUST_TRUE)
#include <omp.h>
#endif // omp support

#include <thrust/iterator/iterator_traits.h>
#include <thrust/system/omp/detail/default_decomposition.h>
#include <thrust/system/detail/generic/select_system.h>
#include <thrust/sort.h>
#include <thrust/merge.h>
#include <thrust/detail/seq.h>
#include <thrust/detail/temporary_array.h>

THRUST_NAMESPACE_BEGIN
namespace system
{
namespace omp
{
namespace detail
{
namespace sort_detail
{


template<typename DerivedPolicy,
         typename RandomAccessIterator,
         typename StrictWeakOrdering>
void inplace_merge(execution_policy<DerivedPolicy> &exec,
                   RandomAccessIterator first,
                   RandomAccessIterator middle,
                   RandomAccessIterator last,
                   StrictWeakOrdering comp)
{
  typedef typename thrust::iterator_value<RandomAccessIterator>::type value_type;

  thrust::detail::temporary_array<value_type,DerivedPolicy> a(exec, first, middle);
  thrust::detail::temporary_array<value_type,DerivedPolicy> b(exec, middle, last);

  thrust::merge(thrust::seq, a.begin(), a.end(), b.begin(), b.end(), first, comp);
}


template<typename DerivedPolicy,
         typename RandomAccessIterator1,
         typename RandomAccessIterator2,
         typename StrictWeakOrdering>
void inplace_merge_by_key(execution_policy<DerivedPolicy> &exec,
                          RandomAccessIterator1 first1,
                          RandomAccessIterator1 middle1,
                          RandomAccessIterator1 last1,
                          RandomAccessIterator2 first2,
                          StrictWeakOrdering comp)
{
  typedef typename thrust::iterator_value<RandomAccessIterator1>::type value_type1;
  typedef typename thrust::iterator_value<RandomAccessIterator2>::type value_type2;

  RandomAccessIterator2 middle2 = first2 + (middle1 - first1);
  RandomAccessIterator2 last2   = first2 + (last1   - first1);

  thrust::detail::temporary_array<value_type1,DerivedPolicy> lhs1(exec, first1, middle1);
  thrust::detail::temporary_array<value_type1,DerivedPolicy> rhs1(exec, middle1, last1);
  thrust::detail::temporary_array<value_type2,DerivedPolicy> lhs2(exec, first2, middle2);
  thrust::detail::temporary_array<value_type2,DerivedPolicy> rhs2(exec, middle2, last2);

  thrust::merge_by_key(thrust::seq,
                       lhs1.begin(), lhs1.end(),
                       rhs1.begin(), rhs1.end(),
                       lhs2.begin(), rhs2.begin(),
                       first1, first2,
                       comp);
}


} // end sort_detail


template<typename DerivedPolicy,
         typename RandomAccessIterator,
         typename StrictWeakOrdering>
void stable_sort(execution_policy<DerivedPolicy> &exec,
                 RandomAccessIterator first,
                 RandomAccessIterator last,
                 StrictWeakOrdering comp)
{
  // we're attempting to launch an omp kernel, assert we're compiling with omp support
  // ========================================================================
  // X Note to the user: If you've found this line due to a compiler error, X
  // X you need to enable OpenMP support in your compiler.                  X
  // ========================================================================
  THRUST_STATIC_ASSERT_MSG(
    (thrust::detail::depend_on_instantiation<
      RandomAccessIterator, (THRUST_DEVICE_COMPILER_IS_OMP_CAPABLE == THRUST_TRUE)
    >::value)
  , "OpenMP compiler support is not enabled"
  );

  // Avoid issues on compilers that don't provide `omp_get_num_threads()`.
#if (THRUST_DEVICE_COMPILER_IS_OMP_CAPABLE == THRUST_TRUE)
  typedef typename thrust::iterator_difference<RandomAccessIterator>::type IndexType;

  if(first == last)
    return;

  THRUST_PRAGMA_OMP(parallel)
  {
    thrust::system::detail::internal::uniform_decomposition<IndexType> decomp(last - first, 1, omp_get_num_threads());

    // process id
    IndexType p_i = omp_get_thread_num();

    // every thread sorts its own tile
    if(p_i < decomp.size())
    {
      thrust::stable_sort(thrust::seq,
                          first + decomp[p_i].begin(),
                          first + decomp[p_i].end(),
                          comp);
    }

    THRUST_PRAGMA_OMP(barrier)

    // XXX For some reason, MSVC 2015 yields an error unless we include this meaningless semicolon here
    ;

    IndexType nseg = decomp.size();
    IndexType h = 2;

    // keep track of which sub-range we're processing
    IndexType a=p_i, b=p_i, c=p_i+1;

    while(nseg>1)
    {
      if(c >= decomp.size())
        c = decomp.size() - 1;

      if((p_i % h) == 0 && c > b)
      {
        sort_detail::inplace_merge(exec,
                                   first + decomp[a].begin(),
                                   first + decomp[b].end(),
                                   first + decomp[c].end(),
                                   comp);

        b = c;
        c += h;
      }

      nseg = (nseg + 1) / 2;
      h *= 2;

      THRUST_PRAGMA_OMP(barrier)
    }
  }
#endif // THRUST_DEVICE_COMPILER_IS_OMP_CAPABLE
}


template<typename DerivedPolicy,
         typename RandomAccessIterator1,
         typename RandomAccessIterator2,
         typename StrictWeakOrdering>
void stable_sort_by_key(execution_policy<DerivedPolicy> &exec,
                        RandomAccessIterator1 keys_first,
                        RandomAccessIterator1 keys_last,
                        RandomAccessIterator2 values_first,
                        StrictWeakOrdering comp)
{
  // we're attempting to launch an omp kernel, assert we're compiling with omp support
  // ========================================================================
  // X Note to the user: If you've found this line due to a compiler error, X
  // X you need to enable OpenMP support in your compiler.                  X
  // ========================================================================
  THRUST_STATIC_ASSERT_MSG(
    (thrust::detail::depend_on_instantiation<
      RandomAccessIterator1, (THRUST_DEVICE_COMPILER_IS_OMP_CAPABLE == THRUST_TRUE)
    >::value)
  , "OpenMP compiler support is not enabled"
  );

  // Avoid issues on compilers that don't provide `omp_get_num_threads()`.
#if (THRUST_DEVICE_COMPILER_IS_OMP_CAPABLE == THRUST_TRUE)
  typedef typename thrust::iterator_difference<RandomAccessIterator1>::type IndexType;

  if(keys_first == keys_last)
    return;

  THRUST_PRAGMA_OMP(parallel)
  {
    thrust::system::detail::internal::uniform_decomposition<IndexType> decomp(keys_last - keys_first, 1, omp_get_num_threads());

    // process id
    IndexType p_i = omp_get_thread_num();

    // every thread sorts its own tile
    if(p_i < decomp.size())
    {
      thrust::stable_sort_by_key(thrust::seq,
                                 keys_first + decomp[p_i].begin(),
                                 keys_first + decomp[p_i].end(),
                                 values_first + decomp[p_i].begin(),
                                 comp);
    }

    THRUST_PRAGMA_OMP(barrier)

    // XXX For some reason, MSVC 2015 yields an error unless we include this meaningless semicolon here
    ;

    IndexType nseg = decomp.size();
    IndexType h = 2;

    // keep track of which sub-range we're processing
    IndexType a=p_i, b=p_i, c=p_i+1;

    while(nseg>1)
    {
      if(c >= decomp.size())
        c = decomp.size() - 1;

      if((p_i % h) == 0 && c > b)
      {
        sort_detail::inplace_merge_by_key(exec,
                                          keys_first + decomp[a].begin(),
                                          keys_first + decomp[b].end(),
                                          keys_first + decomp[c].end(),
                                          values_first + decomp[a].begin(),
                                          comp);

        b = c;
        c += h;
      }

      nseg = (nseg + 1) / 2;
      h *= 2;

      THRUST_PRAGMA_OMP(barrier)
    }
  }
#endif // THRUST_DEVICE_COMPILER_IS_OMP_CAPABLE
}


} // end namespace detail
} // end namespace omp
} // end namespace system
THRUST_NAMESPACE_END

