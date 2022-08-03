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
#include <thrust/detail/temporary_array.h>
#include <thrust/detail/copy.h>
#include <thrust/iterator/iterator_traits.h>
#include <thrust/distance.h>
#include <thrust/merge.h>
#include <thrust/sort.h>
#include <thrust/detail/seq.h>
#include <tbb/parallel_invoke.h>

THRUST_NAMESPACE_BEGIN
namespace system
{
namespace tbb
{
namespace detail
{
namespace sort_detail
{


// TODO tune this based on data type and comp
const static int threshold = 128 * 1024;


template<typename DerivedPolicy, typename Iterator1, typename Iterator2, typename StrictWeakOrdering>
void merge_sort(execution_policy<DerivedPolicy> &exec, Iterator1 first1, Iterator1 last1, Iterator2 first2, StrictWeakOrdering comp, bool inplace);


template<typename DerivedPolicy, typename Iterator1, typename Iterator2, typename StrictWeakOrdering>
struct merge_sort_closure
{
  execution_policy<DerivedPolicy> &exec;
  Iterator1 first1, last1;
  Iterator2 first2;
  StrictWeakOrdering comp;
  bool inplace;

  merge_sort_closure(execution_policy<DerivedPolicy> &exec, Iterator1 first1, Iterator1 last1, Iterator2 first2, StrictWeakOrdering comp, bool inplace)
    : exec(exec), first1(first1), last1(last1), first2(first2), comp(comp), inplace(inplace)
  {}

  void operator()(void) const
  {
    merge_sort(exec, first1, last1, first2, comp, inplace);
  }
};


template<typename DerivedPolicy, typename Iterator1, typename Iterator2, typename StrictWeakOrdering>
void merge_sort(execution_policy<DerivedPolicy> &exec, Iterator1 first1, Iterator1 last1, Iterator2 first2, StrictWeakOrdering comp, bool inplace)
{
  typedef typename thrust::iterator_difference<Iterator1>::type difference_type;

  difference_type n = thrust::distance(first1, last1);

  if (n < threshold)
  {
    thrust::stable_sort(thrust::seq, first1, last1, comp);

    if(!inplace)
    {
      thrust::copy(thrust::seq, first1, last1, first2);
    }

    return;
  }

  Iterator1 mid1  = first1 + (n / 2);
  Iterator2 mid2  = first2 + (n / 2);
  Iterator2 last2 = first2 + n;

  typedef merge_sort_closure<DerivedPolicy,Iterator1,Iterator2,StrictWeakOrdering> Closure;

  Closure left (exec, first1, mid1,  first2, comp, !inplace);
  Closure right(exec, mid1,   last1, mid2,   comp, !inplace);

  ::tbb::parallel_invoke(left, right);

  if(inplace) thrust::merge(exec, first2, mid2, mid2, last2, first1, comp);
  else	      thrust::merge(exec, first1, mid1, mid1, last1, first2, comp);
}


} // end namespace sort_detail


namespace sort_by_key_detail
{


// TODO tune this based on data type and comp
const static int threshold = 128 * 1024;


template<typename DerivedPolicy,
         typename Iterator1,
         typename Iterator2,
         typename Iterator3,
         typename Iterator4,
         typename StrictWeakOrdering>
void merge_sort_by_key(execution_policy<DerivedPolicy> &exec,
                       Iterator1 first1,
                       Iterator1 last1,
                       Iterator2 first2,
                       Iterator3 first3,
                       Iterator4 first4,
                       StrictWeakOrdering comp,
                       bool inplace);


template<typename DerivedPolicy,
         typename Iterator1,
         typename Iterator2,
         typename Iterator3,
         typename Iterator4,
         typename StrictWeakOrdering>
struct merge_sort_by_key_closure
{
  execution_policy<DerivedPolicy> &exec;
  Iterator1 first1, last1;
  Iterator2 first2;
  Iterator3 first3;
  Iterator4 first4;
  StrictWeakOrdering comp;
  bool inplace;

  merge_sort_by_key_closure(execution_policy<DerivedPolicy> &exec,
                            Iterator1 first1,
                            Iterator1 last1,
                            Iterator2 first2,
                            Iterator3 first3,
                            Iterator4 first4,
                            StrictWeakOrdering comp,
                            bool inplace)
    : exec(exec), first1(first1), last1(last1), first2(first2), first3(first3), first4(first4), comp(comp), inplace(inplace)
  {}

  void operator()(void) const
  {
    merge_sort_by_key(exec, first1, last1, first2, first3, first4, comp, inplace);
  }
};


template<typename DerivedPolicy,
         typename Iterator1,
         typename Iterator2,
         typename Iterator3,
         typename Iterator4,
         typename StrictWeakOrdering>
void merge_sort_by_key(execution_policy<DerivedPolicy> &exec,
                       Iterator1 first1,
                       Iterator1 last1,
                       Iterator2 first2,
                       Iterator3 first3,
                       Iterator4 first4,
                       StrictWeakOrdering comp,
                       bool inplace)
{
  typedef typename thrust::iterator_difference<Iterator1>::type difference_type;

  difference_type n = thrust::distance(first1, last1);

  Iterator1 mid1  = first1 + (n / 2);
  Iterator2 mid2  = first2 + (n / 2);
  Iterator3 mid3  = first3 + (n / 2);
  Iterator4 mid4  = first4 + (n / 2);
  Iterator2 last2 = first2 + n;
  Iterator3 last3 = first3 + n;

  if (n < threshold)
  {
    thrust::stable_sort_by_key(thrust::seq, first1, last1, first2, comp);

    if(!inplace)
    {
      thrust::copy(thrust::seq, first1, last1, first3);
      thrust::copy(thrust::seq, first2, last2, first4);
    }

    return;
  }

  typedef merge_sort_by_key_closure<DerivedPolicy,Iterator1,Iterator2,Iterator3,Iterator4,StrictWeakOrdering> Closure;

  Closure left (exec, first1, mid1,  first2, first3, first4, comp, !inplace);
  Closure right(exec, mid1,   last1, mid2,   mid3,   mid4,   comp, !inplace);

  ::tbb::parallel_invoke(left, right);

  if(inplace)
  {
    thrust::merge_by_key(exec, first3, mid3, mid3, last3, first4, mid4, first1, first2, comp);
  }
  else
  {
    thrust::merge_by_key(exec, first1, mid1, mid1, last1, first2, mid2, first3, first4, comp);
  }
}


} // end namespace sort_detail


template<typename DerivedPolicy,
         typename RandomAccessIterator,
         typename StrictWeakOrdering>
void stable_sort(execution_policy<DerivedPolicy> &exec,
                 RandomAccessIterator first,
                 RandomAccessIterator last,
                 StrictWeakOrdering comp)
{
  typedef typename thrust::iterator_value<RandomAccessIterator>::type key_type;

  thrust::detail::temporary_array<key_type, DerivedPolicy> temp(exec, first, last);

  sort_detail::merge_sort(exec, first, last, temp.begin(), comp, true);
}


template<typename DerivedPolicy,
         typename RandomAccessIterator1,
         typename RandomAccessIterator2,
         typename StrictWeakOrdering>
  void stable_sort_by_key(execution_policy<DerivedPolicy> &exec,
                          RandomAccessIterator1 first1,
                          RandomAccessIterator1 last1,
                          RandomAccessIterator2 first2,
                          StrictWeakOrdering comp)
{
  typedef typename thrust::iterator_value<RandomAccessIterator1>::type key_type;
  typedef typename thrust::iterator_value<RandomAccessIterator2>::type val_type;

  RandomAccessIterator2 last2 = first2 + thrust::distance(first1, last1);

  thrust::detail::temporary_array<key_type, DerivedPolicy> temp1(exec, first1, last1);
  thrust::detail::temporary_array<val_type, DerivedPolicy> temp2(exec, first2, last2);

  sort_by_key_detail::merge_sort_by_key(exec, first1, last1, first2, temp1.begin(), temp2.begin(), comp, true);
}


} // end namespace detail
} // end namespace tbb
} // end namespace system
THRUST_NAMESPACE_END

