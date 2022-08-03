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
#include <thrust/iterator/iterator_facade.h>
#include <thrust/iterator/detail/minimum_system.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/detail/type_traits.h>


THRUST_NAMESPACE_BEGIN
namespace detail
{


template<typename RandomAccessIterator1,
         typename RandomAccessIterator2,
         typename Difference,
         typename Reference>
class join_iterator;


namespace join_iterator_detail
{


template<typename RandomAccessIterator1,
         typename RandomAccessIterator2,
         typename Difference,
         typename Reference>
struct join_iterator_base
{
  typedef typename thrust::detail::remove_reference<Reference>::type value_type;

  typedef typename thrust::iterator_system<RandomAccessIterator1>::type  system1;
  typedef typename thrust::iterator_system<RandomAccessIterator2>::type  system2;
  typedef typename thrust::detail::minimum_system<system1,system2>::type system;

  typedef thrust::iterator_adaptor<
    join_iterator<RandomAccessIterator1,RandomAccessIterator2,Difference,Reference>,
    thrust::counting_iterator<Difference>,
    value_type,
    system,
    thrust::random_access_traversal_tag,
    Reference,
    Difference
  > type;
}; // end join_iterator_base


} // end join_iterator_detail


template<typename RandomAccessIterator1,
         typename RandomAccessIterator2,
         typename Difference = typename thrust::iterator_difference<RandomAccessIterator1>::type,
         typename Reference  = typename thrust::iterator_value<RandomAccessIterator1>::type>
class join_iterator
  : public join_iterator_detail::join_iterator_base<RandomAccessIterator1, RandomAccessIterator2, Difference, Reference>::type
{
  private:
    typedef typename join_iterator_detail::join_iterator_base<RandomAccessIterator1, RandomAccessIterator2, Difference, Reference>::type super_t;
    typedef typename super_t::difference_type size_type;

  public:
    inline __host__ __device__
    join_iterator(RandomAccessIterator1 first1, size_type n, RandomAccessIterator2 first2)
      : super_t(thrust::counting_iterator<size_type>(0)),
        m_n1(n),
        m_iter1(first1),
        m_iter2(first2 - m_n1)
    {}


    inline __host__ __device__
    join_iterator(const join_iterator &other)
      : super_t(other),
        m_n1(other.m_n1),
        m_iter1(other.m_iter1),
        m_iter2(other.m_iter2)
    {}


  private:
    friend class thrust::iterator_core_access;

    // MSVC 2013 and 2015 incorrectly warning about returning a reference to
    // a local/temporary here.
    // See goo.gl/LELTNp
    THRUST_DISABLE_MSVC_WARNING_BEGIN(4172)

    __host__ __device__
    typename super_t::reference dereference() const
    {
      size_type i = *super_t::base();
      return (i < m_n1) ? m_iter1[i] : static_cast<typename super_t::reference>(m_iter2[i]);
    } // end dereference()

    THRUST_DISABLE_MSVC_WARNING_END(4172)


    size_type m_n1;
    RandomAccessIterator1 m_iter1;
    RandomAccessIterator2 m_iter2;
}; // end join_iterator


template<typename RandomAccessIterator1, typename Size, typename RandomAccessIterator2>
__host__ __device__
join_iterator<RandomAccessIterator1,RandomAccessIterator2,Size> make_join_iterator(RandomAccessIterator1 first1, Size n1, RandomAccessIterator2 first2)
{
  return join_iterator<RandomAccessIterator1,RandomAccessIterator2,Size>(first1, n1, first2);
} // end make_join_iterator()


} // end detail
THRUST_NAMESPACE_END

