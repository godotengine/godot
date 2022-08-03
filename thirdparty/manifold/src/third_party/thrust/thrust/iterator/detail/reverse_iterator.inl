/*
 *  Copyright 2008-2021 NVIDIA Corporation
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

#include <thrust/iterator/reverse_iterator.h>
#include <thrust/iterator/iterator_traits.h>

THRUST_NAMESPACE_BEGIN

namespace detail
{

__thrust_exec_check_disable__
template<typename Iterator>
  __host__ __device__
  Iterator prior(Iterator x)
{
  return --x;
} // end prior()

} // end detail

template<typename BidirectionalIterator>
  __host__ __device__
  reverse_iterator<BidirectionalIterator>
    ::reverse_iterator(BidirectionalIterator x)
      :super_t(x)
{
} // end reverse_iterator::reverse_iterator()

template<typename BidirectionalIterator>
  template<typename OtherBidirectionalIterator>
    __host__ __device__
    reverse_iterator<BidirectionalIterator>
      ::reverse_iterator(reverse_iterator<OtherBidirectionalIterator> const &r
// XXX msvc screws this up
#if THRUST_HOST_COMPILER != THRUST_HOST_COMPILER_MSVC
                     , typename thrust::detail::enable_if<
                         thrust::detail::is_convertible<
                           OtherBidirectionalIterator,
                           BidirectionalIterator
                         >::value
                       >::type *
#endif // MSVC
                     )
        :super_t(r.base())
{
} // end reverse_iterator::reverse_iterator()

template<typename BidirectionalIterator>
  __host__ __device__
  typename reverse_iterator<BidirectionalIterator>::super_t::reference
    reverse_iterator<BidirectionalIterator>
      ::dereference() const
{
  return *thrust::detail::prior(this->base());
} // end reverse_iterator::increment()

template<typename BidirectionalIterator>
  __host__ __device__
  void reverse_iterator<BidirectionalIterator>
    ::increment()
{
  --this->base_reference();
} // end reverse_iterator::increment()

template<typename BidirectionalIterator>
  __host__ __device__
  void reverse_iterator<BidirectionalIterator>
    ::decrement()
{
  ++this->base_reference();
} // end reverse_iterator::decrement()

template<typename BidirectionalIterator>
  __host__ __device__
  void reverse_iterator<BidirectionalIterator>
    ::advance(typename super_t::difference_type n)
{
  this->base_reference() += -n;
} // end reverse_iterator::advance()

template<typename BidirectionalIterator>
  template<typename OtherBidirectionalIterator>
    __host__ __device__
    typename reverse_iterator<BidirectionalIterator>::super_t::difference_type
      reverse_iterator<BidirectionalIterator>
        ::distance_to(reverse_iterator<OtherBidirectionalIterator> const &y) const
{
  return this->base_reference() - y.base();
} // end reverse_iterator::distance_to()

template<typename BidirectionalIterator>
__host__ __device__
reverse_iterator<BidirectionalIterator> make_reverse_iterator(BidirectionalIterator x)
{
  return reverse_iterator<BidirectionalIterator>(x);
} // end make_reverse_iterator()


THRUST_NAMESPACE_END

