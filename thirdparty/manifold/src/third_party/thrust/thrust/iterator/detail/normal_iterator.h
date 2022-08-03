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


/*! \file normal_iterator.h
 *  \brief Defines the interface to an iterator class
 *         which adapts a pointer type.
 */

#pragma once

#include <thrust/detail/config.h>

#include <thrust/iterator/iterator_adaptor.h>
#include <thrust/detail/type_traits.h>
#include <thrust/type_traits/is_contiguous_iterator.h>

THRUST_NAMESPACE_BEGIN
namespace detail
{


template<typename Pointer>
  class normal_iterator
    : public iterator_adaptor<
        normal_iterator<Pointer>,
        Pointer
      >
{
  typedef iterator_adaptor<normal_iterator<Pointer>, Pointer> super_t;

  public:
    __host__ __device__
    normal_iterator() {}

    __host__ __device__
    normal_iterator(Pointer p)
      : super_t(p) {}
    
    template<typename OtherPointer>
    __host__ __device__
    normal_iterator(const normal_iterator<OtherPointer> &other,
                    typename thrust::detail::enable_if_convertible<
                      OtherPointer,
                      Pointer
                    >::type * = 0)
      : super_t(other.base()) {}

}; // end normal_iterator


template<typename Pointer>
  inline __host__ __device__ normal_iterator<Pointer> make_normal_iterator(Pointer ptr)
{
  return normal_iterator<Pointer>(ptr);
}

} // end detail

template <typename T>
struct proclaim_contiguous_iterator<
  thrust::detail::normal_iterator<T>
> : true_type {};

THRUST_NAMESPACE_END

