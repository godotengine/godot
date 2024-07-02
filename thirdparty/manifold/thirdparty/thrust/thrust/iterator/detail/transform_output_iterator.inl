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

#include <thrust/iterator/transform_output_iterator.h>
#include <thrust/iterator/iterator_adaptor.h>

THRUST_NAMESPACE_BEGIN

template <typename UnaryFunction, typename OutputIterator>
  class transform_output_iterator;

namespace detail
{

// Proxy reference that uses Unary Function to transform the rhs of assigment
// operator before writing the result to OutputIterator
template <typename UnaryFunction, typename OutputIterator>
  class transform_output_iterator_proxy
{
  public:
    __host__ __device__
    transform_output_iterator_proxy(const OutputIterator& out, UnaryFunction fun) : out(out), fun(fun)
    {
    }

    __thrust_exec_check_disable__
    template <typename T>
    __host__ __device__
    transform_output_iterator_proxy operator=(const T& x)
    {
      *out = fun(x);
      return *this;
    }

  private:
    OutputIterator out;
    UnaryFunction fun;
};

// Compute the iterator_adaptor instantiation to be used for transform_output_iterator
template <typename UnaryFunction, typename OutputIterator>
struct transform_output_iterator_base
{
    typedef thrust::iterator_adaptor
    <
        transform_output_iterator<UnaryFunction, OutputIterator>
      , OutputIterator
      , thrust::use_default
      , thrust::use_default
      , thrust::use_default
      , transform_output_iterator_proxy<UnaryFunction, OutputIterator>
    > type;
};

// Register transform_output_iterator_proxy with 'is_proxy_reference' from
// type_traits to enable its use with algorithms.
template <class UnaryFunction, class OutputIterator>
struct is_proxy_reference<
    transform_output_iterator_proxy<UnaryFunction, OutputIterator> >
    : public thrust::detail::true_type {};

} // end detail
THRUST_NAMESPACE_END

