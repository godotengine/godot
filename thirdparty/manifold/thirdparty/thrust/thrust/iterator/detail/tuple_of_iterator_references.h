/*
 *  Copyright 2008-2018 NVIDIA Corporation
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
#include <thrust/tuple.h>
#include <thrust/pair.h>
#include <thrust/detail/reference_forward_declaration.h>

THRUST_NAMESPACE_BEGIN
namespace detail
{

  
template<
  typename... Ts
>
  class tuple_of_iterator_references
    : public thrust::tuple<Ts...>
{
  private:
    typedef thrust::tuple<Ts...> super_t;

  public:
    // allow implicit construction from tuple<refs>
    inline __host__ __device__
    tuple_of_iterator_references(const super_t &other)
      : super_t(other)
    {}

    // allow assignment from tuples
    // XXX might be worthwhile to guard this with an enable_if is_assignable
    __thrust_exec_check_disable__
    template<typename... Us>
    inline __host__ __device__
    tuple_of_iterator_references &operator=(const thrust::tuple<Us...> &other)
    {
      super_t::operator=(other);
      return *this;
    }

    // allow assignment from pairs
    // XXX might be worthwhile to guard this with an enable_if is_assignable
    __thrust_exec_check_disable__
    template<typename U1, typename U2>
    inline __host__ __device__
    tuple_of_iterator_references &operator=(const thrust::pair<U1,U2> &other)
    {
      super_t::operator=(other);
      return *this;
    }

    // allow assignment from reference<tuple>
    // XXX perhaps we should generalize to reference<T>
    //     we could captures reference<pair> this way
    __thrust_exec_check_disable__
    template<typename Pointer, typename Derived,
             typename... Us>
    inline __host__ __device__
// XXX gcc-4.2 crashes on is_assignable
//    typename thrust::detail::enable_if<
//      thrust::detail::is_assignable<
//        super_t,
//        const thrust::tuple<Us...>
//      >::value,
//      tuple_of_iterator_references &
//    >::type
    tuple_of_iterator_references &
    operator=(const thrust::reference<thrust::tuple<Us...>, Pointer, Derived> &other)
    {
      typedef thrust::tuple<Us...> tuple_type;

      // XXX perhaps this could be accelerated
      tuple_type other_tuple = other;
      super_t::operator=(other_tuple);
      return *this;
    }


    // duplicate thrust::tuple's constructors
    inline __host__ __device__
    tuple_of_iterator_references() {}

    inline __host__ __device__
    tuple_of_iterator_references(typename access_traits<Ts>::parameter_type... ts)
      : super_t(ts...)
    {}
};


// this overload of swap() permits swapping tuple_of_iterator_references returned as temporaries from
// iterator dereferences
template<
  typename... Ts,
  typename... Us
>
inline __host__ __device__
void swap(tuple_of_iterator_references<Ts...> x,
          tuple_of_iterator_references<Us...> y)
{
  x.swap(y);
}


} // end detail

// define tuple_size, tuple_element, etc.
template<class... Ts>
struct tuple_size<detail::tuple_of_iterator_references<Ts...>>
  : std::integral_constant<size_t, sizeof...(Ts)>
{};

template<size_t i>
struct tuple_element<i, detail::tuple_of_iterator_references<>> {};


template<class T, class... Ts>
struct tuple_element<0, detail::tuple_of_iterator_references<T,Ts...>>
{
  using type = T;
};


template<size_t i, class T, class... Ts>
struct tuple_element<i, detail::tuple_of_iterator_references<T,Ts...>>
{
  using type = typename tuple_element<i - 1, detail::tuple_of_iterator_references<Ts...>>::type;
};


THRUST_NAMESPACE_END

