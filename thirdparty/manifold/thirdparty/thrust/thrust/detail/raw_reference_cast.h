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
#include <thrust/detail/raw_pointer_cast.h>
#include <thrust/detail/type_traits/has_nested_type.h>
#include <thrust/detail/type_traits.h>
#include <thrust/detail/tuple_transform.h>
#include <thrust/iterator/detail/tuple_of_iterator_references.h>


// the order of declarations and definitions in this file is totally goofy
// this header defines raw_reference_cast, which has a few overloads towards the bottom of the file
// raw_reference_cast depends on metafunctions such as is_unwrappable and raw_reference
// we need to be sure that these metafunctions are completely defined (including specializations) before they are instantiated by raw_reference_cast

THRUST_NAMESPACE_BEGIN
namespace detail
{


__THRUST_DEFINE_HAS_NESTED_TYPE(is_wrapped_reference, wrapped_reference_hint)


// wrapped reference-like things which aren't strictly wrapped references
// (e.g. tuples of wrapped references) are considered unwrappable
template<typename T>
  struct is_unwrappable
    : is_wrapped_reference<T>
{};


// specialize is_unwrappable
// a tuple is_unwrappable if any of its elements is_unwrappable
template<typename... Ts>
  struct is_unwrappable<
    thrust::tuple<Ts...>
  >
    : or_<
        is_unwrappable<Ts>...
      >
{};


// specialize is_unwrappable
// a tuple_of_iterator_references is_unwrappable if any of its elements is_unwrappable
template<
  typename... Ts
>
  struct is_unwrappable<
    thrust::detail::tuple_of_iterator_references<Ts...>
  >
    : or_<
        is_unwrappable<Ts>...
      >
{};


template<typename T, typename Result = void>
  struct enable_if_unwrappable
    : enable_if<
        is_unwrappable<T>::value,
        Result
      >
{};


namespace raw_reference_detail
{


template<typename T, typename Enable = void>
  struct raw_reference_impl
    : add_reference<T>
{};


template<typename T>
  struct raw_reference_impl<
    T,
    typename thrust::detail::enable_if<
      is_wrapped_reference<
        typename remove_cv<T>::type
      >::value
    >::type
  >
{
  typedef typename add_reference<
    typename pointer_element<typename T::pointer>::type
  >::type type;
};


} // end raw_reference_detail


template<typename T>
  struct raw_reference :
    raw_reference_detail::raw_reference_impl<T>
{};


namespace raw_reference_detail
{

// unlike raw_reference,
// raw_reference_tuple_helper needs to return a value
// when it encounters one, rather than a reference
// upon encountering tuple, recurse
//
// we want the following behavior:
//  1. T                                -> T
//  2. T&                               -> T&
//  3. null_type                        -> null_type
//  4. reference<T>                     -> T&
//  5. tuple_of_iterator_references<T>  -> tuple_of_iterator_references<raw_reference_tuple_helper<T>::type>


// wrapped references are unwrapped using raw_reference, otherwise, return T
template<typename T>
  struct raw_reference_tuple_helper
    : eval_if<
        is_unwrappable<
          typename remove_cv<T>::type
        >::value,
        raw_reference<T>,
        identity_<T>
      >
{};


// recurse on tuples
template <
  typename... Ts
>
  struct raw_reference_tuple_helper<
    thrust::tuple<Ts...>
  >
{
  typedef thrust::tuple<
    typename raw_reference_tuple_helper<Ts>::type...
  > type;
};


template <
  typename... Ts
>
  struct raw_reference_tuple_helper<
    thrust::detail::tuple_of_iterator_references<Ts...>
  >
{
  typedef thrust::detail::tuple_of_iterator_references<
    typename raw_reference_tuple_helper<Ts>::type...
  > type;
};


} // end raw_reference_detail


// a couple of specializations of raw_reference for tuples follow


// if a tuple "tuple_type" is_unwrappable,
//   then the raw_reference of tuple_type is a tuple of its members' raw_references
//   else the raw_reference of tuple_type is tuple_type &
template <
  typename... Ts
>
  struct raw_reference<
    thrust::tuple<Ts...>
  >
{
  private:
    typedef thrust::tuple<Ts...> tuple_type;

  public:
    typedef typename eval_if<
      is_unwrappable<tuple_type>::value,
      raw_reference_detail::raw_reference_tuple_helper<tuple_type>,
      add_reference<tuple_type>
    >::type type;
};


template <
  typename... Ts
>
  struct raw_reference<
    thrust::detail::tuple_of_iterator_references<Ts...>
  >
{
  private:
    typedef detail::tuple_of_iterator_references<Ts...> tuple_type;

  public:
    typedef typename raw_reference_detail::raw_reference_tuple_helper<tuple_type>::type type;

    // XXX figure out why is_unwrappable seems to be broken for tuple_of_iterator_references
    //typedef typename eval_if<
    //  is_unwrappable<tuple_type>::value,
    //  raw_reference_detail::raw_reference_tuple_helper<tuple_type>,
    //  add_reference<tuple_type>
    //>::type type;
};


} // end detail


// provide declarations of raw_reference_cast's overloads for raw_reference_caster below
template<typename T>
__host__ __device__
typename detail::raw_reference<T>::type
  raw_reference_cast(T &ref);


template<typename T>
__host__ __device__
typename detail::raw_reference<const T>::type
  raw_reference_cast(const T &ref);


template<
  typename... Ts
>
__host__ __device__
typename detail::enable_if_unwrappable<
  thrust::detail::tuple_of_iterator_references<Ts...>,
  typename detail::raw_reference<
    thrust::detail::tuple_of_iterator_references<Ts...>
  >::type
>::type
raw_reference_cast(thrust::detail::tuple_of_iterator_references<Ts...> t);


namespace detail
{


struct raw_reference_caster
{
  template<typename T>
  __host__ __device__
  typename detail::raw_reference<T>::type operator()(T &ref)
  {
    return thrust::raw_reference_cast(ref);
  }

  template<typename T>
  __host__ __device__
  typename detail::raw_reference<const T>::type operator()(const T &ref)
  {
    return thrust::raw_reference_cast(ref);
  }

  template<
    typename... Ts
  >
  __host__ __device__
  typename detail::raw_reference<
    thrust::detail::tuple_of_iterator_references<Ts...>
  >::type
  operator()(thrust::detail::tuple_of_iterator_references<Ts...> t,
             typename enable_if<
               is_unwrappable<thrust::detail::tuple_of_iterator_references<Ts...> >::value
             >::type * = 0)
  {
    return thrust::raw_reference_cast(t);
  }
}; // end raw_reference_caster


} // end detail


template<typename T>
__host__ __device__
typename detail::raw_reference<T>::type
  raw_reference_cast(T &ref)
{
  return *thrust::raw_pointer_cast(&ref);
} // end raw_reference_cast


template<typename T>
__host__ __device__
typename detail::raw_reference<const T>::type
  raw_reference_cast(const T &ref)
{
  return *thrust::raw_pointer_cast(&ref);
} // end raw_reference_cast


template<
  typename... Ts
>
__host__ __device__
typename detail::enable_if_unwrappable<
  thrust::detail::tuple_of_iterator_references<Ts...>,
  typename detail::raw_reference<
    thrust::detail::tuple_of_iterator_references<Ts...>
  >::type
>::type
raw_reference_cast(thrust::detail::tuple_of_iterator_references<Ts...> t)
{
  thrust::detail::raw_reference_caster f;

  // note that we pass raw_reference_tuple_helper, not raw_reference as the unary metafunction
  // the different way that raw_reference_tuple_helper unwraps tuples is important
  return thrust::detail::tuple_host_device_transform<detail::raw_reference_detail::raw_reference_tuple_helper>(t, f);
} // end raw_reference_cast


THRUST_NAMESPACE_END

