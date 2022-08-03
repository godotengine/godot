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
#include <thrust/functional.h>
#include <thrust/system/detail/generic/replace.h>
#include <thrust/transform.h>
#include <thrust/replace.h>

THRUST_NAMESPACE_BEGIN
namespace system
{
namespace detail
{
namespace generic
{
namespace detail
{


// this functor receives x, and returns a new_value if predicate(x) is true; otherwise,
// it returns x
template<typename Predicate, typename NewType, typename OutputType>
  struct new_value_if
{
  __host__ __device__
  new_value_if(Predicate p, NewType nv):pred(p),new_value(nv){}

  template<typename InputType>
  __host__ __device__
  OutputType operator()(const InputType &x) const
  {
    return pred(x) ? new_value : x;
  } // end operator()()

  // this version of operator()() works like the previous but
  // feeds its second argument to pred
  template<typename InputType, typename PredicateArgumentType>
  __host__ __device__
  OutputType operator()(const InputType &x, const PredicateArgumentType &y)
  {
    return pred(y) ? new_value : x;
  } // end operator()()

  Predicate pred;
  NewType new_value;
}; // end new_value_if


// this unary functor ignores its argument and returns a constant
template<typename T>
  struct constant_unary
{
  __host__ __device__
  constant_unary(T _c):c(_c){}

  template<typename U>
  __host__ __device__
  T operator()(U &)
  {
    return c;
  } // end operator()()

  T c;
}; // end constant_unary


} // end detail


template<typename DerivedPolicy, typename InputIterator, typename OutputIterator, typename Predicate, typename T>
__host__ __device__
  OutputIterator replace_copy_if(thrust::execution_policy<DerivedPolicy> &exec,
                                 InputIterator first,
                                 InputIterator last,
                                 OutputIterator result,
                                 Predicate pred,
                                 const T &new_value)
{
  typedef typename thrust::iterator_traits<OutputIterator>::value_type OutputType;

  detail::new_value_if<Predicate,T,OutputType> op(pred,new_value);
  return thrust::transform(exec, first, last, result, op);
} // end replace_copy_if()


template<typename DerivedPolicy, typename InputIterator1, typename InputIterator2, typename OutputIterator, typename Predicate, typename T>
__host__ __device__
  OutputIterator replace_copy_if(thrust::execution_policy<DerivedPolicy> &exec,
                                 InputIterator1 first,
                                 InputIterator1 last,
                                 InputIterator2 stencil,
                                 OutputIterator result,
                                 Predicate pred,
                                 const T &new_value)
{
  typedef typename thrust::iterator_traits<OutputIterator>::value_type OutputType;

  detail::new_value_if<Predicate,T,OutputType> op(pred,new_value);
  return thrust::transform(exec, first, last, stencil, result, op);
} // end replace_copy_if()


template<typename DerivedPolicy, typename InputIterator, typename OutputIterator, typename T>
__host__ __device__
  OutputIterator replace_copy(thrust::execution_policy<DerivedPolicy> &exec,
                              InputIterator first,
                              InputIterator last,
                              OutputIterator result,
                              const T &old_value,
                              const T &new_value)
{
  using thrust::placeholders::_1;

  return thrust::replace_copy_if(exec, first, last, result, _1 == old_value, new_value);
} // end replace_copy()


template<typename DerivedPolicy, typename ForwardIterator, typename Predicate, typename T>
__host__ __device__
  void replace_if(thrust::execution_policy<DerivedPolicy> &exec,
                  ForwardIterator first,
                  ForwardIterator last,
                  Predicate pred,
                  const T &new_value)
{
  detail::constant_unary<T> f(new_value);
  thrust::transform_if(exec, first, last, first, first, f, pred);
} // end replace_if()


template<typename DerivedPolicy, typename ForwardIterator, typename InputIterator, typename Predicate, typename T>
__host__ __device__
  void replace_if(thrust::execution_policy<DerivedPolicy> &exec,
                  ForwardIterator first,
                  ForwardIterator last,
                  InputIterator stencil,
                  Predicate pred,
                  const T &new_value)
{
  detail::constant_unary<T> f(new_value);
  thrust::transform_if(exec, first, last, stencil, first, f, pred);
} // end replace_if()


template<typename DerivedPolicy, typename ForwardIterator, typename T>
__host__ __device__
  void replace(thrust::execution_policy<DerivedPolicy> &exec,
               ForwardIterator first,
               ForwardIterator last,
               const T &old_value,
               const T &new_value)
{
  using thrust::placeholders::_1;

  return thrust::replace_if(exec, first, last, _1 == old_value, new_value);
} // end replace()


} // end namespace generic
} // end namespace detail
} // end namespace system
THRUST_NAMESPACE_END

