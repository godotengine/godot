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


/*! \file internal_functional.inl
 *  \brief Non-public functionals used to implement algorithm internals.
 */

#pragma once

#include <thrust/tuple.h>
#include <thrust/iterator/iterator_traits.h>
#include <thrust/detail/config.h>
#include <thrust/detail/static_assert.h>
#include <thrust/detail/type_traits.h>
#include <thrust/iterator/detail/tuple_of_iterator_references.h>
#include <thrust/detail/raw_reference_cast.h>
#include <thrust/detail/memory_wrapper.h> // for ::new

THRUST_NAMESPACE_BEGIN

namespace detail
{

// unary_negate does not need to know argument_type
template<typename Predicate>
struct unary_negate
{
  typedef bool result_type;

  Predicate pred;

  __host__ __device__
  explicit unary_negate(const Predicate& pred) : pred(pred) {}

  template <typename T>
  __host__ __device__
  bool operator()(const T& x)
  {
    return !bool(pred(x));
  }
};

// binary_negate does not need to know first_argument_type or second_argument_type
template<typename Predicate>
struct binary_negate
{
  typedef bool result_type;

  Predicate pred;

  __host__ __device__
  explicit binary_negate(const Predicate& pred) : pred(pred) {}

  template <typename T1, typename T2>
  __host__ __device__
  bool operator()(const T1& x, const T2& y)
  {
    return !bool(pred(x,y));
  }
};

template<typename Predicate>
__host__ __device__
thrust::detail::unary_negate<Predicate> not1(const Predicate &pred)
{
  return thrust::detail::unary_negate<Predicate>(pred);
}

template<typename Predicate>
__host__ __device__
thrust::detail::binary_negate<Predicate> not2(const Predicate &pred)
{
  return thrust::detail::binary_negate<Predicate>(pred);
}


// convert a predicate to a 0 or 1 integral value
template<typename Predicate, typename IntegralType>
struct predicate_to_integral
{
  Predicate pred;

  __host__ __device__
  explicit predicate_to_integral(const Predicate& pred) : pred(pred) {}

  template <typename T>
  __host__ __device__
  IntegralType operator()(const T& x)
  {
    return pred(x) ? IntegralType(1) : IntegralType(0);
  }
};


// note that detail::equal_to does not force conversion from T2 -> T1 as equal_to does
template<typename T1>
struct equal_to
{
  typedef bool result_type;

  template <typename T2>
  __host__ __device__
  bool operator()(const T1& lhs, const T2& rhs) const
  {
    return lhs == rhs;
  }
};

// note that equal_to_value does not force conversion from T2 -> T1 as equal_to does
template<typename T2>
struct equal_to_value
{
  T2 rhs;

  __host__ __device__
  equal_to_value(const T2& rhs) : rhs(rhs) {}

  template <typename T1>
  __host__ __device__
  bool operator()(const T1& lhs) const
  {
    return lhs == rhs;
  }
};

template<typename Predicate>
struct tuple_binary_predicate
{
  typedef bool result_type;

  __host__ __device__
  tuple_binary_predicate(const Predicate& p) : pred(p) {}

  template<typename Tuple>
  __host__ __device__
  bool operator()(const Tuple& t) const
  {
    return pred(thrust::get<0>(t), thrust::get<1>(t));
  }

  mutable Predicate pred;
};

template<typename Predicate>
struct tuple_not_binary_predicate
{
  typedef bool result_type;

  __host__ __device__
  tuple_not_binary_predicate(const Predicate& p) : pred(p) {}

  template<typename Tuple>
  __host__ __device__
  bool operator()(const Tuple& t) const
  {
    return !pred(thrust::get<0>(t), thrust::get<1>(t));
  }

  mutable Predicate pred;
};

template<typename Generator>
  struct host_generate_functor
{
  typedef void result_type;

  __thrust_exec_check_disable__
  __host__ __device__
  host_generate_functor(Generator g)
    : gen(g) {}

  // operator() does not take an lvalue reference because some iterators
  // produce temporary proxy references when dereferenced. for example,
  // consider the temporary tuple of references produced by zip_iterator.
  // such temporaries cannot bind to an lvalue reference.
  //
  // to WAR this, accept a const reference (which is bindable to a temporary),
  // and const_cast in the implementation.
  //
  // XXX change to an rvalue reference upon c++0x (which either a named variable
  //     or temporary can bind to)
  template<typename T>
  __host__
  void operator()(const T &x)
  {
    // we have to be naughty and const_cast this to get it to work
    T &lvalue = const_cast<T&>(x);

    // this assigns correctly whether x is a true reference or proxy
    lvalue = gen();
  }

  Generator gen;
};

template<typename Generator>
  struct device_generate_functor
{
  typedef void result_type;

  __thrust_exec_check_disable__
  __host__ __device__
  device_generate_functor(Generator g)
    : gen(g) {}

  // operator() does not take an lvalue reference because some iterators
  // produce temporary proxy references when dereferenced. for example,
  // consider the temporary tuple of references produced by zip_iterator.
  // such temporaries cannot bind to an lvalue reference.
  //
  // to WAR this, accept a const reference (which is bindable to a temporary),
  // and const_cast in the implementation.
  //
  // XXX change to an rvalue reference upon c++0x (which either a named variable
  //     or temporary can bind to)
  template<typename T>
  __host__ __device__
  void operator()(const T &x)
  {
    // we have to be naughty and const_cast this to get it to work
    T &lvalue = const_cast<T&>(x);

    // this assigns correctly whether x is a true reference or proxy
    lvalue = gen();
  }

  Generator gen;
};

template<typename System, typename Generator>
  struct generate_functor
    : thrust::detail::eval_if<
        thrust::detail::is_convertible<System, thrust::host_system_tag>::value,
        thrust::detail::identity_<host_generate_functor<Generator> >,
        thrust::detail::identity_<device_generate_functor<Generator> >
      >
{};


template<typename ResultType, typename BinaryFunction>
  struct zipped_binary_op
{
  typedef ResultType result_type;

  __host__ __device__
  zipped_binary_op(BinaryFunction binary_op)
    : m_binary_op(binary_op) {}

  template<typename Tuple>
  __host__ __device__
  inline result_type operator()(Tuple t)
  {
    return m_binary_op(thrust::get<0>(t), thrust::get<1>(t));
  }

  BinaryFunction m_binary_op;
};


template<typename T>
  struct is_non_const_reference
    : thrust::detail::and_<
        thrust::detail::not_<thrust::detail::is_const<T> >,
        thrust::detail::or_<thrust::detail::is_reference<T>,
                            thrust::detail::is_proxy_reference<T> >
      >
{};

template<typename T> struct is_tuple_of_iterator_references : thrust::detail::false_type {};

template<typename... Ts>
  struct is_tuple_of_iterator_references<
    thrust::detail::tuple_of_iterator_references<
      Ts...
    >
  >
    : thrust::detail::true_type
{};

// use this enable_if to avoid assigning to temporaries in the transform functors below
// XXX revisit this problem with c++11 perfect forwarding
template<typename T>
  struct enable_if_non_const_reference_or_tuple_of_iterator_references
    : thrust::detail::enable_if<
        is_non_const_reference<T>::value || is_tuple_of_iterator_references<T>::value
      >
{};


template<typename UnaryFunction>
  struct unary_transform_functor
{
  typedef void result_type;

  UnaryFunction f;

  __host__ __device__
  unary_transform_functor(UnaryFunction f)
    : f(f)
  {}

  __thrust_exec_check_disable__
  template<typename Tuple>
  inline __host__ __device__
  typename enable_if_non_const_reference_or_tuple_of_iterator_references<
    typename thrust::tuple_element<1,Tuple>::type
  >::type
    operator()(Tuple t)
  {
    thrust::get<1>(t) = f(thrust::get<0>(t));
  }
};


template<typename BinaryFunction>
  struct binary_transform_functor
{
  BinaryFunction f;

  __host__ __device__
  binary_transform_functor(BinaryFunction f)
    : f(f)
  {}

  __thrust_exec_check_disable__
  template<typename Tuple>
  inline __host__ __device__
  typename enable_if_non_const_reference_or_tuple_of_iterator_references<
    typename thrust::tuple_element<2,Tuple>::type
  >::type
    operator()(Tuple t)
  {
    thrust::get<2>(t) = f(thrust::get<0>(t), thrust::get<1>(t));
  }
};


template<typename UnaryFunction, typename Predicate>
struct unary_transform_if_functor
{
  UnaryFunction unary_op;
  Predicate pred;

  __host__ __device__
  unary_transform_if_functor(UnaryFunction unary_op, Predicate pred)
    : unary_op(unary_op), pred(pred)
  {}

  __thrust_exec_check_disable__
  template<typename Tuple>
  inline __host__ __device__
  typename enable_if_non_const_reference_or_tuple_of_iterator_references<
    typename thrust::tuple_element<1,Tuple>::type
  >::type
    operator()(Tuple t)
  {
    if(pred(thrust::get<0>(t)))
    {
      thrust::get<1>(t) = unary_op(thrust::get<0>(t));
    }
  }
}; // end unary_transform_if_functor


template<typename UnaryFunction, typename Predicate>
struct unary_transform_if_with_stencil_functor
{
  UnaryFunction unary_op;
  Predicate pred;

  __host__ __device__
  unary_transform_if_with_stencil_functor(UnaryFunction unary_op, Predicate pred)
    : unary_op(unary_op), pred(pred)
  {}

  __thrust_exec_check_disable__
  template<typename Tuple>
  inline __host__ __device__
  typename enable_if_non_const_reference_or_tuple_of_iterator_references<
    typename thrust::tuple_element<2,Tuple>::type
  >::type
    operator()(Tuple t)
  {
    if(pred(thrust::get<1>(t)))
      thrust::get<2>(t) = unary_op(thrust::get<0>(t));
  }
}; // end unary_transform_if_with_stencil_functor


template<typename BinaryFunction, typename Predicate>
struct binary_transform_if_functor
{
  BinaryFunction binary_op;
  Predicate pred;

  __host__ __device__
  binary_transform_if_functor(BinaryFunction binary_op, Predicate pred)
    : binary_op(binary_op), pred(pred) {}

  __thrust_exec_check_disable__
  template<typename Tuple>
  inline __host__ __device__
  typename enable_if_non_const_reference_or_tuple_of_iterator_references<
    typename thrust::tuple_element<3,Tuple>::type
  >::type
    operator()(Tuple t)
  {
    if(pred(thrust::get<2>(t)))
      thrust::get<3>(t) = binary_op(thrust::get<0>(t), thrust::get<1>(t));
  }
}; // end binary_transform_if_functor


template<typename T>
  struct host_destroy_functor
{
  __host__
  void operator()(T &x) const
  {
    x.~T();
  } // end operator()()
}; // end host_destroy_functor


template<typename T>
  struct device_destroy_functor
{
  // add __host__ to allow the omp backend to compile with nvcc
  __host__ __device__
  void operator()(T &x) const
  {
    x.~T();
  } // end operator()()
}; // end device_destroy_functor


template<typename System, typename T>
  struct destroy_functor
    : thrust::detail::eval_if<
        thrust::detail::is_convertible<System, thrust::host_system_tag>::value,
        thrust::detail::identity_<host_destroy_functor<T> >,
        thrust::detail::identity_<device_destroy_functor<T> >
      >
{};


template <typename T>
struct fill_functor
{
  T exemplar;

  __thrust_exec_check_disable__
  __host__ __device__
  fill_functor(const T& _exemplar)
    : exemplar(_exemplar) {}

  __thrust_exec_check_disable__
  __host__ __device__
  fill_functor(const fill_functor & other)
    :exemplar(other.exemplar){}

  __thrust_exec_check_disable__
  __host__ __device__
  ~fill_functor() {}

  __thrust_exec_check_disable__
  __host__ __device__
  T operator()(void) const
  {
    return exemplar;
  }
};


template<typename T>
  struct uninitialized_fill_functor
{
  T exemplar;

  __thrust_exec_check_disable__
  __host__ __device__
  uninitialized_fill_functor(const T & x):exemplar(x){}

  __thrust_exec_check_disable__
  __host__ __device__
  uninitialized_fill_functor(const uninitialized_fill_functor & other)
    :exemplar(other.exemplar){}

  __thrust_exec_check_disable__
  __host__ __device__
  ~uninitialized_fill_functor() {}

  __thrust_exec_check_disable__
  __host__ __device__
  void operator()(T &x)
  {
    ::new(static_cast<void*>(&x)) T(exemplar);
  } // end operator()()
}; // end uninitialized_fill_functor


// this predicate tests two two-element tuples
// we first use a Compare for the first element
// if the first elements are equivalent, we use
// < for the second elements
template<typename Compare>
  struct compare_first_less_second
{
  compare_first_less_second(Compare c)
    : comp(c) {}

  template<typename T1, typename T2>
  __host__ __device__
  bool operator()(T1 lhs, T2 rhs)
  {
    return comp(thrust::get<0>(lhs), thrust::get<0>(rhs)) || (!comp(thrust::get<0>(rhs), thrust::get<0>(lhs)) && thrust::get<1>(lhs) < thrust::get<1>(rhs));
  }

  Compare comp;
}; // end compare_first_less_second


template<typename Compare>
  struct compare_first
{
  Compare comp;

  __host__ __device__
  compare_first(Compare comp)
    : comp(comp)
  {}

  template<typename Tuple1, typename Tuple2>
  __host__ __device__
  bool operator()(const Tuple1 &x, const Tuple2 &y)
  {
    return comp(thrust::raw_reference_cast(thrust::get<0>(x)), thrust::raw_reference_cast(thrust::get<0>(y)));
  }
}; // end compare_first


} // end namespace detail

THRUST_NAMESPACE_END
