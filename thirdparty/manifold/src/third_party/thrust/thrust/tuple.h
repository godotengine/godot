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


/*! \file tuple.h
 *  \brief A type encapsulating a heterogeneous collection of elements.
 */

/*
 * Copyright (C) 1999, 2000 Jaakko Järvi (jaakko.jarvi@cs.utu.fi)
 *
 * Distributed under the Boost Software License, Version 1.0.
 * (See accompanying NOTICE file for the complete license)
 *
 * For more information, see http://www.boost.org
 */

#pragma once

#include <thrust/detail/config.h>
#include <thrust/detail/tuple.inl>
#include <thrust/pair.h>

THRUST_NAMESPACE_BEGIN

/*! \addtogroup utility
 *  \{
 */

/*! \addtogroup tuple
 *  \{
 */

/*! \cond
 */

struct null_type;

/*! \endcond
 */

/*! This metafunction returns the type of a
 *  \p tuple's <tt>N</tt>th element.
 *
 *  \tparam N This parameter selects the element of interest.
 *  \tparam T A \c tuple type of interest.
 *
 *  \see pair
 *  \see tuple
 */
template <size_t N, class T> struct tuple_element;

/*! This metafunction returns the number of elements
 *  of a \p tuple type of interest.
 *
 *  \tparam T A \c tuple type of interest.
 *
 *  \see pair
 *  \see tuple
 */
template <class T> struct tuple_size;


// get function for non-const cons-lists, returns a reference to the element

/*! The \p get function returns a reference to a \p tuple element of
 *  interest.
 *
 *  \param t A reference to a \p tuple of interest.
 *  \return A reference to \p t's <tt>N</tt>th element.
 *
 *  \tparam N The index of the element of interest.
 *
 *  The following code snippet demonstrates how to use \p get to print
 *  the value of a \p tuple element.
 *
 *  \code
 *  #include <thrust/tuple.h>
 *  #include <iostream>
 *  ...
 *  thrust::tuple<int, const char *> t(13, "thrust");
 *
 *  std::cout << "The 1st value of t is " << thrust::get<0>(t) << std::endl;
 *  \endcode
 *
 *  \see pair
 *  \see tuple
 */
template<int N, class HT, class TT>
__host__ __device__
inline typename access_traits<
                  typename tuple_element<N, detail::cons<HT, TT> >::type
                >::non_const_type
get(detail::cons<HT, TT>& t);


/*! The \p get function returns a \c const reference to a \p tuple element of
 *  interest.
 *
 *  \param t A reference to a \p tuple of interest.
 *  \return A \c const reference to \p t's <tt>N</tt>th element.
 *
 *  \tparam N The index of the element of interest.
 *
 *  The following code snippet demonstrates how to use \p get to print
 *  the value of a \p tuple element.
 *
 *  \code
 *  #include <thrust/tuple.h>
 *  #include <iostream>
 *  ...
 *  thrust::tuple<int, const char *> t(13, "thrust");
 *
 *  std::cout << "The 1st value of t is " << thrust::get<0>(t) << std::endl;
 *  \endcode
 *
 *  \see pair
 *  \see tuple
 */
template<int N, class HT, class TT>
__host__ __device__
inline typename access_traits<
                  typename tuple_element<N, detail::cons<HT, TT> >::type
                >::const_type
get(const detail::cons<HT, TT>& t);



/*! \brief \p tuple is a class template that can be instantiated with up to ten
 *  arguments. Each template argument specifies the type of element in the \p
 *  tuple. Consequently, tuples are heterogeneous, fixed-size collections of
 *  values. An instantiation of \p tuple with two arguments is similar to an
 *  instantiation of \p pair with the same two arguments. Individual elements
 *  of a \p tuple may be accessed with the \p get function.
 *
 *  \tparam TN The type of the <tt>N</tt> \c tuple element. Thrust's \p tuple
 *          type currently supports up to ten elements.
 *
 *  The following code snippet demonstrates how to create a new \p tuple object
 *  and inspect and modify the value of its elements.
 *
 *  \code
 *  #include <thrust/tuple.h>
 *  #include <iostream>
 *  
 *  int main() {
 *    // Create a tuple containing an `int`, a `float`, and a string.
 *    thrust::tuple<int, float, const char*> t(13, 0.1f, "thrust");
 *
 *    // Individual members are accessed with the free function `get`.
 *    std::cout << "The first element's value is " << thrust::get<0>(t) << std::endl;
 *
 *    // ... or the member function `get`.
 *    std::cout << "The second element's value is " << t.get<1>() << std::endl;
 *
 *    // We can also modify elements with the same function.
 *    thrust::get<0>(t) += 10;
 *  }
 *  \endcode
 *
 *  \see pair
 *  \see get
 *  \see make_tuple
 *  \see tuple_element
 *  \see tuple_size
 *  \see tie
 */
template <class T0, class T1, class T2, class T3, class T4,
          class T5, class T6, class T7, class T8, class T9>
  class tuple
  /*! \cond
   */
    : public detail::map_tuple_to_cons<T0, T1, T2, T3, T4, T5, T6, T7, T8, T9>::type
  /*! \endcond
   */
{
  /*! \cond
   */

  private:
  typedef typename detail::map_tuple_to_cons<T0, T1, T2, T3, T4, T5, T6, T7, T8, T9>::type inherited;

  /*! \endcond
   */

  public:

  /*! \p tuple's no-argument constructor initializes each element.
   */
  inline __host__ __device__
  tuple(void) {}

  /*! \p tuple's one-argument constructor copy constructs the first element from the given parameter
   *     and intializes all other elements.
   *  \param t0 The value to assign to this \p tuple's first element.
   */
  inline __host__ __device__
  tuple(typename access_traits<T0>::parameter_type t0)
    : inherited(t0,
                static_cast<const null_type&>(null_type()),
                static_cast<const null_type&>(null_type()),
                static_cast<const null_type&>(null_type()),
                static_cast<const null_type&>(null_type()),
                static_cast<const null_type&>(null_type()),
                static_cast<const null_type&>(null_type()),
                static_cast<const null_type&>(null_type()),
                static_cast<const null_type&>(null_type()),
                static_cast<const null_type&>(null_type())) {}

  /*! \p tuple's one-argument constructor copy constructs the first two elements from the given parameters
   *     and intializes all other elements.
   *  \param t0 The value to assign to this \p tuple's first element.
   *  \param t1 The value to assign to this \p tuple's second element.
   *  \note \p tuple's constructor has ten variants of this form, the rest of which are ommitted here for brevity.
   */
  inline __host__ __device__
  tuple(typename access_traits<T0>::parameter_type t0,
        typename access_traits<T1>::parameter_type t1)
    : inherited(t0, t1,
                static_cast<const null_type&>(null_type()),
                static_cast<const null_type&>(null_type()),
                static_cast<const null_type&>(null_type()),
                static_cast<const null_type&>(null_type()),
                static_cast<const null_type&>(null_type()),
                static_cast<const null_type&>(null_type()),
                static_cast<const null_type&>(null_type()),
                static_cast<const null_type&>(null_type())) {}

  /*! \cond
   */

  inline __host__ __device__
  tuple(typename access_traits<T0>::parameter_type t0,
        typename access_traits<T1>::parameter_type t1,
        typename access_traits<T2>::parameter_type t2)
    : inherited(t0, t1, t2,
                static_cast<const null_type&>(null_type()),
                static_cast<const null_type&>(null_type()),
                static_cast<const null_type&>(null_type()),
                static_cast<const null_type&>(null_type()),
                static_cast<const null_type&>(null_type()),
                static_cast<const null_type&>(null_type()),
                static_cast<const null_type&>(null_type())) {}

  inline __host__ __device__
  tuple(typename access_traits<T0>::parameter_type t0,
        typename access_traits<T1>::parameter_type t1,
        typename access_traits<T2>::parameter_type t2,
        typename access_traits<T3>::parameter_type t3)
    : inherited(t0, t1, t2, t3,
                static_cast<const null_type&>(null_type()),
                static_cast<const null_type&>(null_type()),
                static_cast<const null_type&>(null_type()),
                static_cast<const null_type&>(null_type()),
                static_cast<const null_type&>(null_type()),
                static_cast<const null_type&>(null_type())) {}

  inline __host__ __device__
  tuple(typename access_traits<T0>::parameter_type t0,
        typename access_traits<T1>::parameter_type t1,
        typename access_traits<T2>::parameter_type t2,
        typename access_traits<T3>::parameter_type t3,
        typename access_traits<T4>::parameter_type t4)
    : inherited(t0, t1, t2, t3, t4,
                static_cast<const null_type&>(null_type()),
                static_cast<const null_type&>(null_type()),
                static_cast<const null_type&>(null_type()),
                static_cast<const null_type&>(null_type()),
                static_cast<const null_type&>(null_type())) {}

  inline __host__ __device__
  tuple(typename access_traits<T0>::parameter_type t0,
        typename access_traits<T1>::parameter_type t1,
        typename access_traits<T2>::parameter_type t2,
        typename access_traits<T3>::parameter_type t3,
        typename access_traits<T4>::parameter_type t4,
        typename access_traits<T5>::parameter_type t5)
    : inherited(t0, t1, t2, t3, t4, t5,
                static_cast<const null_type&>(null_type()),
                static_cast<const null_type&>(null_type()),
                static_cast<const null_type&>(null_type()),
                static_cast<const null_type&>(null_type())) {}

  inline __host__ __device__
  tuple(typename access_traits<T0>::parameter_type t0,
        typename access_traits<T1>::parameter_type t1,
        typename access_traits<T2>::parameter_type t2,
        typename access_traits<T3>::parameter_type t3,
        typename access_traits<T4>::parameter_type t4,
        typename access_traits<T5>::parameter_type t5,
        typename access_traits<T6>::parameter_type t6)
    : inherited(t0, t1, t2, t3, t4, t5, t6,
                static_cast<const null_type&>(null_type()),
                static_cast<const null_type&>(null_type()),
                static_cast<const null_type&>(null_type())) {}

  inline __host__ __device__
  tuple(typename access_traits<T0>::parameter_type t0,
        typename access_traits<T1>::parameter_type t1,
        typename access_traits<T2>::parameter_type t2,
        typename access_traits<T3>::parameter_type t3,
        typename access_traits<T4>::parameter_type t4,
        typename access_traits<T5>::parameter_type t5,
        typename access_traits<T6>::parameter_type t6,
        typename access_traits<T7>::parameter_type t7)
    : inherited(t0, t1, t2, t3, t4, t5, t6, t7,
                static_cast<const null_type&>(null_type()),
                static_cast<const null_type&>(null_type())) {}

  inline __host__ __device__
  tuple(typename access_traits<T0>::parameter_type t0,
        typename access_traits<T1>::parameter_type t1,
        typename access_traits<T2>::parameter_type t2,
        typename access_traits<T3>::parameter_type t3,
        typename access_traits<T4>::parameter_type t4,
        typename access_traits<T5>::parameter_type t5,
        typename access_traits<T6>::parameter_type t6,
        typename access_traits<T7>::parameter_type t7,
        typename access_traits<T8>::parameter_type t8)
    : inherited(t0, t1, t2, t3, t4, t5, t6, t7, t8,
                static_cast<const null_type&>(null_type())) {}

  inline __host__ __device__
  tuple(typename access_traits<T0>::parameter_type t0,
        typename access_traits<T1>::parameter_type t1,
        typename access_traits<T2>::parameter_type t2,
        typename access_traits<T3>::parameter_type t3,
        typename access_traits<T4>::parameter_type t4,
        typename access_traits<T5>::parameter_type t5,
        typename access_traits<T6>::parameter_type t6,
        typename access_traits<T7>::parameter_type t7,
        typename access_traits<T8>::parameter_type t8,
        typename access_traits<T9>::parameter_type t9)
    : inherited(t0, t1, t2, t3, t4, t5, t6, t7, t8, t9) {}


  template<class U1, class U2>
  inline __host__ __device__
  tuple(const detail::cons<U1, U2>& p) : inherited(p) {}

  __thrust_exec_check_disable__
  template <class U1, class U2>
  inline __host__ __device__
  tuple& operator=(const detail::cons<U1, U2>& k)
  {
    inherited::operator=(k);
    return *this;
  }

  /*! \endcond
   */

  /*! This assignment operator allows assigning the first two elements of this \p tuple from a \p pair.
   *  \param k A \p pair to assign from.
   */
  __thrust_exec_check_disable__
  template <class U1, class U2>
  __host__ __device__ inline
  tuple& operator=(const thrust::pair<U1, U2>& k) {
    //BOOST_STATIC_ASSERT(length<tuple>::value == 2);// check_length = 2
    this->head = k.first;
    this->tail.head = k.second;
    return *this;
  }

  /*! \p swap swaps the elements of two <tt>tuple</tt>s.
   *
   *  \param t The other <tt>tuple</tt> with which to swap.
   */
  inline __host__ __device__
  void swap(tuple &t)
  {
    inherited::swap(t);
  }
};

/*! \cond
 */

template <>
class tuple<null_type, null_type, null_type, null_type, null_type, null_type, null_type, null_type, null_type, null_type>  :
  public null_type
{
public:
  typedef null_type inherited;
};

/*! \endcond
 */


/*! This version of \p make_tuple creates a new \c tuple object from a
 *  single object.
 *
 *  \param t0 The object to copy from.
 *  \return A \p tuple object with a single member which is a copy of \p t0.
 */
template<class T0>
__host__ __device__ inline
  typename detail::make_tuple_mapper<T0>::type
    make_tuple(const T0& t0);

/*! This version of \p make_tuple creates a new \c tuple object from two
 *  objects.
 *
 *  \param t0 The first object to copy from.
 *  \param t1 The second object to copy from.
 *  \return A \p tuple object with two members which are copies of \p t0
 *          and \p t1.
 *
 *  \note \p make_tuple has ten variants, the rest of which are omitted here
 *        for brevity.
 */
template<class T0, class T1>
__host__ __device__ inline
  typename detail::make_tuple_mapper<T0, T1>::type
    make_tuple(const T0& t0, const T1& t1);

/*! This version of \p tie creates a new \c tuple whose single element is
 *  a reference which refers to this function's argument.
 *
 *  \param t0 The object to reference.
 *  \return A \p tuple object with one member which is a reference to \p t0.
 */
template<typename T0>
__host__ __device__ inline
tuple<T0&> tie(T0& t0);

/*! This version of \p tie creates a new \c tuple of references object which
 *  refers to this function's arguments.
 *
 *  \param t0 The first object to reference.
 *  \param t1 The second object to reference.
 *  \return A \p tuple object with two members which are references to \p t0
 *          and \p t1.
 *
 *  \note \p tie has ten variants, the rest of which are omitted here for
 *           brevity.
 */
template<typename T0, typename T1>
__host__ __device__ inline
tuple<T0&,T1&> tie(T0& t0, T1& t1);

/*! \p swap swaps the contents of two <tt>tuple</tt>s.
 *
 *  \param x The first \p tuple to swap.
 *  \param y The second \p tuple to swap.
 */
template<
  typename T0, typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7, typename T8, typename T9,
  typename U0, typename U1, typename U2, typename U3, typename U4, typename U5, typename U6, typename U7, typename U8, typename U9
>
inline __host__ __device__
void swap(tuple<T0,T1,T2,T3,T4,T5,T6,T7,T8,T9> &x,
          tuple<U0,U1,U2,U3,U4,U5,U6,U7,U8,U9> &y);



/*! \cond
 */

template<class T0, class T1, class T2>
__host__ __device__ inline
  typename detail::make_tuple_mapper<T0, T1, T2>::type
    make_tuple(const T0& t0, const T1& t1, const T2& t2);

template<class T0, class T1, class T2, class T3>
__host__ __device__ inline
  typename detail::make_tuple_mapper<T0, T1, T2, T3>::type
    make_tuple(const T0& t0, const T1& t1, const T2& t2, const T3& t3);

template<class T0, class T1, class T2, class T3, class T4>
__host__ __device__ inline
  typename detail::make_tuple_mapper<T0, T1, T2, T3, T4>::type
    make_tuple(const T0& t0, const T1& t1, const T2& t2, const T3& t3, const T4& t4);

template<class T0, class T1, class T2, class T3, class T4, class T5>
__host__ __device__ inline
  typename detail::make_tuple_mapper<T0, T1, T2, T3, T4, T5>::type
    make_tuple(const T0& t0, const T1& t1, const T2& t2, const T3& t3, const T4& t4, const T5& t5);

template<class T0, class T1, class T2, class T3, class T4, class T5, class T6>
__host__ __device__ inline
  typename detail::make_tuple_mapper<T0, T1, T2, T3, T4, T5, T6>::type
    make_tuple(const T0& t0, const T1& t1, const T2& t2, const T3& t3, const T4& t4, const T5& t5, const T6& t6);

template<class T0, class T1, class T2, class T3, class T4, class T5, class T6, class T7>
__host__ __device__ inline
  typename detail::make_tuple_mapper<T0, T1, T2, T3, T4, T5, T6, T7>::type
    make_tuple(const T0& t0, const T1& t1, const T2& t2, const T3& t3, const T4& t4, const T5& t5, const T6& t6, const T7& t7);

template<class T0, class T1, class T2, class T3, class T4, class T5, class T6, class T7, class T8>
__host__ __device__ inline
  typename detail::make_tuple_mapper<T0, T1, T2, T3, T4, T5, T6, T7, T8>::type
    make_tuple(const T0& t0, const T1& t1, const T2& t2, const T3& t3, const T4& t4, const T5& t5, const T6& t6, const T7& t7, const T8& t8);

template<class T0, class T1, class T2, class T3, class T4, class T5, class T6, class T7, class T8, class T9>
__host__ __device__ inline
  typename detail::make_tuple_mapper<T0, T1, T2, T3, T4, T5, T6, T7, T8, T9>::type
    make_tuple(const T0& t0, const T1& t1, const T2& t2, const T3& t3, const T4& t4, const T5& t5, const T6& t6, const T7& t7, const T8& t8, const T9& t9);

template<typename T0, typename T1, typename T2>
__host__ __device__ inline
tuple<T0&,T1&,T2&> tie(T0 &t0, T1 &t1, T2 &t2);

template<typename T0, typename T1, typename T2, typename T3>
__host__ __device__ inline
tuple<T0&,T1&,T2&,T3&> tie(T0 &t0, T1 &t1, T2 &t2, T3 &t3);

template<typename T0, typename T1, typename T2, typename T3, typename T4>
__host__ __device__ inline
tuple<T0&,T1&,T2&,T3&,T4&> tie(T0 &t0, T1 &t1, T2 &t2, T3 &t3, T4 &t4);

template<typename T0, typename T1, typename T2, typename T3, typename T4, typename T5>
__host__ __device__ inline
tuple<T0&,T1&,T2&,T3&,T4&,T5&> tie(T0 &t0, T1 &t1, T2 &t2, T3 &t3, T4 &t4, T5 &t5);

template<typename T0, typename T1, typename T2, typename T3, typename T4, typename T5, typename T6>
__host__ __device__ inline
tuple<T0&,T1&,T2&,T3&,T4&,T5&,T6&> tie(T0 &t0, T1 &t1, T2 &t2, T3 &t3, T4 &t4, T5 &t5, T6 &t6);

template<typename T0, typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7>
__host__ __device__ inline
tuple<T0&,T1&,T2&,T3&,T4&,T5&,T6&,T7&> tie(T0 &t0, T1 &t1, T2 &t2, T3 &t3, T4 &t4, T5 &t5, T6 &t6, T7 &t7);

template<typename T0, typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7, typename T8>
__host__ __device__ inline
tuple<T0&,T1&,T2&,T3&,T4&,T5&,T6&,T7&,T8&> tie(T0 &t0, T1 &t1, T2 &t2, T3 &t3, T4 &t4, T5 &t5, T6 &t6, T7 &t7, T8 &t8);

template<typename T0, typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7, typename T8, typename T9>
__host__ __device__ inline
tuple<T0&,T1&,T2&,T3&,T4&,T5&,T6&,T7&,T8&,T9&> tie(T0 &t0, T1 &t1, T2 &t2, T3 &t3, T4 &t4, T5 &t5, T6 &t6, T7 &t7, T8 &t8, T9 &t9);


__host__ __device__ inline
bool operator==(const null_type&, const null_type&);

__host__ __device__ inline
bool operator>=(const null_type&, const null_type&);

__host__ __device__ inline
bool operator<=(const null_type&, const null_type&);

__host__ __device__ inline
bool operator!=(const null_type&, const null_type&);

__host__ __device__ inline
bool operator<(const null_type&, const null_type&);

__host__ __device__ inline
bool operator>(const null_type&, const null_type&);

/*! \endcond
 */

/*! \} // tuple
 */

/*! \} // utility
 */

THRUST_NAMESPACE_END
