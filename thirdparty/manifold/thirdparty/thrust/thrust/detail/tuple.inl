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

#include <thrust/detail/type_traits.h>
#include <thrust/detail/swap.h>

THRUST_NAMESPACE_BEGIN

// define null_type
struct null_type {};

// null_type comparisons
__host__ __device__ inline
bool operator==(const null_type&, const null_type&) { return true; }

__host__ __device__ inline
bool operator>=(const null_type&, const null_type&) { return true; }

__host__ __device__ inline
bool operator<=(const null_type&, const null_type&) { return true; }

__host__ __device__ inline
bool operator!=(const null_type&, const null_type&) { return false; }

__host__ __device__ inline
bool operator<(const null_type&, const null_type&) { return false; }

__host__ __device__ inline
bool operator>(const null_type&, const null_type&) { return false; }

// forward declaration for tuple
template <
  class T0 = null_type, class T1 = null_type, class T2 = null_type,
  class T3 = null_type, class T4 = null_type, class T5 = null_type,
  class T6 = null_type, class T7 = null_type, class T8 = null_type,
  class T9 = null_type>
class tuple;


template <size_t N, class T> struct tuple_element;

template<size_t N, class T>
  struct tuple_element_impl
{
  private:
    typedef typename T::tail_type Next;

  public:
    /*! The result of this metafunction is returned in \c type.
     */
    typedef typename tuple_element_impl<N-1, Next>::type type;
}; // end tuple_element

template<class T>
  struct tuple_element_impl<0,T>
{
  typedef typename T::head_type type;
};

template <size_t N, class T>
  struct tuple_element<N, T const>
{
    using type = typename std::add_const<typename tuple_element<N, T>::type>::type;
};

template <size_t N, class T>
struct tuple_element<N, T volatile>
{
    using type = typename std::add_volatile<typename tuple_element<N, T>::type>::type;
};

template <size_t N, class T>
  struct tuple_element<N, T const volatile>
{
    using type = typename std::add_cv<typename tuple_element<N, T>::type>::type;
};

template <size_t N, class T>
struct tuple_element{
    using type = typename tuple_element_impl<N,T>::type;
};

// forward declaration of tuple_size
template<class T> struct tuple_size;

template<class T>
  struct tuple_size<T const> : public tuple_size<T> {};

template<class T>
  struct tuple_size<T volatile> : public tuple_size<T> {};

template<class T>
  struct tuple_size<T const volatile> : public tuple_size<T> {};

/*! This metafunction returns the number of elements
 *  of a \p tuple type of interest.
 *
 *  \tparam T A \c tuple type of interest.
 *
 *  \see pair
 *  \see tuple
 */
template<class T>
  struct tuple_size
{
  /*! The result of this metafunction is returned in \c value.
   */
  static const int value = 1 + tuple_size<typename T::tail_type>::value;
}; // end tuple_size


// specializations for tuple_size
template<>
  struct tuple_size< tuple<> >
{
  static const int value = 0;
}; // end tuple_size< tuple<> >

template<>
  struct tuple_size<null_type>
{
  static const int value = 0;
}; // end tuple_size<null_type>



// forward declaration of detail::cons
namespace detail
{

template <class HT, class TT> struct cons;

} // end detail


// -- some traits classes for get functions
template <class T> struct access_traits
{
  typedef const T& const_type;
  typedef T& non_const_type;

  typedef const typename thrust::detail::remove_cv<T>::type& parameter_type;

// used as the tuple constructors parameter types
// Rationale: non-reference tuple element types can be cv-qualified.
// It should be possible to initialize such types with temporaries,
// and when binding temporaries to references, the reference must
// be non-volatile and const. 8.5.3. (5)
}; // end access_traits

template <class T> struct access_traits<T&>
{
  typedef T& const_type;
  typedef T& non_const_type;

  typedef T& parameter_type;
}; // end access_traits<T&>

// forward declarations of get()
template<int N, class HT, class TT>
__host__ __device__
inline typename access_traits<
                  typename tuple_element<N, detail::cons<HT, TT> >::type
                >::non_const_type
// XXX we probably don't need to do this for any compiler we care about -jph
//get(cons<HT, TT>& c BOOST_APPEND_EXPLICIT_TEMPLATE_NON_TYPE(int, N));
get(detail::cons<HT, TT>& c);

template<int N, class HT, class TT>
__host__ __device__
inline typename access_traits<
                  typename tuple_element<N, detail::cons<HT, TT> >::type
                >::const_type
// XXX we probably don't need to do this for any compiler we care about -jph
//get(const cons<HT, TT>& c BOOST_APPEND_EXPLICIT_TEMPLATE_NON_TYPE(int, N));
get(const detail::cons<HT, TT>& c);

namespace detail
{

// -- generate error template, referencing to non-existing members of this
// template is used to produce compilation errors intentionally
template<class T>
class generate_error;

// - cons getters --------------------------------------------------------
// called: get_class<N>::get<RETURN_TYPE>(aTuple)

template< int N >
struct get_class
{
  template<class RET, class HT, class TT >
  __host__ __device__
  inline static RET get(const cons<HT, TT>& t)
  {
    // XXX we may not need to deal with this for any compiler we care about -jph
    //return get_class<N-1>::BOOST_NESTED_TEMPLATE get<RET>(t.tail);
    return get_class<N-1>::template get<RET>(t.tail);

    // gcc 4.3 couldn't compile this:
    //return get_class<N-1>::get<RET>(t.tail);
  }

  template<class RET, class HT, class TT >
  __host__ __device__
  inline static RET get(cons<HT, TT>& t)
  {
    // XXX we may not need to deal with this for any compiler we care about -jph
    //return get_class<N-1>::BOOST_NESTED_TEMPLATE get<RET>(t.tail);
    return get_class<N-1>::template get<RET>(t.tail);

    // gcc 4.3 couldn't compile this:
    //return get_class<N-1>::get<RET>(t.tail);
  }
}; // end get_class

template<>
struct get_class<0>
{
  template<class RET, class HT, class TT>
  __host__ __device__
  inline static RET get(const cons<HT, TT>& t)
  {
    return t.head;
  }

  template<class RET, class HT, class TT>
  __host__ __device__
  inline static RET get(cons<HT, TT>& t)
  {
    return t.head;
  }
}; // get get_class<0>


template <bool If, class Then, class Else> struct IF
{
  typedef Then RET;
};

template <class Then, class Else> struct IF<false, Then, Else>
{
  typedef Else RET;
};

//  These helper templates wrap void types and plain function types.
//  The rationale is to allow one to write tuple types with those types
//  as elements, even though it is not possible to instantiate such object.
//  E.g: typedef tuple<void> some_type; // ok
//  but: some_type x; // fails

template <class T> class non_storeable_type
{
  __host__ __device__
  non_storeable_type();
};

template <class T> struct wrap_non_storeable_type
{
  // XXX is_function looks complicated; punt for now -jph
  //typedef typename IF<
  //  ::thrust::detail::is_function<T>::value, non_storeable_type<T>, T
  //>::RET type;

  typedef T type;
};

template <> struct wrap_non_storeable_type<void>
{
  typedef non_storeable_type<void> type;
};


template <class HT, class TT>
  struct cons
{
  typedef HT head_type;
  typedef TT tail_type;

  typedef typename
    wrap_non_storeable_type<head_type>::type stored_head_type;

  stored_head_type head;
  tail_type tail;

  inline __host__ __device__
  typename access_traits<stored_head_type>::non_const_type
  get_head() { return head; }

  inline __host__ __device__
  typename access_traits<tail_type>::non_const_type
  get_tail() { return tail; }

  inline __host__ __device__
  typename access_traits<stored_head_type>::const_type
  get_head() const { return head; }

  inline __host__ __device__
  typename access_traits<tail_type>::const_type
  get_tail() const { return tail; }

  inline __host__ __device__
  cons(void) : head(), tail() {}
  //  cons() : head(detail::default_arg<HT>::f()), tail() {}

  // the argument for head is not strictly needed, but it prevents
  // array type elements. This is good, since array type elements
  // cannot be supported properly in any case (no assignment,
  // copy works only if the tails are exactly the same type, ...)

  inline __host__ __device__
  cons(typename access_traits<stored_head_type>::parameter_type h,
       const tail_type& t)
    : head (h), tail(t) {}

  template <class T1, class T2, class T3, class T4, class T5,
            class T6, class T7, class T8, class T9, class T10>
  inline __host__ __device__
  cons( T1& t1, T2& t2, T3& t3, T4& t4, T5& t5,
        T6& t6, T7& t7, T8& t8, T9& t9, T10& t10 )
    : head (t1),
      tail (t2, t3, t4, t5, t6, t7, t8, t9, t10, static_cast<const null_type&>(null_type()))
      {}

  template <class T2, class T3, class T4, class T5,
            class T6, class T7, class T8, class T9, class T10>
  inline __host__ __device__
  cons( const null_type& /*t1*/, T2& t2, T3& t3, T4& t4, T5& t5,
        T6& t6, T7& t7, T8& t8, T9& t9, T10& t10 )
    : head (),
      tail (t2, t3, t4, t5, t6, t7, t8, t9, t10, static_cast<const null_type&>(null_type()))
      {}


  template <class HT2, class TT2>
  inline __host__ __device__
  cons( const cons<HT2, TT2>& u ) : head(u.head), tail(u.tail) {}

#if THRUST_CPP_DIALECT >= 2011
  cons(const cons &) = default;
#endif

  __thrust_exec_check_disable__
  template <class HT2, class TT2>
  inline __host__ __device__
  cons& operator=( const cons<HT2, TT2>& u ) {
    head=u.head; tail=u.tail; return *this;
  }

  // must define assignment operator explicitly, implicit version is
  // illformed if HT is a reference (12.8. (12))
  __thrust_exec_check_disable__
  inline __host__ __device__
  cons& operator=(const cons& u) {
    head = u.head; tail = u.tail;  return *this;
  }

  // XXX enable when we support std::pair -jph
  //template <class T1, class T2>
  //__host__ __device__
  //cons& operator=( const std::pair<T1, T2>& u ) {
  //  //BOOST_STATIC_ASSERT(length<cons>::value == 2); // check length = 2
  //  head = u.first; tail.head = u.second; return *this;
  //}

  // get member functions (non-const and const)
  template <int N>
  __host__ __device__
  typename access_traits<
             typename tuple_element<N, cons<HT, TT> >::type
           >::non_const_type
  get() {
    return thrust::get<N>(*this); // delegate to non-member get
  }

  template <int N>
  __host__ __device__
  typename access_traits<
             typename tuple_element<N, cons<HT, TT> >::type
           >::const_type
  get() const {
    return thrust::get<N>(*this); // delegate to non-member get
  }

  inline __host__ __device__
  void swap(cons &c)
  {
    using thrust::swap;

    swap(head, c.head);
    tail.swap(c.tail);
  }
};

template <class HT>
  struct cons<HT, null_type>
{
  typedef HT head_type;
  typedef null_type tail_type;
  typedef cons<HT, null_type> self_type;

  typedef typename
    wrap_non_storeable_type<head_type>::type stored_head_type;
  stored_head_type head;

  typename access_traits<stored_head_type>::non_const_type
  inline __host__ __device__
  get_head() { return head; }

  inline __host__ __device__
  null_type get_tail() { return null_type(); }

  inline __host__ __device__
  typename access_traits<stored_head_type>::const_type
  get_head() const { return head; }

  inline __host__ __device__
  null_type get_tail() const { return null_type(); }

  inline __host__ __device__
  cons() : head() {}

  inline __host__ __device__
  cons(typename access_traits<stored_head_type>::parameter_type h,
       const null_type& = null_type())
    : head (h) {}

  template<class T1>
  inline __host__ __device__
  cons(T1& t1, const null_type&, const null_type&, const null_type&,
       const null_type&, const null_type&, const null_type&,
       const null_type&, const null_type&, const null_type&)
  : head (t1) {}

  inline __host__ __device__
  cons(const null_type&,
       const null_type&, const null_type&, const null_type&,
       const null_type&, const null_type&, const null_type&,
       const null_type&, const null_type&, const null_type&)
  : head () {}

  template <class HT2>
  inline __host__ __device__
  cons( const cons<HT2, null_type>& u ) : head(u.head) {}

#if THRUST_CPP_DIALECT >= 2011
  cons(const cons &) = default;
#endif

  __thrust_exec_check_disable__
  template <class HT2>
  inline __host__ __device__
  cons& operator=(const cons<HT2, null_type>& u )
  {
    head = u.head;
    return *this;
  }

  // must define assignment operator explicitly, implicit version
  // is illformed if HT is a reference
  inline __host__ __device__
  cons& operator=(const cons& u) { head = u.head; return *this; }

  template <int N>
  inline __host__ __device__
  typename access_traits<
             typename tuple_element<N, self_type>::type
            >::non_const_type
  // XXX we probably don't need this for the compilers we care about -jph
  //get(BOOST_EXPLICIT_TEMPLATE_NON_TYPE(int, N))
  get(void)
  {
    return thrust::get<N>(*this);
  }

  template <int N>
  inline __host__ __device__
  typename access_traits<
             typename tuple_element<N, self_type>::type
           >::const_type
  // XXX we probably don't need this for the compilers we care about -jph
  //get(BOOST_EXPLICIT_TEMPLATE_NON_TYPE(int, N)) const
  get(void) const
  {
    return thrust::get<N>(*this);
  }

  inline __host__ __device__
  void swap(cons &c)
  {
    using thrust::swap;

    swap(head, c.head);
  }
}; // end cons

template <class T0, class T1, class T2, class T3, class T4,
          class T5, class T6, class T7, class T8, class T9>
  struct map_tuple_to_cons
{
  typedef cons<T0,
               typename map_tuple_to_cons<T1, T2, T3, T4, T5,
                                          T6, T7, T8, T9, null_type>::type
              > type;
}; // end map_tuple_to_cons

// The empty tuple is a null_type
template <>
  struct map_tuple_to_cons<null_type, null_type, null_type, null_type, null_type, null_type, null_type, null_type, null_type, null_type>
{
  typedef null_type type;
}; // end map_tuple_to_cons<...>



// ---------------------------------------------------------------------------
// The call_traits for make_tuple

// Must be instantiated with plain or const plain types (not with references)

// from template<class T> foo(const T& t) : make_tuple_traits<const T>::type
// from template<class T> foo(T& t) : make_tuple_traits<T>::type

// Conversions:
// T -> T,
// references -> compile_time_error
// array -> const ref array


template<class T>
struct make_tuple_traits {
  typedef T type;

  // commented away, see below  (JJ)
  //  typedef typename IF<
  //  boost::is_function<T>::value,
  //  T&,
  //  T>::RET type;

};

// The is_function test was there originally for plain function types,
// which can't be stored as such (we must either store them as references or
// pointers). Such a type could be formed if make_tuple was called with a
// reference to a function.
// But this would mean that a const qualified function type was formed in
// the make_tuple function and hence make_tuple can't take a function
// reference as a parameter, and thus T can't be a function type.
// So is_function test was removed.
// (14.8.3. says that type deduction fails if a cv-qualified function type
// is created. (It only applies for the case of explicitly specifying template
// args, though?)) (JJ)

template<class T>
struct make_tuple_traits<T&> {
  typedef typename
     detail::generate_error<T&>::
       do_not_use_with_reference_type error;
};

// Arrays can't be stored as plain types; convert them to references.
// All arrays are converted to const. This is because make_tuple takes its
// parameters as const T& and thus the knowledge of the potential
// non-constness of actual argument is lost.
template<class T, int n>  struct make_tuple_traits <T[n]> {
  typedef const T (&type)[n];
};

template<class T, int n>
struct make_tuple_traits<const T[n]> {
  typedef const T (&type)[n];
};

template<class T, int n>  struct make_tuple_traits<volatile T[n]> {
  typedef const volatile T (&type)[n];
};

template<class T, int n>
struct make_tuple_traits<const volatile T[n]> {
  typedef const volatile T (&type)[n];
};

// XXX enable these if we ever care about reference_wrapper -jph
//template<class T>
//struct make_tuple_traits<reference_wrapper<T> >{
//  typedef T& type;
//};
//
//template<class T>
//struct make_tuple_traits<const reference_wrapper<T> >{
//  typedef T& type;
//};


// a helper traits to make the make_tuple functions shorter (Vesa Karvonen's
// suggestion)
template <
  class T0 = null_type, class T1 = null_type, class T2 = null_type,
  class T3 = null_type, class T4 = null_type, class T5 = null_type,
  class T6 = null_type, class T7 = null_type, class T8 = null_type,
  class T9 = null_type
>
struct make_tuple_mapper {
  typedef
    tuple<typename make_tuple_traits<T0>::type,
          typename make_tuple_traits<T1>::type,
          typename make_tuple_traits<T2>::type,
          typename make_tuple_traits<T3>::type,
          typename make_tuple_traits<T4>::type,
          typename make_tuple_traits<T5>::type,
          typename make_tuple_traits<T6>::type,
          typename make_tuple_traits<T7>::type,
          typename make_tuple_traits<T8>::type,
          typename make_tuple_traits<T9>::type> type;
};

} // end detail


template<int N, class HT, class TT>
__host__ __device__
inline typename access_traits<
                  typename tuple_element<N, detail::cons<HT, TT> >::type
                >::non_const_type
get(detail::cons<HT, TT>& c)
{
  //return detail::get_class<N>::BOOST_NESTED_TEMPLATE

  // gcc 4.3 couldn't compile this:
  //return detail::get_class<N>::

  return detail::get_class<N>::template
         get<
           typename access_traits<
             typename tuple_element<N, detail::cons<HT, TT> >::type
           >::non_const_type,
           HT,TT
         >(c);
}


// get function for const cons-lists, returns a const reference to
// the element. If the element is a reference, returns the reference
// as such (that is, can return a non-const reference)
template<int N, class HT, class TT>
__host__ __device__
inline typename access_traits<
                  typename tuple_element<N, detail::cons<HT, TT> >::type
                >::const_type
get(const detail::cons<HT, TT>& c)
{
  //return detail::get_class<N>::BOOST_NESTED_TEMPLATE

  // gcc 4.3 couldn't compile this:
  //return detail::get_class<N>::

  return detail::get_class<N>::template
         get<
           typename access_traits<
             typename tuple_element<N, detail::cons<HT, TT> >::type
           >::const_type,
           HT,TT
         >(c);
}


template<class T0>
__host__ __device__ inline
  typename detail::make_tuple_mapper<T0>::type
    make_tuple(const T0& t0)
{
  typedef typename detail::make_tuple_mapper<T0>::type t;
  return t(t0);
} // end make_tuple()

template<class T0, class T1>
__host__ __device__ inline
  typename detail::make_tuple_mapper<T0, T1>::type
    make_tuple(const T0& t0, const T1& t1)
{
  typedef typename detail::make_tuple_mapper<T0,T1>::type t;
  return t(t0,t1);
} // end make_tuple()

template<class T0, class T1, class T2>
__host__ __device__ inline
  typename detail::make_tuple_mapper<T0, T1, T2>::type
    make_tuple(const T0& t0, const T1& t1, const T2& t2)
{
  typedef typename detail::make_tuple_mapper<T0,T1,T2>::type t;
  return t(t0,t1,t2);
} // end make_tuple()

template<class T0, class T1, class T2, class T3>
__host__ __device__ inline
  typename detail::make_tuple_mapper<T0, T1, T2, T3>::type
    make_tuple(const T0& t0, const T1& t1, const T2& t2, const T3& t3)
{
  typedef typename detail::make_tuple_mapper<T0,T1,T2,T3>::type t;
  return t(t0,t1,t2,t3);
} // end make_tuple()

template<class T0, class T1, class T2, class T3, class T4>
__host__ __device__ inline
  typename detail::make_tuple_mapper<T0, T1, T2, T3, T4>::type
    make_tuple(const T0& t0, const T1& t1, const T2& t2, const T3& t3, const T4& t4)
{
  typedef typename detail::make_tuple_mapper<T0,T1,T2,T3,T4>::type t;
  return t(t0,t1,t2,t3,t4);
} // end make_tuple()

template<class T0, class T1, class T2, class T3, class T4, class T5>
__host__ __device__ inline
  typename detail::make_tuple_mapper<T0, T1, T2, T3, T4, T5>::type
    make_tuple(const T0& t0, const T1& t1, const T2& t2, const T3& t3, const T4& t4, const T5& t5)
{
  typedef typename detail::make_tuple_mapper<T0,T1,T2,T3,T4,T5>::type t;
  return t(t0,t1,t2,t3,t4,t5);
} // end make_tuple()

template<class T0, class T1, class T2, class T3, class T4, class T5, class T6>
__host__ __device__ inline
  typename detail::make_tuple_mapper<T0, T1, T2, T3, T4, T5, T6>::type
    make_tuple(const T0& t0, const T1& t1, const T2& t2, const T3& t3, const T4& t4, const T5& t5, const T6& t6)
{
  typedef typename detail::make_tuple_mapper<T0,T1,T2,T3,T4,T5,T6>::type t;
  return t(t0,t1,t2,t3,t4,t5,t6);
} // end make_tuple()

template<class T0, class T1, class T2, class T3, class T4, class T5, class T6, class T7>
__host__ __device__ inline
  typename detail::make_tuple_mapper<T0, T1, T2, T3, T4, T5, T6, T7>::type
    make_tuple(const T0& t0, const T1& t1, const T2& t2, const T3& t3, const T4& t4, const T5& t5, const T6& t6, const T7& t7)
{
  typedef typename detail::make_tuple_mapper<T0,T1,T2,T3,T4,T5,T6,T7>::type t;
  return t(t0,t1,t2,t3,t4,t5,t6,t7);
} // end make_tuple()

template<class T0, class T1, class T2, class T3, class T4, class T5, class T6, class T7, class T8>
__host__ __device__ inline
  typename detail::make_tuple_mapper<T0, T1, T2, T3, T4, T5, T6, T7, T8>::type
    make_tuple(const T0& t0, const T1& t1, const T2& t2, const T3& t3, const T4& t4, const T5& t5, const T6& t6, const T7& t7, const T8& t8)
{
  typedef typename detail::make_tuple_mapper<T0,T1,T2,T3,T4,T5,T6,T7,T8>::type t;
  return t(t0,t1,t2,t3,t4,t5,t6,t7,t8);
} // end make_tuple()

template<class T0, class T1, class T2, class T3, class T4, class T5, class T6, class T7, class T8, class T9>
__host__ __device__ inline
  typename detail::make_tuple_mapper<T0, T1, T2, T3, T4, T5, T6, T7, T8, T9>::type
    make_tuple(const T0& t0, const T1& t1, const T2& t2, const T3& t3, const T4& t4, const T5& t5, const T6& t6, const T7& t7, const T8& t8, const T9& t9)
{
  typedef typename detail::make_tuple_mapper<T0,T1,T2,T3,T4,T5,T6,T7,T8,T9>::type t;
  return t(t0,t1,t2,t3,t4,t5,t6,t7,t8,t9);
} // end make_tuple()


template<typename T0>
__host__ __device__ inline
tuple<T0&> tie(T0 &t0)
{
  return tuple<T0&>(t0);
}

template<typename T0,typename T1>
__host__ __device__ inline
tuple<T0&,T1&> tie(T0 &t0, T1 &t1)
{
  return tuple<T0&,T1&>(t0,t1);
}

template<typename T0,typename T1, typename T2>
__host__ __device__ inline
tuple<T0&,T1&,T2&> tie(T0 &t0, T1 &t1, T2 &t2)
{
  return tuple<T0&,T1&,T2&>(t0,t1,t2);
}

template<typename T0,typename T1, typename T2, typename T3>
__host__ __device__ inline
tuple<T0&,T1&,T2&,T3&> tie(T0 &t0, T1 &t1, T2 &t2, T3 &t3)
{
  return tuple<T0&,T1&,T2&,T3&>(t0,t1,t2,t3);
}

template<typename T0,typename T1, typename T2, typename T3, typename T4>
__host__ __device__ inline
tuple<T0&,T1&,T2&,T3&,T4&> tie(T0 &t0, T1 &t1, T2 &t2, T3 &t3, T4 &t4)
{
  return tuple<T0&,T1&,T2&,T3&,T4&>(t0,t1,t2,t3,t4);
}

template<typename T0,typename T1, typename T2, typename T3, typename T4, typename T5>
__host__ __device__ inline
tuple<T0&,T1&,T2&,T3&,T4&,T5&> tie(T0 &t0, T1 &t1, T2 &t2, T3 &t3, T4 &t4, T5 &t5)
{
  return tuple<T0&,T1&,T2&,T3&,T4&,T5&>(t0,t1,t2,t3,t4,t5);
}

template<typename T0,typename T1, typename T2, typename T3, typename T4, typename T5, typename T6>
__host__ __device__ inline
tuple<T0&,T1&,T2&,T3&,T4&,T5&,T6&> tie(T0 &t0, T1 &t1, T2 &t2, T3 &t3, T4 &t4, T5 &t5, T6 &t6)
{
  return tuple<T0&,T1&,T2&,T3&,T4&,T5&,T6&>(t0,t1,t2,t3,t4,t5,t6);
}

template<typename T0,typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7>
__host__ __device__ inline
tuple<T0&,T1&,T2&,T3&,T4&,T5&,T6&,T7&> tie(T0 &t0, T1 &t1, T2 &t2, T3 &t3, T4 &t4, T5 &t5, T6 &t6, T7 &t7)
{
  return tuple<T0&,T1&,T2&,T3&,T4&,T5&,T6&,T7&>(t0,t1,t2,t3,t4,t5,t6,t7);
}

template<typename T0,typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7, typename T8>
__host__ __device__ inline
tuple<T0&,T1&,T2&,T3&,T4&,T5&,T6&,T7&,T8&> tie(T0 &t0, T1 &t1, T2 &t2, T3 &t3, T4 &t4, T5 &t5, T6 &t6, T7 &t7, T8 &t8)
{
  return tuple<T0&,T1&,T2&,T3&,T4&,T5&,T6&,T7&,T8&>(t0,t1,t2,t3,t4,t5,t6,t7,t8);
}

template<typename T0,typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7, typename T8, typename T9>
__host__ __device__ inline
tuple<T0&,T1&,T2&,T3&,T4&,T5&,T6&,T7&,T8&,T9&> tie(T0 &t0, T1 &t1, T2 &t2, T3 &t3, T4 &t4, T5 &t5, T6 &t6, T7 &t7, T8 &t8, T9 &t9)
{
  return tuple<T0&,T1&,T2&,T3&,T4&,T5&,T6&,T7&,T8&,T9&>(t0,t1,t2,t3,t4,t5,t6,t7,t8,t9);
}

template<
  typename T0, typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7, typename T8, typename T9,
  typename U0, typename U1, typename U2, typename U3, typename U4, typename U5, typename U6, typename U7, typename U8, typename U9
>
__host__ __device__ inline
void swap(thrust::tuple<T0,T1,T2,T3,T4,T5,T6,T7,T8,T9> &x,
          thrust::tuple<U0,U1,U2,U3,U4,U5,U6,U7,U8,U9> &y)
{
  return x.swap(y);
}



namespace detail
{

template<class T1, class T2>
__host__ __device__
inline bool eq(const T1& lhs, const T2& rhs) {
  return lhs.get_head() == rhs.get_head() &&
         eq(lhs.get_tail(), rhs.get_tail());
}
template<>
__host__ __device__
inline bool eq<null_type,null_type>(const null_type&, const null_type&) { return true; }

template<class T1, class T2>
__host__ __device__
inline bool neq(const T1& lhs, const T2& rhs) {
  return lhs.get_head() != rhs.get_head()  ||
         neq(lhs.get_tail(), rhs.get_tail());
}
template<>
__host__ __device__
inline bool neq<null_type,null_type>(const null_type&, const null_type&) { return false; }

template<class T1, class T2>
__host__ __device__
inline bool lt(const T1& lhs, const T2& rhs) {
  return (lhs.get_head() < rhs.get_head())  ||
            (!(rhs.get_head() < lhs.get_head()) &&
             lt(lhs.get_tail(), rhs.get_tail()));
}
template<>
__host__ __device__
inline bool lt<null_type,null_type>(const null_type&, const null_type&) { return false; }

template<class T1, class T2>
__host__ __device__
inline bool gt(const T1& lhs, const T2& rhs) {
  return (lhs.get_head() > rhs.get_head())  ||
            (!(rhs.get_head() > lhs.get_head()) &&
             gt(lhs.get_tail(), rhs.get_tail()));
}
template<>
__host__ __device__
inline bool gt<null_type,null_type>(const null_type&, const null_type&) { return false; }

template<class T1, class T2>
__host__ __device__
inline bool lte(const T1& lhs, const T2& rhs) {
  return lhs.get_head() <= rhs.get_head()  &&
          ( !(rhs.get_head() <= lhs.get_head()) ||
            lte(lhs.get_tail(), rhs.get_tail()));
}
template<>
__host__ __device__
inline bool lte<null_type,null_type>(const null_type&, const null_type&) { return true; }

template<class T1, class T2>
__host__ __device__
inline bool gte(const T1& lhs, const T2& rhs) {
  return lhs.get_head() >= rhs.get_head()  &&
          ( !(rhs.get_head() >= lhs.get_head()) ||
            gte(lhs.get_tail(), rhs.get_tail()));
}
template<>
__host__ __device__
inline bool gte<null_type,null_type>(const null_type&, const null_type&) { return true; }

} // end detail



// equal ----

template<class T1, class T2, class S1, class S2>
__host__ __device__
inline bool operator==(const detail::cons<T1, T2>& lhs, const detail::cons<S1, S2>& rhs)
{
  // XXX support this eventually -jph
  //// check that tuple lengths are equal
  //BOOST_STATIC_ASSERT(tuple_size<T2>::value == tuple_size<S2>::value);

  return  detail::eq(lhs, rhs);
} // end operator==()

// not equal -----

template<class T1, class T2, class S1, class S2>
__host__ __device__
inline bool operator!=(const detail::cons<T1, T2>& lhs, const detail::cons<S1, S2>& rhs)
{
  // XXX support this eventually -jph
  //// check that tuple lengths are equal
  //BOOST_STATIC_ASSERT(tuple_size<T2>::value == tuple_size<S2>::value);

  return detail::neq(lhs, rhs);
} // end operator!=()

// <
template<class T1, class T2, class S1, class S2>
__host__ __device__
inline bool operator<(const detail::cons<T1, T2>& lhs, const detail::cons<S1, S2>& rhs)
{
  // XXX support this eventually -jph
  //// check that tuple lengths are equal
  //BOOST_STATIC_ASSERT(tuple_size<T2>::value == tuple_size<S2>::value);

  return detail::lt(lhs, rhs);
} // end operator<()

// >
template<class T1, class T2, class S1, class S2>
__host__ __device__
inline bool operator>(const detail::cons<T1, T2>& lhs, const detail::cons<S1, S2>& rhs)
{
  // XXX support this eventually -jph
  //// check that tuple lengths are equal
  //BOOST_STATIC_ASSERT(tuple_size<T2>::value == tuple_size<S2>::value);

  return detail::gt(lhs, rhs);
} // end operator>()

// <=
template<class T1, class T2, class S1, class S2>
__host__ __device__
inline bool operator<=(const detail::cons<T1, T2>& lhs, const detail::cons<S1, S2>& rhs)
{
  // XXX support this eventually -jph
  //// check that tuple lengths are equal
  //BOOST_STATIC_ASSERT(tuple_size<T2>::value == tuple_size<S2>::value);

  return detail::lte(lhs, rhs);
} // end operator<=()

// >=
template<class T1, class T2, class S1, class S2>
__host__ __device__
inline bool operator>=(const detail::cons<T1, T2>& lhs, const detail::cons<S1, S2>& rhs)
{
  // XXX support this eventually -jph
  //// check that tuple lengths are equal
  //BOOST_STATIC_ASSERT(tuple_size<T2>::value == tuple_size<S2>::value);

  return detail::gte(lhs, rhs);
} // end operator>=()

THRUST_NAMESPACE_END

