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


/*! \file type_traits.h
 *  \brief Temporarily define some type traits
 *         until nvcc can compile tr1::type_traits.
 */

#pragma once

#include <thrust/detail/config.h>

#if THRUST_CPP_DIALECT >= 2011
#  include <type_traits>
#endif

THRUST_NAMESPACE_BEGIN

// forward declaration of device_reference
template<typename T> class device_reference;

namespace detail
{
 /// helper classes [4.3].
 template<typename T, T v>
   struct integral_constant
   {
     THRUST_INLINE_INTEGRAL_MEMBER_CONSTANT T value = v;

     typedef T                       value_type;
     typedef integral_constant<T, v> type;

     // We don't want to switch to std::integral_constant, because we want access
     // to the C++14 operator(), but we'd like standard traits to interoperate
     // with our version when tag dispatching.
     #if THRUST_CPP_DIALECT >= 2011
     integral_constant() = default;

     integral_constant(integral_constant const&) = default;

     integral_constant& operator=(integral_constant const&) = default;

     constexpr __host__ __device__
     integral_constant(std::integral_constant<T, v>) noexcept {}
     #endif

     constexpr __host__ __device__ operator value_type() const noexcept { return value; }
     constexpr __host__ __device__ value_type operator()() const noexcept { return value; }
   };
 
 /// typedef for true_type
 typedef integral_constant<bool, true>  true_type;

 /// typedef for true_type
 typedef integral_constant<bool, false> false_type;

//template<typename T> struct is_integral : public std::tr1::is_integral<T> {};
template<typename T> struct is_integral                           : public false_type {};
template<>           struct is_integral<bool>                     : public true_type {};
template<>           struct is_integral<char>                     : public true_type {};
template<>           struct is_integral<signed char>              : public true_type {};
template<>           struct is_integral<unsigned char>            : public true_type {};
template<>           struct is_integral<short>                    : public true_type {};
template<>           struct is_integral<unsigned short>           : public true_type {};
template<>           struct is_integral<int>                      : public true_type {};
template<>           struct is_integral<unsigned int>             : public true_type {};
template<>           struct is_integral<long>                     : public true_type {};
template<>           struct is_integral<unsigned long>            : public true_type {};
template<>           struct is_integral<long long>                : public true_type {};
template<>           struct is_integral<unsigned long long>       : public true_type {};
template<>           struct is_integral<const bool>               : public true_type {};
template<>           struct is_integral<const char>               : public true_type {};
template<>           struct is_integral<const unsigned char>      : public true_type {};
template<>           struct is_integral<const short>              : public true_type {};
template<>           struct is_integral<const unsigned short>     : public true_type {};
template<>           struct is_integral<const int>                : public true_type {};
template<>           struct is_integral<const unsigned int>       : public true_type {};
template<>           struct is_integral<const long>               : public true_type {};
template<>           struct is_integral<const unsigned long>      : public true_type {};
template<>           struct is_integral<const long long>          : public true_type {};
template<>           struct is_integral<const unsigned long long> : public true_type {};

template<typename T> struct is_floating_point              : public false_type {};
template<>           struct is_floating_point<float>       : public true_type {};
template<>           struct is_floating_point<double>      : public true_type {};
template<>           struct is_floating_point<long double> : public true_type {};

template<typename T> struct is_arithmetic               : public is_integral<T> {};
template<>           struct is_arithmetic<float>        : public true_type {};
template<>           struct is_arithmetic<double>       : public true_type {};
template<>           struct is_arithmetic<const float>  : public true_type {};
template<>           struct is_arithmetic<const double> : public true_type {};

template<typename T> struct is_pointer      : public false_type {};
template<typename T> struct is_pointer<T *> : public true_type  {};

template<typename T> struct is_device_ptr  : public false_type {};

template<typename T> struct is_void             : public false_type {};
template<>           struct is_void<void>       : public true_type {};
template<>           struct is_void<const void> : public true_type {};

template<typename T> struct is_non_bool_integral       : public is_integral<T> {};
template<>           struct is_non_bool_integral<bool> : public false_type {};

template<typename T> struct is_non_bool_arithmetic       : public is_arithmetic<T> {};
template<>           struct is_non_bool_arithmetic<bool> : public false_type {};

template<typename T> struct is_pod
   : public integral_constant<
       bool,
       is_void<T>::value || is_pointer<T>::value || is_arithmetic<T>::value
#if THRUST_HOST_COMPILER == THRUST_HOST_COMPILER_MSVC || \
    THRUST_HOST_COMPILER == THRUST_HOST_COMPILER_CLANG
// use intrinsic type traits
       || __is_pod(T)
#elif THRUST_HOST_COMPILER == THRUST_HOST_COMPILER_GCC
// only use the intrinsic for >= 4.3
#if (__GNUC__ * 100 + __GNUC_MINOR__ >= 403)
       || __is_pod(T)
#endif // GCC VERSION
#endif // THRUST_HOST_COMPILER
     >
 {};


template<typename T> struct has_trivial_constructor
  : public integral_constant<
      bool,
      is_pod<T>::value
#if THRUST_HOST_COMPILER == THRUST_HOST_COMPILER_MSVC || \
    THRUST_HOST_COMPILER == THRUST_HOST_COMPILER_CLANG
      || __has_trivial_constructor(T)
#elif THRUST_HOST_COMPILER == THRUST_HOST_COMPILER_GCC
// only use the intrinsic for >= 4.3
#if (__GNUC__ >= 4) && (__GNUC_MINOR__ >= 3)
      || __has_trivial_constructor(T)
#endif // GCC VERSION
#endif // THRUST_HOST_COMPILER
      >
{};

template<typename T> struct has_trivial_copy_constructor
  : public integral_constant<
      bool,
      is_pod<T>::value
#if THRUST_HOST_COMPILER == THRUST_HOST_COMPILER_MSVC || \
    THRUST_HOST_COMPILER == THRUST_HOST_COMPILER_CLANG
      || __has_trivial_copy(T)
#elif THRUST_HOST_COMPILER == THRUST_HOST_COMPILER_GCC
// only use the intrinsic for >= 4.3
#if (__GNUC__ >= 4) && (__GNUC_MINOR__ >= 3)
      || __has_trivial_copy(T)
#endif // GCC VERSION
#endif // THRUST_HOST_COMPILER
    >
{};

template<typename T> struct has_trivial_destructor : public is_pod<T> {};

template<typename T> struct is_const          : public false_type {};
template<typename T> struct is_const<const T> : public true_type {};

template<typename T> struct is_volatile             : public false_type {};
template<typename T> struct is_volatile<volatile T> : public true_type {};

template<typename T>
  struct add_const
{
  typedef T const type;
}; // end add_const

template<typename T>
  struct remove_const
{
  typedef T type;
}; // end remove_const

template<typename T>
  struct remove_const<const T>
{
  typedef T type;
}; // end remove_const

template<typename T>
  struct add_volatile
{
  typedef volatile T type;
}; // end add_volatile

template<typename T>
  struct remove_volatile
{
  typedef T type;
}; // end remove_volatile

template<typename T>
  struct remove_volatile<volatile T>
{
  typedef T type;
}; // end remove_volatile

template<typename T>
  struct add_cv
{
  typedef const volatile T type;
}; // end add_cv

template<typename T>
  struct remove_cv
{
  typedef typename remove_const<typename remove_volatile<T>::type>::type type;
}; // end remove_cv


template<typename T> struct is_reference     : public false_type {};
template<typename T> struct is_reference<T&> : public true_type {};

template<typename T> struct is_proxy_reference  : public false_type {};

template<typename T> struct is_device_reference                                : public false_type {};
template<typename T> struct is_device_reference< thrust::device_reference<T> > : public true_type {};


// NB: Careful with reference to void.
template<typename _Tp, bool = (is_void<_Tp>::value || is_reference<_Tp>::value)>
  struct __add_reference_helper
  { typedef _Tp&    type; };

template<typename _Tp>
  struct __add_reference_helper<_Tp, true>
  { typedef _Tp     type; };

template<typename _Tp>
  struct add_reference
    : public __add_reference_helper<_Tp>{};

template<typename T>
  struct remove_reference
{
  typedef T type;
}; // end remove_reference

template<typename T>
  struct remove_reference<T&>
{
  typedef T type;
}; // end remove_reference

template<typename T1, typename T2>
  struct is_same
    : public false_type
{
}; // end is_same

template<typename T>
  struct is_same<T,T>
    : public true_type
{
}; // end is_same

template<typename T1, typename T2>
  struct lazy_is_same
    : is_same<typename T1::type, typename T2::type>
{
}; // end lazy_is_same

template<typename T1, typename T2>
  struct is_different
    : public true_type
{
}; // end is_different

template<typename T>
  struct is_different<T,T>
    : public false_type
{
}; // end is_different

template<typename T1, typename T2>
  struct lazy_is_different
    : is_different<typename T1::type, typename T2::type>
{
}; // end lazy_is_different

#if THRUST_CPP_DIALECT >= 2011

using std::is_convertible;

#else

namespace tt_detail
{

template<typename T>
  struct is_int_or_cref
{
  typedef typename remove_reference<T>::type type_sans_ref;
  static const bool value = (is_integral<T>::value
                             || (is_integral<type_sans_ref>::value
                                 && is_const<type_sans_ref>::value
                                 && !is_volatile<type_sans_ref>::value));
}; // end is_int_or_cref


THRUST_DISABLE_MSVC_POSSIBLE_LOSS_OF_DATA_WARNING_BEGIN
THRUST_DISABLE_MSVC_FORCING_VALUE_TO_BOOL_WARNING_BEGIN

template<typename From, typename To>
  struct is_convertible_sfinae
{
  private:
    typedef char                          yes;
    typedef struct { char two_chars[2]; } no;

    static inline yes   test(To) { return yes(); }
    static inline no    test(...) { return no(); } 
    static inline typename remove_reference<From>::type& from() { typename remove_reference<From>::type* ptr = 0; return *ptr; }

  public:
    static const bool value = sizeof(test(from())) == sizeof(yes);
}; // end is_convertible_sfinae


THRUST_DISABLE_MSVC_FORCING_VALUE_TO_BOOL_WARNING_END
THRUST_DISABLE_MSVC_POSSIBLE_LOSS_OF_DATA_WARNING_END


template<typename From, typename To>
  struct is_convertible_needs_simple_test
{
  static const bool from_is_void      = is_void<From>::value;
  static const bool to_is_void        = is_void<To>::value;
  static const bool from_is_float     = is_floating_point<typename remove_reference<From>::type>::value;
  static const bool to_is_int_or_cref = is_int_or_cref<To>::value;

  static const bool value = (from_is_void || to_is_void || (from_is_float && to_is_int_or_cref));
}; // end is_convertible_needs_simple_test


template<typename From, typename To,
         bool = is_convertible_needs_simple_test<From,To>::value>
  struct is_convertible
{
  static const bool value = (is_void<To>::value
                             || (is_int_or_cref<To>::value
                                 && !is_void<From>::value));
}; // end is_convertible


template<typename From, typename To>
  struct is_convertible<From, To, false>
{
  static const bool value = (is_convertible_sfinae<typename
                             add_reference<From>::type, To>::value);
}; // end is_convertible


} // end tt_detail

template<typename From, typename To>
  struct is_convertible
    : public integral_constant<bool, tt_detail::is_convertible<From, To>::value>
{
}; // end is_convertible

#endif

template<typename T1, typename T2>
  struct is_one_convertible_to_the_other
    : public integral_constant<
        bool,
        is_convertible<T1,T2>::value || is_convertible<T2,T1>::value
      >
{};


// mpl stuff
template<typename... Conditions>
  struct or_;

template <>
  struct or_<>
    : public integral_constant<
        bool,
        false_type::value  // identity for or_
      >
{
}; // end or_

template <typename Condition, typename... Conditions>
  struct or_<Condition, Conditions...>
    : public integral_constant<
        bool,
        Condition::value || or_<Conditions...>::value
      >
{
}; // end or_

template <typename... Conditions>
  struct and_;

template<>
  struct and_<>
    : public integral_constant<
        bool,
        true_type::value // identity for and_
      >
{
}; // end and_

template <typename Condition, typename... Conditions>
  struct and_<Condition, Conditions...>
    : public integral_constant<
        bool,
        Condition::value && and_<Conditions...>::value>
{
}; // end and_

template <typename Boolean>
  struct not_
    : public integral_constant<bool, !Boolean::value>
{
}; // end not_

template<bool B, class T, class F>
struct conditional { typedef T type; };
 
template<class T, class F>
struct conditional<false, T, F> { typedef F type; };

template <bool, typename Then, typename Else>
  struct eval_if
{
}; // end eval_if

template<typename Then, typename Else>
  struct eval_if<true, Then, Else>
{
  typedef typename Then::type type;
}; // end eval_if

template<typename Then, typename Else>
  struct eval_if<false, Then, Else>
{
  typedef typename Else::type type;
}; // end eval_if

template<typename T>
//  struct identity
//  XXX WAR nvcc's confusion with thrust::identity
  struct identity_
{
  typedef T type;
}; // end identity

template<bool, typename T = void> struct enable_if {};
template<typename T>              struct enable_if<true, T> {typedef T type;};

template<bool, typename T> struct lazy_enable_if {};
template<typename T>       struct lazy_enable_if<true, T> {typedef typename T::type type;};

template<bool condition, typename T = void> struct disable_if : enable_if<!condition, T> {};
template<bool condition, typename T>        struct lazy_disable_if : lazy_enable_if<!condition, T> {};


template<typename T1, typename T2, typename T = void>
  struct enable_if_convertible
    : enable_if< is_convertible<T1,T2>::value, T >
{};


template<typename T1, typename T2, typename T = void>
  struct disable_if_convertible
    : disable_if< is_convertible<T1,T2>::value, T >
{};


template<typename T1, typename T2, typename Result = void>
  struct enable_if_different
    : enable_if<is_different<T1,T2>::value, Result>
{};


template<typename T>
  struct is_numeric
    : and_<
        is_convertible<int,T>,
        is_convertible<T,int>
      >
{
}; // end is_numeric


template<typename> struct is_reference_to_const             : false_type {};
template<typename T> struct is_reference_to_const<const T&> : true_type {};


// make_unsigned follows

namespace tt_detail
{

template<typename T> struct make_unsigned_simple;

template<> struct make_unsigned_simple<char>                   { typedef unsigned char          type; };
template<> struct make_unsigned_simple<signed char>            { typedef unsigned char          type; };
template<> struct make_unsigned_simple<unsigned char>          { typedef unsigned char          type; };
template<> struct make_unsigned_simple<short>                  { typedef unsigned short         type; };
template<> struct make_unsigned_simple<unsigned short>         { typedef unsigned short         type; };
template<> struct make_unsigned_simple<int>                    { typedef unsigned int           type; };
template<> struct make_unsigned_simple<unsigned int>           { typedef unsigned int           type; };
template<> struct make_unsigned_simple<long int>               { typedef unsigned long int      type; };
template<> struct make_unsigned_simple<unsigned long int>      { typedef unsigned long int      type; };
template<> struct make_unsigned_simple<long long int>          { typedef unsigned long long int type; };
template<> struct make_unsigned_simple<unsigned long long int> { typedef unsigned long long int type; };

template<typename T>
  struct make_unsigned_base
{
  // remove cv
  typedef typename remove_cv<T>::type remove_cv_t;

  // get the simple unsigned type
  typedef typename make_unsigned_simple<remove_cv_t>::type unsigned_remove_cv_t;

  // add back const, volatile, both, or neither to the simple result
  typedef typename eval_if<
    is_const<T>::value && is_volatile<T>::value,
    // add cv back
    add_cv<unsigned_remove_cv_t>,
    // check const & volatile individually
    eval_if<
      is_const<T>::value,
      // add c back
      add_const<unsigned_remove_cv_t>,
      eval_if<
        is_volatile<T>::value,
        // add v back
        add_volatile<unsigned_remove_cv_t>,
        // original type was neither cv, return the simple unsigned result
        identity_<unsigned_remove_cv_t>
      >
    >
  >::type type;
};

} // end tt_detail

template<typename T>
  struct make_unsigned
    : tt_detail::make_unsigned_base<T>
{};

struct largest_available_float
{
#if defined(__CUDA_ARCH__)
#  if (__CUDA_ARCH__ < 130)
  typedef float type;
#  else
  typedef double type;
#  endif
#else
  typedef double type;
#endif
};

// T1 wins if they are both the same size
template<typename T1, typename T2>
  struct larger_type
    : thrust::detail::eval_if<
        (sizeof(T2) > sizeof(T1)),
        thrust::detail::identity_<T2>,
        thrust::detail::identity_<T1>
      >
{};

#if THRUST_CPP_DIALECT >= 2011

using std::is_base_of;

#else

namespace is_base_of_ns
{

typedef char                          yes;
typedef struct { char two_chars[2]; } no;

template<typename Base, typename Derived>
  struct host
{
  operator Base*() const;
  operator Derived*();
}; // end host

template<typename Base, typename Derived>
  struct impl
{
  template<typename T> static yes check(Derived *, T);
  static no check(Base*, int);

  static const bool value = sizeof(check(host<Base,Derived>(), int())) == sizeof(yes);
}; // end impl

} // end is_base_of_ns


template<typename Base, typename Derived>
  struct is_base_of
    : integral_constant<
        bool,
        is_base_of_ns::impl<Base,Derived>::value
      >
{};

#endif

template<typename Base, typename Derived, typename Result = void>
  struct enable_if_base_of
    : enable_if<
        is_base_of<Base,Derived>::value,
        Result
      >
{};


namespace is_assignable_ns
{

template<typename T1, typename T2>
  class is_assignable
{
  typedef char                      yes_type;
  typedef struct { char array[2]; } no_type;

  template<typename T> static typename add_reference<T>::type declval();
  
  template<size_t> struct helper { typedef void * type; };

  template<typename U1, typename U2> static yes_type test(typename helper<sizeof(declval<U1>() = declval<U2>())>::type);

  template<typename,typename> static no_type test(...);

  public:
    static const bool value = sizeof(test<T1,T2>(0)) == 1;
}; // end is_assignable

} // end is_assignable_ns


template<typename T1, typename T2>
  struct is_assignable
    : integral_constant<
        bool,
        is_assignable_ns::is_assignable<T1,T2>::value
      >
{};


template<typename T>
  struct is_copy_assignable
    : is_assignable<
        typename add_reference<T>::type,
        typename add_reference<typename add_const<T>::type>::type
      >
{};


template<typename T1, typename T2, typename Enable = void> struct promoted_numerical_type;

template<typename T1, typename T2> 
  struct promoted_numerical_type<T1,T2,typename enable_if<and_
  <typename is_floating_point<T1>::type,typename is_floating_point<T2>::type>
  ::value>::type>
  {
  typedef typename larger_type<T1,T2>::type type;
  };

template<typename T1, typename T2> 
  struct promoted_numerical_type<T1,T2,typename enable_if<and_
  <typename is_integral<T1>::type,typename is_floating_point<T2>::type>
  ::value>::type>
  {
  typedef T2 type;
  };

template<typename T1, typename T2>
  struct promoted_numerical_type<T1,T2,typename enable_if<and_
  <typename is_floating_point<T1>::type, typename is_integral<T2>::type>
  ::value>::type>
  {
  typedef T1 type;
  };

template<typename T>
  struct is_empty_helper : public T
  {
  };

struct is_empty_helper_base
{
};

template<typename T>
  struct is_empty : integral_constant<bool,
    sizeof(is_empty_helper_base) == sizeof(is_empty_helper<T>)
  >
  {
  };

} // end detail

using detail::integral_constant;
using detail::true_type;
using detail::false_type;

THRUST_NAMESPACE_END

#include <thrust/detail/type_traits/has_trivial_assign.h>

