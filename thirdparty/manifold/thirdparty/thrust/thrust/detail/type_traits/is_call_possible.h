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

#include <thrust/detail/type_traits.h>
#include <thrust/detail/type_traits/has_member_function.h>

// inspired by Roman Perepelitsa's presentation from comp.lang.c++.moderated
// based on the implementation here: http://www.rsdn.ru/forum/cpp/2759773.1.aspx

THRUST_NAMESPACE_BEGIN
namespace detail
{
namespace is_call_possible_detail
{

template<typename T> class void_exp_result {}; 

template<typename T, typename U> 
U const& operator,(U const&, void_exp_result<T>); 

template<typename T, typename U> 
U& operator,(U&, void_exp_result<T>); 

template<typename src_type, typename dest_type> 
struct clone_constness 
{
  typedef dest_type type; 
}; 

template<typename src_type, typename dest_type> 
struct clone_constness<const src_type, dest_type> 
{ 
  typedef const dest_type type; 
};

} // end is_call_possible_detail
} // end detail
THRUST_NAMESPACE_END

#define __THRUST_DEFINE_IS_CALL_POSSIBLE(trait_name, member_function_name)                                                                \
__THRUST_DEFINE_HAS_MEMBER_FUNCTION(trait_name##_has_member, member_function_name)                                                        \
                                                                                                                                          \
template <typename T, typename Signature>                                                                                                 \
struct trait_name                                                                                                                         \
{                                                                                                                                         \
  private:                                                                                                                                \
    struct yes {};                                                                                                                        \
    struct no { yes m[2]; };                                                                                                              \
    struct derived : public T                                                                                                             \
    {                                                                                                                                     \
      using T::member_function_name;                                                                                                      \
      no member_function_name(...) const;                                                                                                 \
    };                                                                                                                                    \
                                                                                                                                          \
    typedef typename thrust::detail::is_call_possible_detail::clone_constness<T, derived>::type derived_type;                             \
                                                                                                                                          \
    template<typename U, typename Result>                                                                                                 \
    struct return_value_check                                                                                                             \
    {                                                                                                                                     \
      static yes deduce(Result);                                                                                                          \
      static no deduce(...);                                                                                                              \
      static no deduce(no);                                                                                                               \
      static no deduce(thrust::detail::is_call_possible_detail::void_exp_result<T>);                                                      \
    };                                                                                                                                    \
                                                                                                                                          \
    template<typename U>                                                                                                                  \
    struct return_value_check<U, void>                                                                                                    \
    {                                                                                                                                     \
      static yes deduce(...);                                                                                                             \
      static no deduce(no);                                                                                                               \
    };                                                                                                                                    \
                                                                                                                                          \
    template<bool has_the_member_of_interest, typename F>                                                                                 \
    struct impl                                                                                                                           \
    {                                                                                                                                     \
      static const bool value = false;                                                                                                    \
    };                                                                                                                                    \
                                                                                                                                          \
    template<typename Result, typename Arg>                                                                                               \
    struct impl<true, Result(Arg)>                                                                                                        \
    {                                                                                                                                     \
      static typename add_reference<derived_type>::type test_me;                                                                          \
      static typename add_reference<Arg>::type          arg;                                                                              \
                                                                                                                                          \
      static const bool value =                                                                                                           \
        sizeof(                                                                                                                           \
                return_value_check<T, Result>::deduce(                                                                                    \
                  (test_me.member_function_name(arg), thrust::detail::is_call_possible_detail::void_exp_result<T>())                      \
                )                                                                                                                         \
              ) == sizeof(yes);                                                                                                           \
    };                                                                                                                                    \
                                                                                                                                          \
    template<typename Result, typename Arg1, typename Arg2>                                                                               \
    struct impl<true, Result(Arg1,Arg2)>                                                                                                  \
    {                                                                                                                                     \
      static typename add_reference<derived_type>::type test_me;                                                                          \
      static typename add_reference<Arg1>::type         arg1;                                                                             \
      static typename add_reference<Arg2>::type         arg2;                                                                             \
                                                                                                                                          \
      static const bool value =                                                                                                           \
        sizeof(                                                                                                                           \
                return_value_check<T, Result>::deduce(                                                                                    \
                  (test_me.member_function_name(arg1,arg2), thrust::detail::is_call_possible_detail::void_exp_result<T>())                \
                )                                                                                                                         \
              ) == sizeof(yes);                                                                                                           \
    };                                                                                                                                    \
                                                                                                                                          \
    template<typename Result, typename Arg1, typename Arg2, typename Arg3>                                                                \
    struct impl<true, Result(Arg1,Arg2,Arg3)>                                                                                             \
    {                                                                                                                                     \
      static typename add_reference<derived_type>::type test_me;                                                                          \
      static typename add_reference<Arg1>::type         arg1;                                                                             \
      static typename add_reference<Arg2>::type         arg2;                                                                             \
      static typename add_reference<Arg3>::type         arg3;                                                                             \
                                                                                                                                          \
      static const bool value =                                                                                                           \
        sizeof(                                                                                                                           \
                return_value_check<T, Result>::deduce(                                                                                    \
                  (test_me.member_function_name(arg1,arg2,arg3), thrust::detail::is_call_possible_detail::void_exp_result<T>())           \
                )                                                                                                                         \
              ) == sizeof(yes);                                                                                                           \
    };                                                                                                                                    \
                                                                                                                                          \
    template<typename Result, typename Arg1, typename Arg2, typename Arg3, typename Arg4>                                                 \
    struct impl<true, Result(Arg1,Arg2,Arg3,Arg4)>                                                                                        \
    {                                                                                                                                     \
      static typename add_reference<derived_type>::type test_me;                                                                          \
      static typename add_reference<Arg1>::type         arg1;                                                                             \
      static typename add_reference<Arg2>::type         arg2;                                                                             \
      static typename add_reference<Arg3>::type         arg3;                                                                             \
      static typename add_reference<Arg4>::type         arg4;                                                                             \
                                                                                                                                          \
      static const bool value =                                                                                                           \
        sizeof(                                                                                                                           \
                return_value_check<T, Result>::deduce(                                                                                    \
                  (test_me.member_function_name(arg1,arg2,arg3,arg4), thrust::detail::is_call_possible_detail::void_exp_result<T>())      \
                )                                                                                                                         \
              ) == sizeof(yes);                                                                                                           \
    };                                                                                                                                    \
                                                                                                                                          \
  public:                                                                                                                                 \
    static const bool value = impl<trait_name##_has_member<T,Signature>::value, Signature>::value;                                        \
    typedef thrust::detail::integral_constant<bool,value> type;                                                                           \
}; 

