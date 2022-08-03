/*! \file meta.h
 *  \brief Defines template classes
 *         for metaprogramming in the
 *         unit tests.
 */

#pragma once

namespace unittest
{

// mark the absence of a type
struct null_type {}; 

// this type encapsulates a list of
// types
template<typename... Ts>
  struct type_list
{
};

// this type provides a way of indexing
// into a type_list
template<typename List, unsigned int i>
  struct get_type
{
  typedef null_type type;
};

template<typename T, typename... Ts>
  struct get_type<type_list<T, Ts...>, 0>
{
  typedef T type;
};

template<typename T, typename... Ts, unsigned int i>
  struct get_type<type_list<T, Ts...>, i>
{
  typedef typename get_type<type_list<Ts...>, i - 1>::type type;
};

// this type and its specialization provides a way to
// iterate over a type_list, and
// applying a unary function to each type
template<typename TypeList,
         template <typename> class Function,
         typename T,
         unsigned int i = 0>
  struct for_each_type
{
  template<typename U>
    void operator()(U n)
  {
    // run the function on type T
    Function<T> f;
    f(n);

    // get the next type
    typedef typename get_type<TypeList,i+1>::type next_type;

    // recurse to i + 1
    for_each_type<TypeList, Function, next_type, i + 1> loop;
    loop(n);
  }

  void operator()(void)
  {
    // run the function on type T
    Function<T> f;
    f();

    // get the next type
    typedef typename get_type<TypeList,i+1>::type next_type;

    // recurse to i + 1
    for_each_type<TypeList, Function, next_type, i + 1> loop;
    loop();
  }
};

// terminal case: do nothing when encountering null_type
template<typename TypeList,
         template <typename> class Function,
         unsigned int i>
  struct for_each_type<TypeList, Function, null_type, i>
{
  template<typename U>
    void operator()(U)
  {
    // no-op
  }

  void operator()(void)
  {
    // no-op
  }
};

// this type and its specialization instantiates
// a template by applying T to Template.
// if T == null_type, then its result is also null_type
template<template <typename> class Template,
         typename T>
  struct ApplyTemplate1
{
  typedef Template<T> type;
};

template<template <typename> class Template>
  struct ApplyTemplate1<Template, null_type>
{
  typedef null_type type;
};

// this type and its specializations instantiates
// a template by applying T1 & T2 to Template.
// if either T1 or T2 == null_type, then its result
// is also null_type
template<template <typename,typename> class Template,
         typename T1,
         typename T2>
  struct ApplyTemplate2
{
  typedef Template<T1,T2> type;
};

template<template <typename,typename> class Template,
         typename T>
  struct ApplyTemplate2<Template, T, null_type>
{
  typedef null_type type;
};

template<template <typename,typename> class Template,
         typename T>
  struct ApplyTemplate2<Template, null_type, T>
{
  typedef null_type type;
};

template<template <typename,typename> class Template>
  struct ApplyTemplate2<Template, null_type, null_type>
{
  typedef null_type type;
};

// this type creates a new type_list by applying a Template to each of
// the Type_list's types
template<typename TypeList,
         template <typename> class Template>
  struct transform1;

template<typename... Ts,
         template <typename> class Template>
  struct transform1<type_list<Ts...>, Template>
{
  typedef type_list<typename ApplyTemplate1<Template, Ts>::type...> type;
};

template<typename TypeList1,
         typename TypeList2,
         template <typename,typename> class Template>
  struct transform2;

template<typename... T1s,
         typename... T2s,
         template <typename,typename> class Template>
  struct transform2<type_list<T1s...>, type_list<T2s...>, Template>
{
  typedef type_list<typename ApplyTemplate2<Template, T1s, T2s>::type...> type;
};

} // end unittest

