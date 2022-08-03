#pragma once

#include <iostream>
#include <string>
#include <typeinfo>
#include <unittest/system.h>

#include <thrust/extrema.h>
#include <thrust/limits.h>
#include <thrust/detail/type_traits.h>

namespace unittest
{

template<typename T>
  std::string type_name(void)
{
  return demangle(typeid(T).name());
} // end type_name()

// Use this with counting_iterator to avoid generating a range larger than we
// can represent.
template <typename T>
typename THRUST_NS_QUALIFIER::detail::disable_if<
  THRUST_NS_QUALIFIER::detail::is_floating_point<T>::value
, T
>::type truncate_to_max_representable(std::size_t n)
{
  return static_cast<T>(THRUST_NS_QUALIFIER::min<std::size_t>(
    n,
    static_cast<std::size_t>(THRUST_NS_QUALIFIER::numeric_limits<T>::max())));
}

// TODO: This probably won't work for `half`.
template <typename T>
typename THRUST_NS_QUALIFIER::detail::enable_if<
  THRUST_NS_QUALIFIER::detail::is_floating_point<T>::value
, T
>::type truncate_to_max_representable(std::size_t n)
{
  return THRUST_NS_QUALIFIER::min<T>(
    static_cast<T>(n),
    THRUST_NS_QUALIFIER::numeric_limits<T>::max());
}

} // end unittest

template <typename Iterator>
void PRINT(Iterator first, Iterator last)
{
  size_t n = 0;
  for (Iterator i = first; i != last; i++, n++)
    std::cout << ">>> [" << n << "] = " << *i << std::endl;
}

template <typename Container>
void PRINT(const Container& c)
{
  PRINT(c.begin(), c.end());
}

template <size_t N>
void PRINT(const char (&c)[N])
{
  std::cout << std::string(c, c + N) << std::endl;
}

