#pragma once

#include <thrust/host_vector.h>
#include <thrust/random.h>
#include <thrust/detail/type_traits.h>

#include <limits>

namespace unittest
{

inline unsigned int hash(unsigned int a)
{
    a = (a+0x7ed55d16) + (a<<12);
    a = (a^0xc761c23c) ^ (a>>19);
    a = (a+0x165667b1) + (a<<5);
    a = (a+0xd3a2646c) ^ (a<<9);
    a = (a+0xfd7046c5) + (a<<3);
    a = (a^0xb55a4f09) ^ (a>>16);
    return a;
}

template<typename T, typename = void>
  struct generate_random_integer;

template<typename T>
  struct generate_random_integer<T,
    typename THRUST_NS_QUALIFIER::detail::disable_if<
      THRUST_NS_QUALIFIER::detail::is_non_bool_arithmetic<T>::value
    >::type
  >
{
  T operator()(unsigned int i) const
  {
      THRUST_NS_QUALIFIER::default_random_engine rng(hash(i));

      return static_cast<T>(rng());
  }
};

template<typename T>
  struct generate_random_integer<T,
    typename THRUST_NS_QUALIFIER::detail::enable_if<
      THRUST_NS_QUALIFIER::detail::is_non_bool_integral<T>::value
    >::type
  >
{
  T operator()(unsigned int i) const
  {
      THRUST_NS_QUALIFIER::default_random_engine rng(hash(i));
      THRUST_NS_QUALIFIER::uniform_int_distribution<T> dist;

      return static_cast<T>(dist(rng));
  }
};

template<typename T>
  struct generate_random_integer<T,
    typename THRUST_NS_QUALIFIER::detail::enable_if<
      THRUST_NS_QUALIFIER::detail::is_floating_point<T>::value
    >::type
  >
{
  T operator()(unsigned int i) const
  {
      T const min = std::numeric_limits<T>::min();
      T const max = std::numeric_limits<T>::max();

      THRUST_NS_QUALIFIER::default_random_engine rng(hash(i));
      THRUST_NS_QUALIFIER::uniform_real_distribution<T> dist(min, max);

      return static_cast<T>(dist(rng));
  }
};

template<>
  struct generate_random_integer<bool>
{
  bool operator()(unsigned int i) const
  {
      THRUST_NS_QUALIFIER::default_random_engine rng(hash(i));
      THRUST_NS_QUALIFIER::uniform_int_distribution<unsigned int> dist(0,1);

      return dist(rng) == 1;
  }
};


template<typename T>
  struct generate_random_sample
{
  T operator()(unsigned int i) const
  {
      THRUST_NS_QUALIFIER::default_random_engine rng(hash(i));
      THRUST_NS_QUALIFIER::uniform_int_distribution<unsigned int> dist(0,20);

      return static_cast<T>(dist(rng));
  } 
}; 



template<typename T>
THRUST_NS_QUALIFIER::host_vector<T> random_integers(const size_t N)
{
    THRUST_NS_QUALIFIER::host_vector<T> vec(N);
    THRUST_NS_QUALIFIER::transform(THRUST_NS_QUALIFIER::counting_iterator<unsigned int>(static_cast<unsigned int>(0)),
                                   THRUST_NS_QUALIFIER::counting_iterator<unsigned int>(static_cast<unsigned int>(N)),
                                   vec.begin(),
                                   generate_random_integer<T>());

    return vec;
}

template<typename T>
T random_integer()
{
    return generate_random_integer<T>()(0);
}

template<typename T>
THRUST_NS_QUALIFIER::host_vector<T> random_samples(const size_t N)
{
    THRUST_NS_QUALIFIER::host_vector<T> vec(N);
    THRUST_NS_QUALIFIER::transform(THRUST_NS_QUALIFIER::counting_iterator<unsigned int>(static_cast<unsigned int>(0)),
                                   THRUST_NS_QUALIFIER::counting_iterator<unsigned int>(static_cast<unsigned int>(N)),
                                   vec.begin(),
                                   generate_random_sample<T>());

    return vec;
}

}; //end namespace unittest

