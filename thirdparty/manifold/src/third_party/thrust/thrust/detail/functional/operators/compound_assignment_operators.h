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
#include <thrust/detail/functional/actor.h>
#include <thrust/detail/functional/composite.h>
#include <thrust/detail/functional/operators/operator_adaptors.h>

THRUST_NAMESPACE_BEGIN
namespace detail
{
namespace functional
{

// there's no standard plus_equal functional, so roll an ad hoc one here
struct plus_equal
{
  using is_transparent = void;

  __thrust_exec_check_disable__
  template <typename T1, typename T2>
  __host__ __device__
  constexpr auto operator()(T1&& t1, T2&& t2) const
  noexcept(noexcept(THRUST_FWD(t1) += THRUST_FWD(t2)))
  THRUST_TRAILING_RETURN(decltype(THRUST_FWD(t1) += THRUST_FWD(t2)))
  {
    return THRUST_FWD(t1) += THRUST_FWD(t2);
  }
};

template<typename T1, typename T2>
__host__ __device__
actor<
  composite<
    transparent_binary_operator<plus_equal>,
    actor<T1>,
    typename as_actor<T2>::type
  >
>
operator+=(const actor<T1> &_1, const T2 &_2)
{
  return compose(transparent_binary_operator<plus_equal>(),
                 make_actor(_1),
                 make_actor(_2));
} // end operator+=()

template<typename T1, typename T2>
__host__ __device__
actor<
  composite<
    transparent_binary_operator<plus_equal>,
    actor<T1>,
    actor<T2>
  >
>
operator+=(const actor<T1> &_1, const actor<T2> &_2)
{
  return compose(transparent_binary_operator<plus_equal>(),
                 make_actor(_1),
                 make_actor(_2));
} // end operator+=()

// there's no standard minus_equal functional, so roll an ad hoc one here
struct minus_equal
{
  using is_transparent = void;

  __thrust_exec_check_disable__
  template <typename T1, typename T2>
  __host__ __device__
  constexpr auto operator()(T1&& t1, T2&& t2) const
  noexcept(noexcept(THRUST_FWD(t1) -= THRUST_FWD(t2)))
  THRUST_TRAILING_RETURN(decltype(THRUST_FWD(t1) -= THRUST_FWD(t2)))
  {
    return THRUST_FWD(t1) -= THRUST_FWD(t2);
  }
};

template<typename T1, typename T2>
__host__ __device__
actor<
  composite<
    transparent_binary_operator<minus_equal>,
    actor<T1>,
    typename as_actor<T2>::type
  >
>
operator-=(const actor<T1> &_1, const T2 &_2)
{
  return compose(transparent_binary_operator<minus_equal>(),
                 make_actor(_1),
                 make_actor(_2));
} // end operator-=()

template<typename T1, typename T2>
__host__ __device__
actor<
  composite<
    transparent_binary_operator<minus_equal>,
    actor<T1>,
    actor<T2>
  >
>
operator-=(const actor<T1> &_1, const actor<T2> &_2)
{
  return compose(transparent_binary_operator<minus_equal>(),
                 make_actor(_1),
                 make_actor(_2));
} // end operator-=()

// there's no standard multiplies_equal functional, so roll an ad hoc one here
struct multiplies_equal
{
  using is_transparent = void;

  __thrust_exec_check_disable__
  template <typename T1, typename T2>
  __host__ __device__
  constexpr auto operator()(T1&& t1, T2&& t2) const
  noexcept(noexcept(THRUST_FWD(t1) *= THRUST_FWD(t2)))
  THRUST_TRAILING_RETURN(decltype(THRUST_FWD(t1) *= THRUST_FWD(t2)))
  {
    return THRUST_FWD(t1) *= THRUST_FWD(t2);
  }
};

template<typename T1, typename T2>
__host__ __device__
actor<
  composite<
    transparent_binary_operator<multiplies_equal>,
    actor<T1>,
    typename as_actor<T2>::type
  >
>
operator*=(const actor<T1> &_1, const T2 &_2)
{
  return compose(transparent_binary_operator<multiplies_equal>(),
                 make_actor(_1),
                 make_actor(_2));
} // end operator*=()

template<typename T1, typename T2>
__host__ __device__
actor<
  composite<
    transparent_binary_operator<multiplies_equal>,
    actor<T1>,
    actor<T2>
  >
>
operator*=(const actor<T1> &_1, const actor<T2> &_2)
{
  return compose(transparent_binary_operator<multiplies_equal>(),
                 make_actor(_1),
                 make_actor(_2));
} // end operator*=()

// there's no standard divides_equal functional, so roll an ad hoc one here
struct divides_equal
{
  using is_transparent = void;

  __thrust_exec_check_disable__
  template <typename T1, typename T2>
  __host__ __device__
  constexpr auto operator()(T1&& t1, T2&& t2) const
  noexcept(noexcept(THRUST_FWD(t1) /= THRUST_FWD(t2)))
  THRUST_TRAILING_RETURN(decltype(THRUST_FWD(t1) /= THRUST_FWD(t2)))
  {
    return THRUST_FWD(t1) /= THRUST_FWD(t2);
  }
};

template<typename T1, typename T2>
__host__ __device__
actor<
  composite<
    transparent_binary_operator<divides_equal>,
    actor<T1>,
    typename as_actor<T2>::type
  >
>
operator/=(const actor<T1> &_1, const T2 &_2)
{
  return compose(transparent_binary_operator<divides_equal>(),
                 make_actor(_1),
                 make_actor(_2));
} // end operator/=()

template<typename T1, typename T2>
__host__ __device__
actor<
  composite<
    transparent_binary_operator<divides_equal>,
    actor<T1>,
    actor<T2>
  >
>
operator/=(const actor<T1> &_1, const actor<T2> &_2)
{
  return compose(transparent_binary_operator<divides_equal>(),
                 make_actor(_1),
                 make_actor(_2));
} // end operator/=()

// there's no standard modulus_equal functional, so roll an ad hoc one here
struct modulus_equal
{
  using is_transparent = void;

  __thrust_exec_check_disable__
  template <typename T1, typename T2>
  __host__ __device__
  constexpr auto operator()(T1&& t1, T2&& t2) const
  noexcept(noexcept(THRUST_FWD(t1) %= THRUST_FWD(t2)))
  THRUST_TRAILING_RETURN(decltype(THRUST_FWD(t1) %= THRUST_FWD(t2)))
  {
    return THRUST_FWD(t1) %= THRUST_FWD(t2);
  }
};

template<typename T1, typename T2>
__host__ __device__
actor<
  composite<
    transparent_binary_operator<modulus_equal>,
    actor<T1>,
    typename as_actor<T2>::type
  >
>
operator%=(const actor<T1> &_1, const T2 &_2)
{
  return compose(transparent_binary_operator<modulus_equal>(),
                 make_actor(_1),
                 make_actor(_2));
} // end operator%=()

template<typename T1, typename T2>
__host__ __device__
actor<
  composite<
    transparent_binary_operator<modulus_equal>,
    actor<T1>,
    actor<T2>
  >
>
operator%=(const actor<T1> &_1, const actor<T2> &_2)
{
  return compose(transparent_binary_operator<modulus_equal>(),
                 make_actor(_1),
                 make_actor(_2));
} // end operator%=()

// there's no standard bit_and_equal functional, so roll an ad hoc one here
struct bit_and_equal
{
  using is_transparent = void;

  __thrust_exec_check_disable__
  template <typename T1, typename T2>
  __host__ __device__
  constexpr auto operator()(T1&& t1, T2&& t2) const
  noexcept(noexcept(THRUST_FWD(t1) &= THRUST_FWD(t2)))
  THRUST_TRAILING_RETURN(decltype(THRUST_FWD(t1) &= THRUST_FWD(t2)))
  {
    return THRUST_FWD(t1) &= THRUST_FWD(t2);
  }
};

template<typename T1, typename T2>
__host__ __device__
actor<
  composite<
    transparent_binary_operator<bit_and_equal>,
    actor<T1>,
    typename as_actor<T2>::type
  >
>
operator&=(const actor<T1> &_1, const T2 &_2)
{
  return compose(transparent_binary_operator<bit_and_equal>(),
                 make_actor(_1),
                 make_actor(_2));
} // end operator&=()

template<typename T1, typename T2>
__host__ __device__
actor<
  composite<
    transparent_binary_operator<bit_and_equal>,
    actor<T1>,
    actor<T2>
  >
>
operator&=(const actor<T1> &_1, const actor<T2> &_2)
{
  return compose(transparent_binary_operator<bit_and_equal>(),
                 make_actor(_1),
                 make_actor(_2));
} // end operator&=()

// there's no standard bit_or_equal functional, so roll an ad hoc one here
struct bit_or_equal
{
  using is_transparent = void;

  __thrust_exec_check_disable__
  template <typename T1, typename T2>
  __host__ __device__
  constexpr auto operator()(T1&& t1, T2&& t2) const
  noexcept(noexcept(THRUST_FWD(t1) |= THRUST_FWD(t2)))
  THRUST_TRAILING_RETURN(decltype(THRUST_FWD(t1) |= THRUST_FWD(t2)))
  {
    return THRUST_FWD(t1) |= THRUST_FWD(t2);
  }
};

template<typename T1, typename T2>
__host__ __device__
actor<
  composite<
    transparent_binary_operator<bit_or_equal>,
    actor<T1>,
    typename as_actor<T2>::type
  >
>
operator|=(const actor<T1> &_1, const T2 &_2)
{
  return compose(transparent_binary_operator<bit_or_equal>(),
                 make_actor(_1),
                 make_actor(_2));
} // end operator|=()

template<typename T1, typename T2>
__host__ __device__
actor<
  composite<
    transparent_binary_operator<bit_or_equal>,
    actor<T1>,
    actor<T2>
  >
>
operator|=(const actor<T1> &_1, const actor<T2> &_2)
{
  return compose(transparent_binary_operator<bit_or_equal>(),
                 make_actor(_1),
                 make_actor(_2));
} // end operator|=()

// there's no standard bit_xor_equal functional, so roll an ad hoc one here
struct bit_xor_equal
{
  using is_transparent = void;

  __thrust_exec_check_disable__
  template <typename T1, typename T2>
  __host__ __device__
  constexpr auto operator()(T1&& t1, T2&& t2) const
  noexcept(noexcept(THRUST_FWD(t1) ^= THRUST_FWD(t2)))
  THRUST_TRAILING_RETURN(decltype(THRUST_FWD(t1) ^= THRUST_FWD(t2)))
  {
    return THRUST_FWD(t1) ^= THRUST_FWD(t2);
  }
};

template<typename T1, typename T2>
__host__ __device__
actor<
  composite<
    transparent_binary_operator<bit_xor_equal>,
    actor<T1>,
    typename as_actor<T2>::type
  >
>
operator^=(const actor<T1> &_1, const T2 &_2)
{
  return compose(transparent_binary_operator<bit_xor_equal>(),
                 make_actor(_1),
                 make_actor(_2));
} // end operator|=()

template<typename T1, typename T2>
__host__ __device__
actor<
  composite<
    transparent_binary_operator<bit_xor_equal>,
    actor<T1>,
    actor<T2>
  >
>
operator^=(const actor<T1> &_1, const actor<T2> &_2)
{
  return compose(transparent_binary_operator<bit_xor_equal>(),
                 make_actor(_1),
                 make_actor(_2));
} // end operator|=()

// there's no standard bit_lshift_equal functional, so roll an ad hoc one here
struct bit_lshift_equal
{
  using is_transparent = void;

  __thrust_exec_check_disable__
  template <typename T1, typename T2>
  __host__ __device__
  constexpr auto operator()(T1&& t1, T2&& t2) const
  noexcept(noexcept(THRUST_FWD(t1) <<= THRUST_FWD(t2)))
  THRUST_TRAILING_RETURN(decltype(THRUST_FWD(t1) <<= THRUST_FWD(t2)))
  {
    return THRUST_FWD(t1) <<= THRUST_FWD(t2);
  }
};
template<typename T1, typename T2>
__host__ __device__
actor<
  composite<
    transparent_binary_operator<bit_lshift_equal>,
    actor<T1>,
    typename as_actor<T2>::type
  >
>
operator<<=(const actor<T1> &_1, const T2 &_2)
{
  return compose(transparent_binary_operator<bit_lshift_equal>(),
                 make_actor(_1),
                 make_actor(_2));
} // end operator<<=()

template<typename T1, typename T2>
__host__ __device__
actor<
  composite<
    transparent_binary_operator<bit_lshift_equal>,
    actor<T1>,
    actor<T2>
  >
>
operator<<=(const actor<T1> &_1, const actor<T2> &_2)
{
  return compose(transparent_binary_operator<bit_lshift_equal>(),
                 make_actor(_1),
                 make_actor(_2));
} // end operator<<=()

// there's no standard bit_rshift_equal functional, so roll an ad hoc one here
struct bit_rshift_equal
{
  using is_transparent = void;

  __thrust_exec_check_disable__
  template <typename T1, typename T2>
  __host__ __device__
  constexpr auto operator()(T1&& t1, T2&& t2) const
  noexcept(noexcept(THRUST_FWD(t1) >>= THRUST_FWD(t2)))
  THRUST_TRAILING_RETURN(decltype(THRUST_FWD(t1) >>= THRUST_FWD(t2)))
  {
    return THRUST_FWD(t1) >>= THRUST_FWD(t2);
  }
};

template<typename T1, typename T2>
__host__ __device__
actor<
  composite<
    transparent_binary_operator<bit_rshift_equal>,
    actor<T1>,
    typename as_actor<T2>::type
  >
>
operator>>=(const actor<T1> &_1, const T2 &_2)
{
  return compose(transparent_binary_operator<bit_rshift_equal>(),
                 make_actor(_1),
                 make_actor(_2));
} // end operator>>=()

template<typename T1, typename T2>
__host__ __device__
actor<
  composite<
    transparent_binary_operator<bit_rshift_equal>,
    actor<T1>,
    actor<T2>
  >
>
operator>>=(const actor<T1> &_1, const actor<T2> &_2)
{
  return compose(transparent_binary_operator<bit_rshift_equal>(),
                 make_actor(_1),
                 make_actor(_2));
} // end operator>>=()

} // end functional
} // end detail
THRUST_NAMESPACE_END

