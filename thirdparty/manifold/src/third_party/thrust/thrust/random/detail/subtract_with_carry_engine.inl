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

#include <thrust/random/linear_congruential_engine.h>
#include <thrust/random/subtract_with_carry_engine.h>
#include <thrust/random/detail/mod.h>
#include <thrust/random/detail/random_core_access.h>

THRUST_NAMESPACE_BEGIN

namespace random
{


template<typename UIntType, size_t w, size_t s, size_t r>
  __host__ __device__
  subtract_with_carry_engine<UIntType,w,s,r>
    ::subtract_with_carry_engine(result_type value)
{
  seed(value);
} // end subtract_with_carry_engine::subtract_with_carry_engine()


template<typename UIntType, size_t w, size_t s, size_t r>
  __host__ __device__
  void subtract_with_carry_engine<UIntType,w,s,r>
    ::seed(result_type value)
{
  thrust::random::linear_congruential_engine<result_type,
    40014u, 0u, 2147483563u> e(value == 0u ? default_seed : value);

  // initialize state
  for(size_t i = 0; i < long_lag; ++i)
  {
    m_x[i] = detail::mod<UIntType, 1, 0, modulus>(e());
  } // end for i

  m_carry = (m_x[long_lag-1] == 0);
  m_k = 0;
} // end subtract_with_carry_engine::seed()


template<typename UIntType, size_t w, size_t s, size_t r>
  __host__ __device__
  typename subtract_with_carry_engine<UIntType,w,s,r>::result_type
    subtract_with_carry_engine<UIntType,w,s,r>
      ::operator()(void)
{
  // XXX we probably need to cache these m_x[m_k] in a register
  //     maybe we need to cache the use of all member variables
  int short_index = m_k - short_lag;
  if(short_index < 0)
    short_index += long_lag;
  result_type xi;
  if (m_x[short_index] >= m_x[m_k] + m_carry)
  {
    // x(n) >= 0
    xi =  m_x[short_index] - m_x[m_k] - m_carry;
    m_carry = 0;
  }
  else
  {
    // x(n) < 0
    xi = modulus - m_x[m_k] - m_carry + m_x[short_index];
    m_carry = 1;
  }
  m_x[m_k] = xi;
  ++m_k;
  if(m_k >= long_lag)
    m_k = 0;
  return xi;
} // end subtract_with_carry_engine::operator()()


template<typename UIntType, size_t w, size_t s, size_t r>
  __host__ __device__
  void subtract_with_carry_engine<UIntType,w,s,r>
    ::discard(unsigned long long z)
{
  for(; z > 0; --z)
  {
    this->operator()();
  } // end for
} // end subtract_with_carry_engine::discard()


template<typename UIntType, size_t w, size_t s, size_t r>
  template<typename CharT, typename Traits>
    std::basic_ostream<CharT,Traits>& subtract_with_carry_engine<UIntType,w,s,r>
      ::stream_out(std::basic_ostream<CharT,Traits> &os) const
{
  typedef std::basic_ostream<CharT,Traits> ostream_type;
  typedef typename ostream_type::ios_base     ios_base;

  const typename ios_base::fmtflags flags = os.flags();
  const CharT fill  = os.fill();
  const CharT space = os.widen(' ');
  os.flags(ios_base::dec | ios_base::fixed | ios_base::left);
  os.fill(space);

  const UIntType long_lag_ = r;

  for(size_t i = 0; i < r; ++i)
    os << m_x[(i + m_k) % long_lag_] << space;
  os << m_carry;

  os.flags(flags);
  os.fill(fill);
  return os;
}


template<typename UIntType, size_t w, size_t s, size_t r>
  template<typename CharType, typename Traits>
    std::basic_istream<CharType,Traits>& subtract_with_carry_engine<UIntType,w,s,r>
      ::stream_in(std::basic_istream<CharType,Traits> &is)
{
  typedef std::basic_istream<CharType,Traits> istream_type;
  typedef typename istream_type::ios_base     ios_base;

  const typename ios_base::fmtflags flags = is.flags();
  is.flags(ios_base::dec | ios_base::skipws);

  for(size_t i = 0; i < r; ++i)
    is >> m_x[i];
  is >> m_carry;

  m_k = 0;

  is.flags(flags);
  return is;
}


template<typename UIntType, size_t w, size_t s, size_t r>
  __host__ __device__
  bool subtract_with_carry_engine<UIntType,w,s,r>
    ::equal(const subtract_with_carry_engine<UIntType,w,s,r> &rhs) const
{
  const UIntType long_lag_ = r;

  bool result = true;
  for(size_t i = 0; i < r; ++i)
  {
    result &= (m_x[(i + m_k) % long_lag_] == rhs.m_x[(i + rhs.m_k) % long_lag_]);
  }

  // XXX not sure if this last check is necessary
  result &= (m_carry == rhs.m_carry);

  return result;
}


template<typename UIntType, size_t w, size_t s, size_t r,
         typename CharT, typename Traits>
  std::basic_ostream<CharT,Traits>&
    operator<<(std::basic_ostream<CharT,Traits> &os,
               const subtract_with_carry_engine<UIntType,w,s,r> &e)
{
  return thrust::random::detail::random_core_access::stream_out(os,e);
}


template<typename UIntType, size_t w, size_t s, size_t r,
         typename CharType, typename Traits>
  std::basic_istream<CharType,Traits>&
    operator>>(std::basic_istream<CharType,Traits> &is,
               subtract_with_carry_engine<UIntType,w,s,r> &e)
{
  return thrust::random::detail::random_core_access::stream_in(is,e);
}


template<typename UIntType, size_t w, size_t s, size_t r>
  __host__ __device__
  bool operator==(const subtract_with_carry_engine<UIntType,w,s,r> &lhs,
                  const subtract_with_carry_engine<UIntType,w,s,r> &rhs)
{
  return thrust::random::detail::random_core_access::equal(lhs,rhs);
}


template<typename UIntType, size_t w, size_t s, size_t r>
  __host__ __device__
  bool operator!=(const subtract_with_carry_engine<UIntType,w,s,r> &lhs,
                  const subtract_with_carry_engine<UIntType,w,s,r> &rhs)
{
  return !(lhs == rhs);
}


} // end random

THRUST_NAMESPACE_END

