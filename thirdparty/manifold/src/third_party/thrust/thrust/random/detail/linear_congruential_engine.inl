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
#include <thrust/random/detail/mod.h>
#include <thrust/random/detail/random_core_access.h>

THRUST_NAMESPACE_BEGIN

namespace random
{


template<typename UIntType, UIntType a, UIntType c, UIntType m>
  __host__ __device__
  linear_congruential_engine<UIntType,a,c,m>
    ::linear_congruential_engine(result_type s)
{
  seed(s);
} // end linear_congruential_engine::linear_congruential_engine()


template<typename UIntType, UIntType a, UIntType c, UIntType m>
  __host__ __device__
  void linear_congruential_engine<UIntType,a,c,m>
    ::seed(result_type s)
{
  if((detail::mod<UIntType, 1, 0, m>(c) == 0) &&
     (detail::mod<UIntType, 1, 0, m>(s) == 0))
    m_x = detail::mod<UIntType, 1, 0, m>(1);
  else
    m_x = detail::mod<UIntType, 1, 0, m>(s);
} // end linear_congruential_engine::seed()


template<typename UIntType, UIntType a, UIntType c, UIntType m>
  __host__ __device__
  typename linear_congruential_engine<UIntType,a,c,m>::result_type
    linear_congruential_engine<UIntType,a,c,m>
      ::operator()(void)
{
  m_x = detail::mod<UIntType,a,c,m>(m_x);
  return m_x;
} // end linear_congruential_engine::operator()()


template<typename UIntType, UIntType a, UIntType c, UIntType m>
  __host__ __device__
  void linear_congruential_engine<UIntType,a,c,m>
    ::discard(unsigned long long z)
{
  thrust::random::detail::linear_congruential_engine_discard::discard(*this,z);
} // end linear_congruential_engine::discard()


template<typename UIntType, UIntType a, UIntType c, UIntType m>
  template<typename CharT, typename Traits>
    std::basic_ostream<CharT,Traits>& linear_congruential_engine<UIntType,a,c,m>
      ::stream_out(std::basic_ostream<CharT,Traits> &os) const
{
  typedef std::basic_ostream<CharT,Traits> ostream_type;
  typedef typename ostream_type::ios_base  ios_base;

  // save old flags & fill character
  const typename ios_base::fmtflags flags = os.flags();
  const CharT fill = os.fill();

  os.flags(ios_base::dec | ios_base::fixed | ios_base::left);
  os.fill(os.widen(' '));

  // output one word of state
  os << m_x;

  // restore flags & fill character
  os.flags(flags);
  os.fill(fill);

  return os;
}


template<typename UIntType, UIntType a, UIntType c, UIntType m>
  template<typename CharT, typename Traits>
    std::basic_istream<CharT,Traits>& linear_congruential_engine<UIntType,a,c,m>
      ::stream_in(std::basic_istream<CharT,Traits> &is)
{
  typedef std::basic_istream<CharT,Traits> istream_type;
  typedef typename istream_type::ios_base     ios_base;

  // save old flags
  const typename ios_base::fmtflags flags = is.flags();

  is.flags(ios_base::dec);

  // input one word of state
  is >> m_x;

  // restore flags
  is.flags(flags);

  return is;
}


template<typename UIntType, UIntType a, UIntType c, UIntType m>
__host__ __device__
bool linear_congruential_engine<UIntType,a,c,m>
  ::equal(const linear_congruential_engine<UIntType,a,c,m> &rhs) const
{
  return m_x == rhs.m_x;
}


template<typename UIntType_, UIntType_ a_, UIntType_ c_, UIntType_ m_>
__host__ __device__
bool operator==(const linear_congruential_engine<UIntType_,a_,c_,m_> &lhs,
                const linear_congruential_engine<UIntType_,a_,c_,m_> &rhs)
{
  return detail::random_core_access::equal(lhs,rhs);
}


template<typename UIntType, UIntType a, UIntType c, UIntType m>
__host__ __device__
bool operator!=(const linear_congruential_engine<UIntType,a,c,m> &lhs,
                const linear_congruential_engine<UIntType,a,c,m> &rhs)
{
  return !(lhs == rhs);
}


template<typename UIntType_, UIntType_ a_, UIntType_ c_, UIntType_ m_,
         typename CharT, typename Traits>
std::basic_ostream<CharT,Traits>&
operator<<(std::basic_ostream<CharT,Traits> &os,
           const linear_congruential_engine<UIntType_,a_,c_,m_> &e)
{
  return detail::random_core_access::stream_out(os,e);
}


template<typename UIntType_, UIntType_ a_, UIntType_ c_, UIntType_ m_,
         typename CharT, typename Traits>
std::basic_istream<CharT,Traits>&
operator>>(std::basic_istream<CharT,Traits> &is,
           linear_congruential_engine<UIntType_,a_,c_,m_> &e)
{
  return detail::random_core_access::stream_in(is,e);
}


} // end random

THRUST_NAMESPACE_END

