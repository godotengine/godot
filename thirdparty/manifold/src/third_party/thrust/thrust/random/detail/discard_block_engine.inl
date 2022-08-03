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

#include <thrust/random/discard_block_engine.h>

THRUST_NAMESPACE_BEGIN

namespace random
{


template<typename Engine, size_t p, size_t r>
  __host__ __device__
  discard_block_engine<Engine,p,r>
    ::discard_block_engine()
      : m_e(), m_n(0)
{}


template<typename Engine, size_t p, size_t r>
  __host__ __device__
  discard_block_engine<Engine,p,r>
    ::discard_block_engine(result_type s)
      : m_e(s), m_n(0)
{}


template<typename Engine, size_t p, size_t r>
  __host__ __device__
  discard_block_engine<Engine,p,r>
    ::discard_block_engine(const base_type &urng)
      : m_e(urng), m_n(0)
{}


template<typename Engine, size_t p, size_t r>
  __host__ __device__
  void discard_block_engine<Engine,p,r>
    ::seed(void)
{
  m_e.seed();
  m_n = 0;
}


template<typename Engine, size_t p, size_t r>
  __host__ __device__
  void discard_block_engine<Engine,p,r>
    ::seed(result_type s)
{
  m_e.seed(s);
  m_n = 0;
}


template<typename Engine, size_t p, size_t r>
  __host__ __device__
  typename discard_block_engine<Engine,p,r>::result_type
    discard_block_engine<Engine,p,r>
      ::operator()(void)
{
  if(m_n >= used_block)
  {
    m_e.discard(block_size - m_n);
//    for(; m_n < block_size; ++m_n)
//      m_e();
    m_n = 0;
  }

  ++m_n;

  return m_e();
}


template<typename Engine, size_t p, size_t r>
  __host__ __device__
  void discard_block_engine<Engine,p,r>
    ::discard(unsigned long long z)
{
  // XXX this should be accelerated
  for(; z > 0; --z)
  {
    this->operator()();
  } // end for
}


template<typename Engine, size_t p, size_t r>
  __host__ __device__
  const typename discard_block_engine<Engine,p,r>::base_type &
    discard_block_engine<Engine,p,r>
      ::base(void) const
{
  return m_e;
}


template<typename Engine, size_t p, size_t r>
  template<typename CharT, typename Traits>
    std::basic_ostream<CharT,Traits>& discard_block_engine<Engine,p,r>
      ::stream_out(std::basic_ostream<CharT,Traits> &os) const
{
  typedef std::basic_ostream<CharT,Traits> ostream_type;
  typedef typename ostream_type::ios_base  ios_base;

  // save old flags & fill character
  const typename ios_base::fmtflags flags = os.flags();
  const CharT fill = os.fill();

  const CharT space = os.widen(' ');
  os.flags(ios_base::dec | ios_base::fixed | ios_base::left);
  os.fill(space);

  // output the base engine followed by n
  os << m_e << space << m_n;

  // restore flags & fill character
  os.flags(flags);
  os.fill(fill);

  return os;
}


template<typename Engine, size_t p, size_t r>
  template<typename CharT, typename Traits>
    std::basic_istream<CharT,Traits>& discard_block_engine<Engine,p,r>
      ::stream_in(std::basic_istream<CharT,Traits> &is)
{
  typedef std::basic_istream<CharT,Traits> istream_type;
  typedef typename istream_type::ios_base  ios_base;

  // save old flags
  const typename ios_base::fmtflags flags = is.flags();

  is.flags(ios_base::skipws);

  // input the base engine and then n
  is >> m_e >> m_n;

  // restore old flags
  is.flags(flags);
  return is;
}


template<typename Engine, size_t p, size_t r>
  __host__ __device__
  bool discard_block_engine<Engine,p,r>
    ::equal(const discard_block_engine<Engine,p,r> &rhs) const
{
  return (m_e == rhs.m_e) && (m_n == rhs.m_n);
}


template<typename Engine, size_t p, size_t r,
         typename CharT, typename Traits>
std::basic_ostream<CharT,Traits>&
operator<<(std::basic_ostream<CharT,Traits> &os,
           const discard_block_engine<Engine,p,r> &e)
{
  return thrust::random::detail::random_core_access::stream_out(os,e);
}


template<typename Engine, size_t p, size_t r,
         typename CharT, typename Traits>
std::basic_istream<CharT,Traits>&
operator>>(std::basic_istream<CharT,Traits> &is,
           discard_block_engine<Engine,p,r> &e)
{
  return thrust::random::detail::random_core_access::stream_in(is,e);
}


template<typename Engine, size_t p, size_t r>
__host__ __device__
bool operator==(const discard_block_engine<Engine,p,r> &lhs,
                const discard_block_engine<Engine,p,r> &rhs)
{
  return thrust::random::detail::random_core_access::equal(lhs,rhs);
}


template<typename Engine, size_t p, size_t r>
__host__ __device__
bool operator!=(const discard_block_engine<Engine,p,r> &lhs,
                const discard_block_engine<Engine,p,r> &rhs)
{
  return !(lhs == rhs);
}


} // end random

THRUST_NAMESPACE_END

