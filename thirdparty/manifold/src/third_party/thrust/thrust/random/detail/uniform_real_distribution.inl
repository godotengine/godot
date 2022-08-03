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

#include <thrust/random/uniform_real_distribution.h>

THRUST_NAMESPACE_BEGIN

namespace random
{


template<typename RealType>
  __host__ __device__
  uniform_real_distribution<RealType>
    ::uniform_real_distribution(RealType a, RealType b)
      :m_param(a,b)
{
} // end uniform_real_distribution::uniform_real_distribution()

template<typename RealType>
  __host__ __device__
  uniform_real_distribution<RealType>
    ::uniform_real_distribution(const param_type &parm)
      :m_param(parm)
{
} // end uniform_real_distribution::uniform_real_distribution()

template<typename RealType>
  __host__ __device__
  void uniform_real_distribution<RealType>
    ::reset(void)
{
} // end uniform_real_distribution::reset()

template<typename RealType>
  template<typename UniformRandomNumberGenerator>
    __host__ __device__
    typename uniform_real_distribution<RealType>::result_type
      uniform_real_distribution<RealType>
        ::operator()(UniformRandomNumberGenerator &urng)
{
  return operator()(urng, m_param);
} // end uniform_real::operator()()

template<typename RealType>
  template<typename UniformRandomNumberGenerator>
    __host__ __device__
    typename uniform_real_distribution<RealType>::result_type
      uniform_real_distribution<RealType>
        ::operator()(UniformRandomNumberGenerator &urng,
                     const param_type &parm)
{
  // call the urng & map its result to [0,1)
  result_type result = static_cast<result_type>(urng() - UniformRandomNumberGenerator::min);

  // adding one to the denominator ensures that the interval is half-open at 1.0
  // XXX adding 1.0 to a potentially large floating point number seems like a bad idea
  // XXX OTOH adding 1 to what is potentially UINT_MAX also seems like a bad idea
  // XXX we could statically check if 1u + (max - min) is representable and do that, otherwise use the current implementation
  result /= (result_type(1) + static_cast<result_type>(UniformRandomNumberGenerator::max - UniformRandomNumberGenerator::min));

  return (result * (parm.second - parm.first)) + parm.first;
} // end uniform_real::operator()()

template<typename RealType>
  __host__ __device__
  typename uniform_real_distribution<RealType>::result_type
    uniform_real_distribution<RealType>
      ::a(void) const
{
  return m_param.first;
} // end uniform_real::a()

template<typename RealType>
  __host__ __device__
  typename uniform_real_distribution<RealType>::result_type
    uniform_real_distribution<RealType>
      ::b(void) const
{
  return m_param.second;
} // end uniform_real_distribution::b()

template<typename RealType>
  __host__ __device__
  typename uniform_real_distribution<RealType>::param_type
    uniform_real_distribution<RealType>
      ::param(void) const
{
  return m_param;;
} // end uniform_real_distribution::param()

template<typename RealType>
  __host__ __device__
  void uniform_real_distribution<RealType>
    ::param(const param_type &parm)
{
  m_param = parm;
} // end uniform_real_distribution::param()

template<typename RealType>
  __host__ __device__
  typename uniform_real_distribution<RealType>::result_type
    uniform_real_distribution<RealType>
      ::min THRUST_PREVENT_MACRO_SUBSTITUTION (void) const
{
  return a();
} // end uniform_real_distribution::min()

template<typename RealType>
  __host__ __device__
  typename uniform_real_distribution<RealType>::result_type
    uniform_real_distribution<RealType>
      ::max THRUST_PREVENT_MACRO_SUBSTITUTION (void) const
{
  return b();
} // end uniform_real_distribution::max()


template<typename RealType>
  __host__ __device__
  bool uniform_real_distribution<RealType>
    ::equal(const uniform_real_distribution &rhs) const
{
  return m_param == rhs.param();
}


template<typename RealType>
  template<typename CharT, typename Traits>
    std::basic_ostream<CharT,Traits>&
      uniform_real_distribution<RealType>
        ::stream_out(std::basic_ostream<CharT,Traits> &os) const
{
  typedef std::basic_ostream<CharT,Traits> ostream_type;
  typedef typename ostream_type::ios_base  ios_base;

  // save old flags and fill character
  const typename ios_base::fmtflags flags = os.flags();
  const CharT fill = os.fill();

  const CharT space = os.widen(' ');
  os.flags(ios_base::dec | ios_base::fixed | ios_base::left);
  os.fill(space);

  os << a() << space << b();

  // restore old flags and fill character
  os.flags(flags);
  os.fill(fill);
  return os;
}


template<typename RealType>
  template<typename CharT, typename Traits>
    std::basic_istream<CharT,Traits>&
      uniform_real_distribution<RealType>
        ::stream_in(std::basic_istream<CharT,Traits> &is)
{
  typedef std::basic_istream<CharT,Traits> istream_type;
  typedef typename istream_type::ios_base  ios_base;

  // save old flags
  const typename ios_base::fmtflags flags = is.flags();

  is.flags(ios_base::skipws);

  is >> m_param.first >> m_param.second;

  // restore old flags
  is.flags(flags);
  return is;
}


template<typename RealType>
__host__ __device__
bool operator==(const uniform_real_distribution<RealType> &lhs,
                const uniform_real_distribution<RealType> &rhs)
{
  return thrust::random::detail::random_core_access::equal(lhs,rhs);
}


template<typename RealType>
__host__ __device__
bool operator!=(const uniform_real_distribution<RealType> &lhs,
                const uniform_real_distribution<RealType> &rhs)
{
  return !(lhs == rhs);
}


template<typename RealType,
         typename CharT, typename Traits>
std::basic_ostream<CharT,Traits>&
operator<<(std::basic_ostream<CharT,Traits> &os,
           const uniform_real_distribution<RealType> &d)
{
  return thrust::random::detail::random_core_access::stream_out(os,d);
}


template<typename RealType,
         typename CharT, typename Traits>
std::basic_istream<CharT,Traits>&
operator>>(std::basic_istream<CharT,Traits> &is,
           uniform_real_distribution<RealType> &d)
{
  return thrust::random::detail::random_core_access::stream_in(is,d);
}


} // end random

THRUST_NAMESPACE_END

