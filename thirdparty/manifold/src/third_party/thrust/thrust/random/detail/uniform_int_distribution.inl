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

#include <thrust/random/uniform_int_distribution.h>
#include <thrust/random/uniform_real_distribution.h>
#include <thrust/detail/type_traits.h>

THRUST_NAMESPACE_BEGIN

namespace random
{


template<typename IntType>
  __host__ __device__
  uniform_int_distribution<IntType>
    ::uniform_int_distribution(IntType a, IntType b)
      :m_param(a,b)
{
} // end uniform_int_distribution::uniform_int_distribution()


template<typename IntType>
  __host__ __device__
  uniform_int_distribution<IntType>
    ::uniform_int_distribution(const param_type &parm)
      :m_param(parm)
{
} // end uniform_int_distribution::uniform_int_distribution()


template<typename IntType>
  __host__ __device__
  void uniform_int_distribution<IntType>
    ::reset(void)
{
} // end uniform_int_distribution::reset()


template<typename IntType>
  template<typename UniformRandomNumberGenerator>
    __host__ __device__
    typename uniform_int_distribution<IntType>::result_type
      uniform_int_distribution<IntType>
        ::operator()(UniformRandomNumberGenerator &urng)
{
  return operator()(urng, m_param);
} // end uniform_int_distribution::operator()()


template<typename IntType>
  template<typename UniformRandomNumberGenerator>
    __host__ __device__
    typename uniform_int_distribution<IntType>::result_type
      uniform_int_distribution<IntType>
        ::operator()(UniformRandomNumberGenerator &urng, const param_type &parm)
{
  // XXX this implementation is somewhat hacky and will skip
  //     values if the range of the RNG is smaller than the range of the distribution
  //     we should improve this implementation in a later version

  typedef typename thrust::detail::largest_available_float::type float_type;

  const float_type real_min(static_cast<float_type>(parm.first));
  const float_type real_max(static_cast<float_type>(parm.second));

  // add one to the right end of the interval because it is half-open
  // XXX adding 1.0 to a potentially large floating point number seems like a bad idea
  uniform_real_distribution<float_type> real_dist(real_min, real_max + float_type(1));

  return static_cast<result_type>(real_dist(urng));
} // end uniform_int_distribution::operator()()


template<typename IntType>
  __host__ __device__
  typename uniform_int_distribution<IntType>::result_type
    uniform_int_distribution<IntType>
      ::a(void) const
{
  return m_param.first;
} // end uniform_int_distribution<IntType>::a()


template<typename IntType>
  __host__ __device__
  typename uniform_int_distribution<IntType>::result_type
    uniform_int_distribution<IntType>
      ::b(void) const
{
  return m_param.second;
} // end uniform_int_distribution::b()


template<typename IntType>
  __host__ __device__
  typename uniform_int_distribution<IntType>::param_type
    uniform_int_distribution<IntType>
      ::param(void) const
{
  return m_param;
} // end uniform_int_distribution::param()


template<typename IntType>
  __host__ __device__
  void uniform_int_distribution<IntType>
    ::param(const param_type &parm)
{
  m_param = parm;
} // end uniform_int_distribution::param()


template<typename IntType>
  __host__ __device__
  typename uniform_int_distribution<IntType>::result_type
    uniform_int_distribution<IntType>
      ::min THRUST_PREVENT_MACRO_SUBSTITUTION (void) const
{
  return a();
} // end uniform_int_distribution::min()


template<typename IntType>
  __host__ __device__
  typename uniform_int_distribution<IntType>::result_type
    uniform_int_distribution<IntType>
      ::max THRUST_PREVENT_MACRO_SUBSTITUTION (void) const
{
  return b();
} // end uniform_int_distribution::max()


template<typename IntType>
  __host__ __device__
  bool uniform_int_distribution<IntType>
    ::equal(const uniform_int_distribution &rhs) const
{
  return param() == rhs.param();
}


template<typename IntType>
  template<typename CharT, typename Traits>
    std::basic_ostream<CharT,Traits>&
      uniform_int_distribution<IntType>
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


template<typename IntType>
  template<typename CharT, typename Traits>
    std::basic_istream<CharT,Traits>&
      uniform_int_distribution<IntType>
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


template<typename IntType>
__host__ __device__
bool operator==(const uniform_int_distribution<IntType> &lhs,
                const uniform_int_distribution<IntType> &rhs)
{
  return thrust::random::detail::random_core_access::equal(lhs,rhs);
}


template<typename IntType>
__host__ __device__
bool operator!=(const uniform_int_distribution<IntType> &lhs,
                const uniform_int_distribution<IntType> &rhs)
{
  return !(lhs == rhs);
}


template<typename IntType,
         typename CharT, typename Traits>
std::basic_ostream<CharT,Traits>&
operator<<(std::basic_ostream<CharT,Traits> &os,
           const uniform_int_distribution<IntType> &d)
{
  return thrust::random::detail::random_core_access::stream_out(os,d);
}


template<typename IntType,
         typename CharT, typename Traits>
std::basic_istream<CharT,Traits>&
operator>>(std::basic_istream<CharT,Traits> &is,
           uniform_int_distribution<IntType> &d)
{
  return thrust::random::detail::random_core_access::stream_in(is,d);
}


} // end random

THRUST_NAMESPACE_END

