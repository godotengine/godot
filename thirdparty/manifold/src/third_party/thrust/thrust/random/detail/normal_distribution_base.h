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

/*
 * Copyright Jens Maurer 2000-2001
 * Distributed under the Boost Software License, Version 1.0. (See
 * accompanying file LICENSE_1_0.txt or copy at
 * http://www.boost.org/LICENSE_1_0.txt)
 */

#pragma once

#include <thrust/detail/config.h>
#include <thrust/pair.h>
#include <thrust/random/uniform_real_distribution.h>
#include <limits>
#include <cmath>

THRUST_NAMESPACE_BEGIN
namespace random
{
namespace detail
{

// this version samples the normal distribution directly
// and uses the non-standard math function erfcinv
template<typename RealType>
  class normal_distribution_nvcc
{
  protected:
    template<typename UniformRandomNumberGenerator>
    __host__ __device__
    RealType sample(UniformRandomNumberGenerator &urng, const RealType mean, const RealType stddev)
    {
      using uint_type = typename UniformRandomNumberGenerator::result_type;
      constexpr uint_type urng_range = UniformRandomNumberGenerator::max - UniformRandomNumberGenerator::min;

      // Constants for conversion
      constexpr RealType S1 = static_cast<RealType>(1. / static_cast<double>(urng_range));
      constexpr RealType S2 = S1 / 2;

      RealType S3 = static_cast<RealType>(-1.4142135623730950488016887242097); // -sqrt(2)

      // Get the integer value
      uint_type u = urng() - UniformRandomNumberGenerator::min;

      // Ensure the conversion to float will give a value in the range [0,0.5)
      if(u > (urng_range / 2))
      {
        u = urng_range - u;
        S3 = -S3;
      }

      // Convert to floating point in [0,0.5)
      RealType p = u*S1 + S2;

      // Apply inverse error function
      return mean + stddev * S3 * erfcinv(2 * p);
    }

    // no-op
    __host__ __device__
    void reset() {}
};

// this version samples the normal distribution using
// Marsaglia's "polar method"
template<typename RealType>
  class normal_distribution_portable
{
  protected:
    normal_distribution_portable()
      : m_r1(), m_r2(), m_cached_rho(), m_valid(false)
    {}

    normal_distribution_portable(const normal_distribution_portable &other)
      : m_r1(other.m_r1), m_r2(other.m_r2), m_cached_rho(other.m_cached_rho), m_valid(other.m_valid)
    {}

    void reset()
    {
      m_valid = false;
    }

    // note that we promise to call this member function with the same mean and stddev
    template<typename UniformRandomNumberGenerator>
    __host__ __device__
    RealType sample(UniformRandomNumberGenerator &urng, const RealType mean, const RealType stddev)
    {
      // implementation from Boost
      // allow for Koenig lookup
      using std::sqrt; using std::log; using std::sin; using std::cos;

      if(!m_valid)
      {
        uniform_real_distribution<RealType> u01;
        m_r1 = u01(urng);
        m_r2 = u01(urng);
        m_cached_rho = sqrt(-RealType(2) * log(RealType(1)-m_r2));

        m_valid = true;
      }
      else
      {
        m_valid = false;
      }

      const RealType pi = RealType(3.14159265358979323846);

      RealType result = m_cached_rho * (m_valid ?
                          cos(RealType(2)*pi*m_r1) :
                          sin(RealType(2)*pi*m_r1));

      return mean + stddev * result;
    }

  private:
    RealType m_r1, m_r2, m_cached_rho;
    bool m_valid;
};

template<typename RealType>
  struct normal_distribution_base
{
#if THRUST_DEVICE_COMPILER == THRUST_DEVICE_COMPILER_NVCC && !defined(_NVHPC_CUDA)
  typedef normal_distribution_nvcc<RealType> type;
#else
  typedef normal_distribution_portable<RealType> type;
#endif
};

} // end detail
} // end random
THRUST_NAMESPACE_END

