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


/*! \file uniform_real_distribution.h
 *  \brief A uniform distribution of real-valued numbers
 */

#pragma once

#include <thrust/detail/config.h>
#include <thrust/pair.h>
#include <thrust/random/detail/random_core_access.h>
#include <iostream>

THRUST_NAMESPACE_BEGIN

namespace random
{


/*! \addtogroup random_number_distributions
 *  \{
 */

/*! \class uniform_real_distribution
 *  \brief A \p uniform_real_distribution random number distribution produces floating point
 *         uniform random numbers from a half-open interval.
 *
 *  \tparam RealType The type of floating point number to produce.
 *
 *  The following code snippet demonstrates examples of using a \p uniform_real_distribution with a 
 *  random number engine to produce random integers drawn from a given range:
 *
 *  \code
 *  #include <thrust/random/linear_congruential_engine.h>
 *  #include <thrust/random/uniform_real_distribution.h>
 *
 *  int main(void)
 *  {
 *    // create a minstd_rand object to act as our source of randomness
 *    thrust::minstd_rand rng;
 *
 *    // create a uniform_real_distribution to produce floats from [-7,13)
 *    thrust::uniform_real_distribution<float> dist(-7,13);
 *
 *    // write a random number from the range [-7,13) to standard output
 *    std::cout << dist(rng) << std::endl;
 *
 *    // write the range of the distribution, just in case we forgot
 *    std::cout << dist.min() << std::endl;
 *
 *    // -7.0 is printed
 *
 *    std::cout << dist.max() << std::endl;
 *
 *    // 13.0 is printed
 *
 *    // write the parameters of the distribution (which happen to be the bounds) to standard output
 *    std::cout << dist.a() << std::endl;
 *
 *    // -7.0 is printed
 *
 *    std::cout << dist.b() << std::endl;
 *
 *    // 13.0 is printed
 *
 *    return 0;
 *  }
 *  \endcode
 */
template<typename RealType = double>
  class uniform_real_distribution
{
  public:
    // types
    
    /*! \typedef result_type
     *  \brief The type of the floating point number produced by this \p uniform_real_distribution.
     */
    typedef RealType result_type;

    /*! \typedef param_type
     *  \brief The type of the object encapsulating this \p uniform_real_distribution's parameters.
     */
    typedef thrust::pair<RealType,RealType> param_type;

    // constructors and reset functions
    
    /*! This constructor creates a new \p uniform_real_distribution from two values defining the
     *  half-open interval of the distribution.
     *  
     *  \param a The smallest floating point number to potentially produce. Defaults to \c 0.0.
     *  \param b The smallest number larger than the largest floating point number to potentially produce. Defaults to \c 1.0.
     */
    __host__ __device__
    explicit uniform_real_distribution(RealType a = 0.0, RealType b = 1.0);

    /*! This constructor creates a new \p uniform_real_distribution from a \p param_type object
     *  encapsulating the range of the distribution.
     *  
     *  \param parm A \p param_type object encapsulating the parameters (i.e., the range) of the distribution.
     */
    __host__ __device__
    explicit uniform_real_distribution(const param_type &parm);

    /*! This does nothing.  It is included to conform to the requirements of the RandomDistribution concept.
     */
    __host__ __device__
    void reset(void);

    // generating functions

    /*! This method produces a new uniform random integer drawn from this \p uniform_real_distribution's
     *  range using a \p UniformRandomNumberGenerator as a source of randomness.
     *
     *  \param urng The \p UniformRandomNumberGenerator to use as a source of randomness.
     */
    template<typename UniformRandomNumberGenerator>
    __host__ __device__
    result_type operator()(UniformRandomNumberGenerator &urng);

    /*! This method produces a new uniform random integer as if by creating a new \p uniform_real_distribution 
     *  from the given \p param_type object, and calling its <tt>operator()</tt> method with the given
     *  \p UniformRandomNumberGenerator as a source of randomness.
     *
     *  \param urng The \p UniformRandomNumberGenerator to use as a source of randomness.
     *  \param parm A \p param_type object encapsulating the parameters of the \p uniform_real_distribution
     *              to draw from.
     */
    template<typename UniformRandomNumberGenerator>
    __host__ __device__
    result_type operator()(UniformRandomNumberGenerator &urng, const param_type &parm);

    // property functions

    /*! This method returns the value of the parameter with which this \p uniform_real_distribution
     *  was constructed.
     *
     *  \return The lower bound of this \p uniform_real_distribution's half-open interval.
     */
    __host__ __device__
    result_type a(void) const;

    /*! This method returns the value of the parameter with which this \p uniform_real_distribution
     *  was constructed.
     *
     *  \return The upper bound of this \p uniform_real_distribution's half-open interval.
     */
    __host__ __device__
    result_type b(void) const;

    /*! This method returns a \p param_type object encapsulating the parameters with which this
     *  \p uniform_real_distribution was constructed.
     *
     *  \return A \p param_type object enapsulating the half-open interval of this \p uniform_real_distribution.
     */
    __host__ __device__
    param_type param(void) const;

    /*! This method changes the parameters of this \p uniform_real_distribution using the values encapsulated
     *  in a given \p param_type object.
     *
     *  \param parm A \p param_type object encapsulating the new half-open interval of this \p uniform_real_distribution.
     */
    __host__ __device__
    void param(const param_type &parm);

    /*! This method returns the smallest floating point number this \p uniform_real_distribution can potentially produce.
     *
     *  \return The lower bound of this \p uniform_real_distribution's half-open interval.
     */
    __host__ __device__
    result_type min THRUST_PREVENT_MACRO_SUBSTITUTION (void) const;

    /*! This method returns the smallest number larger than largest floating point number this \p uniform_real_distribution can potentially produce.
     *
     *  \return The upper bound of this \p uniform_real_distribution's half-open interval.
     */
    __host__ __device__
    result_type max THRUST_PREVENT_MACRO_SUBSTITUTION (void) const;

    /*! \cond
     */
  private:
    param_type m_param;

    friend struct thrust::random::detail::random_core_access;

    __host__ __device__
    bool equal(const uniform_real_distribution &rhs) const;

    template<typename CharT, typename Traits>
    std::basic_ostream<CharT,Traits>& stream_out(std::basic_ostream<CharT,Traits> &os) const;

    template<typename CharT, typename Traits>
    std::basic_istream<CharT,Traits>& stream_in(std::basic_istream<CharT,Traits> &is);
    /*! \endcond
     */
}; // end uniform_real_distribution


/*! This function checks two \p uniform_real_distributions for equality.
 *  \param lhs The first \p uniform_real_distribution to test.
 *  \param rhs The second \p uniform_real_distribution to test.
 *  \return \c true if \p lhs is equal to \p rhs; \c false, otherwise.
 */
template<typename RealType>
__host__ __device__
bool operator==(const uniform_real_distribution<RealType> &lhs,
                const uniform_real_distribution<RealType> &rhs);


/*! This function checks two \p uniform_real_distributions for inequality.
 *  \param lhs The first \p uniform_real_distribution to test.
 *  \param rhs The second \p uniform_real_distribution to test.
 *  \return \c true if \p lhs is not equal to \p rhs; \c false, otherwise.
 */
template<typename RealType>
__host__ __device__
bool operator!=(const uniform_real_distribution<RealType> &lhs,
                const uniform_real_distribution<RealType> &rhs);


/*! This function streams a uniform_real_distribution to a \p std::basic_ostream.
 *  \param os The \p basic_ostream to stream out to.
 *  \param d The \p uniform_real_distribution to stream out.
 *  \return \p os
 */
template<typename RealType,
         typename CharT, typename Traits>
std::basic_ostream<CharT,Traits>&
operator<<(std::basic_ostream<CharT,Traits> &os,
           const uniform_real_distribution<RealType> &d);


/*! This function streams a uniform_real_distribution in from a std::basic_istream.
 *  \param is The \p basic_istream to stream from.
 *  \param d The \p uniform_real_distribution to stream in.
 *  \return \p is
 */
template<typename RealType,
         typename CharT, typename Traits>
std::basic_istream<CharT,Traits>&
operator>>(std::basic_istream<CharT,Traits> &is,
           uniform_real_distribution<RealType> &d);


/*! \} // end random_number_distributions
 */


} // end random

using random::uniform_real_distribution;

THRUST_NAMESPACE_END

#include <thrust/random/detail/uniform_real_distribution.inl>

