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


/*! \file uniform_int_distribution.h
 *  \brief A uniform distribution of integer-valued numbers
 */

#pragma once

#include <thrust/detail/config.h>
#include <thrust/pair.h>
#include <thrust/detail/integer_traits.h>
#include <thrust/random/detail/random_core_access.h>
#include <iostream>

THRUST_NAMESPACE_BEGIN

namespace random
{

/*! \addtogroup random_number_distributions Random Number Distributions Class Templates
 *  \ingroup random
 *  \{
 */

/*! \class uniform_int_distribution
 *  \brief A \p uniform_int_distribution random number distribution produces signed or unsigned integer
 *         uniform random numbers from a given range.
 *
 *  \tparam IntType The type of integer to produce.
 *
 *  The following code snippet demonstrates examples of using a \p uniform_int_distribution with a 
 *  random number engine to produce random integers drawn from a given range:
 *
 *  \code
 *  #include <thrust/random/linear_congruential_engine.h>
 *  #include <thrust/random/uniform_int_distribution.h>
 *
 *  int main(void)
 *  {
 *    // create a minstd_rand object to act as our source of randomness
 *    thrust::minstd_rand rng;
 *
 *    // create a uniform_int_distribution to produce ints from [-7,13]
 *    thrust::uniform_int_distribution<int> dist(-7,13);
 *
 *    // write a random number from the range [-7,13] to standard output
 *    std::cout << dist(rng) << std::endl;
 *
 *    // write the range of the distribution, just in case we forgot
 *    std::cout << dist.min() << std::endl;
 *
 *    // -7 is printed
 *
 *    std::cout << dist.max() << std::endl;
 *
 *    // 13 is printed
 *
 *    // write the parameters of the distribution (which happen to be the bounds) to standard output
 *    std::cout << dist.a() << std::endl;
 *
 *    // -7 is printed
 *
 *    std::cout << dist.b() << std::endl;
 *
 *    // 13 is printed
 *
 *    return 0;
 *  }
 *  \endcode
 */
template<typename IntType = int>
  class uniform_int_distribution
{
  public:
    // types

    /*! \typedef result_type
     *  \brief The type of the integer produced by this \p uniform_int_distribution.
     */
    typedef IntType result_type;

    /*! \typedef param_type
     *  \brief The type of the object encapsulating this \p uniform_int_distribution's parameters.
     */
    typedef thrust::pair<IntType,IntType> param_type;

    // constructors and reset functions

    /*! This constructor creates a new \p uniform_int_distribution from two values defining the
     *  range of the distribution.
     *  
     *  \param a The smallest integer to potentially produce. Defaults to \c 0.
     *  \param b The largest integer to potentially produce. Defaults to the largest representable integer in
     *           the platform.
     */
    __host__ __device__
    explicit uniform_int_distribution(IntType a = 0,
                                      IntType b = THRUST_NS_QUALIFIER::detail::integer_traits<IntType>::const_max);

    /*! This constructor creates a new \p uniform_int_distribution from a \p param_type object
     *  encapsulating the range of the distribution.
     *  
     *  \param parm A \p param_type object encapsulating the parameters (i.e., the range) of the distribution.
     */
    __host__ __device__
    explicit uniform_int_distribution(const param_type &parm);

    /*! This does nothing.  It is included to conform to the requirements of the RandomDistribution concept.
     */
    __host__ __device__
    void reset(void);

    // generating functions

    /*! This method produces a new uniform random integer drawn from this \p uniform_int_distribution's
     *  range using a \p UniformRandomNumberGenerator as a source of randomness.
     *
     *  \param urng The \p UniformRandomNumberGenerator to use as a source of randomness.
     */
    template<typename UniformRandomNumberGenerator>
    __host__ __device__
    result_type operator()(UniformRandomNumberGenerator &urng);

    /*! This method produces a new uniform random integer as if by creating a new \p uniform_int_distribution 
     *  from the given \p param_type object, and calling its <tt>operator()</tt> method with the given
     *  \p UniformRandomNumberGenerator as a source of randomness.
     *
     *  \param urng The \p UniformRandomNumberGenerator to use as a source of randomness.
     *  \param parm A \p param_type object encapsulating the parameters of the \p uniform_int_distribution
     *              to draw from.
     */
    template<typename UniformRandomNumberGenerator>
    __host__ __device__
    result_type operator()(UniformRandomNumberGenerator &urng, const param_type &parm);

    // property functions
    
    /*! This method returns the value of the parameter with which this \p uniform_int_distribution
     *  was constructed.
     *
     *  \return The lower bound of this \p uniform_int_distribution's range.
     */
    __host__ __device__
    result_type a(void) const;

    /*! This method returns the value of the parameter with which this \p uniform_int_distribution
     *  was constructed.
     *
     *  \return The upper bound of this \p uniform_int_distribution's range.
     */
    __host__ __device__
    result_type b(void) const;

    /*! This method returns a \p param_type object encapsulating the parameters with which this
     *  \p uniform_int_distribution was constructed.
     *
     *  \return A \p param_type object enapsulating the range of this \p uniform_int_distribution.
     */
    __host__ __device__
    param_type param(void) const;

    /*! This method changes the parameters of this \p uniform_int_distribution using the values encapsulated
     *  in a given \p param_type object.
     *
     *  \param parm A \p param_type object encapsulating the new range of this \p uniform_int_distribution.
     */
    __host__ __device__
    void param(const param_type &parm);

    /*! This method returns the smallest integer this \p uniform_int_distribution can potentially produce.
     *
     *  \return The lower bound of this \p uniform_int_distribution's range.
     */
    __host__ __device__
    result_type min THRUST_PREVENT_MACRO_SUBSTITUTION (void) const;

    /*! This method returns the largest integer this \p uniform_int_distribution can potentially produce.
     *
     *  \return The upper bound of this \p uniform_int_distribution's range.
     */
    __host__ __device__
    result_type max THRUST_PREVENT_MACRO_SUBSTITUTION (void) const;

    /*! \cond
     */
  private:
    param_type m_param;

    friend struct thrust::random::detail::random_core_access;

    __host__ __device__
    bool equal(const uniform_int_distribution &rhs) const;

    template<typename CharT, typename Traits>
    std::basic_ostream<CharT,Traits>& stream_out(std::basic_ostream<CharT,Traits> &os) const;

    template<typename CharT, typename Traits>
    std::basic_istream<CharT,Traits>& stream_in(std::basic_istream<CharT,Traits> &is);
    /*! \endcond
     */
}; // end uniform_int_distribution


/*! This function checks two \p uniform_int_distributions for equality.
 *  \param lhs The first \p uniform_int_distribution to test.
 *  \param rhs The second \p uniform_int_distribution to test.
 *  \return \c true if \p lhs is equal to \p rhs; \c false, otherwise.
 */
template<typename IntType>
__host__ __device__
bool operator==(const uniform_int_distribution<IntType> &lhs,
                const uniform_int_distribution<IntType> &rhs);


/*! This function checks two \p uniform_int_distributions for inequality.
 *  \param lhs The first \p uniform_int_distribution to test.
 *  \param rhs The second \p uniform_int_distribution to test.
 *  \return \c true if \p lhs is not equal to \p rhs; \c false, otherwise.
 */
template<typename IntType>
__host__ __device__
bool operator!=(const uniform_int_distribution<IntType> &lhs,
                const uniform_int_distribution<IntType> &rhs);


/*! This function streams a uniform_int_distribution to a \p std::basic_ostream.
 *  \param os The \p basic_ostream to stream out to.
 *  \param d The \p uniform_int_distribution to stream out.
 *  \return \p os
 */
template<typename IntType,
         typename CharT, typename Traits>
std::basic_ostream<CharT,Traits>&
operator<<(std::basic_ostream<CharT,Traits> &os,
           const uniform_int_distribution<IntType> &d);


/*! This function streams a uniform_int_distribution in from a std::basic_istream.
 *  \param is The \p basic_istream to stream from.
 *  \param d The \p uniform_int_distribution to stream in.
 *  \return \p is
 */
template<typename IntType,
         typename CharT, typename Traits>
std::basic_istream<CharT,Traits>&
operator>>(std::basic_istream<CharT,Traits> &is,
           uniform_int_distribution<IntType> &d);


/*! \} // end random_number_distributions
 */


} // end random

using random::uniform_int_distribution;

THRUST_NAMESPACE_END

#include <thrust/random/detail/uniform_int_distribution.inl>

