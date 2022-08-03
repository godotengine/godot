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


/*! \file normal_distribution.h
 *  \brief A normal (Gaussian) distribution of real-valued numbers.
 */

#pragma once

#include <thrust/detail/config.h>
#include <thrust/pair.h>
#include <thrust/random/detail/random_core_access.h>
#include <thrust/random/detail/normal_distribution_base.h>
#include <iostream>

THRUST_NAMESPACE_BEGIN

namespace random
{


/*! \addtogroup random_number_distributions
 *  \{
 */

/*! \class normal_distribution
 *  \brief A \p normal_distribution random number distribution produces floating point
 *         Normally distributed random numbers.
 *
 *  \tparam RealType The type of floating point number to produce.
 *
 *  The following code snippet demonstrates examples of using a \p normal_distribution with a 
 *  random number engine to produce random values drawn from the Normal distribution with a given
 *  mean and variance:
 *
 *  \code
 *  #include <thrust/random/linear_congruential_engine.h>
 *  #include <thrust/random/normal_distribution.h>
 *
 *  int main(void)
 *  {
 *    // create a minstd_rand object to act as our source of randomness
 *    thrust::minstd_rand rng;
 *
 *    // create a normal_distribution to produce floats from the Normal distribution
 *    // with mean 2.0 and standard deviation 3.5
 *    thrust::random::normal_distribution<float> dist(2.0f, 3.5f);
 *
 *    // write a random number to standard output
 *    std::cout << dist(rng) << std::endl;
 *
 *    // write the mean of the distribution, just in case we forgot
 *    std::cout << dist.mean() << std::endl;
 *
 *    // 2.0 is printed
 *
 *    // and the standard deviation
 *    std::cout << dist.stddev() << std::endl;
 *
 *    // 3.5 is printed
 *
 *    return 0;
 *  }
 *  \endcode
 */
template<typename RealType = double>
  class normal_distribution
    : public detail::normal_distribution_base<RealType>::type
{
  private:
    typedef typename detail::normal_distribution_base<RealType>::type super_t;

  public:
    // types
    
    /*! \typedef result_type
     *  \brief The type of the floating point number produced by this \p normal_distribution.
     */
    typedef RealType result_type;

    /*! \typedef param_type
     *  \brief The type of the object encapsulating this \p normal_distribution's parameters.
     */
    typedef thrust::pair<RealType,RealType> param_type;

    // constructors and reset functions
    
    /*! This constructor creates a new \p normal_distribution from two values defining the
     *  half-open interval of the distribution.
     *  
     *  \param mean The mean (expected value) of the distribution. Defaults to \c 0.0.
     *  \param stddev The standard deviation of the distribution. Defaults to \c 1.0.
     */
    __host__ __device__
    explicit normal_distribution(RealType mean = 0.0, RealType stddev = 1.0);

    /*! This constructor creates a new \p normal_distribution from a \p param_type object
     *  encapsulating the range of the distribution.
     *  
     *  \param parm A \p param_type object encapsulating the parameters (i.e., the mean and standard deviation) of the distribution.
     */
    __host__ __device__
    explicit normal_distribution(const param_type &parm);

    /*! Calling this member function guarantees that subsequent uses of this
     *  \p normal_distribution do not depend on values produced by any random
     *  number generator prior to invoking this function.
     */
    __host__ __device__
    void reset(void);

    // generating functions

    /*! This method produces a new Normal random integer drawn from this \p normal_distribution's
     *  range using a \p UniformRandomNumberGenerator as a source of randomness.
     *
     *  \param urng The \p UniformRandomNumberGenerator to use as a source of randomness.
     */
    template<typename UniformRandomNumberGenerator>
    __host__ __device__
    result_type operator()(UniformRandomNumberGenerator &urng);

    /*! This method produces a new Normal random integer as if by creating a new \p normal_distribution 
     *  from the given \p param_type object, and calling its <tt>operator()</tt> method with the given
     *  \p UniformRandomNumberGenerator as a source of randomness.
     *
     *  \param urng The \p UniformRandomNumberGenerator to use as a source of randomness.
     *  \param parm A \p param_type object encapsulating the parameters of the \p normal_distribution
     *              to draw from.
     */
    template<typename UniformRandomNumberGenerator>
    __host__ __device__
    result_type operator()(UniformRandomNumberGenerator &urng, const param_type &parm);

    // property functions

    /*! This method returns the value of the parameter with which this \p normal_distribution
     *  was constructed.
     *
     *  \return The mean (expected value) of this \p normal_distribution's output.
     */
    __host__ __device__
    result_type mean(void) const;

    /*! This method returns the value of the parameter with which this \p normal_distribution
     *  was constructed.
     *
     *  \return The standard deviation of this \p uniform_real_distribution's output.
     */
    __host__ __device__
    result_type stddev(void) const;

    /*! This method returns a \p param_type object encapsulating the parameters with which this
     *  \p normal_distribution was constructed.
     *
     *  \return A \p param_type object encapsulating the parameters (i.e., the mean and standard deviation) of this \p normal_distribution.
     */
    __host__ __device__
    param_type param(void) const;

    /*! This method changes the parameters of this \p normal_distribution using the values encapsulated
     *  in a given \p param_type object.
     *
     *  \param parm A \p param_type object encapsulating the new parameters (i.e., the mean and variance) of this \p normal_distribution.
     */
    __host__ __device__
    void param(const param_type &parm);

    /*! This method returns the smallest floating point number this \p normal_distribution can potentially produce.
     *
     *  \return The lower bound of this \p normal_distribution's half-open interval.
     */
    __host__ __device__
    result_type min THRUST_PREVENT_MACRO_SUBSTITUTION (void) const;

    /*! This method returns the smallest number larger than largest floating point number this \p uniform_real_distribution can potentially produce.
     *
     *  \return The upper bound of this \p normal_distribution's half-open interval.
     */
    __host__ __device__
    result_type max THRUST_PREVENT_MACRO_SUBSTITUTION (void) const;

    /*! \cond
     */
  private:
    param_type m_param;

    friend struct thrust::random::detail::random_core_access;

    __host__ __device__
    bool equal(const normal_distribution &rhs) const;

    template<typename CharT, typename Traits>
    std::basic_ostream<CharT,Traits>& stream_out(std::basic_ostream<CharT,Traits> &os) const;

    template<typename CharT, typename Traits>
    std::basic_istream<CharT,Traits>& stream_in(std::basic_istream<CharT,Traits> &is);
    /*! \endcond
     */
}; // end normal_distribution


/*! This function checks two \p normal_distributions for equality.
 *  \param lhs The first \p normal_distribution to test.
 *  \param rhs The second \p normal_distribution to test.
 *  \return \c true if \p lhs is equal to \p rhs; \c false, otherwise.
 */
template<typename RealType>
__host__ __device__
bool operator==(const normal_distribution<RealType> &lhs,
                const normal_distribution<RealType> &rhs);


/*! This function checks two \p normal_distributions for inequality.
 *  \param lhs The first \p normal_distribution to test.
 *  \param rhs The second \p normal_distribution to test.
 *  \return \c true if \p lhs is not equal to \p rhs; \c false, otherwise.
 */
template<typename RealType>
__host__ __device__
bool operator!=(const normal_distribution<RealType> &lhs,
                const normal_distribution<RealType> &rhs);


/*! This function streams a normal_distribution to a \p std::basic_ostream.
 *  \param os The \p basic_ostream to stream out to.
 *  \param d The \p normal_distribution to stream out.
 *  \return \p os
 */
template<typename RealType,
         typename CharT, typename Traits>
std::basic_ostream<CharT,Traits>&
operator<<(std::basic_ostream<CharT,Traits> &os,
           const normal_distribution<RealType> &d);


/*! This function streams a normal_distribution in from a std::basic_istream.
 *  \param is The \p basic_istream to stream from.
 *  \param d The \p normal_distribution to stream in.
 *  \return \p is
 */
template<typename RealType,
         typename CharT, typename Traits>
std::basic_istream<CharT,Traits>&
operator>>(std::basic_istream<CharT,Traits> &is,
           normal_distribution<RealType> &d);


/*! \} // end random_number_distributions
 */


} // end random

using random::normal_distribution;

THRUST_NAMESPACE_END

#include <thrust/random/detail/normal_distribution.inl>

