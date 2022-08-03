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

/*! \file xor_combine_engine.h
 *  \brief A pseudorandom number generator which produces pseudorandom
 *         numbers from two integer base engines by merging their
 *         pseudorandom numbers with bitwise exclusive-or.
 */

#pragma once

#include <thrust/detail/config.h>
#include <thrust/detail/type_traits.h>
#include <thrust/random/detail/xor_combine_engine_max.h>
#include <thrust/random/detail/random_core_access.h>
#include <iostream>
#include <cstddef> // for size_t

THRUST_NAMESPACE_BEGIN

namespace random
{

/*! \addtogroup random_number_engine_adaptors
 *  \{
 */

/*! \class xor_combine_engine
 *  \brief An \p xor_combine_engine adapts two existing base random number engines and
 *         produces random values by combining the values produced by each.
 *
 *  \tparam Engine1 The type of the first base random number engine to adapt.
 *  \tparam s1 The size of the first shift to use in the generation algorithm.
 *  \tparam Engine2 The type of the second base random number engine to adapt.
 *  \tparam s2 The second of the second shift to use in the generation algorithm. Defaults to \c 0.
 *
 *  The following code snippet shows an example of using an \p xor_combine_engine instance:
 *
 *  \code
 *  #include <thrust/random/linear_congruential_engine.h>
 *  #include <thrust/random/xor_combine_engine.h>
 *  #include <iostream>
 *
 *  int main(void)
 *  {
 *    // create an xor_combine_engine from minstd_rand and minstd_rand0
 *    // use a shift of 0 for each
 *    thrust::xor_combine_engine<thrust::minstd_rand,0,thrust::minstd_rand0,0> rng;
 *
 *    // print a random number to standard output
 *    std::cout << rng() << std::endl;
 *
 *    return 0;
 *  }
 *  \endcode
 */
template<typename Engine1, size_t s1,
         typename Engine2, size_t s2=0u>
  class xor_combine_engine
{
  public:
    // types

    /*! \typedef base1_type
     *  \brief The type of the first adapted base random number engine.
     */
    typedef Engine1 base1_type;

    /*! \typedef base2_type
     *  \brief The type of the second adapted base random number engine.
     */
    typedef Engine2 base2_type;

    /*! \typedef result_type
     *  \brief The type of the unsigned integer produced by this \p xor_combine_engine.
     */
    typedef typename thrust::detail::eval_if<
      (sizeof(typename base2_type::result_type) > sizeof(typename base1_type::result_type)),
      thrust::detail::identity_<typename base2_type::result_type>,
      thrust::detail::identity_<typename base1_type::result_type>
    >::type result_type;
    
    /*! The size of the first shift used in the generation algorithm.
     */
    static const size_t shift1 = s1;

    /*! The size of the second shift used in the generation algorithm.
     */
    static const size_t shift2 = s2;

    /*! The smallest value this \p xor_combine_engine may potentially produce.
     */
    static const result_type min = 0;

    /*! The largest value this \p xor_combine_engine may potentially produce.
     */
    static const result_type max =
      detail::xor_combine_engine_max<
        Engine1, s1, Engine2, s2, result_type
      >::value;

    // constructors and seeding functions

    /*! This constructor constructs a new \p xor_combine_engine and constructs
     *  its adapted engines using their null constructors.
     */
    __host__ __device__
    xor_combine_engine(void);

    /*! This constructor constructs a new \p xor_combine_engine using
     *  given \p base1_type and \p base2_type engines to initialize its adapted base engines.
     *
     *  \param urng1 A \p base1_type to use to initialize this \p xor_combine_engine's
     *         first adapted base engine.
     *  \param urng2 A \p base2_type to use to initialize this \p xor_combine_engine's
     *         first adapted base engine.
     */
    __host__ __device__
    xor_combine_engine(const base1_type &urng1, const base2_type &urng2);

    /*! This constructor initializes a new \p xor_combine_engine with a given seed.
     *  
     *  \param s The seed used to intialize this \p xor_combine_engine's adapted base engines.
     */
    __host__ __device__
    xor_combine_engine(result_type s);

    /*! This method initializes the state of this \p xor_combine_engine's adapted base engines
     *  by using their \p default_seed values.
     */
    __host__ __device__
    void seed(void);

    /*! This method initializes the state of this \p xor_combine_engine's adapted base engines
     *  by using the given seed.
     *
     *  \param s The seed with which to intialize this \p xor_combine_engine's adapted base engines.
     */
    __host__ __device__
    void seed(result_type s);

    // generating functions

    /*! This member function produces a new random value and updates this \p xor_combine_engine's state.
     *  \return A new random number.
     */
    __host__ __device__
    result_type operator()(void);

    /*! This member function advances this \p xor_combine_engine's state a given number of times
     *  and discards the results.
     *
     *  \param z The number of random values to discard.
     *  \note This function is provided because an implementation may be able to accelerate it.
     */
    __host__ __device__
    void discard(unsigned long long z);

    // property functions

    /*! This member function returns a const reference to this \p xor_combine_engine's
     *  first adapted base engine.
     *
     *  \return A const reference to the first base engine this \p xor_combine_engine adapts.
     */
    __host__ __device__
    const base1_type &base1(void) const;

    /*! This member function returns a const reference to this \p xor_combine_engine's
     *  second adapted base engine.
     *
     *  \return A const reference to the second base engine this \p xor_combine_engine adapts.
     */
    __host__ __device__
    const base2_type &base2(void) const;

    /*! \cond
     */
  private:
    base1_type m_b1;
    base2_type m_b2;

    friend struct thrust::random::detail::random_core_access;

    __host__ __device__
    bool equal(const xor_combine_engine &rhs) const;

    template<typename CharT, typename Traits>
    std::basic_istream<CharT,Traits>& stream_in(std::basic_istream<CharT,Traits> &is);

    template<typename CharT, typename Traits>
    std::basic_ostream<CharT,Traits>& stream_out(std::basic_ostream<CharT,Traits> &os) const;

    /*! \endcond
     */
}; // end xor_combine_engine


/*! This function checks two \p xor_combine_engines for equality.
 *  \param lhs The first \p xor_combine_engine to test.
 *  \param rhs The second \p xor_combine_engine to test.
 *  \return \c true if \p lhs is equal to \p rhs; \c false, otherwise.
 */
template<typename Engine1_, size_t s1_, typename Engine2_, size_t s2_>
__host__ __device__
bool operator==(const xor_combine_engine<Engine1_,s1_,Engine2_,s2_> &lhs,
                const xor_combine_engine<Engine1_,s1_,Engine2_,s2_> &rhs);


/*! This function checks two \p xor_combine_engines for inequality.
 *  \param lhs The first \p xor_combine_engine to test.
 *  \param rhs The second \p xor_combine_engine to test.
 *  \return \c true if \p lhs is not equal to \p rhs; \c false, otherwise.
 */
template<typename Engine1_, size_t s1_, typename Engine2_, size_t s2_>
__host__ __device__
bool operator!=(const xor_combine_engine<Engine1_,s1_,Engine2_,s2_> &lhs,
                const xor_combine_engine<Engine1_,s1_,Engine2_,s2_> &rhs);


/*! This function streams a xor_combine_engine to a \p std::basic_ostream.
 *  \param os The \p basic_ostream to stream out to.
 *  \param e The \p xor_combine_engine to stream out.
 *  \return \p os
 */
template<typename Engine1_, size_t s1_, typename Engine2_, size_t s2_,
         typename CharT, typename Traits>
std::basic_ostream<CharT,Traits>&
operator<<(std::basic_ostream<CharT,Traits> &os,
           const xor_combine_engine<Engine1_,s1_,Engine2_,s2_> &e);


/*! This function streams a xor_combine_engine in from a std::basic_istream.
 *  \param is The \p basic_istream to stream from.
 *  \param e The \p xor_combine_engine to stream in.
 *  \return \p is
 */
template<typename Engine1_, size_t s1_, typename Engine2_, size_t s2_,
         typename CharT, typename Traits>
std::basic_istream<CharT,Traits>&
operator>>(std::basic_istream<CharT,Traits> &is,
           xor_combine_engine<Engine1_,s1_,Engine2_,s2_> &e);


/*! \} // end random_number_engine_adaptors
 */


} // end random

// import names into thrust::
using random::xor_combine_engine;

THRUST_NAMESPACE_END

#include <thrust/random/detail/xor_combine_engine.inl>

