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


/*! \file linear_congruential_engine.h
 *  \brief A linear congruential pseudorandom number engine.
 */

#pragma once

#include <thrust/detail/config.h>
#include <iostream>
#include <thrust/detail/cstdint.h>
#include <thrust/random/detail/random_core_access.h>
#include <thrust/random/detail/linear_congruential_engine_discard.h>

THRUST_NAMESPACE_BEGIN

namespace random
{

/*! \addtogroup random_number_engine_templates Random Number Engine Class Templates
 *  \ingroup random
 *  \{
 */

/*! \class linear_congruential_engine
 *  \brief A \p linear_congruential_engine random number engine produces unsigned integer
 *         random numbers using a linear congruential random number generation algorithm.
 *
 *         The generation algorithm has the form <tt>x_i = (a * x_{i-1} + c) mod m</tt>.
 *
 *  \tparam UIntType The type of unsigned integer to produce.
 *  \tparam a The multiplier used in the generation algorithm.
 *  \tparam c The increment used in the generation algorithm.
 *  \tparam m The modulus used in the generation algorithm.
 *
 *  \note Inexperienced users should not use this class template directly.  Instead, use
 *  \p minstd_rand or \p minstd_rand0.
 *
 *  The following code snippet shows examples of use of a \p linear_congruential_engine instance:
 *
 *  \code
 *  #include <thrust/random/linear_congruential_engine.h>
 *  #include <iostream>
 *
 *  int main(void)
 *  {
 *    // create a minstd_rand object, which is an instance of linear_congruential_engine
 *    thrust::minstd_rand rng1;
 *
 *    // output some random values to cout
 *    std::cout << rng1() << std::endl;
 *
 *    // a random value is printed
 *
 *    // create a new minstd_rand from a seed
 *    thrust::minstd_rand rng2(13);
 *
 *    // discard some random values
 *    rng2.discard(13);
 *
 *    // stream the object to an iostream
 *    std::cout << rng2 << std::endl;
 *
 *    // rng2's current state is printed
 *
 *    // print the minimum and maximum values that minstd_rand can produce
 *    std::cout << thrust::minstd_rand::min << std::endl;
 *    std::cout << thrust::minstd_rand::max << std::endl;
 *
 *    // the range of minstd_rand is printed
 *
 *    // save the state of rng2 to a different object
 *    thrust::minstd_rand rng3 = rng2;
 *
 *    // compare rng2 and rng3
 *    std::cout << (rng2 == rng3) << std::endl;
 *
 *    // 1 is printed
 *
 *    // re-seed rng2 with a different seed
 *    rng2.seed(7);
 *
 *    // compare rng2 and rng3
 *    std::cout << (rng2 == rng3) << std::endl;
 *
 *    // 0 is printed
 *
 *    return 0;
 *  }
 *
 *  \endcode
 *
 *  \see thrust::random::minstd_rand
 *  \see thrust::random::minstd_rand0
 */
template<typename UIntType, UIntType a, UIntType c, UIntType m>
  class linear_congruential_engine
{
  public:
    // types
    
    /*! \typedef result_type
     *  \brief The type of the unsigned integer produced by this \p linear_congruential_engine.
     */
    typedef UIntType result_type;

    // engine characteristics

    /*! The multiplier used in the generation algorithm.
     */
    static const result_type multiplier = a;

    /*! The increment used in the generation algorithm.
     */
    static const result_type increment = c;

    /*! The modulus used in the generation algorithm.
     */
    static const result_type modulus = m;

    /*! The smallest value this \p linear_congruential_engine may potentially produce.
     */
    static const result_type min = c == 0u ? 1u : 0u;

    /*! The largest value this \p linear_congruential_engine may potentially produce.
     */
    static const result_type max = m - 1u;

    /*! The default seed of this \p linear_congruential_engine.
     */
    static const result_type default_seed = 1u;

    // constructors and seeding functions

    /*! This constructor, which optionally accepts a seed, initializes a new
     *  \p linear_congruential_engine.
     *  
     *  \param s The seed used to intialize this \p linear_congruential_engine's state.
     */
    __host__ __device__
    explicit linear_congruential_engine(result_type s = default_seed);

    /*! This method initializes this \p linear_congruential_engine's state, and optionally accepts
     *  a seed value.
     *
     *  \param s The seed used to initializes this \p linear_congruential_engine's state.
     */
    __host__ __device__
    void seed(result_type s = default_seed);

    // generating functions

    /*! This member function produces a new random value and updates this \p linear_congruential_engine's state.
     *  \return A new random number.
     */
    __host__ __device__
    result_type operator()(void);

    /*! This member function advances this \p linear_congruential_engine's state a given number of times
     *  and discards the results.
     *
     *  \param z The number of random values to discard.
     *  \note This function is provided because an implementation may be able to accelerate it.
     */
    __host__ __device__
    void discard(unsigned long long z);

    /*! \cond
     */
  private:
    result_type m_x;

    static void transition(result_type &state);

    friend struct thrust::random::detail::random_core_access;

    friend struct thrust::random::detail::linear_congruential_engine_discard;

    __host__ __device__
    bool equal(const linear_congruential_engine &rhs) const;

    template<typename CharT, typename Traits>
    std::basic_ostream<CharT,Traits>& stream_out(std::basic_ostream<CharT,Traits> &os) const;

    template<typename CharT, typename Traits>
    std::basic_istream<CharT,Traits>& stream_in(std::basic_istream<CharT,Traits> &is);

    /*! \endcond
     */
}; // end linear_congruential_engine


/*! This function checks two \p linear_congruential_engines for equality.
 *  \param lhs The first \p linear_congruential_engine to test.
 *  \param rhs The second \p linear_congruential_engine to test.
 *  \return \c true if \p lhs is equal to \p rhs; \c false, otherwise.
 */
template<typename UIntType_, UIntType_ a_, UIntType_ c_, UIntType_ m_>
__host__ __device__
bool operator==(const linear_congruential_engine<UIntType_,a_,c_,m_> &lhs,
                const linear_congruential_engine<UIntType_,a_,c_,m_> &rhs);


/*! This function checks two \p linear_congruential_engines for inequality.
 *  \param lhs The first \p linear_congruential_engine to test.
 *  \param rhs The second \p linear_congruential_engine to test.
 *  \return \c true if \p lhs is not equal to \p rhs; \c false, otherwise.
 */
template<typename UIntType_, UIntType_ a_, UIntType_ c_, UIntType_ m_>
__host__ __device__
bool operator!=(const linear_congruential_engine<UIntType_,a_,c_,m_> &lhs,
                const linear_congruential_engine<UIntType_,a_,c_,m_> &rhs);


/*! This function streams a linear_congruential_engine to a \p std::basic_ostream.
 *  \param os The \p basic_ostream to stream out to.
 *  \param e The \p linear_congruential_engine to stream out.
 *  \return \p os
 */
template<typename UIntType_, UIntType_ a_, UIntType_ c_, UIntType_ m_,
         typename CharT, typename Traits>
std::basic_ostream<CharT,Traits>&
operator<<(std::basic_ostream<CharT,Traits> &os,
           const linear_congruential_engine<UIntType_,a_,c_,m_> &e);


/*! This function streams a linear_congruential_engine in from a std::basic_istream.
 *  \param is The \p basic_istream to stream from.
 *  \param e The \p linear_congruential_engine to stream in.
 *  \return \p is
 */
template<typename UIntType_, UIntType_ a_, UIntType_ c_, UIntType_ m_,
         typename CharT, typename Traits>
std::basic_istream<CharT,Traits>&
operator>>(std::basic_istream<CharT,Traits> &is,
           linear_congruential_engine<UIntType_,a_,c_,m_> &e);


/*! \} // random_number_engine_templates
 */


/*! \addtogroup predefined_random
 *  \{
 */

// XXX the type N2111 used here was uint_fast32_t

/*! \typedef minstd_rand0
 *  \brief A random number engine with predefined parameters which implements a version of
 *         the Minimal Standard random number generation algorithm.
 *  \note The 10000th consecutive invocation of a default-constructed object of type \p minstd_rand0
 *        shall produce the value \c 1043618065 .
 */
typedef linear_congruential_engine<thrust::detail::uint32_t, 16807, 0, 2147483647> minstd_rand0;


/*! \typedef minstd_rand
 *  \brief A random number engine with predefined parameters which implements a version of
 *         the Minimal Standard random number generation algorithm.
 *  \note The 10000th consecutive invocation of a default-constructed object of type \p minstd_rand
 *        shall produce the value \c 399268537 .
 */
typedef linear_congruential_engine<thrust::detail::uint32_t, 48271, 0, 2147483647> minstd_rand;

/*! \} // predefined_random
 */
  
} // end random

// import names into thrust::
using random::linear_congruential_engine;
using random::minstd_rand;
using random::minstd_rand0;

THRUST_NAMESPACE_END

#include <thrust/random/detail/linear_congruential_engine.inl>

