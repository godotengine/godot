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

/*! \file subtract_with_carry_engine.h
 *  \brief A subtract-with-carry pseudorandom number generator
 *         based on Marsaglia & Zaman.
 */

#pragma once

#include <thrust/detail/config.h>
#include <thrust/random/detail/random_core_access.h>

#include <thrust/detail/cstdint.h>
#include <cstddef> // for size_t
#include <iostream>

THRUST_NAMESPACE_BEGIN

namespace random
{


/*! \addtogroup random_number_engine_templates
 *  \{
 */

/*! \class subtract_with_carry_engine
 *  \brief A \p subtract_with_carry_engine random number engine produces unsigned
 *         integer random numbers using the subtract with carry algorithm of Marsaglia & Zaman.
 *
 *         The generation algorithm is performed as follows:
 *         -# Let <tt>Y = X_{i-s}- X_{i-r} - c</tt>.
 *         -# Set <tt>X_i</tt> to <tt>y = T mod m</tt>. Set \c c to \c 1 if <tt>Y < 0</tt>, otherwise set \c c to \c 0.
 *
 *         This algorithm corresponds to a modular linear function of the form
 *
 *         <tt>TA(x_i) = (a * x_i) mod b</tt>, where \c b is of the form <tt>m^r - m^s + 1</tt> and
 *         <tt>a = b - (b-1)/m</tt>.
 *
 *  \tparam UIntType The type of unsigned integer to produce.
 *  \tparam w The word size of the produced values (<tt> w <= sizeof(UIntType)</tt>).
 *  \tparam s The short lag of the generation algorithm.
 *  \tparam r The long lag of the generation algorithm.
 *
 *  \note Inexperienced users should not use this class template directly.  Instead, use
 *  \p ranlux24_base or \p ranlux48_base, which are instances of \p subtract_with_carry_engine.
 *
 *  \see thrust::random::ranlux24_base
 *  \see thrust::random::ranlux48_base
 */
template<typename UIntType, size_t w, size_t s, size_t r>
  class subtract_with_carry_engine
{
    /*! \cond
     */
  private:
    static const UIntType modulus = UIntType(1) << w;
    /*! \endcond
     */

  public:
    // types
    
    /*! \typedef result_type
     *  \brief The type of the unsigned integer produced by this \p subtract_with_carry_engine.
     */
    typedef UIntType result_type;

    // engine characteristics

    /*! The word size of the produced values.
     */
    static const size_t word_size = w;

    /*! The size of the short lag used in the generation algorithm.
     */
    static const size_t short_lag = s;

    /*! The size of the long lag used in the generation algorithm.
     */
    static const size_t long_lag = r;

    /*! The smallest value this \p subtract_with_carry_engine may potentially produce.
     */
    static const result_type min = 0;

    /*! The largest value this \p subtract_with_carry_engine may potentially produce.
     */
    static const result_type max = modulus - 1;

    /*! The default seed of this \p subtract_with_carry_engine.
     */
    static const result_type default_seed = 19780503u;

    // constructors and seeding functions

    /*! This constructor, which optionally accepts a seed, initializes a new
     *  \p subtract_with_carry_engine.
     *  
     *  \param value The seed used to intialize this \p subtract_with_carry_engine's state.
     */
    __host__ __device__
    explicit subtract_with_carry_engine(result_type value = default_seed);

    /*! This method initializes this \p subtract_with_carry_engine's state, and optionally accepts
     *  a seed value.
     *
     *  \param value The seed used to initializes this \p subtract_with_carry_engine's state.
     */
    __host__ __device__
    void seed(result_type value = default_seed);

    // generating functions
    
    /*! This member function produces a new random value and updates this \p subtract_with_carry_engine's state.
     *  \return A new random number.
     */
    __host__ __device__
    result_type operator()(void);

    /*! This member function advances this \p subtract_with_carry_engine's state a given number of times
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
    result_type m_x[long_lag];
    unsigned int m_k;
    int m_carry;

    friend struct thrust::random::detail::random_core_access;

    __host__ __device__
    bool equal(const subtract_with_carry_engine &rhs) const;

    template<typename CharT, typename Traits>
    std::basic_ostream<CharT,Traits>& stream_out(std::basic_ostream<CharT,Traits> &os) const;

    template<typename CharT, typename Traits>
    std::basic_istream<CharT,Traits>& stream_in(std::basic_istream<CharT,Traits> &is);

    /*! \endcond
     */
}; // end subtract_with_carry_engine


/*! This function checks two \p subtract_with_carry_engines for equality.
 *  \param lhs The first \p subtract_with_carry_engine to test.
 *  \param rhs The second \p subtract_with_carry_engine to test.
 *  \return \c true if \p lhs is equal to \p rhs; \c false, otherwise.
 */
template<typename UIntType_, size_t w_, size_t s_, size_t r_>
__host__ __device__
bool operator==(const subtract_with_carry_engine<UIntType_,w_,s_,r_> &lhs,
                const subtract_with_carry_engine<UIntType_,w_,s_,r_> &rhs);


/*! This function checks two \p subtract_with_carry_engines for inequality.
 *  \param lhs The first \p subtract_with_carry_engine to test.
 *  \param rhs The second \p subtract_with_carry_engine to test.
 *  \return \c true if \p lhs is not equal to \p rhs; \c false, otherwise.
 */
template<typename UIntType_, size_t w_, size_t s_, size_t r_>
__host__ __device__
bool operator!=(const subtract_with_carry_engine<UIntType_,w_,s_,r_>&lhs,
                const subtract_with_carry_engine<UIntType_,w_,s_,r_>&rhs);


/*! This function streams a subtract_with_carry_engine to a \p std::basic_ostream.
 *  \param os The \p basic_ostream to stream out to.
 *  \param e The \p subtract_with_carry_engine to stream out.
 *  \return \p os
 */
template<typename UIntType_, size_t w_, size_t s_, size_t r_,
         typename CharT, typename Traits>
std::basic_ostream<CharT,Traits>&
operator<<(std::basic_ostream<CharT,Traits> &os,
           const subtract_with_carry_engine<UIntType_,w_,s_,r_> &e);


/*! This function streams a subtract_with_carry_engine in from a std::basic_istream.
 *  \param is The \p basic_istream to stream from.
 *  \param e The \p subtract_with_carry_engine to stream in.
 *  \return \p is
 */
template<typename UIntType_, size_t w_, size_t s_, size_t r_,
         typename CharT, typename Traits>
std::basic_istream<CharT,Traits>&
operator>>(std::basic_istream<CharT,Traits> &is,
           subtract_with_carry_engine<UIntType_,w_,s_,r_> &e);


/*! \} // end random_number_engine_templates
 */


/*! \addtogroup predefined_random
 *  \{
 */

// XXX N2111 uses uint_fast32_t here

/*! \typedef ranlux24_base
 *  \brief A random number engine with predefined parameters which implements the
 *         base engine of the \p ranlux24 random number engine.
 *  \note The 10000th consecutive invocation of a default-constructed object of type \p ranlux24_base
 *        shall produce the value \c 7937952 .
 */
typedef subtract_with_carry_engine<thrust::detail::uint32_t, 24, 10, 24> ranlux24_base;


// XXX N2111 uses uint_fast64_t here

/*! \typedef ranlux48_base
 *  \brief A random number engine with predefined parameters which implements the
 *         base engine of the \p ranlux48 random number engine.
 *  \note The 10000th consecutive invocation of a default-constructed object of type \p ranlux48_base
 *        shall produce the value \c 192113843633948 .
 */
typedef subtract_with_carry_engine<thrust::detail::uint64_t, 48,  5, 12> ranlux48_base;

/*! \} // end predefined_random
 */

} // end random

// import names into thrust::
using random::subtract_with_carry_engine;
using random::ranlux24_base;
using random::ranlux48_base;

THRUST_NAMESPACE_END

#include <thrust/random/detail/subtract_with_carry_engine.inl>

