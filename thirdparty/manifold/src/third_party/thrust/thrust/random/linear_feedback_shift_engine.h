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

/*! \file linear_feedback_shift_engine.h
 *  \brief A linear feedback shift pseudorandom number generator.
 */

/*
 * Copyright Jens Maurer 2002
 *
 * Distributed under the Boost Software License, Version 1.0.
 * (See accompanying NOTICE file for the complete license)
 *
 * For more information, see http://www.boost.org
 */

#pragma once

#include <thrust/detail/config.h>
#include <thrust/random/detail/linear_feedback_shift_engine_wordmask.h>
#include <iostream>
#include <cstddef> // for size_t
#include <thrust/random/detail/random_core_access.h>

THRUST_NAMESPACE_BEGIN


namespace random
{

/*! \addtogroup random_number_engine_templates
 *  \{
 */

/*! \class linear_feedback_shift_engine
 *  \brief A \p linear_feedback_shift_engine random number engine produces
 *         unsigned integer random values using a linear feedback shift random number
 *         generation algorithm.
 *
 *  \tparam UIntType The type of unsigned integer to produce.
 *  \tparam w The word size of the produced values (<tt>w <= sizeof(UIntType)</tt>).
 *  \tparam k The k parameter of Tausworthe's 1965 algorithm.
 *  \tparam q The q exponent of Tausworthe's 1965 algorithm.
 *  \tparam s The step size of Tausworthe's 1965 algorithm.
 *
 *  \note linear_feedback_shift_engine is based on the Boost Template Library's linear_feedback_shift.
 */
template<typename UIntType, size_t w, size_t k, size_t q, size_t s>
  class linear_feedback_shift_engine
{
  public:
    // types

    /*! \typedef result_type
     *  \brief The type of the unsigned integer produced by this \p linear_feedback_shift_engine.
     */
    typedef UIntType result_type;

    // engine characteristics

    /*! The word size of the produced values.
     */
    static const size_t word_size = w;

    /*! A constant used in the generation algorithm.
     */
    static const size_t exponent1 = k;

    /*! A constant used in the generation algorithm.
     */
    static const size_t exponent2 = q;

    /*! The step size used in the generation algorithm.
     */
    static const size_t step_size = s;

    /*! \cond
     */
  private:
    static const result_type wordmask =
      detail::linear_feedback_shift_engine_wordmask<
        result_type,
        w
      >::value;
    /*! \endcond
     */

  public:

    /*! The smallest value this \p linear_feedback_shift_engine may potentially produce.
     */
    static const result_type min = 0;

    /*! The largest value this \p linear_feedback_shift_engine may potentially produce.
     */
    static const result_type max = wordmask;

    /*! The default seed of this \p linear_feedback_shift_engine.
     */
    static const result_type default_seed = 341u;

    // constructors and seeding functions

    /*! This constructor, which optionally accepts a seed, initializes a new
     *  \p linear_feedback_shift_engine.
     *  
     *  \param value The seed used to intialize this \p linear_feedback_shift_engine's state.
     */
    __host__ __device__
    explicit linear_feedback_shift_engine(result_type value = default_seed);

    /*! This method initializes this \p linear_feedback_shift_engine's state, and optionally accepts
     *  a seed value.
     *
     *  \param value The seed used to initializes this \p linear_feedback_shift_engine's state.
     */
    __host__ __device__
    void seed(result_type value = default_seed);

    // generating functions
    
    /*! This member function produces a new random value and updates this \p linear_feedback_shift_engine's state.
     *  \return A new random number.
     */
    __host__ __device__
    result_type operator()(void);

    /*! This member function advances this \p linear_feedback_shift_engine's state a given number of times
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
    result_type m_value;

    friend struct thrust::random::detail::random_core_access;

    __host__ __device__
    bool equal(const linear_feedback_shift_engine &rhs) const;

    template<typename CharT, typename Traits>
    std::basic_ostream<CharT,Traits>& stream_out(std::basic_ostream<CharT,Traits> &os) const;

    template<typename CharT, typename Traits>
    std::basic_istream<CharT,Traits>& stream_in(std::basic_istream<CharT,Traits> &is);

    /*! \endcond
     */
}; // end linear_feedback_shift_engine


/*! This function checks two \p linear_feedback_shift_engines for equality.
 *  \param lhs The first \p linear_feedback_shift_engine to test.
 *  \param rhs The second \p linear_feedback_shift_engine to test.
 *  \return \c true if \p lhs is equal to \p rhs; \c false, otherwise.
 */
template<typename UIntType_, size_t w_, size_t k_, size_t q_, size_t s_>
__host__ __device__
bool operator==(const linear_feedback_shift_engine<UIntType_,w_,k_,q_,s_> &lhs,
                const linear_feedback_shift_engine<UIntType_,w_,k_,q_,s_> &rhs);


/*! This function checks two \p linear_feedback_shift_engines for inequality.
 *  \param lhs The first \p linear_feedback_shift_engine to test.
 *  \param rhs The second \p linear_feedback_shift_engine to test.
 *  \return \c true if \p lhs is not equal to \p rhs; \c false, otherwise.
 */
template<typename UIntType_, size_t w_, size_t k_, size_t q_, size_t s_>
__host__ __device__
bool operator!=(const linear_feedback_shift_engine<UIntType_,w_,k_,q_,s_> &lhs,
                const linear_feedback_shift_engine<UIntType_,w_,k_,q_,s_> &rhs);


/*! This function streams a linear_feedback_shift_engine to a \p std::basic_ostream.
 *  \param os The \p basic_ostream to stream out to.
 *  \param e The \p linear_feedback_shift_engine to stream out.
 *  \return \p os
 */
template<typename UIntType_, size_t w_, size_t k_, size_t q_, size_t s_,
         typename CharT, typename Traits>
std::basic_ostream<CharT,Traits>&
operator<<(std::basic_ostream<CharT,Traits> &os,
           const linear_feedback_shift_engine<UIntType_,w_,k_,q_,s_> &e);


/*! This function streams a linear_feedback_shift_engine in from a std::basic_istream.
 *  \param is The \p basic_istream to stream from.
 *  \param e The \p linear_feedback_shift_engine to stream in.
 *  \return \p is
 */
template<typename UIntType_, size_t w_, size_t k_, size_t q_, size_t s_,
         typename CharT, typename Traits>
std::basic_istream<CharT,Traits>&
operator>>(std::basic_istream<CharT,Traits> &is,
           linear_feedback_shift_engine<UIntType_,w_,k_,q_,s_> &e);


/*! \} // end random_number_engine_templates
 */


} // end random

// import names into thrust::
using random::linear_feedback_shift_engine;

THRUST_NAMESPACE_END

#include <thrust/random/detail/linear_feedback_shift_engine.inl>

