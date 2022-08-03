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


/*! \file discard_block_engine.h
 *  \brief A random number engine which adapts a base engine and produces
 *         numbers by discarding all but a contiguous blocks of its values.
 */

#pragma once

#include <thrust/detail/config.h>

#include <thrust/detail/config.h>
#include <iostream>
#include <thrust/detail/cstdint.h>
#include <thrust/random/detail/random_core_access.h>

THRUST_NAMESPACE_BEGIN

namespace random
{

/*! \addtogroup random_number_engine_adaptors Random Number Engine Adaptor Class Templates
 *  \ingroup random
 *  \{
 */

/*! \class discard_block_engine
 *  \brief A \p discard_block_engine adapts an existing base random number engine and produces
 *         random values by discarding some of the values returned by its base engine.
 *         Each cycle of the compound engine begins by returning \c r values successively produced
 *         by the base engine and ends by discarding <tt>p-r</tt> such values. The engine's state
 *         is the state of its base engine followed by the number of calls to <tt>operator()</tt>
 *         that have occurred since the beginning of the current cycle.
 *
 *  \tparam Engine The type of the base random number engine to adapt.
 *  \tparam p The discard cycle length.
 *  \tparam r The number of values to return of the base engine. Because <tt>p-r</tt> will be
 *            discarded, <tt>r <= p</tt>.
 *
 *  The following code snippet shows an example of using a \p discard_block_engine instance:
 *
 *  \code
 *  #include <thrust/random/linear_congruential_engine.h>
 *  #include <thrust/random/discard_block_engine.h>
 *  #include <iostream>
 *
 *  int main(void)
 *  {
 *    // create a discard_block_engine from minstd_rand, with a cycle length of 13
 *    // keep every first 10 values, and discard the next 3
 *    thrust::discard_block_engine<thrust::minstd_rand, 13, 10> rng;
 *
 *    // print a random number to standard output
 *    std::cout << rng() << std::endl;
 *
 *    return 0;
 *  }
 *  \endcode
 */         
template<typename Engine, size_t p, size_t r>
  class discard_block_engine
{
  public:
    // types

    /*! \typedef base_type
     *  \brief The type of the adapted base random number engine.
     */
    typedef Engine base_type;

    /*! \typedef result_type
     *  \brief The type of the unsigned integer produced by this \p linear_congruential_engine.
     */
    typedef typename base_type::result_type result_type;

    // engine characteristics

    /*! The length of the production cycle.
     */
    static const size_t block_size = p;

    /*! The number of used numbers per production cycle.
     */
    static const size_t used_block = r;

    /*! The smallest value this \p discard_block_engine may potentially produce.
     */
    static const result_type min = base_type::min;

    /*! The largest value this \p discard_block_engine may potentially produce.
     */
    static const result_type max = base_type::max;

    // constructors and seeding functions

    /*! This constructor constructs a new \p discard_block_engine and constructs
     *  its \p base_type engine using its null constructor.
     */
    __host__ __device__
    discard_block_engine();

    /*! This constructor constructs a new \p discard_block_engine using
     *  a given \p base_type engine to initialize its adapted base engine.
     *
     *  \param urng A \p base_type to use to initialize this \p discard_block_engine's
     *         adapted base engine.
     */
    __host__ __device__
    explicit discard_block_engine(const base_type &urng);

    /*! This constructor initializes a new \p discard_block_engine with a given seed.
     *  
     *  \param s The seed used to intialize this \p discard_block_engine's adapted base engine.
     */
    __host__ __device__
    explicit discard_block_engine(result_type s);

    /*! This method initializes the state of this \p discard_block_engine's adapted base engine
     *  by using its \p default_seed value.
     */
    __host__ __device__
    void seed(void);

    /*! This method initializes the state of this \p discard_block_engine's adapted base engine
     *  by using the given seed.
     *
     *  \param s The seed with which to intialize this \p discard_block_engine's adapted base engine.
     */
    __host__ __device__
    void seed(result_type s);

    // generating functions
    
    /*! This member function produces a new random value and updates this \p discard_block_engine's state.
     *  \return A new random number.
     */
    __host__ __device__
    result_type operator()(void);

    /*! This member function advances this \p discard_block_engine's state a given number of times
     *  and discards the results.
     *
     *  \param z The number of random values to discard.
     *  \note This function is provided because an implementation may be able to accelerate it.
     */
    __host__ __device__
    void discard(unsigned long long z);

    // property functions

    /*! This member function returns a const reference to this \p discard_block_engine's
     *  adapted base engine.
     *
     *  \return A const reference to the base engine this \p discard_block_engine adapts.
     */
    __host__ __device__
    const base_type &base(void) const;

    /*! \cond
     */
  private:
    base_type m_e;
    unsigned int m_n;

    friend struct thrust::random::detail::random_core_access;

    __host__ __device__
    bool equal(const discard_block_engine &rhs) const;

    template<typename CharT, typename Traits>
    std::basic_ostream<CharT,Traits>& stream_out(std::basic_ostream<CharT,Traits> &os) const;

    template<typename CharT, typename Traits>
    std::basic_istream<CharT,Traits>& stream_in(std::basic_istream<CharT,Traits> &is);
    /*! \endcond
     */
}; // end discard_block_engine


/*! This function checks two \p discard_block_engines for equality.
 *  \param lhs The first \p discard_block_engine to test.
 *  \param rhs The second \p discard_block_engine to test.
 *  \return \c true if \p lhs is equal to \p rhs; \c false, otherwise.
 */
template<typename Engine, size_t p, size_t r>
__host__ __device__
bool operator==(const discard_block_engine<Engine,p,r> &lhs,
                const discard_block_engine<Engine,p,r> &rhs);


/*! This function checks two \p discard_block_engines for inequality.
 *  \param lhs The first \p discard_block_engine to test.
 *  \param rhs The second \p discard_block_engine to test.
 *  \return \c true if \p lhs is not equal to \p rhs; \c false, otherwise.
 */
template<typename Engine, size_t p, size_t r>
__host__ __device__
bool operator!=(const discard_block_engine<Engine,p,r> &lhs,
                const discard_block_engine<Engine,p,r> &rhs);


/*! This function streams a discard_block_engine to a \p std::basic_ostream.
 *  \param os The \p basic_ostream to stream out to.
 *  \param e The \p discard_block_engine to stream out.
 *  \return \p os
 */
template<typename Engine, size_t p, size_t r,
         typename CharT, typename Traits>
std::basic_ostream<CharT,Traits>&
operator<<(std::basic_ostream<CharT,Traits> &os,
           const discard_block_engine<Engine,p,r> &e);


/*! This function streams a discard_block_engine in from a std::basic_istream.
 *  \param is The \p basic_istream to stream from.
 *  \param e The \p discard_block_engine to stream in.
 *  \return \p is
 */
template<typename Engine, size_t p, size_t r,
         typename CharT, typename Traits>
std::basic_istream<CharT,Traits>&
operator>>(std::basic_istream<CharT,Traits> &is,
           discard_block_engine<Engine,p,r> &e);

/*! \} // end random_number_engine_adaptors
 */

} // end random

// import names into thrust::
using random::discard_block_engine;

THRUST_NAMESPACE_END

#include <thrust/random/detail/discard_block_engine.inl>

