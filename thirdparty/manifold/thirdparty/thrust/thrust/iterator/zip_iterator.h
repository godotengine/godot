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


/*! \file thrust/iterator/zip_iterator.h
 *  \brief An iterator which returns a tuple of the result of dereferencing
 *         a tuple of iterators when dereferenced
 */

/*
 * Copyright David Abrahams and Thomas Becker 2000-2006.
 * 
 * Distributed under the Boost Software License, Version 1.0.
 * (See accompanying NOTICE file for the complete license)
 *
 * For more information, see http://www.boost.org
 */

#pragma once

#include <thrust/detail/config.h>
#include <thrust/iterator/detail/zip_iterator_base.h>
#include <thrust/iterator/iterator_facade.h>
#include <thrust/detail/type_traits.h>

THRUST_NAMESPACE_BEGIN

/*! \addtogroup iterators
 *  \{
 */

/*! \addtogroup fancyiterator Fancy Iterators
 *  \ingroup iterators
 *  \{
 */

/*! \p zip_iterator is an iterator which represents a pointer into a range
 *  of \p tuples whose elements are themselves taken from a \p tuple of input
 *  iterators. This iterator is useful for creating a virtual array of structures
 *  while achieving the same performance and bandwidth as the structure of arrays
 *  idiom. \p zip_iterator also facilitates kernel fusion by providing a convenient
 *  means of amortizing the execution of the same operation over multiple ranges.
 *
 *  The following code snippet demonstrates how to create a \p zip_iterator
 *  which represents the result of "zipping" multiple ranges together.
 *  
 *  \code
 *  #include <thrust/iterator/zip_iterator.h>
 *  #include <thrust/tuple.h>
 *  #include <thrust/device_vector.h>
 *  ...
 *  thrust::device_vector<int> int_v(3);
 *  int_v[0] = 0; int_v[1] = 1; int_v[2] = 2;
 *
 *  thrust::device_vector<float> float_v(3);
 *  float_v[0] = 0.0f; float_v[1] = 1.0f; float_v[2] = 2.0f;
 *
 *  thrust::device_vector<char> char_v(3);
 *  char_v[0] = 'a'; char_v[1] = 'b'; char_v[2] = 'c';
 *
 *  // typedef these iterators for shorthand
 *  typedef thrust::device_vector<int>::iterator   IntIterator;
 *  typedef thrust::device_vector<float>::iterator FloatIterator;
 *  typedef thrust::device_vector<char>::iterator  CharIterator;
 *
 *  // typedef a tuple of these iterators
 *  typedef thrust::tuple<IntIterator, FloatIterator, CharIterator> IteratorTuple;
 *
 *  // typedef the zip_iterator of this tuple
 *  typedef thrust::zip_iterator<IteratorTuple> ZipIterator;
 *
 *  // finally, create the zip_iterator
 *  ZipIterator iter(thrust::make_tuple(int_v.begin(), float_v.begin(), char_v.begin()));
 *
 *  *iter;   // returns (0, 0.0f, 'a')
 *  iter[0]; // returns (0, 0.0f, 'a')
 *  iter[1]; // returns (1, 1.0f, 'b')
 *  iter[2]; // returns (2, 2.0f, 'c')
 *
 *  thrust::get<0>(iter[2]); // returns 2
 *  thrust::get<1>(iter[0]); // returns 0.0f
 *  thrust::get<2>(iter[1]); // returns 'b'
 *
 *  // iter[3] is an out-of-bounds error
 *  \endcode
 *
 *  Defining the type of a \p zip_iterator can be complex. The next code example demonstrates
 *  how to use the \p make_zip_iterator function with the \p make_tuple function to avoid
 *  explicitly specifying the type of the \p zip_iterator. This example shows how to use
 *  \p zip_iterator to copy multiple ranges with a single call to \p thrust::copy.
 *
 *  \code
 *  #include <thrust/zip_iterator.h>
 *  #include <thrust/tuple.h>
 *  #include <thrust/device_vector.h>
 *
 *  int main()
 *  {
 *    thrust::device_vector<int> int_in(3), int_out(3);
 *    int_in[0] = 0;
 *    int_in[1] = 1;
 *    int_in[2] = 2;
 *
 *    thrust::device_vector<float> float_in(3), float_out(3);
 *    float_in[0] =  0.0f;
 *    float_in[1] = 10.0f;
 *    float_in[2] = 20.0f;
 *
 *    thrust::copy(thrust::make_zip_iterator(thrust::make_tuple(int_in.begin(), float_in.begin())),
 *                 thrust::make_zip_iterator(thrust::make_tuple(int_in.end(),   float_in.end())),
 *                 thrust::make_zip_iterator(thrust::make_tuple(int_out.begin(),float_out.begin())));
 *
 *    // int_out is now [0, 1, 2]
 *    // float_out is now [0.0f, 10.0f, 20.0f]
 *
 *    return 0;
 *  }
 *  \endcode
 *
 *  \see make_zip_iterator
 *  \see make_tuple
 *  \see tuple
 *  \see get
 */
template <typename IteratorTuple>
  class zip_iterator
    : public detail::zip_iterator_base<IteratorTuple>::type
{
  public:
    /*! Null constructor does nothing.
     */
    inline __host__ __device__
    zip_iterator();

    /*! This constructor creates a new \p zip_iterator from a
     *  \p tuple of iterators.
     *  
     *  \param iterator_tuple The \p tuple of iterators to copy from.
     */
    inline __host__ __device__
    zip_iterator(IteratorTuple iterator_tuple);

    /*! This copy constructor creates a new \p zip_iterator from another
     *  \p zip_iterator.
     *
     *  \param other The \p zip_iterator to copy.
     */
    template<typename OtherIteratorTuple>
    inline __host__ __device__
    zip_iterator(const zip_iterator<OtherIteratorTuple> &other,
                 typename thrust::detail::enable_if_convertible<
                   OtherIteratorTuple,
                   IteratorTuple
                 >::type * = 0);

    /*! This method returns a \c const reference to this \p zip_iterator's
     *  \p tuple of iterators.
     *
     *  \return A \c const reference to this \p zip_iterator's \p tuple
     *          of iterators.
     */
    inline __host__ __device__
    const IteratorTuple &get_iterator_tuple() const;

    /*! \cond
     */
  private:
    typedef typename
    detail::zip_iterator_base<IteratorTuple>::type super_t;

    friend class thrust::iterator_core_access;

    // Dereferencing returns a tuple built from the dereferenced
    // iterators in the iterator tuple.
    __host__ __device__
    typename super_t::reference dereference() const;

    // Two zip_iterators are equal if the two first iterators of the
    // tuple are equal. Note this differs from Boost's implementation, which
    // considers the entire tuple.
    template<typename OtherIteratorTuple>
    inline __host__ __device__
    bool equal(const zip_iterator<OtherIteratorTuple> &other) const;

    // Advancing a zip_iterator means to advance all iterators in the tuple
    inline __host__ __device__
    void advance(typename super_t::difference_type n);

    // Incrementing a zip iterator means to increment all iterators in the tuple
    inline __host__ __device__
    void increment();

    // Decrementing a zip iterator means to decrement all iterators in the tuple
    inline __host__ __device__
    void decrement();

    // Distance is calculated using the first iterator in the tuple.
    template<typename OtherIteratorTuple>
    inline __host__ __device__
      typename super_t::difference_type
        distance_to(const zip_iterator<OtherIteratorTuple> &other) const;

    // The iterator tuple.
    IteratorTuple m_iterator_tuple;

    /*! \endcond
     */
}; // end zip_iterator

/*! \p make_zip_iterator creates a \p zip_iterator from a \p tuple
 *  of iterators.
 *
 *  \param t The \p tuple of iterators to copy.
 *  \return A newly created \p zip_iterator which zips the iterators encapsulated in \p t.
 *
 *  \see zip_iterator
 */
template<typename... Iterators>
inline __host__ __device__
zip_iterator<thrust::tuple<Iterators...>> make_zip_iterator(thrust::tuple<Iterators...> t);


/*! \p make_zip_iterator creates a \p zip_iterator from
 *  iterators.
 *
 *  \param its The iterators to copy.
 *  \return A newly created \p zip_iterator which zips the iterators.
 *
 *  \see zip_iterator
 */
template<typename... Iterators>
inline __host__ __device__
zip_iterator<thrust::tuple<Iterators...>> make_zip_iterator(Iterators... its);


/*! \} // end fancyiterators
 */

/*! \} // end iterators
 */

THRUST_NAMESPACE_END

#include <thrust/iterator/detail/zip_iterator.inl>

