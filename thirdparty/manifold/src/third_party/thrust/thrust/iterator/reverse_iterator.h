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


/*! \file thrust/iterator/reverse_iterator.h
 *  \brief An iterator adaptor which adapts another iterator to traverse backwards
 */

/*
 * (C) Copyright David Abrahams 2002.
 * (C) Copyright Jeremy Siek    2002.
 * (C) Copyright Thomas Witt    2002.
 * 
 * Distributed under the Boost Software License, Version 1.0.
 * (See accompanying NOTICE file for the complete license)
 *
 * For more information, see http://www.boost.org
 */

#pragma once

#include <thrust/detail/config.h>
#include <thrust/detail/type_traits.h>
#include <thrust/iterator/detail/reverse_iterator_base.h>
#include <thrust/iterator/iterator_facade.h>

THRUST_NAMESPACE_BEGIN

/*! \addtogroup iterators
 *  \{
 */

/*! \addtogroup fancyiterator Fancy Iterators
 *  \ingroup iterators
 *  \{
 */

/*! \p reverse_iterator is an iterator which represents a pointer into a
 *  reversed view of a given range. In this way, \p reverse_iterator allows
 *  backwards iteration through a bidirectional input range.
 *
 *  It is important to note that although \p reverse_iterator is constructed
 *  from a given iterator, it points to the element preceding it. In this way,
 *  the past-the-end \p reverse_iterator of a given range points to the element
 *  preceding the first element of the input range. By the same token, the first
 *  \p reverse_iterator of a given range is constructed from a past-the-end iterator
 *  of the original range yet points to the last element of the input.
 *
 *  The following code snippet demonstrates how to create a \p reverse_iterator
 *  which represents a reversed view of the contents of a \p device_vector.
 *
 *  \code
 *  #include <thrust/iterator/reverse_iterator.h>
 *  #include <thrust/device_vector.h>
 *  ...
 *  thrust::device_vector<float> v(4);
 *  v[0] = 0.0f;
 *  v[1] = 1.0f;
 *  v[2] = 2.0f;
 *  v[3] = 3.0f;
 *
 *  typedef thrust::device_vector<float>::iterator Iterator;
 *
 *  // note that we point the iterator to the *end* of the device_vector
 *  thrust::reverse_iterator<Iterator> iter(values.end());
 *
 *  *iter;   // returns 3.0f;
 *  iter[0]; // returns 3.0f;
 *  iter[1]; // returns 2.0f;
 *  iter[2]; // returns 1.0f;
 *  iter[3]; // returns 0.0f;
 *
 *  // iter[4] is an out-of-bounds error
 *  \endcode
 *
 *  Since reversing a range is a common operation, containers like \p device_vector
 *  have nested typedefs for declaration shorthand and methods for constructing
 *  reverse_iterators. The following code snippet is equivalent to the previous:
 *
 *  \code
 *  #include <thrust/device_vector.h>
 *  ...
 *  thrust::device_vector<float> v(4);
 *  v[0] = 0.0f;
 *  v[1] = 1.0f;
 *  v[2] = 2.0f;
 *  v[3] = 3.0f;
 *
 *  // we use the nested type reverse_iterator to refer to a reversed view of
 *  // a device_vector and the method rbegin() to create a reverse_iterator pointing
 *  // to the beginning of the reversed device_vector
 *  thrust::device_iterator<float>::reverse_iterator iter = values.rbegin();
 *
 *  *iter;   // returns 3.0f;
 *  iter[0]; // returns 3.0f;
 *  iter[1]; // returns 2.0f;
 *  iter[2]; // returns 1.0f;
 *  iter[3]; // returns 0.0f;
 *
 *  // iter[4] is an out-of-bounds error
 *
 *  // similarly, rend() points to the end of the reversed sequence:
 *  assert(values.rend() == (iter + 4));
 *  \endcode
 *
 *  Finally, the following code snippet demonstrates how to use reverse_iterator to
 *  perform a reversed prefix sum operation on the contents of a device_vector:
 *
 *  \code
 *  #include <thrust/device_vector.h>
 *  #include <thrust/scan.h>
 *  ...
 *  thrust::device_vector<int> v(5);
 *  v[0] = 0;
 *  v[1] = 1;
 *  v[2] = 2;
 *  v[3] = 3;
 *  v[4] = 4;
 *
 *  thrust::device_vector<int> result(5);
 *
 *  // exclusive scan v into result in reverse
 *  thrust::exclusive_scan(v.rbegin(), v.rend(), result.begin());
 *
 *  // result is now {0, 4, 7, 9, 10}
 *  \endcode
 *
 *  \see make_reverse_iterator
 */
template<typename BidirectionalIterator>
  class reverse_iterator
    : public detail::reverse_iterator_base<BidirectionalIterator>::type
{
  /*! \cond
   */
  private:
    typedef typename thrust::detail::reverse_iterator_base<
      BidirectionalIterator
    >::type super_t;

    friend class thrust::iterator_core_access;
  /*! \endcond
   */

  public:
    /*! Default constructor does nothing.
     */
    __host__ __device__
    reverse_iterator() {}

    /*! \p Constructor accepts a \c BidirectionalIterator pointing to a range
     *  for this \p reverse_iterator to reverse.
     *
     *  \param x A \c BidirectionalIterator pointing to a range to reverse.
     */
    __host__ __device__
    explicit reverse_iterator(BidirectionalIterator x);

    /*! \p Copy constructor allows construction from a related compatible
     *  \p reverse_iterator.
     *
     *  \param r A \p reverse_iterator to copy from.
     */
    template<typename OtherBidirectionalIterator>
    __host__ __device__
    reverse_iterator(reverse_iterator<OtherBidirectionalIterator> const &r
// XXX msvc screws this up
// XXX remove these guards when we have static_assert
#if THRUST_HOST_COMPILER != THRUST_HOST_COMPILER_MSVC
                     , typename thrust::detail::enable_if<
                         thrust::detail::is_convertible<
                           OtherBidirectionalIterator,
                           BidirectionalIterator
                         >::value
                       >::type * = 0
#endif // MSVC
                     );

  /*! \cond
   */
  private:
    __thrust_exec_check_disable__
    __host__ __device__
    typename super_t::reference dereference() const;

    __host__ __device__
    void increment();

    __host__ __device__
    void decrement();

    __host__ __device__
    void advance(typename super_t::difference_type n);

    template<typename OtherBidirectionalIterator>
    __host__ __device__
    typename super_t::difference_type
    distance_to(reverse_iterator<OtherBidirectionalIterator> const &y) const;
  /*! \endcond
   */
}; // end reverse_iterator


/*! \p make_reverse_iterator creates a \p reverse_iterator
 *  from a \c BidirectionalIterator pointing to a range of elements to reverse.
 *  
 *  \param x A \c BidirectionalIterator pointing to a range to reverse.
 *  \return A new \p reverse_iterator which reverses the range \p x.
 */
template<typename BidirectionalIterator>
__host__ __device__
reverse_iterator<BidirectionalIterator> make_reverse_iterator(BidirectionalIterator x);


/*! \} // end fancyiterators
 */

/*! \} // end iterators
 */

THRUST_NAMESPACE_END

#include <thrust/iterator/detail/reverse_iterator.inl>

