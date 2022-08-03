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

/*! \file thrust/iterator/permutation_iterator.h
 *  \brief An iterator which performs a gather or scatter operation when dereferenced
 */

/*
 * (C) Copyright Toon Knapen    2001.
 * (C) Copyright David Abrahams 2003.
 * (C) Copyright Roland Richter 2003.
 * 
 * Distributed under the Boost Software License, Version 1.0.
 * (See accompanying NOTICE file for the complete license)
 *
 * For more information, see http://www.boost.org
 */

#pragma once

#include <thrust/detail/config.h>
#include <thrust/detail/type_traits.h>
#include <thrust/iterator/detail/permutation_iterator_base.h>
#include <thrust/iterator/iterator_facade.h>
#include <thrust/iterator/iterator_traits.h>

THRUST_NAMESPACE_BEGIN


/*! \addtogroup iterators
 *  \{
 */

/*! \addtogroup fancyiterator Fancy Iterators
 *  \ingroup iterators
 *  \{
 */

/*! \p permutation_iterator is an iterator which represents a pointer into a
 *  reordered view of a given range. \p permutation_iterator is an imprecise name;
 *  the reordered view need not be a strict permutation. This iterator is useful
 *  for fusing a scatter or gather operation with other algorithms.
 *
 *  This iterator takes two arguments:
 *
 *    - an iterator to the range \c V on which the "permutation" will be applied
 *    - the reindexing scheme that defines how the elements of \c V will be permuted.
 *
 *  Note that \p permutation_iterator is not limited to strict permutations of the
 *  given range \c V. The distance between begin and end of the reindexing iterators
 *  is allowed to be smaller compared to the size of the range \c V, in which case
 *  the \p permutation_iterator only provides a "permutation" of a subrange of \c V.
 *  The indices neither need to be unique. In this same context, it must be noted
 *  that the past-the-end \p permutation_iterator is completely defined by means of
 *  the past-the-end iterator to the indices.
 *
 *  The following code snippet demonstrates how to create a \p permutation_iterator
 *  which represents a reordering of the contents of a \p device_vector.
 *
 *  \code
 *  #include <thrust/iterator/permutation_iterator.h>
 *  #include <thrust/device_vector.h>
 *  ...
 *  thrust::device_vector<float> values(8);
 *  values[0] = 10.0f;
 *  values[1] = 20.0f;
 *  values[2] = 30.0f;
 *  values[3] = 40.0f;
 *  values[4] = 50.0f;
 *  values[5] = 60.0f;
 *  values[6] = 70.0f;
 *  values[7] = 80.0f;
 *
 *  thrust::device_vector<int> indices(4);
 *  indices[0] = 2;
 *  indices[1] = 6;
 *  indices[2] = 1;
 *  indices[3] = 3;
 *
 *  typedef thrust::device_vector<float>::iterator ElementIterator;
 *  typedef thrust::device_vector<int>::iterator   IndexIterator;
 *
 *  thrust::permutation_iterator<ElementIterator,IndexIterator> iter(values.begin(), indices.begin());
 *
 *  *iter;   // returns 30.0f;
 *  iter[0]; // returns 30.0f;
 *  iter[1]; // returns 70.0f;
 *  iter[2]; // returns 20.0f;
 *  iter[3]; // returns 40.0f;
 *
 *  // iter[4] is an out-of-bounds error
 *
 *  *iter   = -1.0f; // sets values[2] to -1.0f;
 *  iter[0] = -1.0f; // sets values[2] to -1.0f;
 *  iter[1] = -1.0f; // sets values[6] to -1.0f;
 *  iter[2] = -1.0f; // sets values[1] to -1.0f;
 *  iter[3] = -1.0f; // sets values[3] to -1.0f;
 *
 *  // values is now {10, -1, -1, -1, 50, 60, -1, 80}
 *  \endcode
 *
 *  \see make_permutation_iterator
 */
template <typename ElementIterator,
          typename IndexIterator>
  class permutation_iterator
    : public thrust::detail::permutation_iterator_base<
        ElementIterator,
        IndexIterator
      >::type
{
  /*! \cond
   */
  private:
    typedef typename detail::permutation_iterator_base<ElementIterator,IndexIterator>::type super_t;

    friend class thrust::iterator_core_access;
  /*! \endcond
   */

  public:
    /*! Null constructor calls the null constructor of this \p permutation_iterator's
     *  element iterator.
     */
    __host__ __device__
    permutation_iterator()
      : m_element_iterator() {}

    /*! Constructor accepts an \c ElementIterator into a range of values and an
     *  \c IndexIterator into a range of indices defining the indexing scheme on the
     *  values.
     *
     *  \param x An \c ElementIterator pointing this \p permutation_iterator's range of values.
     *  \param y An \c IndexIterator pointing to an indexing scheme to use on \p x.
     */
    __host__ __device__
    explicit permutation_iterator(ElementIterator x, IndexIterator y)
      : super_t(y), m_element_iterator(x) {}

    /*! Copy constructor accepts a related \p permutation_iterator.
     *  \param r A compatible \p permutation_iterator to copy from.
     */
    template<typename OtherElementIterator, typename OtherIndexIterator>
    __host__ __device__
    permutation_iterator(permutation_iterator<OtherElementIterator,OtherIndexIterator> const &r
    // XXX remove these guards when we have static_assert
    , typename detail::enable_if_convertible<OtherElementIterator, ElementIterator>::type* = 0
    , typename detail::enable_if_convertible<OtherIndexIterator, IndexIterator>::type* = 0
    )
      : super_t(r.base()), m_element_iterator(r.m_element_iterator)
    {}

  /*! \cond
   */
  private:
    // MSVC 2013 and 2015 incorrectly warning about returning a reference to
    // a local/temporary here.
    // See goo.gl/LELTNp
    THRUST_DISABLE_MSVC_WARNING_BEGIN(4172)

    __thrust_exec_check_disable__
    __host__ __device__
    typename super_t::reference dereference() const
    {
      return *(m_element_iterator + *this->base());
    }

    THRUST_DISABLE_MSVC_WARNING_END(4172)

    // make friends for the copy constructor
    template<typename,typename> friend class permutation_iterator;

    ElementIterator m_element_iterator;
  /*! \endcond
   */
}; // end permutation_iterator


/*! \p make_permutation_iterator creates a \p permutation_iterator
 *  from an \c ElementIterator pointing to a range of elements to "permute"
 *  and an \c IndexIterator pointing to a range of indices defining an indexing
 *  scheme on the values.
 *
 *  \param e An \c ElementIterator pointing to a range of values.
 *  \param i An \c IndexIterator pointing to an indexing scheme to use on \p e.
 *  \return A new \p permutation_iterator which permutes the range \p e by \p i.
 *  \see permutation_iterator
 */
template<typename ElementIterator, typename IndexIterator>
__host__ __device__
permutation_iterator<ElementIterator,IndexIterator> make_permutation_iterator(ElementIterator e, IndexIterator i)
{
  return permutation_iterator<ElementIterator,IndexIterator>(e,i);
}

/*! \} // end fancyiterators
 */

/*! \} // end iterators
 */

THRUST_NAMESPACE_END

