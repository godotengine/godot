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


/*! \file thrust/iterator/discard_iterator.h
 *  \brief An iterator which "discards" (ignores) values assigned to it upon dereference
 */

#pragma once

#include <thrust/detail/config.h>
#include <thrust/iterator/detail/discard_iterator_base.h>
#include <thrust/iterator/iterator_facade.h>

THRUST_DISABLE_MSVC_POSSIBLE_LOSS_OF_DATA_WARNING_BEGIN

THRUST_NAMESPACE_BEGIN

/*! \addtogroup iterators
 *  \{
 */

/*! \addtogroup fancyiterator Fancy Iterators
 *  \ingroup iterators
 *  \{
 */

/*! \p discard_iterator is an iterator which represents a special kind of pointer that
 *  ignores values written to it upon dereference. This iterator is useful for ignoring
 *  the output of certain algorithms without wasting memory capacity or bandwidth.
 *  \p discard_iterator may also be used to count the size of an algorithm's output which
 *  may not be known a priori.
 *
 *  The following code snippet demonstrates how to use \p discard_iterator to ignore
 *  ignore one of the output ranges of reduce_by_key
 *
 *  \code
 *  #include <thrust/iterator/discard_iterator.h>
 *  #include <thrust/reduce.h>
 *  #include <thrust/device_vector.h>
 *
 *  int main()
 *  {
 *    thrust::device_vector<int> keys(7), values(7);
 *
 *    keys[0] = 1;
 *    keys[1] = 3;
 *    keys[2] = 3;
 *    keys[3] = 3;
 *    keys[4] = 2;
 *    keys[5] = 2;
 *    keys[6] = 1;
 *
 *    values[0] = 9;
 *    values[1] = 8;
 *    values[2] = 7;
 *    values[3] = 6;
 *    values[4] = 5;
 *    values[5] = 4;
 *    values[6] = 3;
 *
 *    thrust::device_vector<int> result(4);
 *
 *    // we are only interested in the reduced values
 *    // use discard_iterator to ignore the output keys
 *    thrust::reduce_by_key(keys.begin(), keys.end(),
 *                          values.begin(),
 *                          thrust::make_discard_iterator(),
 *                          result.begin());
 *
 *    // result is now [9, 21, 9, 3]
 *
 *    return 0;
 *  }
 *  \endcode
 *
 *  \see make_discard_iterator
 */
template<typename System = use_default>
  class discard_iterator
    : public detail::discard_iterator_base<System>::type
{
    /*! \cond
     */
    friend class thrust::iterator_core_access;
    typedef typename detail::discard_iterator_base<System>::type          super_t;
    typedef typename detail::discard_iterator_base<System>::incrementable incrementable;
    typedef typename detail::discard_iterator_base<System>::base_iterator base_iterator;

  public:
    typedef typename super_t::reference  reference;
    typedef typename super_t::value_type value_type;

    /*! \endcond
     */

    /*! Copy constructor copies from a source discard_iterator.
     *
     *  \p rhs The discard_iterator to copy.
     */
    __host__ __device__
    discard_iterator(discard_iterator const &rhs)
      : super_t(rhs.base()) {}

#if THRUST_CPP_DIALECT >= 2011
    discard_iterator & operator=(const discard_iterator &) = default;
#endif

    /*! This constructor receives an optional index specifying the position of this
     *  \p discard_iterator in a range.
     *
     *  \p i The index of this \p discard_iterator in a range. Defaults to the
     *       value returned by \c Incrementable's null constructor. For example,
     *       when <tt>Incrementable == int</tt>, \c 0.
     */
    __host__ __device__
    discard_iterator(incrementable const &i = incrementable())
      : super_t(base_iterator(i)) {}

    /*! \cond
     */

  private: // Core iterator interface
    __host__ __device__
    reference dereference() const
    {
      return m_element;
    }

    mutable value_type m_element;

    /*! \endcond
     */
}; // end constant_iterator


/*! \p make_discard_iterator creates a \p discard_iterator from an optional index parameter.
 *
 *  \param i The index of the returned \p discard_iterator within a range.
 *           In the default case, the value of this parameter is \c 0.
 *
 *  \return A new \p discard_iterator with index as given by \p i.
 *
 *  \see constant_iterator
 */
inline __host__ __device__
discard_iterator<> make_discard_iterator(discard_iterator<>::difference_type i = discard_iterator<>::difference_type(0))
{
  return discard_iterator<>(i);
} // end make_discard_iterator()

/*! \} // end fancyiterators
 */

/*! \} // end iterators
 */

THRUST_NAMESPACE_END

THRUST_DISABLE_MSVC_POSSIBLE_LOSS_OF_DATA_WARNING_END

