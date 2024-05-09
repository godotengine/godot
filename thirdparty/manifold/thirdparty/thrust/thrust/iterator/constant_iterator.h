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


/*! \file thrust/iterator/constant_iterator.h
 *  \brief An iterator which returns a constant value when
 *         dereferenced
 */

#pragma once

#include <thrust/detail/config.h>
#include <thrust/iterator/detail/constant_iterator_base.h>
#include <thrust/iterator/iterator_facade.h>

THRUST_NAMESPACE_BEGIN

/*! \addtogroup iterators
 *  \{
 */

/*! \addtogroup fancyiterator Fancy Iterators
 *  \ingroup iterators
 *  \{
 */

/*! \p constant_iterator is an iterator which represents a pointer into a range
 *  of constant values. This iterator is useful for creating a range filled with the same
 *  value without explicitly storing it in memory. Using \p constant_iterator saves both
 *  memory capacity and bandwidth.
 *
 *  The following code snippet demonstrates how to create a \p constant_iterator whose
 *  \c value_type is \c int and whose value is \c 10.
 *
 *  \code
 *  #include <thrust/iterator/constant_iterator.h>
 *
 *  thrust::constant_iterator<int> iter(10);
 *
 *  *iter;    // returns 10
 *  iter[0];  // returns 10
 *  iter[1];  // returns 10
 *  iter[13]; // returns 10
 *
 *  // and so on...
 *  \endcode
 *
 *  This next example demonstrates how to use a \p constant_iterator with the
 *  \p thrust::transform function to increment all elements of a sequence by the
 *  same value. We will create a temporary \p constant_iterator with the function
 *  \p make_constant_iterator function in order to avoid explicitly specifying
 *  its type:
 *
 *  \code
 *  #include <thrust/iterator/constant_iterator.h>
 *  #include <thrust/transform.h>
 *  #include <thrust/functional.h>
 *  #include <thrust/device_vector.h>
 *
 *  int main()
 *  {
 *    thrust::device_vector<int> data(4);
 *    data[0] = 3;
 *    data[1] = 7;
 *    data[2] = 2;
 *    data[3] = 5;
 *    
 *    // add 10 to all values in data
 *    thrust::transform(data.begin(), data.end(),
 *                      thrust::make_constant_iterator(10),
 *                      data.begin(),
 *                      thrust::plus<int>());
 *    
 *    // data is now [13, 17, 12, 15]
 *    
 *    return 0;
 *  }
 *  \endcode
 *
 *  \see make_constant_iterator
 */
template<typename Value,
         typename Incrementable = use_default,
         typename System = use_default>
  class constant_iterator
    : public detail::constant_iterator_base<Value, Incrementable, System>::type
{
    /*! \cond
     */
    friend class thrust::iterator_core_access;
    typedef typename detail::constant_iterator_base<Value, Incrementable, System>::type          super_t;
    typedef typename detail::constant_iterator_base<Value, Incrementable, System>::incrementable incrementable;
    typedef typename detail::constant_iterator_base<Value, Incrementable, System>::base_iterator base_iterator;

  public:
    typedef typename super_t::reference  reference;
    typedef typename super_t::value_type value_type;

    /*! \endcond
     */

    /*! Null constructor initializes this \p constant_iterator's constant using its
     *  null constructor.
     */
    __host__ __device__
    constant_iterator()
      : super_t(), m_value() {}

    /*! Copy constructor copies the value of another \p constant_iterator into this
     *  \p constant_iterator.
     *
     *  \p rhs The constant_iterator to copy.
     */
    __host__ __device__
    constant_iterator(constant_iterator const &rhs)
      : super_t(rhs.base()), m_value(rhs.m_value) {}

    /*! Copy constructor copies the value of another \p constant_iterator with related
     *  System type.
     *
     *  \param rhs The \p constant_iterator to copy.
     */
    template<typename OtherSystem>
    __host__ __device__
    constant_iterator(constant_iterator<Value,Incrementable,OtherSystem> const &rhs,
                      typename thrust::detail::enable_if_convertible<
                        typename thrust::iterator_system<constant_iterator<Value,Incrementable,OtherSystem> >::type,
                        typename thrust::iterator_system<super_t>::type
                      >::type * = 0)
      : super_t(rhs.base()), m_value(rhs.value()) {}

    /*! This constructor receives a value to use as the constant value of this
     *  \p constant_iterator and an index specifying the location of this
     *  \p constant_iterator in a sequence.
     *  
     *  \p v The value of this \p constant_iterator's constant value.
     *  \p i The index of this \p constant_iterator in a sequence. Defaults to the
     *       value returned by \c Incrementable's null constructor. For example,
     *       when <tt>Incrementable == int</tt>, \c 0.
     */
    __host__ __device__
    constant_iterator(value_type const& v, incrementable const &i = incrementable())
      : super_t(base_iterator(i)), m_value(v) {}

    /*! This constructor is templated to allow construction from a value type and
     *  incrementable type related this this \p constant_iterator's respective types.
     *
     *  \p v The value of this \p constant_iterator's constant value.
     *  \p i The index of this \p constant_iterator in a sequence. Defaults to the
     *       value returned by \c Incrementable's null constructor. For example,
     *       when <tt>Incrementable == int</tt>, \c 0.
     */
    template<typename OtherValue, typename OtherIncrementable>
    __host__ __device__
    constant_iterator(OtherValue const& v, OtherIncrementable const& i = incrementable())
      : super_t(base_iterator(i)), m_value(v) {}

    /*! This method returns the value of this \p constant_iterator's constant value.
     *  \return A \c const reference to this \p constant_iterator's constant value.
     */
    __host__ __device__
    Value const& value() const
    { return m_value; }

    /*! \cond
     */

  protected:
    __host__ __device__
    Value const& value_reference() const
    { return m_value; }

    __host__ __device__
    Value & value_reference()
    { return m_value; }
  
  private: // Core iterator interface
    __host__ __device__
    reference dereference() const
    {
      return m_value;
    }

  private:
    Value m_value;

    /*! \endcond
     */
}; // end constant_iterator


/*! This version of \p make_constant_iterator creates a \p constant_iterator
 *  from values given for both value and index. The type of \p constant_iterator
 *  may be inferred by the compiler from the types of its parameters.
 *
 *  \param x The value of the returned \p constant_iterator's constant value.
 *  \param i The index of the returned \p constant_iterator within a sequence.
 *           The type of this parameter defaults to \c int. In the default case,
 *           the value of this parameter is \c 0.
 *
 *  \return A new \p constant_iterator with constant value & index as given
 *          by \p x & \p i.
 *
 *  \see constant_iterator
 */
template<typename ValueT, typename IndexT>
inline __host__ __device__
constant_iterator<ValueT, IndexT> make_constant_iterator(ValueT x, IndexT i = int())
{
  return constant_iterator<ValueT, IndexT>(x, i);
} // end make_constant_iterator()


/*! This version of \p make_constant_iterator creates a \p constant_iterator
 *  using only a parameter for the desired constant value. The value of the
 *  returned \p constant_iterator's index is set to \c 0.
 *
 *  \param x The value of the returned \p constant_iterator's constant value.
 *  \return A new \p constant_iterator with constant value equal to \p x and
 *          index equal to \c 0.
 *  \see constant_iterator
 */
template<typename V>
inline __host__ __device__
constant_iterator<V> make_constant_iterator(V x)
{
  return constant_iterator<V>(x, 0);
} // end make_constant_iterator()

/*! \} // end fancyiterators
 */

/*! \} // end iterators
 */

THRUST_NAMESPACE_END

