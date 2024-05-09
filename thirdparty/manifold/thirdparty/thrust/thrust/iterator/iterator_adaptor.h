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


/*! \file thrust/iterator/iterator_adaptor.h
 *  \brief An iterator which adapts a base iterator
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
#include <thrust/iterator/iterator_facade.h>
#include <thrust/detail/use_default.h>
#include <thrust/iterator/detail/iterator_adaptor_base.h>

THRUST_NAMESPACE_BEGIN

/*! \addtogroup iterators
 *  \{
 */

/*! \addtogroup fancyiterator Fancy Iterators
 *  \ingroup iterators
 *  \{
 */

/*! \p iterator_adaptor is an iterator which adapts an existing type of iterator to create a new type of
 *  iterator. Most of Thrust's fancy iterators are defined via inheritance from \p iterator_adaptor.
 *  While composition of these existing Thrust iterators is often sufficient for expressing the desired
 *  functionality, it is occasionally more straightforward to derive from \p iterator_adaptor directly.
 *
 *  To see how to use \p iterator_adaptor to create a novel iterator type, let's examine how to use it to
 *  define \p repeat_iterator, a fancy iterator which repeats elements from another range a given number of time:
 *
 *  \code
 *  #include <thrust/iterator/iterator_adaptor.h>
 *
 *  // derive repeat_iterator from iterator_adaptor
 *  template<typename Iterator>
 *    class repeat_iterator
 *      : public thrust::iterator_adaptor<
 *          repeat_iterator<Iterator>, // the first template parameter is the name of the iterator we're creating
 *          Iterator                   // the second template parameter is the name of the iterator we're adapting
 *                                     // we can use the default for the additional template parameters
 *        >
 *  {
 *    public:
 *      // shorthand for the name of the iterator_adaptor we're deriving from
 *      typedef thrust::iterator_adaptor<
 *        repeat_iterator<Iterator>,
 *        Iterator
 *      > super_t;
 *
 *      __host__ __device__
 *      repeat_iterator(const Iterator &x, int n) : super_t(x), begin(x), n(n) {}
 *
 *      // befriend thrust::iterator_core_access to allow it access to the private interface below
 *      friend class thrust::iterator_core_access;
 *
 *    private:
 *      // repeat each element of the adapted range n times
 *      unsigned int n;
 *
 *      // used to keep track of where we began
 *      const Iterator begin;
 *
 *      // it is private because only thrust::iterator_core_access needs access to it
 *      __host__ __device__
 *      typename super_t::reference dereference() const
 *      {
 *        return *(begin + (this->base() - begin) / n);
 *      }
 *  };
 *  \endcode
 *
 *  Except for the first two, \p iterator_adaptor's template parameters are optional. When omitted, or when the
 *  user specifies \p thrust::use_default in its place, \p iterator_adaptor will use a default type inferred from \p Base.
 *
 *  \p iterator_adaptor's functionality is derived from and generally equivalent to \p boost::iterator_adaptor.
 *  The exception is Thrust's addition of the template parameter \p System, which is necessary to allow Thrust
 *  to dispatch an algorithm to one of several parallel backend systems.
 *
 *  \p iterator_adaptor is a powerful tool for creating custom iterators directly. However, the large set of iterator semantics which must be satisfied
 *  for algorithm compatibility can make \p iterator_adaptor difficult to use correctly. Unless you require the full expressivity of \p iterator_adaptor,
 *  consider building a custom iterator through composition of existing higher-level fancy iterators instead. 
 *
 *  Interested users may refer to <tt>boost::iterator_adaptor</tt>'s documentation for further usage examples.
 */
template<typename Derived,
         typename Base,
         typename Value      = use_default,
         typename System     = use_default,
         typename Traversal  = use_default,
         typename Reference  = use_default,
         typename Difference = use_default>
  class iterator_adaptor:
    public detail::iterator_adaptor_base<
      Derived, Base, Value, System, Traversal, Reference, Difference
    >::type
{
  /*! \cond
   */

    friend class thrust::iterator_core_access;

  protected:
    typedef typename detail::iterator_adaptor_base<
        Derived, Base, Value, System, Traversal, Reference, Difference
    >::type super_t;

  /*! \endcond
   */
  
  public:
    /*! \p iterator_adaptor's default constructor does nothing.
     */
    __host__ __device__
    iterator_adaptor(){}

    /*! This constructor copies from a given instance of the \p Base iterator.
     */
    __thrust_exec_check_disable__
    __host__ __device__
    explicit iterator_adaptor(Base const& iter)
      : m_iterator(iter)
    {}

    /*! The type of iterator this \p iterator_adaptor's \p adapts.
     */
    typedef Base       base_type;
                                                                                              
    /*! \cond
     */
    typedef typename super_t::reference reference;
                                                                                              
    typedef typename super_t::difference_type difference_type;
    /*! \endcond
     */

    /*! \return A \p const reference to the \p Base iterator this \p iterator_adaptor adapts.
     */
    __host__ __device__
    Base const& base() const
    { return m_iterator; }

  protected:
    /*! \return A \p const reference to the \p Base iterator this \p iterator_adaptor adapts.
     */
    __host__ __device__
    Base const& base_reference() const
    { return m_iterator; }

    /*! \return A mutable reference to the \p Base iterator this \p iterator_adaptor adapts.
     */
    __host__ __device__
    Base& base_reference()
    { return m_iterator; }

    /*! \cond
     */
  private: // Core iterator interface for iterator_facade

    __thrust_exec_check_disable__
    __host__ __device__
    typename iterator_adaptor::reference dereference() const
    { return *m_iterator; }

    __thrust_exec_check_disable__
    template<typename OtherDerived, typename OtherIterator, typename V, typename S, typename T, typename R, typename D>
    __host__ __device__
    bool equal(iterator_adaptor<OtherDerived, OtherIterator, V, S, T, R, D> const& x) const
    { return m_iterator == x.base(); }

    __thrust_exec_check_disable__
    __host__ __device__
    void advance(typename iterator_adaptor::difference_type n)
    {
      // XXX statically assert on random_access_traversal_tag

      // counting_iterator will pick eg. diff_t=int64 when base=int32.
      // Explicitly cast to avoid static conversion warnings.
      m_iterator = static_cast<base_type>(m_iterator + n);
    }

    __thrust_exec_check_disable__
    __host__ __device__
    void increment()
    { ++m_iterator; }

    __thrust_exec_check_disable__
    __host__ __device__
    void decrement()
    {
      // XXX statically assert on bidirectional_traversal_tag
      --m_iterator;
    }

    __thrust_exec_check_disable__
    template<typename OtherDerived, typename OtherIterator, typename V, typename S, typename T, typename R, typename D>
    __host__ __device__
    typename iterator_adaptor::difference_type distance_to(iterator_adaptor<OtherDerived, OtherIterator, V, S, T, R, D> const& y) const
    { return y.base() - m_iterator; }

  private:
    Base m_iterator;

    /*! \endcond
     */
}; // end iterator_adaptor

/*! \} // end fancyiterators
 */

/*! \} // end iterators
 */

THRUST_NAMESPACE_END

