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

/*! \file thrust/iterator/iterator_facade.h
 *  \brief A class which exposes a public interface for iterators
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
#include <thrust/iterator/detail/iterator_facade_category.h>
#include <thrust/iterator/detail/distance_from_result.h>

THRUST_NAMESPACE_BEGIN

/*! \addtogroup iterators
 *  \{
 */

/*! \addtogroup fancyiterator Fancy Iterators
 *  \ingroup iterators
 *  \{
 */


// This forward declaration is required for the friend declaration
// in iterator_core_access
template<typename Derived, typename Value, typename System, typename Traversal, typename Reference, typename Difference> class iterator_facade;


/*! \p iterator_core_access is the class which user iterator types derived from \p thrust::iterator_adaptor
 *  or \p thrust::iterator_facade must befriend to allow it to access their private interface.
 */
class iterator_core_access
{
    /*! \cond
     */

    // declare our friends
    template<typename Derived, typename Value, typename System, typename Traversal, typename Reference, typename Difference> friend class iterator_facade;

    // iterator comparisons are our friends
    template <typename Derived1, typename Value1, typename System1, typename Traversal1, typename Reference1, typename Difference1,
              typename Derived2, typename Value2, typename System2, typename Traversal2, typename Reference2, typename Difference2>
    inline __host__ __device__
    friend bool
    operator ==(iterator_facade<Derived1,Value1,System1,Traversal1,Reference1,Difference1> const& lhs,
                iterator_facade<Derived2,Value2,System2,Traversal2,Reference2,Difference2> const& rhs);

    template <typename Derived1, typename Value1, typename System1, typename Traversal1, typename Reference1, typename Difference1,
              typename Derived2, typename Value2, typename System2, typename Traversal2, typename Reference2, typename Difference2>
    inline __host__ __device__
    friend bool
    operator !=(iterator_facade<Derived1,Value1,System1,Traversal1,Reference1,Difference1> const& lhs,
                iterator_facade<Derived2,Value2,System2,Traversal2,Reference2,Difference2> const& rhs);

    template <typename Derived1, typename Value1, typename System1, typename Traversal1, typename Reference1, typename Difference1,
              typename Derived2, typename Value2, typename System2, typename Traversal2, typename Reference2, typename Difference2>
    inline __host__ __device__
    friend bool
    operator <(iterator_facade<Derived1,Value1,System1,Traversal1,Reference1,Difference1> const& lhs,
               iterator_facade<Derived2,Value2,System2,Traversal2,Reference2,Difference2> const& rhs);

    template <typename Derived1, typename Value1, typename System1, typename Traversal1, typename Reference1, typename Difference1,
              typename Derived2, typename Value2, typename System2, typename Traversal2, typename Reference2, typename Difference2>
    inline __host__ __device__
    friend bool
    operator >(iterator_facade<Derived1,Value1,System1,Traversal1,Reference1,Difference1> const& lhs,
               iterator_facade<Derived2,Value2,System2,Traversal2,Reference2,Difference2> const& rhs);

    template <typename Derived1, typename Value1, typename System1, typename Traversal1, typename Reference1, typename Difference1,
              typename Derived2, typename Value2, typename System2, typename Traversal2, typename Reference2, typename Difference2>
    inline __host__ __device__
    friend bool
    operator <=(iterator_facade<Derived1,Value1,System1,Traversal1,Reference1,Difference1> const& lhs,
                iterator_facade<Derived2,Value2,System2,Traversal2,Reference2,Difference2> const& rhs);

    template <typename Derived1, typename Value1, typename System1, typename Traversal1, typename Reference1, typename Difference1,
              typename Derived2, typename Value2, typename System2, typename Traversal2, typename Reference2, typename Difference2>
    inline __host__ __device__
    friend bool
    operator >=(iterator_facade<Derived1,Value1,System1,Traversal1,Reference1,Difference1> const& lhs,
                iterator_facade<Derived2,Value2,System2,Traversal2,Reference2,Difference2> const& rhs);

    // iterator difference is our friend
    template <typename Derived1, typename Value1, typename System1, typename Traversal1, typename Reference1, typename Difference1,
              typename Derived2, typename Value2, typename System2, typename Traversal2, typename Reference2, typename Difference2>
    inline __host__ __device__
    friend
      typename thrust::detail::distance_from_result<
        iterator_facade<Derived1,Value1,System1,Traversal1,Reference1,Difference1>,
        iterator_facade<Derived2,Value2,System2,Traversal2,Reference2,Difference2>
      >::type
    operator-(iterator_facade<Derived1,Value1,System1,Traversal1,Reference1,Difference1> const& lhs,
              iterator_facade<Derived2,Value2,System2,Traversal2,Reference2,Difference2> const& rhs);

    template<typename Facade>
    __host__ __device__
    static typename Facade::reference dereference(Facade const& f)
    {
      return f.dereference();
    }

    template<typename Facade>
    __host__ __device__
    static void increment(Facade& f)
    {
      f.increment();
    }

    template<typename Facade>
    __host__ __device__
    static void decrement(Facade& f)
    {
      f.decrement();
    }

    template <class Facade1, class Facade2>
    __host__ __device__
    static bool equal(Facade1 const& f1, Facade2 const& f2)
    {
      return f1.equal(f2);
    }

    // XXX TODO: Investigate whether we need both of these cases
    //template <class Facade1, class Facade2>
    //__host__ __device__
    //static bool equal(Facade1 const& f1, Facade2 const& f2, mpl::true_)
    //{
    //  return f1.equal(f2);
    //}

    //template <class Facade1, class Facade2>
    //__host__ __device__
    //static bool equal(Facade1 const& f1, Facade2 const& f2, mpl::false_)
    //{
    //  return f2.equal(f1);
    //}

    template <class Facade>
    __host__ __device__
    static void advance(Facade& f, typename Facade::difference_type n)
    {
      f.advance(n);
    }

    // Facade2 is convertible to Facade1,
    // so return Facade1's difference_type
    template <class Facade1, class Facade2>
    __host__ __device__
    static typename Facade1::difference_type
      distance_from(Facade1 const& f1, Facade2 const& f2, thrust::detail::true_type)
    {
      return -f1.distance_to(f2);
    }

    // Facade2 is not convertible to Facade1,
    // so return Facade2's difference_type
    template <class Facade1, class Facade2>
    __host__ __device__
    static typename Facade2::difference_type
      distance_from(Facade1 const& f1, Facade2 const& f2, thrust::detail::false_type)
    {
      return f2.distance_to(f1);
    }
    
    template <class Facade1, class Facade2>
    __host__ __device__
    static typename thrust::detail::distance_from_result<Facade1,Facade2>::type
      distance_from(Facade1 const& f1, Facade2 const& f2)
    {
      // dispatch the implementation of this method upon whether or not
      // Facade2 is convertible to Facade1
      return distance_from(f1, f2,
        typename thrust::detail::is_convertible<Facade2,Facade1>::type());
    }

    //
    // Curiously Recurring Template interface.
    //
    template <typename Derived, typename Value, typename System, typename Traversal, typename Reference, typename Difference>
    __host__ __device__
    static Derived& derived(iterator_facade<Derived,Value,System,Traversal,Reference,Difference>& facade)
    {
      return *static_cast<Derived*>(&facade);
    }

    template <typename Derived, typename Value, typename System, typename Traversal, typename Reference, typename Difference>
    __host__ __device__
    static Derived const& derived(iterator_facade<Derived,Value,System,Traversal,Reference,Difference> const& facade)
    {
      return *static_cast<Derived const*>(&facade);
    }

    /*! \endcond
     */
}; // end iterator_core_access


/*! \p iterator_facade is a template which allows the programmer to define a novel iterator with a standards-conforming interface
 *  which Thrust can use to reason about algorithm acceleration opportunities.
 *
 *  Because most of a standard iterator's interface is defined in terms of a small set of core primitives, \p iterator_facade
 *  defines the non-primitive portion mechanically. In principle a novel iterator could explicitly provide the entire interface in
 *  an ad hoc fashion but doing so might be tedious and prone to subtle errors.
 *
 *  Often \p iterator_facade is too primitive a tool to use for defining novel iterators. In these cases, \p iterator_adaptor
 *  or a specific fancy iterator should be used instead.
 *
 *  \p iterator_facade's functionality is derived from and generally equivalent to \p boost::iterator_facade.
 *  The exception is Thrust's addition of the template parameter \p System, which is necessary to allow Thrust
 *  to dispatch an algorithm to one of several parallel backend systems. An additional exception is Thrust's omission
 *  of the \c operator-> member function.
 *
 *  Interested users may refer to <tt>boost::iterator_facade</tt>'s documentation for usage examples.
 *
 *  \note \p iterator_facade's arithmetic operator free functions exist with the usual meanings but are omitted here for brevity.
 */
template<typename Derived,
         typename Value,
         typename System,
         typename Traversal,
         typename Reference,
         typename Difference = std::ptrdiff_t>
  class iterator_facade
{
  private:
    /*! \cond
     */

    //
    // Curiously Recurring Template interface.
    //
    __host__ __device__
    Derived& derived()
    {
      return *static_cast<Derived*>(this);
    }

    __host__ __device__
    Derived const& derived() const
    {
      return *static_cast<Derived const*>(this);
    }
    /*! \endcond
     */

  public:
    /*! The type of element pointed to by \p iterator_facade.
     */
    typedef typename thrust::detail::remove_const<Value>::type value_type;

    /*! The return type of \p iterator_facade::operator*().
     */
    typedef Reference                                          reference;

    /*! The return type of \p iterator_facade's non-existent \c operator->()
     *  member function. Unlike \c boost::iterator_facade, \p iterator_facade
     *  disallows access to the \p value_type's members through expressions of the
     *  form <tt>iter->member</tt>. \p pointer is defined to \c void to indicate
     *  that these expressions are not allowed. This limitation may be relaxed in a
     *  future version of Thrust.
     */
    typedef void                                               pointer;

    /*! The type of expressions of the form <tt>x - y</tt> where <tt>x</tt> and <tt>y</tt>
     *  are of type \p iterator_facade.
     */
    typedef Difference                                         difference_type;

    /*! The type of iterator category of \p iterator_facade.
     */
    typedef typename thrust::detail::iterator_facade_category<
      System, Traversal, Value, Reference
    >::type                                                    iterator_category;

    /*! \p operator*() dereferences this \p iterator_facade.
     *  \return A reference to the element pointed to by this \p iterator_facade.
     */
    __host__ __device__
    reference operator*() const
    {
      return iterator_core_access::dereference(this->derived());
    }

    // XXX unimplemented for now, consider implementing it later
    //pointer operator->() const
    //{
    //  return;
    //}

    // XXX investigate whether or not we need to go to the lengths
    //     boost does to determine the return type

    /*! \p operator[] performs indexed dereference.
     *  \return A reference to the element \p n distance away from this \p iterator_facade.
     */
    __host__ __device__
    reference operator[](difference_type n) const
    {
      return *(this->derived() + n);
    }

    /*! \p operator++ pre-increments this \p iterator_facade to refer to the element in the next position.
     *  \return <tt>*this</tt>
     */
    __host__ __device__
    Derived& operator++()
    {
      iterator_core_access::increment(this->derived());
      return this->derived();
    }

    /*! \p operator++ post-increments this \p iterator_facade and returns a new \p iterator_facade referring to the element in the next position.
     *  \return A copy of <tt>*this</tt> before increment.
     */
    __host__ __device__
    Derived  operator++(int)
    {
      Derived tmp(this->derived());
      ++*this;
      return tmp;
    }

    /*! \p operator-- pre-decrements this \p iterator_facade to refer to the element in the previous position.
     *  \return <tt>*this</tt>
     */
    __host__ __device__
    Derived& operator--()
    {
      iterator_core_access::decrement(this->derived());
      return this->derived();
    }

    /*! \p operator-- post-decrements this \p iterator_facade and returns a new \p iterator_facade referring to the element in the previous position.
     *  \return A copy of <tt>*this</tt> before decrement.
     */
    __host__ __device__
    Derived  operator--(int)
    {
      Derived tmp(this->derived());
      --*this;
      return tmp;
    }

    /*! \p operator+= increments this \p iterator_facade to refer to an element a given distance after its current position.
     *  \param n The quantity to increment.
     *  \return <tt>*this</tt>
     */
    __host__ __device__
    Derived& operator+=(difference_type n)
    {
      iterator_core_access::advance(this->derived(), n);
      return this->derived();
    }

    /*! \p operator-= decrements this \p iterator_facade to refer to an element a given distance before its current postition.
     *  \param n The quantity to decrement.
     *  \return <tt>*this</tt>
     */
    __host__ __device__
    Derived& operator-=(difference_type n)
    {
      iterator_core_access::advance(this->derived(), -n);
      return this->derived();
    }

    /*! \p operator- subtracts a given quantity from this \p iterator_facade and returns a new \p iterator_facade referring to the element at the given position before this \p iterator_facade.
     *  \param n The quantity to decrement
     *  \return An \p iterator_facade pointing \p n elements before this \p iterator_facade.
     */
    __host__ __device__
    Derived  operator-(difference_type n) const
    {
      Derived result(this->derived());
      return result -= n;
    }
}; // end iterator_facade

/*! \cond
 */

// Comparison operators
template <typename Derived1, typename Value1, typename System1, typename Traversal1, typename Reference1, typename Difference1,
          typename Derived2, typename Value2, typename System2, typename Traversal2, typename Reference2, typename Difference2>
inline __host__ __device__
// XXX it might be nice to implement this at some point
//typename enable_if_interoperable<Dr1,Dr2,bool>::type // exposition
bool
operator ==(iterator_facade<Derived1,Value1,System1,Traversal1,Reference1,Difference1> const& lhs,
            iterator_facade<Derived2,Value2,System2,Traversal2,Reference2,Difference2> const& rhs)
{
  return iterator_core_access
    ::equal(*static_cast<Derived1 const*>(&lhs),
            *static_cast<Derived2 const*>(&rhs));
}

template <typename Derived1, typename Value1, typename System1, typename Traversal1, typename Reference1, typename Difference1,
          typename Derived2, typename Value2, typename System2, typename Traversal2, typename Reference2, typename Difference2>
inline __host__ __device__
// XXX it might be nice to implement this at some point
//typename enable_if_interoperable<Dr1,Dr2,bool>::type // exposition
bool
operator !=(iterator_facade<Derived1,Value1,System1,Traversal1,Reference1,Difference1> const& lhs,
            iterator_facade<Derived2,Value2,System2,Traversal2,Reference2,Difference2> const& rhs)
{
  return !iterator_core_access
    ::equal(*static_cast<Derived1 const*>(&lhs),
            *static_cast<Derived2 const*>(&rhs));
}

template <typename Derived1, typename Value1, typename System1, typename Traversal1, typename Reference1, typename Difference1,
          typename Derived2, typename Value2, typename System2, typename Traversal2, typename Reference2, typename Difference2>
inline __host__ __device__
// XXX it might be nice to implement this at some point
//typename enable_if_interoperable<Dr1,Dr2,bool>::type // exposition
bool
operator <(iterator_facade<Derived1,Value1,System1,Traversal1,Reference1,Difference1> const& lhs,
           iterator_facade<Derived2,Value2,System2,Traversal2,Reference2,Difference2> const& rhs)
{
  return 0 > iterator_core_access
    ::distance_from(*static_cast<Derived1 const*>(&lhs),
                    *static_cast<Derived2 const*>(&rhs));
}

template <typename Derived1, typename Value1, typename System1, typename Traversal1, typename Reference1, typename Difference1,
          typename Derived2, typename Value2, typename System2, typename Traversal2, typename Reference2, typename Difference2>
inline __host__ __device__
// XXX it might be nice to implement this at some point
//typename enable_if_interoperable<Dr1,Dr2,bool>::type // exposition
bool
operator >(iterator_facade<Derived1,Value1,System1,Traversal1,Reference1,Difference1> const& lhs,
           iterator_facade<Derived2,Value2,System2,Traversal2,Reference2,Difference2> const& rhs)
{
  return 0 < iterator_core_access
    ::distance_from(*static_cast<Derived1 const*>(&lhs),
                    *static_cast<Derived2 const*>(&rhs));
}

template <typename Derived1, typename Value1, typename System1, typename Traversal1, typename Reference1, typename Difference1,
          typename Derived2, typename Value2, typename System2, typename Traversal2, typename Reference2, typename Difference2>
inline __host__ __device__
// XXX it might be nice to implement this at some point
//typename enable_if_interoperable<Dr1,Dr2,bool>::type // exposition
bool
operator <=(iterator_facade<Derived1,Value1,System1,Traversal1,Reference1,Difference1> const& lhs,
            iterator_facade<Derived2,Value2,System2,Traversal2,Reference2,Difference2> const& rhs)
{
  return 0 >= iterator_core_access
    ::distance_from(*static_cast<Derived1 const*>(&lhs),
                    *static_cast<Derived2 const*>(&rhs));
}

template <typename Derived1, typename Value1, typename System1, typename Traversal1, typename Reference1, typename Difference1,
          typename Derived2, typename Value2, typename System2, typename Traversal2, typename Reference2, typename Difference2>
inline __host__ __device__
// XXX it might be nice to implement this at some point
//typename enable_if_interoperable<Dr1,Dr2,bool>::type // exposition
bool
operator >=(iterator_facade<Derived1,Value1,System1,Traversal1,Reference1,Difference1> const& lhs,
            iterator_facade<Derived2,Value2,System2,Traversal2,Reference2,Difference2> const& rhs)
{
  return 0 <= iterator_core_access
    ::distance_from(*static_cast<Derived1 const*>(&lhs),
                    *static_cast<Derived2 const*>(&rhs));
}

// Iterator difference
template <typename Derived1, typename Value1, typename System1, typename Traversal1, typename Reference1, typename Difference1,
          typename Derived2, typename Value2, typename System2, typename Traversal2, typename Reference2, typename Difference2>
inline __host__ __device__

// divine the type this operator returns
typename thrust::detail::distance_from_result<
  iterator_facade<Derived1,Value1,System1,Traversal1,Reference1,Difference1>,
  iterator_facade<Derived2,Value2,System2,Traversal2,Reference2,Difference2>
>::type

operator-(iterator_facade<Derived1,Value1,System1,Traversal1,Reference1,Difference1> const& lhs,
          iterator_facade<Derived2,Value2,System2,Traversal2,Reference2,Difference2> const& rhs)
{
  return iterator_core_access
    ::distance_from(*static_cast<Derived1 const*>(&lhs),
                    *static_cast<Derived2 const*>(&rhs));
}

// Iterator addition
template <typename Derived, typename Value, typename System, typename Traversal, typename Reference, typename Difference>
inline __host__ __device__
Derived operator+ (iterator_facade<Derived,Value,System,Traversal,Reference,Difference> const& i,
                   typename Derived::difference_type n)
{
  Derived tmp(static_cast<Derived const&>(i));
  return tmp += n;
}

template <typename Derived, typename Value, typename System, typename Traversal, typename Reference, typename Difference>
inline __host__ __device__
Derived operator+ (typename Derived::difference_type n,
                   iterator_facade<Derived,Value,System,Traversal,Reference,Difference> const& i)
{
  Derived tmp(static_cast<Derived const&>(i));
  return tmp += n;
}

/*! \endcond
 */

/*! \} // end fancyiterators
 */

/*! \} // end iterators
 */

THRUST_NAMESPACE_END

