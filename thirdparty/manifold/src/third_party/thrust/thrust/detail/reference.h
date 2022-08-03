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

/*! \file 
 *  \brief A pointer to a variable which resides in memory associated with a
 *  system.
 */

#pragma once

#include <thrust/detail/config.h>
#include <thrust/detail/reference_forward_declaration.h>
#include <thrust/iterator/iterator_traits.h>
#include <thrust/system/detail/generic/select_system.h>
#include <thrust/system/detail/generic/memory.h>
#include <thrust/system/detail/adl/get_value.h>
#include <thrust/system/detail/adl/assign_value.h>
#include <thrust/system/detail/adl/iter_swap.h>
#include <thrust/type_traits/remove_cvref.h>
#include <type_traits>
#include <ostream>

THRUST_NAMESPACE_BEGIN

namespace detail
{
template <typename>
struct is_wrapped_reference;
}

/*! \p reference acts as a reference-like wrapper for an object residing in
 *  memory that a \p pointer refers to.
 */
template <typename Element, typename Pointer, typename Derived>
class reference
{
private:
  using derived_type = typename std::conditional<
    std::is_same<Derived, use_default>::value, reference, Derived
  >::type;

public:
  using pointer    = Pointer;
  using value_type = typename thrust::remove_cvref<Element>::type;

  reference(reference const&) = default;

  reference(reference&&) = default;

  /*! Construct a \p reference from another \p reference whose pointer type is
   *  convertible to \p pointer. After this \p reference is constructed, it
   *  shall refer to the same object as \p other.
   *
   *  \tparam OtherElement The element type of the other \p reference.
   *  \tparam OtherPointer The pointer type of the other \p reference.
   *  \tparam OtherDerived The derived type of the other \p reference.
   *  \param  other        A \p reference to copy from.
   */
  template <typename OtherElement, typename OtherPointer, typename OtherDerived>
  __host__ __device__
  reference(
    reference<OtherElement, OtherPointer, OtherDerived> const& other
  /*! \cond
   */
  , typename std::enable_if<
      std::is_convertible<
        typename reference<OtherElement, OtherPointer, OtherDerived>::pointer
      , pointer
      >::value
    >::type* = nullptr
  /*! \endcond
   */
  )
    : ptr(other.ptr)
  {}

  /*! Construct a \p reference that refers to an object pointed to by the given
   *  \p pointer. After this \p reference is constructed, it shall refer to the
   *  object pointed to by \p ptr.
   *
   *  \param ptr A \p pointer to construct from.
   */
  __host__ __device__
  explicit reference(pointer const& p) : ptr(p) {}

  /*! Assign the object referred to \p other to the object referred to by
   *  this \p reference.
   *
   *  \param other The other \p reference to assign from.
   *
   *  \return <tt>*this</tt>.
   */
  __host__ __device__
  derived_type& operator=(reference const& other)
  {
    assign_from(&other);
    return derived();
  }

  /*! Assign the object referred to by this \p reference with the object
   *  referred to by another \p reference whose pointer type is convertible to
   *  \p pointer.
   *
   *  \tparam OtherElement The element type of the other \p reference.
   *  \tparam OtherPointer The pointer type of the other \p reference.
   *  \tparam OtherDerived The derived type of the other \p reference.
   *  \param  other        The other \p reference to assign from.
   *
   *  \return <tt>*this</tt>.
   */
  template <typename OtherElement, typename OtherPointer, typename OtherDerived>
  __host__ __device__
  /*! \cond
   */
  typename std::enable_if<
    std::is_convertible<
      typename reference<OtherElement, OtherPointer, OtherDerived>::pointer
    , pointer
    >::value,
  /*! \endcond
   */
    derived_type&
  /*! \cond
   */
  >::type
  /*! \endcond
   */
  operator=(reference<OtherElement, OtherPointer, OtherDerived> const& other)
  {
    assign_from(&other);
    return derived();
  }

  /*! Assign \p rhs to the object referred to by this \p tagged_reference.
   *
   *  \param rhs The \p value_type to assign from.
   *
   *  \return <tt>*this</tt>.
   */
  __host__ __device__
  derived_type& operator=(value_type const& rhs)
  {
    assign_from(&rhs);
    return derived();
  }

  /*! Exchanges the value of the object referred to by this \p tagged_reference
   *  with the object referred to by \p other.
   *
   *  \param other The \p tagged_reference to swap with.
   */
  __host__ __device__
  void swap(derived_type& other)
  {
    // Avoid default-constructing a system; instead, just use a null pointer
    // for dispatch. This assumes that `get_value` will not access any system
    // state.
    typename thrust::iterator_system<pointer>::type* system = nullptr;
    swap(system, other);
  }

  __host__ __device__ pointer operator&() const { return ptr; }

  // This is inherently hazardous, as it discards the strong type information
  // about what system the object is on.
  __host__ __device__ operator value_type() const
  {
    // Avoid default-constructing a system; instead, just use a null pointer
    // for dispatch. This assumes that `get_value` will not access any system
    // state.
    typename thrust::iterator_system<pointer>::type* system = nullptr;
    return convert_to_value_type(system);
  }

  __host__ __device__
  derived_type& operator++()
  {
    // Sadly, this has to make a copy. The only mechanism we have for
    // modifying the value, which may be in memory inaccessible to this
    // system, is to get a copy of it, modify the copy, and then update it.
    value_type tmp = *this;
    ++tmp;
    *this = tmp;
    return derived();
  }

  __host__ __device__
  value_type operator++(int)
  {
    value_type tmp = *this;
    value_type result = tmp++;
    *this = std::move(tmp);
    return result;
  }

  derived_type& operator--()
  {
    // Sadly, this has to make a copy. The only mechanism we have for
    // modifying the value, which may be in memory inaccessible to this
    // system, is to get a copy of it, modify the copy, and then update it.
    value_type tmp = *this;
    --tmp;
    *this = std::move(tmp);
    return derived();
  }

  value_type operator--(int)
  {
    value_type tmp = *this;
    value_type result = tmp--;
    *this = std::move(tmp);
    return derived();
  }

  __host__ __device__
  derived_type& operator+=(value_type const& rhs)
  {
    value_type tmp = *this;
    tmp += rhs;
    *this = tmp;
    return derived();
  }

  derived_type& operator-=(value_type const& rhs)
  {
    value_type tmp = *this;
    tmp -= rhs;
    *this = tmp;
    return derived();
  }

  derived_type& operator*=(value_type const& rhs)
  {
    value_type tmp = *this;
    tmp *= rhs;
    *this = tmp;
    return derived();
  }

  derived_type& operator/=(value_type const& rhs)
  {
    value_type tmp = *this;
    tmp /= rhs;
    *this = tmp;
    return derived();
  }

  derived_type& operator%=(value_type const& rhs)
  {
    value_type tmp = *this;
    tmp %= rhs;
    *this = tmp;
    return derived();
  }

  derived_type& operator<<=(value_type const& rhs)
  {
    value_type tmp = *this;
    tmp <<= rhs;
    *this = tmp;
    return derived();
  }

  derived_type& operator>>=(value_type const& rhs)
  {
    value_type tmp = *this;
    tmp >>= rhs;
    *this = tmp;
    return derived();
  }

  derived_type& operator&=(value_type const& rhs)
  {
    value_type tmp = *this;
    tmp &= rhs;
    *this = tmp;
    return derived();
  }

  derived_type& operator|=(value_type const& rhs)
  {
    value_type tmp = *this;
    tmp |= rhs;
    *this = tmp;
    return derived();
  }

  derived_type& operator^=(value_type const& rhs)
  {
    value_type tmp = *this;
    tmp ^= rhs;
    *this = tmp;
    return derived();
  }

private:
  pointer const ptr;

  // `thrust::detail::is_wrapped_reference` is a trait that indicates whether
  // a type is a fancy reference. It detects such types by loooking for a
  // nested `wrapped_reference_hint` type.
  struct wrapped_reference_hint {};
  template <typename>
  friend struct thrust::detail::is_wrapped_reference;

  template <typename OtherElement, typename OtherPointer, typename OtherDerived>
  friend class reference;

  __host__ __device__
  derived_type& derived() { return static_cast<derived_type&>(*this); }

  template<typename System>
  __host__ __device__
  value_type convert_to_value_type(System* system) const
  {
    using thrust::system::detail::generic::select_system;
    return strip_const_get_value(select_system(*system));
  }

  template <typename System>
  __host__ __device__
  value_type strip_const_get_value(System const& system) const
  {
    System &non_const_system = const_cast<System&>(system);

    using thrust::system::detail::generic::get_value;
    return get_value(thrust::detail::derived_cast(non_const_system), ptr);
  }

  template <typename System0, typename System1, typename OtherPointer>
  __host__ __device__
  void assign_from(System0* system0, System1* system1, OtherPointer src)
  {
    using thrust::system::detail::generic::select_system;
    strip_const_assign_value(select_system(*system0, *system1), src);
  }

  template <typename OtherPointer>
  __host__ __device__
  void assign_from(OtherPointer src)
  {
    // Avoid default-constructing systems; instead, just use a null pointer
    // for dispatch. This assumes that `get_value` will not access any system
    // state.
    typename thrust::iterator_system<pointer>::type*      system0 = nullptr;
    typename thrust::iterator_system<OtherPointer>::type* system1 = nullptr;
    assign_from(system0, system1, src);
  }

  template <typename System, typename OtherPointer>
  __host__ __device__
  void strip_const_assign_value(System const& system, OtherPointer src)
  {
    System& non_const_system = const_cast<System&>(system);

    using thrust::system::detail::generic::assign_value;
    assign_value(thrust::detail::derived_cast(non_const_system), ptr, src);
  }

  template <typename System>
  __host__ __device__
  void swap(System* system, derived_type& other)
  {
    using thrust::system::detail::generic::select_system;
    using thrust::system::detail::generic::iter_swap;

    iter_swap(select_system(*system, *system), ptr, other.ptr);
  }
};

template <typename Pointer, typename Derived>
class reference<void, Pointer, Derived> {};

template <typename Pointer, typename Derived>
class reference<void const, Pointer, Derived> {};

template <
  typename Element, typename Pointer, typename Derived
, typename CharT, typename Traits
>
std::basic_ostream<CharT, Traits>& operator<<(
  std::basic_ostream<CharT, Traits>&os
, reference<Element, Pointer, Derived> const& r
) {
  using value_type = typename reference<Element, Pointer, Derived>::value_type;
  return os << static_cast<value_type>(r);
}

template <typename Element, typename Tag>
class tagged_reference;

/*! \p tagged_reference acts as a reference-like wrapper for an object residing
 *  in memory associated with system \p Tag that a \p pointer refers to.
 */
template <typename Element, typename Tag>
class tagged_reference
  : public thrust::reference<
      Element
    , thrust::pointer<Element, Tag, tagged_reference<Element, Tag>>
    , tagged_reference<Element, Tag>
    >
{
private:
  using base_type = thrust::reference<
    Element
  , thrust::pointer<Element, Tag, tagged_reference<Element, Tag>>
  , tagged_reference<Element, Tag>
  >;

public:
  using value_type = typename base_type::value_type;
  using pointer    = typename base_type::pointer;

  tagged_reference(tagged_reference const&) = default;

  tagged_reference(tagged_reference&&) = default;

  /*! Construct a \p tagged_reference from another \p tagged_reference whose
   *  pointer type is convertible to \p pointer. After this \p tagged_reference
   *  is constructed, it shall refer to the same object as \p other.
   *
   *  \tparam OtherElement The element type of the other \p tagged_reference.
   *  \tparam OtherTag     The tag type of the other \p tagged_reference.
   *  \param  other        A \p tagged_reference to copy from.
   */
  template <typename OtherElement, typename OtherTag>
  __host__ __device__
  tagged_reference(tagged_reference<OtherElement, OtherTag> const& other)
    : base_type(other)
  {}

  /*! Construct a \p tagged_reference that refers to an object pointed to by
   *  the given \p pointer. After this \p tagged_reference is constructed, it
   *  shall refer to the object pointed to by \p ptr.
   *
   *  \param ptr A \p pointer to construct from.
   */
  __host__ __device__ explicit tagged_reference(pointer const& p)
    : base_type(p)
  {}

  /*! Assign the object referred to \p other to the object referred to by
   *  this \p tagged_reference.
   *
   *  \param other The other \p tagged_reference to assign from.
   *
   *  \return <tt>*this</tt>.
   */
  __host__ __device__
  tagged_reference& operator=(tagged_reference const& other)
  {
    return base_type::operator=(other);
  }

  /*! Assign the object referred to by this \p tagged_reference with the object
   *  referred to by another \p tagged_reference whose pointer type is
   *  convertible to \p pointer.
   *
   *  \tparam OtherElement The element type of the other \p tagged_reference.
   *  \tparam OtherTag     The tag type of the other \p tagged_reference.
   *  \param  other        The other \p tagged_reference to assign from.
   *
   *  \return <tt>*this</tt>.
   */
  template <typename OtherElement, typename OtherTag>
  __host__ __device__
  tagged_reference&
  operator=(tagged_reference<OtherElement, OtherTag> const& other)
  {
    return base_type::operator=(other);
  }

  /*! Assign \p rhs to the object referred to by this \p tagged_reference.
   *
   *  \param rhs The \p value_type to assign from.
   *
   *  \return <tt>*this</tt>.
   */
  __host__ __device__
  tagged_reference& operator=(value_type const& rhs)
  {
    return base_type::operator=(rhs);
  }
};

template <typename Tag>
class tagged_reference<void, Tag> {};

template <typename Tag>
class tagged_reference<void const, Tag> {};

/*! Exchanges the values of two objects referred to by \p tagged_reference.
 *
 *  \param x The first \p tagged_reference of interest.
 *  \param y The second \p tagged_reference of interest.
 */
template <typename Element, typename Tag>
__host__ __device__
void swap(tagged_reference<Element, Tag>& x, tagged_reference<Element, Tag>& y)
{
  x.swap(y);
}

THRUST_NAMESPACE_END

