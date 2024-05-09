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

/*! \file thrust/memory.h
 *  \brief Abstractions for Thrust's memory model.
 */

#pragma once

#include <thrust/detail/config.h>
#include <thrust/detail/type_traits/pointer_traits.h>
#include <thrust/detail/pointer.h>
#include <thrust/detail/reference.h>
#include <thrust/detail/raw_pointer_cast.h>
#include <thrust/detail/raw_reference_cast.h>
#include <thrust/detail/malloc_and_free.h>
#include <thrust/detail/temporary_buffer.h>

THRUST_NAMESPACE_BEGIN

/*! \defgroup memory_management Memory Management
 *
 *  All Thrust functionalities related to memory allocation and deallocation.
 *
 */

/** \addtogroup memory_management Memory Management
 *  \{
 */

// define pointer for the purpose of Doxygenating it
// it is actually defined elsewhere
#if 0
/*! \p pointer stores a pointer to an object allocated in memory. Like \p device_ptr, this
 *  type ensures type safety when dispatching standard algorithms on ranges resident in memory.
 *
 *  \p pointer generalizes \p device_ptr by relaxing the backend system associated with the \p pointer.
 *  Instead of the backend system specified by \p THRUST_DEVICE_SYSTEM, \p pointer's
 *  system is given by its second template parameter, \p Tag. For the purpose of Thrust dispatch,
 *  <tt>device_ptr<Element></tt> and <tt>pointer<Element,device_system_tag></tt> are considered equivalent.
 *
 *  The raw pointer encapsulated by a \p pointer may be obtained through its <tt>get</tt> member function
 *  or the \p raw_pointer_cast free function.
 *
 *  \tparam Element specifies the type of the pointed-to object.
 *
 *  \tparam Tag specifies the system with which this \p pointer is associated. This may be any Thrust
 *          backend system, or a user-defined tag.
 *
 *  \tparam Reference allows the client to specify the reference type returned upon derereference.
 *          By default, this type is <tt>reference<Element,pointer></tt>.
 *
 *  \tparam Derived allows the client to specify the name of the derived type when \p pointer is used as
 *          a base class. This is useful to ensure that arithmetic on values of the derived type return
 *          values of the derived type as a result. By default, this type is <tt>pointer<Element,Tag,Reference></tt>.
 *
 *  \note \p pointer is not a smart pointer; it is the client's responsibility to deallocate memory
 *        pointer to by \p pointer.
 *
 *  \see device_ptr
 *  \see reference
 *  \see raw_pointer_cast
 */
template<typename Element, typename Tag, typename Reference = thrust::use_default, typename Derived = thrust::use_default>
  class pointer
{
  public:
    /*! The type of the raw pointer
     */
    typedef typename super_t::base_type raw_pointer;

    /*! \p pointer's default constructor initializes its encapsulated pointer to \c 0
     */
    __host__ __device__
    pointer();

    /*! This constructor allows construction of a <tt>pointer<const T, ...></tt> from a <tt>T*</tt>.
     *
     *  \param ptr A raw pointer to copy from, presumed to point to a location in \p Tag's memory.
     *  \tparam OtherElement \p OtherElement shall be convertible to \p Element.
     */
    template<typename OtherElement>
    __host__ __device__
    explicit pointer(OtherElement *ptr);

    /*! This contructor allows initialization from another pointer-like object.
     *
     *  \param other The \p OtherPointer to copy.
     *
     *  \tparam OtherPointer The tag associated with \p OtherPointer shall be convertible to \p Tag,
     *                       and its element type shall be convertible to \p Element.
     */
    template<typename OtherPointer>
    __host__ __device__
    pointer(const OtherPointer &other,
            typename thrust::detail::enable_if_pointer_is_convertible<
              OtherPointer,
              pointer<Element,Tag,Reference,Derived>
            >::type * = 0);

    /*! Assignment operator allows assigning from another pointer-like object whose element type
     *  is convertible to \c Element.
     *
     *  \param other The other pointer-like object to assign from.
     *  \return <tt>*this</tt>
     *
     *  \tparam OtherPointer The tag associated with \p OtherPointer shall be convertible to \p Tag,
     *                       and its element type shall be convertible to \p Element.
     */
    template<typename OtherPointer>
    __host__ __device__
    typename thrust::detail::enable_if_pointer_is_convertible<
      OtherPointer,
      pointer,
      derived_type &
    >::type
    operator=(const OtherPointer &other);

    /*! \p get returns this \p pointer's encapsulated raw pointer.
     *  \return This \p pointer's raw pointer.
     */
    __host__ __device__
    Element *get() const;
};
#endif

/*! This version of \p malloc allocates untyped uninitialized storage associated with a given system.
 *
 *  \param system The Thrust system with which to associate the storage.
 *  \param n The number of bytes of storage to allocate.
 *  \return If allocation succeeds, a pointer to the allocated storage; a null pointer otherwise.
 *          The pointer must be deallocated with \p thrust::free.
 *
 *  \tparam DerivedPolicy The name of the derived execution policy.
 *
 *  \pre \p DerivedPolicy must be publically derived from <tt>thrust::execution_policy<DerivedPolicy></tt>.
 *
 *  The following code snippet demonstrates how to use \p malloc to allocate a range of memory
 *  associated with Thrust's device system.
 *
 *  \code
 *  #include <thrust/memory.h>
 *  ...
 *  // allocate some memory with thrust::malloc
 *  const int N = 100;
 *  thrust::device_system_tag device_sys;
 *  thrust::pointer<void,thrust::device_space_tag> void_ptr = thrust::malloc(device_sys, N);
 *
 *  // manipulate memory
 *  ...
 *
 *  // deallocate void_ptr with thrust::free
 *  thrust::free(device_sys, void_ptr);
 *  \endcode
 *
 *  \see free
 *  \see device_malloc
 */
template<typename DerivedPolicy>
__host__ __device__
pointer<void,DerivedPolicy> malloc(const thrust::detail::execution_policy_base<DerivedPolicy> &system, std::size_t n);


/*! This version of \p malloc allocates typed uninitialized storage associated with a given system.
 *
 *  \param system The Thrust system with which to associate the storage.
 *  \param n The number of elements of type \c T which the storage should accomodate.
 *  \return If allocation succeeds, a pointer to an allocation large enough to accomodate \c n
 *          elements of type \c T; a null pointer otherwise.
 *          The pointer must be deallocated with \p thrust::free.
 *
 *  \tparam DerivedPolicy The name of the derived execution policy.
 *
 *  \pre \p DerivedPolicy must be publically derived from <tt>thrust::execution_policy<DerivedPolicy></tt>.
 *
 *  The following code snippet demonstrates how to use \p malloc to allocate a range of memory
 *  to accomodate integers associated with Thrust's device system.
 *
 *  \code
 *  #include <thrust/memory.h>
 *  ...
 *  // allocate storage for 100 ints with thrust::malloc
 *  const int N = 100;
 *  thrust::device_system_tag device_sys;
 *  thrust::pointer<int,thrust::device_system_tag> ptr = thrust::malloc<int>(device_sys, N);
 *
 *  // manipulate memory
 *  ...
 *
 *  // deallocate ptr with thrust::free
 *  thrust::free(device_sys, ptr);
 *  \endcode
 *
 *  \see free
 *  \see device_malloc
 */
template<typename T, typename DerivedPolicy>
__host__ __device__
pointer<T,DerivedPolicy> malloc(const thrust::detail::execution_policy_base<DerivedPolicy> &system, std::size_t n);


/*! \p get_temporary_buffer returns a pointer to storage associated with a given Thrust system sufficient to store up to
 *  \p n objects of type \c T. If not enough storage is available to accomodate \p n objects, an implementation may return
 *  a smaller buffer. The number of objects the returned buffer can accomodate is also returned.
 *
 *  Thrust uses \p get_temporary_buffer internally when allocating temporary storage required by algorithm implementations.
 *
 *  The storage allocated with \p get_temporary_buffer must be returned to the system with \p return_temporary_buffer.
 *
 *  \param system The Thrust system with which to associate the storage.
 *  \param n The requested number of objects of type \c T the storage should accomodate.
 *  \return A pair \c p such that <tt>p.first</tt> is a pointer to the allocated storage and <tt>p.second</tt> is the number of
 *          contiguous objects of type \c T that the storage can accomodate. If no storage can be allocated, <tt>p.first</tt> if
 *          no storage can be obtained. The storage must be returned to the system using \p return_temporary_buffer.
 *
 *  \tparam DerivedPolicy The name of the derived execution policy.
 *
 *  \pre \p DerivedPolicy must be publically derived from <tt>thrust::execution_policy<DerivedPolicy></tt>.
 *
 *  The following code snippet demonstrates how to use \p get_temporary_buffer to allocate a range of memory
 *  to accomodate integers associated with Thrust's device system.
 *
 *  \code
 *  #include <thrust/memory.h>
 *  ...
 *  // allocate storage for 100 ints with thrust::get_temporary_buffer
 *  const int N = 100;
 *
 *  typedef thrust::pair<
 *    thrust::pointer<int,thrust::device_system_tag>,
 *    std::ptrdiff_t
 *  > ptr_and_size_t;
 *
 *  thrust::device_system_tag device_sys;
 *  ptr_and_size_t ptr_and_size = thrust::get_temporary_buffer<int>(device_sys, N);
 *
 *  // manipulate up to 100 ints
 *  for(int i = 0; i < ptr_and_size.second; ++i)
 *  {
 *    *ptr_and_size.first = i;
 *  }
 *
 *  // deallocate storage with thrust::return_temporary_buffer
 *  thrust::return_temporary_buffer(device_sys, ptr_and_size.first);
 *  \endcode
 *
 *  \see malloc
 *  \see return_temporary_buffer
 */
template<typename T, typename DerivedPolicy>
__host__ __device__
thrust::pair<thrust::pointer<T,DerivedPolicy>, typename thrust::pointer<T,DerivedPolicy>::difference_type>
get_temporary_buffer(const thrust::detail::execution_policy_base<DerivedPolicy> &system, typename thrust::pointer<T,DerivedPolicy>::difference_type n);

/*! \p free deallocates the storage previously allocated by \p thrust::malloc.
 *
 *  \param system The Thrust system with which the storage is associated.
 *  \param ptr A pointer previously returned by \p thrust::malloc. If \p ptr is null, \p free
 *         does nothing.
 *
 *  \tparam DerivedPolicy The name of the derived execution policy.
 *
 *  \pre \p ptr shall have been returned by a previous call to <tt>thrust::malloc(system, n)</tt> or <tt>thrust::malloc<T>(system, n)</tt> for some type \c T.
 *
 *  The following code snippet demonstrates how to use \p free to deallocate a range of memory
 *  previously allocated with \p thrust::malloc.
 *
 *  \code
 *  #include <thrust/memory.h>
 *  ...
 *  // allocate storage for 100 ints with thrust::malloc
 *  const int N = 100;
 *  thrust::device_system_tag device_sys;
 *  thrust::pointer<int,thrust::device_system_tag> ptr = thrust::malloc<int>(device_sys, N);
 *
 *  // mainpulate memory
 *  ...
 *
 *  // deallocate ptr with thrust::free
 *  thrust::free(device_sys, ptr);
 *  \endcode
 */
template<typename DerivedPolicy, typename Pointer>
__host__ __device__
void free(const thrust::detail::execution_policy_base<DerivedPolicy> &system, Pointer ptr);


/*! \p return_temporary_buffer deallocates storage associated with a given Thrust system previously allocated by \p get_temporary_buffer.
 *
 *  Thrust uses \p return_temporary_buffer internally when deallocating temporary storage required by algorithm implementations.
 *
 *  \param system The Thrust system with which the storage is associated.
 *  \param p A pointer previously returned by \p thrust::get_temporary_buffer. If \p ptr is null, \p return_temporary_buffer does nothing.
 *
 *  \tparam DerivedPolicy The name of the derived execution policy.
 *
 *  \pre \p p shall have been previously allocated by \p thrust::get_temporary_buffer.
 *
 *  The following code snippet demonstrates how to use \p return_temporary_buffer to deallocate a range of memory
 *  previously allocated by \p get_temporary_buffer.
 *
 *  \code
 *  #include <thrust/memory.h>
 *  ...
 *  // allocate storage for 100 ints with thrust::get_temporary_buffer
 *  const int N = 100;
 *
 *  typedef thrust::pair<
 *    thrust::pointer<int,thrust::device_system_tag>,
 *    std::ptrdiff_t
 *  > ptr_and_size_t;
 *
 *  thrust::device_system_tag device_sys;
 *  ptr_and_size_t ptr_and_size = thrust::get_temporary_buffer<int>(device_sys, N);
 *
 *  // manipulate up to 100 ints
 *  for(int i = 0; i < ptr_and_size.second; ++i)
 *  {
 *    *ptr_and_size.first = i;
 *  }
 *
 *  // deallocate storage with thrust::return_temporary_buffer
 *  thrust::return_temporary_buffer(device_sys, ptr_and_size.first);
 *  \endcode
 *
 *  \see free
 *  \see get_temporary_buffer
 */
template<typename DerivedPolicy, typename Pointer>
__host__ __device__
void return_temporary_buffer(const thrust::detail::execution_policy_base<DerivedPolicy> &system, Pointer p, std::ptrdiff_t n);


/*! \p raw_pointer_cast creates a "raw" pointer from a pointer-like type,
 *  simply returning the wrapped pointer, should it exist.
 *
 *  \param ptr The pointer of interest.
 *  \return <tt>ptr.get()</tt>, if the expression is well formed; <tt>ptr</tt>, otherwise.
 *  \see raw_reference_cast
 */
template<typename Pointer>
__host__ __device__
typename thrust::detail::pointer_traits<Pointer>::raw_pointer
  raw_pointer_cast(Pointer ptr);


/*! \p raw_reference_cast creates a "raw" reference from a wrapped reference type,
 *  simply returning the underlying reference, should it exist.
 *
 *  If the argument is not a reference wrapper, the result is a reference to the argument.
 *
 *  \param ref The reference of interest.
 *  \return <tt>*thrust::raw_pointer_cast(&ref)</tt>.
 *  \note There are two versions of \p raw_reference_cast. One for <tt>const</tt> references,
 *        and one for non-<tt>const</tt>.
 *  \see raw_pointer_cast
 */
template<typename T>
__host__ __device__
typename detail::raw_reference<T>::type
  raw_reference_cast(T &ref);


/*! \p raw_reference_cast creates a "raw" reference from a wrapped reference type,
 *  simply returning the underlying reference, should it exist.
 *
 *  If the argument is not a reference wrapper, the result is a reference to the argument.
 *
 *  \param ref The reference of interest.
 *  \return <tt>*thrust::raw_pointer_cast(&ref)</tt>.
 *  \note There are two versions of \p raw_reference_cast. One for <tt>const</tt> references,
 *        and one for non-<tt>const</tt>.
 *  \see raw_pointer_cast
 */
template<typename T>
__host__ __device__
typename detail::raw_reference<const T>::type
  raw_reference_cast(const T &ref);

/*! \} // memory_management
 */

THRUST_NAMESPACE_END
