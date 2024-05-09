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
 *  \brief An allocator which allocates storage with \p device_new.
 */

#pragma once

#include <thrust/detail/config.h>
#include <thrust/device_ptr.h>
#include <thrust/device_reference.h>
#include <thrust/device_new.h>
#include <thrust/device_delete.h>

#include <cuda/std/cstdint>
#include <cuda/std/limits>

#include <stdexcept>

THRUST_NAMESPACE_BEGIN

/*! \addtogroup allocators Allocators
 *  \ingroup memory_management
 *  \{
 */

/*! \p device_new_allocator is a device memory allocator that employs the
 *  \p device_new function for allocation.
 *
 *  \see device_new
 *  \see device_ptr
 *  \see https://en.cppreference.com/w/cpp/memory/allocator
 */
template<typename T>
  class device_new_allocator
{
  public:
    /*! Type of element allocated, \c T. */
    typedef T                                 value_type;

    /*! Pointer to allocation, \c device_ptr<T>. */
    typedef device_ptr<T>                     pointer;

    /*! \c const pointer to allocation, \c device_ptr<const T>. */
    typedef device_ptr<const T>               const_pointer;

    /*! Reference to allocated element, \c device_reference<T>. */
    typedef device_reference<T>               reference;

    /*! \c const reference to allocated element, \c device_reference<const T>. */
    typedef device_reference<const T>         const_reference;

    /*! Type of allocation size, \c ::cuda::std::size_t. */
    typedef ::cuda::std::size_t                 size_type;

    /*! Type of allocation difference, \c pointer::difference_type. */
    typedef typename pointer::difference_type difference_type;

    /*! The \p rebind metafunction provides the type of a \p device_new_allocator
     *  instantiated with another type.
     *
     *  \tparam U The other type to use for instantiation.
     */
    template<typename U>
      struct rebind
    {
      /*! The typedef \p other gives the type of the rebound \p device_new_allocator.
       */
      typedef device_new_allocator<U> other;
    }; // end rebind

    /*! No-argument constructor has no effect. */
    __host__ __device__
    inline device_new_allocator() {}

    /*! No-argument destructor has no effect. */
    __host__ __device__
    inline ~device_new_allocator() {}

    /*! Copy constructor has no effect. */
    __host__ __device__
    inline device_new_allocator(device_new_allocator const&) {}

    /*! Constructor from other \p device_malloc_allocator has no effect. */
    template<typename U>
    __host__ __device__
    inline device_new_allocator(device_new_allocator<U> const&) {}

    /*! Returns the address of an allocated object.
     *  \return <tt>&r</tt>.
     */
    __host__ __device__
    inline pointer address(reference r) { return &r; }
    
    /*! Returns the address an allocated object.
     *  \return <tt>&r</tt>.
     */
    __host__ __device__
    inline const_pointer address(const_reference r) { return &r; }

    /*! Allocates storage for \p cnt objects.
     *  \param cnt The number of objects to allocate.
     *  \return A \p pointer to uninitialized storage for \p cnt objects.
     *  \note Memory allocated by this function must be deallocated with \p deallocate.
     */
    __host__
    inline pointer allocate(size_type cnt,
                            const_pointer = const_pointer(static_cast<T*>(0)))
    {
      if(cnt > this->max_size())
      {
        throw std::bad_alloc();
      } // end if

      // use "::operator new" rather than keyword new
      return pointer(device_new<T>(cnt));
    } // end allocate()

    /*! Deallocates storage for objects allocated with \p allocate.
     *  \param p A \p pointer to the storage to deallocate.
     *  \param cnt The size of the previous allocation.
     *  \note Memory deallocated by this function must previously have been
     *        allocated with \p allocate.
     */
    __host__
    inline void deallocate(pointer p, size_type cnt)
    {
      // use "::operator delete" rather than keyword delete
      (void)cnt;
      device_delete(p);
    } // end deallocate()

    /*! Returns the largest value \c n for which <tt>allocate(n)</tt> might succeed.
     *  \return The largest value \c n for which <tt>allocate(n)</tt> might succeed.
     */
    __host__ __device__
    inline size_type max_size() const
    {
      return ::cuda::std::numeric_limits<size_type>::max THRUST_PREVENT_MACRO_SUBSTITUTION () / sizeof(T);
    } // end max_size()

    /*! Compares against another \p device_malloc_allocator for equality.
     *  \return \c true
     */
    __host__ __device__
    inline bool operator==(device_new_allocator const&) { return true; }

    /*! Compares against another \p device_malloc_allocator for inequality.
     *  \return \c false
     */
    __host__ __device__
    inline bool operator!=(device_new_allocator const &a) {return !operator==(a); }
}; // end device_new_allocator

/*! \} // allocators
 */

THRUST_NAMESPACE_END
