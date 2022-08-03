/*
 *  Copyright 2008-2021 NVIDIA Corporation
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
 *  \brief A pointer to an object which resides in memory associated with the
 *  \c device system.
 */

#pragma once

#include <thrust/detail/config.h>
#include <thrust/memory.h>

THRUST_NAMESPACE_BEGIN

/*! \addtogroup memory_management Memory Management
 *  \{
 */

template <typename T> class device_reference;

/*! \brief \c device_ptr is a pointer-like object which points to an object that
 *  resides in memory associated with the \ref device system.
 *
 *  \c device_ptr has pointer semantics: it may be dereferenced safely from
 *  anywhere, including the \ref host, and may be manipulated with pointer
 *  arithmetic.
 *
 *  \c device_ptr can be created with \ref device_new, \ref device_malloc,
 *  \ref device_malloc_allocator, \ref device_allocator, or
 *  \ref device_pointer_cast, or by explicitly calling its constructor with a
 *  raw pointer.
 *
 *  The raw pointer contained in a \c device_ptr may be obtained via \c get
 *  member function or the \ref raw_pointer_cast free function.
 *
 *  \ref algorithms operating on \c device_ptr types will automatically be
 *  dispatched to the \ref device system.
 *
 *  \note \c device_ptr is not a smart pointer; it is the programmer's
 *  responsibility to deallocate memory pointed to by \c device_ptr.
 *
 *  \see device_new
 *  \see device_malloc
 *  \see device_malloc_allocator
 *  \see device_allocator
 *  \see device_pointer_cast
 *  \see raw_pointer_cast
 */
template <typename T>
class device_ptr
  : public thrust::pointer<
      T,
      thrust::device_system_tag,
      thrust::device_reference<T>,
      thrust::device_ptr<T>
    >
{
  private:
    using super_t = thrust::pointer<
      T,
      thrust::device_system_tag,
      thrust::device_reference<T>,
      thrust::device_ptr<T>
    >;

  public:
    /*! \brief Construct a null \c device_ptr.
     *
     *  \post <tt>get() == nullptr</tt>.
     */
    __host__ __device__
    device_ptr() : super_t() {}

    /*! \brief Construct a null \c device_ptr.
     *
     *  \param ptr A null pointer.
     *
     *  \post <tt>get() == nullptr</tt>.
     */
    __host__ __device__
    device_ptr(std::nullptr_t) : super_t(nullptr) {}

    /*! \brief Construct a \c device_ptr from a raw pointer which is
     *  convertible to \c T*.
     *
     *  \tparam U   A type whose pointer is convertible to \c T*.
     *  \param  ptr A raw pointer to a \c U in device memory to construct from.
     *
     *  \pre <tt>std::is_convertible_v<U*, T*> == true</tt>.
     *
     *  \pre \c ptr points to a location in device memory.
     *
     *  \post <tt>get() == nullptr</tt>.
     */
    template <typename U>
    __host__ __device__
    explicit device_ptr(U* ptr) : super_t(ptr) {}

    /*! \brief Copy construct a \c device_ptr from another \c device_ptr whose
     *  pointer type is convertible to \c T*.
     *
     *  \tparam U     A type whose pointer is convertible to \c T*.
     *  \param  other A \c device_ptr to a \c U to construct from.
     *
     *  \pre <tt>std::is_convertible_v<U*, T*> == true</tt>.
     *
     *  \post <tt>get() == other.get()</tt>.
     */
    template <typename U>
    __host__ __device__
    device_ptr(device_ptr<U> const& other) : super_t(other) {}

    /*! \brief Set this \c device_ptr to point to the same object as another
     *  \c device_ptr whose pointer type is convertible to \c T*.
     *
     *  \tparam U     A type whose pointer is convertible to \c T*.
     *  \param  other A \c device_ptr to a \c U to assign from.
     *
     *  \pre <tt>std::is_convertible_v<U*, T*> == true</tt>.
     *
     *  \post <tt>get() == other.get()</tt>.
     *
     *  \return \c *this.
     */
    template <typename U>
    __host__ __device__
    device_ptr &operator=(device_ptr<U> const& other)
    {
      super_t::operator=(other);
      return *this;
    }

    /*! \brief Set this \c device_ptr to null.
     *
     *  \param ptr A null pointer.
     *
     *  \post <tt>get() == nullptr</tt>.
     *
     *  \return \c *this.
     */
    __host__ __device__
    device_ptr& operator=(std::nullptr_t)
    {
      super_t::operator=(nullptr);
      return *this;
    }

#if THRUST_DOXYGEN
    /*! \brief Return the raw pointer that this \c device_ptr points to.
     */
    __host__ __device__
    T* get() const;
#endif
};

#if THRUST_DOXYGEN
/*! Write the address that a \c device_ptr points to to an output stream.
 *
 *  \param os The output stream.
 *  \param dp The \c device_ptr to output.
 *
 *  \return \c os.
 */
template <typename T, typename CharT, typename Traits>
__host__ std::basic_ostream<CharT, Traits>&
operator<<(std::basic_ostream<CharT, Traits>& os, device_ptr<T> const& dp);
#endif

/*! \brief Create a \c device_ptr from a raw pointer.
 *
 *  \tparam T   Any type.
 *  \param  ptr A raw pointer to a \c T in device memory.
 *
 *  \pre \c ptr points to a location in device memory.
 *
 *  \return A \c device_ptr<T> pointing to \c ptr.
 */
template <typename T>
__host__ __device__
device_ptr<T> device_pointer_cast(T* ptr);

/*! \brief Create a \c device_ptr from another \c device_ptr.
 *
 *  \tparam T    Any type.
 *  \param  dptr A \c device_ptr to a \c T.
 */
template<typename T>
__host__ __device__
device_ptr<T> device_pointer_cast(device_ptr<T> const& dptr);

/*! \} // memory_management
 */

THRUST_NAMESPACE_END

#include <thrust/detail/device_ptr.inl>
#include <thrust/detail/raw_pointer_cast.h>
