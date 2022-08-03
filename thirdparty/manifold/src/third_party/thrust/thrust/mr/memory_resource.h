/*
 *  Copyright 2018 NVIDIA Corporation
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
 *  \brief A base class for the memory resource system, similar to
 *  std::memory_resource, and related utilities.
 */

#pragma once

#include <thrust/detail/config.h>
#include <thrust/detail/config/memory_resource.h>
#ifdef THRUST_MR_STD_MR_HEADER
#  include THRUST_MR_STD_MR_HEADER
#endif

THRUST_NAMESPACE_BEGIN
/*! \brief \p thrust::mr is the namespace containing system agnostic types and functions for \p memory_resource related functionalities.
 */
namespace mr
{

/** \addtogroup memory_resources Memory Resources
 *  \ingroup memory_management
 *  \{
 */

/*! \p memory_resource is the base class for all other memory resources.
 *
 *  \tparam Pointer the pointer type that is allocated and deallocated by the memory resource
 *      derived from this base class. If this is <tt>void *</tt>, this class derives from
 *      <tt>std::pmr::memory_resource</tt>.
 */
template<typename Pointer = void *>
class memory_resource
{
public:
    /*! Alias for the template parameter.
     */
    typedef Pointer pointer;

    /*! Virtual destructor, defaulted when possible.
     */
    virtual ~memory_resource() = default;

    /*! Allocates memory of size at least \p bytes and alignment at least \p alignment.
     *
     *  \param bytes size, in bytes, that is requested from this allocation
     *  \param alignment alignment that is requested from this allocation
     *  \throws thrust::bad_alloc when no memory with requested size and alignment can be allocated.
     *  \return A pointer to void to the newly allocated memory.
     */
    THRUST_NODISCARD
    pointer allocate(std::size_t bytes, std::size_t alignment = THRUST_MR_DEFAULT_ALIGNMENT)
    {
        return do_allocate(bytes, alignment);
    }

    /*! Deallocates memory pointed to by \p p.
     *
     *  \param p pointer to be deallocated
     *  \param bytes the size of the allocation. This must be equivalent to the value of \p bytes that
     *      was passed to the allocation function that returned \p p.
     *  \param alignment the alignment of the allocation. This must be equivalent to the value of \p alignment
     *      that was passed to the allocation function that returned \p p.
     */
    void deallocate(pointer p, std::size_t bytes, std::size_t alignment = THRUST_MR_DEFAULT_ALIGNMENT)
    {
        do_deallocate(p, bytes, alignment);
    }

    /*! Compares this resource to the other one. The default implementation uses identity comparison,
     *      which is often the right thing to do and doesn't require RTTI involvement.
     *
     *  \param other the other resource to compare this resource to
     *  \return whether the two resources are equivalent.
     */
    __host__ __device__
    bool is_equal(const memory_resource & other) const noexcept
    {
        return do_is_equal(other);
    }

    /*! Allocates memory of size at least \p bytes and alignment at least \p alignment.
     *
     *  \param bytes size, in bytes, that is requested from this allocation
     *  \param alignment alignment that is requested from this allocation
     *  \throws thrust::bad_alloc when no memory with requested size and alignment can be allocated.
     *  \return A pointer to void to the newly allocated memory.
     */
    virtual pointer do_allocate(std::size_t bytes, std::size_t alignment) = 0;

    /*! Deallocates memory pointed to by \p p.
     *
     *  \param p pointer to be deallocated
     *  \param bytes the size of the allocation. This must be equivalent to the value of \p bytes that
     *      was passed to the allocation function that returned \p p.
     *  \param alignment the size of the allocation. This must be equivalent to the value of \p alignment
     *      that was passed to the allocation function that returned \p p.
     */
    virtual void do_deallocate(pointer p, std::size_t bytes, std::size_t alignment) = 0;

    /*! Compares this resource to the other one. The default implementation uses identity comparison,
     *      which is often the right thing to do and doesn't require RTTI involvement.
     *
     *  \param other the other resource to compare this resource to
     *  \return whether the two resources are equivalent.
     */
    __host__ __device__
    virtual bool do_is_equal(const memory_resource & other) const noexcept
    {
        return this == &other;
    }
};

template<>
class memory_resource<void *>
#ifdef THRUST_STD_MR_NS
    : THRUST_STD_MR_NS::memory_resource
#endif
{
public:
    typedef void * pointer;

    virtual ~memory_resource() = default;

    THRUST_NODISCARD
    pointer allocate(std::size_t bytes, std::size_t alignment = THRUST_MR_DEFAULT_ALIGNMENT)
    {
        return do_allocate(bytes, alignment);
    }

    void deallocate(pointer p, std::size_t bytes, std::size_t alignment = THRUST_MR_DEFAULT_ALIGNMENT)
    {
        do_deallocate(p, bytes, alignment);
    }

    __host__ __device__
    bool is_equal(const memory_resource & other) const noexcept
    {
        return do_is_equal(other);
    }

    virtual pointer do_allocate(std::size_t bytes, std::size_t alignment) = 0;
    virtual void do_deallocate(pointer p, std::size_t bytes, std::size_t alignment) = 0;
    __host__ __device__
    virtual bool do_is_equal(const memory_resource & other) const noexcept
    {
        return this == &other;
    }

#ifdef THRUST_STD_MR_NS
    // the above do_is_equal is a different function than the one from the standard memory resource
    // can't implement this reasonably without RTTI though; it's reasonable to assume false otherwise

    virtual bool do_is_equal(const THRUST_STD_MR_NS::memory_resource & other) const noexcept override
    {
#  ifdef THRUST_HAS_DYNAMIC_CAST
        auto mr_resource = dynamic_cast<memory_resource<> *>(&other);
        return mr_resource && do_is_equal(*mr_resource);
#  else
        return this == &other;
#  endif
    }
#endif
};

/*! Compares the memory resources for equality, first by identity, then by \p is_equal.
 */
template<typename Pointer>
__host__ __device__
bool operator==(const memory_resource<Pointer> & lhs, const memory_resource<Pointer> & rhs) noexcept
{
    return &lhs == &rhs || rhs.is_equal(rhs);
}

/*! Compares the memory resources for inequality, first by identity, then by \p is_equal.
 */
template<typename Pointer>
__host__ __device__
bool operator!=(const memory_resource<Pointer> & lhs, const memory_resource<Pointer> & rhs) noexcept
{
    return !(lhs == rhs);
}

/*! Returns a global instance of \p MR, created as a function local static variable.
 *
 *  \tparam MR type of a memory resource to get an instance from. Must be \p DefaultConstructible.
 *  \return a pointer to a global instance of \p MR.
 */
template<typename MR>
__host__
MR * get_global_resource()
{
    static MR resource;
    return &resource;
}

/*! \} // memory_resource
 */

} // end mr
THRUST_NAMESPACE_END

