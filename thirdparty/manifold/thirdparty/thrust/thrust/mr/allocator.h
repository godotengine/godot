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
 *  \brief Allocator types usable with \ref Memory Resources.
 */

#pragma once

#include <limits>

#include <thrust/detail/config.h>
#include <thrust/detail/config/exec_check_disable.h>
#include <thrust/detail/config/memory_resource.h>
#include <thrust/detail/type_traits/pointer_traits.h>

#include <thrust/mr/validator.h>
#include <thrust/mr/polymorphic_adaptor.h>

THRUST_NAMESPACE_BEGIN
namespace mr
{

/*! \addtogroup allocators Allocators
 *  \ingroup memory_management
 *  \{
 */

/*! An \p mr::allocator is a template that fulfills the C++ requirements for Allocators,
 *  allowing to use the NPA-based memory resources where an Allocator is required. Unlike
 *  memory resources, but like other allocators, \p mr::allocator is typed and bound to
 *  allocate object of a specific type, however it can be freely rebound to other types.
 *
 *  \tparam T the type that will be allocated by this allocator.
 *  \tparam MR the upstream memory resource to use for memory allocation. Must derive from
 *      \p thrust::mr::memory_resource and must be \p final (in C++11 and beyond).
 */
template<typename T, class MR>
class allocator : private validator<MR>
{
public:
    /*! The pointer to void type of this allocator. */
    typedef typename MR::pointer void_pointer;

    /*! The value type allocated by this allocator. Equivalent to \p T. */
    typedef T value_type;
    /*! The pointer type allocated by this allocator. Equivaled to the pointer type of \p MR rebound to \p T. */
    typedef typename thrust::detail::pointer_traits<void_pointer>::template rebind<T>::other pointer;
    /*! The pointer to const type. Equivalent to a pointer type of \p MR rebound to <tt>const T</tt>. */
    typedef typename thrust::detail::pointer_traits<void_pointer>::template rebind<const T>::other const_pointer;
    /*! The reference to the type allocated by this allocator. Supports smart references. */
    typedef typename thrust::detail::pointer_traits<pointer>::reference reference;
    /*! The const reference to the type allocated by this allocator. Supports smart references. */
    typedef typename thrust::detail::pointer_traits<const_pointer>::reference const_reference;
    /*! The size type of this allocator. Always \p std::size_t. */
    typedef std::size_t size_type;
    /*! The difference type between pointers allocated by this allocator. */
    typedef typename thrust::detail::pointer_traits<pointer>::difference_type difference_type;

    /*! Specifies that the allocator shall be propagated on container copy assignment. */
    typedef detail::true_type propagate_on_container_copy_assignment;
    /*! Specifies that the allocator shall be propagated on container move assignment. */
    typedef detail::true_type propagate_on_container_move_assignment;
    /*! Specifies that the allocator shall be propagated on container swap. */
    typedef detail::true_type propagate_on_container_swap;

    /*! The \p rebind metafunction provides the type of an \p allocator instantiated with another type.
     *
     *  \tparam U the other type to use for instantiation.
     */
    template<typename U>
    struct rebind
    {
        /*! The typedef \p other gives the type of the rebound \p allocator.
         */
        typedef allocator<U, MR> other;
    };

    /*! Calculates the maximum number of elements allocated by this allocator.
     *
     *  \return the maximum value of \p std::size_t, divided by the size of \p T.
     */
    __thrust_exec_check_disable__
    __host__ __device__
    size_type max_size() const
    {
        return (std::numeric_limits<size_type>::max)() / sizeof(T);
    }

    /*! Constructor.
     *
     *  \param resource the resource to be used to allocate raw memory.
     */
    __host__ __device__
    allocator(MR * resource) : mem_res(resource)
    {
    }

    /*! Copy constructor. Copies the resource pointer. */
    template<typename U>
    __host__ __device__
    allocator(const allocator<U, MR> & other) : mem_res(other.resource())
    {
    }

    /*! Allocates objects of type \p T.
     *
     *  \param n number of elements to allocate
     *  \return a pointer to the newly allocated storage.
     */
    THRUST_NODISCARD
    __host__
    pointer allocate(size_type n)
    {
        return static_cast<pointer>(mem_res->do_allocate(n * sizeof(T), THRUST_ALIGNOF(T)));
    }

    /*! Deallocates objects of type \p T.
     *
     *  \param p pointer returned by a previous call to \p allocate
     *  \param n number of elements, passed as an argument to the \p allocate call that produced \p p
     */
    __host__
    void deallocate(pointer p, size_type n)
    {
        return mem_res->do_deallocate(p, n * sizeof(T), THRUST_ALIGNOF(T));
    }

    /*! Extracts the memory resource used by this allocator.
     *
     *  \return the memory resource used by this allocator.
     */
    __host__ __device__
    MR * resource() const
    {
        return mem_res;
    }

private:
    MR * mem_res;
};

/*! Compares the allocators for equality by comparing the underlying memory resources. */
template<typename T, typename MR>
__host__ __device__
bool operator==(const allocator<T, MR> & lhs, const allocator<T, MR> & rhs) noexcept
{
    return *lhs.resource() == *rhs.resource();
}

/*! Compares the allocators for inequality by comparing the underlying memory resources. */
template<typename T, typename MR>
__host__ __device__
bool operator!=(const allocator<T, MR> & lhs, const allocator<T, MR> & rhs) noexcept
{
    return !(lhs == rhs);
}

#if THRUST_CPP_DIALECT >= 2011

template<typename T, typename Pointer>
using polymorphic_allocator = allocator<T, polymorphic_adaptor_resource<Pointer> >;

#else // C++11

template<typename T, typename Pointer>
class polymorphic_allocator : public allocator<T, polymorphic_adaptor_resource<Pointer> >
{
    typedef allocator<T, polymorphic_adaptor_resource<Pointer> > base;

public:
    /*! Initializes the base class with the parameter \p resource.
     */
    polymorphic_allocator(polymorphic_adaptor_resource<Pointer>  * resource) : base(resource)
    {
    }
};

#endif // C++11

/*! A helper allocator class that uses global instances of a given upstream memory resource. Requires the memory resource
 *      to be default constructible.
 *
 *  \tparam T the type that will be allocated by this allocator.
 *  \tparam Upstream the upstream memory resource to use for memory allocation. Must derive from
 *      \p thrust::mr::memory_resource and must be \p final (in C++11 and beyond).
 */
template<typename T, typename Upstream>
class stateless_resource_allocator : public thrust::mr::allocator<T, Upstream>
{
    typedef thrust::mr::allocator<T, Upstream> base;

public:
    /*! The \p rebind metafunction provides the type of an \p stateless_resource_allocator instantiated with another type.
     *
     *  \tparam U the other type to use for instantiation.
     */
    template<typename U>
    struct rebind
    {
        /*! The typedef \p other gives the type of the rebound \p stateless_resource_allocator.
         */
        typedef stateless_resource_allocator<U, Upstream> other;
    };

    /*! Default constructor. Uses \p get_global_resource to get the global instance of \p Upstream and initializes the
     *      \p allocator base subobject with that resource.
     */
    __thrust_exec_check_disable__
    __host__ __device__
    stateless_resource_allocator() : base(get_global_resource<Upstream>())
    {
    }

    /*! Copy constructor. Copies the memory resource pointer. */
    __host__ __device__
    stateless_resource_allocator(const stateless_resource_allocator & other)
        : base(other) {}

    /*! Conversion constructor from an allocator of a different type. Copies the memory resource pointer. */
    template<typename U>
    __host__ __device__
    stateless_resource_allocator(const stateless_resource_allocator<U, Upstream> & other)
        : base(other) {}

#if THRUST_CPP_DIALECT >= 2011
    stateless_resource_allocator & operator=(const stateless_resource_allocator &) = default;
#endif

    /*! Destructor. */
    __host__ __device__
    ~stateless_resource_allocator() {}
};

/*! \} // allocators
 */

} // end mr
THRUST_NAMESPACE_END

