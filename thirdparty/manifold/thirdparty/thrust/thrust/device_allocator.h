/*
 *  Copyright 2008-2018 NVIDIA Corporation
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
 *  \brief An allocator which creates new elements in memory accessible by
 *  devices.
 */

#pragma once

#include <thrust/detail/config.h>
#include <thrust/device_ptr.h>
#include <thrust/mr/allocator.h>
#include <thrust/mr/device_memory_resource.h>

#include <limits>
#include <stdexcept>

THRUST_NAMESPACE_BEGIN

/** \addtogroup allocators Allocators
 *  \ingroup memory_management
 *  \{
 */

/*! Memory resource adaptor that turns any memory resource that returns a fancy
 *      with the same tag as \p device_ptr, and adapts it to a resource that returns
 *      a \p device_ptr.
 */
template<typename Upstream>
class device_ptr_memory_resource final
    : public thrust::mr::memory_resource<
        device_ptr<void>
    >
{
    typedef typename Upstream::pointer upstream_ptr;

public:
    /*! Initialize the adaptor with the global instance of the upstream resource. Obtains
     *      the global instance by calling \p get_global_resource.
     */
    __host__
    device_ptr_memory_resource() : m_upstream(mr::get_global_resource<Upstream>())
    {
    }

    /*! Initialize the adaptor with an upstream resource.
     *
     *  \param upstream the upstream memory resource to adapt.
     */
    __host__
    device_ptr_memory_resource(Upstream * upstream) : m_upstream(upstream)
    {
    }

    THRUST_NODISCARD __host__
    virtual pointer do_allocate(std::size_t bytes, std::size_t alignment = THRUST_MR_DEFAULT_ALIGNMENT) override
    {
        return pointer(m_upstream->do_allocate(bytes, alignment).get());
    }

    __host__
    virtual void do_deallocate(pointer p, std::size_t bytes, std::size_t alignment) override
    {
        m_upstream->do_deallocate(upstream_ptr(p.get()), bytes, alignment);
    }

private:
    Upstream * m_upstream;
};

/*! \brief An allocator which creates new elements in memory accessible by
 *         devices.
 *
 *  \see https://en.cppreference.com/w/cpp/named_req/Allocator
 */
template<typename T>
class device_allocator
    : public thrust::mr::stateless_resource_allocator<
        T,
        device_ptr_memory_resource<device_memory_resource>
    >
{
    typedef thrust::mr::stateless_resource_allocator<
        T,
        device_ptr_memory_resource<device_memory_resource>
    > base;

public:
    /*! The \p rebind metafunction provides the type of a \p device_allocator
     *  instantiated with another type.
     *
     *  \tparam U the other type to use for instantiation.
     */
    template<typename U>
    struct rebind
    {
        /*! The typedef \p other gives the type of the rebound \p device_allocator.
         */
        typedef device_allocator<U> other;
    };

    /*! Default constructor has no effect. */
    __host__ __device__
    device_allocator() {}

    /*! Copy constructor has no effect. */
    __host__ __device__
    device_allocator(const device_allocator& other) : base(other) {}

    /*! Constructor from other \p device_allocator has no effect. */
    template<typename U>
    __host__ __device__
    device_allocator(const device_allocator<U>& other) : base(other) {}

    device_allocator & operator=(const device_allocator &) = default;

    /*! Destructor has no effect. */
    __host__ __device__
    ~device_allocator() {}
};

/*! \} // allocators
 */

THRUST_NAMESPACE_END
