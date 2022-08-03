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
 *  \brief A mutex-synchronized version of \p unsynchronized_pool_resource.
 */

#pragma once

#include <thrust/detail/config.h>
#include <thrust/detail/cpp11_required.h>

#if THRUST_CPP_DIALECT >= 2011

#include <mutex>

#include <thrust/mr/pool.h>

THRUST_NAMESPACE_BEGIN
namespace mr
{

/*! \addtogroup memory_resources Memory Resources
 *  \ingroup memory_management
 *  \{
 */

/*! A mutex-synchronized version of \p unsynchronized_pool_resource. Uses \p std::mutex, and therefore requires C++11.
 *
 *  \tparam Upstream the type of memory resources that will be used for allocating memory
 */
template<typename Upstream>
struct synchronized_pool_resource : public memory_resource<typename Upstream::pointer>
{
    typedef unsynchronized_pool_resource<Upstream> unsync_pool;
    typedef std::lock_guard<std::mutex> lock_t;

    typedef typename Upstream::pointer void_ptr;

public:
    /*! Get the default options for a pool. These are meant to be a sensible set of values for many use cases,
     *      and as such, may be tuned in the future. This function is exposed so that creating a set of options that are
     *      just a slight departure from the defaults is easy.
     */
    static pool_options get_default_options()
    {
        return unsync_pool::get_default_options();
    }

    /*! Constructor.
     *
     *  \param upstream the upstream memory resource for allocations
     *  \param options pool options to use
     */
    synchronized_pool_resource(Upstream * upstream, pool_options options = get_default_options())
        : upstream_pool(upstream, options)
    {
    }

    /*! Constructor. The upstream resource is obtained by calling \p get_global_resource<Upstream>.
     *
     *  \param options pool options to use
     */
    synchronized_pool_resource(pool_options options = get_default_options())
        : upstream_pool(get_global_resource<Upstream>(), options)
    {
    }

    /*! Releases all held memory to upstream.
     */
    void release()
    {
        lock_t lock(mtx);
        upstream_pool.release();
    }

    THRUST_NODISCARD virtual void_ptr do_allocate(std::size_t bytes, std::size_t alignment = THRUST_MR_DEFAULT_ALIGNMENT) override
    {
        lock_t lock(mtx);
        return upstream_pool.do_allocate(bytes, alignment);
    }

    virtual void do_deallocate(void_ptr p, std::size_t n, std::size_t alignment = THRUST_MR_DEFAULT_ALIGNMENT) override
    {
        lock_t lock(mtx);
        upstream_pool.do_deallocate(p, n, alignment);
    }

private:
    std::mutex mtx;
    unsync_pool upstream_pool;
};

/*! \} // memory_resources
 */

} // end mr
THRUST_NAMESPACE_END

#endif // THRUST_CPP_DIALECT >= 2011

