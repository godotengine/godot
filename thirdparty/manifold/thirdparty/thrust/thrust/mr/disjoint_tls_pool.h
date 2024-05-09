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

/*! \file disjoint_tls_pool.h
 *  \brief A function wrapping a thread local instance of a \p disjoint_unsynchronized_pool_resource.
 */

#pragma once

#include <thrust/detail/config.h>
#include <thrust/detail/cpp11_required.h>

#if THRUST_CPP_DIALECT >= 2011

#include <thrust/mr/disjoint_pool.h>

THRUST_NAMESPACE_BEGIN
namespace mr
{

/*! \addtogroup memory_management Memory Management
 *  \addtogroup memory_resources Memory Resources
 *  \ingroup memory_resources
 *  \{
 */

/*! Potentially constructs, if not yet created, and then returns the address of a thread-local
 *      \p disjoint_unsynchronized_pool_resource,
 *
 *  \tparam Upstream the first template argument to the pool template
 *  \tparam Bookkeeper the second template argument to the pool template
 *  \param upstream the first argument to the constructor, if invoked
 *  \param bookkeeper the second argument to the constructor, if invoked
 */
template<typename Upstream, typename Bookkeeper>
__host__
thrust::mr::disjoint_unsynchronized_pool_resource<Upstream, Bookkeeper> & tls_disjoint_pool(
    Upstream * upstream = NULL,
    Bookkeeper * bookkeeper = NULL)
{
    static thread_local auto adaptor = [&]{
        assert(upstream && bookkeeper);
        return thrust::mr::disjoint_unsynchronized_pool_resource<Upstream, Bookkeeper>(upstream, bookkeeper);
    }();

    return adaptor;
}

/*! \}
 */

} // end mr
THRUST_NAMESPACE_END

#endif // THRUST_CPP_DIALECT >= 2011

