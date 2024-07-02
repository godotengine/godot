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

/*! \file tls_pool.h
 *  \brief A function wrapping a thread local instance of a \p unsynchronized_pool_resource.
 */

#pragma once

#include <thrust/detail/config.h>
#include <thrust/detail/cpp11_required.h>

#if THRUST_CPP_DIALECT >= 2011

#include <thrust/mr/pool.h>

THRUST_NAMESPACE_BEGIN
namespace mr
{

/*! \addtogroup memory_management Memory Management
 *  \addtogroup memory_resources Memory Resources
 *  \ingroup memory_resources
 *  \{
 */

/*! Potentially constructs, if not yet created, and then returns the address of a thread-local \p unsynchronized_pool_resource,
 *
 *  \tparam Upstream the template argument to the pool template
 *  \param upstream the argument to the constructor, if invoked
 */
template<typename Upstream, typename Bookkeeper>
__host__
thrust::mr::unsynchronized_pool_resource<Upstream> & tls_pool(Upstream * upstream = NULL)
{
    static thread_local auto adaptor = [&]{
        assert(upstream);
        return thrust::mr::unsynchronized_pool_resource<Upstream>(upstream);
    }();

    return adaptor;
}

/*! \}
 */

} // end mr
THRUST_NAMESPACE_END

#endif // THRUST_CPP_DIALECT >= 2011

