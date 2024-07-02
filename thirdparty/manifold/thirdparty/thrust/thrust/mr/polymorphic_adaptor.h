/*
 *  Copyright 2018-2019 NVIDIA Corporation
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

#pragma once

#include <thrust/detail/config.h>

#include <thrust/mr/memory_resource.h>

THRUST_NAMESPACE_BEGIN
namespace mr
{

template<typename Pointer = void *>
class polymorphic_adaptor_resource final : public memory_resource<Pointer>
{
public:
    polymorphic_adaptor_resource(memory_resource<Pointer> * t) : upstream_resource(t)
    {
    }

    virtual Pointer do_allocate(std::size_t bytes, std::size_t alignment = THRUST_MR_DEFAULT_ALIGNMENT) override
    {
        return upstream_resource->allocate(bytes, alignment);
    }

    virtual void do_deallocate(Pointer p, std::size_t bytes, std::size_t alignment) override
    {
        return upstream_resource->deallocate(p, bytes, alignment);
    }

    __host__ __device__
    virtual bool do_is_equal(const memory_resource<Pointer> & other) const noexcept override
    {
        return upstream_resource->is_equal(other);
    }

private:
    memory_resource<Pointer> * upstream_resource;
};

} // end mr
THRUST_NAMESPACE_END

