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
 *  \brief Global operator new-based memory resource.
 */

#pragma once

#include <thrust/detail/config.h>

#include <thrust/mr/memory_resource.h>

THRUST_NAMESPACE_BEGIN
namespace mr
{

/** \addtogroup memory_resources Memory Resources
 *  \ingroup memory_management
 *  \{
 */

/*! A memory resource that uses global operators new and delete to allocate and deallocate memory. Uses alignment-enabled
 *      overloads when available, otherwise uses regular overloads and implements alignment requirements by itself.
 */
class new_delete_resource final : public memory_resource<>
{
public:
    void * do_allocate(std::size_t bytes, std::size_t alignment = THRUST_MR_DEFAULT_ALIGNMENT) override
    {
#if defined(__cpp_aligned_new)
        return ::operator new(bytes, std::align_val_t(alignment));
#else
        // allocate memory for bytes, plus potential alignment correction,
        // plus store of the correction offset
        void * p = ::operator new(bytes + alignment + sizeof(std::size_t));
        std::size_t ptr_int = reinterpret_cast<std::size_t>(p);
        // calculate the offset, i.e. how many bytes of correction was necessary
        // to get an aligned pointer
        std::size_t offset = (ptr_int % alignment) ? (alignment - ptr_int % alignment) : 0;
        // calculate the return pointer
        char * ptr = static_cast<char *>(p) + offset;
        // store the offset right after the actually returned value
        std::size_t * offset_store = reinterpret_cast<std::size_t *>(ptr + bytes);
        *offset_store = offset;
        return static_cast<void *>(ptr);
#endif
    }

    void do_deallocate(void * p, std::size_t bytes, std::size_t alignment = THRUST_MR_DEFAULT_ALIGNMENT) override
    {
#if defined(__cpp_aligned_new)
# if defined(__cpp_sized_deallocation)
        ::operator delete(p, bytes, std::align_val_t(alignment));
# else
        (void)bytes;
        ::operator delete(p, std::align_val_t(alignment));
# endif
#else
        (void)alignment;
        char * ptr = static_cast<char *>(p);
        // calculate where the offset is stored
        std::size_t * offset = reinterpret_cast<std::size_t *>(ptr + bytes);
        // calculate the original pointer
        p = static_cast<void *>(ptr - *offset);
        ::operator delete(p);
#endif
    }
};

/*! \} // memory_resources
 */

} // end mr
THRUST_NAMESPACE_END

