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
 *  \brief A type used by the pooling resource adaptors to fine-tune their
 *  behavior.
 */

#pragma once

#include <cstddef>

#include <thrust/detail/integer_math.h>

#include <thrust/detail/config.h>
#include <thrust/detail/config/memory_resource.h>

THRUST_NAMESPACE_BEGIN
namespace mr
{

/*! \addtogroup memory_resources Memory Resources
 *  \ingroup memory_management
 *  \{
 */

/*! A type used for configuring pooling resource adaptors, to fine-tune their behavior and parameters.
 */
struct pool_options
{
    /*! The minimal number of blocks, i.e. pieces of memory handed off to the user from a pool of a given size, in a single
     *      chunk allocated from upstream.
     */
    std::size_t min_blocks_per_chunk;
    /*! The minimal number of bytes in a single chunk allocated from upstream.
     */
    std::size_t min_bytes_per_chunk;
    /*! The maximal number of blocks, i.e. pieces of memory handed off to the user from a pool of a given size, in a single
     *      chunk allocated from upstream.
     */
    std::size_t max_blocks_per_chunk;
    /*! The maximal number of bytes in a single chunk allocated from upstream.
     */
    std::size_t max_bytes_per_chunk;

    /*! The size of blocks in the smallest pool covered by the pool resource. All allocation requests below this size will
     *      be rounded up to this size.
     */
    std::size_t smallest_block_size;
    /*! The size of blocks in the largest pool covered by the pool resource. All allocation requests above this size will
     *      be considered oversized, allocated directly from upstream (and not from a pool), and cached only of \p cache_oversized
     *      is true.
     */
    std::size_t largest_block_size;

    /*! The alignment of all blocks in internal pools of the pool resource. All allocation requests above this alignment
     *      will be considered oversized, allocated directly from upstream (and not from a pool), and cached only of
     *      \p cache_oversized is true.
     */
    std::size_t alignment;

    /*! Decides whether oversized and overaligned blocks are cached for later use, or immediately return it to the upstream
     *      resource.
     */
    bool cache_oversized;

    /*! The size factor at which a cached allocation is considered too ridiculously oversized to use to fulfill an allocation
     *      request. For instance: the user requests an allocation of size 1024 bytes. A block of size 32 * 1024 bytes is
     *      cached. If \p cached_size_cutoff_factor is 32 or less, this block will be considered too big for that allocation
     *      request.
     */
    std::size_t cached_size_cutoff_factor;
    /*! The alignment factor at which a cached allocation is considered too ridiculously overaligned to use to fulfill an
     *      allocation request. For instance: the user requests an allocation aligned to 32 bytes. A block aligned to 1024 bytes
     *      is cached. If \p cached_size_cutoff_factor is 32 or less, this block will be considered too overaligned for that
     *      allocation request.
     */
    std::size_t cached_alignment_cutoff_factor;

    /*! Checks if the options are self-consistent.
     *
     *  /returns true if the options are self-consitent, false otherwise.
     */
    bool validate() const
    {
        if (!detail::is_power_of_2(smallest_block_size)) return false;
        if (!detail::is_power_of_2(largest_block_size)) return false;
        if (!detail::is_power_of_2(alignment)) return false;

        if (max_bytes_per_chunk == 0 || max_blocks_per_chunk == 0) return false;
        if (smallest_block_size == 0 || largest_block_size == 0) return false;

        if (min_blocks_per_chunk > max_blocks_per_chunk) return false;
        if (min_bytes_per_chunk > max_bytes_per_chunk) return false;

        if (smallest_block_size > largest_block_size) return false;

        if (min_blocks_per_chunk * smallest_block_size > max_bytes_per_chunk) return false;
        if (min_blocks_per_chunk * largest_block_size > max_bytes_per_chunk) return false;

        if (max_blocks_per_chunk * largest_block_size < min_bytes_per_chunk) return false;
        if (max_blocks_per_chunk * smallest_block_size < min_bytes_per_chunk) return false;

        if (alignment > smallest_block_size) return false;

        return true;
    }
};

/*! \} // memory_resources
 */

} // end mr
THRUST_NAMESPACE_END

