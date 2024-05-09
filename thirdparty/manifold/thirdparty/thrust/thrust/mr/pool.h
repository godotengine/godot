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
 *  \brief A caching and pooling memory resource adaptor which uses a single
 *  upstream resource for memory allocation, and embeds bookkeeping information
 *  in allocated blocks.
 */

#pragma once

#include <thrust/detail/config.h>

#include <thrust/detail/algorithm_wrapper.h>

#include <thrust/host_vector.h>

#include <thrust/mr/memory_resource.h>
#include <thrust/mr/allocator.h>
#include <thrust/mr/pool_options.h>

#include <cassert>

THRUST_NAMESPACE_BEGIN
namespace mr
{

/** \addtogroup memory_resources Memory Resources
 *  \ingroup memory_management
 *  \{
 */

/*! A memory resource adaptor allowing for pooling and caching allocations from \p Upstream, using memory allocated
 *      from it for both blocks then allocated to the user and for internal bookkeeping of the cached memory.
 *
 *  On a typical memory resource, calls to \p allocate and \p deallocate actually allocate and deallocate memory. Pooling
 *      memory resources only allocate and deallocate memory from an external resource (the upstream memory resource) when
 *      there's no suitable memory currently cached; otherwise, they use memory they have acquired beforehand, to make
 *      memory allocation faster and more efficient.
 *
 *  The non-disjoint version of the pool resource uses a single upstream memory resource. Every allocation is larger than
 *      strictly necessary to fulfill the end-user's request, because it needs to account for the memory overhead of tracking
 *      the memory blocks and chunks inside those same memory regions. Nevertheless, this version should be more memory-efficient
 *      than the \p disjoint_unsynchronized_pool_resource, because it doesn't need to allocate additional blocks of memory
 *      from a separate resource, which in turn would necessitate the bookkeeping overhead in the upstream resource.
 *
 *  This version requires that memory allocated from Upstream is accessible from device. It supports smart references,
 *      meaning that the non-managed CUDA resource, returning a device-tagged pointer, will work, but will be much less
 *      efficient than the disjoint version, which wouldn't need to touch device memory at all, and therefore wouldn't need
 *      to transfer it back and forth between the host and the device whenever an allocation or a deallocation happens.
 *
 *  \tparam Upstream the type of memory resources that will be used for allocating memory blocks
 */
template<typename Upstream>
class unsynchronized_pool_resource final
    : public memory_resource<typename Upstream::pointer>,
        private validator<Upstream>
{
public:
    /*! Get the default options for a pool. These are meant to be a sensible set of values for many use cases,
     *      and as such, may be tuned in the future. This function is exposed so that creating a set of options that are
     *      just a slight departure from the defaults is easy.
     */
    static pool_options get_default_options()
    {
        pool_options ret;

        ret.min_blocks_per_chunk = 16;
        ret.min_bytes_per_chunk = 1024;
        ret.max_blocks_per_chunk = static_cast<std::size_t>(1) << 20;
        ret.max_bytes_per_chunk = static_cast<std::size_t>(1) << 30;

        ret.smallest_block_size = THRUST_MR_DEFAULT_ALIGNMENT;
        ret.largest_block_size = static_cast<std::size_t>(1) << 20;

        ret.alignment = THRUST_MR_DEFAULT_ALIGNMENT;

        ret.cache_oversized = true;

        ret.cached_size_cutoff_factor = 16;
        ret.cached_alignment_cutoff_factor = 16;

        return ret;
    }

    /*! Constructor.
     *
     *  \param upstream the upstream memory resource for allocations
     *  \param options pool options to use
     */
    unsynchronized_pool_resource(Upstream * upstream, pool_options options = get_default_options())
        : m_upstream(upstream),
        m_options(options),
        m_smallest_block_log2(detail::log2_ri(m_options.smallest_block_size)),
        m_pools(upstream),
        m_allocated(),
        m_oversized(),
        m_cached_oversized()
    {
        assert(m_options.validate());

        pool p = { block_descriptor_ptr(), 0 };
        m_pools.resize(detail::log2_ri(m_options.largest_block_size) - m_smallest_block_log2 + 1, p);
    }

    // TODO: C++11: use delegating constructors

    /*! Constructor. The upstream resource is obtained by calling \p get_global_resource<Upstream>.
     *
     *  \param options pool options to use
     */
    unsynchronized_pool_resource(pool_options options = get_default_options())
        : m_upstream(get_global_resource<Upstream>()),
        m_options(options),
        m_smallest_block_log2(detail::log2_ri(m_options.smallest_block_size)),
        m_pools(get_global_resource<Upstream>()),
        m_allocated(),
        m_oversized(),
        m_cached_oversized()
    {
        assert(m_options.validate());

        pool p = { block_descriptor_ptr(), 0 };
        m_pools.resize(detail::log2_ri(m_options.largest_block_size) - m_smallest_block_log2 + 1, p);
    }

    /*! Destructor. Releases all held memory to upstream.
     */
    ~unsynchronized_pool_resource()
    {
        release();
    }

private:
    typedef typename Upstream::pointer void_ptr;
    typedef typename thrust::detail::pointer_traits<void_ptr>::template rebind<char>::other char_ptr;

    struct block_descriptor;
    struct chunk_descriptor;
    struct oversized_block_descriptor;

    typedef typename thrust::detail::pointer_traits<void_ptr>::template rebind<block_descriptor>::other block_descriptor_ptr;
    typedef typename thrust::detail::pointer_traits<void_ptr>::template rebind<chunk_descriptor>::other chunk_descriptor_ptr;
    typedef typename thrust::detail::pointer_traits<void_ptr>::template rebind<oversized_block_descriptor>::other oversized_block_descriptor_ptr;

    struct block_descriptor
    {
        block_descriptor_ptr next;
    };

    struct chunk_descriptor
    {
        std::size_t size;
        chunk_descriptor_ptr next;
    };

    // this was originally a forward list, but I made it a doubly linked list
    // because that way deallocation when not caching is faster and doesn't require
    // traversal of a linked list (it's still a forward list for the cached list,
    // because allocation from that list already traverses)
    //
    // TODO: investigate whether it's better to have this be a doubly-linked list
    // with fast do_deallocate when !m_options.cache_oversized, or to have this be
    // a forward list and require traversal in do_deallocate
    //
    // I assume that it is better this way, but the additional pointer could
    // potentially hurt? these are supposed to be oversized and/or overaligned,
    // so they are kinda memory intensive already
    struct oversized_block_descriptor
    {
        std::size_t size;
        std::size_t alignment;
        oversized_block_descriptor_ptr prev;
        oversized_block_descriptor_ptr next;
        oversized_block_descriptor_ptr next_cached;
    };

    struct pool
    {
        block_descriptor_ptr free_list;
        std::size_t previous_allocated_count;
    };

    typedef thrust::host_vector<
        pool,
        allocator<pool, Upstream>
    > pool_vector;

    Upstream * m_upstream;

    pool_options m_options;
    std::size_t m_smallest_block_log2;

    pool_vector m_pools;
    chunk_descriptor_ptr m_allocated;
    oversized_block_descriptor_ptr m_oversized;
    oversized_block_descriptor_ptr m_cached_oversized;

public:
    /*! Releases all held memory to upstream.
     */
    void release()
    {
        // reset the buckets
        for (std::size_t i = 0; i < m_pools.size(); ++i)
        {
            thrust::raw_reference_cast(m_pools[i]).free_list = block_descriptor_ptr();
            thrust::raw_reference_cast(m_pools[i]).previous_allocated_count = 0;
        }

        // deallocate memory allocated for the buckets
        while (detail::pointer_traits<chunk_descriptor_ptr>::get(m_allocated))
        {
            chunk_descriptor_ptr alloc = m_allocated;
            m_allocated = thrust::raw_reference_cast(*m_allocated).next;

            void_ptr p = static_cast<void_ptr>(
                static_cast<char_ptr>(
                    static_cast<void_ptr>(alloc)
                ) - thrust::raw_reference_cast(*alloc).size
            );
            m_upstream->do_deallocate(p, thrust::raw_reference_cast(*alloc).size + sizeof(chunk_descriptor), m_options.alignment);
        }

        // deallocate cached oversized/overaligned memory
        while (detail::pointer_traits<oversized_block_descriptor_ptr>::get(m_oversized))
        {
            oversized_block_descriptor_ptr alloc = m_oversized;
            m_oversized = thrust::raw_reference_cast(*m_oversized).next;

            void_ptr p = static_cast<void_ptr>(
                static_cast<char_ptr>(
                    static_cast<void_ptr>(alloc)
                ) - thrust::raw_reference_cast(*alloc).size
            );
            m_upstream->do_deallocate(p, thrust::raw_reference_cast(*alloc).size + sizeof(oversized_block_descriptor), thrust::raw_reference_cast(*alloc).alignment);
        }

        m_cached_oversized = oversized_block_descriptor_ptr();
    }

    THRUST_NODISCARD virtual void_ptr do_allocate(std::size_t bytes, std::size_t alignment = THRUST_MR_DEFAULT_ALIGNMENT) override
    {
        bytes = (std::max)(bytes, m_options.smallest_block_size);
        assert(detail::is_power_of_2(alignment));

        // an oversized and/or overaligned allocation requested; needs to be allocated separately
        if (bytes > m_options.largest_block_size || alignment > m_options.alignment)
        {
            if (m_options.cache_oversized)
            {
                oversized_block_descriptor_ptr ptr = m_cached_oversized;
                oversized_block_descriptor_ptr * previous = &m_cached_oversized;
                while (detail::pointer_traits<oversized_block_descriptor_ptr>::get(ptr))
                {
                    oversized_block_descriptor desc = *ptr;
                    bool is_good = desc.size >= bytes && desc.alignment >= alignment;

                    // if the size is bigger than the requested size by a factor
                    // bigger than or equal to the specified cutoff for size,
                    // allocate a new block
                    if (is_good)
                    {
                        std::size_t size_factor = desc.size / bytes;
                        if (size_factor >= m_options.cached_size_cutoff_factor)
                        {
                            is_good = false;
                        }
                    }

                    // if the alignment is bigger than the requested one by a factor
                    // bigger than or equal to the specified cutoff for alignment,
                    // allocate a new block
                    if (is_good)
                    {
                        std::size_t alignment_factor = desc.alignment / alignment;
                        if (alignment_factor >= m_options.cached_alignment_cutoff_factor)
                        {
                            is_good = false;
                        }
                    }

                    if (is_good)
                    {
                        if (previous != &m_cached_oversized)
                        {
                            oversized_block_descriptor previous_desc = **previous;
                            previous_desc.next_cached = desc.next_cached;
                            **previous = previous_desc;
                        }
                        else
                        {
                            m_cached_oversized = desc.next_cached;
                        }

                        desc.next_cached = oversized_block_descriptor_ptr();
                        *ptr = desc;

                        return static_cast<void_ptr>(
                            static_cast<char_ptr>(
                                static_cast<void_ptr>(ptr)
                            ) - desc.size
                        );
                    }

                    previous = &thrust::raw_reference_cast(*ptr).next_cached;
                    ptr = *previous;
                }
            }

            // no fitting cached block found; allocate a new one that's just up to the specs
            void_ptr allocated = m_upstream->do_allocate(bytes + sizeof(oversized_block_descriptor), alignment);
            oversized_block_descriptor_ptr block = static_cast<oversized_block_descriptor_ptr>(
                static_cast<void_ptr>(
                    static_cast<char_ptr>(allocated) + bytes
                )
            );

            oversized_block_descriptor desc;
            desc.size = bytes;
            desc.alignment = alignment;
            desc.prev = oversized_block_descriptor_ptr();
            desc.next = m_oversized;
            desc.next_cached = oversized_block_descriptor_ptr();
            *block = desc;
            m_oversized = block;

            if (detail::pointer_traits<oversized_block_descriptor_ptr>::get(desc.next))
            {
                oversized_block_descriptor next = *desc.next;
                next.prev = block;
                *desc.next = next;
            }

            return allocated;
        }

        // the request is NOT for oversized and/or overaligned memory
        // allocate a block from an appropriate bucket
        std::size_t bytes_log2 = thrust::detail::log2_ri(bytes);
        std::size_t bucket_idx = bytes_log2 - m_smallest_block_log2;
        pool & bucket = thrust::raw_reference_cast(m_pools[bucket_idx]);

        bytes = static_cast<std::size_t>(1) << bytes_log2;

        // if the free list of the bucket has no elements, allocate a new chunk
        // and split it into blocks pushed to the free list
        if (!detail::pointer_traits<block_descriptor_ptr>::get(bucket.free_list))
        {
            std::size_t n = bucket.previous_allocated_count;
            if (n == 0)
            {
                n = m_options.min_blocks_per_chunk;
                if (n < (m_options.min_bytes_per_chunk >> bytes_log2))
                {
                    n = m_options.min_bytes_per_chunk >> bytes_log2;
                }
            }
            else
            {
                n = n * 3 / 2;
                if (n > (m_options.max_bytes_per_chunk >> bytes_log2))
                {
                    n = m_options.max_bytes_per_chunk >> bytes_log2;
                }
                if (n > m_options.max_blocks_per_chunk)
                {
                    n = m_options.max_blocks_per_chunk;
                }
            }

            std::size_t descriptor_size = (std::max)(sizeof(block_descriptor), m_options.alignment);
            std::size_t block_size = bytes + descriptor_size;
            block_size += m_options.alignment - block_size % m_options.alignment;
            std::size_t chunk_size = block_size * n;

            void_ptr allocated = m_upstream->do_allocate(chunk_size + sizeof(chunk_descriptor), m_options.alignment);
            chunk_descriptor_ptr chunk = static_cast<chunk_descriptor_ptr>(
                static_cast<void_ptr>(
                    static_cast<char_ptr>(allocated) + chunk_size
                )
            );

            chunk_descriptor chunk_desc;
            chunk_desc.size = chunk_size;
            chunk_desc.next = m_allocated;
            *chunk = chunk_desc;
            m_allocated = chunk;

            for (std::size_t i = 0; i < n; ++i)
            {
                block_descriptor_ptr block = static_cast<block_descriptor_ptr>(
                    static_cast<void_ptr>(
                        static_cast<char_ptr>(allocated) + block_size * i + bytes
                    )
                );

                block_descriptor block_desc;
                block_desc.next = bucket.free_list;
                *block = block_desc;
                bucket.free_list = block;
            }
        }

        // allocate a block from the front of the bucket's free list
        block_descriptor_ptr block = bucket.free_list;
        bucket.free_list = thrust::raw_reference_cast(*block).next;
        return static_cast<void_ptr>(
            static_cast<char_ptr>(
                static_cast<void_ptr>(block)
            ) - bytes
        );
    }

    virtual void do_deallocate(void_ptr p, std::size_t n, std::size_t alignment = THRUST_MR_DEFAULT_ALIGNMENT) override
    {
        n = (std::max)(n, m_options.smallest_block_size);
        assert(detail::is_power_of_2(alignment));

        // verify that the pointer is at least as aligned as claimed
        assert(reinterpret_cast<detail::intmax_t>(detail::pointer_traits<void_ptr>::get(p)) % alignment == 0);

        // the deallocated block is oversized and/or overaligned
        if (n > m_options.largest_block_size || alignment > m_options.alignment)
        {
            oversized_block_descriptor_ptr block = static_cast<oversized_block_descriptor_ptr>(
                static_cast<void_ptr>(
                    static_cast<char_ptr>(p) + n
                )
            );

            oversized_block_descriptor desc = *block;

            if (m_options.cache_oversized)
            {
                desc.next_cached = m_cached_oversized;
                *block = desc;
                m_cached_oversized = block;

                return;
            }

            if (!detail::pointer_traits<oversized_block_descriptor_ptr>::get(desc.prev))
            {
                assert(m_oversized == block);
                m_oversized = desc.next;
            }
            else
            {
                oversized_block_descriptor prev = *desc.prev;
                assert(prev.next == block);
                prev.next = desc.next;
                *desc.prev = prev;
            }

            if (detail::pointer_traits<oversized_block_descriptor_ptr>::get(desc.next))
            {
                oversized_block_descriptor next = *desc.next;
                assert(next.prev == block);
                next.prev = desc.prev;
                *desc.next = next;
            }

            m_upstream->do_deallocate(p, desc.size + sizeof(oversized_block_descriptor), desc.alignment);

            return;
        }

        // push the block to the front of the appropriate bucket's free list
        std::size_t n_log2 = thrust::detail::log2_ri(n);
        std::size_t bucket_idx = n_log2 - m_smallest_block_log2;
        pool & bucket = thrust::raw_reference_cast(m_pools[bucket_idx]);

        n = static_cast<std::size_t>(1) << n_log2;

        block_descriptor_ptr block = static_cast<block_descriptor_ptr>(
            static_cast<void_ptr>(
                static_cast<char_ptr>(p) + n
            )
        );

        block_descriptor desc;
        desc.next = bucket.free_list;
        *block = desc;
        bucket.free_list = block;
    }
};

/*! \} // memory_resources
 */

} // end mr
THRUST_NAMESPACE_END

