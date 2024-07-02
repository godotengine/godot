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
 *  \brief A caching and pooling memory resource adaptor which uses separate upstream resources for memory allocation
 *      and bookkeeping.
 */

#pragma once

#include <thrust/detail/algorithm_wrapper.h>
#include <thrust/detail/config.h>

#include <thrust/host_vector.h>
#include <thrust/binary_search.h>
#include <thrust/detail/seq.h>

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

/*! A memory resource adaptor allowing for pooling and caching allocations from \p Upstream, using \p Bookkeeper for
 *      management of that cached and pooled memory, allowing to cache portions of memory inaccessible from the host.
 *
 *  On a typical memory resource, calls to \p allocate and \p deallocate actually allocate and deallocate memory. Pooling
 *      memory resources only allocate and deallocate memory from an external resource (the upstream memory resource) when
 *      there's no suitable memory currently cached; otherwise, they use memory they have acquired beforehand, to make
 *      memory allocation faster and more efficient.
 *
 *  The disjoint version of the pool resources uses a separate upstream memory resource, \p Bookkeeper, to allocate memory
 *      necessary to manage the cached memory. There may be many reasons to do that; the canonical one is that \p Upstream
 *      allocates memory that is inaccessible to the code of the pool resource, which means that it cannot embed the necessary
 *      information in memory obtained from \p Upstream; for instance, \p Upstream can be a CUDA non-managed memory
 *      resource, or a CUDA managed memory resource whose memory we would prefer to not migrate back and forth between
 *      host and device when executing bookkeeping code.
 *
 *  This is not the only case where it makes sense to use a disjoint pool resource, though. In a multi-core environment
 *      it may be beneficial to avoid stealing cache lines from other cores by writing over bookkeeping information
 *      embedded in an allocated block of memory. In such a case, one can imagine wanting to use a disjoint pool where
 *      both the upstream and the bookkeeper are of the same type, to allocate memory consistently, but separately for
 *      those two purposes.
 *
 *  \tparam Upstream the type of memory resources that will be used for allocating memory blocks to be handed off to the user
 *  \tparam Bookkeeper the type of memory resources that will be used for allocating bookkeeping memory
 */
template<typename Upstream, typename Bookkeeper>
class disjoint_unsynchronized_pool_resource final
    : public memory_resource<typename Upstream::pointer>,
        private validator2<Upstream, Bookkeeper>
{
public:
    /*! Get the default options for a disjoint pool. These are meant to be a sensible set of values for many use cases,
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
     *  \param bookkeeper the upstream memory resource for bookkeeping
     *  \param options pool options to use
     */
    disjoint_unsynchronized_pool_resource(Upstream * upstream, Bookkeeper * bookkeeper,
        pool_options options = get_default_options())
        : m_upstream(upstream),
        m_bookkeeper(bookkeeper),
        m_options(options),
        m_smallest_block_log2(detail::log2_ri(m_options.smallest_block_size)),
        m_pools(m_bookkeeper),
        m_allocated(m_bookkeeper),
        m_cached_oversized(m_bookkeeper),
        m_oversized(m_bookkeeper)
    {
        assert(m_options.validate());

        pointer_vector free(m_bookkeeper);
        pool p(free);
        m_pools.resize(detail::log2_ri(m_options.largest_block_size) - m_smallest_block_log2 + 1, p);
    }

    // TODO: C++11: use delegating constructors

    /*! Constructor. Upstream and bookkeeping resources are obtained by calling \p get_global_resource for their types.
     *
     *  \param options pool options to use
     */
    disjoint_unsynchronized_pool_resource(pool_options options = get_default_options())
        : m_upstream(get_global_resource<Upstream>()),
        m_bookkeeper(get_global_resource<Bookkeeper>()),
        m_options(options),
        m_smallest_block_log2(detail::log2_ri(m_options.smallest_block_size)),
        m_pools(m_bookkeeper),
        m_allocated(m_bookkeeper),
        m_cached_oversized(m_bookkeeper),
        m_oversized(m_bookkeeper)
    {
        assert(m_options.validate());

        pointer_vector free(m_bookkeeper);
        pool p(free);
        m_pools.resize(detail::log2_ri(m_options.largest_block_size) - m_smallest_block_log2 + 1, p);
    }

    /*! Destructor. Releases all held memory to upstream.
     */
    ~disjoint_unsynchronized_pool_resource()
    {
        release();
    }

private:
    typedef typename Upstream::pointer void_ptr;
    typedef typename thrust::detail::pointer_traits<void_ptr>::template rebind<char>::other char_ptr;

    struct chunk_descriptor
    {
        std::size_t size;
        void_ptr pointer;
    };

    typedef thrust::host_vector<
        chunk_descriptor,
        allocator<chunk_descriptor, Bookkeeper>
    > chunk_vector;

    struct oversized_block_descriptor
    {
        std::size_t size;
        std::size_t alignment;
        void_ptr pointer;

        __host__ __device__
        bool operator==(const oversized_block_descriptor & other) const
        {
            return size == other.size && alignment == other.alignment && pointer == other.pointer;
        }

        __host__ __device__
        bool operator<(const oversized_block_descriptor & other) const
        {
            return size < other.size || (size == other.size && alignment < other.alignment);
        }
    };

    struct equal_pointers
    {
    public:
        __host__ __device__
        equal_pointers(void_ptr p) : p(p)
        {
        }

        __host__ __device__
        bool operator()(const oversized_block_descriptor & desc) const
        {
            return desc.pointer == p;
        }

    private:
        void_ptr p;
    };

    struct matching_alignment
    {
    public:
        __host__ __device__
        matching_alignment(std::size_t requested) : requested(requested)
        {
        }

        __host__ __device__
        bool operator()(const oversized_block_descriptor & desc) const
        {
            return desc.alignment >= requested;
        }

    private:
        std::size_t requested;
    };

    typedef thrust::host_vector<
        oversized_block_descriptor,
        allocator<oversized_block_descriptor, Bookkeeper>
    > oversized_block_vector;

    typedef thrust::host_vector<
        void_ptr,
        allocator<void_ptr, Bookkeeper>
    > pointer_vector;

    struct pool
    {
        __host__
        pool(const pointer_vector & free)
            : free_blocks(free),
            previous_allocated_count(0)
        {
        }

        __host__
        pool(const pool & other)
            : free_blocks(other.free_blocks),
            previous_allocated_count(other.previous_allocated_count)
        {
        }

#if THRUST_CPP_DIALECT >= 2011
        pool & operator=(const pool &) = default;
#endif

        __host__
        ~pool() {}

        pointer_vector free_blocks;
        std::size_t previous_allocated_count;
    };

    typedef thrust::host_vector<
        pool,
        allocator<pool, Bookkeeper>
    > pool_vector;

    Upstream * m_upstream;
    Bookkeeper * m_bookkeeper;

    pool_options m_options;
    std::size_t m_smallest_block_log2;

    // buckets containing free lists for each pooled size
    pool_vector m_pools;
    // list of all allocations from upstream for the above
    chunk_vector m_allocated;
    // list of all cached oversized/overaligned blocks that have been returned to the pool to cache
    oversized_block_vector m_cached_oversized;
    // list of all oversized/overaligned allocations from upstream
    oversized_block_vector m_oversized;

public:
    /*! Releases all held memory to upstream.
     */
    void release()
    {
        // reset the buckets
        for (std::size_t i = 0; i < m_pools.size(); ++i)
        {
            m_pools[i].free_blocks.clear();
            m_pools[i].previous_allocated_count = 0;
        }

        // deallocate memory allocated for the buckets
        for (std::size_t i = 0; i < m_allocated.size(); ++i)
        {
            m_upstream->do_deallocate(
                m_allocated[i].pointer,
                m_allocated[i].size,
                m_options.alignment);
        }

        // deallocate cached oversized/overaligned memory
        for (std::size_t i = 0; i < m_oversized.size(); ++i)
        {
            m_upstream->do_deallocate(
                m_oversized[i].pointer,
                m_oversized[i].size,
                m_oversized[i].alignment);
        }

        m_allocated.clear();
        m_oversized.clear();
        m_cached_oversized.clear();
    }

    THRUST_NODISCARD virtual void_ptr do_allocate(std::size_t bytes, std::size_t alignment = THRUST_MR_DEFAULT_ALIGNMENT) override
    {
        bytes = (std::max)(bytes, m_options.smallest_block_size);
        assert(detail::is_power_of_2(alignment));

        // an oversized and/or overaligned allocation requested; needs to be allocated separately
        if (bytes > m_options.largest_block_size || alignment > m_options.alignment)
        {
            oversized_block_descriptor oversized;
            oversized.size = bytes;
            oversized.alignment = alignment;

            if (m_options.cache_oversized && !m_cached_oversized.empty())
            {
                typename oversized_block_vector::iterator it = thrust::lower_bound(
                    thrust::seq,
                    m_cached_oversized.begin(),
                    m_cached_oversized.end(),
                    oversized);

                // if the size is bigger than the requested size by a factor
                // bigger than or equal to the specified cutoff for size,
                // allocate a new block
                if (it != m_cached_oversized.end())
                {
                    std::size_t size_factor = (*it).size / bytes;
                    if (size_factor >= m_options.cached_size_cutoff_factor)
                    {
                        it = m_cached_oversized.end();
                    }
                }

                if (it != m_cached_oversized.end() && (*it).alignment < alignment)
                {
                    it = find_if(it + 1, m_cached_oversized.end(), matching_alignment(alignment));
                }

                // if the alignment is bigger than the requested one by a factor
                // bigger than or equal to the specified cutoff for alignment,
                // allocate a new block
                if (it != m_cached_oversized.end())
                {
                    std::size_t alignment_factor = (*it).alignment / alignment;
                    if (alignment_factor >= m_options.cached_alignment_cutoff_factor)
                    {
                        it = m_cached_oversized.end();
                    }
                }

                if (it != m_cached_oversized.end())
                {
                    oversized.pointer = (*it).pointer;
                    m_cached_oversized.erase(it);
                    return oversized.pointer;
                }
            }

            // no fitting cached block found; allocate a new one that's just up to the specs
            oversized.pointer = m_upstream->do_allocate(bytes, alignment);
            m_oversized.push_back(oversized);

            return oversized.pointer;
        }

        // the request is NOT for oversized and/or overaligned memory
        // allocate a block from an appropriate bucket
        std::size_t bytes_log2 = thrust::detail::log2_ri(bytes);
        std::size_t bucket_idx = bytes_log2 - m_smallest_block_log2;
        pool & bucket = m_pools[bucket_idx];

        // if the free list of the bucket has no elements, allocate a new chunk
        // and split it into blocks pushed to the free list
        if (bucket.free_blocks.empty())
        {
            std::size_t bucket_size = static_cast<std::size_t>(1) << bytes_log2;

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

            bytes = n << bytes_log2;

            assert(n >= m_options.min_blocks_per_chunk);
            assert(n <= m_options.max_blocks_per_chunk);
            assert(bytes >= m_options.min_bytes_per_chunk);
            assert(bytes <= m_options.max_bytes_per_chunk);

            chunk_descriptor allocated;
            allocated.size = bytes;
            allocated.pointer = m_upstream->do_allocate(bytes, m_options.alignment);
            m_allocated.push_back(allocated);
            bucket.previous_allocated_count = n;

            for (std::size_t i = 0; i < n; ++i)
            {
                bucket.free_blocks.push_back(
                    static_cast<void_ptr>(
                        static_cast<char_ptr>(allocated.pointer) + i * bucket_size
                    )
                );
            }
        }

        // allocate a block from the front of the bucket's free list
        void_ptr ret = bucket.free_blocks.back();
        bucket.free_blocks.pop_back();
        return ret;
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
            typename oversized_block_vector::iterator it = find_if(m_oversized.begin(), m_oversized.end(), equal_pointers(p));
            assert(it != m_oversized.end());

            oversized_block_descriptor oversized = *it;

            if (m_options.cache_oversized)
            {
                typename oversized_block_vector::iterator position = lower_bound(m_cached_oversized.begin(), m_cached_oversized.end(), oversized);
                m_cached_oversized.insert(position, oversized);
                return;
            }

            m_oversized.erase(it);

            m_upstream->do_deallocate(p, oversized.size, oversized.alignment);

            return;
        }

        // push the block to the front of the appropriate bucket's free list
        std::size_t n_log2 = thrust::detail::log2_ri(n);
        std::size_t bucket_idx = n_log2 - m_smallest_block_log2;
        pool & bucket = m_pools[bucket_idx];

        bucket.free_blocks.push_back(p);
    }
};

/*! \} // memory_resource
 */

} // end mr
THRUST_NAMESPACE_END

