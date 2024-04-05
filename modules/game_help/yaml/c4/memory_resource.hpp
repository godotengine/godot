#ifndef _C4_MEMORY_RESOURCE_HPP_
#define _C4_MEMORY_RESOURCE_HPP_

/** @file memory_resource.hpp Provides facilities to allocate typeless
 *  memory, via the memory resource model consecrated with C++17. */

/** @defgroup memory memory utilities */

/** @defgroup raw_memory_alloc Raw memory allocation
 * @ingroup memory
 */

/** @defgroup memory_resources Memory resources
 * @ingroup memory
 */

#include "config.hpp"
#include "error.hpp"

namespace c4 {

// need these forward decls here
struct MemoryResource;
struct MemoryResourceMalloc;
struct MemoryResourceStack;
MemoryResourceMalloc* get_memory_resource_malloc();
MemoryResourceStack* get_memory_resource_stack();
namespace detail { MemoryResource*& get_memory_resource(); }


// c-style allocation ---------------------------------------------------------

// this API provides aligned allocation functions.
// These functions forward the call to a user-modifiable function.


// aligned allocation.

/** Aligned allocation. Merely calls the current get_aalloc() function.
 * @see get_aalloc()
 * @ingroup raw_memory_alloc */
void* aalloc(size_t sz, size_t alignment);

/** Aligned free. Merely calls the current get_afree() function.
 * @see get_afree()
 * @ingroup raw_memory_alloc */
void afree(void* ptr);

/** Aligned reallocation. Merely calls the current get_arealloc() function.
 * @see get_arealloc()
 * @ingroup raw_memory_alloc */
void* arealloc(void* ptr, size_t oldsz, size_t newsz, size_t alignment);


// allocation setup facilities.

/** Function pointer type for aligned allocation
 * @see set_aalloc()
 * @ingroup raw_memory_alloc */
using aalloc_pfn = void* (*)(size_t size, size_t alignment);

/** Function pointer type for aligned deallocation
 * @see set_afree()
 * @ingroup raw_memory_alloc */
using afree_pfn = void  (*)(void *ptr);

/** Function pointer type for aligned reallocation
 * @see set_arealloc()
 * @ingroup raw_memory_alloc */
using arealloc_pfn = void* (*)(void *ptr, size_t oldsz, size_t newsz, size_t alignment);


// allocation function pointer setters/getters

/** Set the global aligned allocation function.
 * @see aalloc()
 * @see get_aalloc()
 * @ingroup raw_memory_alloc */
void set_aalloc(aalloc_pfn fn);

/** Set the global aligned deallocation function.
 * @see afree()
 * @see get_afree()
 * @ingroup raw_memory_alloc */
void set_afree(afree_pfn fn);

/** Set the global aligned reallocation function.
 * @see arealloc()
 * @see get_arealloc()
 * @ingroup raw_memory_alloc */
void set_arealloc(arealloc_pfn fn);


/** Get the global aligned reallocation function.
 * @see arealloc()
 * @ingroup raw_memory_alloc */
aalloc_pfn get_aalloc();

/** Get the global aligned deallocation function.
 * @see afree()
 * @ingroup raw_memory_alloc */
afree_pfn get_afree();

/** Get the global aligned reallocation function.
 * @see arealloc()
 * @ingroup raw_memory_alloc */
arealloc_pfn get_arealloc();


//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------
// c++-style allocation -------------------------------------------------------

/** C++17-style memory_resource base class. See http://en.cppreference.com/w/cpp/experimental/memory_resource
 * @ingroup memory_resources */
struct MemoryResource
{
    const char *name = nullptr;
    virtual ~MemoryResource() {}

    void* allocate(size_t sz, size_t alignment=alignof(max_align_t), void *hint=nullptr)
    {
        void *mem = this->do_allocate(sz, alignment, hint);
        C4_CHECK_MSG(mem != nullptr, "could not allocate %lu bytes", sz);
        return mem;
    }

    void* reallocate(void* ptr, size_t oldsz, size_t newsz, size_t alignment=alignof(max_align_t))
    {
        void *mem = this->do_reallocate(ptr, oldsz, newsz, alignment);
        C4_CHECK_MSG(mem != nullptr, "could not reallocate from %lu to %lu bytes", oldsz, newsz);
        return mem;
    }

    void deallocate(void* ptr, size_t sz, size_t alignment=alignof(max_align_t))
    {
        this->do_deallocate(ptr, sz, alignment);
    }

protected:

    virtual void* do_allocate(size_t sz, size_t alignment, void* hint) = 0;
    virtual void* do_reallocate(void* ptr, size_t oldsz, size_t newsz, size_t alignment) = 0;
    virtual void  do_deallocate(void* ptr, size_t sz, size_t alignment) = 0;

};

/** get the current global memory resource. To avoid static initialization
 * order problems, this is implemented using a function call to ensure
 * that it is available when first used.
 * @ingroup memory_resources */
C4_ALWAYS_INLINE MemoryResource* get_memory_resource()
{
    return detail::get_memory_resource();
}

/** set the global memory resource
 * @ingroup memory_resources */
C4_ALWAYS_INLINE void set_memory_resource(MemoryResource* mr)
{
    C4_ASSERT(mr != nullptr);
    detail::get_memory_resource() = mr;
}


//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------
/** A c4::aalloc-based memory resource. Thread-safe if the implementation
 * called by c4::aalloc() is safe.
 * @ingroup memory_resources */
struct MemoryResourceMalloc : public MemoryResource
{

    MemoryResourceMalloc() { name = "malloc"; }
    virtual ~MemoryResourceMalloc() override {}

protected:

    virtual void* do_allocate(size_t sz, size_t alignment, void *hint) override
    {
        C4_UNUSED(hint);
        return c4::aalloc(sz, alignment);
    }

    virtual void  do_deallocate(void* ptr, size_t sz, size_t alignment) override
    {
        C4_UNUSED(sz);
        C4_UNUSED(alignment);
        c4::afree(ptr);
    }

    virtual void* do_reallocate(void* ptr, size_t oldsz, size_t newsz, size_t alignment) override
    {
        return c4::arealloc(ptr, oldsz, newsz, alignment);
    }

};

/** returns a malloc-based memory resource
 * @ingroup memory_resources */
C4_ALWAYS_INLINE MemoryResourceMalloc* get_memory_resource_malloc()
{
    /** @todo use a nifty counter:
     * https://en.wikibooks.org/wiki/More_C%2B%2B_Idioms/Nifty_Counter */
    static MemoryResourceMalloc mr;
    return &mr;
}

namespace detail {
C4_ALWAYS_INLINE MemoryResource* & get_memory_resource()
{
    /** @todo use a nifty counter:
     * https://en.wikibooks.org/wiki/More_C%2B%2B_Idioms/Nifty_Counter */
    thread_local static MemoryResource* mr = get_memory_resource_malloc();
    return mr;
}
} // namespace detail


//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------

namespace detail {

/** Allows a memory resource to obtain its memory from another memory resource.
 * @ingroup memory_resources */
struct DerivedMemoryResource : public MemoryResource
{
public:

    DerivedMemoryResource(MemoryResource *mr_=nullptr) : m_local(mr_ ? mr_ : get_memory_resource()) {}

private:

    MemoryResource *m_local;

protected:

    virtual void* do_allocate(size_t sz, size_t alignment, void* hint) override
    {
        return m_local->allocate(sz, alignment, hint);
    }

    virtual void* do_reallocate(void* ptr, size_t oldsz, size_t newsz, size_t alignment) override
    {
        return m_local->reallocate(ptr, oldsz, newsz, alignment);
    }

    virtual void do_deallocate(void* ptr, size_t sz, size_t alignment) override
    {
        return m_local->deallocate(ptr, sz, alignment);
    }
};

/** Provides common facilities for memory resource consisting of a single memory block
 * @ingroup memory_resources */
struct _MemoryResourceSingleChunk : public DerivedMemoryResource
{

    C4_NO_COPY_OR_MOVE(_MemoryResourceSingleChunk);

    using impl_type = DerivedMemoryResource;

public:

    _MemoryResourceSingleChunk(MemoryResource *impl=nullptr) : DerivedMemoryResource(impl) { name = "linear_malloc"; }

    /** initialize with owned memory, allocated from the given (or the global) memory resource */
    _MemoryResourceSingleChunk(size_t sz, MemoryResource *impl=nullptr) : _MemoryResourceSingleChunk(impl) { acquire(sz); }
    /** initialize with borrowed memory */
    _MemoryResourceSingleChunk(void *mem, size_t sz) : _MemoryResourceSingleChunk() { acquire(mem, sz); }

    virtual ~_MemoryResourceSingleChunk() override { release(); }

public:

    void const* mem() const { return m_mem; }

    size_t capacity() const { return m_size; }
    size_t size() const { return m_pos; }
    size_t slack() const { C4_ASSERT(m_size >= m_pos); return m_size - m_pos; }

public:

    char  *m_mem{nullptr};
    size_t m_size{0};
    size_t m_pos{0};
    bool   m_owner;

public:

    /** set the internal pointer to the beginning of the linear buffer */
    void clear() { m_pos = 0; }

    /** initialize with owned memory, allocated from the global memory resource */
    void acquire(size_t sz);
    /** initialize with borrowed memory */
    void acquire(void *mem, size_t sz);
    /** release the memory */
    void release();

};

} // namespace detail


//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------
/** provides a linear memory resource. Allocates incrementally from a linear
 * buffer, without ever deallocating. Deallocations are a no-op, and the
 * memory is freed only when the resource is release()d. The memory used by
 * this object can be either owned or borrowed. When borrowed, no calls to
 * malloc/free take place.
 *
 * @ingroup memory_resources */
struct MemoryResourceLinear : public detail::_MemoryResourceSingleChunk
{

    C4_NO_COPY_OR_MOVE(MemoryResourceLinear);

public:

    using detail::_MemoryResourceSingleChunk::_MemoryResourceSingleChunk;

protected:

    virtual void* do_allocate(size_t sz, size_t alignment, void *hint) override;
    virtual void  do_deallocate(void* ptr, size_t sz, size_t alignment) override;
    virtual void* do_reallocate(void* ptr, size_t oldsz, size_t newsz, size_t alignment) override;
};


//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------
/** provides a stack-type malloc-based memory resource.
 * @ingroup memory_resources */
struct MemoryResourceStack : public detail::_MemoryResourceSingleChunk
{

    C4_NO_COPY_OR_MOVE(MemoryResourceStack);

public:

    using detail::_MemoryResourceSingleChunk::_MemoryResourceSingleChunk;

protected:

    virtual void* do_allocate(size_t sz, size_t alignment, void *hint) override;
    virtual void  do_deallocate(void* ptr, size_t sz, size_t alignment) override;
    virtual void* do_reallocate(void* ptr, size_t oldsz, size_t newsz, size_t alignment) override;
};


//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------
/** provides a linear array-based memory resource.
 * @see MemoryResourceLinear
 * @ingroup memory_resources */
template<size_t N>
struct MemoryResourceLinearArr : public MemoryResourceLinear
{
    #ifdef _MSC_VER
    #pragma warning(push)
    #pragma warning(disable: 4324) // structure was padded due to alignment specifier
    #endif
    alignas(alignof(max_align_t)) char m_arr[N];
    #ifdef _MSC_VER
    #pragma warning(pop)
    #endif
    MemoryResourceLinearArr() : MemoryResourceLinear(m_arr, N) { name = "linear_arr"; }
};


//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------
struct AllocationCounts
{
    struct Item
    {
        ssize_t allocs;
        ssize_t size;

        void add(size_t sz)
        {
            ++allocs;
            size += static_cast<ssize_t>(sz);
        }
        void rem(size_t sz)
        {
            --allocs;
            size -= static_cast<ssize_t>(sz);
        }
        Item max(Item const& that) const
        {
            Item r(*this);
            r.allocs = r.allocs > that.allocs ? r.allocs : that.allocs;
            r.size = r.size > that.size ? r.size : that.size;
            return r;
        }
    };

    Item curr  = {0, 0};
    Item total = {0, 0};
    Item max   = {0, 0};

    void clear_counts()
    {
        curr  = {0, 0};
        total = {0, 0};
        max   = {0, 0};
    }

    void update(AllocationCounts const& that)
    {
        curr.allocs += that.curr.allocs;
        curr.size += that.curr.size;
        total.allocs += that.total.allocs;
        total.size += that.total.size;
        max.allocs += that.max.allocs;
        max.size += that.max.size;
    }

    void add_counts(void* ptr, size_t sz)
    {
        if(ptr == nullptr) return;
        curr.add(sz);
        total.add(sz);
        max = max.max(curr);
    }

    void rem_counts(void *ptr, size_t sz)
    {
        if(ptr == nullptr) return;
        curr.rem(sz);
    }

    AllocationCounts operator- (AllocationCounts const& that) const
    {
        AllocationCounts r(*this);
        r.curr.allocs -= that.curr.allocs;
        r.curr.size -= that.curr.size;
        r.total.allocs -= that.total.allocs;
        r.total.size -= that.total.size;
        r.max.allocs -= that.max.allocs;
        r.max.size -= that.max.size;
        return r;
    }

    AllocationCounts operator+ (AllocationCounts const& that) const
    {
        AllocationCounts r(*this);
        r.curr.allocs += that.curr.allocs;
        r.curr.size += that.curr.size;
        r.total.allocs += that.total.allocs;
        r.total.size += that.total.size;
        r.max.allocs += that.max.allocs;
        r.max.size += that.max.size;
        return r;
    }
};


//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------
/** a MemoryResource which latches onto another MemoryResource
 * and counts allocations and sizes.
 * @ingroup memory_resources */
class MemoryResourceCounts : public MemoryResource
{
public:

    MemoryResourceCounts() : m_resource(get_memory_resource())
    {
        C4_ASSERT(m_resource != this);
        name = "MemoryResourceCounts";
    }
    MemoryResourceCounts(MemoryResource *res) : m_resource(res)
    {
        C4_ASSERT(m_resource != this);
        name = "MemoryResourceCounts";
    }

    MemoryResource *resource() { return m_resource; }
    AllocationCounts const& counts() const { return m_counts; }

protected:

    MemoryResource *m_resource;
    AllocationCounts m_counts;

protected:

    virtual void* do_allocate(size_t sz, size_t alignment, void * /*hint*/) override
    {
        void *ptr = m_resource->allocate(sz, alignment);
        m_counts.add_counts(ptr, sz);
        return ptr;
    }

    virtual void  do_deallocate(void* ptr, size_t sz, size_t alignment) override
    {
        m_counts.rem_counts(ptr, sz);
        m_resource->deallocate(ptr, sz, alignment);
    }

    virtual void* do_reallocate(void* ptr, size_t oldsz, size_t newsz, size_t alignment) override
    {
        m_counts.rem_counts(ptr, oldsz);
        void* nptr = m_resource->reallocate(ptr, oldsz, newsz, alignment);
        m_counts.add_counts(nptr, newsz);
        return nptr;
    }

};

//-----------------------------------------------------------------------------
/** RAII class which binds a memory resource with a scope duration.
 * @ingroup memory_resources */
struct ScopedMemoryResource
{
    MemoryResource *m_original;

    ScopedMemoryResource(MemoryResource *r)
    :
        m_original(get_memory_resource())
    {
        set_memory_resource(r);
    }

    ~ScopedMemoryResource()
    {
        set_memory_resource(m_original);
    }
};

//-----------------------------------------------------------------------------
/** RAII class which counts allocations and frees inside a scope. Can
 * optionally set also the memory resource to be used.
 * @ingroup memory_resources */
struct ScopedMemoryResourceCounts
{
    MemoryResourceCounts mr;

    ScopedMemoryResourceCounts() : mr()
    {
        set_memory_resource(&mr);
    }
    ScopedMemoryResourceCounts(MemoryResource *m) : mr(m)
    {
        set_memory_resource(&mr);
    }
    ~ScopedMemoryResourceCounts()
    {
        set_memory_resource(mr.resource());
    }
};

} // namespace c4

#endif /* _C4_MEMORY_RESOURCE_HPP_ */
