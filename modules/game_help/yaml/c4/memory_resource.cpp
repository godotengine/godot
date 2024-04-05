#include "memory_resource.hpp"
#include "memory_util.hpp"

#include <stdlib.h>
#include <string.h>
#if defined(C4_POSIX) || defined(C4_IOS) || defined(C4_MACOS) || defined(C4_ARM)
#   include <errno.h>
#endif
#if defined(C4_ARM)
#   include <malloc.h>
#endif

#include <memory>

namespace c4 {

C4_SUPPRESS_WARNING_GCC_CLANG_WITH_PUSH("-Wold-style-cast")

namespace detail {


#ifdef C4_NO_ALLOC_DEFAULTS
aalloc_pfn s_aalloc = nullptr;
free_pfn s_afree = nullptr;
arealloc_pfn s_arealloc = nullptr;
#else


void afree_impl(void *ptr)
{
#if defined(C4_WIN) || defined(C4_XBOX)
    ::_aligned_free(ptr);
#else
    ::free(ptr);
#endif
}


void* aalloc_impl(size_t size, size_t alignment)
{
    void *mem;
#if defined(C4_WIN) || defined(C4_XBOX)
    mem = ::_aligned_malloc(size, alignment);
    C4_CHECK(mem != nullptr || size == 0);
#elif defined(C4_ARM) || defined(C4_ANDROID)
    // https://stackoverflow.com/questions/53614538/undefined-reference-to-posix-memalign-in-arm-gcc
    // https://electronics.stackexchange.com/questions/467382/e2-studio-undefined-reference-to-posix-memalign/467753
    mem = memalign(alignment, size);
    C4_CHECK(mem != nullptr || size == 0);
#elif defined(C4_POSIX) || defined(C4_IOS) || defined(C4_MACOS)
    // NOTE: alignment needs to be sized in multiples of sizeof(void*)
    size_t amult = alignment;
    if(C4_UNLIKELY(alignment < sizeof(void*)))
    {
        amult = sizeof(void*);
    }
    int ret = ::posix_memalign(&mem, amult, size);
    if(C4_UNLIKELY(ret))
    {
        if(ret == EINVAL)
        {
            C4_ERROR("The alignment argument %zu was not a power of two, "
                     "or was not a multiple of sizeof(void*)", alignment);
        }
        else if(ret == ENOMEM)
        {
            C4_ERROR("There was insufficient memory to fulfill the "
                     "allocation request of %zu bytes (alignment=%lu)", size, size);
        }
        return nullptr;
    }
#else
    (void)size;
    (void)alignment;
    mem = nullptr;
    C4_NOT_IMPLEMENTED_MSG("need to implement an aligned allocation for this platform");
#endif
    C4_ASSERT_MSG((uintptr_t(mem) & (alignment-1)) == 0, "address %p is not aligned to %zu boundary", mem, alignment);
    return mem;
}


void* arealloc_impl(void* ptr, size_t oldsz, size_t newsz, size_t alignment)
{
    /** @todo make this more efficient
     * @see https://stackoverflow.com/questions/9078259/does-realloc-keep-the-memory-alignment-of-posix-memalign
     * @see look for qReallocAligned() in http://code.qt.io/cgit/qt/qtbase.git/tree/src/corelib/global/qmalloc.cpp
     */
    void *tmp = aalloc(newsz, alignment);
    size_t min = newsz < oldsz ? newsz : oldsz;
    if(mem_overlaps(ptr, tmp, oldsz, newsz))
    {
        ::memmove(tmp, ptr, min);
    }
    else
    {
        ::memcpy(tmp, ptr, min);
    }
    afree(ptr);
    return tmp;
}

aalloc_pfn s_aalloc = aalloc_impl;
afree_pfn s_afree = afree_impl;
arealloc_pfn s_arealloc = arealloc_impl;

#endif // C4_NO_ALLOC_DEFAULTS

} // namespace detail


aalloc_pfn get_aalloc()
{
    return detail::s_aalloc;
}
void set_aalloc(aalloc_pfn fn)
{
    detail::s_aalloc = fn;
}

afree_pfn get_afree()
{
    return detail::s_afree;
}
void set_afree(afree_pfn fn)
{
    detail::s_afree = fn;
}

arealloc_pfn get_arealloc()
{
    return detail::s_arealloc;
}
void set_arealloc(arealloc_pfn fn)
{
    detail::s_arealloc = fn;
}


void* aalloc(size_t sz, size_t alignment)
{
    C4_ASSERT_MSG(c4::get_aalloc() != nullptr, "did you forget to call set_aalloc()?");
    auto fn = c4::get_aalloc();
    void* ptr = fn(sz, alignment);
    return ptr;
}

void afree(void* ptr)
{
    C4_ASSERT_MSG(c4::get_afree() != nullptr, "did you forget to call set_afree()?");
    auto fn = c4::get_afree();
    fn(ptr);
}

void* arealloc(void *ptr, size_t oldsz, size_t newsz, size_t alignment)
{
    C4_ASSERT_MSG(c4::get_arealloc() != nullptr, "did you forget to call set_arealloc()?");
    auto fn = c4::get_arealloc();
    void* nptr = fn(ptr, oldsz, newsz, alignment);
    return nptr;
}


//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------

void detail::_MemoryResourceSingleChunk::release()
{
    if(m_mem && m_owner)
    {
        impl_type::deallocate(m_mem, m_size);
    }
    m_mem = nullptr;
    m_size = 0;
    m_owner = false;
    m_pos = 0;
}

void detail::_MemoryResourceSingleChunk::acquire(size_t sz)
{
    clear();
    m_owner = true;
    m_mem = (char*) impl_type::allocate(sz, alignof(max_align_t));
    m_size = sz;
    m_pos = 0;
}

void detail::_MemoryResourceSingleChunk::acquire(void *mem, size_t sz)
{
    clear();
    m_owner = false;
    m_mem = (char*) mem;
    m_size = sz;
    m_pos = 0;
}

//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------

void* MemoryResourceLinear::do_allocate(size_t sz, size_t alignment, void *hint)
{
    C4_UNUSED(hint);
    if(sz == 0) return nullptr;
    // make sure there's enough room to allocate
    if(m_pos + sz > m_size)
    {
        C4_ERROR("out of memory");
        return nullptr;
    }
    void *mem = m_mem + m_pos;
    size_t space = m_size - m_pos;
    if(std::align(alignment, sz, mem, space))
    {
        C4_ASSERT(m_pos <= m_size);
        C4_ASSERT(m_size - m_pos >= space);
        m_pos += (m_size - m_pos) - space;
        m_pos += sz;
        C4_ASSERT(m_pos <= m_size);
    }
    else
    {
        C4_ERROR("could not align memory");
        mem = nullptr;
    }
    return mem;
}

void MemoryResourceLinear::do_deallocate(void* ptr, size_t sz, size_t alignment)
{
    C4_UNUSED(ptr);
    C4_UNUSED(sz);
    C4_UNUSED(alignment);
    // nothing to do!!
}

void* MemoryResourceLinear::do_reallocate(void* ptr, size_t oldsz, size_t newsz, size_t alignment)
{
    if(newsz == oldsz) return ptr;
    // is ptr the most recently allocated (MRA) block?
    char *cptr = (char*)ptr;
    bool same_pos = (m_mem + m_pos == cptr + oldsz);
    // no need to get more memory when shrinking
    if(newsz < oldsz)
    {
        // if this is the MRA, we can safely shrink the position
        if(same_pos)
        {
            m_pos -= oldsz - newsz;
        }
        return ptr;
    }
    // we're growing the block, and it fits in size
    else if(same_pos && cptr + newsz <= m_mem + m_size)
    {
        // if this is the MRA, we can safely shrink the position
        m_pos += newsz - oldsz;
        return ptr;
    }
    // we're growing the block or it doesn't fit -
    // delegate any of these situations to do_deallocate()
    return do_allocate(newsz, alignment, ptr);
}


//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------

/** @todo add a free list allocator. A good candidate because of its
 * small size is TLSF.
 *
 * @see https://github.com/mattconte/tlsf
 *
 * Comparisons:
 *
 * @see https://www.researchgate.net/publication/262375150_A_Comparative_Study_on_Memory_Allocators_in_Multicore_and_Multithreaded_Applications_-_SBESC_2011_-_Presentation_Slides
 * @see http://webkit.sed.hu/blog/20100324/war-allocators-tlsf-action
 * @see https://github.com/emeryberger/Malloc-Implementations/tree/master/allocators
 *
 * */

C4_SUPPRESS_WARNING_GCC_CLANG_POP

} // namespace c4


//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------

#ifdef C4_REDEFINE_CPPNEW
#include <new>
void* operator new(size_t size)
{
    auto *mr = ::c4::get_memory_resource();
    return mr->allocate(size);
}
void operator delete(void *p) noexcept
{
    C4_NEVER_REACH();
}
void operator delete(void *p, size_t size)
{
    auto *mr = ::c4::get_memory_resource();
    mr->deallocate(p, size);
}
void* operator new[](size_t size)
{
    return operator new(size);
}
void operator delete[](void *p) noexcept
{
    operator delete(p);
}
void operator delete[](void *p, size_t size)
{
    operator delete(p, size);
}
void* operator new(size_t size, std::nothrow_t)
{
    return operator new(size);
}
void operator delete(void *p, std::nothrow_t)
{
    operator delete(p);
}
void operator delete(void *p, size_t size, std::nothrow_t)
{
    operator delete(p, size);
}
void* operator new[](size_t size, std::nothrow_t)
{
    return operator new(size);
}
void operator delete[](void *p, std::nothrow_t)
{
    operator delete(p);
}
void operator delete[](void *p, size_t, std::nothrow_t)
{
    operator delete(p, size);
}
#endif // C4_REDEFINE_CPPNEW
