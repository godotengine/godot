/*
    Copyright (c) 2005-2020 Intel Corporation

    Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at

        http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.
*/

#ifndef __TBB_cache_aligned_allocator_H
#define __TBB_cache_aligned_allocator_H

#include <new>
#include "tbb_stddef.h"
#if __TBB_ALLOCATOR_CONSTRUCT_VARIADIC
#include <utility> // std::forward
#endif

#if __TBB_CPP17_MEMORY_RESOURCE_PRESENT
#include <memory_resource>
#endif

namespace tbb {

//! @cond INTERNAL
namespace internal {
    //! Cache/sector line size.
    /** @ingroup memory_allocation */
    size_t __TBB_EXPORTED_FUNC NFS_GetLineSize();

    //! Allocate memory on cache/sector line boundary.
    /** @ingroup memory_allocation */
    void* __TBB_EXPORTED_FUNC NFS_Allocate( size_t n_element, size_t element_size, void* hint );

    //! Free memory allocated by NFS_Allocate.
    /** Freeing a NULL pointer is allowed, but has no effect.
        @ingroup memory_allocation */
    void __TBB_EXPORTED_FUNC NFS_Free( void* );
}
//! @endcond

#if _MSC_VER && !defined(__INTEL_COMPILER)
    // Workaround for erroneous "unreferenced parameter" warning in method destroy.
    #pragma warning (push)
    #pragma warning (disable: 4100)
#endif

//! Meets "allocator" requirements of ISO C++ Standard, Section 20.1.5
/** The members are ordered the same way they are in section 20.4.1
    of the ISO C++ standard.
    @ingroup memory_allocation */
template<typename T>
class cache_aligned_allocator {
public:
    typedef typename internal::allocator_type<T>::value_type value_type;
    typedef value_type* pointer;
    typedef const value_type* const_pointer;
    typedef value_type& reference;
    typedef const value_type& const_reference;
    typedef size_t size_type;
    typedef ptrdiff_t difference_type;
    template<typename U> struct rebind {
        typedef cache_aligned_allocator<U> other;
    };
    cache_aligned_allocator() throw() {}
    cache_aligned_allocator( const cache_aligned_allocator& ) throw() {}
    template<typename U> cache_aligned_allocator(const cache_aligned_allocator<U>&) throw() {}

    pointer address(reference x) const {return &x;}
    const_pointer address(const_reference x) const {return &x;}

    //! Allocate space for n objects, starting on a cache/sector line.
    pointer allocate( size_type n, const void* hint=0 ) {
        // The "hint" argument is always ignored in NFS_Allocate thus const_cast shouldn't hurt
        return pointer(internal::NFS_Allocate( n, sizeof(value_type), const_cast<void*>(hint) ));
    }

    //! Free block of memory that starts on a cache line
    void deallocate( pointer p, size_type ) {
        internal::NFS_Free(p);
    }

    //! Largest value for which method allocate might succeed.
    size_type max_size() const throw() {
        return (~size_t(0)-internal::NFS_MaxLineSize)/sizeof(value_type);
    }

    //! Copy-construct value at location pointed to by p.
#if __TBB_ALLOCATOR_CONSTRUCT_VARIADIC
    template<typename U, typename... Args>
    void construct(U *p, Args&&... args)
        { ::new((void *)p) U(std::forward<Args>(args)...); }
#else // __TBB_ALLOCATOR_CONSTRUCT_VARIADIC
#if __TBB_CPP11_RVALUE_REF_PRESENT
    void construct( pointer p, value_type&& value ) {::new((void*)(p)) value_type(std::move(value));}
#endif
    void construct( pointer p, const value_type& value ) {::new((void*)(p)) value_type(value);}
#endif // __TBB_ALLOCATOR_CONSTRUCT_VARIADIC

    //! Destroy value at location pointed to by p.
    void destroy( pointer p ) {p->~value_type();}
};

#if _MSC_VER && !defined(__INTEL_COMPILER)
    #pragma warning (pop)
#endif // warning 4100 is back

//! Analogous to std::allocator<void>, as defined in ISO C++ Standard, Section 20.4.1
/** @ingroup memory_allocation */
template<>
class cache_aligned_allocator<void> {
public:
    typedef void* pointer;
    typedef const void* const_pointer;
    typedef void value_type;
    template<typename U> struct rebind {
        typedef cache_aligned_allocator<U> other;
    };
};

template<typename T, typename U>
inline bool operator==( const cache_aligned_allocator<T>&, const cache_aligned_allocator<U>& ) {return true;}

template<typename T, typename U>
inline bool operator!=( const cache_aligned_allocator<T>&, const cache_aligned_allocator<U>& ) {return false;}

#if __TBB_CPP17_MEMORY_RESOURCE_PRESENT

//! C++17 memory resource wrapper to ensure cache line size alignment
class cache_aligned_resource : public std::pmr::memory_resource {
public:
    cache_aligned_resource() : cache_aligned_resource(std::pmr::get_default_resource()) {}
    explicit cache_aligned_resource(std::pmr::memory_resource* upstream) : m_upstream(upstream) {}

    std::pmr::memory_resource* upstream_resource() const {
        return m_upstream;
    }

private:
    //! We don't know what memory resource set. Use padding to guarantee alignment
    void* do_allocate(size_t bytes, size_t alignment) override {
        size_t cache_line_alignment = correct_alignment(alignment);
        uintptr_t base = (uintptr_t)m_upstream->allocate(correct_size(bytes) + cache_line_alignment);
        __TBB_ASSERT(base != 0, "Upstream resource returned NULL.");
#if _MSC_VER && !defined(__INTEL_COMPILER)
    // unary minus operator applied to unsigned type, result still unsigned
    #pragma warning(push)
    #pragma warning(disable: 4146 4706)
#endif
        // Round up to the next cache line (align the base address)
        uintptr_t result = (base + cache_line_alignment) & -cache_line_alignment;
#if _MSC_VER && !defined(__INTEL_COMPILER)
    #pragma warning(pop)
#endif
        // Record where block actually starts.
        ((uintptr_t*)result)[-1] = base;
        return (void*)result;
    }

    void do_deallocate(void* ptr, size_t bytes, size_t alignment) override {
        if (ptr) {
            // Recover where block actually starts
            uintptr_t base = ((uintptr_t*)ptr)[-1];
            m_upstream->deallocate((void*)base, correct_size(bytes) + correct_alignment(alignment));
        }
    }

    bool do_is_equal(const std::pmr::memory_resource& other) const noexcept override {
        if (this == &other) { return true; }
#if __TBB_USE_OPTIONAL_RTTI
        const cache_aligned_resource* other_res = dynamic_cast<const cache_aligned_resource*>(&other);
        return other_res && (this->upstream_resource() == other_res->upstream_resource());
#else
        return false;
#endif
    }

    size_t correct_alignment(size_t alignment) {
        __TBB_ASSERT(tbb::internal::is_power_of_two(alignment), "Alignment is not a power of 2");
#if __TBB_CPP17_HW_INTERFERENCE_SIZE_PRESENT
        size_t cache_line_size = std::hardware_destructive_interference_size;
#else
        size_t cache_line_size = internal::NFS_GetLineSize();
#endif
        return alignment < cache_line_size ? cache_line_size : alignment;
    }

    size_t correct_size(size_t bytes) {
        // To handle the case, when small size requested. There could be not
        // enough space to store the original pointer.
        return bytes < sizeof(uintptr_t) ? sizeof(uintptr_t) : bytes;
    }

    std::pmr::memory_resource* m_upstream;
};

#endif /* __TBB_CPP17_MEMORY_RESOURCE_PRESENT */

} // namespace tbb

#endif /* __TBB_cache_aligned_allocator_H */

