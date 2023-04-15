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

#ifndef __TBB_memory_pool_H
#define __TBB_memory_pool_H

#if !TBB_PREVIEW_MEMORY_POOL
#error Set TBB_PREVIEW_MEMORY_POOL to include memory_pool.h
#endif
/** @file */

#include "scalable_allocator.h"
#include <new> // std::bad_alloc
#include <stdexcept> // std::runtime_error, std::invalid_argument
// required in C++03 to construct std::runtime_error and std::invalid_argument
#include <string>
#if __TBB_ALLOCATOR_CONSTRUCT_VARIADIC
#include <utility> // std::forward
#endif

#if __TBB_EXTRA_DEBUG
#define __TBBMALLOC_ASSERT ASSERT
#else
#define __TBBMALLOC_ASSERT(a,b) ((void)0)
#endif

namespace tbb {
namespace interface6 {
//! @cond INTERNAL
namespace internal {

//! Base of thread-safe pool allocator for variable-size requests
class pool_base : tbb::internal::no_copy {
    // Pool interface is separate from standard allocator classes because it has
    // to maintain internal state, no copy or assignment. Move and swap are possible.
public:
    //! Reset pool to reuse its memory (free all objects at once)
    void recycle() { rml::pool_reset(my_pool); }

    //! The "malloc" analogue to allocate block of memory of size bytes
    void *malloc(size_t size) { return rml::pool_malloc(my_pool, size); }

    //! The "free" analogue to discard a previously allocated piece of memory.
    void free(void* ptr) { rml::pool_free(my_pool, ptr); }

    //! The "realloc" analogue complementing pool_malloc.
    // Enables some low-level optimization possibilities
    void *realloc(void* ptr, size_t size) {
        return rml::pool_realloc(my_pool, ptr, size);
    }

protected:
    //! destroy pool - must be called in a child class
    void destroy() { rml::pool_destroy(my_pool); }

    rml::MemoryPool *my_pool;
};

} // namespace internal
//! @endcond

#if _MSC_VER && !defined(__INTEL_COMPILER)
    // Workaround for erroneous "unreferenced parameter" warning in method destroy.
    #pragma warning (push)
    #pragma warning (disable: 4100)
#endif

//! Meets "allocator" requirements of ISO C++ Standard, Section 20.1.5
/** @ingroup memory_allocation */
template<typename T, typename P = internal::pool_base>
class memory_pool_allocator {
protected:
    typedef P pool_type;
    pool_type *my_pool;
    template<typename U, typename R>
    friend class memory_pool_allocator;
    template<typename V, typename U, typename R>
    friend bool operator==( const memory_pool_allocator<V,R>& a, const memory_pool_allocator<U,R>& b);
    template<typename V, typename U, typename R>
    friend bool operator!=( const memory_pool_allocator<V,R>& a, const memory_pool_allocator<U,R>& b);
public:
    typedef typename tbb::internal::allocator_type<T>::value_type value_type;
    typedef value_type* pointer;
    typedef const value_type* const_pointer;
    typedef value_type& reference;
    typedef const value_type& const_reference;
    typedef size_t size_type;
    typedef ptrdiff_t difference_type;
    template<typename U> struct rebind {
        typedef memory_pool_allocator<U, P> other;
    };

    explicit memory_pool_allocator(pool_type &pool) throw() : my_pool(&pool) {}
    memory_pool_allocator(const memory_pool_allocator& src) throw() : my_pool(src.my_pool) {}
    template<typename U>
    memory_pool_allocator(const memory_pool_allocator<U,P>& src) throw() : my_pool(src.my_pool) {}

    pointer address(reference x) const { return &x; }
    const_pointer address(const_reference x) const { return &x; }

    //! Allocate space for n objects.
    pointer allocate( size_type n, const void* /*hint*/ = 0) {
        pointer p = static_cast<pointer>( my_pool->malloc( n*sizeof(value_type) ) );
        if (!p)
            tbb::internal::throw_exception(std::bad_alloc());
        return p;
    }
    //! Free previously allocated block of memory.
    void deallocate( pointer p, size_type ) {
        my_pool->free(p);
    }
    //! Largest value for which method allocate might succeed.
    size_type max_size() const throw() {
        size_type max = static_cast<size_type>(-1) / sizeof (value_type);
        return (max > 0 ? max : 1);
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
    void construct( pointer p, const value_type& value ) { ::new((void*)(p)) value_type(value); }
#endif // __TBB_ALLOCATOR_CONSTRUCT_VARIADIC

    //! Destroy value at location pointed to by p.
    void destroy( pointer p ) { p->~value_type(); }

};

#if _MSC_VER && !defined(__INTEL_COMPILER)
    #pragma warning (pop)
#endif // warning 4100 is back

//! Analogous to std::allocator<void>, as defined in ISO C++ Standard, Section 20.4.1
/** @ingroup memory_allocation */
template<typename P>
class memory_pool_allocator<void, P> {
public:
    typedef P pool_type;
    typedef void* pointer;
    typedef const void* const_pointer;
    typedef void value_type;
    template<typename U> struct rebind {
        typedef memory_pool_allocator<U, P> other;
    };

    explicit memory_pool_allocator( pool_type &pool) throw() : my_pool(&pool) {}
    memory_pool_allocator( const memory_pool_allocator& src) throw() : my_pool(src.my_pool) {}
    template<typename U>
    memory_pool_allocator(const memory_pool_allocator<U,P>& src) throw() : my_pool(src.my_pool) {}

protected:
    pool_type *my_pool;
    template<typename U, typename R>
    friend class memory_pool_allocator;
    template<typename V, typename U, typename R>
    friend bool operator==( const memory_pool_allocator<V,R>& a, const memory_pool_allocator<U,R>& b);
    template<typename V, typename U, typename R>
    friend bool operator!=( const memory_pool_allocator<V,R>& a, const memory_pool_allocator<U,R>& b);
};

template<typename T, typename U, typename P>
inline bool operator==( const memory_pool_allocator<T,P>& a, const memory_pool_allocator<U,P>& b) {return a.my_pool==b.my_pool;}

template<typename T, typename U, typename P>
inline bool operator!=( const memory_pool_allocator<T,P>& a, const memory_pool_allocator<U,P>& b) {return a.my_pool!=b.my_pool;}


//! Thread-safe growable pool allocator for variable-size requests
template <typename Alloc>
class memory_pool : public internal::pool_base {
    Alloc my_alloc; // TODO: base-class optimization
    static void *allocate_request(intptr_t pool_id, size_t & bytes);
    static int deallocate_request(intptr_t pool_id, void*, size_t raw_bytes);

public:
    //! construct pool with underlying allocator
    explicit memory_pool(const Alloc &src = Alloc());

    //! destroy pool
    ~memory_pool() { destroy(); } // call the callbacks first and destroy my_alloc latter

};

class fixed_pool : public internal::pool_base {
    void *my_buffer;
    size_t my_size;
    inline static void *allocate_request(intptr_t pool_id, size_t & bytes);

public:
    //! construct pool with underlying allocator
    inline fixed_pool(void *buf, size_t size);
    //! destroy pool
    ~fixed_pool() { destroy(); }
};

//////////////// Implementation ///////////////

template <typename Alloc>
memory_pool<Alloc>::memory_pool(const Alloc &src) : my_alloc(src) {
    rml::MemPoolPolicy args(allocate_request, deallocate_request,
                            sizeof(typename Alloc::value_type));
    rml::MemPoolError res = rml::pool_create_v1(intptr_t(this), &args, &my_pool);
    if (res!=rml::POOL_OK)
        tbb::internal::throw_exception(std::runtime_error("Can't create pool"));
}
template <typename Alloc>
void *memory_pool<Alloc>::allocate_request(intptr_t pool_id, size_t & bytes) {
    memory_pool<Alloc> &self = *reinterpret_cast<memory_pool<Alloc>*>(pool_id);
    const size_t unit_size = sizeof(typename Alloc::value_type);
    __TBBMALLOC_ASSERT( 0 == bytes%unit_size, NULL);
    void *ptr;
    __TBB_TRY { ptr = self.my_alloc.allocate( bytes/unit_size ); }
    __TBB_CATCH(...) { return 0; }
    return ptr;
}
#if __TBB_MSVC_UNREACHABLE_CODE_IGNORED
    // Workaround for erroneous "unreachable code" warning in the template below.
    // Specific for VC++ 17-18 compiler
    #pragma warning (push)
    #pragma warning (disable: 4702)
#endif
template <typename Alloc>
int memory_pool<Alloc>::deallocate_request(intptr_t pool_id, void* raw_ptr, size_t raw_bytes) {
    memory_pool<Alloc> &self = *reinterpret_cast<memory_pool<Alloc>*>(pool_id);
    const size_t unit_size = sizeof(typename Alloc::value_type);
    __TBBMALLOC_ASSERT( 0 == raw_bytes%unit_size, NULL);
    self.my_alloc.deallocate( static_cast<typename Alloc::value_type*>(raw_ptr), raw_bytes/unit_size );
    return 0;
}
#if __TBB_MSVC_UNREACHABLE_CODE_IGNORED
    #pragma warning (pop)
#endif
inline fixed_pool::fixed_pool(void *buf, size_t size) : my_buffer(buf), my_size(size) {
    if (!buf || !size)
        // TODO: improve support for mode with exceptions disabled
        tbb::internal::throw_exception(std::invalid_argument("Zero in parameter is invalid"));
    rml::MemPoolPolicy args(allocate_request, 0, size, /*fixedPool=*/true);
    rml::MemPoolError res = rml::pool_create_v1(intptr_t(this), &args, &my_pool);
    if (res!=rml::POOL_OK)
        tbb::internal::throw_exception(std::runtime_error("Can't create pool"));
}
inline void *fixed_pool::allocate_request(intptr_t pool_id, size_t & bytes) {
    fixed_pool &self = *reinterpret_cast<fixed_pool*>(pool_id);
    __TBBMALLOC_ASSERT(0 != self.my_size, "The buffer must not be used twice.");
    bytes = self.my_size;
    self.my_size = 0; // remember that buffer has been used
    return self.my_buffer;
}

} //namespace interface6
using interface6::memory_pool_allocator;
using interface6::memory_pool;
using interface6::fixed_pool;
} //namespace tbb

#undef __TBBMALLOC_ASSERT
#endif// __TBB_memory_pool_H
