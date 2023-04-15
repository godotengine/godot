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

#ifndef __TBB_scalable_allocator_H
#define __TBB_scalable_allocator_H
/** @file */

#include <stddef.h> /* Need ptrdiff_t and size_t from here. */
#if !_MSC_VER
#include <stdint.h> /* Need intptr_t from here. */
#endif

#if !defined(__cplusplus) && __ICC==1100
    #pragma warning (push)
    #pragma warning (disable: 991)
#endif

#ifdef __cplusplus
extern "C" {
#endif /* __cplusplus */

#if _MSC_VER >= 1400
#define __TBB_EXPORTED_FUNC   __cdecl
#else
#define __TBB_EXPORTED_FUNC
#endif

/** The "malloc" analogue to allocate block of memory of size bytes.
  * @ingroup memory_allocation */
void * __TBB_EXPORTED_FUNC scalable_malloc (size_t size);

/** The "free" analogue to discard a previously allocated piece of memory.
    @ingroup memory_allocation */
void   __TBB_EXPORTED_FUNC scalable_free (void* ptr);

/** The "realloc" analogue complementing scalable_malloc.
    @ingroup memory_allocation */
void * __TBB_EXPORTED_FUNC scalable_realloc (void* ptr, size_t size);

/** The "calloc" analogue complementing scalable_malloc.
    @ingroup memory_allocation */
void * __TBB_EXPORTED_FUNC scalable_calloc (size_t nobj, size_t size);

/** The "posix_memalign" analogue.
    @ingroup memory_allocation */
int __TBB_EXPORTED_FUNC scalable_posix_memalign (void** memptr, size_t alignment, size_t size);

/** The "_aligned_malloc" analogue.
    @ingroup memory_allocation */
void * __TBB_EXPORTED_FUNC scalable_aligned_malloc (size_t size, size_t alignment);

/** The "_aligned_realloc" analogue.
    @ingroup memory_allocation */
void * __TBB_EXPORTED_FUNC scalable_aligned_realloc (void* ptr, size_t size, size_t alignment);

/** The "_aligned_free" analogue.
    @ingroup memory_allocation */
void __TBB_EXPORTED_FUNC scalable_aligned_free (void* ptr);

/** The analogue of _msize/malloc_size/malloc_usable_size.
    Returns the usable size of a memory block previously allocated by scalable_*,
    or 0 (zero) if ptr does not point to such a block.
    @ingroup memory_allocation */
size_t __TBB_EXPORTED_FUNC scalable_msize (void* ptr);

/* Results for scalable_allocation_* functions */
typedef enum {
    TBBMALLOC_OK,
    TBBMALLOC_INVALID_PARAM,
    TBBMALLOC_UNSUPPORTED,
    TBBMALLOC_NO_MEMORY,
    TBBMALLOC_NO_EFFECT
} ScalableAllocationResult;

/* Setting TBB_MALLOC_USE_HUGE_PAGES environment variable to 1 enables huge pages.
   scalable_allocation_mode call has priority over environment variable. */
typedef enum {
    TBBMALLOC_USE_HUGE_PAGES,  /* value turns using huge pages on and off */
    /* deprecated, kept for backward compatibility only */
    USE_HUGE_PAGES = TBBMALLOC_USE_HUGE_PAGES,
    /* try to limit memory consumption value (Bytes), clean internal buffers
       if limit is exceeded, but not prevents from requesting memory from OS */
    TBBMALLOC_SET_SOFT_HEAP_LIMIT,
    /* Lower bound for the size (Bytes), that is interpreted as huge
     * and not released during regular cleanup operations. */
    TBBMALLOC_SET_HUGE_SIZE_THRESHOLD
} AllocationModeParam;

/** Set TBB allocator-specific allocation modes.
    @ingroup memory_allocation */
int __TBB_EXPORTED_FUNC scalable_allocation_mode(int param, intptr_t value);

typedef enum {
    /* Clean internal allocator buffers for all threads.
       Returns TBBMALLOC_NO_EFFECT if no buffers cleaned,
       TBBMALLOC_OK if some memory released from buffers. */
    TBBMALLOC_CLEAN_ALL_BUFFERS,
    /* Clean internal allocator buffer for current thread only.
       Return values same as for TBBMALLOC_CLEAN_ALL_BUFFERS. */
    TBBMALLOC_CLEAN_THREAD_BUFFERS
} ScalableAllocationCmd;

/** Call TBB allocator-specific commands.
    @ingroup memory_allocation */
int __TBB_EXPORTED_FUNC scalable_allocation_command(int cmd, void *param);

#ifdef __cplusplus
} /* extern "C" */
#endif /* __cplusplus */

#ifdef __cplusplus

//! The namespace rml contains components of low-level memory pool interface.
namespace rml {
class MemoryPool;

typedef void *(*rawAllocType)(intptr_t pool_id, size_t &bytes);
// returns non-zero in case of error
typedef int   (*rawFreeType)(intptr_t pool_id, void* raw_ptr, size_t raw_bytes);

/*
MemPoolPolicy extension must be compatible with such structure fields layout

struct MemPoolPolicy {
    rawAllocType pAlloc;
    rawFreeType  pFree;
    size_t       granularity;   // granularity of pAlloc allocations
};
*/

struct MemPoolPolicy {
    enum {
        TBBMALLOC_POOL_VERSION = 1
    };

    rawAllocType pAlloc;
    rawFreeType  pFree;
                 // granularity of pAlloc allocations. 0 means default used.
    size_t       granularity;
    int          version;
                 // all memory consumed at 1st pAlloc call and never returned,
                 // no more pAlloc calls after 1st
    unsigned     fixedPool : 1,
                 // memory consumed but returned only at pool termination
                 keepAllMemory : 1,
                 reserved : 30;

    MemPoolPolicy(rawAllocType pAlloc_, rawFreeType pFree_,
                  size_t granularity_ = 0, bool fixedPool_ = false,
                  bool keepAllMemory_ = false) :
        pAlloc(pAlloc_), pFree(pFree_), granularity(granularity_), version(TBBMALLOC_POOL_VERSION),
        fixedPool(fixedPool_), keepAllMemory(keepAllMemory_),
        reserved(0) {}
};

// enums have same values as appropriate enums from ScalableAllocationResult
// TODO: use ScalableAllocationResult in pool_create directly
enum MemPoolError {
    // pool created successfully
    POOL_OK = TBBMALLOC_OK,
    // invalid policy parameters found
    INVALID_POLICY = TBBMALLOC_INVALID_PARAM,
     // requested pool policy is not supported by allocator library
    UNSUPPORTED_POLICY = TBBMALLOC_UNSUPPORTED,
    // lack of memory during pool creation
    NO_MEMORY = TBBMALLOC_NO_MEMORY,
    // action takes no effect
    NO_EFFECT = TBBMALLOC_NO_EFFECT
};

MemPoolError pool_create_v1(intptr_t pool_id, const MemPoolPolicy *policy,
                            rml::MemoryPool **pool);

bool  pool_destroy(MemoryPool* memPool);
void *pool_malloc(MemoryPool* memPool, size_t size);
void *pool_realloc(MemoryPool* memPool, void *object, size_t size);
void *pool_aligned_malloc(MemoryPool* mPool, size_t size, size_t alignment);
void *pool_aligned_realloc(MemoryPool* mPool, void *ptr, size_t size, size_t alignment);
bool  pool_reset(MemoryPool* memPool);
bool  pool_free(MemoryPool *memPool, void *object);
MemoryPool *pool_identify(void *object);
size_t pool_msize(MemoryPool *memPool, void *object);

} // namespace rml

#include <new>      /* To use new with the placement argument */

/* Ensure that including this header does not cause implicit linkage with TBB */
#ifndef __TBB_NO_IMPLICIT_LINKAGE
    #define __TBB_NO_IMPLICIT_LINKAGE 1
    #include "tbb_stddef.h"
    #undef  __TBB_NO_IMPLICIT_LINKAGE
#else
    #include "tbb_stddef.h"
#endif

#if __TBB_ALLOCATOR_CONSTRUCT_VARIADIC
#include <utility> // std::forward
#endif

#if __TBB_CPP17_MEMORY_RESOURCE_PRESENT
#include <memory_resource>
#endif

namespace tbb {

#if _MSC_VER && !defined(__INTEL_COMPILER)
    // Workaround for erroneous "unreferenced parameter" warning in method destroy.
    #pragma warning (push)
    #pragma warning (disable: 4100)
#endif

//! @cond INTERNAL
namespace internal {

#if TBB_USE_EXCEPTIONS
// forward declaration is for inlining prevention
template<typename E> __TBB_NOINLINE( void throw_exception(const E &e) );
#endif

// keep throw in a separate function to prevent code bloat
template<typename E>
void throw_exception(const E &e) {
    __TBB_THROW(e);
}

} // namespace internal
//! @endcond

//! Meets "allocator" requirements of ISO C++ Standard, Section 20.1.5
/** The members are ordered the same way they are in section 20.4.1
    of the ISO C++ standard.
    @ingroup memory_allocation */
template<typename T>
class scalable_allocator {
public:
    typedef typename internal::allocator_type<T>::value_type value_type;
    typedef value_type* pointer;
    typedef const value_type* const_pointer;
    typedef value_type& reference;
    typedef const value_type& const_reference;
    typedef size_t size_type;
    typedef ptrdiff_t difference_type;
    template<class U> struct rebind {
        typedef scalable_allocator<U> other;
    };

    scalable_allocator() throw() {}
    scalable_allocator( const scalable_allocator& ) throw() {}
    template<typename U> scalable_allocator(const scalable_allocator<U>&) throw() {}

    pointer address(reference x) const {return &x;}
    const_pointer address(const_reference x) const {return &x;}

    //! Allocate space for n objects.
    pointer allocate( size_type n, const void* /*hint*/ =0 ) {
        pointer p = static_cast<pointer>( scalable_malloc( n * sizeof(value_type) ) );
        if (!p)
            internal::throw_exception(std::bad_alloc());
        return p;
    }

    //! Free previously allocated block of memory
    void deallocate( pointer p, size_type ) {
        scalable_free( p );
    }

    //! Largest value for which method allocate might succeed.
    size_type max_size() const throw() {
        size_type absolutemax = static_cast<size_type>(-1) / sizeof (value_type);
        return (absolutemax > 0 ? absolutemax : 1);
    }
#if __TBB_ALLOCATOR_CONSTRUCT_VARIADIC
    template<typename U, typename... Args>
    void construct(U *p, Args&&... args)
        { ::new((void *)p) U(std::forward<Args>(args)...); }
#else /* __TBB_ALLOCATOR_CONSTRUCT_VARIADIC */
#if __TBB_CPP11_RVALUE_REF_PRESENT
    void construct( pointer p, value_type&& value ) { ::new((void*)(p)) value_type( std::move( value ) ); }
#endif
    void construct( pointer p, const value_type& value ) {::new((void*)(p)) value_type(value);}
#endif /* __TBB_ALLOCATOR_CONSTRUCT_VARIADIC */
    void destroy( pointer p ) {p->~value_type();}
};

#if _MSC_VER && !defined(__INTEL_COMPILER)
    #pragma warning (pop)
#endif /* warning 4100 is back */

//! Analogous to std::allocator<void>, as defined in ISO C++ Standard, Section 20.4.1
/** @ingroup memory_allocation */
template<>
class scalable_allocator<void> {
public:
    typedef void* pointer;
    typedef const void* const_pointer;
    typedef void value_type;
    template<class U> struct rebind {
        typedef scalable_allocator<U> other;
    };
};

template<typename T, typename U>
inline bool operator==( const scalable_allocator<T>&, const scalable_allocator<U>& ) {return true;}

template<typename T, typename U>
inline bool operator!=( const scalable_allocator<T>&, const scalable_allocator<U>& ) {return false;}

#if __TBB_CPP17_MEMORY_RESOURCE_PRESENT

namespace internal {

//! C++17 memory resource implementation for scalable allocator
//! ISO C++ Section 23.12.2
class scalable_resource_impl : public std::pmr::memory_resource {
private:
    void* do_allocate(size_t bytes, size_t alignment) override {
        void* ptr = scalable_aligned_malloc( bytes, alignment );
        if (!ptr) {
            throw_exception(std::bad_alloc());
        }
        return ptr;
    }

    void do_deallocate(void* ptr, size_t /*bytes*/, size_t /*alignment*/) override {
        scalable_free(ptr);
    }

    //! Memory allocated by one instance of scalable_resource_impl could be deallocated by any
    //! other instance of this class
    bool do_is_equal(const std::pmr::memory_resource& other) const noexcept override {
        return this == &other ||
#if __TBB_USE_OPTIONAL_RTTI
            dynamic_cast<const scalable_resource_impl*>(&other) != NULL;
#else
            false;
#endif
    }
};

} // namespace internal

//! Global scalable allocator memory resource provider
inline std::pmr::memory_resource* scalable_memory_resource() noexcept {
    static tbb::internal::scalable_resource_impl scalable_res;
    return &scalable_res;
}

#endif /* __TBB_CPP17_MEMORY_RESOURCE_PRESENT */

} // namespace tbb

#if _MSC_VER
    #if (__TBB_BUILD || __TBBMALLOC_BUILD) && !defined(__TBBMALLOC_NO_IMPLICIT_LINKAGE)
        #define __TBBMALLOC_NO_IMPLICIT_LINKAGE 1
    #endif

    #if !__TBBMALLOC_NO_IMPLICIT_LINKAGE
        #ifdef _DEBUG
            #pragma comment(lib, "tbbmalloc_debug.lib")
        #else
            #pragma comment(lib, "tbbmalloc.lib")
        #endif
    #endif


#endif

#endif /* __cplusplus */

#if !defined(__cplusplus) && __ICC==1100
    #pragma warning (pop)
#endif /* ICC 11.0 warning 991 is back */

#endif /* __TBB_scalable_allocator_H */
