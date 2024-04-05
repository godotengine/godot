#ifndef _C4_ALLOCATOR_HPP_
#define _C4_ALLOCATOR_HPP_

#include "memory_resource.hpp"
#include "ctor_dtor.hpp"

#include <memory> // std::allocator_traits
#include <type_traits>

/** @file allocator.hpp Contains classes to make typeful allocations (note
 * that memory resources are typeless) */

/** @defgroup mem_res_providers Memory resource providers
 * @brief Policy classes which provide a memory resource for
 * use in an allocator.
 * @ingroup memory
 */

/** @defgroup allocators Allocators
 * @brief Lightweight classes that act as handles to specific memory
 * resources and provide typeful memory.
 * @ingroup memory
 */

namespace c4 {

C4_SUPPRESS_WARNING_GCC_CLANG_WITH_PUSH("-Wold-style-cast")

namespace detail {
template<class T> inline size_t size_for      (size_t num_objs) noexcept { return num_objs * sizeof(T); }
template<       > inline size_t size_for<void>(size_t num_objs) noexcept { return num_objs;             }
} // namespace detail


//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------

/** provides a per-allocator memory resource
 * @ingroup mem_res_providers */
class MemRes
{
public:

    MemRes() : m_resource(get_memory_resource()) {}
    MemRes(MemoryResource* r) noexcept : m_resource(r ? r : get_memory_resource()) {}

    inline MemoryResource* resource() const { return m_resource; }

private:

    MemoryResource* m_resource;

};


/** the allocators using this will default to the global memory resource
 * @ingroup mem_res_providers */
class MemResGlobal
{
public:

    MemResGlobal() {}
    MemResGlobal(MemoryResource* r) noexcept { C4_UNUSED(r); C4_ASSERT(r == get_memory_resource()); }

    inline MemoryResource* resource() const { return get_memory_resource(); }
};


//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------

namespace detail {
template<class MemRes>
struct _AllocatorUtil;

template<class T, class ...Args>
struct has_no_alloc
    : public std::integral_constant<bool,
                                    !(std::uses_allocator<T, MemoryResource*>::value)
                                    && std::is_constructible<T, Args...>::value> {};

// std::uses_allocator_v<U, MemoryResource> && std::is_constructible<U, std::allocator_arg_t, MemoryResource*, Args...>
// ie can construct(std::allocator_arg_t, MemoryResource*, Args...)
template<class T, class ...Args>
struct has_alloc_arg
    : public std::integral_constant<bool,
                                    std::uses_allocator<T, MemoryResource*>::value
                                    && std::is_constructible<T, std::allocator_arg_t, MemoryResource*, Args...>::value> {};
// std::uses_allocator<U> && std::is_constructible<U, Args..., MemoryResource*>
// ie, can construct(Args..., MemoryResource*)
template<class T, class ...Args>
struct has_alloc
    : public std::integral_constant<bool,
                                    std::uses_allocator<T, MemoryResource*>::value
                                    && std::is_constructible<T, Args..., MemoryResource*>::value> {};

} // namespace detail


template<class MemRes>
struct detail::_AllocatorUtil : public MemRes
{
    using MemRes::MemRes;

    /** for construct:
     * @see http://en.cppreference.com/w/cpp/experimental/polymorphic_allocator/construct */

    // 1. types with no allocators
    template <class U, class... Args>
    C4_ALWAYS_INLINE typename std::enable_if<detail::has_no_alloc<U, Args...>::value, void>::type
    construct(U *ptr, Args &&...args)
    {
        c4::construct(ptr, std::forward<Args>(args)...);
    }
    template<class U, class I, class... Args>
    C4_ALWAYS_INLINE typename std::enable_if<detail::has_no_alloc<U, Args...>::value, void>::type
    construct_n(U* ptr, I n, Args&&... args)
    {
        c4::construct_n(ptr, n, std::forward<Args>(args)...);
    }

    // 2. types using allocators (ie, containers)

    // 2.1. can construct(std::allocator_arg_t, MemoryResource*, Args...)
    template<class U, class... Args>
    C4_ALWAYS_INLINE typename std::enable_if<detail::has_alloc_arg<U, Args...>::value, void>::type
    construct(U* ptr, Args&&... args)
    {
        c4::construct(ptr, std::allocator_arg, this->resource(), std::forward<Args>(args)...);
    }
    template<class U, class I, class... Args>
    C4_ALWAYS_INLINE typename std::enable_if<detail::has_alloc_arg<U, Args...>::value, void>::type
    construct_n(U* ptr, I n, Args&&... args)
    {
        c4::construct_n(ptr, n, std::allocator_arg, this->resource(), std::forward<Args>(args)...);
    }

    // 2.2. can construct(Args..., MemoryResource*)
    template<class U, class... Args>
    C4_ALWAYS_INLINE typename std::enable_if<detail::has_alloc<U, Args...>::value, void>::type
    construct(U* ptr, Args&&... args)
    {
        c4::construct(ptr, std::forward<Args>(args)..., this->resource());
    }
    template<class U, class I, class... Args>
    C4_ALWAYS_INLINE typename std::enable_if<detail::has_alloc<U, Args...>::value, void>::type
    construct_n(U* ptr, I n, Args&&... args)
    {
        c4::construct_n(ptr, n, std::forward<Args>(args)..., this->resource());
    }

    template<class U>
    static C4_ALWAYS_INLINE void destroy(U* ptr)
    {
        c4::destroy(ptr);
    }
    template<class U, class I>
    static C4_ALWAYS_INLINE void destroy_n(U* ptr, I n)
    {
        c4::destroy_n(ptr, n);
    }
};


//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------

/** An allocator is simply a proxy to a memory resource.
 * @param T
 * @param MemResProvider
 * @ingroup allocators */
template<class T, class MemResProvider=MemResGlobal>
class Allocator : public detail::_AllocatorUtil<MemResProvider>
{
public:

    using impl_type = detail::_AllocatorUtil<MemResProvider>;

    using value_type = T;
    using pointer = T*;
    using const_pointer = T const*;
    using reference = T&;
    using const_reference = T const&;
    using size_type = size_t;
    using difference_type = std::ptrdiff_t;
    using propagate_on_container_move_assigment = std::true_type;

public:

    template<class U, class MRProv>
    bool operator== (Allocator<U, MRProv> const& that) const
    {
        return this->resource() == that.resource();
    }
    template<class U, class MRProv>
    bool operator!= (Allocator<U, MRProv> const& that) const
    {
        return this->resource() != that.resource();
    }

public:

    template<class U, class MRProv> friend class Allocator;
    template<class U>
    struct rebind
    {
        using other = Allocator<U, MemResProvider>;
    };
    template<class U>
    typename rebind<U>::other rebound()
    {
        return typename rebind<U>::other(*this);
    }

public:

    using impl_type::impl_type;
    Allocator() : impl_type() {} // VS demands this

    template<class U> Allocator(Allocator<U, MemResProvider> const& that) : impl_type(that.resource()) {}

    Allocator(Allocator const&) = default;
    Allocator(Allocator     &&) = default;

    Allocator& operator= (Allocator const&) = default; // WTF? why? @see http://en.cppreference.com/w/cpp/memory/polymorphic_allocator
    Allocator& operator= (Allocator     &&) = default;

    /** returns a default-constructed polymorphic allocator object
     * @see http://en.cppreference.com/w/cpp/memory/polymorphic_allocator/select_on_container_copy_construction      */
    Allocator select_on_container_copy_construct() const { return Allocator(*this); }

    T* allocate(size_t num_objs, size_t alignment=alignof(T))
    {
        C4_ASSERT(this->resource() != nullptr);
        C4_ASSERT(alignment >= alignof(T));
        void* vmem = this->resource()->allocate(detail::size_for<T>(num_objs), alignment);
        T* mem = static_cast<T*>(vmem);
        return mem;
    }

    void deallocate(T * ptr, size_t num_objs, size_t alignment=alignof(T))
    {
        C4_ASSERT(this->resource() != nullptr);
        C4_ASSERT(alignment>= alignof(T));
        this->resource()->deallocate(ptr, detail::size_for<T>(num_objs), alignment);
    }

    T* reallocate(T* ptr, size_t oldnum, size_t newnum, size_t alignment=alignof(T))
    {
        C4_ASSERT(this->resource() != nullptr);
        C4_ASSERT(alignment >= alignof(T));
        void* vmem = this->resource()->reallocate(ptr, detail::size_for<T>(oldnum), detail::size_for<T>(newnum), alignment);
        T* mem = static_cast<T*>(vmem);
        return mem;
    }

};

//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------

/** @ingroup allocators */
template<class T, size_t N=16, size_t Alignment=alignof(T), class MemResProvider=MemResGlobal>
class SmallAllocator : public detail::_AllocatorUtil<MemResProvider>
{
    static_assert(Alignment >= alignof(T), "invalid alignment");

    using impl_type = detail::_AllocatorUtil<MemResProvider>;

    alignas(Alignment) char m_arr[N * sizeof(T)];
    size_t m_num{0};

public:

    using value_type = T;
    using pointer = T*;
    using const_pointer = T const*;
    using reference = T&;
    using const_reference = T const&;
    using size_type = size_t;
    using difference_type = std::ptrdiff_t;
    using propagate_on_container_move_assigment = std::true_type;

    template<class U>
    bool operator== (SmallAllocator<U,N,Alignment,MemResProvider> const&) const
    {
        return false;
    }
    template<class U>
    bool operator!= (SmallAllocator<U,N,Alignment,MemResProvider> const&) const
    {
        return true;
    }

public:

    template<class U, size_t, size_t, class> friend class SmallAllocator;
    template<class U>
    struct rebind
    {
        using other = SmallAllocator<U, N, alignof(U), MemResProvider>;
    };
    template<class U>
    typename rebind<U>::other rebound()
    {
        return typename rebind<U>::other(*this);
    }

public:

    using impl_type::impl_type;
    SmallAllocator() : impl_type() {} // VS demands this

    template<class U, size_t N2, size_t A2, class MP2>
    SmallAllocator(SmallAllocator<U,N2,A2,MP2> const& that) : impl_type(that.resource())
    {
        C4_ASSERT(that.m_num == 0);
    }

    SmallAllocator(SmallAllocator const&) = default;
    SmallAllocator(SmallAllocator     &&) = default;

    SmallAllocator& operator= (SmallAllocator const&) = default; // WTF? why? @see http://en.cppreference.com/w/cpp/memory/polymorphic_allocator
    SmallAllocator& operator= (SmallAllocator     &&) = default;

    /** returns a default-constructed polymorphic allocator object
     * @see http://en.cppreference.com/w/cpp/memory/polymorphic_allocator/select_on_container_copy_construction      */
    SmallAllocator select_on_container_copy_construct() const { return SmallAllocator(*this); }

    T* allocate(size_t num_objs, size_t alignment=Alignment)
    {
        C4_ASSERT(this->resource() != nullptr);
        C4_ASSERT(alignment >= alignof(T));
        void *vmem;
        if(m_num + num_objs <= N)
        {
            vmem = (m_arr + m_num * sizeof(T));
        }
        else
        {
            vmem = this->resource()->allocate(num_objs * sizeof(T), alignment);
        }
        m_num += num_objs;
        T *mem = static_cast<T*>(vmem);
        return mem;
    }

    void deallocate(T * ptr, size_t num_objs, size_t alignment=Alignment)
    {
        C4_ASSERT(m_num >= num_objs);
        m_num -= num_objs;
        if((char*)ptr >= m_arr && (char*)ptr < m_arr + (N * sizeof(T)))
        {
            return;
        }
        C4_ASSERT(this->resource() != nullptr);
        C4_ASSERT(alignment >= alignof(T));
        this->resource()->deallocate(ptr, num_objs * sizeof(T), alignment);
    }

    T* reallocate(T * ptr, size_t oldnum, size_t newnum, size_t alignment=Alignment)
    {
        C4_ASSERT(this->resource() != nullptr);
        C4_ASSERT(alignment >= alignof(T));
        if(oldnum <= N && newnum <= N)
        {
            return m_arr;
        }
        else if(oldnum <= N && newnum > N)
        {
            return allocate(newnum, alignment);
        }
        else if(oldnum > N && newnum <= N)
        {
            deallocate(ptr, oldnum, alignment);
            return m_arr;
        }
        void* vmem = this->resource()->reallocate(ptr, oldnum * sizeof(T), newnum * sizeof(T), alignment);
        T* mem = static_cast<T*>(vmem);
        return mem;
    }

};

//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------

/** An allocator making use of the global memory resource.
 * @ingroup allocators */
template<class T> using allocator = Allocator<T, MemResGlobal>;
/** An allocator with a per-instance memory resource
 * @ingroup allocators */
template<class T> using allocator_mr = Allocator<T, MemRes>;

/** @ingroup allocators */
template<class T, size_t N=16, size_t Alignment=alignof(T)> using small_allocator = SmallAllocator<T, N, Alignment, MemResGlobal>;
/** @ingroup allocators */
template<class T, size_t N=16, size_t Alignment=alignof(T)> using small_allocator_mr = SmallAllocator<T, N, Alignment, MemRes>;

C4_SUPPRESS_WARNING_GCC_CLANG_POP

} // namespace c4

#endif /* _C4_ALLOCATOR_HPP_ */
