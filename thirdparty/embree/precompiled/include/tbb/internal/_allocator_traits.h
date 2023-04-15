/*
    Copyright (c) 2019-2020 Intel Corporation

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

#ifndef __TBB_allocator_traits_H
#define __TBB_allocator_traits_H

#include "../tbb_stddef.h" // true/false_type

#if __TBB_ALLOCATOR_TRAITS_PRESENT
#include <memory> // for allocator_traits
#endif

#if __TBB_CPP11_RVALUE_REF_PRESENT
#include <utility> // for std::move
#endif

// For allocator_swap helper
#include __TBB_STD_SWAP_HEADER

namespace tbb {
namespace internal {

//! Internal implementation of allocator traits, propagate_on_* use internal boolean_constant.
//! In order to avoid code duplication, check what implementation of boolean constant will likely be passed.
#if __TBB_ALLOCATOR_TRAITS_PRESENT
typedef std::true_type traits_true_type;
typedef std::false_type traits_false_type;
#else
typedef tbb::internal::true_type traits_true_type;
typedef tbb::internal::false_type traits_false_type;
#endif

//! Copy assignment implementation for allocator if propagate_on_container_copy_assignment == true_type
//! Noop if pocca == false_type
template <typename MyAlloc, typename OtherAlloc>
inline void allocator_copy_assignment(MyAlloc& my_allocator, OtherAlloc& other_allocator, traits_true_type) {
    my_allocator = other_allocator;
}
template <typename MyAlloc, typename OtherAlloc>
inline void allocator_copy_assignment(MyAlloc&, OtherAlloc&, traits_false_type) { /* NO COPY */}

#if __TBB_CPP11_RVALUE_REF_PRESENT
//! Move assignment implementation for allocator if propagate_on_container_move_assignment == true_type.
//! Noop if pocma == false_type.
template <typename MyAlloc, typename OtherAlloc>
inline void allocator_move_assignment(MyAlloc& my_allocator, OtherAlloc& other_allocator, traits_true_type) {
    my_allocator = std::move(other_allocator);
}
template <typename MyAlloc, typename OtherAlloc>
inline void allocator_move_assignment(MyAlloc&, OtherAlloc&, traits_false_type) { /* NO MOVE */ }
#endif

//! Swap implementation for allocators if propagate_on_container_swap == true_type.
//! Noop if pocs == false_type.
template <typename MyAlloc, typename OtherAlloc>
inline void allocator_swap(MyAlloc& my_allocator, OtherAlloc& other_allocator, traits_true_type) {
    using std::swap;
    swap(my_allocator, other_allocator);
}
template <typename MyAlloc, typename OtherAlloc>
inline void allocator_swap(MyAlloc&, OtherAlloc&, traits_false_type) { /* NO SWAP */ }

#if __TBB_ALLOCATOR_TRAITS_PRESENT
using std::allocator_traits;
#else
//! Internal allocator_traits implementation, which relies on C++03 standard
//! [20.1.5] allocator requirements
template<typename Alloc>
struct allocator_traits {
    // C++03 allocator doesn't have to be assignable or swappable, therefore
    // define these traits as false_type to do not require additional operations
    // that are not supposed to be in.
    typedef tbb::internal::false_type propagate_on_container_move_assignment;
    typedef tbb::internal::false_type propagate_on_container_copy_assignment;
    typedef tbb::internal::false_type propagate_on_container_swap;

    typedef Alloc allocator_type;
    typedef typename allocator_type::value_type value_type;

    typedef typename allocator_type::pointer pointer;
    typedef typename allocator_type::const_pointer const_pointer;
    typedef typename allocator_type::difference_type difference_type;
    typedef typename allocator_type::size_type size_type;

    template <typename U> struct rebind_alloc {
        typedef typename Alloc::template rebind<U>::other other;
    };

    static pointer allocate(Alloc& a, size_type n) {
        return a.allocate(n);
    }

    static void deallocate(Alloc& a, pointer p, size_type n) {
        a.deallocate(p, n);
    }

    template<typename PT>
    static void construct(Alloc&, PT* p) {
        ::new (static_cast<void*>(p)) PT();
    }

    template<typename PT, typename T1>
    static void construct(Alloc&, PT* p, __TBB_FORWARDING_REF(T1) t1) {
        ::new (static_cast<void*>(p)) PT(tbb::internal::forward<T1>(t1));
    }

    template<typename PT, typename T1, typename T2>
    static void construct(Alloc&, PT* p, __TBB_FORWARDING_REF(T1) t1, __TBB_FORWARDING_REF(T2) t2) {
        ::new (static_cast<void*>(p)) PT(tbb::internal::forward<T1>(t1), tbb::internal::forward<T2>(t2));
    }

    template<typename PT, typename T1, typename T2, typename T3>
    static void construct(Alloc&, PT* p, __TBB_FORWARDING_REF(T1) t1,
                          __TBB_FORWARDING_REF(T2) t2, __TBB_FORWARDING_REF(T3) t3) {
        ::new (static_cast<void*>(p)) PT(tbb::internal::forward<T1>(t1), tbb::internal::forward<T2>(t2),
                                         tbb::internal::forward<T3>(t3));
    }

    template<typename T>
    static void destroy(Alloc&, T* p) {
        p->~T();
        tbb::internal::suppress_unused_warning(p);
    }

    static Alloc select_on_container_copy_construction(const Alloc& a) { return a; }
};
#endif // __TBB_ALLOCATOR_TRAITS_PRESENT

//! C++03/C++11 compliant rebind helper, even if no std::allocator_traits available
//! or rebind is not defined for allocator type
template<typename Alloc, typename T>
struct allocator_rebind {
#if __TBB_ALLOCATOR_TRAITS_PRESENT
    typedef typename allocator_traits<Alloc>::template rebind_alloc<T> type;
#else
    typedef typename allocator_traits<Alloc>::template rebind_alloc<T>::other type;
#endif
};

}} // namespace tbb::internal

#endif // __TBB_allocator_traits_H

