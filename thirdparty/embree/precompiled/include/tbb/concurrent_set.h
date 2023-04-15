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

#ifndef __TBB_concurrent_set_H
#define __TBB_concurrent_set_H

#define __TBB_concurrent_set_H_include_area
#include "internal/_warning_suppress_enable_notice.h"

#if !TBB_PREVIEW_CONCURRENT_ORDERED_CONTAINERS
#error Set TBB_PREVIEW_CONCURRENT_ORDERED_CONTAINERS to include concurrent_set.h
#endif

#include "tbb/tbb_config.h"

// concurrent_set requires C++11 support
#if __TBB_CONCURRENT_ORDERED_CONTAINERS_PRESENT

#include "internal/_concurrent_skip_list_impl.h"

namespace tbb {
namespace interface10 {

// TODO: test this class
template<typename Key, typename KeyCompare, typename RandomGenerator, size_t MAX_LEVELS, typename Allocator, bool AllowMultimapping>
class set_traits {
public:
    static constexpr size_t MAX_LEVEL = MAX_LEVELS;
    using random_level_generator_type = RandomGenerator;
    using key_type = Key;
    using value_type = key_type;
    using compare_type = KeyCompare;
    using value_compare = compare_type;
    using reference = value_type & ;
    using const_reference = const value_type&;
    using allocator_type = Allocator;
    using mutex_type = tbb::spin_mutex;
    using node_type = tbb::internal::node_handle<key_type, value_type, internal::skip_list_node<value_type, mutex_type>, allocator_type>;

    static const bool allow_multimapping = AllowMultimapping;

    static const key_type& get_key(const_reference val) {
        return val;
    }

    static value_compare value_comp(compare_type comp) { return comp; }
};

template <typename Key, typename Comp, typename Allocator>
class concurrent_multiset;

template <typename Key, typename Comp = std::less<Key>, typename Allocator = tbb_allocator<Key>>
class concurrent_set
    : public internal::concurrent_skip_list<set_traits<Key, Comp, internal::concurrent_geometric_level_generator<64>, 64, Allocator, false>> {
    using traits_type = set_traits<Key, Comp, internal::concurrent_geometric_level_generator<64>, 64, Allocator, false>;
    using base_type = internal::concurrent_skip_list<traits_type>;
#if __TBB_EXTRA_DEBUG
public:
#endif
    using base_type::allow_multimapping;
public:
    using key_type = Key;
    using value_type = typename traits_type::value_type;
    using size_type = typename base_type::size_type;
    using difference_type = typename base_type::difference_type;
    using key_compare = Comp;
    using value_compare = typename base_type::value_compare;
    using allocator_type = Allocator;

    using reference = typename base_type::reference;
    using const_reference = typename base_type::const_reference;
    using pointer = typename base_type::pointer;
    using const_pointer = typename base_type::pointer;

    using iterator = typename base_type::iterator;
    using const_iterator = typename base_type::const_iterator;
    using reverse_iterator = typename base_type::reverse_iterator;
    using const_reverse_iterator = typename base_type::const_reverse_iterator;

    using node_type = typename base_type::node_type;

    using base_type::insert;

    concurrent_set() = default;

    explicit concurrent_set(const key_compare& comp, const allocator_type& alloc = allocator_type()) : base_type(comp, alloc) {}

    explicit concurrent_set(const allocator_type& alloc) : base_type(key_compare(), alloc) {}

    template< class InputIt >
    concurrent_set(InputIt first, InputIt last, const key_compare& comp = Comp(), const allocator_type& alloc = allocator_type())
        : base_type(first, last, comp, alloc) {}

    template< class InputIt >
    concurrent_set(InputIt first, InputIt last, const allocator_type& alloc) : base_type(first, last, key_compare(), alloc) {}

    /** Copy constructor */
    concurrent_set(const concurrent_set&) = default;

    concurrent_set(const concurrent_set& other, const allocator_type& alloc) : base_type(other, alloc) {}

    concurrent_set(concurrent_set&&) = default;

    concurrent_set(concurrent_set&& other, const allocator_type& alloc) : base_type(std::move(other), alloc) {}

    concurrent_set(std::initializer_list<value_type> init, const key_compare& comp = Comp(), const allocator_type& alloc = allocator_type())
        : base_type(comp, alloc) {
        insert(init);
    }

    concurrent_set(std::initializer_list<value_type> init, const allocator_type& alloc)
        : base_type(key_compare(), alloc) {
        insert(init);
    }

    concurrent_set& operator=(const concurrent_set& other) {
        return static_cast<concurrent_set&>(base_type::operator=(other));
    }

    concurrent_set& operator=(concurrent_set&& other) {
        return static_cast<concurrent_set&>(base_type::operator=(std::move(other)));
    }

    template<typename C2>
    void merge(concurrent_set<key_type, C2, Allocator>& source) {
        this->internal_merge(source);
    }

    template<typename C2>
    void merge(concurrent_set<key_type, C2, Allocator>&& source) {
        this->internal_merge(std::move(source));
    }

    template<typename C2>
    void merge(concurrent_multiset<key_type, C2, Allocator>& source) {
        this->internal_merge(source);
    }

    template<typename C2>
    void merge(concurrent_multiset<key_type, C2, Allocator>&& source) {
        this->internal_merge(std::move(source));
    }
}; // class concurrent_set

#if __TBB_CPP17_DEDUCTION_GUIDES_PRESENT

namespace internal {

using namespace tbb::internal;

template<template<typename...> typename Set, typename Key, typename... Args>
using c_set_t = Set<Key,
                    std::conditional_t< (sizeof...(Args) > 0) && !is_allocator_v<pack_element_t<0, Args...> >,
                                        pack_element_t<0, Args...>, std::less<Key> >,
                    std::conditional_t< (sizeof...(Args) > 0) && is_allocator_v<pack_element_t<sizeof...(Args)-1, Args...> >,
                                        pack_element_t<sizeof...(Args)-1, Args...>, tbb_allocator<Key> > >;
} // namespace internal

template<typename It, typename... Args>
concurrent_set(It, It, Args...)
-> internal::c_set_t<concurrent_set, internal::iterator_value_t<It>, Args...>;

template<typename Key, typename... Args>
concurrent_set(std::initializer_list<Key>, Args...)
-> internal::c_set_t<concurrent_set, Key, Args...>;

#endif // __TBB_CPP17_DEDUCTION_GUIDES_PRESENT

template <typename Key, typename Comp = std::less<Key>, typename Allocator = tbb_allocator<Key>>
class concurrent_multiset
    : public internal::concurrent_skip_list<set_traits<Key, Comp, internal::concurrent_geometric_level_generator<64>, 64, Allocator, true>> {
    using traits_type = set_traits<Key, Comp, internal::concurrent_geometric_level_generator<64>, 64, Allocator, true>;
    using base_type = internal::concurrent_skip_list<traits_type>;
#if __TBB_EXTRA_DEBUG
public:
#endif
    using base_type::allow_multimapping;
public:
    using key_type = Key;
    using value_type = typename traits_type::value_type;
    using size_type = typename base_type::size_type;
    using difference_type = typename base_type::difference_type;
    using key_compare = Comp;
    using value_compare = typename base_type::value_compare;
    using allocator_type = Allocator;

    using reference = typename base_type::reference;
    using const_reference = typename base_type::const_reference;
    using pointer = typename base_type::pointer;
    using const_pointer = typename base_type::pointer;

    using iterator = typename base_type::iterator;
    using const_iterator = typename base_type::const_iterator;
    using reverse_iterator = typename base_type::reverse_iterator;
    using const_reverse_iterator = typename base_type::const_reverse_iterator;

    using node_type = typename base_type::node_type;

    using base_type::insert;

    concurrent_multiset() = default;

    explicit concurrent_multiset(const key_compare& comp, const allocator_type& alloc = allocator_type()) : base_type(comp, alloc) {}

    explicit concurrent_multiset(const allocator_type& alloc) : base_type(key_compare(), alloc) {}

    template< class InputIt >
    concurrent_multiset(InputIt first, InputIt last, const key_compare& comp = Comp(), const allocator_type& alloc = allocator_type())
        : base_type(comp, alloc) {
        insert(first, last);
    }

    template< class InputIt >
    concurrent_multiset(InputIt first, InputIt last, const allocator_type& alloc) : base_type(key_compare(), alloc) {
        insert(first, last);
    }

    /** Copy constructor */
    concurrent_multiset(const concurrent_multiset&) = default;

    concurrent_multiset(const concurrent_multiset& other, const allocator_type& alloc) : base_type(other, alloc) {}

    concurrent_multiset(concurrent_multiset&&) = default;

    concurrent_multiset(concurrent_multiset&& other, const allocator_type& alloc) : base_type(std::move(other), alloc) {}

    concurrent_multiset(std::initializer_list<value_type> init, const key_compare& comp = Comp(), const allocator_type& alloc = allocator_type())
        : base_type(comp, alloc) {
        insert(init);
    }

    concurrent_multiset(std::initializer_list<value_type> init, const allocator_type& alloc)
        : base_type(key_compare(), alloc) {
        insert(init);
    }

    concurrent_multiset& operator=(const concurrent_multiset& other) {
        return static_cast<concurrent_multiset&>(base_type::operator=(other));
    }

    concurrent_multiset& operator=(concurrent_multiset&& other) {
        return static_cast<concurrent_multiset&>(base_type::operator=(std::move(other)));
    }

    template<typename C2>
    void merge(concurrent_set<key_type, C2, Allocator>& source) {
        this->internal_merge(source);
    }

    template<typename C2>
    void merge(concurrent_set<key_type, C2, Allocator>&& source) {
        this->internal_merge(std::move(source));
    }

    template<typename C2>
    void merge(concurrent_multiset<key_type, C2, Allocator>& source) {
        this->internal_merge(source);
    }

    template<typename C2>
    void merge(concurrent_multiset<key_type, C2, Allocator>&& source) {
        this->internal_merge(std::move(source));
    }
}; // class concurrent_multiset

#if __TBB_CPP17_DEDUCTION_GUIDES_PRESENT


template<typename It, typename... Args>
concurrent_multiset(It, It, Args...)
-> internal::c_set_t<concurrent_multiset, internal::iterator_value_t<It>, Args...>;

template<typename Key, typename... Args>
concurrent_multiset(std::initializer_list<Key>, Args...)
-> internal::c_set_t<concurrent_multiset, Key, Args...>;

#endif // __TBB_CPP17_DEDUCTION_GUIDES_PRESENT

} // namespace interface10

using interface10::concurrent_set;
using interface10::concurrent_multiset;

} // namespace tbb

#endif // __TBB_CONCURRENT_ORDERED_CONTAINERS_PRESENT

#include "internal/_warning_suppress_disable_notice.h"
#undef __TBB_concurrent_set_H_include_area

#endif // __TBB_concurrent_set_H
