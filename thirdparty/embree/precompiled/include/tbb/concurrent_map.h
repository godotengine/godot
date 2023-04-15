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

#ifndef __TBB_concurrent_map_H
#define __TBB_concurrent_map_H

#define __TBB_concurrent_map_H_include_area
#include "internal/_warning_suppress_enable_notice.h"

#if !TBB_PREVIEW_CONCURRENT_ORDERED_CONTAINERS
#error Set TBB_PREVIEW_CONCURRENT_ORDERED_CONTAINERS to include concurrent_map.h
#endif

#include "tbb_config.h"

// concurrent_map requires C++11 support
#if __TBB_CONCURRENT_ORDERED_CONTAINERS_PRESENT

#include "internal/_concurrent_skip_list_impl.h"

namespace tbb {

namespace interface10 {

template<typename Key, typename Value, typename KeyCompare, typename RandomGenerator,
         size_t MAX_LEVELS, typename Allocator, bool AllowMultimapping>
class map_traits {
public:
    static constexpr size_t MAX_LEVEL = MAX_LEVELS;
    using random_level_generator_type = RandomGenerator;
    using key_type = Key;
    using mapped_type = Value;
    using compare_type = KeyCompare;
    using value_type = std::pair<const key_type, mapped_type>;
    using reference = value_type&;
    using const_reference = const value_type&;
    using allocator_type = Allocator;
    using mutex_type = tbb::spin_mutex;
    using node_type = tbb::internal::node_handle<key_type, value_type, internal::skip_list_node<value_type, mutex_type>, allocator_type>;

    static const bool allow_multimapping = AllowMultimapping;

    class value_compare {
    public:
        // TODO: these member types are deprecated in C++17, do we need to let them
        using result_type = bool;
        using first_argument_type = value_type;
        using second_argument_type = value_type;

        bool operator()(const value_type& lhs, const value_type& rhs) const {
            return comp(lhs.first, rhs.first);
        }

    protected:
        value_compare(compare_type c) : comp(c) {}

        friend class map_traits;

        compare_type comp;
    };

    static value_compare value_comp(compare_type comp) { return value_compare(comp); }

    static const key_type& get_key(const_reference val) {
        return val.first;
    }
}; // class map_traits

template <typename Key, typename Value, typename Comp, typename Allocator>
class concurrent_multimap;

template <typename Key, typename Value, typename Comp = std::less<Key>, typename Allocator = tbb_allocator<std::pair<const Key, Value>>>
class concurrent_map
    : public internal::concurrent_skip_list<map_traits<Key, Value, Comp, internal::concurrent_geometric_level_generator<64>, 64, Allocator, false>> {
    using traits_type = map_traits<Key, Value, Comp, internal::concurrent_geometric_level_generator<64>, 64, Allocator, false>;
    using base_type = internal::concurrent_skip_list<traits_type>;
#if __TBB_EXTRA_DEBUG
public:
#endif
    using base_type::allow_multimapping;
public:
    using key_type = Key;
    using mapped_type = Value;
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

    using base_type::end;
    using base_type::find;
    using base_type::emplace;
    using base_type::insert;

    concurrent_map() = default;

    explicit concurrent_map(const key_compare& comp, const allocator_type& alloc = allocator_type()) : base_type(comp, alloc) {}

    explicit concurrent_map(const allocator_type& alloc) : base_type(key_compare(), alloc) {}

    template< class InputIt >
    concurrent_map(InputIt first, InputIt last, const key_compare& comp = Comp(), const allocator_type& alloc = allocator_type())
        : base_type(first, last, comp, alloc) {}

    template< class InputIt >
    concurrent_map(InputIt first, InputIt last, const allocator_type& alloc) : base_type(first, last, key_compare(), alloc) {}

    /** Copy constructor */
    concurrent_map(const concurrent_map&) = default;

    concurrent_map(const concurrent_map& other, const allocator_type& alloc) : base_type(other, alloc) {}

    concurrent_map(concurrent_map&&) = default;

    concurrent_map(concurrent_map&& other, const allocator_type& alloc) : base_type(std::move(other), alloc) {}

    concurrent_map(std::initializer_list<value_type> init, const key_compare& comp = Comp(), const allocator_type& alloc = allocator_type())
        : base_type(comp, alloc) {
        insert(init);
    }

    concurrent_map(std::initializer_list<value_type> init, const allocator_type& alloc)
        : base_type(key_compare(), alloc) {
        insert(init);
    }

    concurrent_map& operator=(const concurrent_map& other) {
        return static_cast<concurrent_map&>(base_type::operator=(other));
    }

    concurrent_map& operator=(concurrent_map&& other) {
        return static_cast<concurrent_map&>(base_type::operator=(std::move(other)));
    }

    mapped_type& at(const key_type& key) {
        iterator it = find(key);

        if (it == end()) {
            tbb::internal::throw_exception(tbb::internal::eid_invalid_key);
        }

        return it->second;
    }

    const mapped_type& at(const key_type& key) const {
        const_iterator it = find(key);

        if (it == end()) {
            tbb::internal::throw_exception(tbb::internal::eid_invalid_key);
        }

        return it->second;
    }

    mapped_type& operator[](const key_type& key) {
        iterator it = find(key);

        if (it == end()) {
            it = emplace(std::piecewise_construct, std::forward_as_tuple(key), std::tuple<>()).first;
        }

        return it->second;
    }

    mapped_type& operator[](key_type&& key) {
        iterator it = find(key);

        if (it == end()) {
            it = emplace(std::piecewise_construct, std::forward_as_tuple(std::move(key)), std::tuple<>()).first;
        }

        return it->second;
    }

    template<typename P, typename std::enable_if<std::is_constructible<value_type, P&&>::value>::type>
    std::pair<iterator, bool> insert(P&& value) {
        return emplace(std::forward<P>(value));
    }

    template<typename P, typename std::enable_if<std::is_constructible<value_type, P&&>::value>::type>
    iterator insert(const_iterator hint, P&& value) {
            return emplace_hint(hint, std::forward<P>(value));
        return end();
    }

    template<typename C2>
    void merge(concurrent_map<key_type, mapped_type, C2, Allocator>& source) {
        this->internal_merge(source);
    }

    template<typename C2>
    void merge(concurrent_map<key_type, mapped_type, C2, Allocator>&& source) {
        this->internal_merge(std::move(source));
    }

    template<typename C2>
    void merge(concurrent_multimap<key_type, mapped_type, C2, Allocator>& source) {
        this->internal_merge(source);
    }

    template<typename C2>
    void merge(concurrent_multimap<key_type, mapped_type, C2, Allocator>&& source) {
        this->internal_merge(std::move(source));
    }
}; // class concurrent_map

#if __TBB_CPP17_DEDUCTION_GUIDES_PRESENT

namespace internal {

using namespace tbb::internal;

template<template<typename...> typename Map, typename Key, typename T, typename... Args>
using c_map_t = Map<Key, T,
                    std::conditional_t< (sizeof...(Args) > 0) && !is_allocator_v<pack_element_t<0, Args...> >,
                                        pack_element_t<0, Args...>, std::less<Key> >,
                    std::conditional_t< (sizeof...(Args) > 0) && is_allocator_v<pack_element_t<sizeof...(Args)-1, Args...> >,
                                        pack_element_t<sizeof...(Args)-1, Args...>, tbb_allocator<std::pair<const Key, T> > > >;
} // namespace internal

template<typename It, typename... Args>
concurrent_map(It, It, Args...)
-> internal::c_map_t<concurrent_map, internal::iterator_key_t<It>, internal::iterator_mapped_t<It>, Args...>;

template<typename Key, typename T, typename... Args>
concurrent_map(std::initializer_list<std::pair<const Key, T>>, Args...)
-> internal::c_map_t<concurrent_map, Key, T, Args...>;

#endif // __TBB_CPP17_DEDUCTION_GUIDES_PRESENT

template <typename Key, typename Value, typename Comp = std::less<Key>, typename Allocator = tbb_allocator<std::pair<const Key, Value>>>
class concurrent_multimap
    : public internal::concurrent_skip_list<map_traits<Key, Value, Comp, internal::concurrent_geometric_level_generator<64>, 64, Allocator, true>> {
    using traits_type = map_traits<Key, Value, Comp, internal::concurrent_geometric_level_generator<64>, 64, Allocator, true>;
    using base_type = internal::concurrent_skip_list<traits_type>;
#if __TBB_EXTRA_DEBUG
public:
#endif
    using base_type::allow_multimapping;
public:
    using key_type = Key;
    using mapped_type = Value;
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

    using base_type::end;
    using base_type::find;
    using base_type::emplace;
    using base_type::insert;

    concurrent_multimap() = default;

    explicit concurrent_multimap(const key_compare& comp, const allocator_type& alloc = allocator_type()) : base_type(comp, alloc) {}

    explicit concurrent_multimap(const allocator_type& alloc) : base_type(key_compare(), alloc) {}

    template< class InputIt >
    concurrent_multimap(InputIt first, InputIt last, const key_compare& comp = Comp(), const allocator_type& alloc = allocator_type())
        : base_type(first, last, comp, alloc) {}

    template< class InputIt >
    concurrent_multimap(InputIt first, InputIt last, const allocator_type& alloc) : base_type(first, last, key_compare(), alloc) {}

    /** Copy constructor */
    concurrent_multimap(const concurrent_multimap&) = default;

    concurrent_multimap(const concurrent_multimap& other, const allocator_type& alloc) : base_type(other, alloc) {}

    concurrent_multimap(concurrent_multimap&&) = default;

    concurrent_multimap(concurrent_multimap&& other, const allocator_type& alloc) : base_type(std::move(other), alloc) {}

    concurrent_multimap(std::initializer_list<value_type> init, const key_compare& comp = Comp(), const allocator_type& alloc = allocator_type())
        : base_type(comp, alloc) {
        insert(init);
    }

    concurrent_multimap(std::initializer_list<value_type> init, const allocator_type& alloc)
        : base_type(key_compare(), alloc) {
        insert(init);
    }

    concurrent_multimap& operator=(const concurrent_multimap& other) {
        return static_cast<concurrent_multimap&>(base_type::operator=(other));
    }

    concurrent_multimap& operator=(concurrent_multimap&& other) {
        return static_cast<concurrent_multimap&>(base_type::operator=(std::move(other)));
    }

    template<typename P, typename std::enable_if<std::is_constructible<value_type, P&&>::value>::type>
    std::pair<iterator, bool> insert(P&& value) {
        return emplace(std::forward<P>(value));
    }

    template<typename P, typename std::enable_if<std::is_constructible<value_type, P&&>::value>::type>
    iterator insert(const_iterator hint, P&& value) {
            return emplace_hint(hint, std::forward<P>(value));
        return end();
    }

    template<typename C2>
    void merge(concurrent_multimap<key_type, mapped_type, C2, Allocator>& source) {
        this->internal_merge(source);
    }

    template<typename C2>
    void merge(concurrent_multimap<key_type, mapped_type, C2, Allocator>&& source) {
        this->internal_merge(std::move(source));
    }

    template<typename C2>
    void merge(concurrent_map<key_type, mapped_type, C2, Allocator>& source) {
        this->internal_merge(source);
    }

    template<typename C2>
    void merge(concurrent_map<key_type, mapped_type, C2, Allocator>&& source) {
        this->internal_merge(std::move(source));
    }

}; // class concurrent_multimap

#if __TBB_CPP17_DEDUCTION_GUIDES_PRESENT

template<typename It, typename... Args>
concurrent_multimap(It, It, Args...)
-> internal::c_map_t<concurrent_multimap, internal::iterator_key_t<It>, internal::iterator_mapped_t<It>, Args...>;

template<typename Key, typename T, typename... Args>
concurrent_multimap(std::initializer_list<std::pair<const Key, T>>, Args...)
-> internal::c_map_t<concurrent_multimap, Key, T, Args...>;

#endif // __TBB_CPP17_DEDUCTION_GUIDES_PRESENT

} // namespace interface10

using interface10::concurrent_map;
using interface10::concurrent_multimap;

} // namespace tbb

#include "internal/_warning_suppress_disable_notice.h"
#undef __TBB_concurrent_map_H_include_area

#endif // __TBB_CONCURRENT_ORDERED_CONTAINERS_PRESENT
#endif // __TBB_concurrent_map_H
