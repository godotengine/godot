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

#ifndef __TBB_concurrent_skip_list_H
#define __TBB_concurrent_skip_list_H

#if !defined(__TBB_concurrent_map_H) && !defined(__TBB_concurrent_set_H)
#error Do not #include this internal file directly; use public TBB headers instead.
#endif

#include "../tbb_config.h"
#include "../tbb_stddef.h"
#include "../tbb_allocator.h"
#include "../spin_mutex.h"
#include "../tbb_exception.h"
#include "../enumerable_thread_specific.h"
#include "_allocator_traits.h"
#include "_template_helpers.h"
#include "_node_handle_impl.h"
#include <utility> // Need std::pair
#include <functional>
#include <initializer_list>
#include <memory> // Need std::allocator_traits
#include <atomic>
#include <mutex>
#include <vector>
#include <array>
#include <type_traits>
#include <cstdlib>
#include <random>
#include <tuple>

#if _MSC_VER
#pragma warning(disable: 4189) // warning 4189 -- local variable is initialized but not referenced
#pragma warning(disable: 4127) // warning 4127 -- while (true) has a constant expression in it
#endif

namespace tbb {
namespace interface10 {
namespace internal {

template <typename Value, typename Mutex>
class skip_list_node {

public:
    using value_type = Value;
    using size_type = std::size_t;
    using reference = value_type & ;
    using const_reference = const value_type & ;
    using pointer = value_type * ;
    using const_pointer = const value_type *;
    using node_pointer = skip_list_node * ;
    using atomic_node_pointer = std::atomic<node_pointer>;

    using mutex_type = Mutex;
    using lock_type = std::unique_lock<mutex_type>;

    skip_list_node(size_type levels) : my_height(levels), my_fullyLinked(false) {
        for (size_type lev = 0; lev < my_height; ++lev)
            new(&my_next(lev)) atomic_node_pointer(nullptr);
        __TBB_ASSERT(height() == levels, "Wrong node height");
    }

    ~skip_list_node() {
        for(size_type lev = 0; lev < my_height; ++lev)
            my_next(lev).~atomic();
    }

    skip_list_node(const skip_list_node&) = delete;

    skip_list_node(skip_list_node&&) = delete;

    skip_list_node& operator=(const skip_list_node&) = delete;

    pointer storage() {
        return reinterpret_cast<pointer>(&my_val);
    }

    reference value() {
        return *storage();
    }

    node_pointer next(size_type level) const {
        __TBB_ASSERT(level < height(), "Cannot get next on the level greater than height");
        return my_next(level).load(std::memory_order_acquire);
    }

    void set_next(size_type level, node_pointer next) {
        __TBB_ASSERT(level < height(), "Cannot set next on the level greater than height");

        my_next(level).store(next, std::memory_order_release);
    }

    /** @return number of layers */
    size_type height() const {
        return my_height;
    }

    bool fully_linked() const {
        return my_fullyLinked.load(std::memory_order_acquire);
    }

    void mark_linked() {
        my_fullyLinked.store(true, std::memory_order_release);
    }

    lock_type acquire() {
        return lock_type(my_mutex);
    }

private:
    using aligned_storage_type = typename std::aligned_storage<sizeof(value_type), alignof(value_type)>::type;

    atomic_node_pointer& my_next(size_type level) {
        atomic_node_pointer* arr = reinterpret_cast<atomic_node_pointer*>(this + 1);
        return arr[level];
    }

    const atomic_node_pointer& my_next(size_type level) const {
        const atomic_node_pointer* arr = reinterpret_cast<const atomic_node_pointer*>(this + 1);
        return arr[level];
    }

    mutex_type my_mutex;
    aligned_storage_type my_val;
    size_type my_height;
    std::atomic_bool my_fullyLinked;
};

template <typename NodeType, bool is_const>
class skip_list_iterator {
    using node_type = NodeType;
    using node_ptr = node_type*;
public:
    using iterator_category = std::forward_iterator_tag;
    using value_type = typename node_type::value_type;
    using difference_type = std::ptrdiff_t;
    using pointer = typename std::conditional<is_const, typename node_type::const_pointer,
                                                        typename node_type::pointer>::type;
    using reference = typename std::conditional<is_const, typename node_type::const_reference,
                                                          typename node_type::reference>::type;

    skip_list_iterator() : my_node_ptr(nullptr) {}

    // TODO: the code above does not compile in VS2015 (seems like a bug) - consider enabling it for all other platforms
    // template <typename T = void, typename = typename std::enable_if<is_const, T>::type>
    // skip_list_iterator(const skip_list_iterator<node_type, false>& other) : my_node_ptr(other.my_node_ptr) {}

    // skip_list_iterator(const skip_list_iterator& other) : my_node_ptr(other.my_node_ptr) {}

    skip_list_iterator(const skip_list_iterator<node_type, false>& other) : my_node_ptr(other.my_node_ptr) {}

    skip_list_iterator& operator=(const skip_list_iterator<node_type, false>& other) {
         my_node_ptr = other.my_node_ptr;
         return *this;
    }

    template <typename T = void, typename = typename std::enable_if<is_const, T>::type>
    skip_list_iterator(const skip_list_iterator<node_type, true>& other) : my_node_ptr(other.my_node_ptr) {}

    reference operator*() const { return *(my_node_ptr->storage()); }
    pointer operator->() const { return &**this; }

    skip_list_iterator& operator++() {
        __TBB_ASSERT(my_node_ptr != nullptr, NULL);
        my_node_ptr = my_node_ptr->next(0);
        return *this;
    }

    skip_list_iterator operator++(int) {
        skip_list_iterator tmp = *this;
        ++*this;
        return tmp;
    }

private:
    skip_list_iterator(node_type* n) : my_node_ptr(n) {}

    node_ptr my_node_ptr;

    template <typename Traits>
    friend class concurrent_skip_list;

    friend class skip_list_iterator<NodeType, true>;

    friend class const_range;
    friend class range;

    template <typename T, bool M, bool U>
    friend bool operator==(const skip_list_iterator<T, M>&, const skip_list_iterator<T, U>&);

    template <typename T, bool M, bool U>
    friend bool operator!=(const skip_list_iterator<T, M>&, const skip_list_iterator<T, U>&);
};

template <typename NodeType, bool is_const1, bool is_const2>
bool operator==(const skip_list_iterator<NodeType, is_const1>& lhs, const skip_list_iterator<NodeType, is_const2>& rhs) {
    return lhs.my_node_ptr == rhs.my_node_ptr;
}

template <typename NodeType, bool is_const1, bool is_const2>
bool operator!=(const skip_list_iterator<NodeType, is_const1>& lhs, const skip_list_iterator<NodeType, is_const2>& rhs) {
    return lhs.my_node_ptr != rhs.my_node_ptr;
}

template <typename Traits>
class concurrent_skip_list {
protected:
    using traits_type = Traits;
    using allocator_type = typename traits_type::allocator_type;
    using allocator_traits_type = std::allocator_traits<allocator_type>;
    using key_compare = typename traits_type::compare_type;
    using value_compare = typename traits_type::value_compare;
    using key_type = typename traits_type::key_type;
    using value_type = typename traits_type::value_type;
    using node_type = typename traits_type::node_type;
    using list_node_type = skip_list_node<value_type, typename traits_type::mutex_type>;

    using iterator = skip_list_iterator<list_node_type, /*is_const=*/false>;
    using const_iterator = skip_list_iterator<list_node_type, /*is_const=*/true>;
    using reverse_iterator = std::reverse_iterator<iterator>;
    using const_reverse_iterator = std::reverse_iterator<const_iterator>;

    using reference = value_type&;
    using const_reference = const value_type&;
    using pointer = typename allocator_traits_type::pointer;
    using const_pointer = typename allocator_traits_type::const_pointer;
    using size_type = std::size_t;
    using difference_type = std::ptrdiff_t;

    using random_level_generator_type = typename traits_type::random_level_generator_type;
    using node_allocator_type = typename std::allocator_traits<allocator_type>::template rebind_alloc<uint8_t>;
    using node_allocator_traits = typename std::allocator_traits<allocator_type>::template rebind_traits<uint8_t>;
    using node_ptr = list_node_type*;

    static constexpr size_type MAX_LEVEL = traits_type::MAX_LEVEL;

    using array_type = std::array<node_ptr, MAX_LEVEL>;
    using lock_array = std::array<typename list_node_type::lock_type, MAX_LEVEL>;

public:
    static bool const allow_multimapping = traits_type::allow_multimapping;

    /**
     * Default constructor. Construct empty skip list.
     */
    concurrent_skip_list() : my_size(0) {
        create_dummy_head();
    }

    explicit concurrent_skip_list(const key_compare& comp, const allocator_type& alloc = allocator_type())
        : my_node_allocator(alloc), my_compare(comp), my_size(0)
    {
        create_dummy_head();
    }

    template<class InputIt>
    concurrent_skip_list(InputIt first, InputIt last, const key_compare& comp = key_compare(),
                         const allocator_type& alloc = allocator_type())
        : my_node_allocator(alloc), my_compare(comp), my_size(0)
    {
        create_dummy_head();
        internal_copy(first, last);
    }

    /** Copy constructor */
    concurrent_skip_list(const concurrent_skip_list& other)
        : my_node_allocator(node_allocator_traits::select_on_container_copy_construction(other.get_allocator())),
          my_compare(other.my_compare), my_rnd_generator(other.my_rnd_generator), my_size(0)
    {
        create_dummy_head();
        internal_copy(other);
        __TBB_ASSERT(my_size == other.my_size, "Wrong size of copy-constructed container");
    }

    concurrent_skip_list(const concurrent_skip_list& other, const allocator_type& alloc)
        : my_node_allocator(alloc), my_compare(other.my_compare),
          my_rnd_generator(other.my_rnd_generator), my_size(0)
    {
        create_dummy_head();
        internal_copy(other);
        __TBB_ASSERT(my_size == other.my_size, "Wrong size of copy-constructed container");
    }

    concurrent_skip_list(concurrent_skip_list&& other)
        : my_node_allocator(std::move(other.my_node_allocator)), my_compare(other.my_compare),
          my_rnd_generator(other.my_rnd_generator)
    {
        internal_move(std::move(other));
    }

    concurrent_skip_list(concurrent_skip_list&& other, const allocator_type& alloc)
        : my_node_allocator(alloc), my_compare(other.my_compare),
          my_rnd_generator(other.my_rnd_generator)
    {
        if (alloc == other.get_allocator()) {
            internal_move(std::move(other));
        } else {
            my_size = 0;
            create_dummy_head();
            internal_copy(std::make_move_iterator(other.begin()), std::make_move_iterator(other.end()));
        }
    }

    ~concurrent_skip_list() {
        clear();
        delete_dummy_head();
    }

    concurrent_skip_list& operator=(const concurrent_skip_list& other) {
        if (this != &other) {
            using pocca_type = typename node_allocator_traits::propagate_on_container_copy_assignment;
            clear();
            tbb::internal::allocator_copy_assignment(my_node_allocator, other.my_node_allocator, pocca_type());
            my_compare = other.my_compare;
            my_rnd_generator = other.my_rnd_generator;
            internal_copy(other);
        }
        return *this;
    }

    concurrent_skip_list& operator=(concurrent_skip_list&& other) {
        if (this != &other) {
            using pocma_type = typename node_allocator_traits::propagate_on_container_move_assignment;
            clear();
            my_compare = other.my_compare;
            my_rnd_generator = other.my_rnd_generator;
            internal_move_assign(std::move(other), pocma_type());
        }
        return *this;
    }

    concurrent_skip_list& operator=(std::initializer_list<value_type> il)
    {
        clear();
        insert(il.begin(),il.end());
        return *this;
    }

    std::pair<iterator, bool> insert(const value_type& value) {
        return internal_insert(value);
    }

    std::pair<iterator, bool> insert(value_type&& value) {
        return internal_insert(std::move(value));
    }

    iterator insert(const_iterator, const_reference value) {
        // Ignore hint
        return insert(value).first;
    }

    iterator insert(const_iterator, value_type&& value) {
        // Ignore hint
        return insert(std::move(value)).first;
    }

    template<typename InputIterator>
    void insert(InputIterator first, InputIterator last) {
        for (InputIterator it = first; it != last; ++it)
            insert(*it);
    }

    void insert(std::initializer_list<value_type> init) {
        insert(init.begin(), init.end());
    }

    std::pair<iterator, bool> insert(node_type&& nh) {
        if(!nh.empty()) {
            std::pair<iterator, bool> insert_result = internal_insert_node(nh.my_node);
            if(insert_result.second) {
                nh.deactivate();
            }
            return insert_result;
        }
        return std::pair<iterator, bool>(end(), false);
    }

    iterator insert(const_iterator, node_type&& nh) {
        // Ignore hint
        return insert(std::move(nh)).first;
    }

    template<typename... Args >
    std::pair<iterator, bool> emplace(Args&&... args) {
        return internal_insert(std::forward<Args>(args)...);
    }

    template<typename... Args>
    iterator emplace_hint(const_iterator, Args&&... args) {
        // Ignore hint
        return emplace(std::forward<Args>(args)...).first;
    }

    iterator unsafe_erase(iterator pos) {
        std::pair<node_ptr, node_ptr> extract_result = internal_extract(pos);
        if(extract_result.first) { // node was extracted
            delete_node(extract_result.first);
            return iterator(extract_result.second);
        }
        return end();
    }

    iterator unsafe_erase(const_iterator pos) {
        return unsafe_erase(get_iterator(pos));
    }

    template <typename K, typename = tbb::internal::is_transparent<key_compare, K>,
                          typename = typename std::enable_if<!std::is_convertible<K, iterator>::value &&
                                                             !std::is_convertible<K, const_iterator>::value>::type>
    size_type unsafe_erase(const K& key) {
        std::pair<iterator, iterator> range = equal_range(key);
        size_type sz = std::distance(range.first, range.second);
        unsafe_erase(range.first, range.second);
        return sz;
    }

    iterator unsafe_erase(const_iterator first, const_iterator last) {
        while(first != last) {
            first = unsafe_erase(get_iterator(first));
        }
        return get_iterator(first);
    }

    size_type unsafe_erase(const key_type& key) {
        std::pair<iterator, iterator> range = equal_range(key);
        size_type sz = std::distance(range.first, range.second);
        unsafe_erase(range.first, range.second);
        return sz;
    }

    node_type unsafe_extract(const_iterator pos) {
        std::pair<node_ptr, node_ptr> extract_result = internal_extract(pos);
        return extract_result.first ? node_type(extract_result.first) : node_type();
    }

    node_type unsafe_extract(const key_type& key) {
        return unsafe_extract(find(key));
    }

    iterator lower_bound(const key_type& key) {
        return internal_get_bound(key, my_compare);
    }

    const_iterator lower_bound(const key_type& key) const {
        return internal_get_bound(key, my_compare);
    }

    template <typename K, typename = typename tbb::internal::is_transparent<key_compare, K> >
    iterator lower_bound(const K& key) {
        return internal_get_bound(key, my_compare);
    }

    template <typename K, typename = typename tbb::internal::is_transparent<key_compare, K> >
    const_iterator lower_bound(const K& key) const {
        return internal_get_bound(key, my_compare);
    }

    iterator upper_bound(const key_type& key) {
        return internal_get_bound(key, not_greater_compare(my_compare));
    }

    const_iterator upper_bound(const key_type& key) const {
        return internal_get_bound(key, not_greater_compare(my_compare));
    }

    template <typename K, typename = typename tbb::internal::is_transparent<key_compare, K> >
    iterator upper_bound(const K& key) {
        return internal_get_bound(key, not_greater_compare(my_compare));
    }

    template <typename K, typename = typename tbb::internal::is_transparent<key_compare, K> >
    const_iterator upper_bound(const K& key) const {
        return internal_get_bound(key, not_greater_compare(my_compare));
    }

    iterator find(const key_type& key) {
        return internal_find(key);
    }

    const_iterator find(const key_type& key) const {
        return internal_find(key);
    }

    template <typename K, typename = typename tbb::internal::is_transparent<key_compare, K> >
    iterator find(const K& key) {
        return internal_find(key);
    }

    template <typename K, typename = typename tbb::internal::is_transparent<key_compare, K> >
    const_iterator find(const K& key) const {
        return internal_find(key);
    }

    size_type count( const key_type& key ) const {
        return internal_count(key);
    }

    template <typename K, typename = typename tbb::internal::is_transparent<key_compare, K> >
    size_type count(const K& key) const {
        return internal_count(key);
    }

    bool contains(const key_type& key) const {
        return find(key) != end();
    }

    template <typename K, typename = typename tbb::internal::is_transparent<key_compare, K> >
    bool contains(const K& key) const {
        return find(key) != end();
    }

    void clear() noexcept {
        __TBB_ASSERT(dummy_head->height() > 0, NULL);

        node_ptr current = dummy_head->next(0);
        while (current) {
            __TBB_ASSERT(current->height() > 0, NULL);
            node_ptr next = current->next(0);
            delete_node(current);
            current = next;
        }

        my_size = 0;
        for (size_type i = 0; i < dummy_head->height(); ++i) {
            dummy_head->set_next(i, nullptr);
        }
    }

    iterator begin() {
        return iterator(dummy_head->next(0));
    }

    const_iterator begin() const {
        return const_iterator(dummy_head->next(0));
    }

    const_iterator cbegin() const {
        return const_iterator(dummy_head->next(0));
    }

    iterator end() {
        return iterator(nullptr);
    }

    const_iterator end() const {
        return const_iterator(nullptr);
    }

    const_iterator cend() const {
        return const_iterator(nullptr);
    }

    size_type size() const {
        return my_size.load(std::memory_order_relaxed);
    }

    size_type max_size() const {
        return my_node_allocator.max_size();
    }

    bool empty() const {
        return 0 == size();
    }

    allocator_type get_allocator() const {
        return my_node_allocator;
    }

    void swap(concurrent_skip_list& other) {
        using std::swap;
        using pocs_type = typename node_allocator_traits::propagate_on_container_swap;
        tbb::internal::allocator_swap(my_node_allocator, other.my_node_allocator, pocs_type());
        swap(my_compare, other.my_compare);
        swap(my_rnd_generator, other.my_rnd_generator);
        swap(dummy_head, other.dummy_head);

        size_type tmp = my_size;
        my_size.store(other.my_size);
        other.my_size.store(tmp);
    }

    std::pair<iterator, iterator> equal_range(const key_type& key) {
        return std::pair<iterator, iterator>(lower_bound(key), upper_bound(key));
    }

    std::pair<const_iterator, const_iterator> equal_range(const key_type& key) const {
        return std::pair<const_iterator, const_iterator>(lower_bound(key), upper_bound(key));
    }

    template <typename K, typename = typename tbb::internal::is_transparent<key_compare, K> >
    std::pair<iterator, iterator> equal_range(const K& key) {
        return std::pair<iterator, iterator>(lower_bound(key), upper_bound(key));
    }

    template <typename K, typename = typename tbb::internal::is_transparent<key_compare, K> >
    std::pair<const_iterator, const_iterator> equal_range(const K& key) const {
        return std::pair<const_iterator, const_iterator>(lower_bound(key), upper_bound(key));
    }

    key_compare key_comp() const { return my_compare; }

    value_compare value_comp() const { return traits_type::value_comp(my_compare); }

    class const_range_type : tbb::internal::no_assign {
    public:
        using size_type = typename concurrent_skip_list::size_type;
        using value_type = typename concurrent_skip_list::value_type;
        using iterator = typename concurrent_skip_list::const_iterator;
    private:
        const_iterator my_end;
        const_iterator my_begin;
        size_type my_level;

    public:

        bool empty() const {
            return my_begin.my_node_ptr->next(0) == my_end.my_node_ptr;
        }

        bool is_divisible() const {
            return my_level != 0 ? my_begin.my_node_ptr->next(my_level - 1) != my_end.my_node_ptr : false;
        }

        size_type size() const { return std::distance(my_begin, my_end);}

        const_range_type( const_range_type& r, split)
            : my_end(r.my_end) {
            my_begin = iterator(r.my_begin.my_node_ptr->next(r.my_level - 1));
            my_level = my_begin.my_node_ptr->height();
            r.my_end = my_begin;
        }

        const_range_type( const concurrent_skip_list& l)
            : my_end(l.end()), my_begin(l.begin()), my_level(my_begin.my_node_ptr->height() ) {}

        iterator begin() const { return my_begin; }
        iterator end() const { return my_end; }
        size_t grainsize() const { return 1; }

    }; // class const_range_type

    class range_type : public const_range_type {
    public:
        using iterator = typename concurrent_skip_list::iterator;

        range_type(range_type& r, split) : const_range_type(r, split()) {}
        range_type(const concurrent_skip_list& l) : const_range_type(l) {}

        iterator begin() const {
            node_ptr node = const_range_type::begin().my_node_ptr;
            return iterator(node);
        }

        iterator end() const {
            node_ptr node = const_range_type::end().my_node_ptr;
            return iterator(node); }
    }; // class range_type

    range_type range() { return range_type(*this); }
    const_range_type range() const { return const_range_type(*this); }

private:
    void internal_move(concurrent_skip_list&& other) {
        dummy_head = other.dummy_head;
        other.dummy_head = nullptr;
        other.create_dummy_head();

        my_size = other.my_size.load();
        other.my_size = 0;
    }

    static const key_type& get_key(node_ptr n) {
        __TBB_ASSERT(n, NULL);
        return traits_type::get_key(n->value());
    }

    template <typename K>
    iterator internal_find(const K& key) {
        iterator it = lower_bound(key);
        return (it == end() || my_compare(key, traits_type::get_key(*it))) ? end() : it;
    }

    template <typename K>
    const_iterator internal_find(const K& key) const {
        const_iterator it = lower_bound(key);
        return (it == end() || my_compare(key, traits_type::get_key(*it))) ? end() : it;
    }

    template <typename K>
    size_type internal_count( const K& key ) const {
        if (allow_multimapping) {
            std::pair<const_iterator, const_iterator> range = equal_range(key);
            return std::distance(range.first, range.second);
        }
        return (find(key) == end()) ? size_type(0) : size_type(1);
    }

    /**
     * Finds position on the @param level using @param cmp
     * @param level - on which level search prev node
     * @param prev - pointer to the start node to search
     * @param key - key to search
     * @param cmp - callable object to compare two objects
     *  (my_compare member is default comparator)
     * @returns pointer to the node which is not satisfy the comparison with @param key
     */
    template <typename K, typename pointer_type, typename comparator>
    pointer_type internal_find_position( size_type level, pointer_type& prev, const K& key,
                                         const comparator& cmp) const {
        __TBB_ASSERT(level < prev->height(), "Wrong level to find position");
        pointer_type curr = prev->next(level);

        while (curr && cmp(get_key(curr), key)) {
            prev = curr;
            __TBB_ASSERT(level < prev->height(), NULL);
            curr = prev->next(level);
        }

        return curr;
    }

    template <typename comparator>
    void fill_prev_next_arrays(array_type& prev_nodes, array_type& next_nodes, node_ptr prev, const key_type& key,
                               const comparator& cmp) {
        prev_nodes.fill(dummy_head);
        next_nodes.fill(nullptr);

        for (size_type h = prev->height(); h > 0; --h) {
            node_ptr next = internal_find_position(h - 1, prev, key, cmp);
            prev_nodes[h - 1] = prev;
            next_nodes[h - 1] = next;
        }
    }

    template <typename comparator>
    void fill_prev_next_by_ptr(array_type& prev_nodes, array_type& next_nodes, const_iterator it, const key_type& key,
                               const comparator& cmp) {
        node_ptr prev = dummy_head;
        node_ptr erase_node = it.my_node_ptr;
        size_type node_height = erase_node->height();

        for (size_type h = prev->height(); h >= node_height; --h) {
            internal_find_position(h - 1, prev, key, cmp);
        }

        for (size_type h = node_height; h > 0; --h) {
            node_ptr curr = prev->next(h - 1);
            while (const_iterator(curr) != it) {
                prev = curr;
                curr = prev->next(h - 1);
            }
            prev_nodes[h - 1] = prev;
        }

        std::fill(next_nodes.begin(), next_nodes.begin() + node_height, erase_node);
    }

    template<typename... Args>
    std::pair<iterator, bool> internal_insert(Args&&... args) {
        node_ptr new_node = create_node(std::forward<Args>(args)...);
        std::pair<iterator, bool> insert_result = internal_insert_node(new_node);
        if(!insert_result.second) {
            delete_node(new_node);
        }
        return insert_result;
    }

    std::pair<iterator, bool> internal_insert_node(node_ptr new_node) {
        array_type prev_nodes;
        array_type next_nodes;
        __TBB_ASSERT(dummy_head->height() >= new_node->height(), "Wrong height for new node");

        do {
            if (allow_multimapping) {
                fill_prev_next_arrays(prev_nodes, next_nodes, dummy_head, get_key(new_node),
                                      not_greater_compare(my_compare));
            } else {
                fill_prev_next_arrays(prev_nodes, next_nodes, dummy_head, get_key(new_node), my_compare);
            }

            node_ptr next = next_nodes[0];
            if (next && !allow_multimapping && !my_compare(get_key(new_node), get_key(next))) {
                // TODO: do we really need to wait?
                while (!next->fully_linked()) {
                    // TODO: atomic backoff
                }

                return std::pair<iterator, bool>(iterator(next), false);
            }
            __TBB_ASSERT(allow_multimapping || !next || my_compare(get_key(new_node), get_key(next)),
                         "Wrong elements order");

        } while (!try_insert_node(new_node, prev_nodes, next_nodes));

        __TBB_ASSERT(new_node, NULL);
        return std::pair<iterator, bool>(iterator(new_node), true);
    }

    bool try_insert_node(node_ptr new_node, array_type& prev_nodes, array_type& next_nodes) {
        __TBB_ASSERT(dummy_head->height() >= new_node->height(), NULL);

        lock_array locks;

        if (!try_lock_nodes(new_node->height(), prev_nodes, next_nodes, locks)) {
            return false;
        }

        __TBB_ASSERT(allow_multimapping ||
                    ((prev_nodes[0] == dummy_head ||
                      my_compare(get_key(prev_nodes[0]), get_key(new_node))) &&
                      (next_nodes[0] == nullptr || my_compare(get_key(new_node), get_key(next_nodes[0])))),
                    "Wrong elements order");

        for (size_type level = 0; level < new_node->height(); ++level) {
            __TBB_ASSERT(prev_nodes[level]->height() > level, NULL);
            __TBB_ASSERT(prev_nodes[level]->next(level) == next_nodes[level], NULL);
            new_node->set_next(level, next_nodes[level]);
            prev_nodes[level]->set_next(level, new_node);
        }
        new_node->mark_linked();

        ++my_size;

        return true;
    }

    bool try_lock_nodes(size_type height, array_type& prevs, array_type& next_nodes, lock_array& locks) {
        for (size_type l = 0; l < height; ++l) {
            if (l == 0 || prevs[l] != prevs[l - 1])
                locks[l] = prevs[l]->acquire();

            node_ptr next = prevs[l]->next(l);
            if ( next != next_nodes[l]) return false;
        }

        return true;
    }

    template <typename K, typename comparator>
    const_iterator internal_get_bound(const K& key, const comparator& cmp) const {
        node_ptr prev = dummy_head;
        __TBB_ASSERT(dummy_head->height() > 0, NULL);
        node_ptr next = nullptr;

        for (size_type h = prev->height(); h > 0; --h) {
            next = internal_find_position(h - 1, prev, key, cmp);
        }

        return const_iterator(next);
    }

    template <typename K, typename comparator>
    iterator internal_get_bound(const K& key, const comparator& cmp){
        node_ptr prev = dummy_head;
        __TBB_ASSERT(dummy_head->height() > 0, NULL);
        node_ptr next = nullptr;

        for (size_type h = prev->height(); h > 0; --h) {
            next = internal_find_position(h - 1, prev, key, cmp);
        }

        return iterator(next);
    }

    // Returns node_ptr to the extracted node and node_ptr to the next node after the extracted
    std::pair<node_ptr, node_ptr> internal_extract(const_iterator it) {
        if ( it != end() ) {
            key_type key = traits_type::get_key(*it);
            __TBB_ASSERT(dummy_head->height() > 0, NULL);

            array_type prev_nodes;
            array_type next_nodes;

            fill_prev_next_by_ptr(prev_nodes, next_nodes, it, key, my_compare);

            node_ptr erase_node = next_nodes[0];
            __TBB_ASSERT(erase_node != nullptr, NULL);
            node_ptr next_node = erase_node->next(0);

            if (!my_compare(key, get_key(erase_node))) {
                for(size_type level = 0; level < erase_node->height(); ++level) {
                    __TBB_ASSERT(prev_nodes[level]->height() > level, NULL);
                    __TBB_ASSERT(next_nodes[level] == erase_node, NULL);
                    prev_nodes[level]->set_next(level, erase_node->next(level));
                }
                --my_size;
                return std::pair<node_ptr, node_ptr>(erase_node, next_node);
            }
        }
        return std::pair<node_ptr, node_ptr>(nullptr, nullptr);
    }

protected:
    template<typename SourceType>
    void internal_merge(SourceType&& source) {
        using source_type = typename std::decay<SourceType>::type;
        using source_iterator = typename source_type::iterator;
        __TBB_STATIC_ASSERT((std::is_same<node_type, typename source_type::node_type>::value), "Incompatible containers cannot be merged");

        for(source_iterator it = source.begin(); it != source.end();) {
            source_iterator where = it++;
            if (allow_multimapping || !contains(traits_type::get_key(*where))) {
                std::pair<node_ptr, node_ptr> extract_result = source.internal_extract(where);

                //If the insertion fails - return the node into source
                node_type handle(extract_result.first);
                __TBB_ASSERT(!handle.empty(), "Extracted handle in merge is empty");

                if (!insert(std::move(handle)).second) {
                    source.insert(std::move(handle));
                }
                handle.deactivate();
            }
        }
    }

private:
    void internal_copy(const concurrent_skip_list& other) {
        internal_copy(other.begin(), other.end());
    }

    template<typename Iterator>
    void internal_copy(Iterator first, Iterator last) {
        clear();
        try {
            for (auto it = first; it != last; ++it)
                insert(*it);
        }
        catch (...) {
            clear();
            delete_dummy_head();
            throw;
        }
    }

    /** Generate random level */
    size_type random_level() {
        return my_rnd_generator();
    }

    static size_type calc_node_size(size_type height) {
        return sizeof(list_node_type) + height*sizeof(typename list_node_type::atomic_node_pointer);
    }

    /** Creates new node */
    template <typename... Args>
    node_ptr create_node(Args&&... args) {
        size_type levels = random_level();

        size_type sz = calc_node_size(levels);

        node_ptr node = reinterpret_cast<node_ptr>(node_allocator_traits::allocate(my_node_allocator, sz));

        try {
            node_allocator_traits::construct(my_node_allocator, node, levels);

        }
        catch(...) {
            deallocate_node(node, sz);
            throw;
        }

        try {
            node_allocator_traits::construct(my_node_allocator, node->storage(), std::forward<Args>(args)...);
        }
        catch (...) {
            node_allocator_traits::destroy(my_node_allocator, node);
            deallocate_node(node, sz);
            throw;
        }

        return node;
    }

    void create_dummy_head() {
        size_type sz = calc_node_size(MAX_LEVEL);

        dummy_head = reinterpret_cast<node_ptr>(node_allocator_traits::allocate(my_node_allocator, sz));
        // TODO: investigate linkage fail in debug without this workaround
        auto max_level = MAX_LEVEL;

        try {
            node_allocator_traits::construct(my_node_allocator, dummy_head, max_level);
        }
        catch(...) {
            deallocate_node(dummy_head, sz);
            throw;
        }
    }

    template <bool is_dummy = false>
    void delete_node(node_ptr node) {
        size_type sz = calc_node_size(node->height());
        // Destroy value
        if (!is_dummy) node_allocator_traits::destroy(my_node_allocator, node->storage());
        // Destroy node
        node_allocator_traits::destroy(my_node_allocator, node);
        // Deallocate memory
        deallocate_node(node, sz);
    }

    void deallocate_node(node_ptr node, size_type sz) {
        node_allocator_traits::deallocate(my_node_allocator, reinterpret_cast<uint8_t*>(node), sz);
    }

    void delete_dummy_head() {
        delete_node<true>(dummy_head);
    }

    static iterator get_iterator(const_iterator it) {
        return iterator(it.my_node_ptr);
    }

    void internal_move_assign(concurrent_skip_list&& other, /*POCMA=*/std::true_type) {
        delete_dummy_head();
        tbb::internal::allocator_move_assignment(my_node_allocator, other.my_node_allocator, std::true_type());
        internal_move(std::move(other));
    }

    void internal_move_assign(concurrent_skip_list&& other, /*POCMA=*/std::false_type) {
        if (my_node_allocator == other.my_node_allocator) {
            delete_dummy_head();
            internal_move(std::move(other));
        } else {
            internal_copy(std::make_move_iterator(other.begin()), std::make_move_iterator(other.end()));
        }
    }

    struct not_greater_compare {
        const key_compare& my_less_compare;

        not_greater_compare(const key_compare& less_compare) : my_less_compare(less_compare) {}

        template <typename K1, typename K2>
        bool operator()(const K1& first, const K2& second) const {
            return !my_less_compare(second, first);
        }
    };

    node_allocator_type my_node_allocator;
    key_compare my_compare;
    random_level_generator_type my_rnd_generator;
    node_ptr dummy_head;

    template<typename OtherTraits>
    friend class concurrent_skip_list;

    std::atomic<size_type> my_size;
}; // class concurrent_skip_list

template <size_t MAX_LEVEL>
class concurrent_geometric_level_generator {
public:
    static constexpr size_t max_level = MAX_LEVEL;

    concurrent_geometric_level_generator() : engines(time(NULL)) {}

    size_t operator()() {
        return (distribution(engines.local()) % MAX_LEVEL) + 1;
    }

private:
    tbb::enumerable_thread_specific<std::mt19937_64> engines;
    std::geometric_distribution<size_t> distribution;
};

} // namespace internal
} // namespace interface10
} // namespace tbb

#endif // __TBB_concurrent_skip_list_H
