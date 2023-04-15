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

/* Container implementations in this header are based on PPL implementations
   provided by Microsoft. */

#ifndef __TBB_concurrent_unordered_map_H
#define __TBB_concurrent_unordered_map_H

#define __TBB_concurrent_unordered_map_H_include_area
#include "internal/_warning_suppress_enable_notice.h"

#include "internal/_concurrent_unordered_impl.h"

namespace tbb
{

namespace interface5 {

// Template class for hash map traits
template<typename Key, typename T, typename Hash_compare, typename Allocator, bool Allow_multimapping>
class concurrent_unordered_map_traits
{
protected:
    typedef std::pair<const Key, T> value_type;
    typedef Key key_type;
    typedef Hash_compare hash_compare;
    typedef typename tbb::internal::allocator_rebind<Allocator, value_type>::type allocator_type;
#if __TBB_UNORDERED_NODE_HANDLE_PRESENT
    typedef tbb::internal::node_handle<key_type, value_type,
                                  typename internal::split_ordered_list<value_type, allocator_type>::node,
                                  allocator_type> node_type;
#endif // __TBB_UNORDERED_NODE_HANDLE_PRESENT

    enum { allow_multimapping = Allow_multimapping };

    concurrent_unordered_map_traits() : my_hash_compare() {}
    concurrent_unordered_map_traits(const hash_compare& hc) : my_hash_compare(hc) {}

    template<class Type1, class Type2>
    static const Key& get_key(const std::pair<Type1, Type2>& value) {
        return (value.first);
    }

    hash_compare my_hash_compare; // the comparator predicate for keys
};

template<typename Key, typename T, typename Hasher, typename Key_equality, typename Allocator>
class concurrent_unordered_multimap;

template <typename Key, typename T, typename Hasher = tbb::tbb_hash<Key>, typename Key_equality = std::equal_to<Key>,
         typename Allocator = tbb::tbb_allocator<std::pair<const Key, T> > >
class concurrent_unordered_map :
    public internal::concurrent_unordered_base< concurrent_unordered_map_traits<Key, T,
    internal::hash_compare<Key, Hasher, Key_equality>, Allocator, false> >
{
    // Base type definitions
    typedef internal::hash_compare<Key, Hasher, Key_equality> hash_compare;
    typedef concurrent_unordered_map_traits<Key, T, hash_compare, Allocator, false> traits_type;
    typedef internal::concurrent_unordered_base< traits_type > base_type;
#if __TBB_EXTRA_DEBUG
public:
#endif
    using traits_type::allow_multimapping;
public:
    using base_type::end;
    using base_type::find;
    using base_type::insert;

    // Type definitions
    typedef Key key_type;
    typedef typename base_type::value_type value_type;
    typedef T mapped_type;
    typedef Hasher hasher;
    typedef Key_equality key_equal;
    typedef hash_compare key_compare;

    typedef typename base_type::allocator_type allocator_type;
    typedef typename base_type::pointer pointer;
    typedef typename base_type::const_pointer const_pointer;
    typedef typename base_type::reference reference;
    typedef typename base_type::const_reference const_reference;

    typedef typename base_type::size_type size_type;
    typedef typename base_type::difference_type difference_type;

    typedef typename base_type::iterator iterator;
    typedef typename base_type::const_iterator const_iterator;
    typedef typename base_type::iterator local_iterator;
    typedef typename base_type::const_iterator const_local_iterator;
#if __TBB_UNORDERED_NODE_HANDLE_PRESENT
    typedef typename base_type::node_type node_type;
#endif // __TBB_UNORDERED_NODE_HANDLE_PRESENT

    // Construction/destruction/copying
    explicit concurrent_unordered_map(size_type n_of_buckets = base_type::initial_bucket_number,
        const hasher& a_hasher = hasher(), const key_equal& a_keyeq = key_equal(),
        const allocator_type& a = allocator_type())
        : base_type(n_of_buckets, key_compare(a_hasher, a_keyeq), a)
    {}

    concurrent_unordered_map(size_type n_of_buckets, const allocator_type& a)
        : base_type(n_of_buckets, key_compare(hasher(), key_equal()), a)
    {}

    concurrent_unordered_map(size_type n_of_buckets, const hasher& a_hasher, const allocator_type& a)
        : base_type(n_of_buckets, key_compare(a_hasher, key_equal()), a)
    {}

    explicit concurrent_unordered_map(const Allocator& a) : base_type(base_type::initial_bucket_number, key_compare(), a)
    {}

    template <typename Iterator>
    concurrent_unordered_map(Iterator first, Iterator last, size_type n_of_buckets = base_type::initial_bucket_number,
        const hasher& a_hasher = hasher(), const key_equal& a_keyeq = key_equal(),
        const allocator_type& a = allocator_type())
        : base_type(n_of_buckets, key_compare(a_hasher, a_keyeq), a)
    {
        insert(first, last);
    }

    template <typename Iterator>
    concurrent_unordered_map(Iterator first, Iterator last, size_type n_of_buckets, const allocator_type& a)
        : base_type(n_of_buckets, key_compare(hasher(), key_equal()), a)
    {
        insert(first, last);
    }

    template <typename Iterator>
    concurrent_unordered_map(Iterator first, Iterator last, size_type n_of_buckets, const hasher& a_hasher,
        const allocator_type& a)
        : base_type(n_of_buckets, key_compare(a_hasher, key_equal()), a)
    {
        insert(first, last);
    }

#if __TBB_INITIALIZER_LISTS_PRESENT
    //! Constructor from initializer_list
    concurrent_unordered_map(std::initializer_list<value_type> il, size_type n_of_buckets = base_type::initial_bucket_number,
        const hasher& a_hasher = hasher(), const key_equal& a_keyeq = key_equal(),
        const allocator_type& a = allocator_type())
        : base_type(n_of_buckets, key_compare(a_hasher, a_keyeq), a)
    {
        insert(il.begin(),il.end());
    }

    concurrent_unordered_map(std::initializer_list<value_type> il, size_type n_of_buckets, const allocator_type& a)
        : base_type(n_of_buckets, key_compare(hasher(), key_equal()), a)
    {
        insert(il.begin(), il.end());
    }

    concurrent_unordered_map(std::initializer_list<value_type> il, size_type n_of_buckets, const hasher& a_hasher,
        const allocator_type& a)
        : base_type(n_of_buckets, key_compare(a_hasher, key_equal()), a)
    {
        insert(il.begin(), il.end());
    }

#endif //# __TBB_INITIALIZER_LISTS_PRESENT


#if __TBB_CPP11_RVALUE_REF_PRESENT && !__TBB_IMPLICIT_MOVE_PRESENT
    concurrent_unordered_map(const concurrent_unordered_map& table)
        : base_type(table)
    {}

    concurrent_unordered_map& operator=(const concurrent_unordered_map& table)
    {
        return static_cast<concurrent_unordered_map&>(base_type::operator=(table));
    }

    concurrent_unordered_map(concurrent_unordered_map&& table)
        : base_type(std::move(table))
    {}

    concurrent_unordered_map& operator=(concurrent_unordered_map&& table)
    {
        return static_cast<concurrent_unordered_map&>(base_type::operator=(std::move(table)));
    }
#endif //__TBB_CPP11_RVALUE_REF_PRESENT && !__TBB_IMPLICIT_MOVE_PRESENT

#if __TBB_CPP11_RVALUE_REF_PRESENT
    concurrent_unordered_map(concurrent_unordered_map&& table, const Allocator& a) : base_type(std::move(table), a)
    {}
#endif /*__TBB_CPP11_RVALUE_REF_PRESENT*/

#if __TBB_UNORDERED_NODE_HANDLE_PRESENT
    template<typename Hash, typename Equality>
    void merge(concurrent_unordered_map<Key, T, Hash, Equality, Allocator>& source)
              { this->internal_merge(source); }

    template<typename Hash, typename Equality>
    void merge(concurrent_unordered_map<Key, T, Hash, Equality, Allocator>&& source)
              { this->internal_merge(source); }

    template<typename Hash, typename Equality>
    void merge(concurrent_unordered_multimap<Key, T, Hash, Equality, Allocator>& source)
              { this->internal_merge(source); }

    template<typename Hash, typename Equality>
    void merge(concurrent_unordered_multimap<Key, T, Hash, Equality, Allocator>&& source)
              { this->internal_merge(source); }

#endif //__TBB_UNORDERED_NODE_HANDLE_PRESENT

    concurrent_unordered_map(const concurrent_unordered_map& table, const Allocator& a)
        : base_type(table, a)
    {}

    // Observers
    mapped_type& operator[](const key_type& key)
    {
        iterator where = find(key);

        if (where == end())
        {
            where = insert(std::pair<key_type, mapped_type>(key, mapped_type())).first;
        }

        return ((*where).second);
    }

    mapped_type& at(const key_type& key)
    {
        iterator where = find(key);

        if (where == end())
        {
            tbb::internal::throw_exception(tbb::internal::eid_invalid_key);
        }

        return ((*where).second);
    }

    const mapped_type& at(const key_type& key) const
    {
        const_iterator where = find(key);

        if (where == end())
        {
            tbb::internal::throw_exception(tbb::internal::eid_invalid_key);
        }

        return ((*where).second);
    }
};

#if __TBB_CPP17_DEDUCTION_GUIDES_PRESENT

namespace internal {
using namespace tbb::internal;

template<template<typename...> typename Map, typename Key, typename Element, typename... Args>
using cu_map_t = Map<
    Key, Element,
    std::conditional_t< (sizeof...(Args)>0) && !is_allocator_v< pack_element_t<0, Args...> >,
                        pack_element_t<0, Args...>, tbb_hash<Key> >,
    std::conditional_t< (sizeof...(Args)>1) && !is_allocator_v< pack_element_t<1, Args...> >,
                        pack_element_t<1, Args...>, std::equal_to<Key> >,
    std::conditional_t< (sizeof...(Args)>0) && is_allocator_v< pack_element_t<sizeof...(Args)-1, Args...> >,
                        pack_element_t<sizeof...(Args)-1, Args...>, tbb_allocator<std::pair<const Key, Element> > >
>;
}

// Deduction guide for the constructor from two iterators
template<typename I>
concurrent_unordered_map (I, I)
-> internal::cu_map_t<concurrent_unordered_map, internal::iterator_key_t<I>, internal::iterator_mapped_t<I>>;

// Deduction guide for the constructor from two iterators and hasher/equality/allocator
template<typename I, typename... Args>
concurrent_unordered_map(I, I, size_t, Args...)
-> internal::cu_map_t<concurrent_unordered_map, internal::iterator_key_t<I>, internal::iterator_mapped_t<I>, Args...>;

// Deduction guide for the constructor from an initializer_list
template<typename Key, typename Element>
concurrent_unordered_map(std::initializer_list<std::pair<const Key, Element>>)
-> internal::cu_map_t<concurrent_unordered_map, Key, Element>;

// Deduction guide for the constructor from an initializer_list and hasher/equality/allocator
template<typename Key, typename Element, typename... Args>
concurrent_unordered_map(std::initializer_list<std::pair<const Key, Element>>, size_t, Args...)
-> internal::cu_map_t<concurrent_unordered_map, Key, Element, Args...>;

#endif /* __TBB_CPP17_DEDUCTION_GUIDES_PRESENT */

template < typename Key, typename T, typename Hasher = tbb::tbb_hash<Key>, typename Key_equality = std::equal_to<Key>,
        typename Allocator = tbb::tbb_allocator<std::pair<const Key, T> > >
class concurrent_unordered_multimap :
    public internal::concurrent_unordered_base< concurrent_unordered_map_traits< Key, T,
    internal::hash_compare<Key, Hasher, Key_equality>, Allocator, true> >
{
    // Base type definitions
    typedef internal::hash_compare<Key, Hasher, Key_equality> hash_compare;
    typedef concurrent_unordered_map_traits<Key, T, hash_compare, Allocator, true> traits_type;
    typedef internal::concurrent_unordered_base<traits_type> base_type;
#if __TBB_EXTRA_DEBUG
public:
#endif
    using traits_type::allow_multimapping;
public:
    using base_type::insert;

    // Type definitions
    typedef Key key_type;
    typedef typename base_type::value_type value_type;
    typedef T mapped_type;
    typedef Hasher hasher;
    typedef Key_equality key_equal;
    typedef hash_compare key_compare;

    typedef typename base_type::allocator_type allocator_type;
    typedef typename base_type::pointer pointer;
    typedef typename base_type::const_pointer const_pointer;
    typedef typename base_type::reference reference;
    typedef typename base_type::const_reference const_reference;

    typedef typename base_type::size_type size_type;
    typedef typename base_type::difference_type difference_type;

    typedef typename base_type::iterator iterator;
    typedef typename base_type::const_iterator const_iterator;
    typedef typename base_type::iterator local_iterator;
    typedef typename base_type::const_iterator const_local_iterator;
#if __TBB_UNORDERED_NODE_HANDLE_PRESENT
    typedef typename base_type::node_type node_type;
#endif //__TBB_UNORDERED_NODE_HANDLE_PRESENT

    // Construction/destruction/copying
    explicit concurrent_unordered_multimap(size_type n_of_buckets = base_type::initial_bucket_number,
        const hasher& a_hasher = hasher(), const key_equal& a_keyeq = key_equal(),
        const allocator_type& a = allocator_type())
        : base_type(n_of_buckets, key_compare(a_hasher, a_keyeq), a)
    {}

    concurrent_unordered_multimap(size_type n_of_buckets, const allocator_type& a)
        : base_type(n_of_buckets, key_compare(hasher(), key_equal()), a)
    {}

    concurrent_unordered_multimap(size_type n_of_buckets, const hasher& a_hasher, const allocator_type& a)
        : base_type(n_of_buckets, key_compare(a_hasher, key_equal()), a)
    {}

    explicit concurrent_unordered_multimap(const Allocator& a) : base_type(base_type::initial_bucket_number, key_compare(), a)
    {}

    template <typename Iterator>
    concurrent_unordered_multimap(Iterator first, Iterator last, size_type n_of_buckets = base_type::initial_bucket_number,
        const hasher& a_hasher = hasher(), const key_equal& a_keyeq = key_equal(),
        const allocator_type& a = allocator_type())
        : base_type(n_of_buckets,key_compare(a_hasher,a_keyeq), a)
    {
        insert(first, last);
    }

    template <typename Iterator>
    concurrent_unordered_multimap(Iterator first, Iterator last, size_type n_of_buckets, const allocator_type& a)
        : base_type(n_of_buckets, key_compare(hasher(), key_equal()), a)
    {
        insert(first, last);
    }

    template <typename Iterator>
    concurrent_unordered_multimap(Iterator first, Iterator last, size_type n_of_buckets, const hasher& a_hasher,
        const allocator_type& a)
        : base_type(n_of_buckets, key_compare(a_hasher, key_equal()), a)
    {
        insert(first, last);
    }

#if __TBB_INITIALIZER_LISTS_PRESENT
    //! Constructor from initializer_list
    concurrent_unordered_multimap(std::initializer_list<value_type> il, size_type n_of_buckets = base_type::initial_bucket_number,
        const hasher& a_hasher = hasher(), const key_equal& a_keyeq = key_equal(),
        const allocator_type& a = allocator_type())
        : base_type(n_of_buckets, key_compare(a_hasher, a_keyeq), a)
    {
        insert(il.begin(),il.end());
    }

    concurrent_unordered_multimap(std::initializer_list<value_type> il, size_type n_of_buckets, const allocator_type& a)
        : base_type(n_of_buckets, key_compare(hasher(), key_equal()), a)
    {
        insert(il.begin(), il.end());
    }

    concurrent_unordered_multimap(std::initializer_list<value_type> il, size_type n_of_buckets, const hasher& a_hasher,
        const allocator_type& a)
        : base_type(n_of_buckets, key_compare(a_hasher, key_equal()), a)
    {
        insert(il.begin(), il.end());
    }

#endif //# __TBB_INITIALIZER_LISTS_PRESENT

#if __TBB_CPP11_RVALUE_REF_PRESENT && !__TBB_IMPLICIT_MOVE_PRESENT
    concurrent_unordered_multimap(const concurrent_unordered_multimap& table)
        : base_type(table)
    {}

    concurrent_unordered_multimap& operator=(const concurrent_unordered_multimap& table)
    {
        return static_cast<concurrent_unordered_multimap&>(base_type::operator=(table));
    }

    concurrent_unordered_multimap(concurrent_unordered_multimap&& table)
        : base_type(std::move(table))
    {}

    concurrent_unordered_multimap& operator=(concurrent_unordered_multimap&& table)
    {
        return static_cast<concurrent_unordered_multimap&>(base_type::operator=(std::move(table)));
    }
#endif //__TBB_CPP11_RVALUE_REF_PRESENT && !__TBB_IMPLICIT_MOVE_PRESENT

#if __TBB_CPP11_RVALUE_REF_PRESENT
    concurrent_unordered_multimap(concurrent_unordered_multimap&& table, const Allocator& a) : base_type(std::move(table), a)
    {}
#endif /*__TBB_CPP11_RVALUE_REF_PRESENT*/

#if __TBB_UNORDERED_NODE_HANDLE_PRESENT
    template<typename Hash, typename Equality>
    void merge(concurrent_unordered_map<Key, T, Hash, Equality, Allocator>& source)
              { this->internal_merge(source); }

    template<typename Hash, typename Equality>
    void merge(concurrent_unordered_map<Key, T, Hash, Equality, Allocator>&& source)
              { this->internal_merge(source); }

    template<typename Hash, typename Equality>
    void merge(concurrent_unordered_multimap<Key, T, Hash, Equality, Allocator>& source)
              { this->internal_merge(source); }

    template<typename Hash, typename Equality>
    void merge(concurrent_unordered_multimap<Key, T, Hash, Equality, Allocator>&& source)
              { this->internal_merge(source); }

#endif //__TBB_UNORDERED_NODE_HANDLE_PRESENT

    concurrent_unordered_multimap(const concurrent_unordered_multimap& table, const Allocator& a)
        : base_type(table, a)
    {}
};

#if __TBB_CPP17_DEDUCTION_GUIDES_PRESENT

// Deduction guide for the constructor from two iterators
template<typename I>
concurrent_unordered_multimap (I, I)
-> internal::cu_map_t<concurrent_unordered_multimap, internal::iterator_key_t<I>, internal::iterator_mapped_t<I>>;

// Deduction guide for the constructor from two iterators and hasher/equality/allocator
template<typename I, typename... Args>
concurrent_unordered_multimap(I, I, size_t, Args...)
-> internal::cu_map_t<concurrent_unordered_multimap, internal::iterator_key_t<I>, internal::iterator_mapped_t<I>, Args...>;

// Deduction guide for the constructor from an initializer_list
template<typename Key, typename Element>
concurrent_unordered_multimap(std::initializer_list<std::pair<const Key, Element>>)
-> internal::cu_map_t<concurrent_unordered_multimap, Key, Element>;

// Deduction guide for the constructor from an initializer_list and hasher/equality/allocator
template<typename Key, typename Element, typename... Args>
concurrent_unordered_multimap(std::initializer_list<std::pair<const Key, Element>>, size_t, Args...)
-> internal::cu_map_t<concurrent_unordered_multimap, Key, Element, Args...>;

#endif /* __TBB_CPP17_DEDUCTION_GUIDES_PRESENT */
} // namespace interface5

using interface5::concurrent_unordered_map;
using interface5::concurrent_unordered_multimap;

} // namespace tbb

#include "internal/_warning_suppress_disable_notice.h"
#undef __TBB_concurrent_unordered_map_H_include_area

#endif// __TBB_concurrent_unordered_map_H
