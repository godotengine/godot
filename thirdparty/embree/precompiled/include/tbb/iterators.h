/*
    Copyright (c) 2017-2020 Intel Corporation

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

#ifndef __TBB_iterators_H
#define __TBB_iterators_H

#include <iterator>
#include <limits>

#include "tbb_config.h"
#include "tbb_stddef.h"

#if __TBB_CPP11_PRESENT

#include <type_traits>

namespace tbb {

template <typename IntType>
class counting_iterator {
    __TBB_STATIC_ASSERT(std::numeric_limits<IntType>::is_integer, "Cannot instantiate counting_iterator with a non-integer type");
public:
    typedef typename std::make_signed<IntType>::type difference_type;
    typedef IntType value_type;
    typedef const IntType* pointer;
    typedef const IntType& reference;
    typedef std::random_access_iterator_tag iterator_category;

    counting_iterator() : my_counter() {}
    explicit counting_iterator(IntType init) : my_counter(init) {}

    reference operator*() const { return my_counter; }
    value_type operator[](difference_type i) const { return *(*this + i); }

    difference_type operator-(const counting_iterator& it) const { return my_counter - it.my_counter; }

    counting_iterator& operator+=(difference_type forward) { my_counter += forward; return *this; }
    counting_iterator& operator-=(difference_type backward) { return *this += -backward; }
    counting_iterator& operator++() { return *this += 1; }
    counting_iterator& operator--() { return *this -= 1; }

    counting_iterator operator++(int) {
        counting_iterator it(*this);
        ++(*this);
        return it;
    }
    counting_iterator operator--(int) {
        counting_iterator it(*this);
        --(*this);
        return it;
    }

    counting_iterator operator-(difference_type backward) const { return counting_iterator(my_counter - backward); }
    counting_iterator operator+(difference_type forward) const { return counting_iterator(my_counter + forward); }
    friend counting_iterator operator+(difference_type forward, const counting_iterator it) { return it + forward; }

    bool operator==(const counting_iterator& it) const { return *this - it == 0; }
    bool operator!=(const counting_iterator& it) const { return !(*this == it); }
    bool operator<(const counting_iterator& it) const {return *this - it < 0; }
    bool operator>(const counting_iterator& it) const { return it < *this; }
    bool operator<=(const counting_iterator& it) const { return !(*this > it); }
    bool operator>=(const counting_iterator& it) const { return !(*this < it); }

private:
    IntType my_counter;
};
} //namespace tbb


#include <tuple>

#include "internal/_template_helpers.h" // index_sequence, make_index_sequence

namespace tbb {
namespace internal {

template<size_t N>
struct tuple_util {
    template<typename TupleType, typename DifferenceType>
    static void increment(TupleType& it, DifferenceType forward) {
        std::get<N-1>(it) += forward;
        tuple_util<N-1>::increment(it, forward);
    }
    template<typename TupleType, typename DifferenceType>
    static bool check_sync(const TupleType& it1, const TupleType& it2, DifferenceType val) {
        if(std::get<N-1>(it1) - std::get<N-1>(it2) != val)
            return false;
        return tuple_util<N-1>::check_sync(it1, it2, val);
    }
};

template<>
struct tuple_util<0> {
    template<typename TupleType, typename DifferenceType>
    static void increment(TupleType&, DifferenceType) {}
    template<typename TupleType, typename DifferenceType>
    static bool check_sync(const TupleType&, const TupleType&, DifferenceType) { return true;}
};

template <typename TupleReturnType>
struct make_references {
    template <typename TupleType, std::size_t... Is>
    TupleReturnType operator()(const TupleType& t, tbb::internal::index_sequence<Is...>) {
        return std::tie( *std::get<Is>(t)... );
    }
};

// A simple wrapper over a tuple of references.
// The class is designed to hold a temporary tuple of reference
// after dereferencing a zip_iterator; in particular, it is needed
// to swap these rvalue tuples. Any other usage is not supported.
template<typename... T>
struct tuplewrapper : public std::tuple<typename std::enable_if<std::is_reference<T>::value, T&&>::type...> {
    // In the context of this class, T is a reference, so T&& is a "forwarding reference"
    typedef std::tuple<T&&...> base_type;
    // Construct from the result of std::tie
    tuplewrapper(const base_type& in) : base_type(in) {}
#if __INTEL_COMPILER
    // ICC cannot generate copy ctor & assignment
    tuplewrapper(const tuplewrapper& rhs) : base_type(rhs) {}
    tuplewrapper& operator=(const tuplewrapper& rhs) {
        *this = base_type(rhs);
        return *this;
    }
#endif
    // Assign any tuple convertible to std::tuple<T&&...>: *it = a_tuple;
    template<typename... U>
    tuplewrapper& operator=(const std::tuple<U...>& other) {
        base_type::operator=(other);
        return *this;
    }
#if _LIBCPP_VERSION
    // (Necessary for libc++ tuples) Convert to a tuple of values: v = *it;
    operator std::tuple<typename std::remove_reference<T>::type...>() { return base_type(*this); }
#endif
    // Swap rvalue tuples: swap(*it1,*it2);
    friend void swap(tuplewrapper&& a, tuplewrapper&& b) {
        std::swap<T&&...>(a,b);
    }
};

} //namespace internal

template <typename... Types>
class zip_iterator {
    __TBB_STATIC_ASSERT(sizeof...(Types)>0, "Cannot instantiate zip_iterator with empty template parameter pack");
    static const std::size_t num_types = sizeof...(Types);
    typedef std::tuple<Types...> it_types;
public:
    typedef typename std::make_signed<std::size_t>::type difference_type;
    typedef std::tuple<typename std::iterator_traits<Types>::value_type...> value_type;
#if __INTEL_COMPILER && __INTEL_COMPILER < 1800 && _MSC_VER
    typedef std::tuple<typename std::iterator_traits<Types>::reference...> reference;
#else
    typedef tbb::internal::tuplewrapper<typename std::iterator_traits<Types>::reference...> reference;
#endif
    typedef std::tuple<typename std::iterator_traits<Types>::pointer...> pointer;
    typedef std::random_access_iterator_tag iterator_category;

    zip_iterator() : my_it() {}
    explicit zip_iterator(Types... args) : my_it(std::make_tuple(args...)) {}
    zip_iterator(const zip_iterator& input) : my_it(input.my_it) {}
    zip_iterator& operator=(const zip_iterator& input) {
        my_it = input.my_it;
        return *this;
    }

    reference operator*() const {
        return tbb::internal::make_references<reference>()(my_it, tbb::internal::make_index_sequence<num_types>());
    }
    reference operator[](difference_type i) const { return *(*this + i); }

    difference_type operator-(const zip_iterator& it) const {
        __TBB_ASSERT(internal::tuple_util<num_types>::check_sync(my_it, it.my_it, std::get<0>(my_it) - std::get<0>(it.my_it)),
                     "Components of zip_iterator are not synchronous");
        return std::get<0>(my_it) - std::get<0>(it.my_it);
    }

    zip_iterator& operator+=(difference_type forward) {
        internal::tuple_util<num_types>::increment(my_it, forward);
        return *this;
    }
    zip_iterator& operator-=(difference_type backward) { return *this += -backward; }
    zip_iterator& operator++() { return *this += 1; }
    zip_iterator& operator--() { return *this -= 1; }

    zip_iterator operator++(int) {
        zip_iterator it(*this);
        ++(*this);
        return it;
    }
    zip_iterator operator--(int) {
        zip_iterator it(*this);
        --(*this);
        return it;
    }

    zip_iterator operator-(difference_type backward) const {
        zip_iterator it(*this);
        return it -= backward;
    }
    zip_iterator operator+(difference_type forward) const {
        zip_iterator it(*this);
        return it += forward;
    }
    friend zip_iterator operator+(difference_type forward, const zip_iterator& it) { return it + forward; }

    bool operator==(const zip_iterator& it) const {
        return *this - it == 0;
    }
    it_types base() const { return my_it; }

    bool operator!=(const zip_iterator& it) const { return !(*this == it); }
    bool operator<(const zip_iterator& it) const { return *this - it < 0; }
    bool operator>(const zip_iterator& it) const { return it < *this; }
    bool operator<=(const zip_iterator& it) const { return !(*this > it); }
    bool operator>=(const zip_iterator& it) const { return !(*this < it); }
private:
    it_types my_it;
};

template<typename... T>
zip_iterator<T...> make_zip_iterator(T... args) { return zip_iterator<T...>(args...); }

template <typename UnaryFunc, typename Iter>
class transform_iterator {
public:
    typedef typename std::iterator_traits<Iter>::value_type value_type;
    typedef typename std::iterator_traits<Iter>::difference_type difference_type;
#if __TBB_CPP17_INVOKE_RESULT_PRESENT
    typedef typename std::invoke_result<UnaryFunc, typename std::iterator_traits<Iter>::reference>::type reference;
#else
    typedef typename std::result_of<UnaryFunc(typename std::iterator_traits<Iter>::reference)>::type reference;
#endif
    typedef typename std::iterator_traits<Iter>::pointer pointer;
    typedef typename std::random_access_iterator_tag iterator_category;

    transform_iterator(Iter it, UnaryFunc unary_func) : my_it(it), my_unary_func(unary_func) {
        __TBB_STATIC_ASSERT((std::is_same<typename std::iterator_traits<Iter>::iterator_category,
                             std::random_access_iterator_tag>::value), "Random access iterator required.");
    }
    transform_iterator(const transform_iterator& input) : my_it(input.my_it), my_unary_func(input.my_unary_func) { }
    transform_iterator& operator=(const transform_iterator& input) {
        my_it = input.my_it;
        return *this;
    }
    reference operator*() const {
        return my_unary_func(*my_it);
    }
    reference operator[](difference_type i) const {
        return *(*this + i);
    }
    transform_iterator& operator++() {
        ++my_it;
        return *this;
    }
    transform_iterator& operator--() {
        --my_it;
        return *this;
    }
    transform_iterator operator++(int) {
        transform_iterator it(*this);
        ++(*this);
        return it;
    }
    transform_iterator operator--(int) {
        transform_iterator it(*this);
        --(*this);
        return it;
    }
    transform_iterator operator+(difference_type forward) const {
        return { my_it + forward, my_unary_func };
    }
    transform_iterator operator-(difference_type backward) const {
        return { my_it - backward, my_unary_func };
    }
    transform_iterator& operator+=(difference_type forward) {
        my_it += forward;
        return *this;
    }
    transform_iterator& operator-=(difference_type backward) {
        my_it -= backward;
        return *this;
    }
    friend transform_iterator operator+(difference_type forward, const transform_iterator& it) {
        return it + forward;
    }
    difference_type operator-(const transform_iterator& it) const {
        return my_it - it.my_it;
    }
    bool operator==(const transform_iterator& it) const { return *this - it == 0; }
    bool operator!=(const transform_iterator& it) const { return !(*this == it); }
    bool operator<(const transform_iterator& it) const { return *this - it < 0; }
    bool operator>(const transform_iterator& it) const { return it < *this; }
    bool operator<=(const transform_iterator& it) const { return !(*this > it); }
    bool operator>=(const transform_iterator& it) const { return !(*this < it); }

    Iter base() const { return my_it; }
private:
    Iter my_it;
    const UnaryFunc my_unary_func;
};

template<typename UnaryFunc, typename Iter>
transform_iterator<UnaryFunc, Iter> make_transform_iterator(Iter it, UnaryFunc unary_func) {
    return transform_iterator<UnaryFunc, Iter>(it, unary_func);
}

} //namespace tbb

#endif //__TBB_CPP11_PRESENT

#endif /* __TBB_iterators_H */
