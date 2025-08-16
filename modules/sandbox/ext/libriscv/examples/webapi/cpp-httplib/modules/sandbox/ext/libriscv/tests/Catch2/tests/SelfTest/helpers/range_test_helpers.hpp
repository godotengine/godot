
//              Copyright Catch2 Authors
// Distributed under the Boost Software License, Version 1.0.
//   (See accompanying file LICENSE.txt or copy at
//        https://www.boost.org/LICENSE_1_0.txt)

// SPDX-License-Identifier: BSL-1.0

#ifndef CATCH_TEST_HELPERS_RANGE_TEST_HELPERS_HPP_INCLUDED
#define CATCH_TEST_HELPERS_RANGE_TEST_HELPERS_HPP_INCLUDED

#include <catch2/catch_tostring.hpp>

#include <initializer_list>
#include <list>
#include <memory>
#include <vector>

namespace unrelated {
    template <typename T>
    class needs_ADL_begin {
        std::vector<T> m_elements;

    public:
        using iterator = typename std::vector<T>::iterator;
        using const_iterator = typename std::vector<T>::const_iterator;

        needs_ADL_begin( std::initializer_list<T> init ): m_elements( init ) {}

        const_iterator Begin() const { return m_elements.begin(); }
        const_iterator End() const { return m_elements.end(); }

        friend const_iterator begin( needs_ADL_begin const& lhs ) {
            return lhs.Begin();
        }
        friend const_iterator end( needs_ADL_begin const& rhs ) {
            return rhs.End();
        }
    };

    struct ADL_empty {
        bool Empty() const { return true; }

        friend bool empty( ADL_empty e ) { return e.Empty(); }
    };

    struct ADL_size {
        size_t sz() const { return 12; }
        friend size_t size( ADL_size s ) { return s.sz(); }
    };

} // namespace unrelated

#if defined( __clang__ )
#    pragma clang diagnostic push
#    pragma clang diagnostic ignored "-Wunused-function"
#endif

template <typename T>
class has_different_begin_end_types {
    // Using std::vector<T> leads to annoying issues when T is bool
    // so we just use list because the perf is not critical and ugh.
    std::list<T> m_elements;

    // Different type for the "end" iterator
    struct iterator_end {};
    // Fake-ish forward iterator that only compares to a different type
    class iterator {
        using underlying_iter = typename std::list<T>::const_iterator;
        underlying_iter m_start;
        underlying_iter m_end;

    public:
        iterator( underlying_iter start, underlying_iter end ):
            m_start( start ), m_end( end ) {}

        using iterator_category = std::forward_iterator_tag;
        using difference_type = std::ptrdiff_t;
        using value_type = T;
        using const_reference = T const&;
        using pointer = T const*;

        friend bool operator==( iterator iter, iterator_end ) {
            return iter.m_start == iter.m_end;
        }
        friend bool operator==(iterator lhs, iterator rhs) {
            return lhs.m_start == rhs.m_start && lhs.m_end == rhs.m_end;
        }
        friend bool operator!=( iterator iter, iterator_end ) {
            return iter.m_start != iter.m_end;
        }
        friend bool operator!=( iterator lhs, iterator rhs ) {
            return !( lhs == rhs );
        }
        iterator& operator++() {
            ++m_start;
            return *this;
        }
        iterator operator++( int ) {
            auto tmp( *this );
            ++m_start;
            return tmp;
        }
        const_reference operator*() const { return *m_start; }
        pointer operator->() const { return m_start; }
    };

public:
    explicit has_different_begin_end_types( std::initializer_list<T> init ):
        m_elements( init ) {}

    iterator begin() const { return { m_elements.begin(), m_elements.end() }; }

    iterator_end end() const { return {}; }
};

#if defined( __clang__ )
#    pragma clang diagnostic pop
#endif

template <typename T>
struct with_mocked_iterator_access {
    std::vector<T> m_elements;

    // use plain arrays to have nicer printouts with CHECK(...)
    mutable std::unique_ptr<bool[]> m_derefed;

    // We want to check which elements were dereferenced when iterating, so
    // we can check whether iterator-using code traverses range correctly
    template <bool is_const>
    class basic_iterator {
        template <typename U>
        using constify_t = std::conditional_t<is_const, std::add_const_t<U>, U>;

        constify_t<with_mocked_iterator_access>* m_origin;
        size_t m_origin_idx;

    public:
        using iterator_category = std::forward_iterator_tag;
        using difference_type = std::ptrdiff_t;
        using value_type = constify_t<T>;
        using const_reference = typename std::vector<T>::const_reference;
        using reference = typename std::vector<T>::reference;
        using pointer = typename std::vector<T>::pointer;

        basic_iterator( constify_t<with_mocked_iterator_access>* origin,
                        std::size_t origin_idx ):
            m_origin{ origin }, m_origin_idx{ origin_idx } {}

        friend bool operator==( basic_iterator lhs, basic_iterator rhs ) {
            return lhs.m_origin == rhs.m_origin &&
                   lhs.m_origin_idx == rhs.m_origin_idx;
        }
        friend bool operator!=( basic_iterator lhs, basic_iterator rhs ) {
            return !( lhs == rhs );
        }
        basic_iterator& operator++() {
            ++m_origin_idx;
            return *this;
        }
        basic_iterator operator++( int ) {
            auto tmp( *this );
            ++( *this );
            return tmp;
        }
        const_reference operator*() const {
            assert( m_origin_idx < m_origin->m_elements.size() &&
                    "Attempted to deref invalid position" );
            m_origin->m_derefed[m_origin_idx] = true;
            return m_origin->m_elements[m_origin_idx];
        }
        pointer operator->() const {
            assert( m_origin_idx < m_origin->m_elements.size() &&
                    "Attempted to deref invalid position" );
            return &m_origin->m_elements[m_origin_idx];
        }
    };

    using iterator = basic_iterator<false>;
    using const_iterator = basic_iterator<true>;

    with_mocked_iterator_access( std::initializer_list<T> init ):
        m_elements( init ),
        m_derefed( std::make_unique<bool[]>( m_elements.size() ) ) {}

    const_iterator begin() const { return { this, 0 }; }
    const_iterator end() const { return { this, m_elements.size() }; }
    iterator begin() { return { this, 0 }; }
    iterator end() { return { this, m_elements.size() }; }
};


namespace Catch {
    // make sure with_mocked_iterator_access is not considered a range by Catch,
    // so that below StringMaker is used instead of the default one for ranges
    template <typename T>
    struct is_range<with_mocked_iterator_access<T>> : std::false_type {};

    template <typename T>
    struct StringMaker<with_mocked_iterator_access<T>> {
        static std::string
        convert( with_mocked_iterator_access<T> const& access ) {
            // We have to avoid the type's iterators, because we check
            // their use in tests
            return ::Catch::Detail::stringify( access.m_elements );
        }
    };
} // namespace Catch

#endif // CATCH_TEST_HELPERS_RANGE_TEST_HELPERS_HPP_INCLUDED
