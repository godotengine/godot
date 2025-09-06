
//              Copyright Catch2 Authors
// Distributed under the Boost Software License, Version 1.0.
//   (See accompanying file LICENSE.txt or copy at
//        https://www.boost.org/LICENSE_1_0.txt)

// SPDX-License-Identifier: BSL-1.0
#ifndef CATCH_MATCHERS_VECTOR_HPP_INCLUDED
#define CATCH_MATCHERS_VECTOR_HPP_INCLUDED

#include <catch2/matchers/catch_matchers.hpp>
#include <catch2/catch_approx.hpp>

#include <algorithm>

namespace Catch {
namespace Matchers {

    template<typename T, typename Alloc>
    class VectorContainsElementMatcher final : public MatcherBase<std::vector<T, Alloc>> {
        T const& m_comparator;

    public:
        VectorContainsElementMatcher(T const& comparator):
            m_comparator(comparator)
        {}

        bool match(std::vector<T, Alloc> const& v) const override {
            for (auto const& el : v) {
                if (el == m_comparator) {
                    return true;
                }
            }
            return false;
        }

        std::string describe() const override {
            return "Contains: " + ::Catch::Detail::stringify( m_comparator );
        }
    };

    template<typename T, typename AllocComp, typename AllocMatch>
    class ContainsMatcher final : public MatcherBase<std::vector<T, AllocMatch>> {
        std::vector<T, AllocComp> const& m_comparator;

    public:
        ContainsMatcher(std::vector<T, AllocComp> const& comparator):
            m_comparator( comparator )
        {}

        bool match(std::vector<T, AllocMatch> const& v) const override {
            // !TBD: see note in EqualsMatcher
            if (m_comparator.size() > v.size())
                return false;
            for (auto const& comparator : m_comparator) {
                auto present = false;
                for (const auto& el : v) {
                    if (el == comparator) {
                        present = true;
                        break;
                    }
                }
                if (!present) {
                    return false;
                }
            }
            return true;
        }
        std::string describe() const override {
            return "Contains: " + ::Catch::Detail::stringify( m_comparator );
        }
    };

    template<typename T, typename AllocComp, typename AllocMatch>
    class EqualsMatcher final : public MatcherBase<std::vector<T, AllocMatch>> {
        std::vector<T, AllocComp> const& m_comparator;

    public:
        EqualsMatcher(std::vector<T, AllocComp> const& comparator):
            m_comparator( comparator )
        {}

        bool match(std::vector<T, AllocMatch> const& v) const override {
            // !TBD: This currently works if all elements can be compared using !=
            // - a more general approach would be via a compare template that defaults
            // to using !=. but could be specialised for, e.g. std::vector<T> etc
            // - then just call that directly
            if ( m_comparator.size() != v.size() ) { return false; }
            for ( std::size_t i = 0; i < v.size(); ++i ) {
                if ( !( m_comparator[i] == v[i] ) ) { return false; }
            }
            return true;
        }
        std::string describe() const override {
            return "Equals: " + ::Catch::Detail::stringify( m_comparator );
        }
    };

    template<typename T, typename AllocComp, typename AllocMatch>
    class ApproxMatcher final : public MatcherBase<std::vector<T, AllocMatch>> {
        std::vector<T, AllocComp> const& m_comparator;
        mutable Catch::Approx approx = Catch::Approx::custom();

    public:
        ApproxMatcher(std::vector<T, AllocComp> const& comparator):
            m_comparator( comparator )
        {}

        bool match(std::vector<T, AllocMatch> const& v) const override {
            if (m_comparator.size() != v.size())
                return false;
            for (std::size_t i = 0; i < v.size(); ++i)
                if (m_comparator[i] != approx(v[i]))
                    return false;
            return true;
        }
        std::string describe() const override {
            return "is approx: " + ::Catch::Detail::stringify( m_comparator );
        }
        template <typename = std::enable_if_t<std::is_constructible<double, T>::value>>
        ApproxMatcher& epsilon( T const& newEpsilon ) {
            approx.epsilon(static_cast<double>(newEpsilon));
            return *this;
        }
        template <typename = std::enable_if_t<std::is_constructible<double, T>::value>>
        ApproxMatcher& margin( T const& newMargin ) {
            approx.margin(static_cast<double>(newMargin));
            return *this;
        }
        template <typename = std::enable_if_t<std::is_constructible<double, T>::value>>
        ApproxMatcher& scale( T const& newScale ) {
            approx.scale(static_cast<double>(newScale));
            return *this;
        }
    };

    template<typename T, typename AllocComp, typename AllocMatch>
    class UnorderedEqualsMatcher final : public MatcherBase<std::vector<T, AllocMatch>> {
        std::vector<T, AllocComp> const& m_target;

    public:
        UnorderedEqualsMatcher(std::vector<T, AllocComp> const& target):
            m_target(target)
        {}
        bool match(std::vector<T, AllocMatch> const& vec) const override {
            if (m_target.size() != vec.size()) {
                return false;
            }
            return std::is_permutation(m_target.begin(), m_target.end(), vec.begin());
        }

        std::string describe() const override {
            return "UnorderedEquals: " + ::Catch::Detail::stringify(m_target);
        }
    };


    // The following functions create the actual matcher objects.
    // This allows the types to be inferred

    //! Creates a matcher that matches vectors that contain all elements in `comparator`
    template<typename T, typename AllocComp = std::allocator<T>, typename AllocMatch = AllocComp>
    ContainsMatcher<T, AllocComp, AllocMatch> Contains( std::vector<T, AllocComp> const& comparator ) {
        return ContainsMatcher<T, AllocComp, AllocMatch>(comparator);
    }

    //! Creates a matcher that matches vectors that contain `comparator` as an element
    template<typename T, typename Alloc = std::allocator<T>>
    VectorContainsElementMatcher<T, Alloc> VectorContains( T const& comparator ) {
        return VectorContainsElementMatcher<T, Alloc>(comparator);
    }

    //! Creates a matcher that matches vectors that are exactly equal to `comparator`
    template<typename T, typename AllocComp = std::allocator<T>, typename AllocMatch = AllocComp>
    EqualsMatcher<T, AllocComp, AllocMatch> Equals( std::vector<T, AllocComp> const& comparator ) {
        return EqualsMatcher<T, AllocComp, AllocMatch>(comparator);
    }

    //! Creates a matcher that matches vectors that `comparator` as an element
    template<typename T, typename AllocComp = std::allocator<T>, typename AllocMatch = AllocComp>
    ApproxMatcher<T, AllocComp, AllocMatch> Approx( std::vector<T, AllocComp> const& comparator ) {
        return ApproxMatcher<T, AllocComp, AllocMatch>(comparator);
    }

    //! Creates a matcher that matches vectors that is equal to `target` modulo permutation
    template<typename T, typename AllocComp = std::allocator<T>, typename AllocMatch = AllocComp>
    UnorderedEqualsMatcher<T, AllocComp, AllocMatch> UnorderedEquals(std::vector<T, AllocComp> const& target) {
        return UnorderedEqualsMatcher<T, AllocComp, AllocMatch>(target);
    }

} // namespace Matchers
} // namespace Catch

#endif // CATCH_MATCHERS_VECTOR_HPP_INCLUDED
