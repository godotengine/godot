
//              Copyright Catch2 Authors
// Distributed under the Boost Software License, Version 1.0.
//   (See accompanying file LICENSE.txt or copy at
//        https://www.boost.org/LICENSE_1_0.txt)

// SPDX-License-Identifier: BSL-1.0
#ifndef CATCH_APPROX_HPP_INCLUDED
#define CATCH_APPROX_HPP_INCLUDED

#include <catch2/catch_tostring.hpp>

#include <type_traits>

namespace Catch {

    class Approx {
    private:
        bool equalityComparisonImpl(double other) const;
        // Sets and validates the new margin (margin >= 0)
        void setMargin(double margin);
        // Sets and validates the new epsilon (0 < epsilon < 1)
        void setEpsilon(double epsilon);

    public:
        explicit Approx ( double value );

        static Approx custom();

        Approx operator-() const;

        template <typename T, typename = std::enable_if_t<std::is_constructible<double, T>::value>>
        Approx operator()( T const& value ) const {
            Approx approx( static_cast<double>(value) );
            approx.m_epsilon = m_epsilon;
            approx.m_margin = m_margin;
            approx.m_scale = m_scale;
            return approx;
        }

        template <typename T, typename = std::enable_if_t<std::is_constructible<double, T>::value>>
        explicit Approx( T const& value ): Approx(static_cast<double>(value))
        {}


        template <typename T, typename = std::enable_if_t<std::is_constructible<double, T>::value>>
        friend bool operator == ( const T& lhs, Approx const& rhs ) {
            auto lhs_v = static_cast<double>(lhs);
            return rhs.equalityComparisonImpl(lhs_v);
        }

        template <typename T, typename = std::enable_if_t<std::is_constructible<double, T>::value>>
        friend bool operator == ( Approx const& lhs, const T& rhs ) {
            return operator==( rhs, lhs );
        }

        template <typename T, typename = std::enable_if_t<std::is_constructible<double, T>::value>>
        friend bool operator != ( T const& lhs, Approx const& rhs ) {
            return !operator==( lhs, rhs );
        }

        template <typename T, typename = std::enable_if_t<std::is_constructible<double, T>::value>>
        friend bool operator != ( Approx const& lhs, T const& rhs ) {
            return !operator==( rhs, lhs );
        }

        template <typename T, typename = std::enable_if_t<std::is_constructible<double, T>::value>>
        friend bool operator <= ( T const& lhs, Approx const& rhs ) {
            return static_cast<double>(lhs) < rhs.m_value || lhs == rhs;
        }

        template <typename T, typename = std::enable_if_t<std::is_constructible<double, T>::value>>
        friend bool operator <= ( Approx const& lhs, T const& rhs ) {
            return lhs.m_value < static_cast<double>(rhs) || lhs == rhs;
        }

        template <typename T, typename = std::enable_if_t<std::is_constructible<double, T>::value>>
        friend bool operator >= ( T const& lhs, Approx const& rhs ) {
            return static_cast<double>(lhs) > rhs.m_value || lhs == rhs;
        }

        template <typename T, typename = std::enable_if_t<std::is_constructible<double, T>::value>>
        friend bool operator >= ( Approx const& lhs, T const& rhs ) {
            return lhs.m_value > static_cast<double>(rhs) || lhs == rhs;
        }

        template <typename T, typename = std::enable_if_t<std::is_constructible<double, T>::value>>
        Approx& epsilon( T const& newEpsilon ) {
            const auto epsilonAsDouble = static_cast<double>(newEpsilon);
            setEpsilon(epsilonAsDouble);
            return *this;
        }

        template <typename T, typename = std::enable_if_t<std::is_constructible<double, T>::value>>
        Approx& margin( T const& newMargin ) {
            const auto marginAsDouble = static_cast<double>(newMargin);
            setMargin(marginAsDouble);
            return *this;
        }

        template <typename T, typename = std::enable_if_t<std::is_constructible<double, T>::value>>
        Approx& scale( T const& newScale ) {
            m_scale = static_cast<double>(newScale);
            return *this;
        }

        std::string toString() const;

    private:
        double m_epsilon;
        double m_margin;
        double m_scale;
        double m_value;
    };

namespace literals {
    Approx operator ""_a(long double val);
    Approx operator ""_a(unsigned long long val);
} // end namespace literals

template<>
struct StringMaker<Catch::Approx> {
    static std::string convert(Catch::Approx const& value);
};

} // end namespace Catch

#endif // CATCH_APPROX_HPP_INCLUDED
