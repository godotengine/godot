
//              Copyright Catch2 Authors
// Distributed under the Boost Software License, Version 1.0.
//   (See accompanying file LICENSE.txt or copy at
//        https://www.boost.org/LICENSE_1_0.txt)

// SPDX-License-Identifier: BSL-1.0
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <catch2/internal/catch_enforce.hpp>
#include <catch2/internal/catch_polyfills.hpp>
#include <catch2/internal/catch_to_string.hpp>
#include <catch2/catch_tostring.hpp>
#include <catch2/internal/catch_floating_point_helpers.hpp>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstdint>
#include <sstream>
#include <iomanip>
#include <limits>


namespace Catch {
namespace {

    template <typename FP>
    bool almostEqualUlps(FP lhs, FP rhs, uint64_t maxUlpDiff) {
        // Comparison with NaN should always be false.
        // This way we can rule it out before getting into the ugly details
        if (Catch::isnan(lhs) || Catch::isnan(rhs)) {
            return false;
        }

        // This should also handle positive and negative zeros, infinities
        const auto ulpDist = ulpDistance(lhs, rhs);

        return ulpDist <= maxUlpDiff;
    }


template <typename FP>
FP step(FP start, FP direction, uint64_t steps) {
    for (uint64_t i = 0; i < steps; ++i) {
        start = Catch::nextafter(start, direction);
    }
    return start;
}

// Performs equivalent check of std::fabs(lhs - rhs) <= margin
// But without the subtraction to allow for INFINITY in comparison
bool marginComparison(double lhs, double rhs, double margin) {
    return (lhs + margin >= rhs) && (rhs + margin >= lhs);
}

template <typename FloatingPoint>
void write(std::ostream& out, FloatingPoint num) {
    out << std::scientific
        << std::setprecision(std::numeric_limits<FloatingPoint>::max_digits10 - 1)
        << num;
}

} // end anonymous namespace

namespace Matchers {
namespace Detail {

    enum class FloatingPointKind : uint8_t {
        Float,
        Double
    };

} // end namespace Detail


    WithinAbsMatcher::WithinAbsMatcher(double target, double margin)
        :m_target{ target }, m_margin{ margin } {
        CATCH_ENFORCE(margin >= 0, "Invalid margin: " << margin << '.'
            << " Margin has to be non-negative.");
    }

    // Performs equivalent check of std::fabs(lhs - rhs) <= margin
    // But without the subtraction to allow for INFINITY in comparison
    bool WithinAbsMatcher::match(double const& matchee) const {
        return (matchee + m_margin >= m_target) && (m_target + m_margin >= matchee);
    }

    std::string WithinAbsMatcher::describe() const {
        return "is within " + ::Catch::Detail::stringify(m_margin) + " of " + ::Catch::Detail::stringify(m_target);
    }


    WithinUlpsMatcher::WithinUlpsMatcher(double target, uint64_t ulps, Detail::FloatingPointKind baseType)
        :m_target{ target }, m_ulps{ ulps }, m_type{ baseType } {
        CATCH_ENFORCE(m_type == Detail::FloatingPointKind::Double
                   || m_ulps < (std::numeric_limits<uint32_t>::max)(),
            "Provided ULP is impossibly large for a float comparison.");
        CATCH_ENFORCE( std::numeric_limits<double>::is_iec559,
                       "WithinUlp matcher only supports platforms with "
                       "IEEE-754 compatible floating point representation" );
    }

#if defined(__clang__)
#pragma clang diagnostic push
// Clang <3.5 reports on the default branch in the switch below
#pragma clang diagnostic ignored "-Wunreachable-code"
#endif

    bool WithinUlpsMatcher::match(double const& matchee) const {
        switch (m_type) {
        case Detail::FloatingPointKind::Float:
            return almostEqualUlps<float>(static_cast<float>(matchee), static_cast<float>(m_target), m_ulps);
        case Detail::FloatingPointKind::Double:
            return almostEqualUlps<double>(matchee, m_target, m_ulps);
        default:
            CATCH_INTERNAL_ERROR( "Unknown Detail::FloatingPointKind value" );
        }
    }

#if defined(__clang__)
#pragma clang diagnostic pop
#endif

    std::string WithinUlpsMatcher::describe() const {
        std::stringstream ret;

        ret << "is within " << m_ulps << " ULPs of ";

        if (m_type == Detail::FloatingPointKind::Float) {
            write(ret, static_cast<float>(m_target));
            ret << 'f';
        } else {
            write(ret, m_target);
        }

        ret << " ([";
        if (m_type == Detail::FloatingPointKind::Double) {
            write( ret,
                   step( m_target,
                         -std::numeric_limits<double>::infinity(),
                         m_ulps ) );
            ret << ", ";
            write( ret,
                   step( m_target,
                         std::numeric_limits<double>::infinity(),
                         m_ulps ) );
        } else {
            // We have to cast INFINITY to float because of MinGW, see #1782
            write( ret,
                   step( static_cast<float>( m_target ),
                         -std::numeric_limits<float>::infinity(),
                         m_ulps ) );
            ret << ", ";
            write( ret,
                   step( static_cast<float>( m_target ),
                         std::numeric_limits<float>::infinity(),
                         m_ulps ) );
        }
        ret << "])";

        return ret.str();
    }

    WithinRelMatcher::WithinRelMatcher(double target, double epsilon):
        m_target(target),
        m_epsilon(epsilon){
        CATCH_ENFORCE(m_epsilon >= 0., "Relative comparison with epsilon <  0 does not make sense.");
        CATCH_ENFORCE(m_epsilon  < 1., "Relative comparison with epsilon >= 1 does not make sense.");
    }

    bool WithinRelMatcher::match(double const& matchee) const {
        const auto relMargin = m_epsilon * (std::max)(std::fabs(matchee), std::fabs(m_target));
        return marginComparison(matchee, m_target,
                                std::isinf(relMargin)? 0 : relMargin);
    }

    std::string WithinRelMatcher::describe() const {
        Catch::ReusableStringStream sstr;
        sstr << "and " << ::Catch::Detail::stringify(m_target) << " are within " << m_epsilon * 100. << "% of each other";
        return sstr.str();
    }


WithinUlpsMatcher WithinULP(double target, uint64_t maxUlpDiff) {
    return WithinUlpsMatcher(target, maxUlpDiff, Detail::FloatingPointKind::Double);
}

WithinUlpsMatcher WithinULP(float target, uint64_t maxUlpDiff) {
    return WithinUlpsMatcher(target, maxUlpDiff, Detail::FloatingPointKind::Float);
}

WithinAbsMatcher WithinAbs(double target, double margin) {
    return WithinAbsMatcher(target, margin);
}

WithinRelMatcher WithinRel(double target, double eps) {
    return WithinRelMatcher(target, eps);
}

WithinRelMatcher WithinRel(double target) {
    return WithinRelMatcher(target, std::numeric_limits<double>::epsilon() * 100);
}

WithinRelMatcher WithinRel(float target, float eps) {
    return WithinRelMatcher(target, eps);
}

WithinRelMatcher WithinRel(float target) {
    return WithinRelMatcher(target, std::numeric_limits<float>::epsilon() * 100);
}



bool IsNaNMatcher::match( double const& matchee ) const {
    return std::isnan( matchee );
}

std::string IsNaNMatcher::describe() const {
    using namespace std::string_literals;
    return "is NaN"s;
}

IsNaNMatcher IsNaN() { return IsNaNMatcher(); }

    } // namespace Matchers
} // namespace Catch
