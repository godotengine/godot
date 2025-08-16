
//              Copyright Catch2 Authors
// Distributed under the Boost Software License, Version 1.0.
//   (See accompanying file LICENSE.txt or copy at
//        https://www.boost.org/LICENSE_1_0.txt)

// SPDX-License-Identifier: BSL-1.0

#include <catch2/internal/catch_case_insensitive_comparisons.hpp>
#include <catch2/internal/catch_string_manip.hpp>

#include <algorithm>

namespace Catch {
    namespace Detail {

        bool CaseInsensitiveLess::operator()( StringRef lhs,
                                              StringRef rhs ) const {
            return std::lexicographical_compare(
                lhs.begin(), lhs.end(),
                rhs.begin(), rhs.end(),
                []( char l, char r ) { return toLower( l ) < toLower( r ); } );
        }

        bool
        CaseInsensitiveEqualTo::operator()( StringRef lhs,
                                            StringRef rhs ) const {
            return std::equal(
                lhs.begin(), lhs.end(),
                rhs.begin(), rhs.end(),
                []( char l, char r ) { return toLower( l ) == toLower( r ); } );
        }

    } // namespace Detail
} // namespace Catch
