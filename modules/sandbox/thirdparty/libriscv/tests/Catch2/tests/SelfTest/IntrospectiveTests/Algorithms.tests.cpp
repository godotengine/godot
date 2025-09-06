
//              Copyright Catch2 Authors
// Distributed under the Boost Software License, Version 1.0.
//   (See accompanying file LICENSE.txt or copy at
//        https://www.boost.org/LICENSE_1_0.txt)

// SPDX-License-Identifier: BSL-1.0

#include <catch2/catch_test_macros.hpp>
#include <catch2/internal/catch_is_permutation.hpp>

#include <helpers/range_test_helpers.hpp>

#include <array>

namespace {
    template <typename Range1, typename Range2>
    static bool is_permutation(Range1 const& r1, Range2 const& r2) {
        using std::begin; using std::end;
        return Catch::Detail::is_permutation(
            begin( r1 ), end( r1 ), begin( r2 ), end( r2 ), std::equal_to<>{} );
    }
}

TEST_CASE("is_permutation", "[algorithms][approvals]") {
    SECTION( "Handle empty ranges" ) {
        std::array<int, 0> empty;
        std::array<int, 2> non_empty{ { 2, 3 } };
        REQUIRE( is_permutation( empty, empty ) );
        REQUIRE_FALSE( is_permutation( empty, non_empty ) );
        REQUIRE_FALSE( is_permutation( non_empty, empty ) );
    }
    SECTION( "Different length ranges" ) {
        std::array<int, 6> arr1{ { 1, 3, 5, 7, 8, 9 } };
        // arr2 is prefix of arr1
        std::array<int, 4> arr2{ { 1, 3, 5, 7 } };
        // arr3 shares prefix with arr1 and arr2, but is not a permutation
        std::array<int, 5> arr3{ { 1, 3, 5, 9, 8 } };
        REQUIRE_FALSE( is_permutation( arr1, arr2 ) );
        REQUIRE_FALSE( is_permutation( arr1, arr3 ) );
        REQUIRE_FALSE( is_permutation( arr2, arr3 ) );
    }
    SECTION( "Same length ranges" ) {
        SECTION( "Shared elements, but different counts" ) {
            const std::array<int, 6>
                arr1{ { 1, 1, 1, 1, 2, 2 } },
                arr2{ { 1, 1, 2, 2, 2, 2 } };
            REQUIRE_FALSE( is_permutation( arr1, arr2 ) );
        }
        SECTION( "Identical ranges" ) {
            const std::array<int, 6>
                arr1{ { 1, 1, 1, 1, 2, 2 } },
                arr2{ { 1, 1, 2, 2, 2, 2 } };
            REQUIRE( is_permutation( arr1, arr1 ) );
            REQUIRE( is_permutation( arr2, arr2 ) );
        }
        SECTION( "Completely distinct elements" ) {
            // Completely distinct elements
            const std::array<int, 4>
                arr1{ { 1, 2, 3, 4 } },
                arr2{ { 10, 20, 30, 40 } };
            REQUIRE_FALSE( is_permutation( arr1, arr2 ) );
        }
        SECTION( "Reverse ranges" ) {
            const std::array<int, 5>
                arr1{ { 1, 2, 3, 4, 5 } },
                arr2{ { 5, 4, 3, 2, 1 } };
            REQUIRE( is_permutation( arr1, arr2 ) );
        }
        SECTION( "Shared prefix & permuted elements" ) {
            const std::array<int, 5>
                arr1{ { 1, 1, 2, 3, 4 } },
                arr2{ { 1, 1, 4, 2, 3 } };
            REQUIRE( is_permutation( arr1, arr2 ) );
        }
        SECTION( "Permutations with element count > 1" ) {
            const std::array<int, 7>
                arr1{ { 2, 2, 3, 3, 3, 1, 1 } },
                arr2{ { 3, 2, 1, 3, 2, 1, 3 } };
            REQUIRE( is_permutation( arr1, arr2 ) );
        }
    }
}

TEST_CASE("is_permutation supports iterator + sentinel pairs",
          "[algorithms][is-permutation][approvals]") {
    const has_different_begin_end_types<int>
        range_1{ 1, 2, 3, 4 },
        range_2{ 4, 3, 2, 1 };
    REQUIRE( is_permutation( range_1, range_2 ) );

    const has_different_begin_end_types<int> range_3{ 3, 3, 2, 1 };
    REQUIRE_FALSE( is_permutation( range_1, range_3 ) );
}
