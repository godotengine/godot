
//              Copyright Catch2 Authors
// Distributed under the Boost Software License, Version 1.0.
//   (See accompanying file LICENSE.txt or copy at
//        https://www.boost.org/LICENSE_1_0.txt)

// SPDX-License-Identifier: BSL-1.0

#include <catch2/catch_test_macros.hpp>
#include <catch2/internal/catch_enforce.hpp>
#include <catch2/internal/catch_case_insensitive_comparisons.hpp>
#include <catch2/internal/catch_optional.hpp>

#include <helpers/type_with_lit_0_comparisons.hpp>

#if defined(_MSC_VER)
#pragma warning(push)
#pragma warning(disable:4702) // unreachable code in the macro expansions
#endif

TEST_CASE("Check that our error handling macros throw the right exceptions", "[!throws][internals][approvals]") {
    REQUIRE_THROWS_AS(CATCH_INTERNAL_ERROR(""), std::logic_error);
    REQUIRE_THROWS_AS(CATCH_ERROR(""), std::domain_error);
    REQUIRE_THROWS_AS(CATCH_RUNTIME_ERROR(""), std::runtime_error);
    REQUIRE_THROWS_AS([](){CATCH_ENFORCE(false, "");}(), std::domain_error);
    REQUIRE_NOTHROW([](){CATCH_ENFORCE(true, "");}());
}

#if defined(_MSC_VER)
#pragma warning(pop) // unreachable code in the macro expansions
#endif

TEST_CASE("CaseInsensitiveLess is case insensitive", "[comparisons][string-case]") {
    Catch::Detail::CaseInsensitiveLess lt;
    SECTION( "Degenerate cases" ) {
        REQUIRE( lt( "", "a" ) );
        REQUIRE_FALSE( lt( "a", "a" ) );
        REQUIRE_FALSE( lt( "", "" ) );
    }
    SECTION("Plain comparisons") {
        REQUIRE( lt( "a", "b" ) );
        REQUIRE( lt( "a", "B" ) );
        REQUIRE( lt( "A", "b" ) );
        REQUIRE( lt( "A", "B" ) );
    }
}

TEST_CASE( "CaseInsensitiveEqualsTo is case insensitive",
           "[comparisons][string-case]" ) {
    Catch::Detail::CaseInsensitiveEqualTo eq;
    SECTION( "Degenerate cases" ) {
        REQUIRE( eq( "", "" ) );
        REQUIRE_FALSE( eq( "", "a" ) );
    }
    SECTION( "Plain comparisons" ) {
        REQUIRE( eq( "a", "a" ) );
        REQUIRE( eq( "a", "A" ) );
        REQUIRE( eq( "A", "a" ) );
        REQUIRE( eq( "A", "A" ) );
        REQUIRE_FALSE( eq( "a", "b" ) );
        REQUIRE_FALSE( eq( "a", "B" ) );
    }
}

TEST_CASE("Optional comparison ops", "[optional][approvals]") {
    using Catch::Optional;

    Optional<int> a, b;

    SECTION( "Empty optionals are equal" ) {
        REQUIRE( a == b );
        REQUIRE_FALSE( a != b );
    }
    SECTION( "Empty and non-empty optionals are never equal" ) {
        a = 1;
        REQUIRE_FALSE( a == b );
        REQUIRE( a != b );
    }
    SECTION(
        "non-empty optionals are equal if the contained elements are equal") {
        a = 1;
        b = 2;
        REQUIRE( a != b );
        REQUIRE_FALSE( a == b );

        a = 2;
        REQUIRE( a == b );
        REQUIRE_FALSE( a != b );
    }
}

namespace {
    struct MoveChecker {
        bool has_moved = false;
        MoveChecker() = default;
        MoveChecker( MoveChecker const& rhs ) = default;
        MoveChecker& operator=( MoveChecker const& rhs ) = default;
        MoveChecker( MoveChecker&& rhs ) noexcept { rhs.has_moved = true; }
        MoveChecker& operator=( MoveChecker&& rhs ) noexcept {
            rhs.has_moved = true;
            return *this;
        }
    };
}

TEST_CASE( "Optional supports move ops", "[optional][approvals]" ) {
    using Catch::Optional;
    MoveChecker a;
    Optional<MoveChecker> opt_A( a );
    REQUIRE_FALSE( a.has_moved );
    REQUIRE_FALSE( opt_A->has_moved );

    SECTION( "Move construction from element" ) {
        Optional<MoveChecker> opt_B( CATCH_MOVE( a ) );
        REQUIRE( a.has_moved );
    }
    SECTION( "Move assignment from element" ) {
        opt_A = CATCH_MOVE( a );
        REQUIRE( a.has_moved );
    }
    SECTION( "Move construction from optional" ) {
        Optional<MoveChecker> opt_B( CATCH_MOVE( opt_A ) );
        REQUIRE( opt_A->has_moved ); // NOLINT(clang-analyzer-cplusplus.Move)
    }
    SECTION( "Move assignment from optional" ) {
        Optional<MoveChecker> opt_B( opt_A );
        REQUIRE_FALSE( opt_A->has_moved );
        opt_B = CATCH_MOVE( opt_A );
        REQUIRE( opt_A->has_moved ); // NOLINT(clang-analyzer-cplusplus.Move)
    }
}

TEST_CASE( "Decomposer checks that the argument is 0 when handling "
           "only-0-comparable types",
           "[decomposition][approvals]" ) {
    TypeWithLit0Comparisons t{};

    CATCH_INTERNAL_START_WARNINGS_SUPPRESSION
    CATCH_INTERNAL_SUPPRESS_PARENTHESES_WARNINGS

    REQUIRE_THROWS( Catch::Decomposer{} <= t == 42 );
    REQUIRE_THROWS( Catch::Decomposer{} <= 42 == t );
    REQUIRE_NOTHROW( Catch::Decomposer{} <= t == 0 );
    REQUIRE_NOTHROW( Catch::Decomposer{} <= 0 == t );

    REQUIRE_THROWS( Catch::Decomposer{} <= t != 42 );
    REQUIRE_THROWS( Catch::Decomposer{} <= 42 != t );
    REQUIRE_NOTHROW( Catch::Decomposer{} <= t != 0 );
    REQUIRE_NOTHROW( Catch::Decomposer{} <= 0 != t );

    REQUIRE_THROWS( Catch::Decomposer{} <= t < 42 );
    REQUIRE_THROWS( Catch::Decomposer{} <= 42 < t );
    REQUIRE_NOTHROW( Catch::Decomposer{} <= t < 0 );
    REQUIRE_NOTHROW( Catch::Decomposer{} <= 0 < t );

    REQUIRE_THROWS( Catch::Decomposer{} <= t <= 42 );
    REQUIRE_THROWS( Catch::Decomposer{} <= 42 <= t );
    REQUIRE_NOTHROW( Catch::Decomposer{} <= t <= 0 );
    REQUIRE_NOTHROW( Catch::Decomposer{} <= 0 <= t );

    REQUIRE_THROWS( Catch::Decomposer{} <= t > 42 );
    REQUIRE_THROWS( Catch::Decomposer{} <= 42 > t );
    REQUIRE_NOTHROW( Catch::Decomposer{} <= t > 0 );
    REQUIRE_NOTHROW( Catch::Decomposer{} <= 0 > t );

    REQUIRE_THROWS( Catch::Decomposer{} <= t >= 42 );
    REQUIRE_THROWS( Catch::Decomposer{} <= 42 >= t );
    REQUIRE_NOTHROW( Catch::Decomposer{} <= t >= 0 );
    REQUIRE_NOTHROW( Catch::Decomposer{} <= 0 >= t );

    CATCH_INTERNAL_STOP_WARNINGS_SUPPRESSION
}
