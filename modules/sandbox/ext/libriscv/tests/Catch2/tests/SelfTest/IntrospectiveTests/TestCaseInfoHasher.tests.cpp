
//              Copyright Catch2 Authors
// Distributed under the Boost Software License, Version 1.0.
//   (See accompanying file LICENSE.txt or copy at
//        https://www.boost.org/LICENSE_1_0.txt)

// SPDX-License-Identifier: BSL-1.0

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_test_case_info.hpp>
#include <catch2/internal/catch_test_case_info_hasher.hpp>

static constexpr Catch::SourceLineInfo dummySourceLineInfo = CATCH_INTERNAL_LINEINFO;

using Catch::TestCaseInfo;
using Catch::TestCaseInfoHasher;

TEST_CASE("Hashers with same seed produce same hash", "[test-case-hash]") {
    TestCaseInfo dummy( "", { "name", "[a-tag]" }, dummySourceLineInfo );

    TestCaseInfoHasher h1( 0x12345678 );
    TestCaseInfoHasher h2( 0x12345678 );

    REQUIRE( h1( dummy ) == h2( dummy ) );
}

TEST_CASE(
    "Hashers with different seed produce different hash with same test case",
    "[test-case-hash]") {
    TestCaseInfo dummy( "", { "name", "[a-tag]" }, dummySourceLineInfo );

    TestCaseInfoHasher h1( 0x12345678 );
    TestCaseInfoHasher h2( 0x87654321 );

    REQUIRE( h1( dummy ) != h2( dummy ) );
}

TEST_CASE("Hashing test case produces same hash across multiple calls",
          "[test-case-hash]") {
    TestCaseInfo dummy( "", { "name", "[a-tag]" }, dummySourceLineInfo );

    TestCaseInfoHasher h( 0x12345678 );

    REQUIRE( h( dummy ) == h( dummy ) );
}

TEST_CASE("Hashing different test cases produces different result", "[test-case-hash]") {
    TestCaseInfoHasher h( 0x12345678 );
    SECTION("Different test name") {
        TestCaseInfo dummy1( "class", { "name-1", "[a-tag]" }, dummySourceLineInfo );
        TestCaseInfo dummy2(
            "class", { "name-2", "[a-tag]" }, dummySourceLineInfo );

        REQUIRE( h( dummy1 ) != h( dummy2 ) );
    }
    SECTION("Different classname") {
        TestCaseInfo dummy1(
            "class-1", { "name", "[a-tag]" }, dummySourceLineInfo );
        TestCaseInfo dummy2(
            "class-2", { "name", "[a-tag]" }, dummySourceLineInfo );

        REQUIRE( h( dummy1 ) != h( dummy2 ) );
    }
    SECTION("Different tags") {
        TestCaseInfo dummy1(
            "class", { "name", "[a-tag]" }, dummySourceLineInfo );
        TestCaseInfo dummy2(
            "class", { "name", "[b-tag]" }, dummySourceLineInfo );

        REQUIRE( h( dummy1 ) != h( dummy2 ) );
    }
}
