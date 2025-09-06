
//              Copyright Catch2 Authors
// Distributed under the Boost Software License, Version 1.0.
//   (See accompanying file LICENSE.txt or copy at
//        https://www.boost.org/LICENSE_1_0.txt)

// SPDX-License-Identifier: BSL-1.0

#include <catch2/catch_test_macros.hpp>
#include <catch2/internal/catch_stringref.hpp>

#include <cstring>

TEST_CASE( "StringRef", "[Strings][StringRef]" ) {
    using Catch::StringRef;

    SECTION( "Empty string" ) {
        StringRef empty;
        REQUIRE( empty.empty() );
        REQUIRE( empty.size() == 0 );
        REQUIRE( std::strcmp( empty.data(), "" ) == 0 );
    }

    SECTION( "From string literal" ) {
        StringRef s = "hello";
        REQUIRE( s.empty() == false );
        REQUIRE( s.size() == 5 );

        auto rawChars = s.data();
        REQUIRE( std::strcmp( rawChars, "hello" ) == 0 );

        REQUIRE(s.data() == rawChars);
    }
    SECTION( "From sub-string" ) {
        StringRef original = StringRef( "original string" ).substr(0, 8);
        REQUIRE( original == "original" );

        REQUIRE_NOTHROW(original.data());
    }
    SECTION( "Copy construction is shallow" ) {
        StringRef original = StringRef( "original string" );
        StringRef copy = original;
        REQUIRE(original.begin() == copy.begin());
    }
    SECTION( "Copy assignment is shallow" ) {
        StringRef original = StringRef( "original string" );
        StringRef copy;
        copy = original;
        REQUIRE(original.begin() == copy.begin());
    }

    SECTION( "Substrings" ) {
        StringRef s = "hello world!";
        StringRef ss = s.substr(0, 5);

        SECTION( "zero-based substring" ) {
            REQUIRE( ss.empty() == false );
            REQUIRE( ss.size() == 5 );
            REQUIRE( std::strncmp( ss.data(), "hello", 5 ) == 0 );
            REQUIRE( ss == "hello" );
        }

        SECTION( "non-zero-based substring") {
            ss = s.substr( 6, 6 );
            REQUIRE( ss.size() == 6 );
            REQUIRE( std::strcmp( ss.data(), "world!" ) == 0 );
        }

        SECTION( "Pointer values of full refs should match" ) {
            StringRef s2 = s;
            REQUIRE( s.data() == s2.data() );
        }

        SECTION( "Pointer values of substring refs should also match" ) {
            REQUIRE( s.data() == ss.data() );
        }

        SECTION("Past the end substring") {
            REQUIRE(s.substr(s.size() + 1, 123).empty());
        }

        SECTION("Substring off the end are trimmed") {
            ss = s.substr(6, 123);
            REQUIRE(std::strcmp(ss.data(), "world!") == 0);
        }
        SECTION("substring start after the end is empty") {
            REQUIRE(s.substr(1'000'000, 1).empty());
        }
    }

    SECTION( "Comparisons are deep" ) {
        char buffer1[] = "Hello";
        char buffer2[] = "Hello";
        CHECK(reinterpret_cast<char*>(buffer1) != reinterpret_cast<char*>(buffer2));

        StringRef left(buffer1), right(buffer2);
        REQUIRE( left == right );
        REQUIRE(left != left.substr(0, 3));
    }

    SECTION( "from std::string" ) {
        std::string stdStr = "a standard string";

        SECTION( "implicitly constructed" ) {
            StringRef sr = stdStr;
            REQUIRE( sr == "a standard string" );
            REQUIRE( sr.size() == stdStr.size() );
        }
        SECTION( "explicitly constructed" ) {
            StringRef sr( stdStr );
            REQUIRE( sr == "a standard string" );
            REQUIRE( sr.size() == stdStr.size() );
        }
        SECTION( "assigned" ) {
            StringRef sr;
            sr = stdStr;
            REQUIRE( sr == "a standard string" );
            REQUIRE( sr.size() == stdStr.size() );
        }
    }

    SECTION( "to std::string" ) {
        StringRef sr = "a stringref";

        SECTION( "explicitly constructed" ) {
            std::string stdStr( sr );
            REQUIRE( stdStr == "a stringref" );
            REQUIRE( stdStr.size() == sr.size() );
        }
        SECTION( "assigned" ) {
            std::string stdStr;
            stdStr = static_cast<std::string>(sr);
            REQUIRE( stdStr == "a stringref" );
            REQUIRE( stdStr.size() == sr.size() );
        }
    }

    SECTION("std::string += StringRef") {
        StringRef sr = "the stringref contents";
        std::string lhs("some string += ");
        lhs += sr;
        REQUIRE(lhs == "some string += the stringref contents");
    }
    SECTION("StringRef + StringRef") {
        StringRef sr1 = "abraka", sr2 = "dabra";
        std::string together = sr1 + sr2;
        REQUIRE(together == "abrakadabra");
    }
}

TEST_CASE("StringRef at compilation time", "[Strings][StringRef][constexpr]") {
    using Catch::StringRef;
    SECTION("Simple constructors") {
        constexpr StringRef empty{};
        STATIC_REQUIRE(empty.size() == 0);
        STATIC_REQUIRE(empty.begin() == empty.end());

        constexpr char const* const abc = "abc";

        constexpr StringRef stringref(abc, 3);
        STATIC_REQUIRE(stringref.size() == 3);
        STATIC_REQUIRE(stringref.data() == abc);
        STATIC_REQUIRE(stringref.begin() == abc);
        STATIC_REQUIRE(stringref.begin() != stringref.end());
        STATIC_REQUIRE(stringref.substr(10, 0).empty());
        STATIC_REQUIRE(stringref.substr(2, 1).data() == abc + 2);
        STATIC_REQUIRE(stringref[1] == 'b');


        constexpr StringRef shortened(abc, 2);
        STATIC_REQUIRE(shortened.size() == 2);
        STATIC_REQUIRE(shortened.data() == abc);
        STATIC_REQUIRE(shortened.begin() != shortened.end());
    }
    SECTION("UDL construction") {
        constexpr auto sr1 = "abc"_catch_sr;
        STATIC_REQUIRE_FALSE(sr1.empty());
        STATIC_REQUIRE(sr1.size() == 3);

        using Catch::operator""_sr;
        constexpr auto sr2 = ""_sr;
        STATIC_REQUIRE(sr2.empty());
        STATIC_REQUIRE(sr2.size() == 0);
    }
}

TEST_CASE("StringRef::compare", "[Strings][StringRef][approvals]") {
    using Catch::StringRef;

    SECTION("Same length on both sides") {
        StringRef sr1("abcdc");
        StringRef sr2("abcdd");
        StringRef sr3("abcdc");

        REQUIRE(sr1.compare(sr2) < 0);
        REQUIRE(sr2.compare(sr1) > 0);
        REQUIRE(sr1.compare(sr3) == 0);
        REQUIRE(sr3.compare(sr1) == 0);
    }
    SECTION("Different lengths") {
        StringRef sr1("def");
        StringRef sr2("deff");
        StringRef sr3("ab");

        REQUIRE(sr1.compare(sr2) < 0);
        REQUIRE(sr2.compare(sr1) > 0);
        REQUIRE(sr1.compare(sr3) > 0);
        REQUIRE(sr2.compare(sr3) > 0);
        REQUIRE(sr3.compare(sr1) < 0);
        REQUIRE(sr3.compare(sr2) < 0);
    }
}
