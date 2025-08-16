
//              Copyright Catch2 Authors
// Distributed under the Boost Software License, Version 1.0.
//   (See accompanying file LICENSE.txt or copy at
//        https://www.boost.org/LICENSE_1_0.txt)

// SPDX-License-Identifier: BSL-1.0

#include <catch2/catch_test_macros.hpp>
#include <catch2/internal/catch_enum_values_registry.hpp>


namespace {
// Enum without user-provided stream operator
enum Enum1 { Enum1Value0, Enum1Value1 };

// Enum with user-provided stream operator
enum Enum2 { Enum2Value0, Enum2Value1 };

std::ostream& operator<<( std::ostream& os, Enum2 v ) {
    return os << "E2{" << static_cast<int>(v) << "}";
}
} // end anonymous namespace

TEST_CASE( "toString(enum)", "[toString][enum]" ) {
    Enum1 e0 = Enum1Value0;
    CHECK( ::Catch::Detail::stringify(e0) == "0" );
    Enum1 e1 = Enum1Value1;
    CHECK( ::Catch::Detail::stringify(e1) == "1" );
}

TEST_CASE( "toString(enum w/operator<<)", "[toString][enum]" ) {
    Enum2 e0 = Enum2Value0;
    CHECK( ::Catch::Detail::stringify(e0) == "E2{0}" );
    Enum2 e1 = Enum2Value1;
    CHECK( ::Catch::Detail::stringify(e1) == "E2{1}" );
}

// Enum class without user-provided stream operator
namespace {
enum class EnumClass1 { EnumClass1Value0, EnumClass1Value1 };

// Enum class with user-provided stream operator
enum class EnumClass2 { EnumClass2Value0, EnumClass2Value1 };

std::ostream& operator<<( std::ostream& os, EnumClass2 e2 ) {
    switch( static_cast<int>( e2 ) ) {
        case static_cast<int>( EnumClass2::EnumClass2Value0 ):
            return os << "E2/V0";
        case static_cast<int>( EnumClass2::EnumClass2Value1 ):
            return os << "E2/V1";
        default:
            return os << "Unknown enum value " << static_cast<int>( e2 );
    }
}

} // end anonymous namespace

TEST_CASE( "toString(enum class)", "[toString][enum][enumClass]" ) {
    EnumClass1 e0 = EnumClass1::EnumClass1Value0;
    CHECK( ::Catch::Detail::stringify(e0) == "0" );
    EnumClass1 e1 = EnumClass1::EnumClass1Value1;
    CHECK( ::Catch::Detail::stringify(e1) == "1" );
}


TEST_CASE( "toString(enum class w/operator<<)", "[toString][enum][enumClass]" ) {
    EnumClass2 e0 = EnumClass2::EnumClass2Value0;
    CHECK( ::Catch::Detail::stringify(e0) == "E2/V0" );
    EnumClass2 e1 = EnumClass2::EnumClass2Value1;
    CHECK( ::Catch::Detail::stringify(e1) == "E2/V1" );

    auto e3 = static_cast<EnumClass2>(10);
    CHECK( ::Catch::Detail::stringify(e3) == "Unknown enum value 10" );
}

enum class EnumClass3 { Value1, Value2, Value3, Value4 };

CATCH_REGISTER_ENUM( EnumClass3, EnumClass3::Value1, EnumClass3::Value2, EnumClass3::Value3 )


TEST_CASE( "Enums can quickly have stringification enabled using CATCH_REGISTER_ENUM" ) {
    using Catch::Detail::stringify;
    REQUIRE( stringify( EnumClass3::Value1 ) == "Value1" );
    REQUIRE( stringify( EnumClass3::Value2 ) == "Value2" );
    REQUIRE( stringify( EnumClass3::Value3 ) == "Value3" );
    REQUIRE( stringify( EnumClass3::Value4 ) == "{** unexpected enum value **}" );

    EnumClass3 ec3 = EnumClass3::Value2;
    REQUIRE( stringify( ec3 ) == "Value2" );
}

namespace Bikeshed {
    enum class Colours { Red, Green, Blue };
}

// Important!: This macro must appear at top level scope - not inside a namespace
// You can fully qualify the names, or use a using if you prefer
CATCH_REGISTER_ENUM( Bikeshed::Colours,
                     Bikeshed::Colours::Red,
                     Bikeshed::Colours::Green,
                     Bikeshed::Colours::Blue )

TEST_CASE( "Enums in namespaces can quickly have stringification enabled using CATCH_REGISTER_ENUM" ) {
    using Catch::Detail::stringify;
    REQUIRE( stringify( Bikeshed::Colours::Red ) == "Red" );
    REQUIRE( stringify( Bikeshed::Colours::Blue ) == "Blue" );
}
