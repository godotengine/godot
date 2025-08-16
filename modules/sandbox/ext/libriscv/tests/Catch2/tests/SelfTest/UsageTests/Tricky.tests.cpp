
//              Copyright Catch2 Authors
// Distributed under the Boost Software License, Version 1.0.
//   (See accompanying file LICENSE.txt or copy at
//        https://www.boost.org/LICENSE_1_0.txt)

// SPDX-License-Identifier: BSL-1.0

#ifdef __clang__
#pragma clang diagnostic ignored "-Wpadded"
#endif

#ifdef _MSC_VER
#pragma warning (disable : 4702) // Disable unreachable code warning for the last test
                                 // that is triggered when compiling as Win32|Release
#endif

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/generators/catch_generators_range.hpp>

#include <cstdio>
#include <sstream>
#include <iostream>

struct Opaque
{
    int val;
    bool operator ==( const Opaque& o ) const
    {
        return val == o.val;
    }
};

///////////////////////////////////////////////////////////////////////////////
TEST_CASE
(
    "A failing expression with a non streamable type is still captured",
    "[Tricky][failing][.]"
)
{

    Opaque o1, o2;
    o1.val = 7;
    o2.val = 8;

    CHECK( &o1 == &o2 );
    CHECK( o1 == o2 );
}

///////////////////////////////////////////////////////////////////////////////
TEST_CASE
(
    "An expression with side-effects should only be evaluated once",
    "[Tricky]"
)
{
    int i = 7;

    REQUIRE( i++ == 7 );
    REQUIRE( i++ == 8 );

}

namespace A {
    struct X
    {
        X() : a(4), b(2), c(7) {}
        X(int v) : a(v), b(2), c(7) {}
        int a;
        int b;
        int c;
    };
}

namespace B {
    struct Y
    {
        Y() : a(4), b(2), c(7) {}
        Y(int v) : a(v), b(2), c(7) {}
        int a;
        int b;
        int c;
    };
}

inline bool operator==(const A::X& lhs, const B::Y& rhs)
{
    return (lhs.a == rhs.a);
}

inline bool operator==(const B::Y& lhs, const A::X& rhs)
{
    return (lhs.a == rhs.a);
}


///////////////////////////////////////////////////////////////////////////////
/* This, currently, does not compile with LLVM
TEST_CASE
(
    "Operators at different namespace levels not hijacked by Koenig lookup"
    "[Tricky]"
)
{
    A::X x;
    B::Y y;
    REQUIRE( x == y );
}
*/

namespace ObjectWithConversions
{
    struct Object
    {
        operator unsigned int() const {return 0xc0000000;}
    };

    ///////////////////////////////////////////////////////////////////////////////
    TEST_CASE
    (
        "Implicit conversions are supported inside assertion macros",
        "[Tricky][approvals]"
    )
    {
        Object o;
        REQUIRE(0xc0000000 == o );
    }
}

namespace EnumBitFieldTests
{
    enum Bits : uint32_t {
        bit0 = 0x0001,
        bit1 = 0x0002,
        bit2 = 0x0004,
        bit3 = 0x0008,
        bit1and2 = bit1 | bit2,
        bit30 = 0x40000000,
        bit31 = 0x80000000,
        bit30and31 = bit30 | bit31
    };

    TEST_CASE( "Test enum bit values", "[Tricky]" )
    {
        REQUIRE( 0xc0000000 == bit30and31 );
    }
}

struct Obj
{
    Obj():prop(&p){}

    int p = 0;
    int* prop;
};

TEST_CASE("boolean member", "[Tricky]")
{
    Obj obj;
    REQUIRE( obj.prop != nullptr );
}

// Tests for a problem submitted by Ralph McArdell
//
// The static bool value should not need to be defined outside the
// struct it is declared in - but when evaluating it in a deduced
// context it appears to require the extra definition.
// The issue was fixed by adding bool overloads to bypass the
// templates that were there to deduce it.
template <bool B>
struct is_true
{
    static const bool value = B;
};

TEST_CASE( "(unimplemented) static bools can be evaluated", "[Tricky]" )
{
    SECTION("compare to true")
    {
        REQUIRE( is_true<true>::value == true );
        REQUIRE( true == is_true<true>::value );
    }
    SECTION("compare to false")
    {
        REQUIRE( is_true<false>::value == false );
        REQUIRE( false == is_true<false>::value );
    }

    SECTION("negation")
    {
        REQUIRE( !is_true<false>::value );
    }

    SECTION("double negation")
    {
        REQUIRE( !!is_true<true>::value );
    }

    SECTION("direct")
    {
        REQUIRE( is_true<true>::value );
        REQUIRE_FALSE( is_true<false>::value );
    }
}

struct Boolable
{
    explicit Boolable( bool value ) : m_value( value ) {}

    explicit operator bool() const {
        return m_value;
    }

    bool m_value;
};

TEST_CASE( "Objects that evaluated in boolean contexts can be checked", "[Tricky][SafeBool]" )
{
    Boolable True( true );
    Boolable False( false );

    CHECK( True );
    CHECK( !False );
    CHECK_FALSE( False );
}

TEST_CASE( "Assertions then sections", "[Tricky]" )
{
    // This was causing a failure due to the way the console reporter was handling
    // the current section

    REQUIRE( true );

    SECTION( "A section" )
    {
        REQUIRE( true );

        SECTION( "Another section" )
        {
            REQUIRE( true );
        }
        SECTION( "Another other section" )
        {
            REQUIRE( true );
        }
    }
}

struct Awkward
{
    operator int() const { return 7; }
};

TEST_CASE( "non streamable - with conv. op", "[Tricky]" )
{
    Awkward awkward;
    std::string s = ::Catch::Detail::stringify( awkward );
    REQUIRE( s == "7" );
}

inline void foo() {}

using fooptr_t = void (*)();

TEST_CASE( "Comparing function pointers", "[Tricky][function pointer]" )
{
    // This was giving a warning in VS2010
    // #179
    fooptr_t a = foo;

    REQUIRE( a );
    REQUIRE( a == &foo );
}

struct S
{
    void f() {}
};


TEST_CASE( "Comparing member function pointers", "[Tricky][member function pointer][approvals]" )
{
    using MF = void (S::*)();
    MF m = &S::f;

    CHECK( m == &S::f );
}

class ClassName {};

TEST_CASE( "pointer to class", "[Tricky]" )
{
   ClassName *p = 0;
   REQUIRE( p == 0 );
}

#include <memory>

TEST_CASE( "null_ptr", "[Tricky]" )
{
    std::unique_ptr<int> ptr;
    REQUIRE(ptr.get() == nullptr);
}

TEST_CASE( "X/level/0/a", "[Tricky]" )      { SUCCEED(""); }
TEST_CASE( "X/level/0/b", "[Tricky][fizz]" ){ SUCCEED(""); }
TEST_CASE( "X/level/1/a", "[Tricky]" )      { SUCCEED(""); }
TEST_CASE( "X/level/1/b", "[Tricky]" )      { SUCCEED(""); }

TEST_CASE( "has printf" ) {

    // This can cause problems as, currently, stdout itself is not redirected - only the cout (and cerr) buffer
    printf( "loose text artifact\n" );
}

namespace {
    struct constructor_throws {
        [[noreturn]] constructor_throws() {
            throw 1;
        }
    };
}

TEST_CASE("Commas in various macros are allowed") {
    REQUIRE_THROWS( std::vector<constructor_throws>{constructor_throws{}, constructor_throws{}} );
    CHECK_THROWS( std::vector<constructor_throws>{constructor_throws{}, constructor_throws{}} );
    REQUIRE_NOTHROW( std::vector<int>{1, 2, 3} == std::vector<int>{1, 2, 3} );
    CHECK_NOTHROW( std::vector<int>{1, 2, 3} == std::vector<int>{1, 2, 3} );

    REQUIRE(std::vector<int>{1, 2} == std::vector<int>{1, 2});
    CHECK( std::vector<int>{1, 2} == std::vector<int>{1, 2} );
    REQUIRE_FALSE(std::vector<int>{1, 2} == std::vector<int>{1, 2, 3});
    CHECK_FALSE( std::vector<int>{1, 2} == std::vector<int>{1, 2, 3} );

    CHECK_NOFAIL( std::vector<int>{1, 2} == std::vector<int>{1, 2} );
    CHECKED_IF( std::vector<int>{1, 2} == std::vector<int>{1, 2} ) {
        REQUIRE(true);
    } CHECKED_ELSE( std::vector<int>{1, 2} == std::vector<int>{1, 2} ) {
        CHECK(true);
    }
}

TEST_CASE( "non-copyable objects", "[.][failing]" ) {
    // Thanks to Agustin Bergé (@k-ballo on the cpplang Slack) for raising this
    std::type_info const& ti = typeid(int);
    CHECK( ti == typeid(int) );
}

TEST_CASE("#1514: stderr/stdout is not captured in tests aborted by an exception", "[output-capture][regression][.]") {
    std::cout << "This would not be caught previously\n" << std::flush;
    std::clog << "Nor would this\n" << std::flush;
    // FAIL aborts the test by throwing a Catch exception
    FAIL("1514");
}


TEST_CASE( "#2025: -c shouldn't cause infinite loop", "[sections][generators][regression][.approvals]" ) {
    SECTION( "Check cursor from buffer offset" ) {
        auto bufPos = GENERATE_REF( range( 0, 44 ) );
        WHEN( "Buffer position is " << bufPos ) { REQUIRE( 1 == 1 ); }
    }
}

TEST_CASE("#2025: original repro", "[sections][generators][regression][.approvals]") {
    auto fov = GENERATE(true, false);
    DYNAMIC_SECTION("fov_" << fov) {
        std::cout << "inside with fov: " << fov << '\n';
    }
}

TEST_CASE("#2025: same-level sections", "[sections][generators][regression][.approvals]") {
    SECTION("A") {
        SUCCEED();
    }
    auto i = GENERATE(1, 2, 3);
    SECTION("B") {
        REQUIRE(i < 4);
    }
}
