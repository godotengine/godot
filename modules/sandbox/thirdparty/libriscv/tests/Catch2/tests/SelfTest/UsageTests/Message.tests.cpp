
//              Copyright Catch2 Authors
// Distributed under the Boost Software License, Version 1.0.
//   (See accompanying file LICENSE.txt or copy at
//        https://www.boost.org/LICENSE_1_0.txt)

// SPDX-License-Identifier: BSL-1.0

#include <catch2/catch_test_macros.hpp>
#include <iostream>

TEST_CASE( "INFO and WARN do not abort tests", "[messages][.]" ) {
    INFO( "this is a " << "message" );    // This should output the message if a failure occurs
    WARN( "this is a " << "warning" );    // This should always output the message but then continue
}

TEST_CASE( "#1455 - INFO and WARN can start with a linebreak", "[messages][.]" ) {
    // Previously these would be hidden from the console reporter output,
    // because it would fail at properly reflowing the text
    INFO( "\nThis info message starts with a linebreak" );
    WARN( "\nThis warning message starts with a linebreak" );
}

TEST_CASE( "SUCCEED counts as a test pass", "[messages]" ) {
    SUCCEED( "this is a " << "success" );
}

TEST_CASE( "INFO gets logged on failure", "[failing][messages][.]" ) {
    INFO( "this message should be logged" );
    INFO( "so should this" );
    int a = 2;
    REQUIRE( a == 1 );
}

TEST_CASE( "INFO gets logged on failure, even if captured before successful assertions", "[failing][messages][.]" ) {
    INFO( "this message may be logged later" );
    int a = 2;
    CHECK( a == 2 );

    INFO( "this message should be logged" );

    CHECK( a == 1 );

    INFO( "and this, but later" );

    CHECK( a == 0 );

    INFO( "but not this" );

    CHECK( a == 2 );
}

TEST_CASE( "FAIL aborts the test", "[failing][messages][.]" ) {
    FAIL( "This is a " << "failure" );    // This should output the message and abort
    WARN( "We should never see this");
}

TEST_CASE( "FAIL_CHECK does not abort the test", "[failing][messages][.]" ) {
    FAIL_CHECK( "This is a " << "failure" );    // This should output the message then continue
    WARN( "This message appears in the output");
}

TEST_CASE( "FAIL does not require an argument", "[failing][messages][.]" ) {
    FAIL();
}

TEST_CASE( "SUCCEED does not require an argument", "[messages][.]" ) {
   SUCCEED();
}

TEST_CASE( "Output from all sections is reported", "[failing][messages][.]" ) {
    SECTION( "one" ) {
        FAIL( "Message from section one" );
    }

    SECTION( "two" ) {
        FAIL( "Message from section two" );
    }
}

TEST_CASE( "Standard output from all sections is reported", "[messages][.]" ) {
    SECTION( "one" ) {
        std::cout << "Message from section one\n";
    }

    SECTION( "two" ) {
        std::cout << "Message from section two\n";
    }
}

TEST_CASE( "Standard error is reported and redirected", "[messages][.][approvals]" ) {
    SECTION( "std::cerr" ) {
        std::cerr << "Write to std::cerr\n";
    }
    SECTION( "std::clog" ) {
        std::clog << "Write to std::clog\n";
    }
    SECTION( "Interleaved writes to cerr and clog" ) {
        std::cerr << "Inter";
        std::clog << "leaved";
        std::cerr << ' ';
        std::clog << "writes";
        std::cerr << " to error";
        std::clog << " streams\n" << std::flush;
    }
}

TEST_CASE( "INFO is reset for each loop", "[messages][failing][.]" ) {
    for( int i=0; i<100; i++ )
    {
        INFO( "current counter " << i );
        CAPTURE( i );
        REQUIRE( i < 10 );
    }
}

TEST_CASE( "The NO_FAIL macro reports a failure but does not fail the test", "[messages]" ) {
    CHECK_NOFAIL( 1 == 2 );
}

TEST_CASE( "just info", "[info][isolated info][messages]" ) {
    INFO( "this should never be seen" );
}
TEST_CASE( "just failure", "[fail][isolated info][.][messages]" ) {
    FAIL( "Previous info should not be seen" );
}


TEST_CASE( "sends information to INFO", "[.][failing]" ) {
    INFO( "hi" );
    int i = 7;
    CAPTURE( i );
    REQUIRE( false );
}

TEST_CASE( "Pointers can be converted to strings", "[messages][.][approvals]" ) {
    int p;
    WARN( "actual address of p: " << &p );
    WARN( "toString(p): " << ::Catch::Detail::stringify( &p ) );
}

template <typename T>
static void unscoped_info( T msg ) {
    UNSCOPED_INFO( msg );
}

TEST_CASE( "just unscoped info", "[unscoped][info]" ) {
    unscoped_info( "this should NOT be seen" );
    unscoped_info( "this also should NOT be seen" );
}

TEST_CASE( "just failure after unscoped info", "[failing][.][unscoped][info]" ) {
    FAIL( "previous unscoped info SHOULD not be seen" );
}

TEST_CASE( "print unscoped info if passing unscoped info is printed", "[unscoped][info]" ) {
    unscoped_info( "this MAY be seen IF info is printed for passing assertions" );
    REQUIRE( true );
}

TEST_CASE( "prints unscoped info on failure", "[failing][.][unscoped][info]" ) {
    unscoped_info( "this SHOULD be seen" );
    unscoped_info( "this SHOULD also be seen" );
    REQUIRE( false );
    unscoped_info( "but this should NOT be seen" );
}

TEST_CASE( "not prints unscoped info from previous failures", "[failing][.][unscoped][info]" ) {
    unscoped_info( "this MAY be seen only for the FIRST assertion IF info is printed for passing assertions" );
    REQUIRE( true );
    unscoped_info( "this MAY be seen only for the SECOND assertion IF info is printed for passing assertions" );
    REQUIRE( true );
    unscoped_info( "this SHOULD be seen" );
    REQUIRE( false );
}

TEST_CASE( "prints unscoped info only for the first assertion", "[failing][.][unscoped][info]" ) {
    unscoped_info( "this SHOULD be seen only ONCE" );
    CHECK( false );
    CHECK( true );
    unscoped_info( "this MAY also be seen only ONCE IF info is printed for passing assertions" );
    CHECK( true );
    CHECK( true );
}

TEST_CASE( "stacks unscoped info in loops", "[failing][.][unscoped][info]" ) {
    UNSCOPED_INFO("Count 1 to 3...");
    for (int i = 1; i <= 3; i++) {
        unscoped_info(i);
    }
    CHECK( false );

    UNSCOPED_INFO("Count 4 to 6...");
    for (int i = 4; i <= 6; i++) {
        unscoped_info(i);
    }
    CHECK( false );
}

TEST_CASE( "mix info, unscoped info and warning", "[unscoped][info]" ) {
    INFO("info");
    unscoped_info("unscoped info");
    WARN("and warn may mix");
    WARN("they are not cleared after warnings");
}

TEST_CASE( "CAPTURE can deal with complex expressions", "[messages][capture]" ) {
    int a = 1;
    int b = 2;
    int c = 3;
    CAPTURE( a, b, c, a + b, a+b, c > b, a == 1 );
    SUCCEED();
}

#ifdef __clang__
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunused-value" // In (1, 2), the "1" is unused ...
#endif
#ifdef __GNUC__
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-value" // All the comma operators are side-effect free
#endif
#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable:4709) // comma in indexing operator
#endif

template <typename T1, typename T2>
struct helper_1436 {
    helper_1436(T1 t1_, T2 t2_):
        t1{ t1_ },
        t2{ t2_ }
    {}
    T1 t1;
    T2 t2;
};

template <typename T1, typename T2>
std::ostream& operator<<(std::ostream& out, helper_1436<T1, T2> const& helper) {
    out << "{ " << helper.t1 << ", " << helper.t2 << " }";
    return out;
}

// Clang and gcc have different names for this warning, and clang also
// warns about an unused value. This warning must be disabled for C++20.
#if defined(__GNUG__) && !defined(__clang__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wpragmas"
#pragma GCC diagnostic ignored "-Wcomma-subscript"
#elif defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunknown-pragmas"
#pragma clang diagnostic ignored "-Wunknown-warning-option"
#pragma clang diagnostic ignored "-Wdeprecated-comma-subscript"
#pragma clang diagnostic ignored "-Wunused-value"
#endif

namespace {
    template <typename T>
    struct custom_index_op {
        constexpr custom_index_op( std::initializer_list<T> ) {}
        constexpr T operator[]( size_t ) { return T{}; }
#if defined( __cpp_multidimensional_subscript ) && \
    __cpp_multidimensional_subscript >= 202110L
        constexpr T operator[]( size_t, size_t, size_t ) const noexcept {
            return T{};
        }
#endif
    };
}

TEST_CASE("CAPTURE can deal with complex expressions involving commas", "[messages][capture]") {
    CAPTURE(custom_index_op<int>{1, 2, 3}[0, 1, 2],
            custom_index_op<int>{1, 2, 3}[(0, 1)],
            custom_index_op<int>{1, 2, 3}[0]);
    CAPTURE((helper_1436<int, int>{12, -12}),
            (helper_1436<int, int>(-12, 12)));
    CAPTURE( (1, 2), (2, 3) );
    SUCCEED();
}

#ifdef __GNUG__
#pragma GCC diagnostic pop
#endif

TEST_CASE("CAPTURE parses string and character constants", "[messages][capture]") {
    CAPTURE(("comma, in string", "escaped, \", "), "single quote in string,',", "some escapes, \\,\\\\");
    CAPTURE("some, ), unmatched, } prenheses {[<");
    CAPTURE('"', '\'', ',', '}', ')', '(', '{');
    SUCCEED();
}

#ifdef __clang__
#pragma clang diagnostic pop
#endif
#ifdef __GNUC__
#pragma GCC diagnostic pop
#endif
#ifdef _MSC_VER
#pragma warning(pop)
#endif

TEST_CASE( "INFO and UNSCOPED_INFO can stream multiple arguments",
           "[messages][info][.failing]" ) {
    INFO( "This info"
          << " has multiple"
          << " parts." );
    UNSCOPED_INFO( "This unscoped info"
                   << " has multiple"
                   << " parts." );
    FAIL( "Show infos!" );
}

TEST_CASE( "Scoped messages do not leave block with an exception", "[messages][info][.failing]" ) {
    INFO( "Should be in scope at the end" );
    { INFO( "This should go out of scope immediately" ); }

    try {
        INFO( "Should not be in scope at the end" );
        throw std::runtime_error( "ex" );
    } catch (std::exception const&) {}

    REQUIRE( false );
}

TEST_CASE( "Captures do not leave block with an exception",
           "[messages][capture][.failing]" ) {
    int a = 1, b = 2, c = 3;

    CAPTURE( a );
    { CAPTURE( b ); }

    try {
        CAPTURE( c );
        throw std::runtime_error( "ex" );
    } catch ( std::exception const& ) {}

    REQUIRE( false );
}

TEST_CASE( "Scoped messages outlive section end",
           "[messages][info][.failing]" ) {
    INFO( "Should survive a section end" );
    SECTION( "Dummy section" ) { CHECK( true ); }

    REQUIRE( false );
}

TEST_CASE( "Captures outlive section end", "[messages][info][.failing]" ) {
    int a = 1;
    CAPTURE( a );
    SECTION( "Dummy section" ) { CHECK( true ); }

    REQUIRE( false );
}

TEST_CASE( "Scoped message applies to all assertions in scope",
           "[messages][info][.failing]" ) {
    INFO( "This will be reported multiple times" );
    CHECK( false );
    CHECK( false );
}
