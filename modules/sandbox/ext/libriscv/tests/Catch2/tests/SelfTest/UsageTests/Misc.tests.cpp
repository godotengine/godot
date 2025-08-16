
//              Copyright Catch2 Authors
// Distributed under the Boost Software License, Version 1.0.
//   (See accompanying file LICENSE.txt or copy at
//        https://www.boost.org/LICENSE_1_0.txt)

// SPDX-License-Identifier: BSL-1.0

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_template_test_macros.hpp>
#include <catch2/internal/catch_config_wchar.hpp>
#include <catch2/internal/catch_windows_h_proxy.hpp>


#include <iostream>
#include <cerrno>
#include <limits>
#include <array>
#include <tuple>

namespace {

    static const char* makeString(bool makeNull) {
        return makeNull ? nullptr : "valid string";
    }
    static bool testCheckedIf(bool flag) {
        CHECKED_IF(flag)
            return true;
    else
    return false;
    }
    static bool testCheckedElse(bool flag) {
        CHECKED_ELSE(flag)
            return false;

        return true;
    }

    static unsigned int Factorial(unsigned int number) {
        return number > 1 ? Factorial(number - 1) * number : 1;
    }

    static int f() {
        return 1;
    }

    static void manuallyRegisteredTestFunction() {
        SUCCEED("was called");
    }

    struct AutoTestReg {
        AutoTestReg() {
            REGISTER_TEST_CASE(manuallyRegisteredTestFunction, "ManuallyRegistered");
        }
    };

    CATCH_INTERNAL_START_WARNINGS_SUPPRESSION
    CATCH_INTERNAL_SUPPRESS_GLOBALS_WARNINGS
    static const AutoTestReg autoTestReg;
    CATCH_INTERNAL_STOP_WARNINGS_SUPPRESSION

        template<typename T>
    struct Foo {
        size_t size() { return 0; }
    };

    template<typename T, size_t S>
    struct Bar {
        size_t size() { return S; }
    };

}

TEST_CASE( "random SECTION tests", "[.][sections][failing]" ) {
    int a = 1;
    int b = 2;

    SECTION( "doesn't equal" ) {
        REQUIRE( a != b );
        REQUIRE( b != a );
    }

    SECTION( "not equal" ) {
        REQUIRE( a != b);
    }
}

TEST_CASE( "nested SECTION tests", "[.][sections][failing]" ) {
    int a = 1;
    int b = 2;

    SECTION( "doesn't equal" ) {
        REQUIRE( a != b );
        REQUIRE( b != a );

        SECTION( "not equal" ) {
            REQUIRE( a != b);
        }
    }
}

TEST_CASE( "more nested SECTION tests", "[sections][failing][.]" ) {
    int a = 1;
    int b = 2;

    SECTION( "doesn't equal" ) {
        SECTION( "equal" ) {
            REQUIRE( a == b );
        }

        SECTION( "not equal" ) {
            REQUIRE( a != b );
        }
        SECTION( "less than" ) {
            REQUIRE( a < b );
        }
    }
}

TEST_CASE( "even more nested SECTION tests", "[sections]" ) {
    SECTION( "c" ) {
        SECTION( "d (leaf)" ) {
            SUCCEED(); // avoid failing due to no tests
        }

        SECTION( "e (leaf)" ) {
            SUCCEED(); // avoid failing due to no tests
        }
    }

    SECTION( "f (leaf)" ) {
        SUCCEED(); // avoid failing due to no tests
    }
}

TEST_CASE( "looped SECTION tests", "[.][failing][sections]" ) {
    int a = 1;

    for( int b = 0; b < 10; ++b ) {
        DYNAMIC_SECTION( "b is currently: " << b ) {
            CHECK( b > a );
        }
    }
}

TEST_CASE( "looped tests", "[.][failing]" ) {
    static const int fib[]  = { 1, 1, 2, 3, 5, 8, 13, 21 };

    for( std::size_t i=0; i < sizeof(fib)/sizeof(int); ++i ) {
        INFO( "Testing if fib[" << i << "] (" << fib[i] << ") is even" );
        CHECK( ( fib[i] % 2 ) == 0 );
    }
}

TEST_CASE( "Sends stuff to stdout and stderr", "[.]" ) {
    std::cout << "A string sent directly to stdout\n" << std::flush;
    std::cerr << "A string sent directly to stderr\n" << std::flush;
    std::clog << "A string sent to stderr via clog\n" << std::flush;
}

TEST_CASE( "null strings" ) {
    REQUIRE( makeString( false ) != static_cast<char*>(nullptr));
    REQUIRE( makeString( true ) == static_cast<char*>(nullptr));
}

TEST_CASE( "checkedIf" ) {
    REQUIRE( testCheckedIf( true ) );
}

TEST_CASE( "checkedIf, failing", "[failing][.]" ) {
    REQUIRE( testCheckedIf( false ) );
}

TEST_CASE( "checkedElse" ) {
    REQUIRE( testCheckedElse( true ) );
}

TEST_CASE( "checkedElse, failing", "[failing][.]" ) {
    REQUIRE( testCheckedElse( false ) );
}

TEST_CASE("Testing checked-if", "[checked-if]") {
    CHECKED_IF(true) {
        SUCCEED();
    }
    CHECKED_IF(false) {
        FAIL();
    }
    CHECKED_ELSE(true) {
        FAIL();
    }
    CHECKED_ELSE(false) {
        SUCCEED();
    }
}

TEST_CASE("Testing checked-if 2", "[checked-if][!shouldfail]") {
    CHECKED_IF(true) {
        FAIL();
    }
    // If the checked if is not entered, this passes and the test
    // fails, because of the [!shouldfail] tag.
    SUCCEED();
}

TEST_CASE("Testing checked-if 3", "[checked-if][!shouldfail]") {
    CHECKED_ELSE(false) {
        FAIL();
    }
    // If the checked false is not entered, this passes and the test
    // fails, because of the [!shouldfail] tag.
    SUCCEED();
}

[[noreturn]]
TEST_CASE("Testing checked-if 4", "[checked-if][!shouldfail]") {
    CHECKED_ELSE(true) {}
    throw std::runtime_error("Uncaught exception should fail!");
}

[[noreturn]]
TEST_CASE("Testing checked-if 5", "[checked-if][!shouldfail]") {
    CHECKED_ELSE(false) {}
    throw std::runtime_error("Uncaught exception should fail!");
}

TEST_CASE( "xmlentitycheck" ) {
    SECTION( "embedded xml: <test>it should be possible to embed xml characters, such as <, \" or &, or even whole <xml>documents</xml> within an attribute</test>" ) {
        SUCCEED(); // We need this here to stop it failing due to no tests
    }
    SECTION( "encoded chars: these should all be encoded: &&&\"\"\"<<<&\"<<&\"" ) {
        SUCCEED(); // We need this here to stop it failing due to no tests
    }
}

TEST_CASE( "send a single char to INFO", "[failing][.]" ) {
    INFO(3);
    REQUIRE(false);
}

TEST_CASE( "Factorials are computed", "[factorial]" ) {
  REQUIRE( Factorial(0) == 1 );
  REQUIRE( Factorial(1) == 1 );
  REQUIRE( Factorial(2) == 2 );
  REQUIRE( Factorial(3) == 6 );
  REQUIRE( Factorial(10) == 3628800 );
}

TEST_CASE( "An empty test with no assertions", "[empty]" ) {}

TEST_CASE( "Nice descriptive name", "[tag1][tag2][tag3][.]" ) {
    WARN( "This one ran" );
}
TEST_CASE( "first tag", "[tag1]" ) {}
TEST_CASE( "second tag", "[tag2]" ) {}

TEST_CASE( "vectors can be sized and resized", "[vector]" ) {

    std::vector<int> v( 5 );

    REQUIRE( v.size() == 5 );
    REQUIRE( v.capacity() >= 5 );

    SECTION( "resizing bigger changes size and capacity" ) {
        v.resize( 10 );

        REQUIRE( v.size() == 10 );
        REQUIRE( v.capacity() >= 10 );
    }
    SECTION( "resizing smaller changes size but not capacity" ) {
        v.resize( 0 );

        REQUIRE( v.size() == 0 );
        REQUIRE( v.capacity() >= 5 );

        SECTION( "We can use the 'swap trick' to reset the capacity" ) {
            std::vector<int> empty;
            empty.swap( v );

            REQUIRE( v.capacity() == 0 );
        }
    }
    SECTION( "reserving bigger changes capacity but not size" ) {
        v.reserve( 10 );

        REQUIRE( v.size() == 5 );
        REQUIRE( v.capacity() >= 10 );
    }
    SECTION( "reserving smaller does not change size or capacity" ) {
        v.reserve( 0 );

        REQUIRE( v.size() == 5 );
        REQUIRE( v.capacity() >= 5 );
    }
}

TEMPLATE_TEST_CASE( "TemplateTest: vectors can be sized and resized", "[vector][template]", int, float, std::string, (std::tuple<int,float>) ) {

    std::vector<TestType> v( 5 );

    REQUIRE( v.size() == 5 );
    REQUIRE( v.capacity() >= 5 );

    SECTION( "resizing bigger changes size and capacity" ) {
        v.resize( 10 );

        REQUIRE( v.size() == 10 );
        REQUIRE( v.capacity() >= 10 );
    }
    SECTION( "resizing smaller changes size but not capacity" ) {
        v.resize( 0 );

        REQUIRE( v.size() == 0 );
        REQUIRE( v.capacity() >= 5 );

        SECTION( "We can use the 'swap trick' to reset the capacity" ) {
            std::vector<TestType> empty;
            empty.swap( v );

            REQUIRE( v.capacity() == 0 );
        }
    }
    SECTION( "reserving bigger changes capacity but not size" ) {
        v.reserve( 10 );

        REQUIRE( v.size() == 5 );
        REQUIRE( v.capacity() >= 10 );
    }
    SECTION( "reserving smaller does not change size or capacity" ) {
        v.reserve( 0 );

        REQUIRE( v.size() == 5 );
        REQUIRE( v.capacity() >= 5 );
    }
}

TEMPLATE_TEST_CASE_SIG("TemplateTestSig: compiles with a single int parameter", "[template][singleint]", ((int V), V), 1, 3, 6) {}

TEMPLATE_TEST_CASE_SIG("TemplateTestSig: compiles with two type parameters", "[template][onlytypes]", ((typename U, typename V), U, V), (int,int)) {}

TEMPLATE_TEST_CASE_SIG("TemplateTestSig: vectors can be sized and resized", "[vector][template][nttp]", ((typename TestType, int V), TestType, V), (int,5), (float,4), (std::string,15), ((std::tuple<int, float>), 6)) {

    std::vector<TestType> v(V);

    REQUIRE(v.size() == V);
    REQUIRE(v.capacity() >= V);

    SECTION("resizing bigger changes size and capacity") {
        v.resize(2 * V);

        REQUIRE(v.size() == 2 * V);
        REQUIRE(v.capacity() >= 2 * V);
    }
    SECTION("resizing smaller changes size but not capacity") {
        v.resize(0);

        REQUIRE(v.size() == 0);
        REQUIRE(v.capacity() >= V);

        SECTION("We can use the 'swap trick' to reset the capacity") {
            std::vector<TestType> empty;
            empty.swap(v);

            REQUIRE(v.capacity() == 0);
        }
    }
    SECTION("reserving bigger changes capacity but not size") {
        v.reserve(2 * V);

        REQUIRE(v.size() == V);
        REQUIRE(v.capacity() >= 2 * V);
    }
    SECTION("reserving smaller does not change size or capacity") {
        v.reserve(0);

        REQUIRE(v.size() == V);
        REQUIRE(v.capacity() >= V);
    }
}

TEMPLATE_PRODUCT_TEST_CASE("A Template product test case", "[template][product]", (std::vector, Foo), (int, float)) {
    TestType x;
    REQUIRE(x.size() == 0);
}

TEMPLATE_PRODUCT_TEST_CASE_SIG("A Template product test case with array signature", "[template][product][nttp]", ((typename T, size_t S), T, S), (std::array, Bar), ((int, 9), (float, 42))) {
    TestType x;
    REQUIRE(x.size() > 0);
}

TEMPLATE_PRODUCT_TEST_CASE("Product with differing arities", "[template][product]", std::tuple, (int, (int, double), (int, double, float))) {
    REQUIRE(std::tuple_size<TestType>::value >= 1);
}

using MyTypes = std::tuple<int, char, float>;
TEMPLATE_LIST_TEST_CASE("Template test case with test types specified inside std::tuple", "[template][list]", MyTypes)
{
    REQUIRE(std::is_arithmetic<TestType>::value);
}

struct NonDefaultConstructibleType {
    NonDefaultConstructibleType() = delete;
};

using MyNonDefaultConstructibleTypes = std::tuple<NonDefaultConstructibleType, float>;
TEMPLATE_LIST_TEST_CASE("Template test case with test types specified inside non-default-constructible std::tuple", "[template][list]", MyNonDefaultConstructibleTypes)
{
    REQUIRE(std::is_trivially_copyable<TestType>::value);
}

struct NonCopyableAndNonMovableType {
    NonCopyableAndNonMovableType() = default;

    NonCopyableAndNonMovableType(NonCopyableAndNonMovableType const &) = delete;
    NonCopyableAndNonMovableType(NonCopyableAndNonMovableType &&) = delete;
    auto operator=(NonCopyableAndNonMovableType const &) -> NonCopyableAndNonMovableType & = delete;
    auto operator=(NonCopyableAndNonMovableType &&) -> NonCopyableAndNonMovableType & = delete;
};

using NonCopyableAndNonMovableTypes = std::tuple<NonCopyableAndNonMovableType, float>;
TEMPLATE_LIST_TEST_CASE("Template test case with test types specified inside non-copyable and non-movable std::tuple", "[template][list]", NonCopyableAndNonMovableTypes)
{
    REQUIRE(std::is_default_constructible<TestType>::value);
}

// https://github.com/philsquared/Catch/issues/166
TEST_CASE("A couple of nested sections followed by a failure", "[failing][.]") {
    SECTION("Outer")
        SECTION("Inner")
            SUCCEED("that's not flying - that's failing in style");

    FAIL("to infinity and beyond");
}

TEST_CASE("not allowed", "[!throws]") {
    // This test case should not be included if you run with -e on the command line
    SUCCEED();
}

TEST_CASE( "Tabs and newlines show in output", "[.][whitespace][failing]" ) {

    // Based on issue #242
    std::string s1 = "if ($b == 10) {\n\t\t$a\t= 20;\n}";
    std::string s2 = "if ($b == 10) {\n\t$a = 20;\n}\n";
    CHECK( s1 == s2 );
}


#if defined(CATCH_CONFIG_WCHAR)
TEST_CASE( "toString on const wchar_t const pointer returns the string contents", "[toString]" ) {
        const wchar_t * const s = L"wide load";
        std::string result = ::Catch::Detail::stringify( s );
        CHECK( result == "\"wide load\"" );
}

TEST_CASE( "toString on const wchar_t pointer returns the string contents", "[toString]" ) {
        const wchar_t * s = L"wide load";
        std::string result = ::Catch::Detail::stringify( s );
        CHECK( result == "\"wide load\"" );
}

TEST_CASE( "toString on wchar_t const pointer returns the string contents", "[toString]" ) {
        auto const s = const_cast<wchar_t*>( L"wide load" );
        std::string result = ::Catch::Detail::stringify( s );
        CHECK( result == "\"wide load\"" );
}

TEST_CASE( "toString on wchar_t returns the string contents", "[toString]" ) {
        auto s = const_cast<wchar_t*>( L"wide load" );
        std::string result = ::Catch::Detail::stringify( s );
        CHECK( result == "\"wide load\"" );
}
#endif // CATCH_CONFIG_WCHAR

TEST_CASE( "long long" ) {
    constexpr long long l = std::numeric_limits<long long>::max();

    REQUIRE( l == std::numeric_limits<long long>::max() );
}

TEST_CASE( "This test 'should' fail but doesn't", "[.][failing][!shouldfail]" ) {
    SUCCEED( "oops!" );
}

TEST_CASE( "# A test name that starts with a #" ) {
    SUCCEED( "yay" );
}

TEST_CASE( "#835 -- errno should not be touched by Catch2", "[.][failing][!shouldfail]" ) {
    errno = 1;
    // Check that reporting failed test doesn't change errno.
    CHECK(f() == 0);
    // We want to avoid expanding `errno` macro in assertion, because
    // we capture the expression after macro expansion, and would have
    // to normalize the ways different platforms spell `errno`.
    const auto errno_after = errno;
    REQUIRE(errno_after == 1);
}

TEST_CASE( "#961 -- Dynamically created sections should all be reported", "[.]" ) {
    for (char i = '0'; i < '5'; ++i) {
        SECTION(std::string("Looped section ") + i) {
            SUCCEED( "Everything is OK" );
        }
    }
}

TEST_CASE( "#1175 - Hidden Test", "[.]" ) {
  // Just for checking that hidden test is not listed by default
  SUCCEED();
}

TEMPLATE_TEST_CASE_SIG("#1954 - 7 arg template test case sig compiles", "[regression][.compilation]",
                       ((int Tnx, int Tnu, int Tny, int Tph, int Tch, int Tineq, int Teq), Tnx, Tnu, Tny, Tph, Tch, Tineq, Teq),
                       (1, 1, 1, 1, 1, 0, 0), (5, 1, 1, 1, 1, 0, 0), (5, 3, 1, 1, 1, 0, 0)) {
    SUCCEED();
}

TEST_CASE("Same test name but with different tags is fine", "[.approvals][some-tag]") {}
TEST_CASE("Same test name but with different tags is fine", "[.approvals][other-tag]") {}

// MinGW doesn't support __try, and Clang has only very partial support
#if defined(_MSC_VER)
void throw_and_catch()
{
    __try {
        RaiseException(0xC0000005, 0, 0, NULL);
    }
    __except (1)
    {

    }
}


TEST_CASE("Validate SEH behavior - handled", "[approvals][FatalConditionHandler][CATCH_PLATFORM_WINDOWS]")
{
    // Validate that Catch2 framework correctly handles tests raising and handling SEH exceptions.
    throw_and_catch();
}

void throw_no_catch()
{
    RaiseException(0xC0000005, 0, 0, NULL);
}

TEST_CASE("Validate SEH behavior - unhandled", "[.approvals][FatalConditionHandler][CATCH_PLATFORM_WINDOWS]")
{
    // Validate that Catch2 framework correctly handles tests raising and not handling SEH exceptions.
    throw_no_catch();
}

static LONG CALLBACK dummyExceptionFilter(PEXCEPTION_POINTERS ExceptionInfo) {
    return EXCEPTION_CONTINUE_SEARCH;
}

TEST_CASE("Validate SEH behavior - no crash for stack unwinding", "[approvals][!throws][!shouldfail][FatalConditionHandler][CATCH_PLATFORM_WINDOWS]")
{
    // Trigger stack unwinding with SEH top-level filter changed and validate the test fails expectedly with no application crash
    SetUnhandledExceptionFilter(dummyExceptionFilter);
    throw 1;
}

#endif // _MSC_VER

TEST_CASE( "Comparing (and stringifying) volatile pointers works",
           "[volatile]" ) {
    volatile int* ptr = nullptr;
    REQUIRE_FALSE( ptr );
    REQUIRE( ptr == ptr );
    REQUIRE_FALSE( ptr != ptr );
    REQUIRE_FALSE( ptr < ptr );
    REQUIRE( ptr <= ptr );
    REQUIRE_FALSE( ptr > ptr );
    REQUIRE( ptr >= ptr );
}
