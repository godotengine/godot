
//              Copyright Catch2 Authors
// Distributed under the Boost Software License, Version 1.0.
//   (See accompanying file LICENSE.txt or copy at
//        https://www.boost.org/LICENSE_1_0.txt)

// SPDX-License-Identifier: BSL-1.0

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_template_test_macros.hpp>
#include <catch2/matchers/catch_matchers_exception.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <catch2/matchers/catch_matchers_predicate.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>
#include <catch2/matchers/catch_matchers_vector.hpp>
#include <catch2/matchers/catch_matchers_templated.hpp>

#include <algorithm>
#include <exception>
#include <cmath>
#include <list>
#include <sstream>

#ifdef __clang__
#    pragma clang diagnostic push
#    pragma clang diagnostic ignored "-Wweak-vtables"
#    pragma clang diagnostic ignored "-Wpadded"
#endif

namespace {

    static const char* testStringForMatching() {
        return "this string contains 'abc' as a substring";
    }

    static const char* testStringForMatching2() {
        return "some completely different text that contains one common word";
    }

    static bool alwaysTrue( int ) { return true; }
    static bool alwaysFalse( int ) { return false; }

#ifdef _MSC_VER
#    pragma warning( disable : 4702 ) // Unreachable code -- MSVC 19 (VS 2015)
                                      // sees right through the indirection
#endif

    struct SpecialException : std::exception {
        SpecialException( int i_ ): i( i_ ) {}

        char const* what() const noexcept override {
            return "SpecialException::what";
        }

        int i;
    };

    struct DerivedException : std::exception {
        char const* what() const noexcept override {
            return "DerivedException::what";
        }
    };

    static void doesNotThrow() {}

    [[noreturn]] static void throwsSpecialException( int i ) {
        throw SpecialException{ i };
    }

    [[noreturn]] static void throwsAsInt( int i ) { throw i; }

    [[noreturn]] static void throwsDerivedException() {
        throw DerivedException{};
    }

    class ExceptionMatcher
        : public Catch::Matchers::MatcherBase<SpecialException> {
        int m_expected;

    public:
        ExceptionMatcher( int i ): m_expected( i ) {}

        bool match( SpecialException const& se ) const override {
            return se.i == m_expected;
        }

        std::string describe() const override {
            std::ostringstream ss;
            ss << "special exception has value of " << m_expected;
            return ss.str();
        }
    };

    using namespace Catch::Matchers;

#ifdef __DJGPP__
    static float nextafter( float from, float to ) {
        return ::nextafterf( from, to );
    }

    static double nextafter( double from, double to ) {
        return ::nextafter( from, to );
    }
#else
    using std::nextafter;
#endif

} // end unnamed namespace

TEST_CASE( "String matchers", "[matchers]" ) {
    REQUIRE_THAT( testStringForMatching(), ContainsSubstring( "string" ) );
    REQUIRE_THAT( testStringForMatching(),
                  ContainsSubstring( "string", Catch::CaseSensitive::No ) );
    CHECK_THAT( testStringForMatching(), ContainsSubstring( "abc" ) );
    CHECK_THAT( testStringForMatching(),
                ContainsSubstring( "aBC", Catch::CaseSensitive::No ) );

    CHECK_THAT( testStringForMatching(), StartsWith( "this" ) );
    CHECK_THAT( testStringForMatching(),
                StartsWith( "THIS", Catch::CaseSensitive::No ) );
    CHECK_THAT( testStringForMatching(), EndsWith( "substring" ) );
    CHECK_THAT( testStringForMatching(),
                EndsWith( " SuBsTrInG", Catch::CaseSensitive::No ) );
}

TEST_CASE( "Contains string matcher", "[.][failing][matchers]" ) {
    CHECK_THAT( testStringForMatching(),
                ContainsSubstring( "not there", Catch::CaseSensitive::No ) );
    CHECK_THAT( testStringForMatching(), ContainsSubstring( "STRING" ) );
}

TEST_CASE( "StartsWith string matcher", "[.][failing][matchers]" ) {
    CHECK_THAT( testStringForMatching(), StartsWith( "This String" ) );
    CHECK_THAT( testStringForMatching(),
                StartsWith( "string", Catch::CaseSensitive::No ) );
}

TEST_CASE( "EndsWith string matcher", "[.][failing][matchers]" ) {
    CHECK_THAT( testStringForMatching(), EndsWith( "Substring" ) );
    CHECK_THAT( testStringForMatching(),
                EndsWith( "this", Catch::CaseSensitive::No ) );
}

TEST_CASE( "Equals string matcher", "[.][failing][matchers]" ) {
    CHECK_THAT( testStringForMatching(),
                Equals( "this string contains 'ABC' as a substring" ) );
    CHECK_THAT( testStringForMatching(),
                Equals( "something else", Catch::CaseSensitive::No ) );
}

TEST_CASE( "Equals", "[matchers]" ) {
    CHECK_THAT( testStringForMatching(),
                Equals( "this string contains 'abc' as a substring" ) );
    CHECK_THAT( testStringForMatching(),
                Equals( "this string contains 'ABC' as a substring",
                        Catch::CaseSensitive::No ) );
}

TEST_CASE( "Regex string matcher -- libstdc++-4.8 workaround",
           "[matchers][approvals]" ) {
// DJGPP has similar problem with its regex support as libstdc++ 4.8
#ifndef __DJGPP__
    REQUIRE_THAT( testStringForMatching(),
                  Matches( "this string contains 'abc' as a substring" ) );
    REQUIRE_THAT( testStringForMatching(),
                  Matches( "this string CONTAINS 'abc' as a substring",
                           Catch::CaseSensitive::No ) );
    REQUIRE_THAT( testStringForMatching(),
                  Matches( "^this string contains 'abc' as a substring$" ) );
    REQUIRE_THAT( testStringForMatching(), Matches( "^.* 'abc' .*$" ) );
    REQUIRE_THAT( testStringForMatching(),
                  Matches( "^.* 'ABC' .*$", Catch::CaseSensitive::No ) );
#endif

    REQUIRE_THAT( testStringForMatching2(),
                  !Matches( "this string contains 'abc' as a substring" ) );
}

TEST_CASE( "Regex string matcher", "[matchers][.failing]" ) {
    CHECK_THAT( testStringForMatching(),
                Matches( "this STRING contains 'abc' as a substring" ) );
    CHECK_THAT( testStringForMatching(),
                Matches( "contains 'abc' as a substring" ) );
    CHECK_THAT( testStringForMatching(),
                Matches( "this string contains 'abc' as a" ) );
}

TEST_CASE( "Matchers can be (AllOf) composed with the && operator",
           "[matchers][operators][operator&&]" ) {
    CHECK_THAT( testStringForMatching(),
                ContainsSubstring( "string" ) && ContainsSubstring( "abc" ) &&
                    ContainsSubstring( "substring" ) && ContainsSubstring( "contains" ) );
}

TEST_CASE( "Matchers can be (AnyOf) composed with the || operator",
           "[matchers][operators][operator||]" ) {
    CHECK_THAT( testStringForMatching(),
                ContainsSubstring( "string" ) || ContainsSubstring( "different" ) ||
                    ContainsSubstring( "random" ) );
    CHECK_THAT( testStringForMatching2(),
                ContainsSubstring( "string" ) || ContainsSubstring( "different" ) ||
                    ContainsSubstring( "random" ) );
}

TEST_CASE( "Matchers can be composed with both && and ||",
           "[matchers][operators][operator||][operator&&]" ) {
    CHECK_THAT( testStringForMatching(),
                ( ContainsSubstring( "string" ) || ContainsSubstring( "different" ) ) &&
                    ContainsSubstring( "substring" ) );
}

TEST_CASE( "Matchers can be composed with both && and || - failing",
           "[matchers][operators][operator||][operator&&][.failing]" ) {
    CHECK_THAT( testStringForMatching(),
                ( ContainsSubstring( "string" ) || ContainsSubstring( "different" ) ) &&
                    ContainsSubstring( "random" ) );
}

TEST_CASE( "Matchers can be negated (Not) with the ! operator",
           "[matchers][operators][not]" ) {
    CHECK_THAT( testStringForMatching(), !ContainsSubstring( "different" ) );
}

TEST_CASE( "Matchers can be negated (Not) with the ! operator - failing",
           "[matchers][operators][not][.failing]" ) {
    CHECK_THAT( testStringForMatching(), !ContainsSubstring( "substring" ) );
}

template <typename T> struct CustomAllocator : private std::allocator<T> {
    using size_type = size_t;
    using difference_type = ptrdiff_t;
    using pointer = T*;
    using const_pointer = const T*;
    using reference = T&;
    using const_reference = const T&;
    using value_type = T;

    template <typename U> struct rebind { using other = CustomAllocator<U>; };

    using propagate_on_container_move_assignment = std::true_type;
    using is_always_equal = std::true_type;

    CustomAllocator() = default;

    CustomAllocator( const CustomAllocator& other ):
        std::allocator<T>( other ) {}

    template <typename U> CustomAllocator( const CustomAllocator<U>& ) {}

    ~CustomAllocator() = default;

    using std::allocator<T>::allocate;
    using std::allocator<T>::deallocate;
};

TEST_CASE( "Vector matchers", "[matchers][vector]" ) {
    std::vector<int> v;
    v.push_back( 1 );
    v.push_back( 2 );
    v.push_back( 3 );

    std::vector<int> v2;
    v2.push_back( 1 );
    v2.push_back( 2 );

    std::vector<double> v3;
    v3.push_back( 1 );
    v3.push_back( 2 );
    v3.push_back( 3 );

    std::vector<double> v4;
    v4.push_back( 1 + 1e-8 );
    v4.push_back( 2 + 1e-8 );
    v4.push_back( 3 + 1e-8 );

    std::vector<int, CustomAllocator<int>> v5;
    v5.push_back( 1 );
    v5.push_back( 2 );
    v5.push_back( 3 );

    std::vector<int, CustomAllocator<int>> v6;
    v6.push_back( 1 );
    v6.push_back( 2 );

    std::vector<int> empty;

    SECTION( "Contains (element)" ) {
        CHECK_THAT( v, VectorContains( 1 ) );
        CHECK_THAT( v, VectorContains( 2 ) );
        CHECK_THAT( v5, ( VectorContains<int, CustomAllocator<int>>( 2 ) ) );
    }
    SECTION( "Contains (vector)" ) {
        CHECK_THAT( v, Contains( v2 ) );
        CHECK_THAT( v, Contains<int>( { 1, 2 } ) );
        CHECK_THAT( v5,
                    ( Contains<int, std::allocator<int>, CustomAllocator<int>>(
                        v2 ) ) );

        v2.push_back( 3 ); // now exactly matches
        CHECK_THAT( v, Contains( v2 ) );

        CHECK_THAT( v, Contains( empty ) );
        CHECK_THAT( empty, Contains( empty ) );

        CHECK_THAT( v5,
                    ( Contains<int, std::allocator<int>, CustomAllocator<int>>(
                        v2 ) ) );
        CHECK_THAT( v5, Contains( v6 ) );
    }
    SECTION( "Contains (element), composed" ) {
        CHECK_THAT( v, VectorContains( 1 ) && VectorContains( 2 ) );
    }

    SECTION( "Equals" ) {

        // Same vector
        CHECK_THAT( v, Equals( v ) );

        CHECK_THAT( empty, Equals( empty ) );

        // Different vector with same elements
        CHECK_THAT( v, Equals<int>( { 1, 2, 3 } ) );
        v2.push_back( 3 );
        CHECK_THAT( v, Equals( v2 ) );

        CHECK_THAT(
            v5,
            ( Equals<int, std::allocator<int>, CustomAllocator<int>>( v2 ) ) );

        v6.push_back( 3 );
        CHECK_THAT( v5, Equals( v6 ) );
    }
    SECTION( "UnorderedEquals" ) {
        CHECK_THAT( v, UnorderedEquals( v ) );
        CHECK_THAT( v, UnorderedEquals<int>( { 3, 2, 1 } ) );
        CHECK_THAT( empty, UnorderedEquals( empty ) );

        auto permuted = v;
        std::next_permutation( begin( permuted ), end( permuted ) );
        REQUIRE_THAT( permuted, UnorderedEquals( v ) );

        std::reverse( begin( permuted ), end( permuted ) );
        REQUIRE_THAT( permuted, UnorderedEquals( v ) );

        CHECK_THAT(
            v5,
            ( UnorderedEquals<int, std::allocator<int>, CustomAllocator<int>>(
                permuted ) ) );

        auto v5_permuted = v5;
        std::next_permutation( begin( v5_permuted ), end( v5_permuted ) );
        CHECK_THAT( v5_permuted, UnorderedEquals( v5 ) );
    }
}

TEST_CASE( "Vector matchers that fail", "[matchers][vector][.][failing]" ) {
    std::vector<int> v;
    v.push_back( 1 );
    v.push_back( 2 );
    v.push_back( 3 );

    std::vector<int> v2;
    v2.push_back( 1 );
    v2.push_back( 2 );

    std::vector<double> v3;
    v3.push_back( 1 );
    v3.push_back( 2 );
    v3.push_back( 3 );

    std::vector<double> v4;
    v4.push_back( 1.1 );
    v4.push_back( 2.1 );
    v4.push_back( 3.1 );

    std::vector<int> empty;

    SECTION( "Contains (element)" ) {
        CHECK_THAT( v, VectorContains( -1 ) );
        CHECK_THAT( empty, VectorContains( 1 ) );
    }
    SECTION( "Contains (vector)" ) {
        CHECK_THAT( empty, Contains( v ) );
        v2.push_back( 4 );
        CHECK_THAT( v, Contains( v2 ) );
    }

    SECTION( "Equals" ) {

        CHECK_THAT( v, Equals( v2 ) );
        CHECK_THAT( v2, Equals( v ) );
        CHECK_THAT( empty, Equals( v ) );
        CHECK_THAT( v, Equals( empty ) );
    }
    SECTION( "UnorderedEquals" ) {
        CHECK_THAT( v, UnorderedEquals( empty ) );
        CHECK_THAT( empty, UnorderedEquals( v ) );

        auto permuted = v;
        std::next_permutation( begin( permuted ), end( permuted ) );
        permuted.pop_back();
        CHECK_THAT( permuted, UnorderedEquals( v ) );

        std::reverse( begin( permuted ), end( permuted ) );
        CHECK_THAT( permuted, UnorderedEquals( v ) );
    }
}

namespace {
    struct SomeType {
        int i;
        friend bool operator==( SomeType lhs, SomeType rhs ) {
            return lhs.i == rhs.i;
        }
    };
} // end anonymous namespace

TEST_CASE( "Vector matcher with elements without !=", "[matchers][vector][approvals]" ) {
    std::vector<SomeType> lhs, rhs;
    lhs.push_back( { 1 } );
    lhs.push_back( { 2 } );
    rhs.push_back( { 1 } );
    rhs.push_back( { 1 } );

    REQUIRE_THAT( lhs, !Equals(rhs) );
}

TEST_CASE( "Exception matchers that succeed",
           "[matchers][exceptions][!throws]" ) {
    CHECK_THROWS_MATCHES(
        throwsSpecialException( 1 ), SpecialException, ExceptionMatcher{ 1 } );
    REQUIRE_THROWS_MATCHES(
        throwsSpecialException( 2 ), SpecialException, ExceptionMatcher{ 2 } );
}

TEST_CASE( "Exception matchers that fail",
           "[matchers][exceptions][!throws][.failing]" ) {
    SECTION( "No exception" ) {
        CHECK_THROWS_MATCHES(
            doesNotThrow(), SpecialException, ExceptionMatcher{ 1 } );
        REQUIRE_THROWS_MATCHES(
            doesNotThrow(), SpecialException, ExceptionMatcher{ 1 } );
    }
    SECTION( "Type mismatch" ) {
        CHECK_THROWS_MATCHES(
            throwsAsInt( 1 ), SpecialException, ExceptionMatcher{ 1 } );
        REQUIRE_THROWS_MATCHES(
            throwsAsInt( 1 ), SpecialException, ExceptionMatcher{ 1 } );
    }
    SECTION( "Contents are wrong" ) {
        CHECK_THROWS_MATCHES( throwsSpecialException( 3 ),
                              SpecialException,
                              ExceptionMatcher{ 1 } );
        REQUIRE_THROWS_MATCHES( throwsSpecialException( 4 ),
                                SpecialException,
                                ExceptionMatcher{ 1 } );
    }
}

TEST_CASE( "Floating point matchers: float", "[matchers][floating-point]" ) {
    SECTION( "Relative" ) {
        REQUIRE_THAT( 10.f, WithinRel( 11.1f, 0.1f ) );
        REQUIRE_THAT( 10.f, !WithinRel( 11.2f, 0.1f ) );
        REQUIRE_THAT( 1.f, !WithinRel( 0.f, 0.99f ) );
        REQUIRE_THAT( -0.f, WithinRel( 0.f ) );
        SECTION( "Some subnormal values" ) {
            auto v1 = std::numeric_limits<float>::min();
            auto v2 = v1;
            for ( int i = 0; i < 5; ++i ) {
                v2 = std::nextafter( v1, 0.f );
            }
            REQUIRE_THAT( v1, WithinRel( v2 ) );
        }
    }
    SECTION( "Margin" ) {
        REQUIRE_THAT( 1.f, WithinAbs( 1.f, 0 ) );
        REQUIRE_THAT( 0.f, WithinAbs( 1.f, 1 ) );

        REQUIRE_THAT( 0.f, !WithinAbs( 1.f, 0.99f ) );
        REQUIRE_THAT( 0.f, !WithinAbs( 1.f, 0.99f ) );

        REQUIRE_THAT( 0.f, WithinAbs( -0.f, 0 ) );

        REQUIRE_THAT( 11.f, !WithinAbs( 10.f, 0.5f ) );
        REQUIRE_THAT( 10.f, !WithinAbs( 11.f, 0.5f ) );
        REQUIRE_THAT( -10.f, WithinAbs( -10.f, 0.5f ) );
        REQUIRE_THAT( -10.f, WithinAbs( -9.6f, 0.5f ) );
    }
    SECTION( "ULPs" ) {
        REQUIRE_THAT( 1.f, WithinULP( 1.f, 0 ) );
        REQUIRE_THAT(-1.f, WithinULP( -1.f, 0 ) );

        REQUIRE_THAT( nextafter( 1.f, 2.f ), WithinULP( 1.f, 1 ) );
        REQUIRE_THAT( 0.f, WithinULP( nextafter( 0.f, 1.f ), 1 ) );
        REQUIRE_THAT( 1.f, WithinULP( nextafter( 1.f, 0.f ), 1 ) );
        REQUIRE_THAT( 1.f, !WithinULP( nextafter( 1.f, 2.f ), 0 ) );

        REQUIRE_THAT( 1.f, WithinULP( 1.f, 0 ) );
        REQUIRE_THAT( -0.f, WithinULP( 0.f, 0 ) );
    }
    SECTION( "Composed" ) {
        REQUIRE_THAT( 1.f, WithinAbs( 1.f, 0.5 ) || WithinULP( 1.f, 1 ) );
        REQUIRE_THAT( 1.f, WithinAbs( 2.f, 0.5 ) || WithinULP( 1.f, 0 ) );
        REQUIRE_THAT( 0.0001f,
                      WithinAbs( 0.f, 0.001f ) || WithinRel( 0.f, 0.1f ) );
    }
    SECTION( "Constructor validation" ) {
        REQUIRE_NOTHROW( WithinAbs( 1.f, 0.f ) );
        REQUIRE_THROWS_AS( WithinAbs( 1.f, -1.f ), std::domain_error );

        REQUIRE_NOTHROW( WithinULP( 1.f, 0 ) );
        REQUIRE_THROWS_AS( WithinULP( 1.f, static_cast<uint64_t>( -1 ) ),
                           std::domain_error );

        REQUIRE_NOTHROW( WithinRel( 1.f, 0.f ) );
        REQUIRE_THROWS_AS( WithinRel( 1.f, -0.2f ), std::domain_error );
        REQUIRE_THROWS_AS( WithinRel( 1.f, 1.f ), std::domain_error );
    }
    SECTION( "IsNaN" ) {
        REQUIRE_THAT( 1., !IsNaN() );
    }
}

TEST_CASE( "Floating point matchers: double", "[matchers][floating-point]" ) {
    SECTION( "Relative" ) {
        REQUIRE_THAT( 10., WithinRel( 11.1, 0.1 ) );
        REQUIRE_THAT( 10., !WithinRel( 11.2, 0.1 ) );
        REQUIRE_THAT( 1., !WithinRel( 0., 0.99 ) );
        REQUIRE_THAT( -0., WithinRel( 0. ) );
        SECTION( "Some subnormal values" ) {
            auto v1 = std::numeric_limits<double>::min();
            auto v2 = v1;
            for ( int i = 0; i < 5; ++i ) {
                v2 = std::nextafter( v1, 0 );
            }
            REQUIRE_THAT( v1, WithinRel( v2 ) );
        }
    }
    SECTION( "Margin" ) {
        REQUIRE_THAT( 1., WithinAbs( 1., 0 ) );
        REQUIRE_THAT( 0., WithinAbs( 1., 1 ) );

        REQUIRE_THAT( 0., !WithinAbs( 1., 0.99 ) );
        REQUIRE_THAT( 0., !WithinAbs( 1., 0.99 ) );

        REQUIRE_THAT( 11., !WithinAbs( 10., 0.5 ) );
        REQUIRE_THAT( 10., !WithinAbs( 11., 0.5 ) );
        REQUIRE_THAT( -10., WithinAbs( -10., 0.5 ) );
        REQUIRE_THAT( -10., WithinAbs( -9.6, 0.5 ) );
    }
    SECTION( "ULPs" ) {
        REQUIRE_THAT( 1., WithinULP( 1., 0 ) );

        REQUIRE_THAT( nextafter( 1., 2. ), WithinULP( 1., 1 ) );
        REQUIRE_THAT( 0., WithinULP( nextafter( 0., 1. ), 1 ) );
        REQUIRE_THAT( 1., WithinULP( nextafter( 1., 0. ), 1 ) );
        REQUIRE_THAT( 1., !WithinULP( nextafter( 1., 2. ), 0 ) );

        REQUIRE_THAT( 1., WithinULP( 1., 0 ) );
        REQUIRE_THAT( -0., WithinULP( 0., 0 ) );
    }
    SECTION( "Composed" ) {
        REQUIRE_THAT( 1., WithinAbs( 1., 0.5 ) || WithinULP( 2., 1 ) );
        REQUIRE_THAT( 1., WithinAbs( 2., 0.5 ) || WithinULP( 1., 0 ) );
        REQUIRE_THAT( 0.0001, WithinAbs( 0., 0.001 ) || WithinRel( 0., 0.1 ) );
    }
    SECTION( "Constructor validation" ) {
        REQUIRE_NOTHROW( WithinAbs( 1., 0. ) );
        REQUIRE_THROWS_AS( WithinAbs( 1., -1. ), std::domain_error );

        REQUIRE_NOTHROW( WithinULP( 1., 0 ) );

        REQUIRE_NOTHROW( WithinRel( 1., 0. ) );
        REQUIRE_THROWS_AS( WithinRel( 1., -0.2 ), std::domain_error );
        REQUIRE_THROWS_AS( WithinRel( 1., 1. ), std::domain_error );
    }
    SECTION("IsNaN") {
        REQUIRE_THAT( 1., !IsNaN() );
    }
}

TEST_CASE( "Floating point matchers that are problematic in approvals",
           "[approvals][matchers][floating-point]" ) {
    REQUIRE_THAT( NAN, !WithinAbs( NAN, 0 ) );
    REQUIRE_THAT( NAN, !( WithinAbs( NAN, 100 ) || WithinULP( NAN, 123 ) ) );
    REQUIRE_THAT( NAN, !WithinULP( NAN, 123 ) );
    REQUIRE_THAT( INFINITY, WithinRel( INFINITY ) );
    REQUIRE_THAT( -INFINITY, !WithinRel( INFINITY ) );
    REQUIRE_THAT( 1., !WithinRel( INFINITY ) );
    REQUIRE_THAT( INFINITY, !WithinRel( 1. ) );
    REQUIRE_THAT( NAN, !WithinRel( NAN ) );
    REQUIRE_THAT( 1., !WithinRel( NAN ) );
    REQUIRE_THAT( NAN, !WithinRel( 1. ) );
    REQUIRE_THAT( NAN, IsNaN() );
    REQUIRE_THAT( static_cast<double>(NAN), IsNaN() );
}

TEST_CASE( "Arbitrary predicate matcher", "[matchers][generic]" ) {
    SECTION( "Function pointer" ) {
        REQUIRE_THAT( 1, Predicate<int>( alwaysTrue, "always true" ) );
        REQUIRE_THAT( 1, !Predicate<int>( alwaysFalse, "always false" ) );
    }
    SECTION( "Lambdas + different type" ) {
        REQUIRE_THAT( "Hello olleH",
                      Predicate<std::string>(
                          []( std::string const& str ) -> bool {
                              return str.front() == str.back();
                          },
                          "First and last character should be equal" ) );

        REQUIRE_THAT(
            "This wouldn't pass",
            !Predicate<std::string>( []( std::string const& str ) -> bool {
                return str.front() == str.back();
            } ) );
    }
}

TEST_CASE( "Regression test #1", "[matchers][vector]" ) {
    // At some point, UnorderedEqualsMatcher skipped
    // mismatched prefixed before doing the comparison itself
    std::vector<char> actual = { 'a', 'b' };
    std::vector<char> expected = { 'c', 'b' };

    CHECK_THAT( actual, !UnorderedEquals( expected ) );
}

TEST_CASE( "Predicate matcher can accept const char*",
           "[matchers][compilation]" ) {
    REQUIRE_THAT( "foo", Predicate<const char*>( []( const char* const& ) {
                      return true;
                  } ) );
}

TEST_CASE( "Vector Approx matcher", "[matchers][approx][vector]" ) {
    using Catch::Matchers::Approx;
    SECTION( "Empty vector is roughly equal to an empty vector" ) {
        std::vector<double> empty;
        REQUIRE_THAT( empty, Approx( empty ) );
    }
    SECTION( "Vectors with elements" ) {
        std::vector<double> v1( { 1., 2., 3. } );
        SECTION( "A vector is approx equal to itself" ) {
            REQUIRE_THAT( v1, Approx( v1 ) );
            REQUIRE_THAT( v1, Approx<double>( { 1., 2., 3. } ) );
        }
        std::vector<double> v2( { 1.5, 2.5, 3.5 } );
        SECTION( "Different length" ) {
            auto temp( v1 );
            temp.push_back( 4 );
            REQUIRE_THAT( v1, !Approx( temp ) );
        }
        SECTION( "Same length, different elements" ) {
            REQUIRE_THAT( v1, !Approx( v2 ) );
            REQUIRE_THAT( v1, Approx( v2 ).margin( 0.5 ) );
            REQUIRE_THAT( v1, Approx( v2 ).epsilon( 0.5 ) );
            REQUIRE_THAT( v1, Approx( v2 ).epsilon( 0.1 ).scale( 500 ) );
        }
    }
}

TEST_CASE( "Vector Approx matcher -- failing",
           "[matchers][approx][vector][.failing]" ) {
    using Catch::Matchers::Approx;
    SECTION( "Empty and non empty vectors are not approx equal" ) {
        std::vector<double> empty, t1( { 1, 2 } );
        CHECK_THAT( empty, Approx( t1 ) );
    }
    SECTION( "Just different vectors" ) {
        std::vector<double> v1( { 2., 4., 6. } ), v2( { 1., 3., 5. } );
        CHECK_THAT( v1, Approx( v2 ) );
    }
}

TEST_CASE( "Exceptions matchers", "[matchers][exceptions][!throws]" ) {
    REQUIRE_THROWS_MATCHES( throwsDerivedException(),
                            DerivedException,
                            Message( "DerivedException::what" ) );
    REQUIRE_THROWS_MATCHES( throwsDerivedException(),
                            DerivedException,
                            !Message( "derivedexception::what" ) );
    REQUIRE_THROWS_MATCHES( throwsSpecialException( 2 ),
                            SpecialException,
                            !Message( "DerivedException::what" ) );
    REQUIRE_THROWS_MATCHES( throwsSpecialException( 2 ),
                            SpecialException,
                            Message( "SpecialException::what" ) );
}

TEST_CASE( "Exception message can be matched", "[matchers][exceptions][!throws]" ) {
    REQUIRE_THROWS_MATCHES( throwsDerivedException(),
                            DerivedException,
                            MessageMatches( StartsWith( "Derived" ) ) );
    REQUIRE_THROWS_MATCHES( throwsDerivedException(),
                            DerivedException,
                            MessageMatches( EndsWith( "::what" ) ) );
    REQUIRE_THROWS_MATCHES( throwsDerivedException(),
                            DerivedException,
                            MessageMatches( !StartsWith( "::what" ) ) );
    REQUIRE_THROWS_MATCHES( throwsSpecialException( 2 ),
                            SpecialException,
                            MessageMatches( StartsWith( "Special" ) ) );
}

struct CheckedTestingMatcher : Catch::Matchers::MatcherBase<int> {
    mutable bool matchCalled = false;
    bool matchSucceeds = false;

    bool match( int const& ) const override {
        matchCalled = true;
        return matchSucceeds;
    }
    std::string describe() const override {
        return "CheckedTestingMatcher set to " +
               ( matchSucceeds ? std::string( "succeed" )
                               : std::string( "fail" ) );
    }
};

TEST_CASE( "Composed matchers shortcircuit", "[matchers][composed]" ) {
    // Check that if first returns false, second is not touched
    CheckedTestingMatcher first, second;
    SECTION( "MatchAllOf" ) {
        first.matchSucceeds = false;

        Detail::MatchAllOf<int> matcher =
            Detail::MatchAllOf<int>{} && first && second;
        CHECK_FALSE( matcher.match( 1 ) );

        // These two assertions are the important ones
        REQUIRE( first.matchCalled );
        REQUIRE( !second.matchCalled );
    }
    // Check that if first returns true, second is not touched
    SECTION( "MatchAnyOf" ) {
        first.matchSucceeds = true;

        Detail::MatchAnyOf<int> matcher =
            Detail::MatchAnyOf<int>{} || first || second;
        CHECK( matcher.match( 1 ) );

        // These two assertions are the important ones
        REQUIRE( first.matchCalled );
        REQUIRE( !second.matchCalled );
    }
}

struct CheckedTestingGenericMatcher : Catch::Matchers::MatcherGenericBase {
    mutable bool matchCalled = false;
    bool matchSucceeds = false;

    bool match( int const& ) const {
        matchCalled = true;
        return matchSucceeds;
    }
    std::string describe() const override {
        return "CheckedTestingGenericMatcher set to " +
               ( matchSucceeds ? std::string( "succeed" )
                               : std::string( "fail" ) );
    }
};

TEST_CASE( "Composed generic matchers shortcircuit",
           "[matchers][composed][generic]" ) {
    // Check that if first returns false, second is not touched
    CheckedTestingGenericMatcher first, second;
    SECTION( "MatchAllOf" ) {
        first.matchSucceeds = false;

        Detail::MatchAllOfGeneric<CheckedTestingGenericMatcher,
                                  CheckedTestingGenericMatcher>
            matcher{ first, second };

        CHECK_FALSE( matcher.match( 1 ) );

        // These two assertions are the important ones
        REQUIRE( first.matchCalled );
        REQUIRE( !second.matchCalled );
    }
    // Check that if first returns true, second is not touched
    SECTION( "MatchAnyOf" ) {
        first.matchSucceeds = true;

        Detail::MatchAnyOfGeneric<CheckedTestingGenericMatcher,
                                  CheckedTestingGenericMatcher>
            matcher{ first, second };
        CHECK( matcher.match( 1 ) );

        // These two assertions are the important ones
        REQUIRE( first.matchCalled );
        REQUIRE( !second.matchCalled );
    }
}

template <typename Range>
struct EqualsRangeMatcher : Catch::Matchers::MatcherGenericBase {

    EqualsRangeMatcher( Range const& range ): m_range{ range } {}

    template <typename OtherRange> bool match( OtherRange const& other ) const {
        using std::begin;
        using std::end;

        return std::equal(
            begin( m_range ), end( m_range ), begin( other ), end( other ) );
    }

    std::string describe() const override {
        return "Equals: " + Catch::rangeToString( m_range );
    }

private:
    Range const& m_range;
};

template <typename Range>
auto EqualsRange( const Range& range ) -> EqualsRangeMatcher<Range> {
    return EqualsRangeMatcher<Range>{ range };
}

TEST_CASE( "Combining templated matchers", "[matchers][templated]" ) {
    std::array<int, 3> container{ { 1, 2, 3 } };

    std::array<int, 3> a{ { 1, 2, 3 } };
    std::vector<int> b{ 0, 1, 2 };
    std::list<int> c{ 4, 5, 6 };

    REQUIRE_THAT( container,
                  EqualsRange( a ) || EqualsRange( b ) || EqualsRange( c ) );
}

TEST_CASE( "Combining templated and concrete matchers",
           "[matchers][templated]" ) {
    std::vector<int> vec{ 1, 3, 5 };

    std::array<int, 3> a{ { 5, 3, 1 } };

    REQUIRE_THAT( vec,
                  Predicate<std::vector<int>>(
                      []( auto const& v ) {
                          return std::all_of(
                              v.begin(), v.end(), []( int elem ) {
                                  return elem % 2 == 1;
                              } );
                      },
                      "All elements are odd" ) &&
                      !EqualsRange( a ) );

    const std::string str = "foobar";
    const std::array<char, 6> arr{ { 'f', 'o', 'o', 'b', 'a', 'r' } };
    const std::array<char, 6> bad_arr{ { 'o', 'o', 'f', 'b', 'a', 'r' } };

    using Catch::Matchers::EndsWith;
    using Catch::Matchers::StartsWith;

    REQUIRE_THAT(
        str, StartsWith( "foo" ) && EqualsRange( arr ) && EndsWith( "bar" ) );
    REQUIRE_THAT( str,
                  StartsWith( "foo" ) && !EqualsRange( bad_arr ) &&
                      EndsWith( "bar" ) );

    REQUIRE_THAT(
        str, EqualsRange( arr ) && StartsWith( "foo" ) && EndsWith( "bar" ) );
    REQUIRE_THAT( str,
                  !EqualsRange( bad_arr ) && StartsWith( "foo" ) &&
                      EndsWith( "bar" ) );

    REQUIRE_THAT( str,
                  EqualsRange( bad_arr ) ||
                      ( StartsWith( "foo" ) && EndsWith( "bar" ) ) );
    REQUIRE_THAT( str,
                  ( StartsWith( "foo" ) && EndsWith( "bar" ) ) ||
                      EqualsRange( bad_arr ) );
}

TEST_CASE( "Combining concrete matchers does not use templated matchers",
           "[matchers][templated]" ) {
    using Catch::Matchers::EndsWith;
    using Catch::Matchers::StartsWith;

    STATIC_REQUIRE(
        std::is_same<decltype( StartsWith( "foo" ) ||
                               ( StartsWith( "bar" ) && EndsWith( "bar" ) &&
                                 !EndsWith( "foo" ) ) ),
                     Catch::Matchers::Detail::MatchAnyOf<std::string>>::value );
}

struct MatcherA : Catch::Matchers::MatcherGenericBase {
    std::string describe() const override {
        return "equals: (int) 1 or (string) \"1\"";
    }
    bool match( int i ) const { return i == 1; }
    bool match( std::string const& s ) const { return s == "1"; }
};

struct MatcherB : Catch::Matchers::MatcherGenericBase {
    std::string describe() const override { return "equals: (long long) 1"; }
    bool match( long long l ) const { return l == 1ll; }
};

struct MatcherC : Catch::Matchers::MatcherGenericBase {
    std::string describe() const override { return "equals: (T) 1"; }
    template <typename T> bool match( T t ) const { return t == T{ 1 }; }
};

struct MatcherD : Catch::Matchers::MatcherGenericBase {
    std::string describe() const override { return "equals: true"; }
    bool match( bool b ) const { return b == true; }
};

TEST_CASE( "Combining only templated matchers", "[matchers][templated]" ) {
    STATIC_REQUIRE(
        std::is_same<decltype( MatcherA() || MatcherB() ),
                     Catch::Matchers::Detail::
                         MatchAnyOfGeneric<MatcherA, MatcherB>>::value );

    REQUIRE_THAT( 1, MatcherA() || MatcherB() );

    STATIC_REQUIRE(
        std::is_same<decltype( MatcherA() && MatcherB() ),
                     Catch::Matchers::Detail::
                         MatchAllOfGeneric<MatcherA, MatcherB>>::value );

    REQUIRE_THAT( 1, MatcherA() && MatcherB() );

    STATIC_REQUIRE(
        std::is_same<
            decltype( MatcherA() || !MatcherB() ),
            Catch::Matchers::Detail::MatchAnyOfGeneric<
                MatcherA,
                Catch::Matchers::Detail::MatchNotOfGeneric<MatcherB>>>::value );

    REQUIRE_THAT( 1, MatcherA() || !MatcherB() );
}

TEST_CASE( "Combining MatchAnyOfGeneric does not nest",
           "[matchers][templated]" ) {
    // MatchAnyOfGeneric LHS + some matcher RHS
    STATIC_REQUIRE(
        std::is_same<
            decltype( ( MatcherA() || MatcherB() ) || MatcherC() ),
            Catch::Matchers::Detail::
                MatchAnyOfGeneric<MatcherA, MatcherB, MatcherC>>::value );

    REQUIRE_THAT( 1, ( MatcherA() || MatcherB() ) || MatcherC() );

    // some matcher LHS + MatchAnyOfGeneric RHS
    STATIC_REQUIRE(
        std::is_same<
            decltype( MatcherA() || ( MatcherB() || MatcherC() ) ),
            Catch::Matchers::Detail::
                MatchAnyOfGeneric<MatcherA, MatcherB, MatcherC>>::value );

    REQUIRE_THAT( 1, MatcherA() || ( MatcherB() || MatcherC() ) );

    // MatchAnyOfGeneric LHS + MatchAnyOfGeneric RHS
    STATIC_REQUIRE(
        std::is_same<
            decltype( ( MatcherA() || MatcherB() ) ||
                      ( MatcherC() || MatcherD() ) ),
            Catch::Matchers::Detail::
                MatchAnyOfGeneric<MatcherA, MatcherB, MatcherC, MatcherD>>::
            value );

    REQUIRE_THAT(
        1, ( MatcherA() || MatcherB() ) || ( MatcherC() || MatcherD() ) );
}

TEST_CASE( "Combining MatchAllOfGeneric does not nest",
           "[matchers][templated]" ) {
    // MatchAllOfGeneric lhs + some matcher RHS
    STATIC_REQUIRE(
        std::is_same<
            decltype( ( MatcherA() && MatcherB() ) && MatcherC() ),
            Catch::Matchers::Detail::
                MatchAllOfGeneric<MatcherA, MatcherB, MatcherC>>::value );

    REQUIRE_THAT( 1, ( MatcherA() && MatcherB() ) && MatcherC() );

    // some matcher LHS + MatchAllOfGeneric RSH
    STATIC_REQUIRE(
        std::is_same<
            decltype( MatcherA() && ( MatcherB() && MatcherC() ) ),
            Catch::Matchers::Detail::
                MatchAllOfGeneric<MatcherA, MatcherB, MatcherC>>::value );

    REQUIRE_THAT( 1, MatcherA() && ( MatcherB() && MatcherC() ) );

    // MatchAllOfGeneric LHS + MatchAllOfGeneric RHS
    STATIC_REQUIRE(
        std::is_same<
            decltype( ( MatcherA() && MatcherB() ) &&
                      ( MatcherC() && MatcherD() ) ),
            Catch::Matchers::Detail::
                MatchAllOfGeneric<MatcherA, MatcherB, MatcherC, MatcherD>>::
            value );

    REQUIRE_THAT(
        1, ( MatcherA() && MatcherB() ) && ( MatcherC() && MatcherD() ) );
}

TEST_CASE( "Combining MatchNotOfGeneric does not nest",
           "[matchers][templated]" ) {
    STATIC_REQUIRE(
        std::is_same<
            decltype( !MatcherA() ),
            Catch::Matchers::Detail::MatchNotOfGeneric<MatcherA>>::value );

    REQUIRE_THAT( 0, !MatcherA() );

    STATIC_REQUIRE(
        std::is_same<decltype( !!MatcherA() ), MatcherA const&>::value );

    REQUIRE_THAT( 1, !!MatcherA() );

    STATIC_REQUIRE(
        std::is_same<
            decltype( !!!MatcherA() ),
            Catch::Matchers::Detail::MatchNotOfGeneric<MatcherA>>::value );

    REQUIRE_THAT( 0, !!!MatcherA() );

    STATIC_REQUIRE(
        std::is_same<decltype( !!!!MatcherA() ), MatcherA const&>::value );

    REQUIRE_THAT( 1, !!!!MatcherA() );
}

struct EvilAddressOfOperatorUsed : std::exception {
    const char* what() const noexcept override {
        return "overloaded address-of operator of matcher was used instead of "
               "std::addressof";
    }
};

struct EvilCommaOperatorUsed : std::exception {
    const char* what() const noexcept override {
        return "overloaded comma operator of matcher was used";
    }
};

struct EvilMatcher : Catch::Matchers::MatcherGenericBase {
    std::string describe() const override { return "equals: 45"; }

    bool match( int i ) const { return i == 45; }

    EvilMatcher const* operator&() const { throw EvilAddressOfOperatorUsed(); }

    int operator,( EvilMatcher const& ) const { throw EvilCommaOperatorUsed(); }
};

TEST_CASE( "Overloaded comma or address-of operators are not used",
           "[matchers][templated]" ) {
    REQUIRE_THROWS_AS( ( EvilMatcher(), EvilMatcher() ),
                       EvilCommaOperatorUsed );
    REQUIRE_THROWS_AS( &EvilMatcher(), EvilAddressOfOperatorUsed );
    REQUIRE_NOTHROW( EvilMatcher() || ( EvilMatcher() && !EvilMatcher() ) );
    REQUIRE_NOTHROW( ( EvilMatcher() && EvilMatcher() ) || !EvilMatcher() );
}

struct ImmovableMatcher : Catch::Matchers::MatcherGenericBase {
    ImmovableMatcher() = default;
    ImmovableMatcher( ImmovableMatcher const& ) = delete;
    ImmovableMatcher( ImmovableMatcher&& ) = delete;
    ImmovableMatcher& operator=( ImmovableMatcher const& ) = delete;
    ImmovableMatcher& operator=( ImmovableMatcher&& ) = delete;

    std::string describe() const override { return "always false"; }

    template <typename T> bool match( T&& ) const { return false; }
};

struct MatcherWasMovedOrCopied : std::exception {
    const char* what() const noexcept override {
        return "attempted to copy or move a matcher";
    }
};

struct ThrowOnCopyOrMoveMatcher : Catch::Matchers::MatcherGenericBase {
    ThrowOnCopyOrMoveMatcher() = default;

    [[noreturn]] ThrowOnCopyOrMoveMatcher( ThrowOnCopyOrMoveMatcher const& other ):
        Catch::Matchers::MatcherGenericBase( other ) {
        throw MatcherWasMovedOrCopied();
    }
    // NOLINTNEXTLINE(performance-noexcept-move-constructor)
    [[noreturn]] ThrowOnCopyOrMoveMatcher( ThrowOnCopyOrMoveMatcher&& other ):
        Catch::Matchers::MatcherGenericBase( CATCH_MOVE(other) ) {
        throw MatcherWasMovedOrCopied();
    }
    ThrowOnCopyOrMoveMatcher& operator=( ThrowOnCopyOrMoveMatcher const& ) {
        throw MatcherWasMovedOrCopied();
    }
    // NOLINTNEXTLINE(performance-noexcept-move-constructor)
    ThrowOnCopyOrMoveMatcher& operator=( ThrowOnCopyOrMoveMatcher&& ) {
        throw MatcherWasMovedOrCopied();
    }

    std::string describe() const override { return "always false"; }

    template <typename T> bool match( T&& ) const { return false; }
};

TEST_CASE( "Matchers are not moved or copied",
           "[matchers][templated][approvals]" ) {
    REQUIRE_NOTHROW(
        ( ThrowOnCopyOrMoveMatcher() && ThrowOnCopyOrMoveMatcher() ) ||
        !ThrowOnCopyOrMoveMatcher() );
}

TEST_CASE( "Immovable matchers can be used",
           "[matchers][templated][approvals]" ) {
    REQUIRE_THAT( 123,
                  ( ImmovableMatcher() && ImmovableMatcher() ) ||
                      !ImmovableMatcher() );
}

struct ReferencingMatcher : Catch::Matchers::MatcherGenericBase {
    std::string describe() const override { return "takes reference"; }
    bool match( int& i ) const { return i == 22; }
};

TEST_CASE( "Matchers can take references",
           "[matchers][templated][approvals]" ) {
    REQUIRE_THAT( 22, ReferencingMatcher{} );
}

#ifdef __clang__
#    pragma clang diagnostic pop
#endif

TEMPLATE_TEST_CASE(
    "#2152 - ULP checks between differently signed values were wrong",
    "[matchers][floating-point][ulp]",
    float,
    double ) {
    using Catch::Matchers::WithinULP;

    static constexpr TestType smallest_non_zero =
        std::numeric_limits<TestType>::denorm_min();

    CHECK_THAT( smallest_non_zero, WithinULP( -smallest_non_zero, 2 ) );
    CHECK_THAT( smallest_non_zero, !WithinULP( -smallest_non_zero, 1 ) );
}
