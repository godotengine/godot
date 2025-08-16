
//              Copyright Catch2 Authors
// Distributed under the Boost Software License, Version 1.0.
//   (See accompanying file LICENSE.txt or copy at
//        https://www.boost.org/LICENSE_1_0.txt)

// SPDX-License-Identifier: BSL-1.0

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include <cmath>

using Catch::Approx;

namespace {
    static double divide(double a, double b) {
        return a / b;
    }

    class StrongDoubleTypedef {
        double d_ = 0.0;

    public:
        explicit StrongDoubleTypedef(double d) : d_(d) {}
        explicit operator double() const { return d_; }
    };

    static std::ostream& operator<<(std::ostream& os, StrongDoubleTypedef td) {
        return os << "StrongDoubleTypedef(" << static_cast<double>(td) << ")";
    }
} // end unnamed namespace

using namespace Catch::literals;

///////////////////////////////////////////////////////////////////////////////
TEST_CASE( "A comparison that uses literals instead of the normal constructor", "[Approx]" ) {
    double d = 1.23;

    REQUIRE( d == 1.23_a );
    REQUIRE( d != 1.22_a );
    REQUIRE( -d == -1.23_a );

    REQUIRE( d == 1.2_a .epsilon(.1) );
    REQUIRE( d != 1.2_a .epsilon(.001) );
    REQUIRE( d == 1_a .epsilon(.3) );
}

TEST_CASE( "Some simple comparisons between doubles", "[Approx]" ) {
    double d = 1.23;

    REQUIRE( d == Approx( 1.23 ) );
    REQUIRE( d != Approx( 1.22 ) );
    REQUIRE( d != Approx( 1.24 ) );

    REQUIRE( d == 1.23_a );
    REQUIRE( d != 1.22_a );

    REQUIRE( Approx( d ) == 1.23 );
    REQUIRE( Approx( d ) != 1.22 );
    REQUIRE( Approx( d ) != 1.24 );
}

///////////////////////////////////////////////////////////////////////////////
TEST_CASE( "Approximate comparisons with different epsilons", "[Approx]" ) {
    double d = 1.23;

    REQUIRE( d != Approx( 1.231 ) );
    REQUIRE( d == Approx( 1.231 ).epsilon( 0.1 ) );
}

///////////////////////////////////////////////////////////////////////////////
TEST_CASE( "Less-than inequalities with different epsilons", "[Approx]" ) {
  double d = 1.23;

  REQUIRE( d <= Approx( 1.24 ) );
  REQUIRE( d <= Approx( 1.23 ) );
  REQUIRE_FALSE( d <= Approx( 1.22 ) );
  REQUIRE( d <= Approx( 1.22 ).epsilon(0.1) );
}

///////////////////////////////////////////////////////////////////////////////
TEST_CASE( "Greater-than inequalities with different epsilons", "[Approx]" ) {
  double d = 1.23;

  REQUIRE( d >= Approx( 1.22 ) );
  REQUIRE( d >= Approx( 1.23 ) );
  REQUIRE_FALSE( d >= Approx( 1.24 ) );
  REQUIRE( d >= Approx( 1.24 ).epsilon(0.1) );
}

///////////////////////////////////////////////////////////////////////////////
TEST_CASE( "Approximate comparisons with floats", "[Approx]" ) {
    REQUIRE( 1.23f == Approx( 1.23f ) );
    REQUIRE( 0.0f == Approx( 0.0f ) );
}

///////////////////////////////////////////////////////////////////////////////
TEST_CASE( "Approximate comparisons with ints", "[Approx]" ) {
    REQUIRE( 1 == Approx( 1 ) );
    REQUIRE( 0 == Approx( 0 ) );
}

///////////////////////////////////////////////////////////////////////////////
TEST_CASE( "Approximate comparisons with mixed numeric types", "[Approx]" ) {
    const double dZero = 0;
    const double dSmall = 0.00001;
    const double dMedium = 1.234;

    REQUIRE( 1.0f == Approx( 1 ) );
    REQUIRE( 0 == Approx( dZero) );
    REQUIRE( 0 == Approx( dSmall ).margin( 0.001 ) );
    REQUIRE( 1.234f == Approx( dMedium ) );
    REQUIRE( dMedium == Approx( 1.234f ) );
}

///////////////////////////////////////////////////////////////////////////////
TEST_CASE( "Use a custom approx", "[Approx][custom]" ) {
    double d = 1.23;

    Approx approx = Approx::custom().epsilon( 0.01 );

    REQUIRE( d == approx( 1.23 ) );
    REQUIRE( d == approx( 1.22 ) );
    REQUIRE( d == approx( 1.24 ) );
    REQUIRE( d != approx( 1.25 ) );

    REQUIRE( approx( d ) == 1.23 );
    REQUIRE( approx( d ) == 1.22 );
    REQUIRE( approx( d ) == 1.24 );
    REQUIRE( approx( d ) != 1.25 );
}

TEST_CASE( "Approximate PI", "[Approx][PI]" ) {
    REQUIRE( divide( 22, 7 ) == Approx( 3.141 ).epsilon( 0.001 ) );
    REQUIRE( divide( 22, 7 ) != Approx( 3.141 ).epsilon( 0.0001 ) );
}

///////////////////////////////////////////////////////////////////////////////

TEST_CASE( "Absolute margin", "[Approx]" ) {
    REQUIRE( 104.0 != Approx(100.0) );
    REQUIRE( 104.0 == Approx(100.0).margin(5) );
    REQUIRE( 104.0 == Approx(100.0).margin(4) );
    REQUIRE( 104.0 != Approx(100.0).margin(3) );
    REQUIRE( 100.3 != Approx(100.0) );
    REQUIRE( 100.3 == Approx(100.0).margin(0.5) );
}

TEST_CASE("Approx with exactly-representable margin", "[Approx]") {
    CHECK( 0.25f == Approx(0.0f).margin(0.25f) );

    CHECK( 0.0f == Approx(0.25f).margin(0.25f) );
    CHECK( 0.5f == Approx(0.25f).margin(0.25f) );

    CHECK( 245.0f == Approx(245.25f).margin(0.25f) );
    CHECK( 245.5f == Approx(245.25f).margin(0.25f) );
}

TEST_CASE("Approx setters validate their arguments", "[Approx]") {
    REQUIRE_NOTHROW(Approx(0).margin(0));
    REQUIRE_NOTHROW(Approx(0).margin(1234656));

    REQUIRE_THROWS_AS(Approx(0).margin(-2), std::domain_error);

    REQUIRE_NOTHROW(Approx(0).epsilon(0));
    REQUIRE_NOTHROW(Approx(0).epsilon(1));

    REQUIRE_THROWS_AS(Approx(0).epsilon(-0.001), std::domain_error);
    REQUIRE_THROWS_AS(Approx(0).epsilon(1.0001), std::domain_error);
}

TEST_CASE("Default scale is invisible to comparison", "[Approx]") {
    REQUIRE(101.000001 != Approx(100).epsilon(0.01));
    REQUIRE(std::pow(10, -5) != Approx(std::pow(10, -7)));
}

TEST_CASE("Epsilon only applies to Approx's value", "[Approx]") {
    REQUIRE(101.01 != Approx(100).epsilon(0.01));
}

TEST_CASE("Assorted miscellaneous tests", "[Approx][approvals]") {
    REQUIRE(INFINITY == Approx(INFINITY));
    REQUIRE(-INFINITY != Approx(INFINITY));
    REQUIRE(1 != Approx(INFINITY));
    REQUIRE(INFINITY != Approx(1));
    REQUIRE(NAN != Approx(NAN));
    REQUIRE_FALSE(NAN == Approx(NAN));
}

TEST_CASE( "Comparison with explicitly convertible types", "[Approx]" )
{
  StrongDoubleTypedef td(10.0);

  REQUIRE(td == Approx(10.0));
  REQUIRE(Approx(10.0) == td);

  REQUIRE(td != Approx(11.0));
  REQUIRE(Approx(11.0) != td);

  REQUIRE(td <= Approx(10.0));
  REQUIRE(td <= Approx(11.0));
  REQUIRE(Approx(10.0) <= td);
  REQUIRE(Approx(9.0) <= td);

  REQUIRE(td >= Approx(9.0));
  REQUIRE(td >= Approx(td));
  REQUIRE(Approx(td) >= td);
  REQUIRE(Approx(11.0) >= td);

}

TEST_CASE("Approx::operator() is const correct", "[Approx][.approvals]") {
    const Approx ap = Approx(0.0).margin(0.01);

    // As long as this compiles, the test should be considered passing
    REQUIRE(1.0 == ap(1.0));
}
