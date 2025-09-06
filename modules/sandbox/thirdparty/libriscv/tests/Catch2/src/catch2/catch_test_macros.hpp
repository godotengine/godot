
//              Copyright Catch2 Authors
// Distributed under the Boost Software License, Version 1.0.
//   (See accompanying file LICENSE.txt or copy at
//        https://www.boost.org/LICENSE_1_0.txt)

// SPDX-License-Identifier: BSL-1.0
#ifndef CATCH_TEST_MACROS_HPP_INCLUDED
#define CATCH_TEST_MACROS_HPP_INCLUDED

#include <catch2/internal/catch_test_macro_impl.hpp>
#include <catch2/catch_message.hpp>
#include <catch2/catch_user_config.hpp>
#include <catch2/internal/catch_section.hpp>
#include <catch2/internal/catch_test_registry.hpp>
#include <catch2/internal/catch_unique_name.hpp>
#include <catch2/internal/catch_unreachable.hpp>


// All of our user-facing macros support configuration toggle, that
// forces them to be defined prefixed with CATCH_. We also like to
// support another toggle that can minimize (disable) their implementation.
// Given this, we have 4 different configuration options below

#if defined(CATCH_CONFIG_PREFIX_ALL) && !defined(CATCH_CONFIG_DISABLE)

  #define CATCH_REQUIRE( ... ) INTERNAL_CATCH_TEST( "CATCH_REQUIRE", Catch::ResultDisposition::Normal, __VA_ARGS__ )
  #define CATCH_REQUIRE_FALSE( ... ) INTERNAL_CATCH_TEST( "CATCH_REQUIRE_FALSE", Catch::ResultDisposition::Normal | Catch::ResultDisposition::FalseTest, __VA_ARGS__ )

  #define CATCH_REQUIRE_THROWS( ... ) INTERNAL_CATCH_THROWS( "CATCH_REQUIRE_THROWS", Catch::ResultDisposition::Normal, __VA_ARGS__ )
  #define CATCH_REQUIRE_THROWS_AS( expr, exceptionType ) INTERNAL_CATCH_THROWS_AS( "CATCH_REQUIRE_THROWS_AS", exceptionType, Catch::ResultDisposition::Normal, expr )
  #define CATCH_REQUIRE_NOTHROW( ... ) INTERNAL_CATCH_NO_THROW( "CATCH_REQUIRE_NOTHROW", Catch::ResultDisposition::Normal, __VA_ARGS__ )

  #define CATCH_CHECK( ... ) INTERNAL_CATCH_TEST( "CATCH_CHECK", Catch::ResultDisposition::ContinueOnFailure, __VA_ARGS__ )
  #define CATCH_CHECK_FALSE( ... ) INTERNAL_CATCH_TEST( "CATCH_CHECK_FALSE", Catch::ResultDisposition::ContinueOnFailure | Catch::ResultDisposition::FalseTest, __VA_ARGS__ )
  #define CATCH_CHECKED_IF( ... ) INTERNAL_CATCH_IF( "CATCH_CHECKED_IF", Catch::ResultDisposition::ContinueOnFailure | Catch::ResultDisposition::SuppressFail, __VA_ARGS__ )
  #define CATCH_CHECKED_ELSE( ... ) INTERNAL_CATCH_ELSE( "CATCH_CHECKED_ELSE", Catch::ResultDisposition::ContinueOnFailure | Catch::ResultDisposition::SuppressFail, __VA_ARGS__ )
  #define CATCH_CHECK_NOFAIL( ... ) INTERNAL_CATCH_TEST( "CATCH_CHECK_NOFAIL", Catch::ResultDisposition::ContinueOnFailure | Catch::ResultDisposition::SuppressFail, __VA_ARGS__ )

  #define CATCH_CHECK_THROWS( ... )  INTERNAL_CATCH_THROWS( "CATCH_CHECK_THROWS", Catch::ResultDisposition::ContinueOnFailure, __VA_ARGS__ )
  #define CATCH_CHECK_THROWS_AS( expr, exceptionType ) INTERNAL_CATCH_THROWS_AS( "CATCH_CHECK_THROWS_AS", exceptionType, Catch::ResultDisposition::ContinueOnFailure, expr )
  #define CATCH_CHECK_NOTHROW( ... ) INTERNAL_CATCH_NO_THROW( "CATCH_CHECK_NOTHROW", Catch::ResultDisposition::ContinueOnFailure, __VA_ARGS__ )

  #define CATCH_TEST_CASE( ... ) INTERNAL_CATCH_TESTCASE( __VA_ARGS__ )
  #define CATCH_TEST_CASE_METHOD( className, ... ) INTERNAL_CATCH_TEST_CASE_METHOD( className, __VA_ARGS__ )
  #define CATCH_METHOD_AS_TEST_CASE( method, ... ) INTERNAL_CATCH_METHOD_AS_TEST_CASE( method, __VA_ARGS__ )
  #define CATCH_TEST_CASE_PERSISTENT_FIXTURE( className, ... ) INTERNAL_CATCH_TEST_CASE_PERSISTENT_FIXTURE( className, __VA_ARGS__ )
  #define CATCH_REGISTER_TEST_CASE( Function, ... ) INTERNAL_CATCH_REGISTER_TESTCASE( Function, __VA_ARGS__ )
  #define CATCH_SECTION( ... ) INTERNAL_CATCH_SECTION( __VA_ARGS__ )
  #define CATCH_DYNAMIC_SECTION( ... ) INTERNAL_CATCH_DYNAMIC_SECTION( __VA_ARGS__ )
  #define CATCH_FAIL( ... ) do { \
           INTERNAL_CATCH_MSG("CATCH_FAIL", Catch::ResultWas::ExplicitFailure, Catch::ResultDisposition::Normal, __VA_ARGS__ );  \
           Catch::Detail::Unreachable(); \
         } while ( false )
  #define CATCH_FAIL_CHECK( ... ) INTERNAL_CATCH_MSG( "CATCH_FAIL_CHECK", Catch::ResultWas::ExplicitFailure, Catch::ResultDisposition::ContinueOnFailure, __VA_ARGS__ )
  #define CATCH_SUCCEED( ... ) INTERNAL_CATCH_MSG( "CATCH_SUCCEED", Catch::ResultWas::Ok, Catch::ResultDisposition::ContinueOnFailure, __VA_ARGS__ )
  #define CATCH_SKIP( ... ) do { \
           INTERNAL_CATCH_MSG( "CATCH_SKIP", Catch::ResultWas::ExplicitSkip, Catch::ResultDisposition::Normal, __VA_ARGS__ ); \
           Catch::Detail::Unreachable(); \
         } while (false)


  #if !defined(CATCH_CONFIG_RUNTIME_STATIC_REQUIRE)
    #define CATCH_STATIC_REQUIRE( ... )       static_assert(   __VA_ARGS__ ,      #__VA_ARGS__ );     CATCH_SUCCEED( #__VA_ARGS__ )
    #define CATCH_STATIC_REQUIRE_FALSE( ... ) static_assert( !(__VA_ARGS__), "!(" #__VA_ARGS__ ")" ); CATCH_SUCCEED( #__VA_ARGS__ )
    #define CATCH_STATIC_CHECK( ... )       static_assert(   __VA_ARGS__ ,      #__VA_ARGS__ );     CATCH_SUCCEED( #__VA_ARGS__ )
    #define CATCH_STATIC_CHECK_FALSE( ... ) static_assert( !(__VA_ARGS__), "!(" #__VA_ARGS__ ")" ); CATCH_SUCCEED( #__VA_ARGS__ )
  #else
    #define CATCH_STATIC_REQUIRE( ... )       CATCH_REQUIRE( __VA_ARGS__ )
    #define CATCH_STATIC_REQUIRE_FALSE( ... ) CATCH_REQUIRE_FALSE( __VA_ARGS__ )
    #define CATCH_STATIC_CHECK( ... )       CATCH_CHECK( __VA_ARGS__ )
    #define CATCH_STATIC_CHECK_FALSE( ... ) CATCH_CHECK_FALSE( __VA_ARGS__ )
  #endif


  // "BDD-style" convenience wrappers
  #define CATCH_SCENARIO( ... ) CATCH_TEST_CASE( "Scenario: " __VA_ARGS__ )
  #define CATCH_SCENARIO_METHOD( className, ... ) INTERNAL_CATCH_TEST_CASE_METHOD( className, "Scenario: " __VA_ARGS__ )
  #define CATCH_GIVEN( desc )     INTERNAL_CATCH_DYNAMIC_SECTION( "    Given: " << desc )
  #define CATCH_AND_GIVEN( desc ) INTERNAL_CATCH_DYNAMIC_SECTION( "And given: " << desc )
  #define CATCH_WHEN( desc )      INTERNAL_CATCH_DYNAMIC_SECTION( "     When: " << desc )
  #define CATCH_AND_WHEN( desc )  INTERNAL_CATCH_DYNAMIC_SECTION( " And when: " << desc )
  #define CATCH_THEN( desc )      INTERNAL_CATCH_DYNAMIC_SECTION( "     Then: " << desc )
  #define CATCH_AND_THEN( desc )  INTERNAL_CATCH_DYNAMIC_SECTION( "      And: " << desc )

#elif defined(CATCH_CONFIG_PREFIX_ALL) && defined(CATCH_CONFIG_DISABLE) // ^^ prefixed, implemented | vv prefixed, disabled

  #define CATCH_REQUIRE( ... )        (void)(0)
  #define CATCH_REQUIRE_FALSE( ... )  (void)(0)

  #define CATCH_REQUIRE_THROWS( ... ) (void)(0)
  #define CATCH_REQUIRE_THROWS_AS( expr, exceptionType ) (void)(0)
  #define CATCH_REQUIRE_NOTHROW( ... ) (void)(0)

  #define CATCH_CHECK( ... )         (void)(0)
  #define CATCH_CHECK_FALSE( ... )   (void)(0)
  #define CATCH_CHECKED_IF( ... )    if (__VA_ARGS__)
  #define CATCH_CHECKED_ELSE( ... )  if (!(__VA_ARGS__))
  #define CATCH_CHECK_NOFAIL( ... )  (void)(0)

  #define CATCH_CHECK_THROWS( ... )  (void)(0)
  #define CATCH_CHECK_THROWS_AS( expr, exceptionType ) (void)(0)
  #define CATCH_CHECK_NOTHROW( ... ) (void)(0)

  #define CATCH_TEST_CASE( ... ) INTERNAL_CATCH_TESTCASE_NO_REGISTRATION(INTERNAL_CATCH_UNIQUE_NAME( CATCH2_INTERNAL_TEST_ ))
  #define CATCH_TEST_CASE_METHOD( className, ... ) INTERNAL_CATCH_TESTCASE_NO_REGISTRATION(INTERNAL_CATCH_UNIQUE_NAME( CATCH2_INTERNAL_TEST_ ))
  #define CATCH_METHOD_AS_TEST_CASE( method, ... )
  #define CATCH_TEST_CASE_PERSISTENT_FIXTURE( className, ... ) INTERNAL_CATCH_TESTCASE_NO_REGISTRATION(INTERNAL_CATCH_UNIQUE_NAME( CATCH2_INTERNAL_TEST_ ))
  #define CATCH_REGISTER_TEST_CASE( Function, ... ) (void)(0)
  #define CATCH_SECTION( ... )
  #define CATCH_DYNAMIC_SECTION( ... )
  #define CATCH_FAIL( ... ) (void)(0)
  #define CATCH_FAIL_CHECK( ... ) (void)(0)
  #define CATCH_SUCCEED( ... ) (void)(0)
  #define CATCH_SKIP( ... ) (void)(0)

  #define CATCH_STATIC_REQUIRE( ... )       (void)(0)
  #define CATCH_STATIC_REQUIRE_FALSE( ... ) (void)(0)
  #define CATCH_STATIC_CHECK( ... )       (void)(0)
  #define CATCH_STATIC_CHECK_FALSE( ... ) (void)(0)

  // "BDD-style" convenience wrappers
  #define CATCH_SCENARIO( ... ) INTERNAL_CATCH_TESTCASE_NO_REGISTRATION(INTERNAL_CATCH_UNIQUE_NAME( CATCH2_INTERNAL_TEST_ ))
  #define CATCH_SCENARIO_METHOD( className, ... ) INTERNAL_CATCH_TESTCASE_METHOD_NO_REGISTRATION(INTERNAL_CATCH_UNIQUE_NAME( CATCH2_INTERNAL_TEST_ ), className )
  #define CATCH_GIVEN( desc )
  #define CATCH_AND_GIVEN( desc )
  #define CATCH_WHEN( desc )
  #define CATCH_AND_WHEN( desc )
  #define CATCH_THEN( desc )
  #define CATCH_AND_THEN( desc )

#elif !defined(CATCH_CONFIG_PREFIX_ALL) && !defined(CATCH_CONFIG_DISABLE) // ^^ prefixed, disabled | vv unprefixed, implemented

  #define REQUIRE( ... ) INTERNAL_CATCH_TEST( "REQUIRE", Catch::ResultDisposition::Normal, __VA_ARGS__  )
  #define REQUIRE_FALSE( ... ) INTERNAL_CATCH_TEST( "REQUIRE_FALSE", Catch::ResultDisposition::Normal | Catch::ResultDisposition::FalseTest, __VA_ARGS__ )

  #define REQUIRE_THROWS( ... ) INTERNAL_CATCH_THROWS( "REQUIRE_THROWS", Catch::ResultDisposition::Normal, __VA_ARGS__ )
  #define REQUIRE_THROWS_AS( expr, exceptionType ) INTERNAL_CATCH_THROWS_AS( "REQUIRE_THROWS_AS", exceptionType, Catch::ResultDisposition::Normal, expr )
  #define REQUIRE_NOTHROW( ... ) INTERNAL_CATCH_NO_THROW( "REQUIRE_NOTHROW", Catch::ResultDisposition::Normal, __VA_ARGS__ )

  #define CHECK( ... ) INTERNAL_CATCH_TEST( "CHECK", Catch::ResultDisposition::ContinueOnFailure, __VA_ARGS__ )
  #define CHECK_FALSE( ... ) INTERNAL_CATCH_TEST( "CHECK_FALSE", Catch::ResultDisposition::ContinueOnFailure | Catch::ResultDisposition::FalseTest, __VA_ARGS__ )
  #define CHECKED_IF( ... ) INTERNAL_CATCH_IF( "CHECKED_IF", Catch::ResultDisposition::ContinueOnFailure | Catch::ResultDisposition::SuppressFail, __VA_ARGS__ )
  #define CHECKED_ELSE( ... ) INTERNAL_CATCH_ELSE( "CHECKED_ELSE", Catch::ResultDisposition::ContinueOnFailure | Catch::ResultDisposition::SuppressFail, __VA_ARGS__ )
  #define CHECK_NOFAIL( ... ) INTERNAL_CATCH_TEST( "CHECK_NOFAIL", Catch::ResultDisposition::ContinueOnFailure | Catch::ResultDisposition::SuppressFail, __VA_ARGS__ )

  #define CHECK_THROWS( ... )  INTERNAL_CATCH_THROWS( "CHECK_THROWS", Catch::ResultDisposition::ContinueOnFailure, __VA_ARGS__ )
  #define CHECK_THROWS_AS( expr, exceptionType ) INTERNAL_CATCH_THROWS_AS( "CHECK_THROWS_AS", exceptionType, Catch::ResultDisposition::ContinueOnFailure, expr )
  #define CHECK_NOTHROW( ... ) INTERNAL_CATCH_NO_THROW( "CHECK_NOTHROW", Catch::ResultDisposition::ContinueOnFailure, __VA_ARGS__ )

  #define TEST_CASE( ... ) INTERNAL_CATCH_TESTCASE( __VA_ARGS__ )
  #define TEST_CASE_METHOD( className, ... ) INTERNAL_CATCH_TEST_CASE_METHOD( className, __VA_ARGS__ )
  #define METHOD_AS_TEST_CASE( method, ... ) INTERNAL_CATCH_METHOD_AS_TEST_CASE( method, __VA_ARGS__ )
  #define TEST_CASE_PERSISTENT_FIXTURE( className, ... ) INTERNAL_CATCH_TEST_CASE_PERSISTENT_FIXTURE( className, __VA_ARGS__ )
  #define REGISTER_TEST_CASE( Function, ... ) INTERNAL_CATCH_REGISTER_TESTCASE( Function, __VA_ARGS__ )
  #define SECTION( ... ) INTERNAL_CATCH_SECTION( __VA_ARGS__ )
  #define DYNAMIC_SECTION( ... ) INTERNAL_CATCH_DYNAMIC_SECTION( __VA_ARGS__ )
  #define FAIL( ... ) do { \
           INTERNAL_CATCH_MSG( "FAIL", Catch::ResultWas::ExplicitFailure, Catch::ResultDisposition::Normal, __VA_ARGS__ ); \
           Catch::Detail::Unreachable(); \
         } while (false)
  #define FAIL_CHECK( ... ) INTERNAL_CATCH_MSG( "FAIL_CHECK", Catch::ResultWas::ExplicitFailure, Catch::ResultDisposition::ContinueOnFailure, __VA_ARGS__ )
  #define SUCCEED( ... ) INTERNAL_CATCH_MSG( "SUCCEED", Catch::ResultWas::Ok, Catch::ResultDisposition::ContinueOnFailure, __VA_ARGS__ )
  #define SKIP( ... ) do { \
           INTERNAL_CATCH_MSG( "SKIP", Catch::ResultWas::ExplicitSkip, Catch::ResultDisposition::Normal, __VA_ARGS__ ); \
           Catch::Detail::Unreachable(); \
         } while (false)


  #if !defined(CATCH_CONFIG_RUNTIME_STATIC_REQUIRE)
    #define STATIC_REQUIRE( ... )       static_assert(   __VA_ARGS__,  #__VA_ARGS__ ); SUCCEED( #__VA_ARGS__ )
    #define STATIC_REQUIRE_FALSE( ... ) static_assert( !(__VA_ARGS__), "!(" #__VA_ARGS__ ")" ); SUCCEED( "!(" #__VA_ARGS__ ")" )
    #define STATIC_CHECK( ... )       static_assert(   __VA_ARGS__,  #__VA_ARGS__ ); SUCCEED( #__VA_ARGS__ )
    #define STATIC_CHECK_FALSE( ... ) static_assert( !(__VA_ARGS__), "!(" #__VA_ARGS__ ")" ); SUCCEED( "!(" #__VA_ARGS__ ")" )
  #else
    #define STATIC_REQUIRE( ... )       REQUIRE( __VA_ARGS__ )
    #define STATIC_REQUIRE_FALSE( ... ) REQUIRE_FALSE( __VA_ARGS__ )
    #define STATIC_CHECK( ... )       CHECK( __VA_ARGS__ )
    #define STATIC_CHECK_FALSE( ... ) CHECK_FALSE( __VA_ARGS__ )
  #endif

  // "BDD-style" convenience wrappers
  #define SCENARIO( ... ) TEST_CASE( "Scenario: " __VA_ARGS__ )
  #define SCENARIO_METHOD( className, ... ) INTERNAL_CATCH_TEST_CASE_METHOD( className, "Scenario: " __VA_ARGS__ )
  #define GIVEN( desc )     INTERNAL_CATCH_DYNAMIC_SECTION( "    Given: " << desc )
  #define AND_GIVEN( desc ) INTERNAL_CATCH_DYNAMIC_SECTION( "And given: " << desc )
  #define WHEN( desc )      INTERNAL_CATCH_DYNAMIC_SECTION( "     When: " << desc )
  #define AND_WHEN( desc )  INTERNAL_CATCH_DYNAMIC_SECTION( " And when: " << desc )
  #define THEN( desc )      INTERNAL_CATCH_DYNAMIC_SECTION( "     Then: " << desc )
  #define AND_THEN( desc )  INTERNAL_CATCH_DYNAMIC_SECTION( "      And: " << desc )

#elif !defined(CATCH_CONFIG_PREFIX_ALL) && defined(CATCH_CONFIG_DISABLE) // ^^ unprefixed, implemented | vv unprefixed, disabled

  #define REQUIRE( ... )       (void)(0)
  #define REQUIRE_FALSE( ... ) (void)(0)

  #define REQUIRE_THROWS( ... ) (void)(0)
  #define REQUIRE_THROWS_AS( expr, exceptionType ) (void)(0)
  #define REQUIRE_NOTHROW( ... ) (void)(0)

  #define CHECK( ... ) (void)(0)
  #define CHECK_FALSE( ... ) (void)(0)
  #define CHECKED_IF( ... ) if (__VA_ARGS__)
  #define CHECKED_ELSE( ... ) if (!(__VA_ARGS__))
  #define CHECK_NOFAIL( ... ) (void)(0)

  #define CHECK_THROWS( ... )  (void)(0)
  #define CHECK_THROWS_AS( expr, exceptionType ) (void)(0)
  #define CHECK_NOTHROW( ... ) (void)(0)

  #define TEST_CASE( ... )  INTERNAL_CATCH_TESTCASE_NO_REGISTRATION(INTERNAL_CATCH_UNIQUE_NAME( CATCH2_INTERNAL_TEST_ ), __VA_ARGS__)
  #define TEST_CASE_METHOD( className, ... ) INTERNAL_CATCH_TESTCASE_NO_REGISTRATION(INTERNAL_CATCH_UNIQUE_NAME( CATCH2_INTERNAL_TEST_ ))
  #define METHOD_AS_TEST_CASE( method, ... )
  #define TEST_CASE_PERSISTENT_FIXTURE( className, ... ) INTERNAL_CATCH_TESTCASE_NO_REGISTRATION(INTERNAL_CATCH_UNIQUE_NAME( CATCH2_INTERNAL_TEST_ ), __VA_ARGS__)
  #define REGISTER_TEST_CASE( Function, ... ) (void)(0)
  #define SECTION( ... )
  #define DYNAMIC_SECTION( ... )
  #define FAIL( ... ) (void)(0)
  #define FAIL_CHECK( ... ) (void)(0)
  #define SUCCEED( ... ) (void)(0)
  #define SKIP( ... ) (void)(0)

  #define STATIC_REQUIRE( ... )       (void)(0)
  #define STATIC_REQUIRE_FALSE( ... ) (void)(0)
  #define STATIC_CHECK( ... )       (void)(0)
  #define STATIC_CHECK_FALSE( ... ) (void)(0)

  // "BDD-style" convenience wrappers
  #define SCENARIO( ... ) INTERNAL_CATCH_TESTCASE_NO_REGISTRATION(INTERNAL_CATCH_UNIQUE_NAME( CATCH2_INTERNAL_TEST_ ) )
  #define SCENARIO_METHOD( className, ... ) INTERNAL_CATCH_TESTCASE_METHOD_NO_REGISTRATION(INTERNAL_CATCH_UNIQUE_NAME( CATCH2_INTERNAL_TEST_ ), className )

  #define GIVEN( desc )
  #define AND_GIVEN( desc )
  #define WHEN( desc )
  #define AND_WHEN( desc )
  #define THEN( desc )
  #define AND_THEN( desc )

#endif // ^^ unprefixed, disabled

// end of user facing macros

#endif // CATCH_TEST_MACROS_HPP_INCLUDED
