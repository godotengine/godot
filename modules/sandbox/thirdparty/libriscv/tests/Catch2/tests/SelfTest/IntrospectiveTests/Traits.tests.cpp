
//              Copyright Catch2 Authors
// Distributed under the Boost Software License, Version 1.0.
//   (See accompanying file LICENSE.txt or copy at
//        https://www.boost.org/LICENSE_1_0.txt)

// SPDX-License-Identifier: BSL-1.0

#include <catch2/catch_test_macros.hpp>
#include <catch2/internal/catch_compare_traits.hpp>
#include <helpers/type_with_lit_0_comparisons.hpp>


#define ADD_TRAIT_TEST_CASE( op )                                         \
    TEST_CASE( "is_" #op "_comparable",                                   \
               "[traits][is_comparable][approvals]" ) {                   \
        using Catch::Detail::is_##op##_0_comparable;                      \
        using Catch::Detail::is_##op##_comparable;                        \
                                                                          \
        STATIC_REQUIRE( is_##op##_comparable<int, int>::value );          \
        STATIC_REQUIRE(                                                   \
            is_##op##_comparable<std::string, std::string>::value );      \
        STATIC_REQUIRE( !is_##op##_comparable<int, std::string>::value ); \
        STATIC_REQUIRE(                                                   \
            !is_##op##_comparable<TypeWithLit0Comparisons, int>::value ); \
        STATIC_REQUIRE(                                                   \
            !is_##op##_comparable<int, TypeWithLit0Comparisons>::value ); \
                                                                          \
        STATIC_REQUIRE( is_##op##_0_comparable<int>::value );             \
        STATIC_REQUIRE(                                                   \
            is_##op##_0_comparable<TypeWithLit0Comparisons>::value );     \
        STATIC_REQUIRE( !is_##op##_0_comparable<std::string>::value );    \
                                                                          \
    /* This test fails with MSVC in permissive mode, because of course it does */ \
    /* STATIC_REQUIRE( !is_##op##_0_comparable<int*>::value ); */ \
}

ADD_TRAIT_TEST_CASE(lt)
ADD_TRAIT_TEST_CASE(gt)
ADD_TRAIT_TEST_CASE(le)
ADD_TRAIT_TEST_CASE(ge)
ADD_TRAIT_TEST_CASE(eq)
ADD_TRAIT_TEST_CASE(ne)

#undef ADD_TRAIT_TEST_CASE
