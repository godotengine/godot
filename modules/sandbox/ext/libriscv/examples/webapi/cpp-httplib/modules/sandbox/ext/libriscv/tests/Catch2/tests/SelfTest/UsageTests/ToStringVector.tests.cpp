
//              Copyright Catch2 Authors
// Distributed under the Boost Software License, Version 1.0.
//   (See accompanying file LICENSE.txt or copy at
//        https://www.boost.org/LICENSE_1_0.txt)

// SPDX-License-Identifier: BSL-1.0

#include <catch2/catch_test_macros.hpp>
#include <vector>
#include <array>

// vector
TEST_CASE( "vector<int> -> toString", "[toString][vector]" )
{
    std::vector<int> vv;
    REQUIRE( ::Catch::Detail::stringify(vv) == "{  }" );
    vv.push_back( 42 );
    REQUIRE( ::Catch::Detail::stringify(vv) == "{ 42 }" );
    vv.push_back( 250 );
    REQUIRE( ::Catch::Detail::stringify(vv) == "{ 42, 250 }" );
}

TEST_CASE( "vector<string> -> toString", "[toString][vector]" )
{
    std::vector<std::string> vv;
    REQUIRE( ::Catch::Detail::stringify(vv) == "{  }" );
    vv.emplace_back( "hello" );
    REQUIRE( ::Catch::Detail::stringify(vv) == "{ \"hello\" }" );
    vv.emplace_back( "world" );
    REQUIRE( ::Catch::Detail::stringify(vv) == "{ \"hello\", \"world\" }" );
}

namespace {
    /* Minimal Allocator */
    template<typename T>
    struct minimal_allocator {
        using value_type = T;
        using size_type = std::size_t;

        minimal_allocator() = default;
        template <typename U>
        minimal_allocator(const minimal_allocator<U>&) {}


        T *allocate( size_type n ) {
            return static_cast<T *>( ::operator new( n * sizeof(T) ) );
        }
        void deallocate( T *p, size_type /*n*/ ) {
            ::operator delete( static_cast<void *>(p) );
        }
        template<typename U>
        bool operator==( const minimal_allocator<U>& ) const { return true; }
        template<typename U>
        bool operator!=( const minimal_allocator<U>& ) const { return false; }
    };
}

TEST_CASE( "vector<int,allocator> -> toString", "[toString][vector,allocator]" ) {
    std::vector<int,minimal_allocator<int> > vv;
    REQUIRE( ::Catch::Detail::stringify(vv) == "{  }" );
    vv.push_back( 42 );
    REQUIRE( ::Catch::Detail::stringify(vv) == "{ 42 }" );
    vv.push_back( 250 );
    REQUIRE( ::Catch::Detail::stringify(vv) == "{ 42, 250 }" );
}

TEST_CASE( "vec<vec<string,alloc>> -> toString", "[toString][vector,allocator]" ) {
    using inner = std::vector<std::string, minimal_allocator<std::string>>;
    using vector = std::vector<inner>;
    vector v;
    REQUIRE( ::Catch::Detail::stringify(v) == "{  }" );
    v.push_back( inner { "hello" } );
    v.push_back( inner { "world" } );
    REQUIRE( ::Catch::Detail::stringify(v) == "{ { \"hello\" }, { \"world\" } }" );
}

// Based on PR by mat-so: https://github.com/catchorg/Catch2/pull/606/files#diff-43562f40f8c6dcfe2c54557316e0f852
TEST_CASE( "vector<bool> -> toString", "[toString][containers][vector]" ) {
    std::vector<bool> bools;
    REQUIRE( ::Catch::Detail::stringify(bools) == "{  }");
    bools.push_back(true);
    REQUIRE( ::Catch::Detail::stringify(bools) == "{ true }");
    bools.push_back(false);
    REQUIRE( ::Catch::Detail::stringify(bools) == "{ true, false }");
}
TEST_CASE( "array<int, N> -> toString", "[toString][containers][array]" ) {
    std::array<int, 0> empty;
    REQUIRE( Catch::Detail::stringify( empty ) == "{  }" );
    std::array<int, 1> oneValue = {{ 42 }};
    REQUIRE( Catch::Detail::stringify( oneValue ) == "{ 42 }" );
    std::array<int, 2> twoValues = {{ 42, 250 }};
    REQUIRE( Catch::Detail::stringify( twoValues ) == "{ 42, 250 }" );
}
