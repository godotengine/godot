
//              Copyright Catch2 Authors
// Distributed under the Boost Software License, Version 1.0.
//   (See accompanying file LICENSE.txt or copy at
//        https://www.boost.org/LICENSE_1_0.txt)

// SPDX-License-Identifier: BSL-1.0

#include <catch2/catch_test_macros.hpp>



#if defined(__GNUC__)
// This has to be left enabled until end of the TU, because the GCC
// frontend reports operator<<(std::ostream& os, const has_maker_and_operator&)
// as unused anyway
#    pragma GCC diagnostic ignored "-Wunused-function"
#endif

namespace {

struct has_operator { };
struct has_maker {};
struct has_maker_and_operator {};
struct has_neither {};
struct has_template_operator {};

std::ostream& operator<<(std::ostream& os, const has_operator&) {
    os << "operator<<( has_operator )";
    return os;
}

std::ostream& operator<<(std::ostream& os, const has_maker_and_operator&) {
    os << "operator<<( has_maker_and_operator )";
    return os;
}

template <typename StreamT>
StreamT& operator<<(StreamT& os, const has_template_operator&) {
    os << "operator<<( has_template_operator )";
    return os;
}

} // end anonymous namespace

namespace Catch {
    template<>
    struct StringMaker<has_maker> {
        static std::string convert( const has_maker& ) {
            return "StringMaker<has_maker>";
        }
    };
    template<>
    struct StringMaker<has_maker_and_operator> {
        static std::string convert( const has_maker_and_operator& ) {
            return "StringMaker<has_maker_and_operator>";
        }
    };
}

// Call the operator
TEST_CASE( "stringify( has_operator )", "[toString]" ) {
    has_operator item;
    REQUIRE( ::Catch::Detail::stringify( item ) == "operator<<( has_operator )" );
}

// Call the stringmaker
TEST_CASE( "stringify( has_maker )", "[toString]" ) {
    has_maker item;
    REQUIRE( ::Catch::Detail::stringify( item ) == "StringMaker<has_maker>" );
}

// Call the stringmaker
TEST_CASE( "stringify( has_maker_and_operator )", "[toString]" ) {
    has_maker_and_operator item;
    REQUIRE( ::Catch::Detail::stringify( item ) == "StringMaker<has_maker_and_operator>" );
}

TEST_CASE("stringify( has_neither )", "[toString]") {
    has_neither item;
    REQUIRE( ::Catch::Detail::stringify(item) == "{?}" );
}

// Call the templated operator
TEST_CASE( "stringify( has_template_operator )", "[toString]" ) {
    has_template_operator item;
    REQUIRE( ::Catch::Detail::stringify( item ) == "operator<<( has_template_operator )" );
}


// Vectors...

TEST_CASE( "stringify( vectors<has_operator> )", "[toString]" ) {
    std::vector<has_operator> v(1);
    REQUIRE( ::Catch::Detail::stringify( v ) == "{ operator<<( has_operator ) }" );
}

TEST_CASE( "stringify( vectors<has_maker> )", "[toString]" ) {
    std::vector<has_maker> v(1);
    REQUIRE( ::Catch::Detail::stringify( v ) == "{ StringMaker<has_maker> }" );
}

TEST_CASE( "stringify( vectors<has_maker_and_operator> )", "[toString]" ) {
    std::vector<has_maker_and_operator> v(1);
    REQUIRE( ::Catch::Detail::stringify( v ) == "{ StringMaker<has_maker_and_operator> }" );
}

namespace {

// Range-based conversion should only be used if other possibilities fail
struct int_iterator {
    using iterator_category = std::input_iterator_tag;
    using difference_type = std::ptrdiff_t;
    using value_type = int;
    using reference = int&;
    using pointer = int*;

    int_iterator() = default;
    int_iterator(int i) :val(i) {}

    value_type operator*() const { return val; }
    bool operator==(int_iterator rhs) const { return val == rhs.val; }
    bool operator!=(int_iterator rhs) const { return val != rhs.val; }
    int_iterator operator++() { ++val; return *this; }
    int_iterator operator++(int) {
        auto temp(*this);
        ++val;
        return temp;
    }
private:
    int val = 5;
};

struct streamable_range {
    int_iterator begin() const { return int_iterator{ 1 }; }
    int_iterator end() const { return {}; }
};

std::ostream& operator<<(std::ostream& os, const streamable_range&) {
    os << "op<<(streamable_range)";
    return os;
}

struct stringmaker_range {
    int_iterator begin() const { return int_iterator{ 1 }; }
    int_iterator end() const { return {}; }
};

} // end anonymous namespace

namespace Catch {
template <>
struct StringMaker<stringmaker_range> {
    static std::string convert(stringmaker_range const&) {
        return "stringmaker(streamable_range)";
    }
};
}

namespace {

struct just_range {
    int_iterator begin() const { return int_iterator{ 1 }; }
    int_iterator end() const { return {}; }
};

struct disabled_range {
    int_iterator begin() const { return int_iterator{ 1 }; }
    int_iterator end() const { return {}; }
};

} // end anonymous namespace

namespace Catch {
template <>
struct is_range<disabled_range> {
    static const bool value = false;
};
}

TEST_CASE("stringify ranges", "[toString]") {
    REQUIRE(::Catch::Detail::stringify(streamable_range{}) == "op<<(streamable_range)");
    REQUIRE(::Catch::Detail::stringify(stringmaker_range{}) == "stringmaker(streamable_range)");
    REQUIRE(::Catch::Detail::stringify(just_range{}) == "{ 1, 2, 3, 4 }");
    REQUIRE(::Catch::Detail::stringify(disabled_range{}) == "{?}");
}
