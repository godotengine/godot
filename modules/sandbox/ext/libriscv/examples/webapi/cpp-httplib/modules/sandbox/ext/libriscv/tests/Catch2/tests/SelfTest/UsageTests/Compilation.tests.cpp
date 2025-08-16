
//              Copyright Catch2 Authors
// Distributed under the Boost Software License, Version 1.0.
//   (See accompanying file LICENSE.txt or copy at
//        https://www.boost.org/LICENSE_1_0.txt)

// SPDX-License-Identifier: BSL-1.0

#include <helpers/type_with_lit_0_comparisons.hpp>

#include <array>
#include <type_traits>

// Setup for #1403 -- look for global overloads of operator << for classes
// in a different namespace.
#include <ostream>

namespace foo {
    struct helper_1403 {
        bool operator==(helper_1403) const { return true; }
    };
}

namespace bar {
    template <typename... Ts>
    struct TypeList {};
}

#ifdef __GNUC__
#pragma GCC diagnostic ignored "-Wmissing-declarations"
#endif
static std::ostream& operator<<(std::ostream& out, foo::helper_1403 const&) {
    return out << "[1403 helper]";
}
///////////////////////////////

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators_range.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include <cstring>

// Comparison operators can return non-booleans.
// This is unusual, but should be supported.
struct logic_t {
    logic_t operator< (logic_t) const { return {}; }
    logic_t operator<=(logic_t) const { return {}; }
    logic_t operator> (logic_t) const { return {}; }
    logic_t operator>=(logic_t) const { return {}; }
    logic_t operator==(logic_t) const { return {}; }
    logic_t operator!=(logic_t) const { return {}; }
    explicit operator bool() const { return true; }
};


static void throws_int(bool b) {
    if (b) {
        throw 1;
    }
}

template<typename T>
bool templated_tests(T t) {
    int a = 3;
    REQUIRE(a == t);
    CHECK(a == t);
    REQUIRE_THROWS(throws_int(true));
    CHECK_THROWS_AS(throws_int(true), int);
    REQUIRE_NOTHROW(throws_int(false));
    REQUIRE_THAT("aaa", Catch::Matchers::EndsWith("aaa"));
    return true;
}

struct A {};

static std::ostream &operator<<(std::ostream &o, const A &) { return o << 0; }

struct B : private A {
    bool operator==(int) const { return true; }
};

#ifdef __clang__
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunused-function"
#endif
#ifdef __GNUC__
// Note that because -~GCC~-, this warning cannot be silenced temporarily, by pushing diagnostic stack...
// Luckily it is firing in test files and thus can be silenced for the whole file, without losing much.
#pragma GCC diagnostic ignored "-Wunused-function"
#endif

B f();

std::ostream g();

#ifdef __clang__
#pragma clang diagnostic pop
#endif

template <typename, typename>
struct Fixture_1245 {};

// This is a minimal example for an issue we have found in 1.7.0
struct dummy_809 {
    int i;
};

template<typename T>
bool operator==(const T& val, dummy_809 f) {
    return val == f.i;
}

TEST_CASE("#809") {
    dummy_809 f;
    f.i = 42;
    REQUIRE(42 == f);
}


// ------------------------------------------------------------------
// Changes to REQUIRE_THROWS_AS made it stop working in a template in
// an unfixable way (as long as C++03 compatibility is being kept).
// To prevent these from happening in the future, this needs to compile

    TEST_CASE("#833") {
        REQUIRE(templated_tests<int>(3));
    }


// Test containing example where original stream insertable check breaks compilation
TEST_CASE("#872") {
    A dummy;
    CAPTURE(dummy);
    B x;
    REQUIRE (x == 4);
}

TEST_CASE("#1027: Bitfields can be captured") {
    struct Y {
        uint32_t v : 1;
    };
    Y y{ 0 };
    REQUIRE(y.v == 0);
    REQUIRE(0 == y.v);
}

TEST_CASE( "#3001: Enum-based bitfields can be captured" ) {
    enum E {
        ZERO = 0,
        ONE = 1,
        TWO = 2,
    };

    struct BF {
        E e : 2;
    };

    BF bf{};
    bf.e = ONE;
    REQUIRE( bf.e == 1 );
    REQUIRE( 1 == bf.e );
}

// Comparison operators can return non-booleans.
// This is unusual, but should be supported.
TEST_CASE("#1147") {
    logic_t t1, t2;
    REQUIRE(t1 == t2);
    REQUIRE(t1 != t2);
    REQUIRE(t1 <  t2);
    REQUIRE(t1 >  t2);
    REQUIRE(t1 <= t2);
    REQUIRE(t1 >= t2);
}

// unsigned array
TEST_CASE("#1238") {
    unsigned char uarr[] = "123";
    CAPTURE(uarr);
    signed char sarr[] = "456";
    CAPTURE(sarr);

    REQUIRE(std::memcmp(uarr, "123", sizeof(uarr)) == 0);
    REQUIRE(std::memcmp(sarr, "456", sizeof(sarr)) == 0);
}

TEST_CASE_METHOD((Fixture_1245<int, int>), "#1245", "[compilation]") {
    SUCCEED();
}

TEST_CASE("#1403", "[compilation]") {
    ::foo::helper_1403 h1, h2;
    REQUIRE(h1 == h2);
}

TEST_CASE("Optionally static assertions", "[compilation]") {
    STATIC_REQUIRE( std::is_void<void>::value );
    STATIC_REQUIRE_FALSE( std::is_void<int>::value );
    STATIC_CHECK( std::is_void<void>::value );
    STATIC_CHECK_FALSE( std::is_void<int>::value );
}

TEST_CASE("#1548", "[compilation]") {
    using namespace bar;
    REQUIRE(std::is_same<TypeList<int>, TypeList<int>>::value);
}

    // #925
    using signal_t = void (*) (void*);

    struct TestClass {
        signal_t testMethod_uponComplete_arg = nullptr;
    };

    namespace utility {
        inline static void synchronizing_callback( void * ) { }
    }

#if defined (_MSC_VER)
#pragma warning(push)
// The function pointer comparison below triggers warning because of
// calling conventions
#pragma warning(disable:4244)
#endif
    TEST_CASE("#925: comparing function pointer to function address failed to compile", "[!nonportable]" ) {
        TestClass test;
        REQUIRE(utility::synchronizing_callback != test.testMethod_uponComplete_arg);
    }
#if defined (_MSC_VER)
#pragma warning(pop)
#endif

TEST_CASE( "#1319: Sections can have description (even if it is not saved",
               "[compilation]" ) {
    SECTION( "SectionName", "This is a long form section description" ) {
        SUCCEED();
    }
}

TEST_CASE("Lambdas in assertions") {
    REQUIRE([]() { return true; }());
}

namespace {
    struct HasBitOperators {
        int value;

        friend HasBitOperators operator| (HasBitOperators lhs, HasBitOperators rhs) {
            return { lhs.value | rhs.value };
        }
        friend HasBitOperators operator& (HasBitOperators lhs, HasBitOperators rhs) {
            return { lhs.value & rhs.value };
        }
        friend HasBitOperators operator^ (HasBitOperators lhs, HasBitOperators rhs) {
            return { lhs.value ^ rhs.value };
        }
        explicit operator bool() const {
            return !!value;
        }

        friend std::ostream& operator<<(std::ostream& out, HasBitOperators val) {
            out << "Val: " << val.value;
            return out;
        }
    };
}

TEST_CASE("Assertion macros support bit operators and bool conversions", "[compilation][bitops]") {
    HasBitOperators lhs{ 1 }, rhs{ 2 };
    REQUIRE(lhs | rhs);
    REQUIRE_FALSE(lhs & rhs);
    REQUIRE(HasBitOperators{ 1 } & HasBitOperators{ 1 });
    REQUIRE(lhs ^ rhs);
    REQUIRE_FALSE(lhs ^ lhs);
}

namespace {
    struct ImmovableType {
        ImmovableType() = default;

        ImmovableType(ImmovableType const&) = delete;
        ImmovableType& operator=(ImmovableType const&) = delete;
        ImmovableType(ImmovableType&&) = delete;
        ImmovableType& operator=(ImmovableType&&) = delete;

        friend bool operator==(ImmovableType const&, ImmovableType const&) {
            return true;
        }
    };
}

TEST_CASE("Immovable types are supported in basic assertions", "[compilation][.approvals]") {
    REQUIRE(ImmovableType{} == ImmovableType{});
}

namespace adl {

struct always_true {
    explicit operator bool() const { return true; }
};

#define COMPILATION_TEST_DEFINE_UNIVERSAL_OPERATOR(op) \
template <class T, class U> \
auto operator op (T&&, U&&) { \
    return always_true{}; \
}

COMPILATION_TEST_DEFINE_UNIVERSAL_OPERATOR(==)
COMPILATION_TEST_DEFINE_UNIVERSAL_OPERATOR(!=)
COMPILATION_TEST_DEFINE_UNIVERSAL_OPERATOR(<)
COMPILATION_TEST_DEFINE_UNIVERSAL_OPERATOR(>)
COMPILATION_TEST_DEFINE_UNIVERSAL_OPERATOR(<=)
COMPILATION_TEST_DEFINE_UNIVERSAL_OPERATOR(>=)
COMPILATION_TEST_DEFINE_UNIVERSAL_OPERATOR(|)
COMPILATION_TEST_DEFINE_UNIVERSAL_OPERATOR(&)
COMPILATION_TEST_DEFINE_UNIVERSAL_OPERATOR(^)

#undef COMPILATION_TEST_DEFINE_UNIVERSAL_OPERATOR

}

TEST_CASE("ADL universal operators don't hijack expression deconstruction", "[compilation][.approvals]") {
    REQUIRE(adl::always_true{});
    REQUIRE(0 == adl::always_true{});
    REQUIRE(0 != adl::always_true{});
    REQUIRE(0 < adl::always_true{});
    REQUIRE(0 > adl::always_true{});
    REQUIRE(0 <= adl::always_true{});
    REQUIRE(0 >= adl::always_true{});
    REQUIRE(0 | adl::always_true{});
    REQUIRE(0 & adl::always_true{});
    REQUIRE(0 ^ adl::always_true{});
}

TEST_CASE( "#2555 - types that can only be compared with 0 literal implemented as pointer conversion are supported",
           "[compilation][approvals]" ) {
    REQUIRE( TypeWithLit0Comparisons{} < 0 );
    REQUIRE_FALSE( 0 < TypeWithLit0Comparisons{} );
    REQUIRE( TypeWithLit0Comparisons{} <= 0 );
    REQUIRE_FALSE( 0 <= TypeWithLit0Comparisons{} );

    REQUIRE( TypeWithLit0Comparisons{} > 0 );
    REQUIRE_FALSE( 0 > TypeWithLit0Comparisons{} );
    REQUIRE( TypeWithLit0Comparisons{} >= 0 );
    REQUIRE_FALSE( 0 >= TypeWithLit0Comparisons{} );

    REQUIRE( TypeWithLit0Comparisons{} == 0 );
    REQUIRE_FALSE( 0 == TypeWithLit0Comparisons{} );
    REQUIRE( TypeWithLit0Comparisons{} != 0 );
    REQUIRE_FALSE( 0 != TypeWithLit0Comparisons{} );
}

// These tests require `consteval` to propagate through `constexpr` calls
// which is a late DR against C++20.
#if defined( CATCH_CPP20_OR_GREATER ) && defined( __cpp_consteval ) && \
    __cpp_consteval >= 202211L
// Can't have internal linkage to avoid warnings
void ZeroLiteralErrorFunc();
namespace {
    struct ZeroLiteralConsteval {
        template <class T, std::enable_if_t<std::is_same_v<T, int>, int> = 0>
        consteval ZeroLiteralConsteval( T zero ) noexcept {
            if ( zero != 0 ) { ZeroLiteralErrorFunc(); }
        }
    };

    // Should only be constructible from literal 0. Uses the propagating
    // consteval constructor trick (currently used by MSVC, might be used
    // by libc++ in the future as well).
    struct TypeWithConstevalLit0Comparison {
#    define DEFINE_COMP_OP( op )                                               \
        constexpr friend bool operator op( TypeWithConstevalLit0Comparison,    \
                                           ZeroLiteralConsteval ) {            \
            return true;                                                       \
        }                                                                      \
        constexpr friend bool operator op( ZeroLiteralConsteval,               \
                                           TypeWithConstevalLit0Comparison ) { \
            return false;                                                      \
        }                                                                      \
        /* std::orderings only have these for ==, but we add them for all      \
           operators so we can test all overloads for decomposer */            \
        constexpr friend bool operator op( TypeWithConstevalLit0Comparison,    \
                                           TypeWithConstevalLit0Comparison ) { \
            return true;                                                       \
        }

        DEFINE_COMP_OP( < )
        DEFINE_COMP_OP( <= )
        DEFINE_COMP_OP( > )
        DEFINE_COMP_OP( >= )
        DEFINE_COMP_OP( == )
        DEFINE_COMP_OP( != )

#undef DEFINE_COMP_OP
    };

} // namespace

namespace Catch {
    template <>
    struct capture_by_value<TypeWithConstevalLit0Comparison> : std::true_type {};
}

TEST_CASE( "#2555 - types that can only be compared with 0 literal implemented as consteval check are supported",
           "[compilation][approvals]" ) {
    REQUIRE( TypeWithConstevalLit0Comparison{} < 0 );
    REQUIRE_FALSE( 0 < TypeWithConstevalLit0Comparison{} );
    REQUIRE( TypeWithConstevalLit0Comparison{} <= 0 );
    REQUIRE_FALSE( 0 <= TypeWithConstevalLit0Comparison{} );

    REQUIRE( TypeWithConstevalLit0Comparison{} > 0 );
    REQUIRE_FALSE( 0 > TypeWithConstevalLit0Comparison{} );
    REQUIRE( TypeWithConstevalLit0Comparison{} >= 0 );
    REQUIRE_FALSE( 0 >= TypeWithConstevalLit0Comparison{} );

    REQUIRE( TypeWithConstevalLit0Comparison{} == 0 );
    REQUIRE_FALSE( 0 == TypeWithConstevalLit0Comparison{} );
    REQUIRE( TypeWithConstevalLit0Comparison{} != 0 );
    REQUIRE_FALSE( 0 != TypeWithConstevalLit0Comparison{} );
}

// We check all comparison ops to test, even though orderings, the primary
// motivation for this functionality, only have self-comparison (and thus
// have the ambiguity issue) for `==` and `!=`.
TEST_CASE( "Comparing const instances of type registered with capture_by_value",
           "[regression][approvals][compilation]" ) {
    SECTION("Type with consteval-int constructor") {
        auto const const_Lit0Type_1 = TypeWithConstevalLit0Comparison{};
        auto const const_Lit0Type_2 = TypeWithConstevalLit0Comparison{};
        REQUIRE( const_Lit0Type_1 == const_Lit0Type_2 );
        REQUIRE( const_Lit0Type_1 <= const_Lit0Type_2 );
        REQUIRE( const_Lit0Type_1 < const_Lit0Type_2 );
        REQUIRE( const_Lit0Type_1 >= const_Lit0Type_2 );
        REQUIRE( const_Lit0Type_1 > const_Lit0Type_2 );
        REQUIRE( const_Lit0Type_1 != const_Lit0Type_2 );
    }
    SECTION("Type with constexpr-int constructor") {
        auto const const_Lit0Type_1 = TypeWithLit0Comparisons{};
        auto const const_Lit0Type_2 = TypeWithLit0Comparisons{};
        REQUIRE( const_Lit0Type_1 == const_Lit0Type_2 );
        REQUIRE( const_Lit0Type_1 <= const_Lit0Type_2 );
        REQUIRE( const_Lit0Type_1 < const_Lit0Type_2 );
        REQUIRE( const_Lit0Type_1 >= const_Lit0Type_2 );
        REQUIRE( const_Lit0Type_1 > const_Lit0Type_2 );
        REQUIRE( const_Lit0Type_1 != const_Lit0Type_2 );
    }
}

#endif // C++20 consteval


namespace {
    struct MultipleImplicitConstructors {
        MultipleImplicitConstructors( double ) {}
        MultipleImplicitConstructors( int64_t ) {}
        bool operator==( MultipleImplicitConstructors ) const { return true; }
        bool operator!=( MultipleImplicitConstructors ) const { return true; }
        bool operator<( MultipleImplicitConstructors ) const { return true; }
        bool operator<=( MultipleImplicitConstructors ) const { return true; }
        bool operator>( MultipleImplicitConstructors ) const { return true; }
        bool operator>=( MultipleImplicitConstructors ) const { return true; }
    };
}
TEST_CASE("#2571 - tests compile types that have multiple implicit constructors from lit 0",
          "[compilation][approvals]") {
    MultipleImplicitConstructors mic1( 0.0 );
    MultipleImplicitConstructors mic2( 0.0 );
    REQUIRE( mic1 == mic2 );
    REQUIRE( mic1 != mic2 );
    REQUIRE( mic1 < mic2 );
    REQUIRE( mic1 <= mic2 );
    REQUIRE( mic1 > mic2 );
    REQUIRE( mic1 >= mic2 );
}

#if defined( CATCH_CONFIG_CPP20_COMPARE_OVERLOADS )
// This test does not test all the related codepaths, but it is the original
// reproducer
TEST_CASE( "Comparing const std::weak_ordering instances must compile",
           "[compilation][approvals][regression]" ) {
    auto const const_ordering_1 = std::weak_ordering::less;
    auto const const_ordering_2 = std::weak_ordering::less;
    auto plain_ordering_1 = std::weak_ordering::less;
    REQUIRE( const_ordering_1 == plain_ordering_1 );
    REQUIRE( const_ordering_1 == const_ordering_2 );
    REQUIRE( plain_ordering_1 == const_ordering_1 );
}
#endif

// Reproduce issue with yaml-cpp iterators, where the `const_iterator`
// for Node type has `const T` as the value_type. This is wrong for
// multitude of reasons, but there might be other libraries in the wild
// that share this issue, and the workaround needed to support
// `from_range(iter, iter)` helper with those libraries is easy enough.
class HasBadIterator {
    std::array<int, 10> m_arr{};

public:
    class iterator {
        const int* m_ptr = nullptr;

    public:
        iterator( const int* ptr ): m_ptr( ptr ) {}

        using difference_type = std::ptrdiff_t;
        using value_type = const int;
        using pointer = const int*;
        using reference = const int&;
        using iterator_category = std::input_iterator_tag;

        iterator& operator++() {
            ++m_ptr;
            return *this;
        }

        iterator operator++( int ) {
            auto ret( *this );
            ++( *this );
            return ret;
        }

        friend bool operator==( iterator lhs, iterator rhs ) {
            return lhs.m_ptr == rhs.m_ptr;
        }
        friend bool operator!=( iterator lhs, iterator rhs ) {
            return !( lhs == rhs );
        }

        int operator*() const { return *m_ptr; }
    };

    iterator cbegin() const { return { m_arr.data() }; }
    iterator cend() const { return { m_arr.data() + m_arr.size() }; }
};

TEST_CASE("from_range(iter, iter) supports const_iterators", "[generators][from-range][approvals]") {
    using namespace Catch::Generators;

    HasBadIterator data;
    auto gen = from_range(data.cbegin(), data.cend());
    (void)gen;
}
