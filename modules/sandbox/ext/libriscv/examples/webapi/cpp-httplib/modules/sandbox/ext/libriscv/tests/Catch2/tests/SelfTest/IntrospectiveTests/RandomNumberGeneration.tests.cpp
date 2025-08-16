
//              Copyright Catch2 Authors
// Distributed under the Boost Software License, Version 1.0.
//   (See accompanying file LICENSE.txt or copy at
//        https://www.boost.org/LICENSE_1_0.txt)

// SPDX-License-Identifier: BSL-1.0

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_template_test_macros.hpp>
#include <catch2/internal/catch_floating_point_helpers.hpp>
#include <catch2/internal/catch_random_integer_helpers.hpp>
#include <catch2/internal/catch_random_number_generator.hpp>
#include <catch2/internal/catch_random_seed_generation.hpp>
#include <catch2/internal/catch_uniform_floating_point_distribution.hpp>
#include <catch2/internal/catch_uniform_integer_distribution.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/matchers/catch_matchers_range_equals.hpp>

#include <random>

TEST_CASE("Our PCG implementation provides expected results for known seeds", "[rng]") {
    Catch::SimplePcg32 rng;
    SECTION("Default seeded") {
        REQUIRE(rng() == 0xfcdb943b);
        REQUIRE(rng() == 0x6f55b921);
        REQUIRE(rng() == 0x4c17a916);
        REQUIRE(rng() == 0x71eae25f);
        REQUIRE(rng() == 0x6ce7909c);
    }
    SECTION("Specific seed") {
        rng.seed(0xabcd1234);
        REQUIRE(rng() == 0x57c08495);
        REQUIRE(rng() == 0x33c956ac);
        REQUIRE(rng() == 0x2206fd76);
        REQUIRE(rng() == 0x3501a35b);
        REQUIRE(rng() == 0xfdffb30f);

        // Also check repeated output after reseeding
        rng.seed(0xabcd1234);
        REQUIRE(rng() == 0x57c08495);
        REQUIRE(rng() == 0x33c956ac);
        REQUIRE(rng() == 0x2206fd76);
        REQUIRE(rng() == 0x3501a35b);
        REQUIRE(rng() == 0xfdffb30f);
    }
}

TEST_CASE("Comparison ops", "[rng]") {
    using Catch::SimplePcg32;
    REQUIRE(SimplePcg32{} == SimplePcg32{});
    REQUIRE(SimplePcg32{ 0 } != SimplePcg32{});
    REQUIRE_FALSE(SimplePcg32{ 1 } == SimplePcg32{ 2 });
    REQUIRE_FALSE(SimplePcg32{ 1 } != SimplePcg32{ 1 });
}

TEST_CASE("Random seed generation reports unknown methods", "[rng][seed]") {
    REQUIRE_THROWS(Catch::generateRandomSeed(static_cast<Catch::GenerateFrom>(77)));
}

TEST_CASE("Random seed generation accepts known methods", "[rng][seed]") {
    using Catch::GenerateFrom;
    const auto method = GENERATE(
        GenerateFrom::Time,
        GenerateFrom::RandomDevice,
        GenerateFrom::Default
    );

    REQUIRE_NOTHROW(Catch::generateRandomSeed(method));
}

TEMPLATE_TEST_CASE("uniform_floating_point_distribution never returns infs from finite range",
          "[rng][distribution][floating-point][approvals]", float, double) {
    std::random_device rd{};
    Catch::SimplePcg32 pcg( rd() );
    Catch::uniform_floating_point_distribution<TestType> dist(
        -std::numeric_limits<TestType>::max(),
        std::numeric_limits<TestType>::max() );

    for (size_t i = 0; i < 10'000; ++i) {
        auto ret = dist( pcg );
        REQUIRE_FALSE( std::isinf( ret ) );
        REQUIRE_FALSE( std::isnan( ret ) );
    }
}

TEST_CASE( "fillBitsFrom - shortening and stretching", "[rng][approvals]" ) {
    using Catch::Detail::fillBitsFrom;

    // The seed is not important, but the numbers below have to be repeatable.
    // They should also exhibit the same general pattern of being prefixes
    Catch::SimplePcg32 pcg( 0xaabb'ccdd );

    SECTION( "Shorten to 8 bits" ) {
        // We cast the result to avoid dealing with char-like type in uint8_t
        auto shortened = static_cast<uint32_t>( fillBitsFrom<uint8_t>( pcg ) );
        REQUIRE( shortened == 0xcc );
    }
    SECTION( "Shorten to 16 bits" ) {
        auto shortened = fillBitsFrom<uint16_t>( pcg );
        REQUIRE( shortened == 0xccbe );
    }
    SECTION( "Keep at 32 bits" ) {
        auto n = fillBitsFrom<uint32_t>( pcg );
        REQUIRE( n == 0xccbe'5f04 );
    }
    SECTION( "Stretch to 64 bits" ) {
        auto stretched = fillBitsFrom<uint64_t>( pcg );
        REQUIRE( stretched == 0xccbe'5f04'a424'a486 );
    }
}

TEST_CASE("uniform_integer_distribution can return the bounds", "[rng][distribution]") {
    Catch::uniform_integer_distribution<int32_t> dist( -10, 10 );
    REQUIRE( dist.a() == -10 );
    REQUIRE( dist.b() == 10 );
}

namespace {
    template <typename T>
    static void CheckReturnValue(Catch::uniform_integer_distribution<T>& dist,
                                 Catch::SimplePcg32& rng,
                                 T target) {
        REQUIRE( dist.a() == dist.b() );
        for (int i = 0; i < 1'000; ++i) {
            REQUIRE( dist( rng ) == target );
        }
    }
}

TEMPLATE_TEST_CASE( "uniform_integer_distribution can handle unit ranges",
                    "[rng][distribution][approvals]",
                    unsigned char,
                    signed char,
                    char,
                    uint8_t,
                    int8_t,
                    uint16_t,
                    int16_t,
                    uint32_t,
                    int32_t,
                    uint64_t,
                    int64_t,
                    size_t,
                    ptrdiff_t) {
    // We want random seed to sample different parts of the rng state,
    // the output is predetermined anyway
    std::random_device rd;
    auto seed = rd();
    CAPTURE( seed );
    Catch::SimplePcg32 pcg( seed );

    // We check unitary ranges of 3 different values, min for type, max for type,
    // some value in between just to make sure
    SECTION("lowest value") {
        constexpr auto lowest = std::numeric_limits<TestType>::min();
        Catch::uniform_integer_distribution<TestType> dist( lowest, lowest );
        CheckReturnValue( dist, pcg, lowest );
    }
    SECTION( "highest value" ) {
        constexpr auto highest = std::numeric_limits<TestType>::max();
        Catch::uniform_integer_distribution<TestType> dist( highest, highest );
        CheckReturnValue( dist, pcg, highest );
    }
    SECTION( "some value" ) {
        constexpr auto some = TestType( 42 );
        Catch::uniform_integer_distribution<TestType> dist( some, some );
        CheckReturnValue( dist, pcg, some );
    }
}

// Bool needs its own test because it doesn't have a valid "third" value
TEST_CASE( "uniform_integer_distribution can handle boolean unit ranges",
           "[rng][distribution][approvals]" ) {
    // We want random seed to sample different parts of the rng state,
    // the output is predetermined anyway
    std::random_device rd;
    auto seed = rd();
    CAPTURE( seed );
    Catch::SimplePcg32 pcg( seed );

    // We check unitary ranges of 3 different values, min for type, max for
    // type, some value in between just to make sure
    SECTION( "lowest value" ) {
        Catch::uniform_integer_distribution<bool> dist( false, false );
        CheckReturnValue( dist, pcg, false );
    }
    SECTION( "highest value" ) {
        Catch::uniform_integer_distribution<bool> dist( true, true );
        CheckReturnValue( dist, pcg, true );
    }
}

TEMPLATE_TEST_CASE( "uniform_integer_distribution can handle full width ranges",
                    "[rng][distribution][approvals]",
                    unsigned char,
                    signed char,
                    char,
                    uint8_t,
                    int8_t,
                    uint16_t,
                    int16_t,
                    uint32_t,
                    int32_t,
                    uint64_t,
                    int64_t ) {
    // We want random seed to sample different parts of the rng state,
    // the output is predetermined anyway
    std::random_device rd;
    auto seed = rd();
    CAPTURE( seed );
    Catch::SimplePcg32 pcg( seed );

    constexpr auto lowest = std::numeric_limits<TestType>::min();
    constexpr auto highest = std::numeric_limits<TestType>::max();
    Catch::uniform_integer_distribution<TestType> dist( lowest, highest );
    STATIC_REQUIRE( std::is_same<TestType, decltype( dist( pcg ) )>::value );

    // We need to do bit operations on the results, so we will have to
    // cast them to unsigned type.
    using BitType = std::make_unsigned_t<TestType>;
    BitType ORs = 0;
    BitType ANDs = BitType(-1);
    for (int i = 0; i < 100; ++i) {
        auto bits = static_cast<BitType>( dist( pcg ) );
        ORs |= bits;
        ANDs &= bits;
    }
    // Assuming both our RNG and distribution are unbiased, asking for
    // the full range should essentially give us random bit generator.
    // Over long run, OR of all the generated values should have all
    // bits set to 1, while AND should have all bits set to 0.
    // The chance of this test failing for unbiased pipeline is
    // 1 / 2**iters, which for 100 iterations is astronomical.
    REQUIRE( ORs == BitType( -1 ) );
    REQUIRE( ANDs == 0 );
}

namespace {
    template <typename T>
    struct uniform_integer_test_params;

    template <>
    struct uniform_integer_test_params<bool> {
        static constexpr bool lowest = false;
        static constexpr bool highest = true;
        //  This seems weird, but it is an artifact of the specific seed
        static constexpr bool expected[] = { true,
                                             true,
                                             true,
                                             true,
                                             true,
                                             true,
                                             false,
                                             true,
                                             true,
                                             true,
                                             true,
                                             true,
                                             false,
                                             true,
                                             true };
    };

    template <>
    struct uniform_integer_test_params<char> {
        static constexpr char lowest = 32;
        static constexpr char highest = 126;
        static constexpr char expected[] = { 'k',
                                             '\\',
                                             'Z',
                                             'X',
                                             '`',
                                             'Q',
                                             ';',
                                             'o',
                                             ']',
                                             'T',
                                             'v',
                                             'p',
                                             ':',
                                             'S',
                                             't' };
    };

    template <>
    struct uniform_integer_test_params<uint8_t> {
        static constexpr uint8_t lowest = 3;
        static constexpr uint8_t highest = 123;
        static constexpr uint8_t expected[] = { 'c',
                                                'P',
                                                'M',
                                                'J',
                                                'U',
                                                'A',
                                                '%',
                                                'h',
                                                'Q',
                                                'F',
                                                'q',
                                                'i',
                                                '$',
                                                'E',
                                                'o' };
    };

    template <>
    struct uniform_integer_test_params<int8_t> {
        static constexpr int8_t lowest = -27;
        static constexpr int8_t highest = 73;
        static constexpr int8_t expected[] = { '5',
                                               '%',
                                               '#',
                                               ' ',
                                               '*',
                                               25,
                                               2,
                                               '9',
                                               '&',
                                               29,
                                               'A',
                                               ':',
                                               1,
                                               28,
                                               '?' };
    };

    template <>
    struct uniform_integer_test_params<uint16_t> {
        static constexpr uint16_t lowest = 123;
        static constexpr uint16_t highest = 33333;
        static constexpr uint16_t expected[] = { 26684,
                                                 21417,
                                                 20658,
                                                 19791,
                                                 22896,
                                                 17433,
                                                 9806,
                                                 27948,
                                                 21767,
                                                 18588,
                                                 30556,
                                                 28244,
                                                 9439,
                                                 18293,
                                                 29949 };
    };

    template <>
    struct uniform_integer_test_params<int16_t> {
        static constexpr int16_t lowest = -17222;
        static constexpr int16_t highest = 17222;
        static constexpr int16_t expected[] = { 10326,
                                                 4863,
                                                 4076,
                                                 3177,
                                                 6397,
                                                 731,
                                                 -7179,
                                                 11637,
                                                 5226,
                                                 1929,
                                                 14342,
                                                 11944,
                                                 -7560,
                                                 1623,
                                                 13712 };
    };

    template <>
    struct uniform_integer_test_params<uint32_t> {
        static constexpr uint32_t lowest = 17222;
        static constexpr uint32_t highest = 234234;
        static constexpr uint32_t expected[] = { 190784,
                                                 156367,
                                                 151409,
                                                 145743,
                                                 166032,
                                                 130337,
                                                 80501,
                                                 199046,
                                                 158654,
                                                 137883,
                                                 216091,
                                                 200981,
                                                 78099,
                                                 135954,
                                                 212120 };
    };

    template <>
    struct uniform_integer_test_params<int32_t> {
        static constexpr int32_t lowest = -237272;
        static constexpr int32_t highest = 234234;
        static constexpr int32_t expected[] = { 139829,
                                                65050,
                                                54278,
                                                41969,
                                                86051,
                                                8494,
                                                -99785,
                                                157781,
                                                70021,
                                                24890,
                                                194815,
                                                161985,
                                                -105004,
                                                20699,
                                                186186 };
    };

    template <>
    struct uniform_integer_test_params<uint64_t> {
        static constexpr uint64_t lowest = 1234;
        static constexpr uint64_t highest = 1234567890;
        static constexpr uint64_t expected[] = { 987382749,
                                                 763380386,
                                                 846572137,
                                                 359990258,
                                                 804599765,
                                                 1131353566,
                                                 346324913,
                                                 1108760730,
                                                 1141693933,
                                                 856999148,
                                                 879390623,
                                                 1149485521,
                                                 900556586,
                                                 952385958,
                                                 807916408 };
    };

    template <>
    struct uniform_integer_test_params<int64_t> {
        static constexpr int64_t lowest = -1234567890;
        static constexpr int64_t highest = 1234567890;
        static constexpr int64_t expected[] = { 740197113,
                                                292191940,
                                                458575608,
                                                -514589122,
                                                374630781,
                                                1028139036,
                                                -541919840,
                                                982953318,
                                                1048819790,
                                                479429651,
                                                524212647,
                                                1064402981,
                                                566544615,
                                                670203462,
                                                381264073 };
    };

    // We need these definitions for C++14 and earlier, but
    // GCC will complain about them in newer C++ standards
#if __cplusplus <= 201402L
    constexpr bool uniform_integer_test_params<bool>::expected[];
    constexpr char uniform_integer_test_params<char>::expected[];
    constexpr uint8_t uniform_integer_test_params<uint8_t>::expected[];
    constexpr int8_t uniform_integer_test_params<int8_t>::expected[];
    constexpr uint16_t uniform_integer_test_params<uint16_t>::expected[];
    constexpr int16_t uniform_integer_test_params<int16_t>::expected[];
    constexpr uint32_t uniform_integer_test_params<uint32_t>::expected[];
    constexpr int32_t uniform_integer_test_params<int32_t>::expected[];
    constexpr uint64_t uniform_integer_test_params<uint64_t>::expected[];
    constexpr int64_t uniform_integer_test_params<int64_t>::expected[];
#endif

}

TEMPLATE_TEST_CASE( "uniform_integer_distribution is reproducible",
                    "[rng][distribution][approvals]",
                   bool,
                   char,
                   uint8_t,
                   int8_t,
                   uint16_t,
                   int16_t,
                   uint32_t,
                   int32_t,
                   uint64_t,
                   int64_t) {
    Catch::SimplePcg32 pcg( 0xaabb'ccdd );

    constexpr auto lowest = uniform_integer_test_params<TestType>::lowest;
    constexpr auto highest = uniform_integer_test_params<TestType>::highest;
    Catch::uniform_integer_distribution<TestType> dist(lowest, highest);

    constexpr auto iters = 15;
    std::array<TestType, iters> generated;
    for (int i = 0; i < iters; ++i) {
        generated[i] = dist( pcg );
    }

    REQUIRE_THAT(generated, Catch::Matchers::RangeEquals(uniform_integer_test_params<TestType>::expected));
}

// The reproducibility tests assume that operations on `float`/`double`
// happen in the same precision as the operated-upon type. This is
// generally true, unless the code is compiled for 32 bit targets without
// SSE2 enabled, in which case the operations are done in the x87 FPU,
// which usually implies doing math in 80 bit floats, and then rounding
// into smaller type when the type is saved into memory. This obviously
// leads to a different answer, than doing the math in the correct precision.
#if ( defined( _MSC_VER ) && _M_IX86_FP < 2 ) ||              \
    ( defined( __GNUC__ ) &&                                  \
      ( ( defined( __i386__ ) || defined( __x86_64__ ) ) ) && \
      !defined( __SSE2_MATH__ ) )
#    define CATCH_TEST_CONFIG_DISABLE_FLOAT_REPRODUCIBILITY_TESTS
#endif

#if !defined( CATCH_TEST_CONFIG_DISABLE_FLOAT_REPRODUCIBILITY_TESTS )

namespace {
    template <typename T>
    struct uniform_fp_test_params;

    template<>
    struct uniform_fp_test_params<float> {
        // These are exactly representable
        static constexpr float lowest = -256.125f;
        static constexpr float highest = 385.125f;
        // These are just round-trip formatted
        static constexpr float expected[] = { 92.56961f,
                                              -23.170044f,
                                              310.81833f,
                                              -53.023132f,
                                              105.03287f,
                                              198.77591f,
                                              -172.72931f,
                                              51.805176f,
                                              -241.10156f,
                                              64.66101f,
                                              212.12509f,
                                              -49.24292f,
                                              -177.1399f,
                                              245.23679f,
                                              173.22421f };
    };
    template <>
    struct uniform_fp_test_params<double> {
        // These are exactly representable
        static constexpr double lowest = -234582.9921875;
        static constexpr double highest = 261238.015625;
        // These are just round-trip formatted
        static constexpr double expected[] = { 35031.207052832615,
                                               203783.3401838024,
                                               44667.940405848756,
                                               -170100.5877224467,
                                               -222966.7418051684,
                                               127472.72630072923,
                                               -173510.88209096913,
                                               97394.16172239158,
                                               119123.6921592663,
                                               22595.741022785165,
                                               8988.68409120926,
                                               136906.86520606978,
                                               33369.19104222473,
                                               60912.7615841752,
                                               -149060.05936760217 };
    };

// We need these definitions for C++14 and earlier, but
// GCC will complain about them in newer C++ standards
#if __cplusplus <= 201402L
    constexpr float uniform_fp_test_params<float>::expected[];
    constexpr double uniform_fp_test_params<double>::expected[];
#endif
} // namespace

TEMPLATE_TEST_CASE( "uniform_floating_point_distribution is reproducible",
                    "[rng][distribution][floating-point][approvals]",
                    float,
                    double ) {
    Catch::SimplePcg32 pcg( 0xaabb'aabb );

    const auto lowest = uniform_fp_test_params<TestType>::lowest;
    const auto highest = uniform_fp_test_params<TestType>::highest;
    Catch::uniform_floating_point_distribution<TestType> dist( lowest, highest );

    constexpr auto iters = 15;
    std::array<TestType, iters> generated;
    for ( int i = 0; i < iters; ++i ) {
        generated[i] = dist( pcg );
    }

    REQUIRE_THAT( generated, Catch::Matchers::RangeEquals( uniform_fp_test_params<TestType>::expected ) );
}

#endif // ^^ float reproducibility tests are enabled

TEMPLATE_TEST_CASE( "uniform_floating_point_distribution can handle unitary ranges",
                    "[rng][distribution][floating-point][approvals]",
                    float,
                    double ) {
    std::random_device rd;
    auto seed = rd();
    CAPTURE( seed );
    Catch::SimplePcg32 pcg( seed );

    const auto highest = TestType(385.125);
    Catch::uniform_floating_point_distribution<TestType> dist( highest,
                                                               highest );

    constexpr auto iters = 20;
    for (int i = 0; i < iters; ++i) {
        REQUIRE( Catch::Detail::directCompare( dist( pcg ), highest ) );
    }
}
