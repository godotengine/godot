
//              Copyright Catch2 Authors
// Distributed under the Boost Software License, Version 1.0.
//   (See accompanying file LICENSE.txt or copy at
//        https://www.boost.org/LICENSE_1_0.txt)

// SPDX-License-Identifier: BSL-1.0

#ifndef CATCH_RANDOM_INTEGER_HELPERS_HPP_INCLUDED
#define CATCH_RANDOM_INTEGER_HELPERS_HPP_INCLUDED

#include <climits>
#include <cstddef>
#include <cstdint>
#include <type_traits>

// Note: We use the usual enable-disable-autodetect dance here even though
//       we do not support these in CMake configuration options (yet?).
//       It is highly unlikely that we will need to make these actually
//       user-configurable, but this will make it simpler if weend up needing
//       it, and it provides an escape hatch to the users who need it.
#if defined( __SIZEOF_INT128__ )
#    define CATCH_CONFIG_INTERNAL_UINT128
// Unlike GCC, MSVC does not polyfill umul as mulh + mul pair on ARM machines.
// Currently we do not bother doing this ourselves, but we could if it became
// important for perf.
#elif defined( _MSC_VER ) && defined( _M_X64 )
#    define CATCH_CONFIG_INTERNAL_MSVC_UMUL128
#endif

#if defined( CATCH_CONFIG_INTERNAL_UINT128 ) && \
    !defined( CATCH_CONFIG_NO_UINT128 ) &&      \
    !defined( CATCH_CONFIG_UINT128 )
#define CATCH_CONFIG_UINT128
#endif

#if defined( CATCH_CONFIG_INTERNAL_MSVC_UMUL128 ) && \
    !defined( CATCH_CONFIG_NO_MSVC_UMUL128 ) &&      \
    !defined( CATCH_CONFIG_MSVC_UMUL128 )
#    define CATCH_CONFIG_MSVC_UMUL128
#    include <intrin.h>
#endif


namespace Catch {
    namespace Detail {

        template <std::size_t>
        struct SizedUnsignedType;
#define SizedUnsignedTypeHelper( TYPE )        \
    template <>                                \
    struct SizedUnsignedType<sizeof( TYPE )> { \
        using type = TYPE;                     \
    }

        SizedUnsignedTypeHelper( std::uint8_t );
        SizedUnsignedTypeHelper( std::uint16_t );
        SizedUnsignedTypeHelper( std::uint32_t );
        SizedUnsignedTypeHelper( std::uint64_t );
#undef SizedUnsignedTypeHelper

        template <std::size_t sz>
        using SizedUnsignedType_t = typename SizedUnsignedType<sz>::type;

        template <typename T>
        using DoubleWidthUnsignedType_t = SizedUnsignedType_t<2 * sizeof( T )>;

        template <typename T>
        struct ExtendedMultResult {
            T upper;
            T lower;
            constexpr bool operator==( ExtendedMultResult const& rhs ) const {
                return upper == rhs.upper && lower == rhs.lower;
            }
        };

        /**
         * Returns 128 bit result of lhs * rhs using portable C++ code
         *
         * This implementation is almost twice as fast as naive long multiplication,
         * and unlike intrinsic-based approach, it supports constexpr evaluation.
         */
        constexpr ExtendedMultResult<std::uint64_t>
        extendedMultPortable(std::uint64_t lhs, std::uint64_t rhs) {
#define CarryBits( x ) ( x >> 32 )
#define Digits( x ) ( x & 0xFF'FF'FF'FF )
            std::uint64_t lhs_low = Digits( lhs );
            std::uint64_t rhs_low = Digits( rhs );
            std::uint64_t low_low = ( lhs_low * rhs_low );
            std::uint64_t high_high = CarryBits( lhs ) * CarryBits( rhs );

            // We add in carry bits from low-low already
            std::uint64_t high_low =
                ( CarryBits( lhs ) * rhs_low ) + CarryBits( low_low );
            // Note that we can add only low bits from high_low, to avoid
            // overflow with large inputs
            std::uint64_t low_high =
                ( lhs_low * CarryBits( rhs ) ) + Digits( high_low );

            return { high_high + CarryBits( high_low ) + CarryBits( low_high ),
                     ( low_high << 32 ) | Digits( low_low ) };
#undef CarryBits
#undef Digits
        }

        //! Returns 128 bit result of lhs * rhs
        inline ExtendedMultResult<std::uint64_t>
        extendedMult( std::uint64_t lhs, std::uint64_t rhs ) {
#if defined( CATCH_CONFIG_UINT128 )
            auto result = __uint128_t( lhs ) * __uint128_t( rhs );
            return { static_cast<std::uint64_t>( result >> 64 ),
                     static_cast<std::uint64_t>( result ) };
#elif defined( CATCH_CONFIG_MSVC_UMUL128 )
            std::uint64_t high;
            std::uint64_t low = _umul128( lhs, rhs, &high );
            return { high, low };
#else
            return extendedMultPortable( lhs, rhs );
#endif
        }


        template <typename UInt>
        constexpr ExtendedMultResult<UInt> extendedMult( UInt lhs, UInt rhs ) {
            static_assert( std::is_unsigned<UInt>::value,
                           "extendedMult can only handle unsigned integers" );
            static_assert( sizeof( UInt ) < sizeof( std::uint64_t ),
                           "Generic extendedMult can only handle types smaller "
                           "than uint64_t" );
            using WideType = DoubleWidthUnsignedType_t<UInt>;

            auto result = WideType( lhs ) * WideType( rhs );
            return {
                static_cast<UInt>( result >> ( CHAR_BIT * sizeof( UInt ) ) ),
                static_cast<UInt>( result & UInt( -1 ) ) };
        }


        template <typename TargetType,
                  typename Generator>
            std::enable_if_t<sizeof(typename Generator::result_type) >= sizeof(TargetType),
            TargetType> fillBitsFrom(Generator& gen) {
            using gresult_type = typename Generator::result_type;
            static_assert( std::is_unsigned<TargetType>::value, "Only unsigned integers are supported" );
            static_assert( Generator::min() == 0 &&
                           Generator::max() == static_cast<gresult_type>( -1 ),
                           "Generator must be able to output all numbers in its result type (effectively it must be a random bit generator)" );

            // We want to return the top bits from a generator, as they are
            // usually considered higher quality.
            constexpr auto generated_bits = sizeof( gresult_type ) * CHAR_BIT;
            constexpr auto return_bits = sizeof( TargetType ) * CHAR_BIT;

            return static_cast<TargetType>( gen() >>
                                            ( generated_bits - return_bits) );
        }

        template <typename TargetType,
                  typename Generator>
            std::enable_if_t<sizeof(typename Generator::result_type) < sizeof(TargetType),
            TargetType> fillBitsFrom(Generator& gen) {
            using gresult_type = typename Generator::result_type;
            static_assert( std::is_unsigned<TargetType>::value,
                           "Only unsigned integers are supported" );
            static_assert( Generator::min() == 0 &&
                           Generator::max() == static_cast<gresult_type>( -1 ),
                           "Generator must be able to output all numbers in its result type (effectively it must be a random bit generator)" );

            constexpr auto generated_bits = sizeof( gresult_type ) * CHAR_BIT;
            constexpr auto return_bits = sizeof( TargetType ) * CHAR_BIT;
            std::size_t filled_bits = 0;
            TargetType ret = 0;
            do {
                ret <<= generated_bits;
                ret |= gen();
                filled_bits += generated_bits;
            } while ( filled_bits < return_bits );

            return ret;
        }

        /*
         * Transposes numbers into unsigned type while keeping their ordering
         *
         * This means that signed types are changed so that the ordering is
         * [INT_MIN, ..., -1, 0, ..., INT_MAX], rather than order we would
         * get by simple casting ([0, ..., INT_MAX, INT_MIN, ..., -1])
         */
        template <typename OriginalType, typename UnsignedType>
        constexpr
        std::enable_if_t<std::is_signed<OriginalType>::value, UnsignedType>
        transposeToNaturalOrder( UnsignedType in ) {
            static_assert(
                sizeof( OriginalType ) == sizeof( UnsignedType ),
                "reordering requires the same sized types on both sides" );
            static_assert( std::is_unsigned<UnsignedType>::value,
                           "Input type must be unsigned" );
            // Assuming 2s complement (standardized in current C++), the
            // positive and negative numbers are already internally ordered,
            // and their difference is in the top bit. Swapping it orders
            // them the desired way.
            constexpr auto highest_bit =
                UnsignedType( 1 ) << ( sizeof( UnsignedType ) * CHAR_BIT - 1 );
            return static_cast<UnsignedType>( in ^ highest_bit );
        }



        template <typename OriginalType,
                  typename UnsignedType>
        constexpr
        std::enable_if_t<std::is_unsigned<OriginalType>::value, UnsignedType>
            transposeToNaturalOrder(UnsignedType in) {
            static_assert(
                sizeof( OriginalType ) == sizeof( UnsignedType ),
                "reordering requires the same sized types on both sides" );
            static_assert( std::is_unsigned<UnsignedType>::value, "Input type must be unsigned" );
            // No reordering is needed for unsigned -> unsigned
            return in;
        }
    } // namespace Detail
} // namespace Catch

#endif // CATCH_RANDOM_INTEGER_HELPERS_HPP_INCLUDED
