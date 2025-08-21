
//              Copyright Catch2 Authors
// Distributed under the Boost Software License, Version 1.0.
//   (See accompanying file LICENSE.txt or copy at
//        https://www.boost.org/LICENSE_1_0.txt)

// SPDX-License-Identifier: BSL-1.0

#include <catch2/generators/catch_generators_random.hpp>
#include <catch2/internal/catch_context.hpp>

#include <random>

namespace Catch {
    namespace Generators {
        namespace Detail {
            std::uint32_t getSeed() { return sharedRng()(); }
        } // namespace Detail

        struct RandomFloatingGenerator<long double>::PImpl {
            PImpl( long double a, long double b, uint32_t seed ):
                rng( seed ), dist( a, b ) {}

            Catch::SimplePcg32 rng;
            std::uniform_real_distribution<long double> dist;
        };

        RandomFloatingGenerator<long double>::RandomFloatingGenerator(
            long double a, long double b, std::uint32_t seed) :
            m_pimpl(Catch::Detail::make_unique<PImpl>(a, b, seed)) {
            static_cast<void>( next() );
        }

        RandomFloatingGenerator<long double>::~RandomFloatingGenerator() =
            default;
        bool RandomFloatingGenerator<long double>::next() {
            m_current_number = m_pimpl->dist( m_pimpl->rng );
            return true;
        }
    } // namespace Generators
} // namespace Catch
