
//              Copyright Catch2 Authors
// Distributed under the Boost Software License, Version 1.0.
//   (See accompanying file LICENSE.txt or copy at
//        https://www.boost.org/LICENSE_1_0.txt)

// SPDX-License-Identifier: BSL-1.0

#include <catch2/catch_test_case_info.hpp>
#include <catch2/internal/catch_test_case_info_hasher.hpp>

namespace Catch {
    TestCaseInfoHasher::TestCaseInfoHasher( hash_t seed ): m_seed( seed ) {}

    uint32_t TestCaseInfoHasher::operator()( TestCaseInfo const& t ) const {
        // FNV-1a hash algorithm that is designed for uniqueness:
        const hash_t prime = 1099511628211u;
        hash_t hash = 14695981039346656037u;
        for ( const char c : t.name ) {
            hash ^= c;
            hash *= prime;
        }
        for ( const char c : t.className ) {
            hash ^= c;
            hash *= prime;
        }
        for ( const Tag& tag : t.tags ) {
            for ( const char c : tag.original ) {
                hash ^= c;
                hash *= prime;
            }
        }
        hash ^= m_seed;
        hash *= prime;
        const uint32_t low{ static_cast<uint32_t>( hash ) };
        const uint32_t high{ static_cast<uint32_t>( hash >> 32 ) };
        return low * high;
    }
} // namespace Catch
