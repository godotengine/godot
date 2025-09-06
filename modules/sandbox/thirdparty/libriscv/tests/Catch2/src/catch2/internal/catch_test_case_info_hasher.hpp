
//              Copyright Catch2 Authors
// Distributed under the Boost Software License, Version 1.0.
//   (See accompanying file LICENSE.txt or copy at
//        https://www.boost.org/LICENSE_1_0.txt)

// SPDX-License-Identifier: BSL-1.0
#ifndef CATCH_TEST_CASE_INFO_HASHER_HPP_INCLUDED
#define CATCH_TEST_CASE_INFO_HASHER_HPP_INCLUDED

#include <cstdint>

namespace Catch {

    struct TestCaseInfo;

    class TestCaseInfoHasher {
    public:
        using hash_t = std::uint64_t;
        TestCaseInfoHasher( hash_t seed );
        uint32_t operator()( TestCaseInfo const& t ) const;

    private:
        hash_t m_seed;
    };

} // namespace Catch

#endif /* CATCH_TEST_CASE_INFO_HASHER_HPP_INCLUDED */
