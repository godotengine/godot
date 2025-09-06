
//              Copyright Catch2 Authors
// Distributed under the Boost Software License, Version 1.0.
//   (See accompanying file LICENSE.txt or copy at
//        https://www.boost.org/LICENSE_1_0.txt)

// SPDX-License-Identifier: BSL-1.0
#ifndef CATCH_TOTALS_HPP_INCLUDED
#define CATCH_TOTALS_HPP_INCLUDED

#include <cstdint>

namespace Catch {

    struct Counts {
        Counts operator - ( Counts const& other ) const;
        Counts& operator += ( Counts const& other );

        std::uint64_t total() const;
        bool allPassed() const;
        bool allOk() const;

        std::uint64_t passed = 0;
        std::uint64_t failed = 0;
        std::uint64_t failedButOk = 0;
        std::uint64_t skipped = 0;
    };

    struct Totals {

        Totals operator - ( Totals const& other ) const;
        Totals& operator += ( Totals const& other );

        Totals delta( Totals const& prevTotals ) const;

        Counts assertions;
        Counts testCases;
    };
}

#endif // CATCH_TOTALS_HPP_INCLUDED
