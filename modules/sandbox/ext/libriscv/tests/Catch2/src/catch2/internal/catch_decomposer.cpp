
//              Copyright Catch2 Authors
// Distributed under the Boost Software License, Version 1.0.
//   (See accompanying file LICENSE.txt or copy at
//        https://www.boost.org/LICENSE_1_0.txt)

// SPDX-License-Identifier: BSL-1.0

#include <catch2/internal/catch_decomposer.hpp>

namespace Catch {

    void ITransientExpression::streamReconstructedExpression(
        std::ostream& os ) const {
        // We can't make this function pure virtual to keep ITransientExpression
        // constexpr, so we write error message instead
        os << "Some class derived from ITransientExpression without overriding streamReconstructedExpression";
    }

    void formatReconstructedExpression( std::ostream &os, std::string const& lhs, StringRef op, std::string const& rhs ) {
        if( lhs.size() + rhs.size() < 40 &&
                lhs.find('\n') == std::string::npos &&
                rhs.find('\n') == std::string::npos )
            os << lhs << ' ' << op << ' ' << rhs;
        else
            os << lhs << '\n' << op << '\n' << rhs;
    }
}
