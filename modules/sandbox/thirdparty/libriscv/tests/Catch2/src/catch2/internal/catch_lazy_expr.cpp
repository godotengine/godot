
//              Copyright Catch2 Authors
// Distributed under the Boost Software License, Version 1.0.
//   (See accompanying file LICENSE.txt or copy at
//        https://www.boost.org/LICENSE_1_0.txt)

// SPDX-License-Identifier: BSL-1.0

#include <catch2/internal/catch_lazy_expr.hpp>
#include <catch2/internal/catch_decomposer.hpp>

namespace Catch {

    auto operator << (std::ostream& os, LazyExpression const& lazyExpr) -> std::ostream& {
        if (lazyExpr.m_isNegated)
            os << '!';

        if (lazyExpr) {
            if (lazyExpr.m_isNegated && lazyExpr.m_transientExpression->isBinaryExpression())
                os << '(' << *lazyExpr.m_transientExpression << ')';
            else
                os << *lazyExpr.m_transientExpression;
        } else {
            os << "{** error - unchecked empty expression requested **}";
        }
        return os;
    }

} // namespace Catch
