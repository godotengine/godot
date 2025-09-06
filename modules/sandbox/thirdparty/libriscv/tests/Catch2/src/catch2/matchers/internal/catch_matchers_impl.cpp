
//              Copyright Catch2 Authors
// Distributed under the Boost Software License, Version 1.0.
//   (See accompanying file LICENSE.txt or copy at
//        https://www.boost.org/LICENSE_1_0.txt)

// SPDX-License-Identifier: BSL-1.0

#include <catch2/matchers/internal/catch_matchers_impl.hpp>
#include <catch2/matchers/catch_matchers.hpp>
#include <catch2/interfaces/catch_interfaces_registry_hub.hpp>
#include <catch2/internal/catch_move_and_forward.hpp>

namespace Catch {

    // This is the general overload that takes a any string matcher
    // There is another overload, in catch_assertionhandler.h/.cpp, that only takes a string and infers
    // the Equals matcher (so the header does not mention matchers)
    void handleExceptionMatchExpr( AssertionHandler& handler, StringMatcher const& matcher ) {
        std::string exceptionMessage = Catch::translateActiveException();
        MatchExpr<std::string, StringMatcher const&> expr( CATCH_MOVE(exceptionMessage), matcher );
        handler.handleExpr( expr );
    }

} // namespace Catch
