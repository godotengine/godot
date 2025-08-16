
//              Copyright Catch2 Authors
// Distributed under the Boost Software License, Version 1.0.
//   (See accompanying file LICENSE.txt or copy at
//        https://www.boost.org/LICENSE_1_0.txt)

// SPDX-License-Identifier: BSL-1.0
#include <catch2/matchers/catch_matchers_quantifiers.hpp>

namespace Catch {
    namespace Matchers {
        std::string AllTrueMatcher::describe() const { return "contains only true"; }

        AllTrueMatcher AllTrue() { return AllTrueMatcher{}; }

        std::string NoneTrueMatcher::describe() const { return "contains no true"; }

        NoneTrueMatcher NoneTrue() { return NoneTrueMatcher{}; }

        std::string AnyTrueMatcher::describe() const { return "contains at least one true"; }

        AnyTrueMatcher AnyTrue() { return AnyTrueMatcher{}; }
    } // namespace Matchers
} // namespace Catch
