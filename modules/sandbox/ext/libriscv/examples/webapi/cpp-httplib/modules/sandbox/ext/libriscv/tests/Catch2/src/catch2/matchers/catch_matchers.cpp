
//              Copyright Catch2 Authors
// Distributed under the Boost Software License, Version 1.0.
//   (See accompanying file LICENSE.txt or copy at
//        https://www.boost.org/LICENSE_1_0.txt)

// SPDX-License-Identifier: BSL-1.0


#include <catch2/matchers/catch_matchers.hpp>

namespace Catch {
namespace Matchers {

    std::string MatcherUntypedBase::toString() const {
        if (m_cachedToString.empty()) {
            m_cachedToString = describe();
        }
        return m_cachedToString;
    }

    MatcherUntypedBase::~MatcherUntypedBase() = default;

} // namespace Matchers
} // namespace Catch
