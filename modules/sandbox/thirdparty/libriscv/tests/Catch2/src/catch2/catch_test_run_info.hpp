
//              Copyright Catch2 Authors
// Distributed under the Boost Software License, Version 1.0.
//   (See accompanying file LICENSE.txt or copy at
//        https://www.boost.org/LICENSE_1_0.txt)

// SPDX-License-Identifier: BSL-1.0
#ifndef CATCH_TEST_RUN_INFO_HPP_INCLUDED
#define CATCH_TEST_RUN_INFO_HPP_INCLUDED

#include <catch2/internal/catch_stringref.hpp>

namespace Catch {

    struct TestRunInfo {
        constexpr TestRunInfo(StringRef _name) : name(_name) {}
        StringRef name;
    };

} // end namespace Catch

#endif // CATCH_TEST_RUN_INFO_HPP_INCLUDED
