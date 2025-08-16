
//              Copyright Catch2 Authors
// Distributed under the Boost Software License, Version 1.0.
//   (See accompanying file LICENSE.txt or copy at
//        https://www.boost.org/LICENSE_1_0.txt)

// SPDX-License-Identifier: BSL-1.0

#ifndef CATCH_TEST_HELPERS_PARSE_TEST_SPEC_HPP_INCLUDED
#define CATCH_TEST_HELPERS_PARSE_TEST_SPEC_HPP_INCLUDED

#include <catch2/catch_test_spec.hpp>

#include <string>

namespace Catch {
    TestSpec parseTestSpec( std::string const& arg );
}

#endif // CATCH_TEST_HELPERS_PARSE_TEST_SPEC_HPP_INCLUDED
