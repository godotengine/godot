
//              Copyright Catch2 Authors
// Distributed under the Boost Software License, Version 1.0.
//   (See accompanying file LICENSE.txt or copy at
//        https://www.boost.org/LICENSE_1_0.txt)

// SPDX-License-Identifier: BSL-1.0

#include <helpers/parse_test_spec.hpp>

#include <catch2/internal/catch_test_spec_parser.hpp>
#include <catch2/interfaces/catch_interfaces_tag_alias_registry.hpp>

namespace Catch {

    TestSpec parseTestSpec( std::string const& arg ) {
        return TestSpecParser( ITagAliasRegistry::get() )
            .parse( arg )
            .testSpec();
    }

} // namespace Catch
