
//              Copyright Catch2 Authors
// Distributed under the Boost Software License, Version 1.0.
//   (See accompanying file LICENSE.txt or copy at
//        https://www.boost.org/LICENSE_1_0.txt)

// SPDX-License-Identifier: BSL-1.0
#ifndef CATCH_PARSE_NUMBERS_HPP_INCLUDED
#define CATCH_PARSE_NUMBERS_HPP_INCLUDED

#include <catch2/internal/catch_optional.hpp>

#include <string>

namespace Catch {

    /**
     * Parses unsigned int from the input, using provided base
     *
     * Effectively a wrapper around std::stoul but with better error checking
     * e.g. "-1" is rejected, instead of being parsed as UINT_MAX.
     */
    Optional<unsigned int> parseUInt(std::string const& input, int base = 10);
}

#endif // CATCH_PARSE_NUMBERS_HPP_INCLUDED
