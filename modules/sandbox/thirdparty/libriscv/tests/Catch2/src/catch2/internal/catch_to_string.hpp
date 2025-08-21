
//              Copyright Catch2 Authors
// Distributed under the Boost Software License, Version 1.0.
//   (See accompanying file LICENSE.txt or copy at
//        https://www.boost.org/LICENSE_1_0.txt)

// SPDX-License-Identifier: BSL-1.0
#ifndef CATCH_TO_STRING_HPP_INCLUDED
#define CATCH_TO_STRING_HPP_INCLUDED

#include <string>

#include <catch2/internal/catch_compiler_capabilities.hpp>
#include <catch2/internal/catch_reusable_string_stream.hpp>

namespace Catch {
    template <typename T>
    std::string to_string(T const& t) {
#if defined(CATCH_CONFIG_CPP11_TO_STRING)
        return std::to_string(t);
#else
        ReusableStringStream rss;
        rss << t;
        return rss.str();
#endif
    }
} // end namespace Catch

#endif // CATCH_TO_STRING_HPP_INCLUDED
