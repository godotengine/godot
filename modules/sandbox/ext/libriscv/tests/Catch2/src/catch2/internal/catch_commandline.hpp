
//              Copyright Catch2 Authors
// Distributed under the Boost Software License, Version 1.0.
//   (See accompanying file LICENSE.txt or copy at
//        https://www.boost.org/LICENSE_1_0.txt)

// SPDX-License-Identifier: BSL-1.0
#ifndef CATCH_COMMANDLINE_HPP_INCLUDED
#define CATCH_COMMANDLINE_HPP_INCLUDED

#include <catch2/internal/catch_clara.hpp>

namespace Catch {

    struct ConfigData;

    Clara::Parser makeCommandLineParser( ConfigData& config );

} // end namespace Catch

#endif // CATCH_COMMANDLINE_HPP_INCLUDED
