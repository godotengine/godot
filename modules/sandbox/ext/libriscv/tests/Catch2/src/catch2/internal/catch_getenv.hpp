
//              Copyright Catch2 Authors
// Distributed under the Boost Software License, Version 1.0.
//   (See accompanying file LICENSE.txt or copy at
//        https://www.boost.org/LICENSE_1_0.txt)

// SPDX-License-Identifier: BSL-1.0
#ifndef CATCH_GETENV_HPP_INCLUDED
#define CATCH_GETENV_HPP_INCLUDED

namespace Catch {
namespace Detail {

    //! Wrapper over `std::getenv` that compiles on UWP (and always returns nullptr there)
    char const* getEnv(char const* varName);

}
}

#endif // CATCH_GETENV_HPP_INCLUDED
