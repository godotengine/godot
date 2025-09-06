
//              Copyright Catch2 Authors
// Distributed under the Boost Software License, Version 1.0.
//   (See accompanying file LICENSE.txt or copy at
//        https://www.boost.org/LICENSE_1_0.txt)

// SPDX-License-Identifier: BSL-1.0

#include <catch2/internal/catch_getenv.hpp>

#include <catch2/internal/catch_platform.hpp>
#include <catch2/internal/catch_compiler_capabilities.hpp>

#include <cstdlib>

namespace Catch {
    namespace Detail {

#if !defined (CATCH_CONFIG_GETENV)
        char const* getEnv( char const* ) { return nullptr; }
#else

        char const* getEnv( char const* varName ) {
#    if defined( _MSC_VER )
#        pragma warning( push )
#        pragma warning( disable : 4996 ) // use getenv_s instead of getenv
#    endif

            return std::getenv( varName );

#    if defined( _MSC_VER )
#        pragma warning( pop )
#    endif
        }
#endif
} // namespace Detail
} // namespace Catch
