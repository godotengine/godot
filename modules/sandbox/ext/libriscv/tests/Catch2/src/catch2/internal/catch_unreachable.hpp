
//              Copyright Catch2 Authors
// Distributed under the Boost Software License, Version 1.0.
//   (See accompanying file LICENSE.txt or copy at
//        https://www.boost.org/LICENSE_1_0.txt)

// SPDX-License-Identifier: BSL-1.0
#ifndef CATCH_UNREACHABLE_HPP_INCLUDED
#define CATCH_UNREACHABLE_HPP_INCLUDED

/**\file
 * Polyfill `std::unreachable`
 *
 * We need something like `std::unreachable` to tell the compiler that
 * some macros, e.g. `FAIL` or `SKIP`, do not continue execution in normal
 * manner, and should handle it as such, e.g. not warn if there is no return
 * from non-void function after a `FAIL` or `SKIP`.
 */

#include <exception>

#if defined( __cpp_lib_unreachable ) && __cpp_lib_unreachable > 202202L
#    include <utility>
namespace Catch {
    namespace Detail {
        using Unreachable = std::unreachable;
    }
} // namespace Catch

#else // vv If we do not have std::unreachable, we implement something similar

namespace Catch {
    namespace Detail {

        [[noreturn]]
        inline void Unreachable() noexcept {
#    if defined( NDEBUG )
#        if defined( _MSC_VER ) && !defined( __clang__ )
            __assume( false );
#        elif defined( __GNUC__ )
            __builtin_unreachable();
#        else // vv platform without known optimization hint
            std::terminate();
#        endif
#    else  // ^^ NDEBUG
            // For non-release builds, we prefer termination on bug over UB
            std::terminate();
#    endif //
        }

    } // namespace Detail
} // end namespace Catch

#endif

#endif // CATCH_UNREACHABLE_HPP_INCLUDED
