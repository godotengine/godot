
//              Copyright Catch2 Authors
// Distributed under the Boost Software License, Version 1.0.
//   (See accompanying file LICENSE.txt or copy at
//        https://www.boost.org/LICENSE_1_0.txt)

// SPDX-License-Identifier: BSL-1.0

#include <catch2/internal/catch_test_failure_exception.hpp>
#include <catch2/internal/catch_enforce.hpp>
#include <catch2/catch_user_config.hpp>

namespace Catch {

    void throw_test_failure_exception() {
#if !defined( CATCH_CONFIG_DISABLE_EXCEPTIONS )
        throw TestFailureException{};
#else
        CATCH_ERROR( "Test failure requires aborting test!" );
#endif
    }

    void throw_test_skip_exception() {
#if !defined( CATCH_CONFIG_DISABLE_EXCEPTIONS )
        throw Catch::TestSkipException();
#else
        CATCH_ERROR( "Explicitly skipping tests during runtime requires exceptions" );
#endif
    }

} // namespace Catch
