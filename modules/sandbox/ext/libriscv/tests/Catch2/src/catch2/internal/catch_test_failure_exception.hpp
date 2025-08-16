
//              Copyright Catch2 Authors
// Distributed under the Boost Software License, Version 1.0.
//   (See accompanying file LICENSE.txt or copy at
//        https://www.boost.org/LICENSE_1_0.txt)

// SPDX-License-Identifier: BSL-1.0
#ifndef CATCH_TEST_FAILURE_EXCEPTION_HPP_INCLUDED
#define CATCH_TEST_FAILURE_EXCEPTION_HPP_INCLUDED

namespace Catch {

    //! Used to signal that an assertion macro failed
    struct TestFailureException{};
    //! Used to signal that the remainder of a test should be skipped
    struct TestSkipException {};

    /**
     * Outlines throwing of `TestFailureException` into a single TU
     *
     * Also handles `CATCH_CONFIG_DISABLE_EXCEPTIONS` for callers.
     */
    [[noreturn]] void throw_test_failure_exception();

    /**
     * Outlines throwing of `TestSkipException` into a single TU
     *
     * Also handles `CATCH_CONFIG_DISABLE_EXCEPTIONS` for callers.
     */
    [[noreturn]] void throw_test_skip_exception();

} // namespace Catch

#endif // CATCH_TEST_FAILURE_EXCEPTION_HPP_INCLUDED
