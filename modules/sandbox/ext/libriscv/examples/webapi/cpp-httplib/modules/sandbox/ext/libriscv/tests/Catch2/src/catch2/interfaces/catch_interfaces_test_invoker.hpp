
//              Copyright Catch2 Authors
// Distributed under the Boost Software License, Version 1.0.
//   (See accompanying file LICENSE.txt or copy at
//        https://www.boost.org/LICENSE_1_0.txt)

// SPDX-License-Identifier: BSL-1.0
#ifndef CATCH_INTERFACES_TEST_INVOKER_HPP_INCLUDED
#define CATCH_INTERFACES_TEST_INVOKER_HPP_INCLUDED

namespace Catch {

    class ITestInvoker {
    public:
        virtual void prepareTestCase();
        virtual void tearDownTestCase();
        virtual void invoke() const = 0;
        virtual ~ITestInvoker(); // = default
    };

} // namespace Catch

#endif // CATCH_INTERFACES_TEST_INVOKER_HPP_INCLUDED
