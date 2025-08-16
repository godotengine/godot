
//              Copyright Catch2 Authors
// Distributed under the Boost Software License, Version 1.0.
//   (See accompanying file LICENSE.txt or copy at
//        https://www.boost.org/LICENSE_1_0.txt)

// SPDX-License-Identifier: BSL-1.0
#ifndef CATCH_TIMER_HPP_INCLUDED
#define CATCH_TIMER_HPP_INCLUDED

#include <cstdint>

namespace Catch {

    class Timer {
        uint64_t m_nanoseconds = 0;
    public:
        void start();
        auto getElapsedNanoseconds() const -> uint64_t;
        auto getElapsedMicroseconds() const -> uint64_t;
        auto getElapsedMilliseconds() const -> unsigned int;
        auto getElapsedSeconds() const -> double;
    };

} // namespace Catch

#endif // CATCH_TIMER_HPP_INCLUDED
