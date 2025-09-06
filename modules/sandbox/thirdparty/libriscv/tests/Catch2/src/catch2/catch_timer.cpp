
//              Copyright Catch2 Authors
// Distributed under the Boost Software License, Version 1.0.
//   (See accompanying file LICENSE.txt or copy at
//        https://www.boost.org/LICENSE_1_0.txt)

// SPDX-License-Identifier: BSL-1.0
#include <catch2/catch_timer.hpp>

#include <chrono>

namespace Catch {

    namespace {
        static auto getCurrentNanosecondsSinceEpoch() -> uint64_t {
            return std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();
        }
    } // end unnamed namespace

    void Timer::start() {
       m_nanoseconds = getCurrentNanosecondsSinceEpoch();
    }
    auto Timer::getElapsedNanoseconds() const -> uint64_t {
        return getCurrentNanosecondsSinceEpoch() - m_nanoseconds;
    }
    auto Timer::getElapsedMicroseconds() const -> uint64_t {
        return getElapsedNanoseconds()/1000;
    }
    auto Timer::getElapsedMilliseconds() const -> unsigned int {
        return static_cast<unsigned int>(getElapsedMicroseconds()/1000);
    }
    auto Timer::getElapsedSeconds() const -> double {
        return static_cast<double>(getElapsedMicroseconds())/1000000.0;
    }


} // namespace Catch
