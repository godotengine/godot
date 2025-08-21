
//              Copyright Catch2 Authors
// Distributed under the Boost Software License, Version 1.0.
//   (See accompanying file LICENSE.txt or copy at
//        https://www.boost.org/LICENSE_1_0.txt)

// SPDX-License-Identifier: BSL-1.0
#ifndef CATCH_THREAD_SUPPORT_HPP_INCLUDED
#define CATCH_THREAD_SUPPORT_HPP_INCLUDED

#include <catch2/catch_user_config.hpp>

#if defined( CATCH_CONFIG_EXPERIMENTAL_THREAD_SAFE_ASSERTIONS )
#    include <atomic>
#    include <mutex>
#endif

#include <catch2/catch_totals.hpp>

namespace Catch {
    namespace Detail {
#if defined( CATCH_CONFIG_EXPERIMENTAL_THREAD_SAFE_ASSERTIONS )
        using Mutex = std::mutex;
        using LockGuard = std::lock_guard<std::mutex>;
        struct AtomicCounts {
            std::atomic<std::uint64_t> passed = 0;
            std::atomic<std::uint64_t> failed = 0;
            std::atomic<std::uint64_t> failedButOk = 0;
            std::atomic<std::uint64_t> skipped = 0;
        };
#else // ^^ Use actual mutex, lock and atomics
      // vv Dummy implementations for single-thread performance

        struct Mutex {
            void lock() {}
            void unlock() {}
        };

        struct LockGuard {
            LockGuard( Mutex ) {}
        };

        using AtomicCounts = Counts;
#endif

    } // namespace Detail
} // namespace Catch

#endif // CATCH_THREAD_SUPPORT_HPP_INCLUDED
