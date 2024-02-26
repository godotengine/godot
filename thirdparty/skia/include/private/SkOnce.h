/*
 * Copyright 2013 Google Inc.
 *
 * Use of this source code is governed by a BSD-style license that can be
 * found in the LICENSE file.
 */

#ifndef SkOnce_DEFINED
#define SkOnce_DEFINED

#include "include/private/SkThreadAnnotations.h"
#include <atomic>
#include <utility>

// SkOnce provides call-once guarantees for Skia, much like std::once_flag/std::call_once().
//
// There should be no particularly error-prone gotcha use cases when using SkOnce.
// It works correctly as a class member, a local, a global, a function-scoped static, whatever.

class SkOnce {
public:
    constexpr SkOnce() = default;

    template <typename Fn, typename... Args>
    void operator()(Fn&& fn, Args&&... args) {
        auto state = fState.load(std::memory_order_acquire);

        if (state == Done) {
            return;
        }

        // If it looks like no one has started calling fn(), try to claim that job.
        if (state == NotStarted && fState.compare_exchange_strong(state, Claimed,
                                                                  std::memory_order_relaxed,
                                                                  std::memory_order_relaxed)) {
            // Great!  We'll run fn() then notify the other threads by releasing Done into fState.
            fn(std::forward<Args>(args)...);
            return fState.store(Done, std::memory_order_release);
        }

        // Some other thread is calling fn().
        // We'll just spin here acquiring until it releases Done into fState.
        SK_POTENTIALLY_BLOCKING_REGION_BEGIN;
        while (fState.load(std::memory_order_acquire) != Done) { /*spin*/ }
        SK_POTENTIALLY_BLOCKING_REGION_END;
    }

private:
    enum State : uint8_t { NotStarted, Claimed, Done};
    std::atomic<uint8_t> fState{NotStarted};
};

#endif  // SkOnce_DEFINED
