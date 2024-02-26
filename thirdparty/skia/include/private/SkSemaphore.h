/*
 * Copyright 2015 Google Inc.
 *
 * Use of this source code is governed by a BSD-style license that can be
 * found in the LICENSE file.
 */

#ifndef SkSemaphore_DEFINED
#define SkSemaphore_DEFINED

#include "include/core/SkTypes.h"
#include "include/private/SkOnce.h"
#include "include/private/SkThreadAnnotations.h"
#include <algorithm>
#include <atomic>

class SkSemaphore {
public:
    constexpr SkSemaphore(int count = 0) : fCount(count), fOSSemaphore(nullptr) {}

    // Cleanup the underlying OS semaphore.
    SK_SPI ~SkSemaphore();

    // Increment the counter n times.
    // Generally it's better to call signal(n) instead of signal() n times.
    void signal(int n = 1);

    // Decrement the counter by 1,
    // then if the counter is < 0, sleep this thread until the counter is >= 0.
    void wait();

    // If the counter is positive, decrement it by 1 and return true, otherwise return false.
    SK_SPI bool try_wait();

private:
    // This implementation follows the general strategy of
    //     'A Lightweight Semaphore with Partial Spinning'
    // found here
    //     http://preshing.com/20150316/semaphores-are-surprisingly-versatile/
    // That article (and entire blog) are very much worth reading.
    //
    // We wrap an OS-provided semaphore with a user-space atomic counter that
    // lets us avoid interacting with the OS semaphore unless strictly required:
    // moving the count from >=0 to <0 or vice-versa, i.e. sleeping or waking threads.
    struct OSSemaphore;

    SK_SPI void osSignal(int n);
    SK_SPI void osWait();

    std::atomic<int> fCount;
    SkOnce           fOSSemaphoreOnce;
    OSSemaphore*     fOSSemaphore;
};

inline void SkSemaphore::signal(int n) {
    int prev = fCount.fetch_add(n, std::memory_order_release);

    // We only want to call the OS semaphore when our logical count crosses
    // from <0 to >=0 (when we need to wake sleeping threads).
    //
    // This is easiest to think about with specific examples of prev and n.
    // If n == 5 and prev == -3, there are 3 threads sleeping and we signal
    // std::min(-(-3), 5) == 3 times on the OS semaphore, leaving the count at 2.
    //
    // If prev >= 0, no threads are waiting, std::min(-prev, n) is always <= 0,
    // so we don't call the OS semaphore, leaving the count at (prev + n).
    int toSignal = std::min(-prev, n);
    if (toSignal > 0) {
        this->osSignal(toSignal);
    }
}

inline void SkSemaphore::wait() {
    // Since this fetches the value before the subtract, zero and below means that there are no
    // resources left, so the thread needs to wait.
    if (fCount.fetch_sub(1, std::memory_order_acquire) <= 0) {
        SK_POTENTIALLY_BLOCKING_REGION_BEGIN;
        this->osWait();
        SK_POTENTIALLY_BLOCKING_REGION_END;
    }
}

#endif//SkSemaphore_DEFINED
