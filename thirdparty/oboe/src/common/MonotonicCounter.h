/*
 * Copyright 2016 The Android Open Source Project
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef COMMON_MONOTONIC_COUNTER_H
#define COMMON_MONOTONIC_COUNTER_H

#include <cstdint>

/**
 * Maintain a 64-bit monotonic counter.
 * Can be used to track a 32-bit counter that wraps or gets reset.
 *
 * Note that this is not atomic and has no interior locks.
 * A caller will need to provide their own exterior locking
 * if they need to use it from multiple threads.
 */
class MonotonicCounter {

public:
    MonotonicCounter() {}
    virtual ~MonotonicCounter() {}

    /**
     * @return current value of the counter
     */
    int64_t get() const {
        return mCounter64;
    }

    /**
     * set the current value of the counter
     */
    void set(int64_t counter) {
        mCounter64 = counter;
    }

    /**
     * Advance the counter if delta is positive.
     * @return current value of the counter
     */
    int64_t increment(int64_t delta) {
        if (delta > 0) {
            mCounter64 += delta;
        }
        return mCounter64;
    }

    /**
     * Advance the 64-bit counter if (current32 - previousCurrent32) > 0.
     * This can be used to convert a 32-bit counter that may be wrapping into
     * a monotonic 64-bit counter.
     *
     * This counter32 should NOT be allowed to advance by more than 0x7FFFFFFF between calls.
     * Think of the wrapping counter like a sine wave. If the frequency of the signal
     * is more than half the sampling rate (Nyquist rate) then you cannot measure it properly.
     * If the counter wraps around every 24 hours then we should measure it with a period
     * of less than 12 hours.
     *
     * @return current value of the 64-bit counter
     */
    int64_t update32(int32_t counter32) {
        int32_t delta = counter32 - mCounter32;
        // protect against the mCounter64 going backwards
        if (delta > 0) {
            mCounter64 += delta;
            mCounter32 = counter32;
        }
        return mCounter64;
    }

    /**
     * Reset the stored value of the 32-bit counter.
     * This is used if your counter32 has been reset to zero.
     */
    void reset32() {
        mCounter32 = 0;
    }

    /**
     * Round 64-bit counter up to a multiple of the period.
     *
     * The period must be positive.
     *
     * @param period might be, for example, a buffer capacity
     */
    void roundUp64(int32_t period) {
        if (period > 0) {
            int64_t numPeriods = (mCounter64 + period - 1) / period;
            mCounter64 = numPeriods * period;
        }
    }

private:
    int64_t mCounter64 = 0;
    int32_t mCounter32 = 0;
};


#endif //COMMON_MONOTONIC_COUNTER_H
