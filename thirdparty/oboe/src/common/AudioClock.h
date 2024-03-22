/*
 * Copyright (C) 2016 The Android Open Source Project
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

#ifndef OBOE_AUDIO_CLOCK_H
#define OBOE_AUDIO_CLOCK_H

#include <sys/types.h>
#include <ctime>
#include "oboe/Definitions.h"

namespace oboe {

// TODO: Move this class into the public headers because it is useful when calculating stream latency
class AudioClock {
public:
    static int64_t getNanoseconds(clockid_t clockId = CLOCK_MONOTONIC) {
        struct timespec time;
        int result = clock_gettime(clockId, &time);
        if (result < 0) {
            return result;
        }
        return (time.tv_sec * kNanosPerSecond) + time.tv_nsec;
    }

    /**
     * Sleep until the specified time.
     *
     * @param nanoTime time to wake up
     * @param clockId CLOCK_MONOTONIC is default
     * @return 0 or a negative error, eg. -EINTR
     */

    static int sleepUntilNanoTime(int64_t nanoTime, clockid_t clockId = CLOCK_MONOTONIC) {
        struct timespec time;
        time.tv_sec = nanoTime / kNanosPerSecond;
        time.tv_nsec = nanoTime - (time.tv_sec * kNanosPerSecond);
        return 0 - clock_nanosleep(clockId, TIMER_ABSTIME, &time, NULL);
    }

    /**
     * Sleep for the specified number of nanoseconds in real-time.
     * Return immediately with 0 if a negative nanoseconds is specified.
     *
     * @param nanoseconds time to sleep
     * @param clockId CLOCK_REALTIME is default
     * @return 0 or a negative error, eg. -EINTR
     */

    static int sleepForNanos(int64_t nanoseconds, clockid_t clockId = CLOCK_REALTIME) {
        if (nanoseconds > 0) {
            struct timespec time;
            time.tv_sec = nanoseconds / kNanosPerSecond;
            time.tv_nsec = nanoseconds - (time.tv_sec * kNanosPerSecond);
            return 0 - clock_nanosleep(clockId, 0, &time, NULL);
        }
        return 0;
    }
};

} // namespace oboe

#endif //OBOE_AUDIO_CLOCK_H
