/*
 * Copyright 2021 The Android Open Source Project
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

#ifndef SYNTHMARK_ADPF_WRAPPER_H
#define SYNTHMARK_ADPF_WRAPPER_H

#include <algorithm>
#include <functional>
#include <stdint.h>
#include <sys/types.h>
#include <unistd.h>
#include <mutex>

struct APerformanceHintManager;
struct APerformanceHintSession;

typedef struct APerformanceHintManager APerformanceHintManager;
typedef struct APerformanceHintSession APerformanceHintSession;

class AdpfWrapper {
public:
     /**
      * Create an ADPF session that can be used to boost performance.
      * @param threadId
      * @param targetDurationNanos - nominal period of isochronous task
      * @return zero or negative error
      */
    int open(pid_t threadId,
             int64_t targetDurationNanos);

    bool isOpen() const {
        return (mHintSession != nullptr);
    }

    void close();

    /**
     * Call this at the beginning of the callback that you are measuring.
     */
    void onBeginCallback();

    /**
     * Call this at the end of the callback that you are measuring.
     * It is OK to skip this if you have a short callback.
     */
    void onEndCallback(double durationScaler);

    /**
     * For internal use only!
     * This is a hack for communicating with experimental versions of ADPF.
     * @param enabled
     */
    static void setUseAlternative(bool enabled) {
        sUseAlternativeHack = enabled;
    }

    /**
     * Report the measured duration of a callback.
     * This is normally called by onEndCallback().
     * You may want to call this directly in order to give an advance hint of a jump in workload.
     * @param actualDurationNanos
     */
    void reportActualDuration(int64_t actualDurationNanos);

private:
    std::mutex               mLock;
    APerformanceHintSession* mHintSession = nullptr;
    int64_t                  mBeginCallbackNanos = 0;
    static bool              sUseAlternativeHack;
};

#endif //SYNTHMARK_ADPF_WRAPPER_H
