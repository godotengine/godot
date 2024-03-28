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

#include <dlfcn.h>
#include <stdint.h>
#include <sys/types.h>

#include "AdpfWrapper.h"
#include "AudioClock.h"
#include "OboeDebug.h"

typedef APerformanceHintManager* (*APH_getManager)();
typedef APerformanceHintSession* (*APH_createSession)(APerformanceHintManager*, const int32_t*,
                                                      size_t, int64_t);
typedef void (*APH_reportActualWorkDuration)(APerformanceHintSession*, int64_t);
typedef void (*APH_closeSession)(APerformanceHintSession* session);

static bool gAPerformanceHintBindingInitialized = false;
static APH_getManager gAPH_getManagerFn = nullptr;
static APH_createSession gAPH_createSessionFn = nullptr;
static APH_reportActualWorkDuration gAPH_reportActualWorkDurationFn = nullptr;
static APH_closeSession gAPH_closeSessionFn = nullptr;

static int loadAphFunctions() {
    if (gAPerformanceHintBindingInitialized) return true;

    void* handle_ = dlopen("libandroid.so", RTLD_NOW | RTLD_NODELETE);
    if (handle_ == nullptr) {
        return -1000;
    }

    gAPH_getManagerFn = (APH_getManager)dlsym(handle_, "APerformanceHint_getManager");
    if (gAPH_getManagerFn == nullptr) {
        return -1001;
    }

    gAPH_createSessionFn = (APH_createSession)dlsym(handle_, "APerformanceHint_createSession");
    if (gAPH_getManagerFn == nullptr) {
        return -1002;
    }

    gAPH_reportActualWorkDurationFn = (APH_reportActualWorkDuration)dlsym(
            handle_, "APerformanceHint_reportActualWorkDuration");
    if (gAPH_getManagerFn == nullptr) {
        return -1003;
    }

    gAPH_closeSessionFn = (APH_closeSession)dlsym(handle_, "APerformanceHint_closeSession");
    if (gAPH_getManagerFn == nullptr) {
        return -1004;
    }

    gAPerformanceHintBindingInitialized = true;
    return 0;
}

bool AdpfWrapper::sUseAlternativeHack = false; // TODO remove hack

int AdpfWrapper::open(pid_t threadId,
                      int64_t targetDurationNanos) {
    std::lock_guard<std::mutex> lock(mLock);
    int result = loadAphFunctions();
    if (result < 0) return result;

    // This is a singleton.
    APerformanceHintManager* manager = gAPH_getManagerFn();

    int32_t thread32 = threadId;
    if (sUseAlternativeHack) {
        // TODO Remove this hack when we finish experimenting with alternative algorithms.
        // The A5 is an arbitrary signal to a hacked version of ADPF to try an alternative
        // algorithm that is not based on PID.
        targetDurationNanos = (targetDurationNanos & ~0xFF) | 0xA5;
    }
    mHintSession = gAPH_createSessionFn(manager, &thread32, 1 /* size */, targetDurationNanos);
    if (mHintSession == nullptr) {
        return -1;
    }
    return 0;
}

void AdpfWrapper::reportActualDuration(int64_t actualDurationNanos) {
    //LOGD("ADPF Oboe %s(dur=%lld)", __func__, (long long)actualDurationNanos);
    std::lock_guard<std::mutex> lock(mLock);
    if (mHintSession != nullptr) {
        gAPH_reportActualWorkDurationFn(mHintSession, actualDurationNanos);
    }
}

void AdpfWrapper::close() {
    std::lock_guard<std::mutex> lock(mLock);
    if (mHintSession != nullptr) {
        gAPH_closeSessionFn(mHintSession);
        mHintSession = nullptr;
    }
}

void AdpfWrapper::onBeginCallback() {
    if (isOpen()) {
        mBeginCallbackNanos = oboe::AudioClock::getNanoseconds(CLOCK_REALTIME);
    }
}

void AdpfWrapper::onEndCallback(double durationScaler) {
    if (isOpen()) {
        int64_t endCallbackNanos = oboe::AudioClock::getNanoseconds(CLOCK_REALTIME);
        int64_t actualDurationNanos = endCallbackNanos - mBeginCallbackNanos;
        int64_t scaledDurationNanos = static_cast<int64_t>(actualDurationNanos * durationScaler);
        reportActualDuration(scaledDurationNanos);
    }
}
