/*
 * Copyright 2017 The Android Open Source Project
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


#include "common/OboeDebug.h"
#include "EngineOpenSLES.h"
#include "OpenSLESUtilities.h"
#include "OutputMixerOpenSLES.h"

using namespace oboe;

OutputMixerOpenSL &OutputMixerOpenSL::getInstance() {
    static OutputMixerOpenSL sInstance;
    return sInstance;
}

SLresult OutputMixerOpenSL::open() {
    std::lock_guard<std::mutex> lock(mLock);

    SLresult result = SL_RESULT_SUCCESS;
    if (mOpenCount++ == 0) {
        // get the output mixer
        result = EngineOpenSLES::getInstance().createOutputMix(&mOutputMixObject);
        if (SL_RESULT_SUCCESS != result) {
            LOGE("OutputMixerOpenSL() - createOutputMix() result:%s", getSLErrStr(result));
            goto error;
        }

        // realize the output mix
        result = (*mOutputMixObject)->Realize(mOutputMixObject, SL_BOOLEAN_FALSE);
        if (SL_RESULT_SUCCESS != result) {
            LOGE("OutputMixerOpenSL() - Realize() mOutputMixObject result:%s", getSLErrStr(result));
            goto error;
        }
    }

    return result;

error:
    close();
    return result;
}

void OutputMixerOpenSL::close() {
    std::lock_guard<std::mutex> lock(mLock);

    if (--mOpenCount == 0) {
        // destroy output mix object, and invalidate all associated interfaces
        if (mOutputMixObject != nullptr) {
            (*mOutputMixObject)->Destroy(mOutputMixObject);
            mOutputMixObject = nullptr;
        }
    }
}

SLresult OutputMixerOpenSL::createAudioPlayer(SLObjectItf *objectItf,
                                              SLDataSource *audioSource) {
    SLDataLocator_OutputMix loc_outmix = {SL_DATALOCATOR_OUTPUTMIX, mOutputMixObject};
    SLDataSink audioSink = {&loc_outmix, NULL};
    return EngineOpenSLES::getInstance().createAudioPlayer(objectItf, audioSource, &audioSink);
}
