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

#ifndef AUDIO_INPUT_STREAM_OPENSL_ES_H_
#define AUDIO_INPUT_STREAM_OPENSL_ES_H_


#include <SLES/OpenSLES.h>
#include <SLES/OpenSLES_Android.h>

#include "oboe/Oboe.h"
#include "AudioStreamOpenSLES.h"

namespace oboe {

/**
 * INTERNAL USE ONLY
 */

class AudioInputStreamOpenSLES : public AudioStreamOpenSLES {
public:
    AudioInputStreamOpenSLES();
    explicit AudioInputStreamOpenSLES(const AudioStreamBuilder &builder);

    virtual ~AudioInputStreamOpenSLES();

    Result open() override;
    Result close() override;

    Result requestStart() override;
    Result requestPause() override;
    Result requestFlush() override;
    Result requestStop() override;

protected:
    Result requestStop_l();

    Result updateServiceFrameCounter() override;

    void updateFramesWritten() override;

private:

    SLuint32 channelCountToChannelMask(int chanCount) const;

    Result setRecordState_l(SLuint32 newState);

    SLRecordItf mRecordInterface = nullptr;
};

} // namespace oboe

#endif //AUDIO_INPUT_STREAM_OPENSL_ES_H_
