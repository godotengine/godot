/*
 * Copyright 2019 The Android Open Source Project
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
#include "oboe/Version.h"

namespace oboe {

    // This variable enables the version information to be read from the resulting binary e.g.
    // by running `objdump -s --section=.data <binary>`
    // Please do not optimize or change in any way.
    char kVersionText[] = "OboeVersion" OBOE_VERSION_TEXT;

    const char * getVersionText(){
        return kVersionText;
    }
} // namespace oboe
