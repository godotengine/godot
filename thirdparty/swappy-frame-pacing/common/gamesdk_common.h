/*
 * Copyright 2020 The Android Open Source Project
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

/*
 * This is the main interface to the Android Performance Tuner library, also
 * known as Tuning Fork.
 *
 * It is part of the Android Games SDK and produces best results when integrated
 * with the Swappy Frame Pacing Library.
 *
 * See the documentation at
 * https://developer.android.com/games/sdk/performance-tuner/custom-engine for
 * more information on using this library in a native Android game.
 *
 */

#pragma once

// There are separate versions for each GameSDK component that use this format:
#define ANDROID_GAMESDK_PACKED_VERSION(MAJOR, MINOR, BUGFIX) \
    ((MAJOR << 16) | (MINOR << 8) | (BUGFIX))
// Accessors
#define ANDROID_GAMESDK_MAJOR_VERSION(PACKED) ((PACKED) >> 16)
#define ANDROID_GAMESDK_MINOR_VERSION(PACKED) (((PACKED) >> 8) & 0xff)
#define ANDROID_GAMESDK_BUGFIX_VERSION(PACKED) ((PACKED) & 0xff)

#define AGDK_STRING_VERSION(MAJOR, MINOR, BUGFIX, GIT) \
#MAJOR "." #MINOR "." #BUGFIX "." #GIT
