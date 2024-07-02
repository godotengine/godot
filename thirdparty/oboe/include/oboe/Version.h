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

#ifndef OBOE_VERSIONINFO_H
#define OBOE_VERSIONINFO_H

#include <cstdint>

/**
 * A note on use of preprocessor defines:
 *
 * This is one of the few times when it's suitable to use preprocessor defines rather than constexpr
 * Why? Because C++11 requires a lot of boilerplate code to convert integers into compile-time
 * string literals. The preprocessor, despite it's lack of type checking, is more suited to the task
 *
 * See: https://stackoverflow.com/questions/6713420/c-convert-integer-to-string-at-compile-time/26824971#26824971
 *
 */

// Type: 8-bit unsigned int. Min value: 0 Max value: 255. See below for description.
#define OBOE_VERSION_MAJOR 1

// Type: 8-bit unsigned int. Min value: 0 Max value: 255. See below for description.
#define OBOE_VERSION_MINOR 8

// Type: 16-bit unsigned int. Min value: 0 Max value: 65535. See below for description.
#define OBOE_VERSION_PATCH 2

#define OBOE_STRINGIFY(x) #x
#define OBOE_TOSTRING(x) OBOE_STRINGIFY(x)

// Type: String literal. See below for description.
#define OBOE_VERSION_TEXT \
        OBOE_TOSTRING(OBOE_VERSION_MAJOR) "." \
        OBOE_TOSTRING(OBOE_VERSION_MINOR) "." \
        OBOE_TOSTRING(OBOE_VERSION_PATCH)

// Type: 32-bit unsigned int. See below for description.
#define OBOE_VERSION_NUMBER ((OBOE_VERSION_MAJOR << 24) | (OBOE_VERSION_MINOR << 16) | OBOE_VERSION_PATCH)

namespace oboe {

const char * getVersionText();

/**
 * Oboe versioning object
 */
struct Version {
    /**
     * This is incremented when we make breaking API changes. Based loosely on https://semver.org/.
     */
    static constexpr uint8_t Major = OBOE_VERSION_MAJOR;

    /**
     * This is incremented when we add backwards compatible functionality. Or set to zero when MAJOR is
     * incremented.
     */
    static constexpr uint8_t Minor = OBOE_VERSION_MINOR;

    /**
     * This is incremented when we make backwards compatible bug fixes. Or set to zero when MINOR is
     * incremented.
     */
    static constexpr uint16_t Patch = OBOE_VERSION_PATCH;

    /**
     * Version string in the form MAJOR.MINOR.PATCH.
     */
    static constexpr const char * Text = OBOE_VERSION_TEXT;

    /**
     * Integer representation of the current Oboe library version. This will always increase when the
     * version number changes so can be compared using integer comparison.
     */
    static constexpr uint32_t Number = OBOE_VERSION_NUMBER;
};

} // namespace oboe
#endif //OBOE_VERSIONINFO_H
