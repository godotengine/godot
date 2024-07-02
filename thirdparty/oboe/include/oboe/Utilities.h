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

#ifndef OBOE_UTILITIES_H
#define OBOE_UTILITIES_H

#include <unistd.h>
#include <sys/types.h>
#include <string>
#include "oboe/Definitions.h"

namespace oboe {

/**
 * Convert an array of floats to an array of 16-bit integers.
 *
 * @param source the input array.
 * @param destination the output array.
 * @param numSamples the number of values to convert.
 */
void convertFloatToPcm16(const float *source, int16_t *destination, int32_t numSamples);

/**
 * Convert an array of 16-bit integers to an array of floats.
 *
 * @param source the input array.
 * @param destination the output array.
 * @param numSamples the number of values to convert.
 */
void convertPcm16ToFloat(const int16_t *source, float *destination, int32_t numSamples);

/**
 * @return the size of a sample of the given format in bytes or 0 if format is invalid
 */
int32_t convertFormatToSizeInBytes(AudioFormat format);

/**
 * The text is the ASCII symbol corresponding to the supplied Oboe enum value,
 * or an English message saying the value is unrecognized.
 * This is intended for developers to use when debugging.
 * It is not for displaying to users.
 *
 * @param input object to convert from. @see common/Utilities.cpp for concrete implementations
 * @return text representation of an Oboe enum value. There is no need to call free on this.
 */
template <typename FromType>
const char * convertToText(FromType input);

/**
 * @param name
 * @return the value of a named system property in a string or empty string
 */
std::string getPropertyString(const char * name);

/**
 * @param name
 * @param defaultValue
 * @return integer value associated with a property or the default value
 */
int getPropertyInteger(const char * name, int defaultValue);

/**
 * Return the version of the SDK that is currently running.
 *
 * For example, on Android, this would return 27 for Oreo 8.1.
 * If the version number cannot be determined then this will return -1.
 *
 * @return version number or -1
 */
int getSdkVersion();

/**
 * Returns whether a device is on a pre-release SDK that is at least the specified codename
 * version.
 *
 * @param codename the code name to verify.
 * @return boolean of whether the device is on a pre-release SDK and is at least the specified
 * codename
 */
bool isAtLeastPreReleaseCodename(const std::string& codename);

int getChannelCountFromChannelMask(ChannelMask channelMask);

} // namespace oboe

#endif //OBOE_UTILITIES_H
