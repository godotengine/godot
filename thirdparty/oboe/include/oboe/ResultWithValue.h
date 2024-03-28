/*
 * Copyright (C) 2018 The Android Open Source Project
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

#ifndef OBOE_RESULT_WITH_VALUE_H
#define OBOE_RESULT_WITH_VALUE_H

#include "oboe/Definitions.h"
#include <iostream>
#include <sstream>

namespace oboe {

/**
 * A ResultWithValue can store both the result of an operation (either OK or an error) and a value.
 *
 * It has been designed for cases where the caller needs to know whether an operation succeeded and,
 * if it did, a value which was obtained during the operation.
 *
 * For example, when reading from a stream the caller needs to know the result of the read operation
 * and, if it was successful, how many frames were read. Note that ResultWithValue can be evaluated
 * as a boolean so it's simple to check whether the result is OK.
 *
 * <code>
 * ResultWithValue<int32_t> resultOfRead = myStream.read(&buffer, numFrames, timeoutNanoseconds);
 *
 * if (resultOfRead) {
 *     LOGD("Frames read: %d", resultOfRead.value());
 * } else {
 *     LOGD("Error reading from stream: %s", resultOfRead.error());
 * }
 * </code>
 */
template <typename T>
class ResultWithValue {
public:

    /**
     * Construct a ResultWithValue containing an error result.
     *
     * @param error The error
     */
    ResultWithValue(oboe::Result error)
            : mValue{}
            , mError(error) {}

    /**
     * Construct a ResultWithValue containing an OK result and a value.
     *
     * @param value the value to store
     */
    explicit ResultWithValue(T value)
            : mValue(value)
            , mError(oboe::Result::OK) {}

    /**
     * Get the result.
     *
     * @return the result
     */
    oboe::Result error() const {
        return mError;
    }

    /**
     * Get the value
     * @return
     */
    T value() const {
        return mValue;
    }

    /**
     * @return true if OK
     */
    explicit operator bool() const { return mError == oboe::Result::OK; }

    /**
     * Quick way to check for an error.
     *
     * The caller could write something like this:
     * <code>
     *     if (!result) { printf("Got error %s\n", convertToText(result.error())); }
     * </code>
     *
     * @return true if an error occurred
     */
    bool operator !() const { return mError != oboe::Result::OK; }

    /**
     * Implicitly convert to a Result. This enables easy comparison with Result values. Example:
     *
     * <code>
     *     ResultWithValue result = openStream();
     *     if (result == Result::ErrorNoMemory){ // tell user they're out of memory }
     * </code>
     */
    operator Result() const {
        return mError;
    }

    /**
     * Create a ResultWithValue from a number. If the number is positive the ResultWithValue will
     * have a result of Result::OK and the value will contain the number. If the number is negative
     * the result will be obtained from the negative number (numeric error codes can be found in
     * AAudio.h) and the value will be null.
     *
     */
    static ResultWithValue<T> createBasedOnSign(T numericResult){

        // Ensure that the type is either an integer or float
        static_assert(std::is_arithmetic<T>::value,
                      "createBasedOnSign can only be called for numeric types (int or float)");

        if (numericResult >= 0){
            return ResultWithValue<T>(numericResult);
        } else {
            return ResultWithValue<T>(static_cast<Result>(numericResult));
        }
    }

private:
    const T             mValue;
    const oboe::Result  mError;
};

/**
 * If the result is `OK` then return the value, otherwise return a human-readable error message.
 */
template <typename T>
std::ostream& operator<<(std::ostream &strm, const ResultWithValue<T> &result) {
    if (!result) {
        strm << convertToText(result.error());
    } else {
        strm << result.value();
    }
   return strm;
}

} // namespace oboe


#endif //OBOE_RESULT_WITH_VALUE_H
