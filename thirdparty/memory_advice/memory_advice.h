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

/**
 * @defgroup memory_advice Memory Advice main interface
 * The main interface to use Memory Advice.
 * @{
 */

#pragma once

#include <jni.h>
#include <stdint.h>

#include "gamesdk_common.h"

#ifdef __cplusplus
extern "C" {
#endif

#define MEMORY_ADVICE_MAJOR_VERSION 2
#define MEMORY_ADVICE_MINOR_VERSION 0
#define MEMORY_ADVICE_BUGFIX_VERSION 0
#define MEMORY_ADVICE_PACKED_VERSION                         \
    ANDROID_GAMESDK_PACKED_VERSION(TUNINGFORK_MAJOR_VERSION, \
                                   TUNINGFORK_MINOR_VERSION, \
                                   TUNINGFORK_BUGFIX_VERSION)

/**
 * @brief All the error codes that can be returned by MemoryAdvice functions.
 */
typedef enum MemoryAdvice_ErrorCode : int32_t {
    MEMORYADVICE_ERROR_OK = 0,  ///< No error
    MEMORYADVICE_ERROR_NOT_INITIALIZED =
        -1,  ///< A call was made before MemoryAdvice was initialized.
    MEMORYADVICE_ERROR_ALREADY_INITIALIZED =
        -2,  ///< MemoryAdvice_init was called more than once.
    MEMORYADVICE_ERROR_LOOKUP_TABLE_INVALID =
        -3,  ///< The provided lookup table was not a valid json object.
    MEMORYADVICE_ERROR_ADVISOR_PARAMETERS_INVALID =
        -4,  ///< The provided advisor parameters was not a valid json object.
    MEMORYADVICE_ERROR_WATCHER_NOT_FOUND =
        -5,  ///< UnregisterWatcher was called with an invalid callback.
    MEMORYADVICE_ERROR_TFLITE_MODEL_INVALID =
        -6,  ///< A correct TFLite model was not provided.
} MemoryAdvice_ErrorCode;

/**
 * @brief All possible memory states that can be reported by the library.
 */
typedef enum MemoryAdvice_MemoryState : int32_t {
    MEMORYADVICE_STATE_UNKNOWN = 0,  ///< The memory state cannot be determined.
    MEMORYADVICE_STATE_OK = 1,  ///< The application can safely allocate memory.
    MEMORYADVICE_STATE_APPROACHING_LIMIT =
        2,  ///< The application should minimize memory allocation.
    MEMORYADVICE_STATE_CRITICAL =
        3,  ///< The application should free memory as soon as possible, until
            ///< the memory state changes.
} MemoryAdvice_MemoryState;

typedef void (*MemoryAdvice_WatcherCallback)(MemoryAdvice_MemoryState state,
                                             void *user_data);

/**
 * @brief Initialize the Memory Advice library. This must be called before any
 * other functions.
 *
 * This version of the init function will use the library provided default
 * params.
 *
 * @param env a JNIEnv
 * @param context the app context
 *
 * @return MEMORYADVICE_ERROR_OK if successful,
 * @return MEMORYADVICE_ERROR_ALREADY_INITIALIZED if Memory Advice was already
 * initialized.
 */
MemoryAdvice_ErrorCode MemoryAdvice_init(JNIEnv *env, jobject context);

/**
 * @brief Initialize the Memory Advice library. This must be called before any
 * other functions.
 *
 * This version of the init function will read the given params instead of
 * using the library provided default params.
 *
 * @param env a JNIEnv
 * @param context the app context
 * @param params the advisor parameters to run the library with
 *
 * @return MEMORYADVICE_ERROR_OK if successful,
 * @return MEMORYADVICE_ERROR_ADVISOR_PARAMETERS_INVALID if the provided
 * parameters are not a valid JSON object,
 * @return MEMORYADVICE_ERROR_ALREADY_INITIALIZED if Memory Advice was already
 * initialized.
 */
MemoryAdvice_ErrorCode MemoryAdvice_initWithParams(JNIEnv *env, jobject context,
                                                   const char *params);

/**
 * @brief Returns the current memory state.
 *
 * @param state a pointer to a MemoryAdvice_MemoryState, in which the
 * memory state will be written
 *
 * @return A MemoryAdvice_MemoryState, if successful,
 * @return MEMORYADVICE_ERROR_NOT_INITIALIZED (a negative number) if Memory
 * Advice was not yet initialized.
 */
MemoryAdvice_MemoryState MemoryAdvice_getMemoryState();

/**
 * @brief Calculates an estimate for the amount of memory that can safely be
 * allocated, as a percentage of the total memory.
 *
 * @return A positive number between 0 and 100 with an estimate of the
 * percentage memory available.
 * @return MEMORYADVICE_ERROR_NOT_INITIALIZED (a negative number) if Memory
 * Advice was not yet initialized.
 */
float MemoryAdvice_getPercentageAvailableMemory();

/**
 * @brief Calculates the total memory available on the device, as reported by
 * ActivityManager#getMemoryInfo()
 *
 * @return The total memory of the device, in bytes.
 * @return MEMORYADVICE_ERROR_NOT_INITIALIZED (a negative number) if Memory
 * Advice was not yet initialized.
 */
int64_t MemoryAdvice_getTotalMemory();

/**
 * @brief Registers a watcher that polls the Memory Advice library periodically,
 * and invokes the watcher callback when the memory state goes critical.
 *
 * This function creates another thread that calls MemoryAdvice_getMemoryState
 * every `intervalMillis` milliseconds. If the returned state is not
 * MEMORYADVICE_STATE_OK, then calls the watcher callback with the current
 * state.
 *
 * @param intervalMillis the interval at which the Memory Advice library will be
 * polled
 * @param callback the callback function that will be invoked if memory goes
 * critical
 * @param user_data context to pass to the callback function
 *
 * @return MEMORYADVICE_ERROR_OK if successful,
 * @return MEMORYADVICE_ERROR_NOT_INITIALIZED if Memory Advice was not yet
 * initialized,
 */
MemoryAdvice_ErrorCode MemoryAdvice_registerWatcher(
    uint64_t intervalMillis, MemoryAdvice_WatcherCallback callback,
    void *user_data);

/**
 * @brief Removes all watchers with the given callback that were previously
 * registered using
 * {@link MemoryAdvice_registerWatcher}.
 *
 * @return MEMORYADVICE_ERROR_OK if successful,
 * @return MEMORYADVICE_ERROR_NOT_INITIALIZED if Memory Advice was not yet
 * initialized.
 * @return MEMORYADVICE_ERROR_WATCHER_NOT_FOUND if the given callback wasn't
 * previously registered.
 */
MemoryAdvice_ErrorCode MemoryAdvice_unregisterWatcher(
    MemoryAdvice_WatcherCallback callback);

#ifdef __cplusplus
}  // extern "C" {
#endif
