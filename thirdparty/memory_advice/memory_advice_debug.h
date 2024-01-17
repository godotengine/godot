/*
 * Copyright 2022 The Android Open Source Project
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
 * @defgroup memory_advice_debug Memory Advice debugging functions
 * @{
 */

#pragma once

#include <jni.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief A char* representing a serialized json object.
 * @see MemoryAdvice_JsonSerialization_free for how to deallocate
 * the memory once finished with the buffer.
 */
typedef struct MemoryAdvice_JsonSerialization {
    char *json;     ///< String for the json object.
    uint32_t size;  ///< Size of the json string.
    ///< Deallocation callback (may be NULL if not owned).
    void (*dealloc)(struct MemoryAdvice_JsonSerialization *);
} MemoryAdvice_JsonSerialization;

/**
 * @brief Returns the advice regarding the current memory state.
 *
 * @param advice a pointer to a struct, in which the memory advice will be
 * written.
 *
 * @return MEMORYADVICE_ERROR_OK if successful,
 * @return MEMORYADVICE_ERROR_NOT_INITIALIZED if Memory Advice was not yet
 * initialized.
 */
MemoryAdvice_ErrorCode MemoryAdvice_getAdvice(
    MemoryAdvice_JsonSerialization *advice);

/**
 * @brief Deallocate any memory owned by the json serialization.
 * @param ser A json serialization
 */
void MemoryAdvice_JsonSerialization_free(MemoryAdvice_JsonSerialization *ser);

/**
 * @brief Perform tests on the memory advice library.
 *
 * @return 0 if the tests pass.
 * @return non-zero if the tests fail. Error messages will be printed to logcat
 * in this case.
 */
int32_t MemoryAdvice_test();

#ifdef __cplusplus
}  // extern "C" {
#endif
