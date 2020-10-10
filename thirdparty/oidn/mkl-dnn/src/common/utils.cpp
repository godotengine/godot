/*******************************************************************************
* Copyright 2018 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/

#include <string.h>
#ifdef _WIN32
#include <malloc.h>
#include <windows.h>
#endif
#include <limits.h>
#include <stdlib.h>
#include <stdio.h>

#include "mkldnn.h"
#include "utils.hpp"

namespace mkldnn {
namespace impl {

int getenv(const char *name, char *buffer, int buffer_size) {
    if (name == NULL || buffer_size < 0 || (buffer == NULL && buffer_size > 0))
        return INT_MIN;

    int result = 0;
    int term_zero_idx = 0;
    size_t value_length = 0;

#ifdef _WIN32
    value_length = GetEnvironmentVariable(name, buffer, buffer_size);
#else
    const char *value = ::getenv(name);
    value_length = value == NULL ? 0 : strlen(value);
#endif

    if (value_length > INT_MAX)
        result = INT_MIN;
    else {
        int int_value_length = (int)value_length;
        if (int_value_length >= buffer_size) {
            result = -int_value_length;
        } else {
            term_zero_idx = int_value_length;
            result = int_value_length;
#ifndef _WIN32
            strncpy(buffer, value, value_length);
#endif
        }
    }

    if (buffer != NULL)
        buffer[term_zero_idx] = '\0';
    return result;
}

int getenv_int(const char *name, int default_value)
{
    int value = default_value;
    // # of digits in the longest 32-bit signed int + sign + terminating null
    const int len = 12;
    char value_str[len];
    if (getenv(name, value_str, len) > 0)
        value = atoi(value_str);
    return value;
}

FILE *fopen(const char *filename, const char *mode) {
#ifdef _WIN32
    FILE *fp = NULL;
    return ::fopen_s(&fp, filename, mode) ? NULL : fp;
#else
    return ::fopen(filename, mode);
#endif
}

void *malloc(size_t size, int alignment) {
    void *ptr;

#ifdef _WIN32
    ptr = _aligned_malloc(size, alignment);
    int rc = ptr ? 0 : -1;
#else
    int rc = ::posix_memalign(&ptr, alignment, size);
#endif

    return (rc == 0) ? ptr : 0;
}

void free(void *p) {
#ifdef _WIN32
    _aligned_free(p);
#else
    ::free(p);
#endif
}

// Atomic operations
int32_t fetch_and_add(int32_t *dst, int32_t val) {
#ifdef _WIN32
    return InterlockedExchangeAdd(reinterpret_cast<long*>(dst), val);
#else
    return __sync_fetch_and_add(dst, val);
#endif
}

static int jit_dump_flag = 0;
static bool jit_dump_flag_initialized = false;
bool jit_dump_enabled() {
    if (!jit_dump_flag_initialized) {
        jit_dump_flag = getenv_int("MKLDNN_JIT_DUMP");
        jit_dump_flag_initialized = true;
    }
    return jit_dump_flag != 0;
}

}
}

mkldnn_status_t mkldnn_set_jit_dump(int enabled) {
    using namespace mkldnn::impl::status;
    mkldnn::impl::jit_dump_flag = enabled;
    mkldnn::impl::jit_dump_flag_initialized = true;
    return success;
}
