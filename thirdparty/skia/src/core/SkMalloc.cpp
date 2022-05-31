// Copyright 2019 Google LLC.
// Use of this source code is governed by a BSD-style license that can be found in the LICENSE file.

#include "include/private/SkMalloc.h"

#include "src/core/SkSafeMath.h"

void* sk_calloc_throw(size_t count, size_t elemSize) {
    return sk_calloc_throw(SkSafeMath::Mul(count, elemSize));
}

void* sk_malloc_throw(size_t count, size_t elemSize) {
    return sk_malloc_throw(SkSafeMath::Mul(count, elemSize));
}

void* sk_realloc_throw(void* buffer, size_t count, size_t elemSize) {
    return sk_realloc_throw(buffer, SkSafeMath::Mul(count, elemSize));
}

void* sk_malloc_canfail(size_t count, size_t elemSize) {
    return sk_malloc_canfail(SkSafeMath::Mul(count, elemSize));
}
