/*
 * Copyright 2018 Google Inc.
 *
 * Use of this source code is governed by a BSD-style license that can be
 * found in the LICENSE file.
 */

#ifndef SkSafe32_DEFINED
#define SkSafe32_DEFINED

#include "include/core/SkTypes.h"

static constexpr int32_t Sk64_pin_to_s32(int64_t x) {
    return x < SK_MinS32 ? SK_MinS32 : (x > SK_MaxS32 ? SK_MaxS32 : (int32_t)x);
}

static constexpr int32_t Sk32_sat_add(int32_t a, int32_t b) {
    return Sk64_pin_to_s32((int64_t)a + (int64_t)b);
}

static constexpr int32_t Sk32_sat_sub(int32_t a, int32_t b) {
    return Sk64_pin_to_s32((int64_t)a - (int64_t)b);
}

// To avoid UBSAN complaints about 2's compliment overflows
//
static constexpr int32_t Sk32_can_overflow_add(int32_t a, int32_t b) {
    return (int32_t)((uint32_t)a + (uint32_t)b);
}
static constexpr int32_t Sk32_can_overflow_sub(int32_t a, int32_t b) {
    return (int32_t)((uint32_t)a - (uint32_t)b);
}

#endif
