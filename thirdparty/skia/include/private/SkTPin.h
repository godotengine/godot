/*
 * Copyright 2020 Google Inc.
 *
 * Use of this source code is governed by a BSD-style license that can be
 * found in the LICENSE file.
 */

#ifndef SkTPin_DEFINED
#define SkTPin_DEFINED

#include <algorithm>

/** @return x pinned (clamped) between lo and hi, inclusively.

    Unlike std::clamp(), SkTPin() always returns a value between lo and hi.
    If x is NaN, SkTPin() returns lo but std::clamp() returns NaN.
*/
template <typename T>
static constexpr const T& SkTPin(const T& x, const T& lo, const T& hi) {
    return std::max(lo, std::min(x, hi));
}

#endif
