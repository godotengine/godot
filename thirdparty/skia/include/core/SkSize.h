/*
 * Copyright 2011 Google Inc.
 *
 * Use of this source code is governed by a BSD-style license that can be
 * found in the LICENSE file.
 */

#ifndef SkSize_DEFINED
#define SkSize_DEFINED

#include "include/core/SkScalar.h"

struct SkISize {
    int32_t fWidth;
    int32_t fHeight;

    static constexpr SkISize Make(int32_t w, int32_t h) { return {w, h}; }

    static constexpr SkISize MakeEmpty() { return {0, 0}; }

    void set(int32_t w, int32_t h) { *this = SkISize{w, h}; }

    /** Returns true iff fWidth == 0 && fHeight == 0
     */
    bool isZero() const { return 0 == fWidth && 0 == fHeight; }

    /** Returns true if either width or height are <= 0 */
    bool isEmpty() const { return fWidth <= 0 || fHeight <= 0; }

    /** Set the width and height to 0 */
    void setEmpty() { fWidth = fHeight = 0; }

    constexpr int32_t width() const { return fWidth; }
    constexpr int32_t height() const { return fHeight; }

    constexpr int64_t area() const { return fWidth * fHeight; }

    bool equals(int32_t w, int32_t h) const { return fWidth == w && fHeight == h; }
};

static inline bool operator==(const SkISize& a, const SkISize& b) {
    return a.fWidth == b.fWidth && a.fHeight == b.fHeight;
}

static inline bool operator!=(const SkISize& a, const SkISize& b) { return !(a == b); }

///////////////////////////////////////////////////////////////////////////////

struct SkSize {
    SkScalar fWidth;
    SkScalar fHeight;

    static SkSize Make(SkScalar w, SkScalar h) { return {w, h}; }

    static SkSize Make(const SkISize& src) {
        return {SkIntToScalar(src.width()), SkIntToScalar(src.height())};
    }

    static SkSize MakeEmpty() { return {0, 0}; }

    void set(SkScalar w, SkScalar h) { *this = SkSize{w, h}; }

    /** Returns true iff fWidth == 0 && fHeight == 0
     */
    bool isZero() const { return 0 == fWidth && 0 == fHeight; }

    /** Returns true if either width or height are <= 0 */
    bool isEmpty() const { return fWidth <= 0 || fHeight <= 0; }

    /** Set the width and height to 0 */
    void setEmpty() { *this = SkSize{0, 0}; }

    SkScalar width() const { return fWidth; }
    SkScalar height() const { return fHeight; }

    bool equals(SkScalar w, SkScalar h) const { return fWidth == w && fHeight == h; }

    SkISize toRound() const { return {SkScalarRoundToInt(fWidth), SkScalarRoundToInt(fHeight)}; }

    SkISize toCeil() const { return {SkScalarCeilToInt(fWidth), SkScalarCeilToInt(fHeight)}; }

    SkISize toFloor() const { return {SkScalarFloorToInt(fWidth), SkScalarFloorToInt(fHeight)}; }
};

static inline bool operator==(const SkSize& a, const SkSize& b) {
    return a.fWidth == b.fWidth && a.fHeight == b.fHeight;
}

static inline bool operator!=(const SkSize& a, const SkSize& b) { return !(a == b); }
#endif
