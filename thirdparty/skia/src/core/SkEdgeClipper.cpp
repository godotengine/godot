/*
 * Copyright 2009 The Android Open Source Project
 *
 * Use of this source code is governed by a BSD-style license that can be
 * found in the LICENSE file.
 */

#include "include/private/SkMacros.h"
#include "src/core/SkEdgeClipper.h"
#include "src/core/SkGeometry.h"
#include "src/core/SkLineClipper.h"

#include <utility>

static bool quick_reject(const SkRect& bounds, const SkRect& clip) {
    return bounds.fTop >= clip.fBottom || bounds.fBottom <= clip.fTop;
}

static inline void clamp_le(SkScalar& value, SkScalar max) {
    if (value > max) {
        value = max;
    }
}

static inline void clamp_ge(SkScalar& value, SkScalar min) {
    if (value < min) {
        value = min;
    }
}

/*  src[] must be monotonic in Y. This routine copies src into dst, and sorts
 it to be increasing in Y. If it had to reverse the order of the points,
 it returns true, otherwise it returns false
 */
static bool sort_increasing_Y(SkPoint dst[], const SkPoint src[], int count) {
    // we need the data to be monotonically increasing in Y
    if (src[0].fY > src[count - 1].fY) {
        for (int i = 0; i < count; i++) {
            dst[i] = src[count - i - 1];
        }
        return true;
    } else {
        memcpy(dst, src, count * sizeof(SkPoint));
        return false;
    }
}

bool SkEdgeClipper::clipLine(SkPoint p0, SkPoint p1, const SkRect& clip) {
    fCurrPoint = fPoints;
    fCurrVerb = fVerbs;

    SkPoint lines[SkLineClipper::kMaxPoints];
    const SkPoint pts[] = { p0, p1 };
    int lineCount = SkLineClipper::ClipLine(pts, clip, lines, fCanCullToTheRight);
    for (int i = 0; i < lineCount; i++) {
        this->appendLine(lines[i], lines[i + 1]);
    }

    *fCurrVerb = SkPath::kDone_Verb;
    fCurrPoint = fPoints;
    fCurrVerb = fVerbs;
    return SkPath::kDone_Verb != fVerbs[0];
}

///////////////////////////////////////////////////////////////////////////////

static bool chopMonoQuadAt(SkScalar c0, SkScalar c1, SkScalar c2,
                           SkScalar target, SkScalar* t) {
    /* Solve F(t) = y where F(t) := [0](1-t)^2 + 2[1]t(1-t) + [2]t^2
     *  We solve for t, using quadratic equation, hence we have to rearrange
     * our cooefficents to look like At^2 + Bt + C
     */
    SkScalar A = c0 - c1 - c1 + c2;
    SkScalar B = 2*(c1 - c0);
    SkScalar C = c0 - target;

    SkScalar roots[2];  // we only expect one, but make room for 2 for safety
    int count = SkFindUnitQuadRoots(A, B, C, roots);
    if (count) {
        *t = roots[0];
        return true;
    }
    return false;
}

static bool chopMonoQuadAtY(SkPoint pts[3], SkScalar y, SkScalar* t) {
    return chopMonoQuadAt(pts[0].fY, pts[1].fY, pts[2].fY, y, t);
}

static bool chopMonoQuadAtX(SkPoint pts[3], SkScalar x, SkScalar* t) {
    return chopMonoQuadAt(pts[0].fX, pts[1].fX, pts[2].fX, x, t);
}

// Modify pts[] in place so that it is clipped in Y to the clip rect
static void chop_quad_in_Y(SkPoint pts[3], const SkRect& clip) {
    SkScalar t;
    SkPoint tmp[5]; // for SkChopQuadAt

    // are we partially above
    if (pts[0].fY < clip.fTop) {
        if (chopMonoQuadAtY(pts, clip.fTop, &t)) {
            // take the 2nd chopped quad
            SkChopQuadAt(pts, tmp, t);
            // clamp to clean up imprecise numerics in the chop
            tmp[2].fY = clip.fTop;
            clamp_ge(tmp[3].fY, clip.fTop);

            pts[0] = tmp[2];
            pts[1] = tmp[3];
        } else {
            // if chopMonoQuadAtY failed, then we may have hit inexact numerics
            // so we just clamp against the top
            for (int i = 0; i < 3; i++) {
                if (pts[i].fY < clip.fTop) {
                    pts[i].fY = clip.fTop;
                }
            }
        }
    }

    // are we partially below
    if (pts[2].fY > clip.fBottom) {
        if (chopMonoQuadAtY(pts, clip.fBottom, &t)) {
            SkChopQuadAt(pts, tmp, t);
            // clamp to clean up imprecise numerics in the chop
            clamp_le(tmp[1].fY, clip.fBottom);
            tmp[2].fY = clip.fBottom;

            pts[1] = tmp[1];
            pts[2] = tmp[2];
        } else {
            // if chopMonoQuadAtY failed, then we may have hit inexact numerics
            // so we just clamp against the bottom
            for (int i = 0; i < 3; i++) {
                if (pts[i].fY > clip.fBottom) {
                    pts[i].fY = clip.fBottom;
                }
            }
        }
    }
}

// srcPts[] must be monotonic in X and Y
void SkEdgeClipper::clipMonoQuad(const SkPoint srcPts[3], const SkRect& clip) {
    SkPoint pts[3];
    bool reverse = sort_increasing_Y(pts, srcPts, 3);

    // are we completely above or below
    if (pts[2].fY <= clip.fTop || pts[0].fY >= clip.fBottom) {
        return;
    }

    // Now chop so that pts is contained within clip in Y
    chop_quad_in_Y(pts, clip);

    if (pts[0].fX > pts[2].fX) {
        using std::swap;
        swap(pts[0], pts[2]);
        reverse = !reverse;
    }
    SkASSERT(pts[0].fX <= pts[1].fX);
    SkASSERT(pts[1].fX <= pts[2].fX);

    // Now chop in X has needed, and record the segments

    if (pts[2].fX <= clip.fLeft) {  // wholly to the left
        this->appendVLine(clip.fLeft, pts[0].fY, pts[2].fY, reverse);
        return;
    }
    if (pts[0].fX >= clip.fRight) {  // wholly to the right
        if (!this->canCullToTheRight()) {
            this->appendVLine(clip.fRight, pts[0].fY, pts[2].fY, reverse);
        }
        return;
    }

    SkScalar t;
    SkPoint tmp[5]; // for SkChopQuadAt

    // are we partially to the left
    if (pts[0].fX < clip.fLeft) {
        if (chopMonoQuadAtX(pts, clip.fLeft, &t)) {
            SkChopQuadAt(pts, tmp, t);
            this->appendVLine(clip.fLeft, tmp[0].fY, tmp[2].fY, reverse);
            // clamp to clean up imprecise numerics in the chop
            tmp[2].fX = clip.fLeft;
            clamp_ge(tmp[3].fX, clip.fLeft);

            pts[0] = tmp[2];
            pts[1] = tmp[3];
        } else {
            // if chopMonoQuadAtY failed, then we may have hit inexact numerics
            // so we just clamp against the left
            this->appendVLine(clip.fLeft, pts[0].fY, pts[2].fY, reverse);
            return;
        }
    }

    // are we partially to the right
    if (pts[2].fX > clip.fRight) {
        if (chopMonoQuadAtX(pts, clip.fRight, &t)) {
            SkChopQuadAt(pts, tmp, t);
            // clamp to clean up imprecise numerics in the chop
            clamp_le(tmp[1].fX, clip.fRight);
            tmp[2].fX = clip.fRight;

            this->appendQuad(tmp, reverse);
            this->appendVLine(clip.fRight, tmp[2].fY, tmp[4].fY, reverse);
        } else {
            // if chopMonoQuadAtY failed, then we may have hit inexact numerics
            // so we just clamp against the right
            pts[1].fX = std::min(pts[1].fX, clip.fRight);
            pts[2].fX = std::min(pts[2].fX, clip.fRight);
            this->appendQuad(pts, reverse);
        }
    } else {    // wholly inside the clip
        this->appendQuad(pts, reverse);
    }
}

bool SkEdgeClipper::clipQuad(const SkPoint srcPts[3], const SkRect& clip) {
    fCurrPoint = fPoints;
    fCurrVerb = fVerbs;

    SkRect  bounds;
    bounds.setBounds(srcPts, 3);

    if (!quick_reject(bounds, clip)) {
        SkPoint monoY[5];
        int countY = SkChopQuadAtYExtrema(srcPts, monoY);
        for (int y = 0; y <= countY; y++) {
            SkPoint monoX[5];
            int countX = SkChopQuadAtXExtrema(&monoY[y * 2], monoX);
            for (int x = 0; x <= countX; x++) {
                this->clipMonoQuad(&monoX[x * 2], clip);
                SkASSERT(fCurrVerb - fVerbs < kMaxVerbs);
                SkASSERT(fCurrPoint - fPoints <= kMaxPoints);
            }
        }
    }

    *fCurrVerb = SkPath::kDone_Verb;
    fCurrPoint = fPoints;
    fCurrVerb = fVerbs;
    return SkPath::kDone_Verb != fVerbs[0];
}

///////////////////////////////////////////////////////////////////////////////

static SkScalar mono_cubic_closestT(const SkScalar src[], SkScalar x) {
    SkScalar t = 0.5f;
    SkScalar lastT;
    SkScalar bestT  SK_INIT_TO_AVOID_WARNING;
    SkScalar step = 0.25f;
    SkScalar D = src[0];
    SkScalar A = src[6] + 3*(src[2] - src[4]) - D;
    SkScalar B = 3*(src[4] - src[2] - src[2] + D);
    SkScalar C = 3*(src[2] - D);
    x -= D;
    SkScalar closest = SK_ScalarMax;
    do {
        SkScalar loc = ((A * t + B) * t + C) * t;
        SkScalar dist = SkScalarAbs(loc - x);
        if (closest > dist) {
            closest = dist;
            bestT = t;
        }
        lastT = t;
        t += loc < x ? step : -step;
        step *= 0.5f;
    } while (closest > 0.25f && lastT != t);
    return bestT;
}

static void chop_mono_cubic_at_y(SkPoint src[4], SkScalar y, SkPoint dst[7]) {
    if (SkChopMonoCubicAtY(src, y, dst)) {
        return;
    }
    SkChopCubicAt(src, dst, mono_cubic_closestT(&src->fY, y));
}

// Modify pts[] in place so that it is clipped in Y to the clip rect
static void chop_cubic_in_Y(SkPoint pts[4], const SkRect& clip) {

    // are we partially above
    if (pts[0].fY < clip.fTop) {
        SkPoint tmp[7];
        chop_mono_cubic_at_y(pts, clip.fTop, tmp);

        /*
         *  For a large range in the points, we can do a poor job of chopping, such that the t
         *  we computed resulted in the lower cubic still being partly above the clip.
         *
         *  If just the first or first 2 Y values are above the fTop, we can just smash them
         *  down. If the first 3 Ys are above fTop, we can't smash all 3, as that can really
         *  distort the cubic. In this case, we take the first output (tmp[3..6] and treat it as
         *  a guess, and re-chop against fTop. Then we fall through to checking if we need to
         *  smash the first 1 or 2 Y values.
         */
        if (tmp[3].fY < clip.fTop && tmp[4].fY < clip.fTop && tmp[5].fY < clip.fTop) {
            SkPoint tmp2[4];
            memcpy(tmp2, &tmp[3].fX, 4 * sizeof(SkPoint));
            chop_mono_cubic_at_y(tmp2, clip.fTop, tmp);
        }

        // tmp[3, 4].fY should all be to the below clip.fTop.
        // Since we can't trust the numerics of the chopper, we force those conditions now
        tmp[3].fY = clip.fTop;
        clamp_ge(tmp[4].fY, clip.fTop);

        pts[0] = tmp[3];
        pts[1] = tmp[4];
        pts[2] = tmp[5];
    }

    // are we partially below
    if (pts[3].fY > clip.fBottom) {
        SkPoint tmp[7];
        chop_mono_cubic_at_y(pts, clip.fBottom, tmp);
        tmp[3].fY = clip.fBottom;
        clamp_le(tmp[2].fY, clip.fBottom);

        pts[1] = tmp[1];
        pts[2] = tmp[2];
        pts[3] = tmp[3];
    }
}

static void chop_mono_cubic_at_x(SkPoint src[4], SkScalar x, SkPoint dst[7]) {
    if (SkChopMonoCubicAtX(src, x, dst)) {
        return;
    }
    SkChopCubicAt(src, dst, mono_cubic_closestT(&src->fX, x));
}

// srcPts[] must be monotonic in X and Y
void SkEdgeClipper::clipMonoCubic(const SkPoint src[4], const SkRect& clip) {
    SkPoint pts[4];
    bool reverse = sort_increasing_Y(pts, src, 4);

    // are we completely above or below
    if (pts[3].fY <= clip.fTop || pts[0].fY >= clip.fBottom) {
        return;
    }

    // Now chop so that pts is contained within clip in Y
    chop_cubic_in_Y(pts, clip);

    if (pts[0].fX > pts[3].fX) {
        using std::swap;
        swap(pts[0], pts[3]);
        swap(pts[1], pts[2]);
        reverse = !reverse;
    }

    // Now chop in X has needed, and record the segments

    if (pts[3].fX <= clip.fLeft) {  // wholly to the left
        this->appendVLine(clip.fLeft, pts[0].fY, pts[3].fY, reverse);
        return;
    }
    if (pts[0].fX >= clip.fRight) {  // wholly to the right
        if (!this->canCullToTheRight()) {
            this->appendVLine(clip.fRight, pts[0].fY, pts[3].fY, reverse);
        }
        return;
    }

    // are we partially to the left
    if (pts[0].fX < clip.fLeft) {
        SkPoint tmp[7];
        chop_mono_cubic_at_x(pts, clip.fLeft, tmp);
        this->appendVLine(clip.fLeft, tmp[0].fY, tmp[3].fY, reverse);

        // tmp[3, 4].fX should all be to the right of clip.fLeft.
        // Since we can't trust the numerics of
        // the chopper, we force those conditions now
        tmp[3].fX = clip.fLeft;
        clamp_ge(tmp[4].fX, clip.fLeft);

        pts[0] = tmp[3];
        pts[1] = tmp[4];
        pts[2] = tmp[5];
    }

    // are we partially to the right
    if (pts[3].fX > clip.fRight) {
        SkPoint tmp[7];
        chop_mono_cubic_at_x(pts, clip.fRight, tmp);
        tmp[3].fX = clip.fRight;
        clamp_le(tmp[2].fX, clip.fRight);

        this->appendCubic(tmp, reverse);
        this->appendVLine(clip.fRight, tmp[3].fY, tmp[6].fY, reverse);
    } else {    // wholly inside the clip
        this->appendCubic(pts, reverse);
    }
}

static SkRect compute_cubic_bounds(const SkPoint pts[4]) {
    SkRect r;
    r.setBounds(pts, 4);
    return r;
}

static bool too_big_for_reliable_float_math(const SkRect& r) {
    // limit set as the largest float value for which we can still reliably compute things like
    // - chopping at XY extrema
    // - chopping at Y or X values for clipping
    //
    // Current value chosen just by experiment. Larger (and still succeeds) is always better.
    //
    const SkScalar limit = 1 << 22;
    return r.fLeft < -limit || r.fTop < -limit || r.fRight > limit || r.fBottom > limit;
}

bool SkEdgeClipper::clipCubic(const SkPoint srcPts[4], const SkRect& clip) {
    fCurrPoint = fPoints;
    fCurrVerb = fVerbs;

    const SkRect bounds = compute_cubic_bounds(srcPts);
    // check if we're clipped out vertically
    if (bounds.fBottom > clip.fTop && bounds.fTop < clip.fBottom) {
        if (too_big_for_reliable_float_math(bounds)) {
            // can't safely clip the cubic, so we give up and draw a line (which we can safely clip)
            //
            // If we rewrote chopcubicat*extrema and chopmonocubic using doubles, we could very
            // likely always handle the cubic safely, but (it seems) at a big loss in speed, so
            // we'd only want to take that alternate impl if needed. Perhaps a TODO to try it.
            //
            return this->clipLine(srcPts[0], srcPts[3], clip);
        } else {
            SkPoint monoY[10];
            int countY = SkChopCubicAtYExtrema(srcPts, monoY);
            for (int y = 0; y <= countY; y++) {
                SkPoint monoX[10];
                int countX = SkChopCubicAtXExtrema(&monoY[y * 3], monoX);
                for (int x = 0; x <= countX; x++) {
                    this->clipMonoCubic(&monoX[x * 3], clip);
                    SkASSERT(fCurrVerb - fVerbs < kMaxVerbs);
                    SkASSERT(fCurrPoint - fPoints <= kMaxPoints);
                }
            }
        }
    }

    *fCurrVerb = SkPath::kDone_Verb;
    fCurrPoint = fPoints;
    fCurrVerb = fVerbs;
    return SkPath::kDone_Verb != fVerbs[0];
}

///////////////////////////////////////////////////////////////////////////////

void SkEdgeClipper::appendLine(SkPoint p0, SkPoint p1) {
    *fCurrVerb++ = SkPath::kLine_Verb;
    fCurrPoint[0] = p0;
    fCurrPoint[1] = p1;
    fCurrPoint += 2;
}

void SkEdgeClipper::appendVLine(SkScalar x, SkScalar y0, SkScalar y1, bool reverse) {
    *fCurrVerb++ = SkPath::kLine_Verb;

    if (reverse) {
        using std::swap;
        swap(y0, y1);
    }
    fCurrPoint[0].set(x, y0);
    fCurrPoint[1].set(x, y1);
    fCurrPoint += 2;
}

void SkEdgeClipper::appendQuad(const SkPoint pts[3], bool reverse) {
    *fCurrVerb++ = SkPath::kQuad_Verb;

    if (reverse) {
        fCurrPoint[0] = pts[2];
        fCurrPoint[2] = pts[0];
    } else {
        fCurrPoint[0] = pts[0];
        fCurrPoint[2] = pts[2];
    }
    fCurrPoint[1] = pts[1];
    fCurrPoint += 3;
}

void SkEdgeClipper::appendCubic(const SkPoint pts[4], bool reverse) {
    *fCurrVerb++ = SkPath::kCubic_Verb;

    if (reverse) {
        for (int i = 0; i < 4; i++) {
            fCurrPoint[i] = pts[3 - i];
        }
    } else {
        memcpy(fCurrPoint, pts, 4 * sizeof(SkPoint));
    }
    fCurrPoint += 4;
}

SkPath::Verb SkEdgeClipper::next(SkPoint pts[]) {
    SkPath::Verb verb = *fCurrVerb;

    switch (verb) {
        case SkPath::kLine_Verb:
            memcpy(pts, fCurrPoint, 2 * sizeof(SkPoint));
            fCurrPoint += 2;
            fCurrVerb += 1;
            break;
        case SkPath::kQuad_Verb:
            memcpy(pts, fCurrPoint, 3 * sizeof(SkPoint));
            fCurrPoint += 3;
            fCurrVerb += 1;
            break;
        case SkPath::kCubic_Verb:
            memcpy(pts, fCurrPoint, 4 * sizeof(SkPoint));
            fCurrPoint += 4;
            fCurrVerb += 1;
            break;
        case SkPath::kDone_Verb:
            break;
        default:
            SkDEBUGFAIL("unexpected verb in quadclippper2 iter");
            break;
    }
    return verb;
}

///////////////////////////////////////////////////////////////////////////////

#ifdef SK_DEBUG
static void assert_monotonic(const SkScalar coord[], int count) {
    if (coord[0] > coord[(count - 1) * 2]) {
        for (int i = 1; i < count; i++) {
            SkASSERT(coord[2 * (i - 1)] >= coord[i * 2]);
        }
    } else if (coord[0] < coord[(count - 1) * 2]) {
        for (int i = 1; i < count; i++) {
            SkASSERT(coord[2 * (i - 1)] <= coord[i * 2]);
        }
    } else {
        for (int i = 1; i < count; i++) {
            SkASSERT(coord[2 * (i - 1)] == coord[i * 2]);
        }
    }
}

void sk_assert_monotonic_y(const SkPoint pts[], int count) {
    if (count > 1) {
        assert_monotonic(&pts[0].fY, count);
    }
}

void sk_assert_monotonic_x(const SkPoint pts[], int count) {
    if (count > 1) {
        assert_monotonic(&pts[0].fX, count);
    }
}
#endif

#include "src/core/SkPathPriv.h"

void SkEdgeClipper::ClipPath(const SkPath& path, const SkRect& clip, bool canCullToTheRight,
                             void (*consume)(SkEdgeClipper*, bool newCtr, void* ctx), void* ctx) {
    SkASSERT(path.isFinite());

    SkAutoConicToQuads quadder;
    const SkScalar conicTol = SK_Scalar1 / 4;

    SkPathEdgeIter iter(path);
    SkEdgeClipper clipper(canCullToTheRight);

    while (auto e = iter.next()) {
        switch (e.fEdge) {
            case SkPathEdgeIter::Edge::kLine:
                if (clipper.clipLine(e.fPts[0], e.fPts[1], clip)) {
                    consume(&clipper, e.fIsNewContour, ctx);
                }
                break;
            case SkPathEdgeIter::Edge::kQuad:
                if (clipper.clipQuad(e.fPts, clip)) {
                    consume(&clipper, e.fIsNewContour, ctx);
                }
                break;
            case SkPathEdgeIter::Edge::kConic: {
                const SkPoint* quadPts = quadder.computeQuads(e.fPts, iter.conicWeight(), conicTol);
                for (int i = 0; i < quadder.countQuads(); ++i) {
                    if (clipper.clipQuad(quadPts, clip)) {
                        consume(&clipper, e.fIsNewContour, ctx);
                    }
                    quadPts += 2;
                }
            } break;
            case SkPathEdgeIter::Edge::kCubic:
                if (clipper.clipCubic(e.fPts, clip)) {
                    consume(&clipper, e.fIsNewContour, ctx);
                }
                break;
        }
    }
}
