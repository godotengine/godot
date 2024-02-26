/*
 * Copyright 2012 Google Inc.
 *
 * Use of this source code is governed by a BSD-style license that can be
 * found in the LICENSE file.
 */
#include "src/pathops/SkAddIntersections.h"
#include "src/pathops/SkOpCoincidence.h"
#include "src/pathops/SkPathOpsBounds.h"

#include <utility>

#if DEBUG_ADD_INTERSECTING_TS

static void debugShowLineIntersection(int pts, const SkIntersectionHelper& wt,
                                      const SkIntersectionHelper& wn, const SkIntersections& i) {
    SkASSERT(i.used() == pts);
    if (!pts) {
        SkDebugf("%s no intersect " LINE_DEBUG_STR " " LINE_DEBUG_STR "\n",
                __FUNCTION__, LINE_DEBUG_DATA(wt.pts()), LINE_DEBUG_DATA(wn.pts()));
        return;
    }
    SkDebugf("%s " T_DEBUG_STR(wtTs, 0) " " LINE_DEBUG_STR " " PT_DEBUG_STR, __FUNCTION__,
            i[0][0], LINE_DEBUG_DATA(wt.pts()), PT_DEBUG_DATA(i, 0));
    if (pts == 2) {
        SkDebugf(" " T_DEBUG_STR(wtTs, 1) " " PT_DEBUG_STR, i[0][1], PT_DEBUG_DATA(i, 1));
    }
    SkDebugf(" wnTs[0]=%g " LINE_DEBUG_STR, i[1][0], LINE_DEBUG_DATA(wn.pts()));
    if (pts == 2) {
        SkDebugf(" " T_DEBUG_STR(wnTs, 1), i[1][1]);
    }
    SkDebugf("\n");
}

static void debugShowQuadLineIntersection(int pts, const SkIntersectionHelper& wt,
                                          const SkIntersectionHelper& wn,
                                          const SkIntersections& i) {
    SkASSERT(i.used() == pts);
    if (!pts) {
        SkDebugf("%s no intersect " QUAD_DEBUG_STR " " LINE_DEBUG_STR "\n",
                __FUNCTION__, QUAD_DEBUG_DATA(wt.pts()), LINE_DEBUG_DATA(wn.pts()));
        return;
    }
    SkDebugf("%s " T_DEBUG_STR(wtTs, 0) " " QUAD_DEBUG_STR " " PT_DEBUG_STR, __FUNCTION__,
            i[0][0], QUAD_DEBUG_DATA(wt.pts()), PT_DEBUG_DATA(i, 0));
    for (int n = 1; n < pts; ++n) {
        SkDebugf(" " TX_DEBUG_STR(wtTs) " " PT_DEBUG_STR, n, i[0][n], PT_DEBUG_DATA(i, n));
    }
    SkDebugf(" wnTs[0]=%g " LINE_DEBUG_STR, i[1][0], LINE_DEBUG_DATA(wn.pts()));
    for (int n = 1; n < pts; ++n) {
        SkDebugf(" " TX_DEBUG_STR(wnTs), n, i[1][n]);
    }
    SkDebugf("\n");
}

static void debugShowQuadIntersection(int pts, const SkIntersectionHelper& wt,
        const SkIntersectionHelper& wn, const SkIntersections& i) {
    SkASSERT(i.used() == pts);
    if (!pts) {
        SkDebugf("%s no intersect " QUAD_DEBUG_STR " " QUAD_DEBUG_STR "\n",
                __FUNCTION__, QUAD_DEBUG_DATA(wt.pts()), QUAD_DEBUG_DATA(wn.pts()));
        return;
    }
    SkDebugf("%s " T_DEBUG_STR(wtTs, 0) " " QUAD_DEBUG_STR " " PT_DEBUG_STR, __FUNCTION__,
            i[0][0], QUAD_DEBUG_DATA(wt.pts()), PT_DEBUG_DATA(i, 0));
    for (int n = 1; n < pts; ++n) {
        SkDebugf(" " TX_DEBUG_STR(wtTs) " " PT_DEBUG_STR, n, i[0][n], PT_DEBUG_DATA(i, n));
    }
    SkDebugf(" wnTs[0]=%g " QUAD_DEBUG_STR, i[1][0], QUAD_DEBUG_DATA(wn.pts()));
    for (int n = 1; n < pts; ++n) {
        SkDebugf(" " TX_DEBUG_STR(wnTs), n, i[1][n]);
    }
    SkDebugf("\n");
}

static void debugShowConicLineIntersection(int pts, const SkIntersectionHelper& wt,
        const SkIntersectionHelper& wn, const SkIntersections& i) {
    SkASSERT(i.used() == pts);
    if (!pts) {
        SkDebugf("%s no intersect " CONIC_DEBUG_STR " " LINE_DEBUG_STR "\n",
                __FUNCTION__, CONIC_DEBUG_DATA(wt.pts(), wt.weight()), LINE_DEBUG_DATA(wn.pts()));
        return;
    }
    SkDebugf("%s " T_DEBUG_STR(wtTs, 0) " " CONIC_DEBUG_STR " " PT_DEBUG_STR, __FUNCTION__,
            i[0][0], CONIC_DEBUG_DATA(wt.pts(), wt.weight()), PT_DEBUG_DATA(i, 0));
    for (int n = 1; n < pts; ++n) {
        SkDebugf(" " TX_DEBUG_STR(wtTs) " " PT_DEBUG_STR, n, i[0][n], PT_DEBUG_DATA(i, n));
    }
    SkDebugf(" wnTs[0]=%g " LINE_DEBUG_STR, i[1][0], LINE_DEBUG_DATA(wn.pts()));
    for (int n = 1; n < pts; ++n) {
        SkDebugf(" " TX_DEBUG_STR(wnTs), n, i[1][n]);
    }
    SkDebugf("\n");
}

static void debugShowConicQuadIntersection(int pts, const SkIntersectionHelper& wt,
        const SkIntersectionHelper& wn, const SkIntersections& i) {
    SkASSERT(i.used() == pts);
    if (!pts) {
        SkDebugf("%s no intersect " CONIC_DEBUG_STR " " QUAD_DEBUG_STR "\n",
                __FUNCTION__, CONIC_DEBUG_DATA(wt.pts(), wt.weight()), QUAD_DEBUG_DATA(wn.pts()));
        return;
    }
    SkDebugf("%s " T_DEBUG_STR(wtTs, 0) " " CONIC_DEBUG_STR " " PT_DEBUG_STR, __FUNCTION__,
            i[0][0], CONIC_DEBUG_DATA(wt.pts(), wt.weight()), PT_DEBUG_DATA(i, 0));
    for (int n = 1; n < pts; ++n) {
        SkDebugf(" " TX_DEBUG_STR(wtTs) " " PT_DEBUG_STR, n, i[0][n], PT_DEBUG_DATA(i, n));
    }
    SkDebugf(" wnTs[0]=%g " QUAD_DEBUG_STR, i[1][0], QUAD_DEBUG_DATA(wn.pts()));
    for (int n = 1; n < pts; ++n) {
        SkDebugf(" " TX_DEBUG_STR(wnTs), n, i[1][n]);
    }
    SkDebugf("\n");
}

static void debugShowConicIntersection(int pts, const SkIntersectionHelper& wt,
        const SkIntersectionHelper& wn, const SkIntersections& i) {
    SkASSERT(i.used() == pts);
    if (!pts) {
        SkDebugf("%s no intersect " CONIC_DEBUG_STR " " CONIC_DEBUG_STR "\n",
                __FUNCTION__, CONIC_DEBUG_DATA(wt.pts(), wt.weight()),
                CONIC_DEBUG_DATA(wn.pts(), wn.weight()));
        return;
    }
    SkDebugf("%s " T_DEBUG_STR(wtTs, 0) " " CONIC_DEBUG_STR " " PT_DEBUG_STR, __FUNCTION__,
            i[0][0], CONIC_DEBUG_DATA(wt.pts(), wt.weight()), PT_DEBUG_DATA(i, 0));
    for (int n = 1; n < pts; ++n) {
        SkDebugf(" " TX_DEBUG_STR(wtTs) " " PT_DEBUG_STR, n, i[0][n], PT_DEBUG_DATA(i, n));
    }
    SkDebugf(" wnTs[0]=%g " CONIC_DEBUG_STR, i[1][0], CONIC_DEBUG_DATA(wn.pts(), wn.weight()));
    for (int n = 1; n < pts; ++n) {
        SkDebugf(" " TX_DEBUG_STR(wnTs), n, i[1][n]);
    }
    SkDebugf("\n");
}

static void debugShowCubicLineIntersection(int pts, const SkIntersectionHelper& wt,
        const SkIntersectionHelper& wn, const SkIntersections& i) {
    SkASSERT(i.used() == pts);
    if (!pts) {
        SkDebugf("%s no intersect " CUBIC_DEBUG_STR " " LINE_DEBUG_STR "\n",
                __FUNCTION__, CUBIC_DEBUG_DATA(wt.pts()), LINE_DEBUG_DATA(wn.pts()));
        return;
    }
    SkDebugf("%s " T_DEBUG_STR(wtTs, 0) " " CUBIC_DEBUG_STR " " PT_DEBUG_STR, __FUNCTION__,
            i[0][0], CUBIC_DEBUG_DATA(wt.pts()), PT_DEBUG_DATA(i, 0));
    for (int n = 1; n < pts; ++n) {
        SkDebugf(" " TX_DEBUG_STR(wtTs) " " PT_DEBUG_STR, n, i[0][n], PT_DEBUG_DATA(i, n));
    }
    SkDebugf(" wnTs[0]=%g " LINE_DEBUG_STR, i[1][0], LINE_DEBUG_DATA(wn.pts()));
    for (int n = 1; n < pts; ++n) {
        SkDebugf(" " TX_DEBUG_STR(wnTs), n, i[1][n]);
    }
    SkDebugf("\n");
}

static void debugShowCubicQuadIntersection(int pts, const SkIntersectionHelper& wt,
        const SkIntersectionHelper& wn, const SkIntersections& i) {
    SkASSERT(i.used() == pts);
    if (!pts) {
        SkDebugf("%s no intersect " CUBIC_DEBUG_STR " " QUAD_DEBUG_STR "\n",
                __FUNCTION__, CUBIC_DEBUG_DATA(wt.pts()), QUAD_DEBUG_DATA(wn.pts()));
        return;
    }
    SkDebugf("%s " T_DEBUG_STR(wtTs, 0) " " CUBIC_DEBUG_STR " " PT_DEBUG_STR, __FUNCTION__,
            i[0][0], CUBIC_DEBUG_DATA(wt.pts()), PT_DEBUG_DATA(i, 0));
    for (int n = 1; n < pts; ++n) {
        SkDebugf(" " TX_DEBUG_STR(wtTs) " " PT_DEBUG_STR, n, i[0][n], PT_DEBUG_DATA(i, n));
    }
    SkDebugf(" wnTs[0]=%g " QUAD_DEBUG_STR, i[1][0], QUAD_DEBUG_DATA(wn.pts()));
    for (int n = 1; n < pts; ++n) {
        SkDebugf(" " TX_DEBUG_STR(wnTs), n, i[1][n]);
    }
    SkDebugf("\n");
}

static void debugShowCubicConicIntersection(int pts, const SkIntersectionHelper& wt,
        const SkIntersectionHelper& wn, const SkIntersections& i) {
    SkASSERT(i.used() == pts);
    if (!pts) {
        SkDebugf("%s no intersect " CUBIC_DEBUG_STR " " CONIC_DEBUG_STR "\n",
                __FUNCTION__, CUBIC_DEBUG_DATA(wt.pts()), CONIC_DEBUG_DATA(wn.pts(), wn.weight()));
        return;
    }
    SkDebugf("%s " T_DEBUG_STR(wtTs, 0) " " CUBIC_DEBUG_STR " " PT_DEBUG_STR, __FUNCTION__,
            i[0][0], CUBIC_DEBUG_DATA(wt.pts()), PT_DEBUG_DATA(i, 0));
    for (int n = 1; n < pts; ++n) {
        SkDebugf(" " TX_DEBUG_STR(wtTs) " " PT_DEBUG_STR, n, i[0][n], PT_DEBUG_DATA(i, n));
    }
    SkDebugf(" wnTs[0]=%g " CONIC_DEBUG_STR, i[1][0], CONIC_DEBUG_DATA(wn.pts(), wn.weight()));
    for (int n = 1; n < pts; ++n) {
        SkDebugf(" " TX_DEBUG_STR(wnTs), n, i[1][n]);
    }
    SkDebugf("\n");
}

static void debugShowCubicIntersection(int pts, const SkIntersectionHelper& wt,
        const SkIntersectionHelper& wn, const SkIntersections& i) {
    SkASSERT(i.used() == pts);
    if (!pts) {
        SkDebugf("%s no intersect " CUBIC_DEBUG_STR " " CUBIC_DEBUG_STR "\n",
                __FUNCTION__, CUBIC_DEBUG_DATA(wt.pts()), CUBIC_DEBUG_DATA(wn.pts()));
        return;
    }
    SkDebugf("%s " T_DEBUG_STR(wtTs, 0) " " CUBIC_DEBUG_STR " " PT_DEBUG_STR, __FUNCTION__,
            i[0][0], CUBIC_DEBUG_DATA(wt.pts()), PT_DEBUG_DATA(i, 0));
    for (int n = 1; n < pts; ++n) {
        SkDebugf(" " TX_DEBUG_STR(wtTs) " " PT_DEBUG_STR, n, i[0][n], PT_DEBUG_DATA(i, n));
    }
    SkDebugf(" wnTs[0]=%g " CUBIC_DEBUG_STR, i[1][0], CUBIC_DEBUG_DATA(wn.pts()));
    for (int n = 1; n < pts; ++n) {
        SkDebugf(" " TX_DEBUG_STR(wnTs), n, i[1][n]);
    }
    SkDebugf("\n");
}

#else
static void debugShowLineIntersection(int , const SkIntersectionHelper& ,
        const SkIntersectionHelper& , const SkIntersections& ) {
}

static void debugShowQuadLineIntersection(int , const SkIntersectionHelper& ,
        const SkIntersectionHelper& , const SkIntersections& ) {
}

static void debugShowQuadIntersection(int , const SkIntersectionHelper& ,
        const SkIntersectionHelper& , const SkIntersections& ) {
}

static void debugShowConicLineIntersection(int , const SkIntersectionHelper& ,
        const SkIntersectionHelper& , const SkIntersections& ) {
}

static void debugShowConicQuadIntersection(int , const SkIntersectionHelper& ,
        const SkIntersectionHelper& , const SkIntersections& ) {
}

static void debugShowConicIntersection(int , const SkIntersectionHelper& ,
        const SkIntersectionHelper& , const SkIntersections& ) {
}

static void debugShowCubicLineIntersection(int , const SkIntersectionHelper& ,
        const SkIntersectionHelper& , const SkIntersections& ) {
}

static void debugShowCubicQuadIntersection(int , const SkIntersectionHelper& ,
        const SkIntersectionHelper& , const SkIntersections& ) {
}

static void debugShowCubicConicIntersection(int , const SkIntersectionHelper& ,
        const SkIntersectionHelper& , const SkIntersections& ) {
}

static void debugShowCubicIntersection(int , const SkIntersectionHelper& ,
        const SkIntersectionHelper& , const SkIntersections& ) {
}
#endif

bool AddIntersectTs(SkOpContour* test, SkOpContour* next, SkOpCoincidence* coincidence) {
    if (test != next) {
        if (AlmostLessUlps(test->bounds().fBottom, next->bounds().fTop)) {
            return false;
        }
        // OPTIMIZATION: outset contour bounds a smidgen instead?
        if (!SkPathOpsBounds::Intersects(test->bounds(), next->bounds())) {
            return true;
        }
    }
    SkIntersectionHelper wt;
    wt.init(test);
    do {
        SkIntersectionHelper wn;
        wn.init(next);
        test->debugValidate();
        next->debugValidate();
        if (test == next && !wn.startAfter(wt)) {
            continue;
        }
        do {
            if (!SkPathOpsBounds::Intersects(wt.bounds(), wn.bounds())) {
                continue;
            }
            int pts = 0;
            SkIntersections ts { SkDEBUGCODE(test->globalState()) };
            bool swap = false;
            SkDQuad quad1, quad2;
            SkDConic conic1, conic2;
            SkDCubic cubic1, cubic2;
            switch (wt.segmentType()) {
                case SkIntersectionHelper::kHorizontalLine_Segment:
                    swap = true;
                    switch (wn.segmentType()) {
                        case SkIntersectionHelper::kHorizontalLine_Segment:
                        case SkIntersectionHelper::kVerticalLine_Segment:
                        case SkIntersectionHelper::kLine_Segment:
                            pts = ts.lineHorizontal(wn.pts(), wt.left(),
                                    wt.right(), wt.y(), wt.xFlipped());
                            debugShowLineIntersection(pts, wn, wt, ts);
                            break;
                        case SkIntersectionHelper::kQuad_Segment:
                            pts = ts.quadHorizontal(wn.pts(), wt.left(),
                                    wt.right(), wt.y(), wt.xFlipped());
                            debugShowQuadLineIntersection(pts, wn, wt, ts);
                            break;
                        case SkIntersectionHelper::kConic_Segment:
                            pts = ts.conicHorizontal(wn.pts(), wn.weight(), wt.left(),
                                    wt.right(), wt.y(), wt.xFlipped());
                            debugShowConicLineIntersection(pts, wn, wt, ts);
                            break;
                        case SkIntersectionHelper::kCubic_Segment:
                            pts = ts.cubicHorizontal(wn.pts(), wt.left(),
                                    wt.right(), wt.y(), wt.xFlipped());
                            debugShowCubicLineIntersection(pts, wn, wt, ts);
                            break;
                        default:
                            SkASSERT(0);
                    }
                    break;
                case SkIntersectionHelper::kVerticalLine_Segment:
                    swap = true;
                    switch (wn.segmentType()) {
                        case SkIntersectionHelper::kHorizontalLine_Segment:
                        case SkIntersectionHelper::kVerticalLine_Segment:
                        case SkIntersectionHelper::kLine_Segment: {
                            pts = ts.lineVertical(wn.pts(), wt.top(),
                                    wt.bottom(), wt.x(), wt.yFlipped());
                            debugShowLineIntersection(pts, wn, wt, ts);
                            break;
                        }
                        case SkIntersectionHelper::kQuad_Segment: {
                            pts = ts.quadVertical(wn.pts(), wt.top(),
                                    wt.bottom(), wt.x(), wt.yFlipped());
                            debugShowQuadLineIntersection(pts, wn, wt, ts);
                            break;
                        }
                        case SkIntersectionHelper::kConic_Segment: {
                            pts = ts.conicVertical(wn.pts(), wn.weight(), wt.top(),
                                    wt.bottom(), wt.x(), wt.yFlipped());
                            debugShowConicLineIntersection(pts, wn, wt, ts);
                            break;
                        }
                        case SkIntersectionHelper::kCubic_Segment: {
                            pts = ts.cubicVertical(wn.pts(), wt.top(),
                                    wt.bottom(), wt.x(), wt.yFlipped());
                            debugShowCubicLineIntersection(pts, wn, wt, ts);
                            break;
                        }
                        default:
                            SkASSERT(0);
                    }
                    break;
                case SkIntersectionHelper::kLine_Segment:
                    switch (wn.segmentType()) {
                        case SkIntersectionHelper::kHorizontalLine_Segment:
                            pts = ts.lineHorizontal(wt.pts(), wn.left(),
                                    wn.right(), wn.y(), wn.xFlipped());
                            debugShowLineIntersection(pts, wt, wn, ts);
                            break;
                        case SkIntersectionHelper::kVerticalLine_Segment:
                            pts = ts.lineVertical(wt.pts(), wn.top(),
                                    wn.bottom(), wn.x(), wn.yFlipped());
                            debugShowLineIntersection(pts, wt, wn, ts);
                            break;
                        case SkIntersectionHelper::kLine_Segment:
                            pts = ts.lineLine(wt.pts(), wn.pts());
                            debugShowLineIntersection(pts, wt, wn, ts);
                            break;
                        case SkIntersectionHelper::kQuad_Segment:
                            swap = true;
                            pts = ts.quadLine(wn.pts(), wt.pts());
                            debugShowQuadLineIntersection(pts, wn, wt, ts);
                            break;
                        case SkIntersectionHelper::kConic_Segment:
                            swap = true;
                            pts = ts.conicLine(wn.pts(), wn.weight(), wt.pts());
                            debugShowConicLineIntersection(pts, wn, wt, ts);
                            break;
                        case SkIntersectionHelper::kCubic_Segment:
                            swap = true;
                            pts = ts.cubicLine(wn.pts(), wt.pts());
                            debugShowCubicLineIntersection(pts, wn, wt, ts);
                            break;
                        default:
                            SkASSERT(0);
                    }
                    break;
                case SkIntersectionHelper::kQuad_Segment:
                    switch (wn.segmentType()) {
                        case SkIntersectionHelper::kHorizontalLine_Segment:
                            pts = ts.quadHorizontal(wt.pts(), wn.left(),
                                    wn.right(), wn.y(), wn.xFlipped());
                            debugShowQuadLineIntersection(pts, wt, wn, ts);
                            break;
                        case SkIntersectionHelper::kVerticalLine_Segment:
                            pts = ts.quadVertical(wt.pts(), wn.top(),
                                    wn.bottom(), wn.x(), wn.yFlipped());
                            debugShowQuadLineIntersection(pts, wt, wn, ts);
                            break;
                        case SkIntersectionHelper::kLine_Segment:
                            pts = ts.quadLine(wt.pts(), wn.pts());
                            debugShowQuadLineIntersection(pts, wt, wn, ts);
                            break;
                        case SkIntersectionHelper::kQuad_Segment: {
                            pts = ts.intersect(quad1.set(wt.pts()), quad2.set(wn.pts()));
                            debugShowQuadIntersection(pts, wt, wn, ts);
                            break;
                        }
                        case SkIntersectionHelper::kConic_Segment: {
                            swap = true;
                            pts = ts.intersect(conic2.set(wn.pts(), wn.weight()),
                                    quad1.set(wt.pts()));
                            debugShowConicQuadIntersection(pts, wn, wt, ts);
                            break;
                        }
                        case SkIntersectionHelper::kCubic_Segment: {
                            swap = true;
                            pts = ts.intersect(cubic2.set(wn.pts()), quad1.set(wt.pts()));
                            debugShowCubicQuadIntersection(pts, wn, wt, ts);
                            break;
                        }
                        default:
                            SkASSERT(0);
                    }
                    break;
                case SkIntersectionHelper::kConic_Segment:
                    switch (wn.segmentType()) {
                        case SkIntersectionHelper::kHorizontalLine_Segment:
                            pts = ts.conicHorizontal(wt.pts(), wt.weight(), wn.left(),
                                    wn.right(), wn.y(), wn.xFlipped());
                            debugShowConicLineIntersection(pts, wt, wn, ts);
                            break;
                        case SkIntersectionHelper::kVerticalLine_Segment:
                            pts = ts.conicVertical(wt.pts(), wt.weight(), wn.top(),
                                    wn.bottom(), wn.x(), wn.yFlipped());
                            debugShowConicLineIntersection(pts, wt, wn, ts);
                            break;
                        case SkIntersectionHelper::kLine_Segment:
                            pts = ts.conicLine(wt.pts(), wt.weight(), wn.pts());
                            debugShowConicLineIntersection(pts, wt, wn, ts);
                            break;
                        case SkIntersectionHelper::kQuad_Segment: {
                            pts = ts.intersect(conic1.set(wt.pts(), wt.weight()),
                                    quad2.set(wn.pts()));
                            debugShowConicQuadIntersection(pts, wt, wn, ts);
                            break;
                        }
                        case SkIntersectionHelper::kConic_Segment: {
                            pts = ts.intersect(conic1.set(wt.pts(), wt.weight()),
                                    conic2.set(wn.pts(), wn.weight()));
                            debugShowConicIntersection(pts, wt, wn, ts);
                            break;
                        }
                        case SkIntersectionHelper::kCubic_Segment: {
                            swap = true;
                            pts = ts.intersect(cubic2.set(wn.pts()
                                    SkDEBUGPARAMS(ts.globalState())),
                                    conic1.set(wt.pts(), wt.weight()
                                    SkDEBUGPARAMS(ts.globalState())));
                            debugShowCubicConicIntersection(pts, wn, wt, ts);
                            break;
                        }
                    }
                    break;
                case SkIntersectionHelper::kCubic_Segment:
                    switch (wn.segmentType()) {
                        case SkIntersectionHelper::kHorizontalLine_Segment:
                            pts = ts.cubicHorizontal(wt.pts(), wn.left(),
                                    wn.right(), wn.y(), wn.xFlipped());
                            debugShowCubicLineIntersection(pts, wt, wn, ts);
                            break;
                        case SkIntersectionHelper::kVerticalLine_Segment:
                            pts = ts.cubicVertical(wt.pts(), wn.top(),
                                    wn.bottom(), wn.x(), wn.yFlipped());
                            debugShowCubicLineIntersection(pts, wt, wn, ts);
                            break;
                        case SkIntersectionHelper::kLine_Segment:
                            pts = ts.cubicLine(wt.pts(), wn.pts());
                            debugShowCubicLineIntersection(pts, wt, wn, ts);
                            break;
                        case SkIntersectionHelper::kQuad_Segment: {
                            pts = ts.intersect(cubic1.set(wt.pts()), quad2.set(wn.pts()));
                            debugShowCubicQuadIntersection(pts, wt, wn, ts);
                            break;
                        }
                        case SkIntersectionHelper::kConic_Segment: {
                            pts = ts.intersect(cubic1.set(wt.pts()
                                    SkDEBUGPARAMS(ts.globalState())),
                                    conic2.set(wn.pts(), wn.weight()
                                    SkDEBUGPARAMS(ts.globalState())));
                            debugShowCubicConicIntersection(pts, wt, wn, ts);
                            break;
                        }
                        case SkIntersectionHelper::kCubic_Segment: {
                            pts = ts.intersect(cubic1.set(wt.pts()), cubic2.set(wn.pts()));
                            debugShowCubicIntersection(pts, wt, wn, ts);
                            break;
                        }
                        default:
                            SkASSERT(0);
                    }
                    break;
                default:
                    SkASSERT(0);
            }
#if DEBUG_T_SECT_LOOP_COUNT
            test->globalState()->debugAddLoopCount(&ts, wt, wn);
#endif
            int coinIndex = -1;
            SkOpPtT* coinPtT[2];
            for (int pt = 0; pt < pts; ++pt) {
                SkASSERT(ts[0][pt] >= 0 && ts[0][pt] <= 1);
                SkASSERT(ts[1][pt] >= 0 && ts[1][pt] <= 1);
                wt.segment()->debugValidate();
                // if t value is used to compute pt in addT, error may creep in and
                // rect intersections may result in non-rects. if pt value from intersection
                // is passed in, current tests break. As a workaround, pass in pt
                // value from intersection only if pt.x and pt.y is integral
                SkPoint iPt = ts.pt(pt).asSkPoint();
                bool iPtIsIntegral = iPt.fX == floor(iPt.fX) && iPt.fY == floor(iPt.fY);
                SkOpPtT* testTAt = iPtIsIntegral ? wt.segment()->addT(ts[swap][pt], iPt)
                        : wt.segment()->addT(ts[swap][pt]);
                wn.segment()->debugValidate();
                SkOpPtT* nextTAt = iPtIsIntegral ? wn.segment()->addT(ts[!swap][pt], iPt)
                        : wn.segment()->addT(ts[!swap][pt]);
                if (!testTAt->contains(nextTAt)) {
                    SkOpPtT* oppPrev = testTAt->oppPrev(nextTAt);  //  Returns nullptr if pair
                    if (oppPrev) {                                 //  already share a pt-t loop.
                        testTAt->span()->mergeMatches(nextTAt->span());
                        testTAt->addOpp(nextTAt, oppPrev);
                    }
                    if (testTAt->fPt != nextTAt->fPt) {
                        testTAt->span()->unaligned();
                        nextTAt->span()->unaligned();
                    }
                    wt.segment()->debugValidate();
                    wn.segment()->debugValidate();
                }
                if (!ts.isCoincident(pt)) {
                    continue;
                }
                if (coinIndex < 0) {
                    coinPtT[0] = testTAt;
                    coinPtT[1] = nextTAt;
                    coinIndex = pt;
                    continue;
                }
                if (coinPtT[0]->span() == testTAt->span()) {
                    coinIndex = -1;
                    continue;
                }
                if (coinPtT[1]->span() == nextTAt->span()) {
                    coinIndex = -1;  // coincidence span collapsed
                    continue;
                }
                if (swap) {
                    using std::swap;
                    swap(coinPtT[0], coinPtT[1]);
                    swap(testTAt, nextTAt);
                }
                SkASSERT(coincidence->globalState()->debugSkipAssert()
                        || coinPtT[0]->span()->t() < testTAt->span()->t());
                if (coinPtT[0]->span()->deleted()) {
                    coinIndex = -1;
                    continue;
                }
                if (testTAt->span()->deleted()) {
                    coinIndex = -1;
                    continue;
                }
                coincidence->add(coinPtT[0], testTAt, coinPtT[1], nextTAt);
                wt.segment()->debugValidate();
                wn.segment()->debugValidate();
                coinIndex = -1;
            }
            SkOPOBJASSERT(coincidence, coinIndex < 0);  // expect coincidence to be paired
        } while (wn.advance());
    } while (wt.advance());
    return true;
}
