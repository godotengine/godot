/*
 * Copyright 2012 Google Inc.
 *
 * Use of this source code is governed by a BSD-style license that can be
 * found in the LICENSE file.
 */
#include "src/core/SkGeometry.h"
#include "src/pathops/SkReduceOrder.h"

int SkReduceOrder::reduce(const SkDLine& line) {
    fLine[0] = line[0];
    int different = line[0] != line[1];
    fLine[1] = line[different];
    return 1 + different;
}

static int coincident_line(const SkDQuad& quad, SkDQuad& reduction) {
    reduction[0] = reduction[1] = quad[0];
    return 1;
}

static int reductionLineCount(const SkDQuad& reduction) {
    return 1 + !reduction[0].approximatelyEqual(reduction[1]);
}

static int vertical_line(const SkDQuad& quad, SkDQuad& reduction) {
    reduction[0] = quad[0];
    reduction[1] = quad[2];
    return reductionLineCount(reduction);
}

static int horizontal_line(const SkDQuad& quad, SkDQuad& reduction) {
    reduction[0] = quad[0];
    reduction[1] = quad[2];
    return reductionLineCount(reduction);
}

static int check_linear(const SkDQuad& quad,
        int minX, int maxX, int minY, int maxY, SkDQuad& reduction) {
    if (!quad.isLinear(0, 2)) {
        return 0;
    }
    // four are colinear: return line formed by outside
    reduction[0] = quad[0];
    reduction[1] = quad[2];
    return reductionLineCount(reduction);
}

// reduce to a quadratic or smaller
// look for identical points
// look for all four points in a line
    // note that three points in a line doesn't simplify a cubic
// look for approximation with single quadratic
    // save approximation with multiple quadratics for later
int SkReduceOrder::reduce(const SkDQuad& quad) {
    int index, minX, maxX, minY, maxY;
    int minXSet, minYSet;
    minX = maxX = minY = maxY = 0;
    minXSet = minYSet = 0;
    for (index = 1; index < 3; ++index) {
        if (quad[minX].fX > quad[index].fX) {
            minX = index;
        }
        if (quad[minY].fY > quad[index].fY) {
            minY = index;
        }
        if (quad[maxX].fX < quad[index].fX) {
            maxX = index;
        }
        if (quad[maxY].fY < quad[index].fY) {
            maxY = index;
        }
    }
    for (index = 0; index < 3; ++index) {
        if (AlmostEqualUlps(quad[index].fX, quad[minX].fX)) {
            minXSet |= 1 << index;
        }
        if (AlmostEqualUlps(quad[index].fY, quad[minY].fY)) {
            minYSet |= 1 << index;
        }
    }
    if ((minXSet & 0x05) == 0x5 && (minYSet & 0x05) == 0x5) { // test for degenerate
        // this quad starts and ends at the same place, so never contributes
        // to the fill
        return coincident_line(quad, fQuad);
    }
    if (minXSet == 0x7) {  // test for vertical line
        return vertical_line(quad, fQuad);
    }
    if (minYSet == 0x7) {  // test for horizontal line
        return horizontal_line(quad, fQuad);
    }
    int result = check_linear(quad, minX, maxX, minY, maxY, fQuad);
    if (result) {
        return result;
    }
    fQuad = quad;
    return 3;
}

////////////////////////////////////////////////////////////////////////////////////

static int coincident_line(const SkDCubic& cubic, SkDCubic& reduction) {
    reduction[0] = reduction[1] = cubic[0];
    return 1;
}

static int reductionLineCount(const SkDCubic& reduction) {
    return 1 + !reduction[0].approximatelyEqual(reduction[1]);
}

static int vertical_line(const SkDCubic& cubic, SkDCubic& reduction) {
    reduction[0] = cubic[0];
    reduction[1] = cubic[3];
    return reductionLineCount(reduction);
}

static int horizontal_line(const SkDCubic& cubic, SkDCubic& reduction) {
    reduction[0] = cubic[0];
    reduction[1] = cubic[3];
    return reductionLineCount(reduction);
}

// check to see if it is a quadratic or a line
static int check_quadratic(const SkDCubic& cubic, SkDCubic& reduction) {
    double dx10 = cubic[1].fX - cubic[0].fX;
    double dx23 = cubic[2].fX - cubic[3].fX;
    double midX = cubic[0].fX + dx10 * 3 / 2;
    double sideAx = midX - cubic[3].fX;
    double sideBx = dx23 * 3 / 2;
    if (approximately_zero(sideAx) ? !approximately_equal(sideAx, sideBx)
            : !AlmostEqualUlps_Pin(sideAx, sideBx)) {
        return 0;
    }
    double dy10 = cubic[1].fY - cubic[0].fY;
    double dy23 = cubic[2].fY - cubic[3].fY;
    double midY = cubic[0].fY + dy10 * 3 / 2;
    double sideAy = midY - cubic[3].fY;
    double sideBy = dy23 * 3 / 2;
    if (approximately_zero(sideAy) ? !approximately_equal(sideAy, sideBy)
            : !AlmostEqualUlps_Pin(sideAy, sideBy)) {
        return 0;
    }
    reduction[0] = cubic[0];
    reduction[1].fX = midX;
    reduction[1].fY = midY;
    reduction[2] = cubic[3];
    return 3;
}

static int check_linear(const SkDCubic& cubic,
        int minX, int maxX, int minY, int maxY, SkDCubic& reduction) {
    if (!cubic.isLinear(0, 3)) {
        return 0;
    }
    // four are colinear: return line formed by outside
    reduction[0] = cubic[0];
    reduction[1] = cubic[3];
    return reductionLineCount(reduction);
}

/* food for thought:
http://objectmix.com/graphics/132906-fast-precision-driven-cubic-quadratic-piecewise-degree-reduction-algos-2-a.html

Given points c1, c2, c3 and c4 of a cubic Bezier, the points of the
corresponding quadratic Bezier are (given in convex combinations of
points):

q1 = (11/13)c1 + (3/13)c2 -(3/13)c3 + (2/13)c4
q2 = -c1 + (3/2)c2 + (3/2)c3 - c4
q3 = (2/13)c1 - (3/13)c2 + (3/13)c3 + (11/13)c4

Of course, this curve does not interpolate the end-points, but it would
be interesting to see the behaviour of such a curve in an applet.

--
Kalle Rutanen
http://kaba.hilvi.org

*/

// reduce to a quadratic or smaller
// look for identical points
// look for all four points in a line
    // note that three points in a line doesn't simplify a cubic
// look for approximation with single quadratic
    // save approximation with multiple quadratics for later
int SkReduceOrder::reduce(const SkDCubic& cubic, Quadratics allowQuadratics) {
    int index, minX, maxX, minY, maxY;
    int minXSet, minYSet;
    minX = maxX = minY = maxY = 0;
    minXSet = minYSet = 0;
    for (index = 1; index < 4; ++index) {
        if (cubic[minX].fX > cubic[index].fX) {
            minX = index;
        }
        if (cubic[minY].fY > cubic[index].fY) {
            minY = index;
        }
        if (cubic[maxX].fX < cubic[index].fX) {
            maxX = index;
        }
        if (cubic[maxY].fY < cubic[index].fY) {
            maxY = index;
        }
    }
    for (index = 0; index < 4; ++index) {
        double cx = cubic[index].fX;
        double cy = cubic[index].fY;
        double denom = std::max(fabs(cx), std::max(fabs(cy),
                std::max(fabs(cubic[minX].fX), fabs(cubic[minY].fY))));
        if (denom == 0) {
            minXSet |= 1 << index;
            minYSet |= 1 << index;
            continue;
        }
        double inv = 1 / denom;
        if (approximately_equal_half(cx * inv, cubic[minX].fX * inv)) {
            minXSet |= 1 << index;
        }
        if (approximately_equal_half(cy * inv, cubic[minY].fY * inv)) {
            minYSet |= 1 << index;
        }
    }
    if (minXSet == 0xF) {  // test for vertical line
        if (minYSet == 0xF) {  // return 1 if all four are coincident
            return coincident_line(cubic, fCubic);
        }
        return vertical_line(cubic, fCubic);
    }
    if (minYSet == 0xF) {  // test for horizontal line
        return horizontal_line(cubic, fCubic);
    }
    int result = check_linear(cubic, minX, maxX, minY, maxY, fCubic);
    if (result) {
        return result;
    }
    if (allowQuadratics == SkReduceOrder::kAllow_Quadratics
            && (result = check_quadratic(cubic, fCubic))) {
        return result;
    }
    fCubic = cubic;
    return 4;
}

SkPath::Verb SkReduceOrder::Quad(const SkPoint a[3], SkPoint* reducePts) {
    SkDQuad quad;
    quad.set(a);
    SkReduceOrder reducer;
    int order = reducer.reduce(quad);
    if (order == 2) {  // quad became line
        for (int index = 0; index < order; ++index) {
            *reducePts++ = reducer.fLine[index].asSkPoint();
        }
    }
    return SkPathOpsPointsToVerb(order - 1);
}

SkPath::Verb SkReduceOrder::Conic(const SkConic& c, SkPoint* reducePts) {
    SkPath::Verb verb = SkReduceOrder::Quad(c.fPts, reducePts);
    if (verb > SkPath::kLine_Verb && c.fW == 1) {
        return SkPath::kQuad_Verb;
    }
    return verb == SkPath::kQuad_Verb ? SkPath::kConic_Verb : verb;
}

SkPath::Verb SkReduceOrder::Cubic(const SkPoint a[4], SkPoint* reducePts) {
    if (SkDPoint::ApproximatelyEqual(a[0], a[1]) && SkDPoint::ApproximatelyEqual(a[0], a[2])
            && SkDPoint::ApproximatelyEqual(a[0], a[3])) {
        reducePts[0] = a[0];
        return SkPath::kMove_Verb;
    }
    SkDCubic cubic;
    cubic.set(a);
    SkReduceOrder reducer;
    int order = reducer.reduce(cubic, kAllow_Quadratics);
    if (order == 2 || order == 3) {  // cubic became line or quad
        for (int index = 0; index < order; ++index) {
            *reducePts++ = reducer.fQuad[index].asSkPoint();
        }
    }
    return SkPathOpsPointsToVerb(order - 1);
}
