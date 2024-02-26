/*
 * Copyright 2012 Google Inc.
 *
 * Use of this source code is governed by a BSD-style license that can be
 * found in the LICENSE file.
 */
#include "src/pathops/SkIntersections.h"
#include "src/pathops/SkPathOpsLine.h"

#include <utility>

void SkIntersections::cleanUpParallelLines(bool parallel) {
    while (fUsed > 2) {
        removeOne(1);
    }
    if (fUsed == 2 && !parallel) {
        bool startMatch = fT[0][0] == 0 || zero_or_one(fT[1][0]);
        bool endMatch = fT[0][1] == 1 || zero_or_one(fT[1][1]);
        if ((!startMatch && !endMatch) || approximately_equal(fT[0][0], fT[0][1])) {
            SkASSERT(startMatch || endMatch);
            if (startMatch && endMatch && (fT[0][0] != 0 || !zero_or_one(fT[1][0]))
                    && fT[0][1] == 1 && zero_or_one(fT[1][1])) {
                removeOne(0);
            } else {
                removeOne(endMatch);
            }
        }
    }
    if (fUsed == 2) {
        fIsCoincident[0] = fIsCoincident[1] = 0x03;
    }
}

void SkIntersections::computePoints(const SkDLine& line, int used) {
    fPt[0] = line.ptAtT(fT[0][0]);
    if ((fUsed = used) == 2) {
        fPt[1] = line.ptAtT(fT[0][1]);
    }
}

int SkIntersections::intersectRay(const SkDLine& a, const SkDLine& b) {
    fMax = 2;
    SkDVector aLen = a[1] - a[0];
    SkDVector bLen = b[1] - b[0];
    /* Slopes match when denom goes to zero:
                      axLen / ayLen ==                   bxLen / byLen
    (ayLen * byLen) * axLen / ayLen == (ayLen * byLen) * bxLen / byLen
             byLen  * axLen         ==  ayLen          * bxLen
             byLen  * axLen         -   ayLen          * bxLen == 0 ( == denom )
     */
    double denom = bLen.fY * aLen.fX - aLen.fY * bLen.fX;
    int used;
    if (!approximately_zero(denom)) {
        SkDVector ab0 = a[0] - b[0];
        double numerA = ab0.fY * bLen.fX - bLen.fY * ab0.fX;
        double numerB = ab0.fY * aLen.fX - aLen.fY * ab0.fX;
        numerA /= denom;
        numerB /= denom;
        fT[0][0] = numerA;
        fT[1][0] = numerB;
        used = 1;
    } else {
       /* See if the axis intercepts match:
                  ay - ax * ayLen / axLen  ==          by - bx * ayLen / axLen
         axLen * (ay - ax * ayLen / axLen) == axLen * (by - bx * ayLen / axLen)
         axLen *  ay - ax * ayLen          == axLen *  by - bx * ayLen
        */
        if (!AlmostEqualUlps(aLen.fX * a[0].fY - aLen.fY * a[0].fX,
                aLen.fX * b[0].fY - aLen.fY * b[0].fX)) {
            return fUsed = 0;
        }
        // there's no great answer for intersection points for coincident rays, but return something
        fT[0][0] = fT[1][0] = 0;
        fT[1][0] = fT[1][1] = 1;
        used = 2;
    }
    computePoints(a, used);
    return fUsed;
}

// note that this only works if both lines are neither horizontal nor vertical
int SkIntersections::intersect(const SkDLine& a, const SkDLine& b) {
    fMax = 3;  // note that we clean up so that there is no more than two in the end
    // see if end points intersect the opposite line
    double t;
    for (int iA = 0; iA < 2; ++iA) {
        if ((t = b.exactPoint(a[iA])) >= 0) {
            insert(iA, t, a[iA]);
        }
    }
    for (int iB = 0; iB < 2; ++iB) {
        if ((t = a.exactPoint(b[iB])) >= 0) {
            insert(t, iB, b[iB]);
        }
    }
    /* Determine the intersection point of two line segments
       Return FALSE if the lines don't intersect
       from: http://paulbourke.net/geometry/lineline2d/ */
    double axLen = a[1].fX - a[0].fX;
    double ayLen = a[1].fY - a[0].fY;
    double bxLen = b[1].fX - b[0].fX;
    double byLen = b[1].fY - b[0].fY;
    /* Slopes match when denom goes to zero:
                      axLen / ayLen ==                   bxLen / byLen
    (ayLen * byLen) * axLen / ayLen == (ayLen * byLen) * bxLen / byLen
             byLen  * axLen         ==  ayLen          * bxLen
             byLen  * axLen         -   ayLen          * bxLen == 0 ( == denom )
     */
    double axByLen = axLen * byLen;
    double ayBxLen = ayLen * bxLen;
    // detect parallel lines the same way here and in SkOpAngle operator <
    // so that non-parallel means they are also sortable
    bool unparallel = fAllowNear ? NotAlmostEqualUlps_Pin(axByLen, ayBxLen)
            : NotAlmostDequalUlps(axByLen, ayBxLen);
    if (unparallel && fUsed == 0) {
        double ab0y = a[0].fY - b[0].fY;
        double ab0x = a[0].fX - b[0].fX;
        double numerA = ab0y * bxLen - byLen * ab0x;
        double numerB = ab0y * axLen - ayLen * ab0x;
        double denom = axByLen - ayBxLen;
        if (between(0, numerA, denom) && between(0, numerB, denom)) {
            fT[0][0] = numerA / denom;
            fT[1][0] = numerB / denom;
            computePoints(a, 1);
        }
    }
/* Allow tracking that both sets of end points are near each other -- the lines are entirely
   coincident -- even when the end points are not exactly the same.
   Mark this as a 'wild card' for the end points, so that either point is considered totally
   coincident. Then, avoid folding the lines over each other, but allow either end to mate
   to the next set of lines.
 */
    if (fAllowNear || !unparallel) {
        double aNearB[2];
        double bNearA[2];
        bool aNotB[2] = {false, false};
        bool bNotA[2] = {false, false};
        int nearCount = 0;
        for (int index = 0; index < 2; ++index) {
            aNearB[index] = t = b.nearPoint(a[index], &aNotB[index]);
            nearCount += t >= 0;
            bNearA[index] = t = a.nearPoint(b[index], &bNotA[index]);
            nearCount += t >= 0;
        }
        if (nearCount > 0) {
            // Skip if each segment contributes to one end point.
            if (nearCount != 2 || aNotB[0] == aNotB[1]) {
                for (int iA = 0; iA < 2; ++iA) {
                    if (!aNotB[iA]) {
                        continue;
                    }
                    int nearer = aNearB[iA] > 0.5;
                    if (!bNotA[nearer]) {
                        continue;
                    }
                    SkASSERT(a[iA] != b[nearer]);
                    SkOPASSERT(iA == (bNearA[nearer] > 0.5));
                    insertNear(iA, nearer, a[iA], b[nearer]);
                    aNearB[iA] = -1;
                    bNearA[nearer] = -1;
                    nearCount -= 2;
                }
            }
            if (nearCount > 0) {
                for (int iA = 0; iA < 2; ++iA) {
                    if (aNearB[iA] >= 0) {
                        insert(iA, aNearB[iA], a[iA]);
                    }
                }
                for (int iB = 0; iB < 2; ++iB) {
                    if (bNearA[iB] >= 0) {
                        insert(bNearA[iB], iB, b[iB]);
                    }
                }
            }
        }
    }
    cleanUpParallelLines(!unparallel);
    SkASSERT(fUsed <= 2);
    return fUsed;
}

static int horizontal_coincident(const SkDLine& line, double y) {
    double min = line[0].fY;
    double max = line[1].fY;
    if (min > max) {
        using std::swap;
        swap(min, max);
    }
    if (min > y || max < y) {
        return 0;
    }
    if (AlmostEqualUlps(min, max) && max - min < fabs(line[0].fX - line[1].fX)) {
        return 2;
    }
    return 1;
}

double SkIntersections::HorizontalIntercept(const SkDLine& line, double y) {
    SkASSERT(line[1].fY != line[0].fY);
    return SkPinT((y - line[0].fY) / (line[1].fY - line[0].fY));
}

int SkIntersections::horizontal(const SkDLine& line, double left, double right,
                                double y, bool flipped) {
    fMax = 3;  // clean up parallel at the end will limit the result to 2 at the most
    // see if end points intersect the opposite line
    double t;
    const SkDPoint leftPt = { left, y };
    if ((t = line.exactPoint(leftPt)) >= 0) {
        insert(t, (double) flipped, leftPt);
    }
    if (left != right) {
        const SkDPoint rightPt = { right, y };
        if ((t = line.exactPoint(rightPt)) >= 0) {
            insert(t, (double) !flipped, rightPt);
        }
        for (int index = 0; index < 2; ++index) {
            if ((t = SkDLine::ExactPointH(line[index], left, right, y)) >= 0) {
                insert((double) index, flipped ? 1 - t : t, line[index]);
            }
        }
    }
    int result = horizontal_coincident(line, y);
    if (result == 1 && fUsed == 0) {
        fT[0][0] = HorizontalIntercept(line, y);
        double xIntercept = line[0].fX + fT[0][0] * (line[1].fX - line[0].fX);
        if (between(left, xIntercept, right)) {
            fT[1][0] = (xIntercept - left) / (right - left);
            if (flipped) {
                // OPTIMIZATION: ? instead of swapping, pass original line, use [1].fX - [0].fX
                for (int index = 0; index < result; ++index) {
                    fT[1][index] = 1 - fT[1][index];
                }
            }
            fPt[0].fX = xIntercept;
            fPt[0].fY = y;
            fUsed = 1;
        }
    }
    if (fAllowNear || result == 2) {
        if ((t = line.nearPoint(leftPt, nullptr)) >= 0) {
            insert(t, (double) flipped, leftPt);
        }
        if (left != right) {
            const SkDPoint rightPt = { right, y };
            if ((t = line.nearPoint(rightPt, nullptr)) >= 0) {
                insert(t, (double) !flipped, rightPt);
            }
            for (int index = 0; index < 2; ++index) {
                if ((t = SkDLine::NearPointH(line[index], left, right, y)) >= 0) {
                    insert((double) index, flipped ? 1 - t : t, line[index]);
                }
            }
        }
    }
    cleanUpParallelLines(result == 2);
    return fUsed;
}

static int vertical_coincident(const SkDLine& line, double x) {
    double min = line[0].fX;
    double max = line[1].fX;
    if (min > max) {
        using std::swap;
        swap(min, max);
    }
    if (!precisely_between(min, x, max)) {
        return 0;
    }
    if (AlmostEqualUlps(min, max)) {
        return 2;
    }
    return 1;
}

double SkIntersections::VerticalIntercept(const SkDLine& line, double x) {
    SkASSERT(line[1].fX != line[0].fX);
    return SkPinT((x - line[0].fX) / (line[1].fX - line[0].fX));
}

int SkIntersections::vertical(const SkDLine& line, double top, double bottom,
                              double x, bool flipped) {
    fMax = 3;  // cleanup parallel lines will bring this back line
    // see if end points intersect the opposite line
    double t;
    SkDPoint topPt = { x, top };
    if ((t = line.exactPoint(topPt)) >= 0) {
        insert(t, (double) flipped, topPt);
    }
    if (top != bottom) {
        SkDPoint bottomPt = { x, bottom };
        if ((t = line.exactPoint(bottomPt)) >= 0) {
            insert(t, (double) !flipped, bottomPt);
        }
        for (int index = 0; index < 2; ++index) {
            if ((t = SkDLine::ExactPointV(line[index], top, bottom, x)) >= 0) {
                insert((double) index, flipped ? 1 - t : t, line[index]);
            }
        }
    }
    int result = vertical_coincident(line, x);
    if (result == 1 && fUsed == 0) {
        fT[0][0] = VerticalIntercept(line, x);
        double yIntercept = line[0].fY + fT[0][0] * (line[1].fY - line[0].fY);
        if (between(top, yIntercept, bottom)) {
            fT[1][0] = (yIntercept - top) / (bottom - top);
            if (flipped) {
                // OPTIMIZATION: instead of swapping, pass original line, use [1].fY - [0].fY
                for (int index = 0; index < result; ++index) {
                    fT[1][index] = 1 - fT[1][index];
                }
            }
            fPt[0].fX = x;
            fPt[0].fY = yIntercept;
            fUsed = 1;
        }
    }
    if (fAllowNear || result == 2) {
        if ((t = line.nearPoint(topPt, nullptr)) >= 0) {
            insert(t, (double) flipped, topPt);
        }
        if (top != bottom) {
            SkDPoint bottomPt = { x, bottom };
            if ((t = line.nearPoint(bottomPt, nullptr)) >= 0) {
                insert(t, (double) !flipped, bottomPt);
            }
            for (int index = 0; index < 2; ++index) {
                if ((t = SkDLine::NearPointV(line[index], top, bottom, x)) >= 0) {
                    insert((double) index, flipped ? 1 - t : t, line[index]);
                }
            }
        }
    }
    cleanUpParallelLines(result == 2);
    SkASSERT(fUsed <= 2);
    return fUsed;
}

