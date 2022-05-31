/*
 * Copyright 2012 Google Inc.
 *
 * Use of this source code is governed by a BSD-style license that can be
 * found in the LICENSE file.
 */
#include "src/pathops/SkPathOpsCubic.h"

static bool rotate(const SkDCubic& cubic, int zero, int index, SkDCubic& rotPath) {
    double dy = cubic[index].fY - cubic[zero].fY;
    double dx = cubic[index].fX - cubic[zero].fX;
    if (approximately_zero(dy)) {
        if (approximately_zero(dx)) {
            return false;
        }
        rotPath = cubic;
        if (dy) {
            rotPath[index].fY = cubic[zero].fY;
            int mask = other_two(index, zero);
            int side1 = index ^ mask;
            int side2 = zero ^ mask;
            if (approximately_equal(cubic[side1].fY, cubic[zero].fY)) {
                rotPath[side1].fY = cubic[zero].fY;
            }
            if (approximately_equal(cubic[side2].fY, cubic[zero].fY)) {
                rotPath[side2].fY = cubic[zero].fY;
            }
        }
        return true;
    }
    for (int i = 0; i < 4; ++i) {
        rotPath[i].fX = cubic[i].fX * dx + cubic[i].fY * dy;
        rotPath[i].fY = cubic[i].fY * dx - cubic[i].fX * dy;
    }
    return true;
}


// Returns 0 if negative, 1 if zero, 2 if positive
static int side(double x) {
    return (x > 0) + (x >= 0);
}

/* Given a cubic, find the convex hull described by the end and control points.
   The hull may have 3 or 4 points. Cubics that degenerate into a point or line
   are not considered.

   The hull is computed by assuming that three points, if unique and non-linear,
   form a triangle. The fourth point may replace one of the first three, may be
   discarded if in the triangle or on an edge, or may be inserted between any of
   the three to form a convex quadralateral.

   The indices returned in order describe the convex hull.
*/
int SkDCubic::convexHull(char order[4]) const {
    size_t index;
    // find top point
    size_t yMin = 0;
    for (index = 1; index < 4; ++index) {
        if (fPts[yMin].fY > fPts[index].fY || (fPts[yMin].fY == fPts[index].fY
                && fPts[yMin].fX > fPts[index].fX)) {
            yMin = index;
        }
    }
    order[0] = yMin;
    int midX = -1;
    int backupYMin = -1;
    for (int pass = 0; pass < 2; ++pass) {
        for (index = 0; index < 4; ++index) {
            if (index == yMin) {
                continue;
            }
            // rotate line from (yMin, index) to axis
            // see if remaining two points are both above or below
            // use this to find mid
            int mask = other_two(yMin, index);
            int side1 = yMin ^ mask;
            int side2 = index ^ mask;
            SkDCubic rotPath;
            if (!rotate(*this, yMin, index, rotPath)) { // ! if cbc[yMin]==cbc[idx]
                order[1] = side1;
                order[2] = side2;
                return 3;
            }
            int sides = side(rotPath[side1].fY - rotPath[yMin].fY);
            sides ^= side(rotPath[side2].fY - rotPath[yMin].fY);
            if (sides == 2) { // '2' means one remaining point <0, one >0
                if (midX >= 0) {
                    // one of the control points is equal to an end point
                    order[0] = 0;
                    order[1] = 3;
                    if (fPts[1] == fPts[0] || fPts[1] == fPts[3]) {
                        order[2] = 2;
                        return 3;
                    }
                    if (fPts[2] == fPts[0] || fPts[2] == fPts[3]) {
                        order[2] = 1;
                        return 3;
                    }
                    // one of the control points may be very nearly but not exactly equal --
                    double dist1_0 = fPts[1].distanceSquared(fPts[0]);
                    double dist1_3 = fPts[1].distanceSquared(fPts[3]);
                    double dist2_0 = fPts[2].distanceSquared(fPts[0]);
                    double dist2_3 = fPts[2].distanceSquared(fPts[3]);
                    double smallest1distSq = std::min(dist1_0, dist1_3);
                    double smallest2distSq = std::min(dist2_0, dist2_3);
                    if (approximately_zero(std::min(smallest1distSq, smallest2distSq))) {
                        order[2] = smallest1distSq < smallest2distSq ? 2 : 1;
                        return 3;
                    }
                }
                midX = index;
            } else if (sides == 0) { // '0' means both to one side or the other
                backupYMin = index;
            }
        }
        if (midX >= 0) {
            break;
        }
        if (backupYMin < 0) {
            break;
        }
        yMin = backupYMin;
        backupYMin = -1;
    }
    if (midX < 0) {
        midX = yMin ^ 3; // choose any other point
    }
    int mask = other_two(yMin, midX);
    int least = yMin ^ mask;
    int most = midX ^ mask;
    order[0] = yMin;
    order[1] = least;

    // see if mid value is on same side of line (least, most) as yMin
    SkDCubic midPath;
    if (!rotate(*this, least, most, midPath)) { // ! if cbc[least]==cbc[most]
        order[2] = midX;
        return 3;
    }
    int midSides = side(midPath[yMin].fY - midPath[least].fY);
    midSides ^= side(midPath[midX].fY - midPath[least].fY);
    if (midSides != 2) {  // if mid point is not between
        order[2] = most;
        return 3; // result is a triangle
    }
    order[2] = midX;
    order[3] = most;
    return 4; // result is a quadralateral
}
