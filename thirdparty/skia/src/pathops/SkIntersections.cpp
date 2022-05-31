/*
 * Copyright 2012 Google Inc.
 *
 * Use of this source code is governed by a BSD-style license that can be
 * found in the LICENSE file.
 */

#include "src/pathops/SkIntersections.h"

int SkIntersections::closestTo(double rangeStart, double rangeEnd, const SkDPoint& testPt,
        double* closestDist) const {
    int closest = -1;
    *closestDist = SK_ScalarMax;
    for (int index = 0; index < fUsed; ++index) {
        if (!between(rangeStart, fT[0][index], rangeEnd)) {
            continue;
        }
        const SkDPoint& iPt = fPt[index];
        double dist = testPt.distanceSquared(iPt);
        if (*closestDist > dist) {
            *closestDist = dist;
            closest = index;
        }
    }
    return closest;
}

void SkIntersections::flip() {
    for (int index = 0; index < fUsed; ++index) {
        fT[1][index] = 1 - fT[1][index];
    }
}

int SkIntersections::insert(double one, double two, const SkDPoint& pt) {
    if (fIsCoincident[0] == 3 && between(fT[0][0], one, fT[0][1])) {
        // For now, don't allow a mix of coincident and non-coincident intersections
        return -1;
    }
    SkASSERT(fUsed <= 1 || fT[0][0] <= fT[0][1]);
    int index;
    for (index = 0; index < fUsed; ++index) {
        double oldOne = fT[0][index];
        double oldTwo = fT[1][index];
        if (one == oldOne && two == oldTwo) {
            return -1;
        }
        if (more_roughly_equal(oldOne, one) && more_roughly_equal(oldTwo, two)) {
            if ((!precisely_zero(one) || precisely_zero(oldOne))
                    && (!precisely_equal(one, 1) || precisely_equal(oldOne, 1))
                    && (!precisely_zero(two) || precisely_zero(oldTwo))
                    && (!precisely_equal(two, 1) || precisely_equal(oldTwo, 1))) {
                return -1;
            }
            SkASSERT(one >= 0 && one <= 1);
            SkASSERT(two >= 0 && two <= 1);
            // remove this and reinsert below in case replacing would make list unsorted
            int remaining = fUsed - index - 1;
            memmove(&fPt[index], &fPt[index + 1], sizeof(fPt[0]) * remaining);
            memmove(&fT[0][index], &fT[0][index + 1], sizeof(fT[0][0]) * remaining);
            memmove(&fT[1][index], &fT[1][index + 1], sizeof(fT[1][0]) * remaining);
            int clearMask = ~((1 << index) - 1);
            fIsCoincident[0] -= (fIsCoincident[0] >> 1) & clearMask;
            fIsCoincident[1] -= (fIsCoincident[1] >> 1) & clearMask;
            --fUsed;
            break;
        }
    #if ONE_OFF_DEBUG
        if (pt.roughlyEqual(fPt[index])) {
            SkDebugf("%s t=%1.9g pts roughly equal\n", __FUNCTION__, one);
        }
    #endif
    }
    for (index = 0; index < fUsed; ++index) {
        if (fT[0][index] > one) {
            break;
        }
    }
    if (fUsed >= fMax) {
        SkOPASSERT(0);  // FIXME : this error, if it is to be handled at runtime in release, must
                      // be propagated all the way back down to the caller, and return failure.
        fUsed = 0;
        return 0;
    }
    int remaining = fUsed - index;
    if (remaining > 0) {
        memmove(&fPt[index + 1], &fPt[index], sizeof(fPt[0]) * remaining);
        memmove(&fT[0][index + 1], &fT[0][index], sizeof(fT[0][0]) * remaining);
        memmove(&fT[1][index + 1], &fT[1][index], sizeof(fT[1][0]) * remaining);
        int clearMask = ~((1 << index) - 1);
        fIsCoincident[0] += fIsCoincident[0] & clearMask;
        fIsCoincident[1] += fIsCoincident[1] & clearMask;
    }
    fPt[index] = pt;
    if (one < 0 || one > 1) {
        return -1;
    }
    if (two < 0 || two > 1) {
        return -1;
    }
    fT[0][index] = one;
    fT[1][index] = two;
    ++fUsed;
    SkASSERT(fUsed <= SK_ARRAY_COUNT(fPt));
    return index;
}

void SkIntersections::insertNear(double one, double two, const SkDPoint& pt1, const SkDPoint& pt2) {
    SkASSERT(one == 0 || one == 1);
    SkASSERT(two == 0 || two == 1);
    SkASSERT(pt1 != pt2);
    fNearlySame[one ? 1 : 0] = true;
    (void) insert(one, two, pt1);
    fPt2[one ? 1 : 0] = pt2;
}

int SkIntersections::insertCoincident(double one, double two, const SkDPoint& pt) {
    int index = insertSwap(one, two, pt);
    if (index >= 0) {
        setCoincident(index);
    }
    return index;
}

void SkIntersections::setCoincident(int index) {
    SkASSERT(index >= 0);
    int bit = 1 << index;
    fIsCoincident[0] |= bit;
    fIsCoincident[1] |= bit;
}

void SkIntersections::merge(const SkIntersections& a, int aIndex, const SkIntersections& b,
        int bIndex) {
    this->reset();
    fT[0][0] = a.fT[0][aIndex];
    fT[1][0] = b.fT[0][bIndex];
    fPt[0] = a.fPt[aIndex];
    fPt2[0] = b.fPt[bIndex];
    fUsed = 1;
}

int SkIntersections::mostOutside(double rangeStart, double rangeEnd, const SkDPoint& origin) const {
    int result = -1;
    for (int index = 0; index < fUsed; ++index) {
        if (!between(rangeStart, fT[0][index], rangeEnd)) {
            continue;
        }
        if (result < 0) {
            result = index;
            continue;
        }
        SkDVector best = fPt[result] - origin;
        SkDVector test = fPt[index] - origin;
        if (test.crossCheck(best) < 0) {
            result = index;
        }
    }
    return result;
}

void SkIntersections::removeOne(int index) {
    int remaining = --fUsed - index;
    if (remaining <= 0) {
        return;
    }
    memmove(&fPt[index], &fPt[index + 1], sizeof(fPt[0]) * remaining);
    memmove(&fT[0][index], &fT[0][index + 1], sizeof(fT[0][0]) * remaining);
    memmove(&fT[1][index], &fT[1][index + 1], sizeof(fT[1][0]) * remaining);
//    SkASSERT(fIsCoincident[0] == 0);
    int coBit = fIsCoincident[0] & (1 << index);
    fIsCoincident[0] -= ((fIsCoincident[0] >> 1) & ~((1 << index) - 1)) + coBit;
    SkASSERT(!(coBit ^ (fIsCoincident[1] & (1 << index))));
    fIsCoincident[1] -= ((fIsCoincident[1] >> 1) & ~((1 << index) - 1)) + coBit;
}
