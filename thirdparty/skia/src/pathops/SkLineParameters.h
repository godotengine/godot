/*
 * Copyright 2012 Google Inc.
 *
 * Use of this source code is governed by a BSD-style license that can be
 * found in the LICENSE file.
 */

#ifndef SkLineParameters_DEFINED
#define SkLineParameters_DEFINED

#include "src/pathops/SkPathOpsCubic.h"
#include "src/pathops/SkPathOpsLine.h"
#include "src/pathops/SkPathOpsQuad.h"

// Sources
// computer-aided design - volume 22 number 9 november 1990 pp 538 - 549
// online at http://cagd.cs.byu.edu/~tom/papers/bezclip.pdf

// This turns a line segment into a parameterized line, of the form
// ax + by + c = 0
// When a^2 + b^2 == 1, the line is normalized.
// The distance to the line for (x, y) is d(x,y) = ax + by + c
//
// Note that the distances below are not necessarily normalized. To get the true
// distance, it's necessary to either call normalize() after xxxEndPoints(), or
// divide the result of xxxDistance() by sqrt(normalSquared())

class SkLineParameters {
public:

    bool cubicEndPoints(const SkDCubic& pts) {
        int endIndex = 1;
        cubicEndPoints(pts, 0, endIndex);
        if (dy() != 0) {
            return true;
        }
        if (dx() == 0) {
            cubicEndPoints(pts, 0, ++endIndex);
            SkASSERT(endIndex == 2);
            if (dy() != 0) {
                return true;
            }
            if (dx() == 0) {
                cubicEndPoints(pts, 0, ++endIndex);  // line
                SkASSERT(endIndex == 3);
                return false;
            }
        }
        // FIXME: after switching to round sort, remove bumping fA
        if (dx() < 0) { // only worry about y bias when breaking cw/ccw tie
            return true;
        }
        // if cubic tangent is on x axis, look at next control point to break tie
        // control point may be approximate, so it must move significantly to account for error
        if (NotAlmostEqualUlps(pts[0].fY, pts[++endIndex].fY)) {
            if (pts[0].fY > pts[endIndex].fY) {
                fA = DBL_EPSILON; // push it from 0 to slightly negative (y() returns -a)
            }
            return true;
        }
        if (endIndex == 3) {
            return true;
        }
        SkASSERT(endIndex == 2);
        if (pts[0].fY > pts[3].fY) {
            fA = DBL_EPSILON; // push it from 0 to slightly negative (y() returns -a)
        }
        return true;
    }

    void cubicEndPoints(const SkDCubic& pts, int s, int e) {
        fA = pts[s].fY - pts[e].fY;
        fB = pts[e].fX - pts[s].fX;
        fC = pts[s].fX * pts[e].fY - pts[e].fX * pts[s].fY;
    }

    double cubicPart(const SkDCubic& part) {
        cubicEndPoints(part);
        if (part[0] == part[1] || ((const SkDLine& ) part[0]).nearRay(part[2])) {
            return pointDistance(part[3]);
        }
        return pointDistance(part[2]);
    }

    void lineEndPoints(const SkDLine& pts) {
        fA = pts[0].fY - pts[1].fY;
        fB = pts[1].fX - pts[0].fX;
        fC = pts[0].fX * pts[1].fY - pts[1].fX * pts[0].fY;
    }

    bool quadEndPoints(const SkDQuad& pts) {
        quadEndPoints(pts, 0, 1);
        if (dy() != 0) {
            return true;
        }
        if (dx() == 0) {
            quadEndPoints(pts, 0, 2);
            return false;
        }
        if (dx() < 0) { // only worry about y bias when breaking cw/ccw tie
            return true;
        }
        // FIXME: after switching to round sort, remove this
        if (pts[0].fY > pts[2].fY) {
            fA = DBL_EPSILON;
        }
        return true;
    }

    void quadEndPoints(const SkDQuad& pts, int s, int e) {
        fA = pts[s].fY - pts[e].fY;
        fB = pts[e].fX - pts[s].fX;
        fC = pts[s].fX * pts[e].fY - pts[e].fX * pts[s].fY;
    }

    double quadPart(const SkDQuad& part) {
        quadEndPoints(part);
        return pointDistance(part[2]);
    }

    double normalSquared() const {
        return fA * fA + fB * fB;
    }

    bool normalize() {
        double normal = sqrt(normalSquared());
        if (approximately_zero(normal)) {
            fA = fB = fC = 0;
            return false;
        }
        double reciprocal = 1 / normal;
        fA *= reciprocal;
        fB *= reciprocal;
        fC *= reciprocal;
        return true;
    }

    void cubicDistanceY(const SkDCubic& pts, SkDCubic& distance) const {
        double oneThird = 1 / 3.0;
        for (int index = 0; index < 4; ++index) {
            distance[index].fX = index * oneThird;
            distance[index].fY = fA * pts[index].fX + fB * pts[index].fY + fC;
        }
    }

    void quadDistanceY(const SkDQuad& pts, SkDQuad& distance) const {
        double oneHalf = 1 / 2.0;
        for (int index = 0; index < 3; ++index) {
            distance[index].fX = index * oneHalf;
            distance[index].fY = fA * pts[index].fX + fB * pts[index].fY + fC;
        }
    }

    double controlPtDistance(const SkDCubic& pts, int index) const {
        SkASSERT(index == 1 || index == 2);
        return fA * pts[index].fX + fB * pts[index].fY + fC;
    }

    double controlPtDistance(const SkDQuad& pts) const {
        return fA * pts[1].fX + fB * pts[1].fY + fC;
    }

    double pointDistance(const SkDPoint& pt) const {
        return fA * pt.fX + fB * pt.fY + fC;
    }

    double dx() const {
        return fB;
    }

    double dy() const {
        return -fA;
    }

private:
    double fA;
    double fB;
    double fC;
};

#endif
