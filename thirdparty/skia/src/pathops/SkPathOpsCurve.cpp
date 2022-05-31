/*
 * Copyright 2015 Google Inc.
 *
 * Use of this source code is governed by a BSD-style license that can be
 * found in the LICENSE file.
 */
#include "src/pathops/SkPathOpsBounds.h"
#include "src/pathops/SkPathOpsCurve.h"
#include "src/pathops/SkPathOpsRect.h"

 // this cheats and assumes that the perpendicular to the point is the closest ray to the curve
 // this case (where the line and the curve are nearly coincident) may be the only case that counts
double SkDCurve::nearPoint(SkPath::Verb verb, const SkDPoint& xy, const SkDPoint& opp) const {
    int count = SkPathOpsVerbToPoints(verb);
    double minX = fCubic.fPts[0].fX;
    double maxX = minX;
    for (int index = 1; index <= count; ++index) {
        minX = std::min(minX, fCubic.fPts[index].fX);
        maxX = std::max(maxX, fCubic.fPts[index].fX);
    }
    if (!AlmostBetweenUlps(minX, xy.fX, maxX)) {
        return -1;
    }
    double minY = fCubic.fPts[0].fY;
    double maxY = minY;
    for (int index = 1; index <= count; ++index) {
        minY = std::min(minY, fCubic.fPts[index].fY);
        maxY = std::max(maxY, fCubic.fPts[index].fY);
    }
    if (!AlmostBetweenUlps(minY, xy.fY, maxY)) {
        return -1;
    }
    SkIntersections i;
    SkDLine perp = {{ xy, { xy.fX + opp.fY - xy.fY, xy.fY + xy.fX - opp.fX }}};
    (*CurveDIntersectRay[verb])(*this, perp, &i);
    int minIndex = -1;
    double minDist = FLT_MAX;
    for (int index = 0; index < i.used(); ++index) {
        double dist = xy.distance(i.pt(index));
        if (minDist > dist) {
            minDist = dist;
            minIndex = index;
        }
    }
    if (minIndex < 0) {
        return -1;
    }
    double largest = std::max(std::max(maxX, maxY), -std::min(minX, minY));
    if (!AlmostEqualUlps_Pin(largest, largest + minDist)) { // is distance within ULPS tolerance?
        return -1;
    }
    return SkPinT(i[0][minIndex]);
}

void SkDCurve::offset(SkPath::Verb verb, const SkDVector& off) {
    int count = SkPathOpsVerbToPoints(verb);
    for (int index = 0; index <= count; ++index) {
        fCubic.fPts[index] += off;
    }
}

void SkDCurve::setConicBounds(const SkPoint curve[3], SkScalar curveWeight,
        double tStart, double tEnd, SkPathOpsBounds* bounds) {
    SkDConic dCurve;
    dCurve.set(curve, curveWeight);
    SkDRect dRect;
    dRect.setBounds(dCurve, fConic, tStart, tEnd);
    bounds->setLTRB(SkDoubleToScalar(dRect.fLeft), SkDoubleToScalar(dRect.fTop),
                    SkDoubleToScalar(dRect.fRight), SkDoubleToScalar(dRect.fBottom));
}

void SkDCurve::setCubicBounds(const SkPoint curve[4], SkScalar ,
        double tStart, double tEnd, SkPathOpsBounds* bounds) {
    SkDCubic dCurve;
    dCurve.set(curve);
    SkDRect dRect;
    dRect.setBounds(dCurve, fCubic, tStart, tEnd);
    bounds->setLTRB(SkDoubleToScalar(dRect.fLeft), SkDoubleToScalar(dRect.fTop),
                    SkDoubleToScalar(dRect.fRight), SkDoubleToScalar(dRect.fBottom));
}

void SkDCurve::setQuadBounds(const SkPoint curve[3], SkScalar ,
        double tStart, double tEnd, SkPathOpsBounds* bounds) {
    SkDQuad dCurve;
    dCurve.set(curve);
    SkDRect dRect;
    dRect.setBounds(dCurve, fQuad, tStart, tEnd);
    bounds->setLTRB(SkDoubleToScalar(dRect.fLeft), SkDoubleToScalar(dRect.fTop),
                    SkDoubleToScalar(dRect.fRight), SkDoubleToScalar(dRect.fBottom));
}

void SkDCurveSweep::setCurveHullSweep(SkPath::Verb verb) {
    fOrdered = true;
    fSweep[0] = fCurve[1] - fCurve[0];
    if (SkPath::kLine_Verb == verb) {
        fSweep[1] = fSweep[0];
        fIsCurve = false;
        return;
    }
    fSweep[1] = fCurve[2] - fCurve[0];
    // OPTIMIZE: I do the following float check a lot -- probably need a
    // central place for this val-is-small-compared-to-curve check
    double maxVal = 0;
    for (int index = 0; index <= SkPathOpsVerbToPoints(verb); ++index) {
        maxVal = std::max(maxVal, std::max(SkTAbs(fCurve[index].fX),
                SkTAbs(fCurve[index].fY)));
    }
    {
        if (SkPath::kCubic_Verb != verb) {
            if (roughly_zero_when_compared_to(fSweep[0].fX, maxVal)
                    && roughly_zero_when_compared_to(fSweep[0].fY, maxVal)) {
                fSweep[0] = fSweep[1];
            }
            goto setIsCurve;
        }
        SkDVector thirdSweep = fCurve[3] - fCurve[0];
        if (fSweep[0].fX == 0 && fSweep[0].fY == 0) {
            fSweep[0] = fSweep[1];
            fSweep[1] = thirdSweep;
            if (roughly_zero_when_compared_to(fSweep[0].fX, maxVal)
                    && roughly_zero_when_compared_to(fSweep[0].fY, maxVal)) {
                fSweep[0] = fSweep[1];
                fCurve[1] = fCurve[3];
            }
            goto setIsCurve;
        }
        double s1x3 = fSweep[0].crossCheck(thirdSweep);
        double s3x2 = thirdSweep.crossCheck(fSweep[1]);
        if (s1x3 * s3x2 >= 0) {  // if third vector is on or between first two vectors
            goto setIsCurve;
        }
        double s2x1 = fSweep[1].crossCheck(fSweep[0]);
        // FIXME: If the sweep of the cubic is greater than 180 degrees, we're in trouble
        // probably such wide sweeps should be artificially subdivided earlier so that never happens
        SkASSERT(s1x3 * s2x1 < 0 || s1x3 * s3x2 < 0);
        if (s3x2 * s2x1 < 0) {
            SkASSERT(s2x1 * s1x3 > 0);
            fSweep[0] = fSweep[1];
            fOrdered = false;
        }
        fSweep[1] = thirdSweep;
    }
setIsCurve:
    fIsCurve = fSweep[0].crossCheck(fSweep[1]) != 0;
}
