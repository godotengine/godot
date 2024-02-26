/*
 * Copyright 2015 Google Inc.
 *
 * Use of this source code is governed by a BSD-style license that can be
 * found in the LICENSE file.
 */
#include "src/pathops/SkIntersections.h"
#include "src/pathops/SkLineParameters.h"
#include "src/pathops/SkPathOpsConic.h"
#include "src/pathops/SkPathOpsCubic.h"
#include "src/pathops/SkPathOpsQuad.h"
#include "src/pathops/SkPathOpsRect.h"

// cribbed from the float version in SkGeometry.cpp
static void conic_deriv_coeff(const double src[],
                              SkScalar w,
                              double coeff[3]) {
    const double P20 = src[4] - src[0];
    const double P10 = src[2] - src[0];
    const double wP10 = w * P10;
    coeff[0] = w * P20 - P20;
    coeff[1] = P20 - 2 * wP10;
    coeff[2] = wP10;
}

static double conic_eval_tan(const double coord[], SkScalar w, double t) {
    double coeff[3];
    conic_deriv_coeff(coord, w, coeff);
    return t * (t * coeff[0] + coeff[1]) + coeff[2];
}

int SkDConic::FindExtrema(const double src[], SkScalar w, double t[1]) {
    double coeff[3];
    conic_deriv_coeff(src, w, coeff);

    double tValues[2];
    int roots = SkDQuad::RootsValidT(coeff[0], coeff[1], coeff[2], tValues);
    // In extreme cases, the number of roots returned can be 2. Pathops
    // will fail later on, so there's no advantage to plumbing in an error
    // return here.
    // SkASSERT(0 == roots || 1 == roots);

    if (1 == roots) {
        t[0] = tValues[0];
        return 1;
    }
    return 0;
}

SkDVector SkDConic::dxdyAtT(double t) const {
    SkDVector result = {
        conic_eval_tan(&fPts[0].fX, fWeight, t),
        conic_eval_tan(&fPts[0].fY, fWeight, t)
    };
    if (result.fX == 0 && result.fY == 0) {
        if (zero_or_one(t)) {
            result = fPts[2] - fPts[0];
        } else {
            // incomplete
            SkDebugf("!k");
        }
    }
    return result;
}

static double conic_eval_numerator(const double src[], SkScalar w, double t) {
    SkASSERT(src);
    SkASSERT(t >= 0 && t <= 1);
    double src2w = src[2] * w;
    double C = src[0];
    double A = src[4] - 2 * src2w + C;
    double B = 2 * (src2w - C);
    return (A * t + B) * t + C;
}


static double conic_eval_denominator(SkScalar w, double t) {
    double B = 2 * (w - 1);
    double C = 1;
    double A = -B;
    return (A * t + B) * t + C;
}

bool SkDConic::hullIntersects(const SkDCubic& cubic, bool* isLinear) const {
    return cubic.hullIntersects(*this, isLinear);
}

SkDPoint SkDConic::ptAtT(double t) const {
    if (t == 0) {
        return fPts[0];
    }
    if (t == 1) {
        return fPts[2];
    }
    double denominator = conic_eval_denominator(fWeight, t);
    SkDPoint result = {
        sk_ieee_double_divide(conic_eval_numerator(&fPts[0].fX, fWeight, t), denominator),
        sk_ieee_double_divide(conic_eval_numerator(&fPts[0].fY, fWeight, t), denominator)
    };
    return result;
}

/* see quad subdivide for point rationale */
/* w rationale : the mid point between t1 and t2 could be determined from the computed a/b/c
   values if the computed w was known. Since we know the mid point at (t1+t2)/2, we'll assume
   that it is the same as the point on the new curve t==(0+1)/2.

    d / dz == conic_poly(dst, unknownW, .5) / conic_weight(unknownW, .5);

    conic_poly(dst, unknownW, .5)
                  =   a / 4 + (b * unknownW) / 2 + c / 4
                  =  (a + c) / 4 + (bx * unknownW) / 2

    conic_weight(unknownW, .5)
                  =   unknownW / 2 + 1 / 2

    d / dz                  == ((a + c) / 2 + b * unknownW) / (unknownW + 1)
    d / dz * (unknownW + 1) ==  (a + c) / 2 + b * unknownW
              unknownW       = ((a + c) / 2 - d / dz) / (d / dz - b)

    Thus, w is the ratio of the distance from the mid of end points to the on-curve point, and the
    distance of the on-curve point to the control point.
 */
SkDConic SkDConic::subDivide(double t1, double t2) const {
    double ax, ay, az;
    if (t1 == 0) {
        ax = fPts[0].fX;
        ay = fPts[0].fY;
        az = 1;
    } else if (t1 != 1) {
        ax = conic_eval_numerator(&fPts[0].fX, fWeight, t1);
        ay = conic_eval_numerator(&fPts[0].fY, fWeight, t1);
        az = conic_eval_denominator(fWeight, t1);
    } else {
        ax = fPts[2].fX;
        ay = fPts[2].fY;
        az = 1;
    }
    double midT = (t1 + t2) / 2;
    double dx = conic_eval_numerator(&fPts[0].fX, fWeight, midT);
    double dy = conic_eval_numerator(&fPts[0].fY, fWeight, midT);
    double dz = conic_eval_denominator(fWeight, midT);
    double cx, cy, cz;
    if (t2 == 1) {
        cx = fPts[2].fX;
        cy = fPts[2].fY;
        cz = 1;
    } else if (t2 != 0) {
        cx = conic_eval_numerator(&fPts[0].fX, fWeight, t2);
        cy = conic_eval_numerator(&fPts[0].fY, fWeight, t2);
        cz = conic_eval_denominator(fWeight, t2);
    } else {
        cx = fPts[0].fX;
        cy = fPts[0].fY;
        cz = 1;
    }
    double bx = 2 * dx - (ax + cx) / 2;
    double by = 2 * dy - (ay + cy) / 2;
    double bz = 2 * dz - (az + cz) / 2;
    if (!bz) {
        bz = 1; // if bz is 0, weight is 0, control point has no effect: any value will do
    }
    SkDConic dst = {{{{ax / az, ay / az}, {bx / bz, by / bz}, {cx / cz, cy / cz}}
            SkDEBUGPARAMS(fPts.fDebugGlobalState) },
            SkDoubleToScalar(bz / sqrt(az * cz)) };
    return dst;
}

SkDPoint SkDConic::subDivide(const SkDPoint& a, const SkDPoint& c, double t1, double t2,
        SkScalar* weight) const {
    SkDConic chopped = this->subDivide(t1, t2);
    *weight = chopped.fWeight;
    return chopped[1];
}

int SkTConic::intersectRay(SkIntersections* i, const SkDLine& line) const {
    return i->intersectRay(fConic, line);
}

bool SkTConic::hullIntersects(const SkDQuad& quad, bool* isLinear) const  {
    return quad.hullIntersects(fConic, isLinear);
}

bool SkTConic::hullIntersects(const SkDCubic& cubic, bool* isLinear) const {
    return cubic.hullIntersects(fConic, isLinear);
}

void SkTConic::setBounds(SkDRect* rect) const {
    rect->setBounds(fConic);
}
