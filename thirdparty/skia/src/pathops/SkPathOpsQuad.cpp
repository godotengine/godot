/*
 * Copyright 2012 Google Inc.
 *
 * Use of this source code is governed by a BSD-style license that can be
 * found in the LICENSE file.
 */
#include "src/pathops/SkIntersections.h"
#include "src/pathops/SkLineParameters.h"
#include "src/pathops/SkPathOpsCubic.h"
#include "src/pathops/SkPathOpsCurve.h"
#include "src/pathops/SkPathOpsQuad.h"
#include "src/pathops/SkPathOpsRect.h"

// from blackpawn.com/texts/pointinpoly
static bool pointInTriangle(const SkDPoint fPts[3], const SkDPoint& test) {
    SkDVector v0 = fPts[2] - fPts[0];
    SkDVector v1 = fPts[1] - fPts[0];
    SkDVector v2 = test - fPts[0];
    double dot00 = v0.dot(v0);
    double dot01 = v0.dot(v1);
    double dot02 = v0.dot(v2);
    double dot11 = v1.dot(v1);
    double dot12 = v1.dot(v2);
    // Compute barycentric coordinates
    double denom = dot00 * dot11 - dot01 * dot01;
    double u = dot11 * dot02 - dot01 * dot12;
    double v = dot00 * dot12 - dot01 * dot02;
    // Check if point is in triangle
    if (denom >= 0) {
        return u >= 0 && v >= 0 && u + v < denom;
    }
    return u <= 0 && v <= 0 && u + v > denom;
}

static bool matchesEnd(const SkDPoint fPts[3], const SkDPoint& test) {
    return fPts[0] == test || fPts[2] == test;
}

/* started with at_most_end_pts_in_common from SkDQuadIntersection.cpp */
// Do a quick reject by rotating all points relative to a line formed by
// a pair of one quad's points. If the 2nd quad's points
// are on the line or on the opposite side from the 1st quad's 'odd man', the
// curves at most intersect at the endpoints.
/* if returning true, check contains true if quad's hull collapsed, making the cubic linear
   if returning false, check contains true if the the quad pair have only the end point in common
*/
bool SkDQuad::hullIntersects(const SkDQuad& q2, bool* isLinear) const {
    bool linear = true;
    for (int oddMan = 0; oddMan < kPointCount; ++oddMan) {
        const SkDPoint* endPt[2];
        this->otherPts(oddMan, endPt);
        double origX = endPt[0]->fX;
        double origY = endPt[0]->fY;
        double adj = endPt[1]->fX - origX;
        double opp = endPt[1]->fY - origY;
        double sign = (fPts[oddMan].fY - origY) * adj - (fPts[oddMan].fX - origX) * opp;
        if (approximately_zero(sign)) {
            continue;
        }
        linear = false;
        bool foundOutlier = false;
        for (int n = 0; n < kPointCount; ++n) {
            double test = (q2[n].fY - origY) * adj - (q2[n].fX - origX) * opp;
            if (test * sign > 0 && !precisely_zero(test)) {
                foundOutlier = true;
                break;
            }
        }
        if (!foundOutlier) {
            return false;
        }
    }
    if (linear && !matchesEnd(fPts, q2.fPts[0]) && !matchesEnd(fPts, q2.fPts[2])) {
        // if the end point of the opposite quad is inside the hull that is nearly a line,
        // then representing the quad as a line may cause the intersection to be missed.
        // Check to see if the endpoint is in the triangle.
        if (pointInTriangle(fPts, q2.fPts[0]) || pointInTriangle(fPts, q2.fPts[2])) {
            linear = false;
        }
    }
    *isLinear = linear;
    return true;
}

bool SkDQuad::hullIntersects(const SkDConic& conic, bool* isLinear) const {
    return conic.hullIntersects(*this, isLinear);
}

bool SkDQuad::hullIntersects(const SkDCubic& cubic, bool* isLinear) const {
    return cubic.hullIntersects(*this, isLinear);
}

/* bit twiddling for finding the off curve index (x&~m is the pair in [0,1,2] excluding oddMan)
oddMan    opp   x=oddMan^opp  x=x-oddMan  m=x>>2   x&~m
    0       1         1            1         0       1
            2         2            2         0       2
    1       1         0           -1        -1       0
            2         3            2         0       2
    2       1         3            1         0       1
            2         0           -2        -1       0
*/
void SkDQuad::otherPts(int oddMan, const SkDPoint* endPt[2]) const {
    for (int opp = 1; opp < kPointCount; ++opp) {
        int end = (oddMan ^ opp) - oddMan;  // choose a value not equal to oddMan
        end &= ~(end >> 2);  // if the value went negative, set it to zero
        endPt[opp - 1] = &fPts[end];
    }
}

int SkDQuad::AddValidTs(double s[], int realRoots, double* t) {
    int foundRoots = 0;
    for (int index = 0; index < realRoots; ++index) {
        double tValue = s[index];
        if (approximately_zero_or_more(tValue) && approximately_one_or_less(tValue)) {
            if (approximately_less_than_zero(tValue)) {
                tValue = 0;
            } else if (approximately_greater_than_one(tValue)) {
                tValue = 1;
            }
            for (int idx2 = 0; idx2 < foundRoots; ++idx2) {
                if (approximately_equal(t[idx2], tValue)) {
                    goto nextRoot;
                }
            }
            t[foundRoots++] = tValue;
        }
nextRoot:
        {}
    }
    return foundRoots;
}

// note: caller expects multiple results to be sorted smaller first
// note: http://en.wikipedia.org/wiki/Loss_of_significance has an interesting
//  analysis of the quadratic equation, suggesting why the following looks at
//  the sign of B -- and further suggesting that the greatest loss of precision
//  is in b squared less two a c
int SkDQuad::RootsValidT(double A, double B, double C, double t[2]) {
    double s[2];
    int realRoots = RootsReal(A, B, C, s);
    int foundRoots = AddValidTs(s, realRoots, t);
    return foundRoots;
}

static int handle_zero(const double B, const double C, double s[2]) {
    if (approximately_zero(B)) {
        s[0] = 0;
        return C == 0;
    }
    s[0] = -C / B;
    return 1;
}

/*
Numeric Solutions (5.6) suggests to solve the quadratic by computing
       Q = -1/2(B + sgn(B)Sqrt(B^2 - 4 A C))
and using the roots
      t1 = Q / A
      t2 = C / Q
*/
// this does not discard real roots <= 0 or >= 1
int SkDQuad::RootsReal(const double A, const double B, const double C, double s[2]) {
    if (!A) {
        return handle_zero(B, C, s);
    }
    const double p = B / (2 * A);
    const double q = C / A;
    if (approximately_zero(A) && (approximately_zero_inverse(p) || approximately_zero_inverse(q))) {
        return handle_zero(B, C, s);
    }
    /* normal form: x^2 + px + q = 0 */
    const double p2 = p * p;
    if (!AlmostDequalUlps(p2, q) && p2 < q) {
        return 0;
    }
    double sqrt_D = 0;
    if (p2 > q) {
        sqrt_D = sqrt(p2 - q);
    }
    s[0] = sqrt_D - p;
    s[1] = -sqrt_D - p;
    return 1 + !AlmostDequalUlps(s[0], s[1]);
}

bool SkDQuad::isLinear(int startIndex, int endIndex) const {
    SkLineParameters lineParameters;
    lineParameters.quadEndPoints(*this, startIndex, endIndex);
    // FIXME: maybe it's possible to avoid this and compare non-normalized
    lineParameters.normalize();
    double distance = lineParameters.controlPtDistance(*this);
    double tiniest = std::min(std::min(std::min(std::min(std::min(fPts[0].fX, fPts[0].fY),
            fPts[1].fX), fPts[1].fY), fPts[2].fX), fPts[2].fY);
    double largest = std::max(std::max(std::max(std::max(std::max(fPts[0].fX, fPts[0].fY),
            fPts[1].fX), fPts[1].fY), fPts[2].fX), fPts[2].fY);
    largest = std::max(largest, -tiniest);
    return approximately_zero_when_compared_to(distance, largest);
}

SkDVector SkDQuad::dxdyAtT(double t) const {
    double a = t - 1;
    double b = 1 - 2 * t;
    double c = t;
    SkDVector result = { a * fPts[0].fX + b * fPts[1].fX + c * fPts[2].fX,
            a * fPts[0].fY + b * fPts[1].fY + c * fPts[2].fY };
    if (result.fX == 0 && result.fY == 0) {
        if (zero_or_one(t)) {
            result = fPts[2] - fPts[0];
        } else {
            // incomplete
            SkDebugf("!q");
        }
    }
    return result;
}

// OPTIMIZE: assert if caller passes in t == 0 / t == 1 ?
SkDPoint SkDQuad::ptAtT(double t) const {
    if (0 == t) {
        return fPts[0];
    }
    if (1 == t) {
        return fPts[2];
    }
    double one_t = 1 - t;
    double a = one_t * one_t;
    double b = 2 * one_t * t;
    double c = t * t;
    SkDPoint result = { a * fPts[0].fX + b * fPts[1].fX + c * fPts[2].fX,
            a * fPts[0].fY + b * fPts[1].fY + c * fPts[2].fY };
    return result;
}

static double interp_quad_coords(const double* src, double t) {
    if (0 == t) {
        return src[0];
    }
    if (1 == t) {
        return src[4];
    }
    double ab = SkDInterp(src[0], src[2], t);
    double bc = SkDInterp(src[2], src[4], t);
    double abc = SkDInterp(ab, bc, t);
    return abc;
}

bool SkDQuad::monotonicInX() const {
    return between(fPts[0].fX, fPts[1].fX, fPts[2].fX);
}

bool SkDQuad::monotonicInY() const {
    return between(fPts[0].fY, fPts[1].fY, fPts[2].fY);
}

/*
Given a quadratic q, t1, and t2, find a small quadratic segment.

The new quadratic is defined by A, B, and C, where
 A = c[0]*(1 - t1)*(1 - t1) + 2*c[1]*t1*(1 - t1) + c[2]*t1*t1
 C = c[3]*(1 - t1)*(1 - t1) + 2*c[2]*t1*(1 - t1) + c[1]*t1*t1

To find B, compute the point halfway between t1 and t2:

q(at (t1 + t2)/2) == D

Next, compute where D must be if we know the value of B:

_12 = A/2 + B/2
12_ = B/2 + C/2
123 = A/4 + B/2 + C/4
    = D

Group the known values on one side:

B   = D*2 - A/2 - C/2
*/

// OPTIMIZE? : special case  t1 = 1 && t2 = 0
SkDQuad SkDQuad::subDivide(double t1, double t2) const {
    if (0 == t1 && 1 == t2) {
        return *this;
    }
    SkDQuad dst;
    double ax = dst[0].fX = interp_quad_coords(&fPts[0].fX, t1);
    double ay = dst[0].fY = interp_quad_coords(&fPts[0].fY, t1);
    double dx = interp_quad_coords(&fPts[0].fX, (t1 + t2) / 2);
    double dy = interp_quad_coords(&fPts[0].fY, (t1 + t2) / 2);
    double cx = dst[2].fX = interp_quad_coords(&fPts[0].fX, t2);
    double cy = dst[2].fY = interp_quad_coords(&fPts[0].fY, t2);
    /* bx = */ dst[1].fX = 2 * dx - (ax + cx) / 2;
    /* by = */ dst[1].fY = 2 * dy - (ay + cy) / 2;
    return dst;
}

void SkDQuad::align(int endIndex, SkDPoint* dstPt) const {
    if (fPts[endIndex].fX == fPts[1].fX) {
        dstPt->fX = fPts[endIndex].fX;
    }
    if (fPts[endIndex].fY == fPts[1].fY) {
        dstPt->fY = fPts[endIndex].fY;
    }
}

SkDPoint SkDQuad::subDivide(const SkDPoint& a, const SkDPoint& c, double t1, double t2) const {
    SkASSERT(t1 != t2);
    SkDPoint b;
    SkDQuad sub = subDivide(t1, t2);
    SkDLine b0 = {{a, sub[1] + (a - sub[0])}};
    SkDLine b1 = {{c, sub[1] + (c - sub[2])}};
    SkIntersections i;
    i.intersectRay(b0, b1);
    if (i.used() == 1 && i[0][0] >= 0 && i[1][0] >= 0) {
        b = i.pt(0);
    } else {
        SkASSERT(i.used() <= 2);
        return SkDPoint::Mid(b0[1], b1[1]);
    }
    if (t1 == 0 || t2 == 0) {
        align(0, &b);
    }
    if (t1 == 1 || t2 == 1) {
        align(2, &b);
    }
    if (AlmostBequalUlps(b.fX, a.fX)) {
        b.fX = a.fX;
    } else if (AlmostBequalUlps(b.fX, c.fX)) {
        b.fX = c.fX;
    }
    if (AlmostBequalUlps(b.fY, a.fY)) {
        b.fY = a.fY;
    } else if (AlmostBequalUlps(b.fY, c.fY)) {
        b.fY = c.fY;
    }
    return b;
}

/* classic one t subdivision */
static void interp_quad_coords(const double* src, double* dst, double t) {
    double ab = SkDInterp(src[0], src[2], t);
    double bc = SkDInterp(src[2], src[4], t);
    dst[0] = src[0];
    dst[2] = ab;
    dst[4] = SkDInterp(ab, bc, t);
    dst[6] = bc;
    dst[8] = src[4];
}

SkDQuadPair SkDQuad::chopAt(double t) const
{
    SkDQuadPair dst;
    interp_quad_coords(&fPts[0].fX, &dst.pts[0].fX, t);
    interp_quad_coords(&fPts[0].fY, &dst.pts[0].fY, t);
    return dst;
}

static int valid_unit_divide(double numer, double denom, double* ratio)
{
    if (numer < 0) {
        numer = -numer;
        denom = -denom;
    }
    if (denom == 0 || numer == 0 || numer >= denom) {
        return 0;
    }
    double r = numer / denom;
    if (r == 0) {  // catch underflow if numer <<<< denom
        return 0;
    }
    *ratio = r;
    return 1;
}

/** Quad'(t) = At + B, where
    A = 2(a - 2b + c)
    B = 2(b - a)
    Solve for t, only if it fits between 0 < t < 1
*/
int SkDQuad::FindExtrema(const double src[], double tValue[1]) {
    /*  At + B == 0
        t = -B / A
    */
    double a = src[0];
    double b = src[2];
    double c = src[4];
    return valid_unit_divide(a - b, a - b - b + c, tValue);
}

/* Parameterization form, given A*t*t + 2*B*t*(1-t) + C*(1-t)*(1-t)
 *
 * a = A - 2*B +   C
 * b =     2*B - 2*C
 * c =             C
 */
void SkDQuad::SetABC(const double* quad, double* a, double* b, double* c) {
    *a = quad[0];      // a = A
    *b = 2 * quad[2];  // b =     2*B
    *c = quad[4];      // c =             C
    *b -= *c;          // b =     2*B -   C
    *a -= *b;          // a = A - 2*B +   C
    *b -= *c;          // b =     2*B - 2*C
}

int SkTQuad::intersectRay(SkIntersections* i, const SkDLine& line) const {
    return i->intersectRay(fQuad, line);
}

bool SkTQuad::hullIntersects(const SkDConic& conic, bool* isLinear) const  {
    return conic.hullIntersects(fQuad, isLinear);
}

bool SkTQuad::hullIntersects(const SkDCubic& cubic, bool* isLinear) const {
    return cubic.hullIntersects(fQuad, isLinear);
}

void SkTQuad::setBounds(SkDRect* rect) const {
    rect->setBounds(fQuad);
}
