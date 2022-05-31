/*
 * Copyright 2012 Google Inc.
 *
 * Use of this source code is governed by a BSD-style license that can be
 * found in the LICENSE file.
 */
#include "src/pathops/SkIntersections.h"
#include "src/pathops/SkPathOpsCubic.h"
#include "src/pathops/SkPathOpsCurve.h"
#include "src/pathops/SkPathOpsLine.h"

/*
Find the interection of a line and cubic by solving for valid t values.

Analogous to line-quadratic intersection, solve line-cubic intersection by
representing the cubic as:
  x = a(1-t)^3 + 2b(1-t)^2t + c(1-t)t^2 + dt^3
  y = e(1-t)^3 + 2f(1-t)^2t + g(1-t)t^2 + ht^3
and the line as:
  y = i*x + j  (if the line is more horizontal)
or:
  x = i*y + j  (if the line is more vertical)

Then using Mathematica, solve for the values of t where the cubic intersects the
line:

  (in) Resultant[
        a*(1 - t)^3 + 3*b*(1 - t)^2*t + 3*c*(1 - t)*t^2 + d*t^3 - x,
        e*(1 - t)^3 + 3*f*(1 - t)^2*t + 3*g*(1 - t)*t^2 + h*t^3 - i*x - j, x]
  (out) -e     +   j     +
       3 e t   - 3 f t   -
       3 e t^2 + 6 f t^2 - 3 g t^2 +
         e t^3 - 3 f t^3 + 3 g t^3 - h t^3 +
     i ( a     -
       3 a t + 3 b t +
       3 a t^2 - 6 b t^2 + 3 c t^2 -
         a t^3 + 3 b t^3 - 3 c t^3 + d t^3 )

if i goes to infinity, we can rewrite the line in terms of x. Mathematica:

  (in) Resultant[
        a*(1 - t)^3 + 3*b*(1 - t)^2*t + 3*c*(1 - t)*t^2 + d*t^3 - i*y - j,
        e*(1 - t)^3 + 3*f*(1 - t)^2*t + 3*g*(1 - t)*t^2 + h*t^3 - y,       y]
  (out)  a     -   j     -
       3 a t   + 3 b t   +
       3 a t^2 - 6 b t^2 + 3 c t^2 -
         a t^3 + 3 b t^3 - 3 c t^3 + d t^3 -
     i ( e     -
       3 e t   + 3 f t   +
       3 e t^2 - 6 f t^2 + 3 g t^2 -
         e t^3 + 3 f t^3 - 3 g t^3 + h t^3 )

Solving this with Mathematica produces an expression with hundreds of terms;
instead, use Numeric Solutions recipe to solve the cubic.

The near-horizontal case, in terms of:  Ax^3 + Bx^2 + Cx + D == 0
    A =   (-(-e + 3*f - 3*g + h) + i*(-a + 3*b - 3*c + d)     )
    B = 3*(-( e - 2*f +   g    ) + i*( a - 2*b +   c    )     )
    C = 3*(-(-e +   f          ) + i*(-a +   b          )     )
    D =   (-( e                ) + i*( a                ) + j )

The near-vertical case, in terms of:  Ax^3 + Bx^2 + Cx + D == 0
    A =   ( (-a + 3*b - 3*c + d) - i*(-e + 3*f - 3*g + h)     )
    B = 3*( ( a - 2*b +   c    ) - i*( e - 2*f +   g    )     )
    C = 3*( (-a +   b          ) - i*(-e +   f          )     )
    D =   ( ( a                ) - i*( e                ) - j )

For horizontal lines:
(in) Resultant[
      a*(1 - t)^3 + 3*b*(1 - t)^2*t + 3*c*(1 - t)*t^2 + d*t^3 - j,
      e*(1 - t)^3 + 3*f*(1 - t)^2*t + 3*g*(1 - t)*t^2 + h*t^3 - y, y]
(out)  e     -   j     -
     3 e t   + 3 f t   +
     3 e t^2 - 6 f t^2 + 3 g t^2 -
       e t^3 + 3 f t^3 - 3 g t^3 + h t^3
 */

class LineCubicIntersections {
public:
    enum PinTPoint {
        kPointUninitialized,
        kPointInitialized
    };

    LineCubicIntersections(const SkDCubic& c, const SkDLine& l, SkIntersections* i)
        : fCubic(c)
        , fLine(l)
        , fIntersections(i)
        , fAllowNear(true) {
        i->setMax(4);
    }

    void allowNear(bool allow) {
        fAllowNear = allow;
    }

    void checkCoincident() {
        int last = fIntersections->used() - 1;
        for (int index = 0; index < last; ) {
            double cubicMidT = ((*fIntersections)[0][index] + (*fIntersections)[0][index + 1]) / 2;
            SkDPoint cubicMidPt = fCubic.ptAtT(cubicMidT);
            double t = fLine.nearPoint(cubicMidPt, nullptr);
            if (t < 0) {
                ++index;
                continue;
            }
            if (fIntersections->isCoincident(index)) {
                fIntersections->removeOne(index);
                --last;
            } else if (fIntersections->isCoincident(index + 1)) {
                fIntersections->removeOne(index + 1);
                --last;
            } else {
                fIntersections->setCoincident(index++);
            }
            fIntersections->setCoincident(index);
        }
    }

    // see parallel routine in line quadratic intersections
    int intersectRay(double roots[3]) {
        double adj = fLine[1].fX - fLine[0].fX;
        double opp = fLine[1].fY - fLine[0].fY;
        SkDCubic c;
        SkDEBUGCODE(c.fDebugGlobalState = fIntersections->globalState());
        for (int n = 0; n < 4; ++n) {
            c[n].fX = (fCubic[n].fY - fLine[0].fY) * adj - (fCubic[n].fX - fLine[0].fX) * opp;
        }
        double A, B, C, D;
        SkDCubic::Coefficients(&c[0].fX, &A, &B, &C, &D);
        int count = SkDCubic::RootsValidT(A, B, C, D, roots);
        for (int index = 0; index < count; ++index) {
            SkDPoint calcPt = c.ptAtT(roots[index]);
            if (!approximately_zero(calcPt.fX)) {
                for (int n = 0; n < 4; ++n) {
                    c[n].fY = (fCubic[n].fY - fLine[0].fY) * opp
                            + (fCubic[n].fX - fLine[0].fX) * adj;
                }
                double extremeTs[6];
                int extrema = SkDCubic::FindExtrema(&c[0].fX, extremeTs);
                count = c.searchRoots(extremeTs, extrema, 0, SkDCubic::kXAxis, roots);
                break;
            }
        }
        return count;
    }

    int intersect() {
        addExactEndPoints();
        if (fAllowNear) {
            addNearEndPoints();
        }
        double rootVals[3];
        int roots = intersectRay(rootVals);
        for (int index = 0; index < roots; ++index) {
            double cubicT = rootVals[index];
            double lineT = findLineT(cubicT);
            SkDPoint pt;
            if (pinTs(&cubicT, &lineT, &pt, kPointUninitialized) && uniqueAnswer(cubicT, pt)) {
                fIntersections->insert(cubicT, lineT, pt);
            }
        }
        checkCoincident();
        return fIntersections->used();
    }

    static int HorizontalIntersect(const SkDCubic& c, double axisIntercept, double roots[3]) {
        double A, B, C, D;
        SkDCubic::Coefficients(&c[0].fY, &A, &B, &C, &D);
        D -= axisIntercept;
        int count = SkDCubic::RootsValidT(A, B, C, D, roots);
        for (int index = 0; index < count; ++index) {
            SkDPoint calcPt = c.ptAtT(roots[index]);
            if (!approximately_equal(calcPt.fY, axisIntercept)) {
                double extremeTs[6];
                int extrema = SkDCubic::FindExtrema(&c[0].fY, extremeTs);
                count = c.searchRoots(extremeTs, extrema, axisIntercept, SkDCubic::kYAxis, roots);
                break;
            }
        }
        return count;
    }

    int horizontalIntersect(double axisIntercept, double left, double right, bool flipped) {
        addExactHorizontalEndPoints(left, right, axisIntercept);
        if (fAllowNear) {
            addNearHorizontalEndPoints(left, right, axisIntercept);
        }
        double roots[3];
        int count = HorizontalIntersect(fCubic, axisIntercept, roots);
        for (int index = 0; index < count; ++index) {
            double cubicT = roots[index];
            SkDPoint pt = { fCubic.ptAtT(cubicT).fX,  axisIntercept };
            double lineT = (pt.fX - left) / (right - left);
            if (pinTs(&cubicT, &lineT, &pt, kPointInitialized) && uniqueAnswer(cubicT, pt)) {
                fIntersections->insert(cubicT, lineT, pt);
            }
        }
        if (flipped) {
            fIntersections->flip();
        }
        checkCoincident();
        return fIntersections->used();
    }

        bool uniqueAnswer(double cubicT, const SkDPoint& pt) {
            for (int inner = 0; inner < fIntersections->used(); ++inner) {
                if (fIntersections->pt(inner) != pt) {
                    continue;
                }
                double existingCubicT = (*fIntersections)[0][inner];
                if (cubicT == existingCubicT) {
                    return false;
                }
                // check if midway on cubic is also same point. If so, discard this
                double cubicMidT = (existingCubicT + cubicT) / 2;
                SkDPoint cubicMidPt = fCubic.ptAtT(cubicMidT);
                if (cubicMidPt.approximatelyEqual(pt)) {
                    return false;
                }
            }
#if ONE_OFF_DEBUG
            SkDPoint cPt = fCubic.ptAtT(cubicT);
            SkDebugf("%s pt=(%1.9g,%1.9g) cPt=(%1.9g,%1.9g)\n", __FUNCTION__, pt.fX, pt.fY,
                    cPt.fX, cPt.fY);
#endif
            return true;
        }

    static int VerticalIntersect(const SkDCubic& c, double axisIntercept, double roots[3]) {
        double A, B, C, D;
        SkDCubic::Coefficients(&c[0].fX, &A, &B, &C, &D);
        D -= axisIntercept;
        int count = SkDCubic::RootsValidT(A, B, C, D, roots);
        for (int index = 0; index < count; ++index) {
            SkDPoint calcPt = c.ptAtT(roots[index]);
            if (!approximately_equal(calcPt.fX, axisIntercept)) {
                double extremeTs[6];
                int extrema = SkDCubic::FindExtrema(&c[0].fX, extremeTs);
                count = c.searchRoots(extremeTs, extrema, axisIntercept, SkDCubic::kXAxis, roots);
                break;
            }
        }
        return count;
    }

    int verticalIntersect(double axisIntercept, double top, double bottom, bool flipped) {
        addExactVerticalEndPoints(top, bottom, axisIntercept);
        if (fAllowNear) {
            addNearVerticalEndPoints(top, bottom, axisIntercept);
        }
        double roots[3];
        int count = VerticalIntersect(fCubic, axisIntercept, roots);
        for (int index = 0; index < count; ++index) {
            double cubicT = roots[index];
            SkDPoint pt = { axisIntercept, fCubic.ptAtT(cubicT).fY };
            double lineT = (pt.fY - top) / (bottom - top);
            if (pinTs(&cubicT, &lineT, &pt, kPointInitialized) && uniqueAnswer(cubicT, pt)) {
                fIntersections->insert(cubicT, lineT, pt);
            }
        }
        if (flipped) {
            fIntersections->flip();
        }
        checkCoincident();
        return fIntersections->used();
    }

    protected:

    void addExactEndPoints() {
        for (int cIndex = 0; cIndex < 4; cIndex += 3) {
            double lineT = fLine.exactPoint(fCubic[cIndex]);
            if (lineT < 0) {
                continue;
            }
            double cubicT = (double) (cIndex >> 1);
            fIntersections->insert(cubicT, lineT, fCubic[cIndex]);
        }
    }

    /* Note that this does not look for endpoints of the line that are near the cubic.
       These points are found later when check ends looks for missing points */
    void addNearEndPoints() {
        for (int cIndex = 0; cIndex < 4; cIndex += 3) {
            double cubicT = (double) (cIndex >> 1);
            if (fIntersections->hasT(cubicT)) {
                continue;
            }
            double lineT = fLine.nearPoint(fCubic[cIndex], nullptr);
            if (lineT < 0) {
                continue;
            }
            fIntersections->insert(cubicT, lineT, fCubic[cIndex]);
        }
        this->addLineNearEndPoints();
    }

    void addLineNearEndPoints() {
        for (int lIndex = 0; lIndex < 2; ++lIndex) {
            double lineT = (double) lIndex;
            if (fIntersections->hasOppT(lineT)) {
                continue;
            }
            double cubicT = ((SkDCurve*) &fCubic)->nearPoint(SkPath::kCubic_Verb,
                fLine[lIndex], fLine[!lIndex]);
            if (cubicT < 0) {
                continue;
            }
            fIntersections->insert(cubicT, lineT, fLine[lIndex]);
        }
    }

    void addExactHorizontalEndPoints(double left, double right, double y) {
        for (int cIndex = 0; cIndex < 4; cIndex += 3) {
            double lineT = SkDLine::ExactPointH(fCubic[cIndex], left, right, y);
            if (lineT < 0) {
                continue;
            }
            double cubicT = (double) (cIndex >> 1);
            fIntersections->insert(cubicT, lineT, fCubic[cIndex]);
        }
    }

    void addNearHorizontalEndPoints(double left, double right, double y) {
        for (int cIndex = 0; cIndex < 4; cIndex += 3) {
            double cubicT = (double) (cIndex >> 1);
            if (fIntersections->hasT(cubicT)) {
                continue;
            }
            double lineT = SkDLine::NearPointH(fCubic[cIndex], left, right, y);
            if (lineT < 0) {
                continue;
            }
            fIntersections->insert(cubicT, lineT, fCubic[cIndex]);
        }
        this->addLineNearEndPoints();
    }

    void addExactVerticalEndPoints(double top, double bottom, double x) {
        for (int cIndex = 0; cIndex < 4; cIndex += 3) {
            double lineT = SkDLine::ExactPointV(fCubic[cIndex], top, bottom, x);
            if (lineT < 0) {
                continue;
            }
            double cubicT = (double) (cIndex >> 1);
            fIntersections->insert(cubicT, lineT, fCubic[cIndex]);
        }
    }

    void addNearVerticalEndPoints(double top, double bottom, double x) {
        for (int cIndex = 0; cIndex < 4; cIndex += 3) {
            double cubicT = (double) (cIndex >> 1);
            if (fIntersections->hasT(cubicT)) {
                continue;
            }
            double lineT = SkDLine::NearPointV(fCubic[cIndex], top, bottom, x);
            if (lineT < 0) {
                continue;
            }
            fIntersections->insert(cubicT, lineT, fCubic[cIndex]);
        }
        this->addLineNearEndPoints();
    }

    double findLineT(double t) {
        SkDPoint xy = fCubic.ptAtT(t);
        double dx = fLine[1].fX - fLine[0].fX;
        double dy = fLine[1].fY - fLine[0].fY;
        if (fabs(dx) > fabs(dy)) {
            return (xy.fX - fLine[0].fX) / dx;
        }
        return (xy.fY - fLine[0].fY) / dy;
    }

    bool pinTs(double* cubicT, double* lineT, SkDPoint* pt, PinTPoint ptSet) {
        if (!approximately_one_or_less(*lineT)) {
            return false;
        }
        if (!approximately_zero_or_more(*lineT)) {
            return false;
        }
        double cT = *cubicT = SkPinT(*cubicT);
        double lT = *lineT = SkPinT(*lineT);
        SkDPoint lPt = fLine.ptAtT(lT);
        SkDPoint cPt = fCubic.ptAtT(cT);
        if (!lPt.roughlyEqual(cPt)) {
            return false;
        }
        // FIXME: if points are roughly equal but not approximately equal, need to do
        // a binary search like quad/quad intersection to find more precise t values
        if (lT == 0 || lT == 1 || (ptSet == kPointUninitialized && cT != 0 && cT != 1)) {
            *pt = lPt;
        } else if (ptSet == kPointUninitialized) {
            *pt = cPt;
        }
        SkPoint gridPt = pt->asSkPoint();
        if (gridPt == fLine[0].asSkPoint()) {
            *lineT = 0;
        } else if (gridPt == fLine[1].asSkPoint()) {
            *lineT = 1;
        }
        if (gridPt == fCubic[0].asSkPoint() && approximately_equal(*cubicT, 0)) {
            *cubicT = 0;
        } else if (gridPt == fCubic[3].asSkPoint() && approximately_equal(*cubicT, 1)) {
            *cubicT = 1;
        }
        return true;
    }

private:
    const SkDCubic& fCubic;
    const SkDLine& fLine;
    SkIntersections* fIntersections;
    bool fAllowNear;
};

int SkIntersections::horizontal(const SkDCubic& cubic, double left, double right, double y,
        bool flipped) {
    SkDLine line = {{{ left, y }, { right, y }}};
    LineCubicIntersections c(cubic, line, this);
    return c.horizontalIntersect(y, left, right, flipped);
}

int SkIntersections::vertical(const SkDCubic& cubic, double top, double bottom, double x,
        bool flipped) {
    SkDLine line = {{{ x, top }, { x, bottom }}};
    LineCubicIntersections c(cubic, line, this);
    return c.verticalIntersect(x, top, bottom, flipped);
}

int SkIntersections::intersect(const SkDCubic& cubic, const SkDLine& line) {
    LineCubicIntersections c(cubic, line, this);
    c.allowNear(fAllowNear);
    return c.intersect();
}

int SkIntersections::intersectRay(const SkDCubic& cubic, const SkDLine& line) {
    LineCubicIntersections c(cubic, line, this);
    fUsed = c.intersectRay(fT[0]);
    for (int index = 0; index < fUsed; ++index) {
        fPt[index] = cubic.ptAtT(fT[0][index]);
    }
    return fUsed;
}

// SkDCubic accessors to Intersection utilities

int SkDCubic::horizontalIntersect(double yIntercept, double roots[3]) const {
    return LineCubicIntersections::HorizontalIntersect(*this, yIntercept, roots);
}

int SkDCubic::verticalIntersect(double xIntercept, double roots[3]) const {
    return LineCubicIntersections::VerticalIntersect(*this, xIntercept, roots);
}
