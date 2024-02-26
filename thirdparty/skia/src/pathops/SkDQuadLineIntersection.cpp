/*
 * Copyright 2012 Google Inc.
 *
 * Use of this source code is governed by a BSD-style license that can be
 * found in the LICENSE file.
 */
#include "src/pathops/SkIntersections.h"
#include "src/pathops/SkPathOpsCurve.h"
#include "src/pathops/SkPathOpsLine.h"
#include "src/pathops/SkPathOpsQuad.h"

/*
Find the interection of a line and quadratic by solving for valid t values.

From http://stackoverflow.com/questions/1853637/how-to-find-the-mathematical-function-defining-a-bezier-curve

"A Bezier curve is a parametric function. A quadratic Bezier curve (i.e. three
control points) can be expressed as: F(t) = A(1 - t)^2 + B(1 - t)t + Ct^2 where
A, B and C are points and t goes from zero to one.

This will give you two equations:

  x = a(1 - t)^2 + b(1 - t)t + ct^2
  y = d(1 - t)^2 + e(1 - t)t + ft^2

If you add for instance the line equation (y = kx + m) to that, you'll end up
with three equations and three unknowns (x, y and t)."

Similar to above, the quadratic is represented as
  x = a(1-t)^2 + 2b(1-t)t + ct^2
  y = d(1-t)^2 + 2e(1-t)t + ft^2
and the line as
  y = g*x + h

Using Mathematica, solve for the values of t where the quadratic intersects the
line:

  (in)  t1 = Resultant[a*(1 - t)^2 + 2*b*(1 - t)*t + c*t^2 - x,
                       d*(1 - t)^2 + 2*e*(1 - t)*t  + f*t^2 - g*x - h, x]
  (out) -d + h + 2 d t - 2 e t - d t^2 + 2 e t^2 - f t^2 +
         g  (a - 2 a t + 2 b t + a t^2 - 2 b t^2 + c t^2)
  (in)  Solve[t1 == 0, t]
  (out) {
    {t -> (-2 d + 2 e +   2 a g - 2 b g    -
      Sqrt[(2 d - 2 e -   2 a g + 2 b g)^2 -
          4 (-d + 2 e - f + a g - 2 b g    + c g) (-d + a g + h)]) /
         (2 (-d + 2 e - f + a g - 2 b g    + c g))
         },
    {t -> (-2 d + 2 e +   2 a g - 2 b g    +
      Sqrt[(2 d - 2 e -   2 a g + 2 b g)^2 -
          4 (-d + 2 e - f + a g - 2 b g    + c g) (-d + a g + h)]) /
         (2 (-d + 2 e - f + a g - 2 b g    + c g))
         }
        }

Using the results above (when the line tends towards horizontal)
       A =   (-(d - 2*e + f) + g*(a - 2*b + c)     )
       B = 2*( (d -   e    ) - g*(a -   b    )     )
       C =   (-(d          ) + g*(a          ) + h )

If g goes to infinity, we can rewrite the line in terms of x.
  x = g'*y + h'

And solve accordingly in Mathematica:

  (in)  t2 = Resultant[a*(1 - t)^2 + 2*b*(1 - t)*t + c*t^2 - g'*y - h',
                       d*(1 - t)^2 + 2*e*(1 - t)*t  + f*t^2 - y, y]
  (out)  a - h' - 2 a t + 2 b t + a t^2 - 2 b t^2 + c t^2 -
         g'  (d - 2 d t + 2 e t + d t^2 - 2 e t^2 + f t^2)
  (in)  Solve[t2 == 0, t]
  (out) {
    {t -> (2 a - 2 b -   2 d g' + 2 e g'    -
    Sqrt[(-2 a + 2 b +   2 d g' - 2 e g')^2 -
          4 (a - 2 b + c - d g' + 2 e g' - f g') (a - d g' - h')]) /
         (2 (a - 2 b + c - d g' + 2 e g' - f g'))
         },
    {t -> (2 a - 2 b -   2 d g' + 2 e g'    +
    Sqrt[(-2 a + 2 b +   2 d g' - 2 e g')^2 -
          4 (a - 2 b + c - d g' + 2 e g' - f g') (a - d g' - h')])/
         (2 (a - 2 b + c - d g' + 2 e g' - f g'))
         }
        }

Thus, if the slope of the line tends towards vertical, we use:
       A =   ( (a - 2*b + c) - g'*(d  - 2*e + f)      )
       B = 2*(-(a -   b    ) + g'*(d  -   e    )      )
       C =   ( (a          ) - g'*(d           ) - h' )
 */

class LineQuadraticIntersections {
public:
    enum PinTPoint {
        kPointUninitialized,
        kPointInitialized
    };

    LineQuadraticIntersections(const SkDQuad& q, const SkDLine& l, SkIntersections* i)
        : fQuad(q)
        , fLine(&l)
        , fIntersections(i)
        , fAllowNear(true) {
        i->setMax(5);  // allow short partial coincidence plus discrete intersections
    }

    LineQuadraticIntersections(const SkDQuad& q)
        : fQuad(q)
        SkDEBUGPARAMS(fLine(nullptr))
        SkDEBUGPARAMS(fIntersections(nullptr))
        SkDEBUGPARAMS(fAllowNear(false)) {
    }

    void allowNear(bool allow) {
        fAllowNear = allow;
    }

    void checkCoincident() {
        int last = fIntersections->used() - 1;
        for (int index = 0; index < last; ) {
            double quadMidT = ((*fIntersections)[0][index] + (*fIntersections)[0][index + 1]) / 2;
            SkDPoint quadMidPt = fQuad.ptAtT(quadMidT);
            double t = fLine->nearPoint(quadMidPt, nullptr);
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

    int intersectRay(double roots[2]) {
    /*
        solve by rotating line+quad so line is horizontal, then finding the roots
        set up matrix to rotate quad to x-axis
        |cos(a) -sin(a)|
        |sin(a)  cos(a)|
        note that cos(a) = A(djacent) / Hypoteneuse
                  sin(a) = O(pposite) / Hypoteneuse
        since we are computing Ts, we can ignore hypoteneuse, the scale factor:
        |  A     -O    |
        |  O      A    |
        A = line[1].fX - line[0].fX (adjacent side of the right triangle)
        O = line[1].fY - line[0].fY (opposite side of the right triangle)
        for each of the three points (e.g. n = 0 to 2)
        quad[n].fY' = (quad[n].fY - line[0].fY) * A - (quad[n].fX - line[0].fX) * O
    */
        double adj = (*fLine)[1].fX - (*fLine)[0].fX;
        double opp = (*fLine)[1].fY - (*fLine)[0].fY;
        double r[3];
        for (int n = 0; n < 3; ++n) {
            r[n] = (fQuad[n].fY - (*fLine)[0].fY) * adj - (fQuad[n].fX - (*fLine)[0].fX) * opp;
        }
        double A = r[2];
        double B = r[1];
        double C = r[0];
        A += C - 2 * B;  // A = a - 2*b + c
        B -= C;  // B = -(b - c)
        return SkDQuad::RootsValidT(A, 2 * B, C, roots);
    }

    int intersect() {
        addExactEndPoints();
        if (fAllowNear) {
            addNearEndPoints();
        }
        double rootVals[2];
        int roots = intersectRay(rootVals);
        for (int index = 0; index < roots; ++index) {
            double quadT = rootVals[index];
            double lineT = findLineT(quadT);
            SkDPoint pt;
            if (pinTs(&quadT, &lineT, &pt, kPointUninitialized) && uniqueAnswer(quadT, pt)) {
                fIntersections->insert(quadT, lineT, pt);
            }
        }
        checkCoincident();
        return fIntersections->used();
    }

    int horizontalIntersect(double axisIntercept, double roots[2]) {
        double D = fQuad[2].fY;  // f
        double E = fQuad[1].fY;  // e
        double F = fQuad[0].fY;  // d
        D += F - 2 * E;         // D = d - 2*e + f
        E -= F;                 // E = -(d - e)
        F -= axisIntercept;
        return SkDQuad::RootsValidT(D, 2 * E, F, roots);
    }

    int horizontalIntersect(double axisIntercept, double left, double right, bool flipped) {
        addExactHorizontalEndPoints(left, right, axisIntercept);
        if (fAllowNear) {
            addNearHorizontalEndPoints(left, right, axisIntercept);
        }
        double rootVals[2];
        int roots = horizontalIntersect(axisIntercept, rootVals);
        for (int index = 0; index < roots; ++index) {
            double quadT = rootVals[index];
            SkDPoint pt = fQuad.ptAtT(quadT);
            double lineT = (pt.fX - left) / (right - left);
            if (pinTs(&quadT, &lineT, &pt, kPointInitialized) && uniqueAnswer(quadT, pt)) {
                fIntersections->insert(quadT, lineT, pt);
            }
        }
        if (flipped) {
            fIntersections->flip();
        }
        checkCoincident();
        return fIntersections->used();
    }

    bool uniqueAnswer(double quadT, const SkDPoint& pt) {
        for (int inner = 0; inner < fIntersections->used(); ++inner) {
            if (fIntersections->pt(inner) != pt) {
                continue;
            }
            double existingQuadT = (*fIntersections)[0][inner];
            if (quadT == existingQuadT) {
                return false;
            }
            // check if midway on quad is also same point. If so, discard this
            double quadMidT = (existingQuadT + quadT) / 2;
            SkDPoint quadMidPt = fQuad.ptAtT(quadMidT);
            if (quadMidPt.approximatelyEqual(pt)) {
                return false;
            }
        }
#if ONE_OFF_DEBUG
        SkDPoint qPt = fQuad.ptAtT(quadT);
        SkDebugf("%s pt=(%1.9g,%1.9g) cPt=(%1.9g,%1.9g)\n", __FUNCTION__, pt.fX, pt.fY,
                qPt.fX, qPt.fY);
#endif
        return true;
    }

    int verticalIntersect(double axisIntercept, double roots[2]) {
        double D = fQuad[2].fX;  // f
        double E = fQuad[1].fX;  // e
        double F = fQuad[0].fX;  // d
        D += F - 2 * E;         // D = d - 2*e + f
        E -= F;                 // E = -(d - e)
        F -= axisIntercept;
        return SkDQuad::RootsValidT(D, 2 * E, F, roots);
    }

    int verticalIntersect(double axisIntercept, double top, double bottom, bool flipped) {
        addExactVerticalEndPoints(top, bottom, axisIntercept);
        if (fAllowNear) {
            addNearVerticalEndPoints(top, bottom, axisIntercept);
        }
        double rootVals[2];
        int roots = verticalIntersect(axisIntercept, rootVals);
        for (int index = 0; index < roots; ++index) {
            double quadT = rootVals[index];
            SkDPoint pt = fQuad.ptAtT(quadT);
            double lineT = (pt.fY - top) / (bottom - top);
            if (pinTs(&quadT, &lineT, &pt, kPointInitialized) && uniqueAnswer(quadT, pt)) {
                fIntersections->insert(quadT, lineT, pt);
            }
        }
        if (flipped) {
            fIntersections->flip();
        }
        checkCoincident();
        return fIntersections->used();
    }

protected:
    // add endpoints first to get zero and one t values exactly
    void addExactEndPoints() {
        for (int qIndex = 0; qIndex < 3; qIndex += 2) {
            double lineT = fLine->exactPoint(fQuad[qIndex]);
            if (lineT < 0) {
                continue;
            }
            double quadT = (double) (qIndex >> 1);
            fIntersections->insert(quadT, lineT, fQuad[qIndex]);
        }
    }

    void addNearEndPoints() {
        for (int qIndex = 0; qIndex < 3; qIndex += 2) {
            double quadT = (double) (qIndex >> 1);
            if (fIntersections->hasT(quadT)) {
                continue;
            }
            double lineT = fLine->nearPoint(fQuad[qIndex], nullptr);
            if (lineT < 0) {
                continue;
            }
            fIntersections->insert(quadT, lineT, fQuad[qIndex]);
        }
        this->addLineNearEndPoints();
    }

    void addLineNearEndPoints() {
        for (int lIndex = 0; lIndex < 2; ++lIndex) {
            double lineT = (double) lIndex;
            if (fIntersections->hasOppT(lineT)) {
                continue;
            }
            double quadT = ((SkDCurve*) &fQuad)->nearPoint(SkPath::kQuad_Verb,
                    (*fLine)[lIndex], (*fLine)[!lIndex]);
            if (quadT < 0) {
                continue;
            }
            fIntersections->insert(quadT, lineT, (*fLine)[lIndex]);
        }
    }

    void addExactHorizontalEndPoints(double left, double right, double y) {
        for (int qIndex = 0; qIndex < 3; qIndex += 2) {
            double lineT = SkDLine::ExactPointH(fQuad[qIndex], left, right, y);
            if (lineT < 0) {
                continue;
            }
            double quadT = (double) (qIndex >> 1);
            fIntersections->insert(quadT, lineT, fQuad[qIndex]);
        }
    }

    void addNearHorizontalEndPoints(double left, double right, double y) {
        for (int qIndex = 0; qIndex < 3; qIndex += 2) {
            double quadT = (double) (qIndex >> 1);
            if (fIntersections->hasT(quadT)) {
                continue;
            }
            double lineT = SkDLine::NearPointH(fQuad[qIndex], left, right, y);
            if (lineT < 0) {
                continue;
            }
            fIntersections->insert(quadT, lineT, fQuad[qIndex]);
        }
        this->addLineNearEndPoints();
    }

    void addExactVerticalEndPoints(double top, double bottom, double x) {
        for (int qIndex = 0; qIndex < 3; qIndex += 2) {
            double lineT = SkDLine::ExactPointV(fQuad[qIndex], top, bottom, x);
            if (lineT < 0) {
                continue;
            }
            double quadT = (double) (qIndex >> 1);
            fIntersections->insert(quadT, lineT, fQuad[qIndex]);
        }
    }

    void addNearVerticalEndPoints(double top, double bottom, double x) {
        for (int qIndex = 0; qIndex < 3; qIndex += 2) {
            double quadT = (double) (qIndex >> 1);
            if (fIntersections->hasT(quadT)) {
                continue;
            }
            double lineT = SkDLine::NearPointV(fQuad[qIndex], top, bottom, x);
            if (lineT < 0) {
                continue;
            }
            fIntersections->insert(quadT, lineT, fQuad[qIndex]);
        }
        this->addLineNearEndPoints();
    }

    double findLineT(double t) {
        SkDPoint xy = fQuad.ptAtT(t);
        double dx = (*fLine)[1].fX - (*fLine)[0].fX;
        double dy = (*fLine)[1].fY - (*fLine)[0].fY;
        if (fabs(dx) > fabs(dy)) {
            return (xy.fX - (*fLine)[0].fX) / dx;
        }
        return (xy.fY - (*fLine)[0].fY) / dy;
    }

    bool pinTs(double* quadT, double* lineT, SkDPoint* pt, PinTPoint ptSet) {
        if (!approximately_one_or_less_double(*lineT)) {
            return false;
        }
        if (!approximately_zero_or_more_double(*lineT)) {
            return false;
        }
        double qT = *quadT = SkPinT(*quadT);
        double lT = *lineT = SkPinT(*lineT);
        if (lT == 0 || lT == 1 || (ptSet == kPointUninitialized && qT != 0 && qT != 1)) {
            *pt = (*fLine).ptAtT(lT);
        } else if (ptSet == kPointUninitialized) {
            *pt = fQuad.ptAtT(qT);
        }
        SkPoint gridPt = pt->asSkPoint();
        if (SkDPoint::ApproximatelyEqual(gridPt, (*fLine)[0].asSkPoint())) {
            *pt = (*fLine)[0];
            *lineT = 0;
        } else if (SkDPoint::ApproximatelyEqual(gridPt, (*fLine)[1].asSkPoint())) {
            *pt = (*fLine)[1];
            *lineT = 1;
        }
        if (fIntersections->used() > 0 && approximately_equal((*fIntersections)[1][0], *lineT)) {
            return false;
        }
        if (gridPt == fQuad[0].asSkPoint()) {
            *pt = fQuad[0];
            *quadT = 0;
        } else if (gridPt == fQuad[2].asSkPoint()) {
            *pt = fQuad[2];
            *quadT = 1;
        }
        return true;
    }

private:
    const SkDQuad& fQuad;
    const SkDLine* fLine;
    SkIntersections* fIntersections;
    bool fAllowNear;
};

int SkIntersections::horizontal(const SkDQuad& quad, double left, double right, double y,
                                bool flipped) {
    SkDLine line = {{{ left, y }, { right, y }}};
    LineQuadraticIntersections q(quad, line, this);
    return q.horizontalIntersect(y, left, right, flipped);
}

int SkIntersections::vertical(const SkDQuad& quad, double top, double bottom, double x,
                              bool flipped) {
    SkDLine line = {{{ x, top }, { x, bottom }}};
    LineQuadraticIntersections q(quad, line, this);
    return q.verticalIntersect(x, top, bottom, flipped);
}

int SkIntersections::intersect(const SkDQuad& quad, const SkDLine& line) {
    LineQuadraticIntersections q(quad, line, this);
    q.allowNear(fAllowNear);
    return q.intersect();
}

int SkIntersections::intersectRay(const SkDQuad& quad, const SkDLine& line) {
    LineQuadraticIntersections q(quad, line, this);
    fUsed = q.intersectRay(fT[0]);
    for (int index = 0; index < fUsed; ++index) {
        fPt[index] = quad.ptAtT(fT[0][index]);
    }
    return fUsed;
}

int SkIntersections::HorizontalIntercept(const SkDQuad& quad, SkScalar y, double* roots) {
    LineQuadraticIntersections q(quad);
    return q.horizontalIntersect(y, roots);
}

int SkIntersections::VerticalIntercept(const SkDQuad& quad, SkScalar x, double* roots) {
    LineQuadraticIntersections q(quad);
    return q.verticalIntersect(x, roots);
}

// SkDQuad accessors to Intersection utilities

int SkDQuad::horizontalIntersect(double yIntercept, double roots[2]) const {
    return SkIntersections::HorizontalIntercept(*this, yIntercept, roots);
}

int SkDQuad::verticalIntersect(double xIntercept, double roots[2]) const {
    return SkIntersections::VerticalIntercept(*this, xIntercept, roots);
}
