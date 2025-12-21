
#include "convergent-curve-ordering.h"

#include "arithmetics.hpp"
#include "Vector2.hpp"

/*
 * For non-degenerate curves A(t), B(t) (ones where all control points are distinct) both originating at P = A(0) = B(0) = *corner,
 * we are computing the limit of
 *
 *     sign(crossProduct( A(t / |A'(0)|) - P, B(t / |B'(0)|) - P ))
 *
 * for t -> 0 from 1. Of note is that the curves' parameter has to be normed by the first derivative at P,
 * which ensures that the limit approaches P at the same rate along both curves - omitting this was the main error of earlier versions of deconverge.
 *
 * For degenerate cubic curves (ones where the first control point equals the origin point), the denominator |A'(0)| is zero,
 * so to address that, we approach with the square root of t and use the derivative of A(sqrt(t)), which at t = 0 equals A''(0)/2
 * Therefore, in these cases, we replace one factor of the cross product with A(sqrt(2*t / |A''(0)|)) - P
 *
 * The cross product results in a polynomial (in respect to t or t^2 in the degenerate case),
 * the limit of sign of which at zero can be determined by the lowest order non-zero derivative,
 * which equals to the sign of the first non-zero polynomial coefficient in the order of increasing exponents.
 *
 * The polynomial's constant and linear terms are zero, so the first derivative is definitely zero as well.
 * The second derivative is assumed to be zero (or near zero) due to the curves being convergent - this is an input requirement
 * (otherwise the correct result is the sign of the cross product of their directions at t = 0).
 * Therefore, we skip the first and second derivatives.
 */

namespace msdfgen {

static void simplifyDegenerateCurve(Point2 *controlPoints, int &order) {
    if (order == 3 && (controlPoints[1] == controlPoints[0] || controlPoints[1] == controlPoints[3]) && (controlPoints[2] == controlPoints[0] || controlPoints[2] == controlPoints[3])) {
        controlPoints[1] = controlPoints[3];
        order = 1;
    }
    if (order == 2 && (controlPoints[1] == controlPoints[0] || controlPoints[1] == controlPoints[2])) {
        controlPoints[1] = controlPoints[2];
        order = 1;
    }
    if (order == 1 && controlPoints[0] == controlPoints[1])
        order = 0;
}

int convergentCurveOrdering(const Point2 *corner, int controlPointsBefore, int controlPointsAfter) {
    if (!(controlPointsBefore > 0 && controlPointsAfter > 0))
        return 0;
    Vector2 a1, a2, a3, b1, b2, b3;
    a1 = *(corner-1)-*corner;
    b1 = *(corner+1)-*corner;
    if (controlPointsBefore >= 2)
        a2 = *(corner-2)-*(corner-1)-a1;
    if (controlPointsAfter >= 2)
        b2 = *(corner+2)-*(corner+1)-b1;
    if (controlPointsBefore >= 3) {
        a3 = *(corner-3)-*(corner-2)-(*(corner-2)-*(corner-1))-a2;
        a2 *= 3;
    }
    if (controlPointsAfter >= 3) {
        b3 = *(corner+3)-*(corner+2)-(*(corner+2)-*(corner+1))-b2;
        b2 *= 3;
    }
    a1 *= controlPointsBefore;
    b1 *= controlPointsAfter;
    // Non-degenerate case
    if (a1 && b1) {
        double as = a1.length();
        double bs = b1.length();
        // Third derivative
        if (double d = as*crossProduct(a1, b2) + bs*crossProduct(a2, b1))
            return sign(d);
        // Fourth derivative
        if (double d = as*as*crossProduct(a1, b3) + as*bs*crossProduct(a2, b2) + bs*bs*crossProduct(a3, b1))
            return sign(d);
        // Fifth derivative
        if (double d = as*crossProduct(a2, b3) + bs*crossProduct(a3, b2))
            return sign(d);
        // Sixth derivative
        return sign(crossProduct(a3, b3));
    }
    // Degenerate curve after corner (control point after corner equals corner)
    int s = 1;
    if (a1) { // !b1
        // Swap aN <-> bN and handle in if (b1)
        b1 = a1;
        a1 = b2, b2 = a2, a2 = a1;
        a1 = b3, b3 = a3, a3 = a1;
        s = -1; // make sure to also flip output
    }
    // Degenerate curve before corner (control point before corner equals corner)
    if (b1) { // !a1
        // Two-and-a-half-th derivative
        if (double d = crossProduct(a3, b1))
            return s*sign(d);
        // Third derivative
        if (double d = crossProduct(a2, b2))
            return s*sign(d);
        // Three-and-a-half-th derivative
        if (double d = crossProduct(a3, b2))
            return s*sign(d);
        // Fourth derivative
        if (double d = crossProduct(a2, b3))
            return s*sign(d);
        // Four-and-a-half-th derivative
        return s*sign(crossProduct(a3, b3));
    }
    // Degenerate curves on both sides of the corner (control point before and after corner equals corner)
    { // !a1 && !b1
        // Two-and-a-half-th derivative
        if (double d = sqrt(a2.length())*crossProduct(a2, b3) + sqrt(b2.length())*crossProduct(a3, b2))
            return sign(d);
        // Third derivative
        return sign(crossProduct(a3, b3));
    }
}

int convergentCurveOrdering(const EdgeSegment *a, const EdgeSegment *b) {
    Point2 controlPoints[12];
    Point2 *corner = controlPoints+4;
    Point2 *aCpTmp = controlPoints+8;
    int aOrder = int(a->type());
    int bOrder = int(b->type());
    if (!(aOrder >= 1 && aOrder <= 3 && bOrder >= 1 && bOrder <= 3)) {
        // Not implemented - only linear, quadratic, and cubic curves supported
        return 0;
    }
    for (int i = 0; i <= aOrder; ++i)
        aCpTmp[i] = a->controlPoints()[i];
    for (int i = 0; i <= bOrder; ++i)
        corner[i] = b->controlPoints()[i];
    if (aCpTmp[aOrder] != *corner)
        return 0;
    simplifyDegenerateCurve(aCpTmp, aOrder);
    simplifyDegenerateCurve(corner, bOrder);
    for (int i = 0; i < aOrder; ++i)
        corner[i-aOrder] = aCpTmp[i];
    return convergentCurveOrdering(corner, aOrder, bOrder);
}

}
