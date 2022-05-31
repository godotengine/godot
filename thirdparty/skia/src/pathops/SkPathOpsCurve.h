/*
 * Copyright 2012 Google Inc.
 *
 * Use of this source code is governed by a BSD-style license that can be
 * found in the LICENSE file.
 */
#ifndef SkPathOpsCurve_DEFINE
#define SkPathOpsCurve_DEFINE

#include "src/pathops/SkIntersections.h"

#ifndef SK_RELEASE
#include "include/core/SkPath.h"
#endif

struct SkPathOpsBounds;

struct SkOpCurve {
    SkPoint fPts[4];
    SkScalar fWeight;
    SkDEBUGCODE(SkPath::Verb fVerb);

    const SkPoint& operator[](int n) const {
        SkASSERT(n >= 0 && n <= SkPathOpsVerbToPoints(fVerb));
        return fPts[n];
    }

    void dump() const;

    void set(const SkDQuad& quad) {
        for (int index = 0; index < SkDQuad::kPointCount; ++index) {
            fPts[index] = quad[index].asSkPoint();
        }
        SkDEBUGCODE(fWeight = 1);
        SkDEBUGCODE(fVerb = SkPath::kQuad_Verb);
    }

    void set(const SkDCubic& cubic) {
        for (int index = 0; index < SkDCubic::kPointCount; ++index) {
            fPts[index] = cubic[index].asSkPoint();
        }
        SkDEBUGCODE(fWeight = 1);
        SkDEBUGCODE(fVerb = SkPath::kCubic_Verb);
    }

};

struct SkDCurve {
    union {
        SkDLine fLine;
        SkDQuad fQuad;
        SkDConic fConic;
        SkDCubic fCubic;
    };
    SkDEBUGCODE(SkPath::Verb fVerb);

    const SkDPoint& operator[](int n) const {
        SkASSERT(n >= 0 && n <= SkPathOpsVerbToPoints(fVerb));
        return fCubic[n];
    }

    SkDPoint& operator[](int n) {
        SkASSERT(n >= 0 && n <= SkPathOpsVerbToPoints(fVerb));
        return fCubic[n];
    }

    SkDPoint conicTop(const SkPoint curve[3], SkScalar curveWeight,
                      double s, double e, double* topT);
    SkDPoint cubicTop(const SkPoint curve[4], SkScalar , double s, double e, double* topT);
    void dump() const;
    void dumpID(int ) const;
    SkDPoint lineTop(const SkPoint[2], SkScalar , double , double , double* topT);
    double nearPoint(SkPath::Verb verb, const SkDPoint& xy, const SkDPoint& opp) const;
    void offset(SkPath::Verb verb, const SkDVector& );
    SkDPoint quadTop(const SkPoint curve[3], SkScalar , double s, double e, double* topT);

    void setConicBounds(const SkPoint curve[3], SkScalar curveWeight,
                        double s, double e, SkPathOpsBounds* );
    void setCubicBounds(const SkPoint curve[4], SkScalar ,
                        double s, double e, SkPathOpsBounds* );
    void setQuadBounds(const SkPoint curve[3], SkScalar ,
                       double s, double e, SkPathOpsBounds*);
};

class SkDCurveSweep {
public:
    bool isCurve() const { return fIsCurve; }
    bool isOrdered() const { return fOrdered; }
    void setCurveHullSweep(SkPath::Verb verb);

    SkDCurve fCurve;
    SkDVector fSweep[2];
private:
    bool fIsCurve;
    bool fOrdered;  // cleared when a cubic's control point isn't between the sweep vectors

};

extern SkDPoint (SkDCurve::* const Top[])(const SkPoint curve[], SkScalar cWeight,
    double tStart, double tEnd, double* topT);

static SkDPoint dline_xy_at_t(const SkPoint a[2], SkScalar , double t) {
    SkDLine line;
    line.set(a);
    return line.ptAtT(t);
}

static SkDPoint dquad_xy_at_t(const SkPoint a[3], SkScalar , double t) {
    SkDQuad quad;
    quad.set(a);
    return quad.ptAtT(t);
}

static SkDPoint dconic_xy_at_t(const SkPoint a[3], SkScalar weight, double t) {
    SkDConic conic;
    conic.set(a, weight);
    return conic.ptAtT(t);
}

static SkDPoint dcubic_xy_at_t(const SkPoint a[4], SkScalar , double t) {
    SkDCubic cubic;
    cubic.set(a);
    return cubic.ptAtT(t);
}

static SkDPoint (* const CurveDPointAtT[])(const SkPoint[], SkScalar , double ) = {
    nullptr,
    dline_xy_at_t,
    dquad_xy_at_t,
    dconic_xy_at_t,
    dcubic_xy_at_t
};

static SkDPoint ddline_xy_at_t(const SkDCurve& c, double t) {
    return c.fLine.ptAtT(t);
}

static SkDPoint ddquad_xy_at_t(const SkDCurve& c, double t) {
    return c.fQuad.ptAtT(t);
}

static SkDPoint ddconic_xy_at_t(const SkDCurve& c, double t) {
    return c.fConic.ptAtT(t);
}

static SkDPoint ddcubic_xy_at_t(const SkDCurve& c, double t) {
    return c.fCubic.ptAtT(t);
}

static SkDPoint (* const CurveDDPointAtT[])(const SkDCurve& , double ) = {
    nullptr,
    ddline_xy_at_t,
    ddquad_xy_at_t,
    ddconic_xy_at_t,
    ddcubic_xy_at_t
};

static SkPoint fline_xy_at_t(const SkPoint a[2], SkScalar weight, double t) {
    return dline_xy_at_t(a, weight, t).asSkPoint();
}

static SkPoint fquad_xy_at_t(const SkPoint a[3], SkScalar weight, double t) {
    return dquad_xy_at_t(a, weight, t).asSkPoint();
}

static SkPoint fconic_xy_at_t(const SkPoint a[3], SkScalar weight, double t) {
    return dconic_xy_at_t(a, weight, t).asSkPoint();
}

static SkPoint fcubic_xy_at_t(const SkPoint a[4], SkScalar weight, double t) {
    return dcubic_xy_at_t(a, weight, t).asSkPoint();
}

static SkPoint (* const CurvePointAtT[])(const SkPoint[], SkScalar , double ) = {
    nullptr,
    fline_xy_at_t,
    fquad_xy_at_t,
    fconic_xy_at_t,
    fcubic_xy_at_t
};

static SkDVector dline_dxdy_at_t(const SkPoint a[2], SkScalar , double ) {
    SkDLine line;
    line.set(a);
    return line[1] - line[0];
}

static SkDVector dquad_dxdy_at_t(const SkPoint a[3], SkScalar , double t) {
    SkDQuad quad;
    quad.set(a);
    return quad.dxdyAtT(t);
}

static SkDVector dconic_dxdy_at_t(const SkPoint a[3], SkScalar weight, double t) {
    SkDConic conic;
    conic.set(a, weight);
    return conic.dxdyAtT(t);
}

static SkDVector dcubic_dxdy_at_t(const SkPoint a[4], SkScalar , double t) {
    SkDCubic cubic;
    cubic.set(a);
    return cubic.dxdyAtT(t);
}

static SkDVector (* const CurveDSlopeAtT[])(const SkPoint[], SkScalar , double ) = {
    nullptr,
    dline_dxdy_at_t,
    dquad_dxdy_at_t,
    dconic_dxdy_at_t,
    dcubic_dxdy_at_t
};

static SkDVector ddline_dxdy_at_t(const SkDCurve& c, double ) {
    return c.fLine.fPts[1] - c.fLine.fPts[0];
}

static SkDVector ddquad_dxdy_at_t(const SkDCurve& c, double t) {
    return c.fQuad.dxdyAtT(t);
}

static SkDVector ddconic_dxdy_at_t(const SkDCurve& c, double t) {
    return c.fConic.dxdyAtT(t);
}

static SkDVector ddcubic_dxdy_at_t(const SkDCurve& c, double t) {
    return c.fCubic.dxdyAtT(t);
}

static SkDVector (* const CurveDDSlopeAtT[])(const SkDCurve& , double ) = {
    nullptr,
    ddline_dxdy_at_t,
    ddquad_dxdy_at_t,
    ddconic_dxdy_at_t,
    ddcubic_dxdy_at_t
};

static SkVector fline_dxdy_at_t(const SkPoint a[2], SkScalar , double ) {
    return a[1] - a[0];
}

static SkVector fquad_dxdy_at_t(const SkPoint a[3], SkScalar weight, double t) {
    return dquad_dxdy_at_t(a, weight, t).asSkVector();
}

static SkVector fconic_dxdy_at_t(const SkPoint a[3], SkScalar weight, double t) {
    return dconic_dxdy_at_t(a, weight, t).asSkVector();
}

static SkVector fcubic_dxdy_at_t(const SkPoint a[4], SkScalar weight, double t) {
    return dcubic_dxdy_at_t(a, weight, t).asSkVector();
}

static SkVector (* const CurveSlopeAtT[])(const SkPoint[], SkScalar , double ) = {
    nullptr,
    fline_dxdy_at_t,
    fquad_dxdy_at_t,
    fconic_dxdy_at_t,
    fcubic_dxdy_at_t
};

static bool line_is_vertical(const SkPoint a[2], SkScalar , double startT, double endT) {
    SkDLine line;
    line.set(a);
    SkDPoint dst[2] = { line.ptAtT(startT), line.ptAtT(endT) };
    return AlmostEqualUlps(dst[0].fX, dst[1].fX);
}

static bool quad_is_vertical(const SkPoint a[3], SkScalar , double startT, double endT) {
    SkDQuad quad;
    quad.set(a);
    SkDQuad dst = quad.subDivide(startT, endT);
    return AlmostEqualUlps(dst[0].fX, dst[1].fX) && AlmostEqualUlps(dst[1].fX, dst[2].fX);
}

static bool conic_is_vertical(const SkPoint a[3], SkScalar weight, double startT, double endT) {
    SkDConic conic;
    conic.set(a, weight);
    SkDConic dst = conic.subDivide(startT, endT);
    return AlmostEqualUlps(dst[0].fX, dst[1].fX) && AlmostEqualUlps(dst[1].fX, dst[2].fX);
}

static bool cubic_is_vertical(const SkPoint a[4], SkScalar , double startT, double endT) {
    SkDCubic cubic;
    cubic.set(a);
    SkDCubic dst = cubic.subDivide(startT, endT);
    return AlmostEqualUlps(dst[0].fX, dst[1].fX) && AlmostEqualUlps(dst[1].fX, dst[2].fX)
            && AlmostEqualUlps(dst[2].fX, dst[3].fX);
}

static bool (* const CurveIsVertical[])(const SkPoint[], SkScalar , double , double) = {
    nullptr,
    line_is_vertical,
    quad_is_vertical,
    conic_is_vertical,
    cubic_is_vertical
};

static void line_intersect_ray(const SkPoint a[2], SkScalar , const SkDLine& ray,
        SkIntersections* i) {
    SkDLine line;
    line.set(a);
    i->intersectRay(line, ray);
}

static void quad_intersect_ray(const SkPoint a[3], SkScalar , const SkDLine& ray,
        SkIntersections* i) {
    SkDQuad quad;
    quad.set(a);
    i->intersectRay(quad, ray);
}

static void conic_intersect_ray(const SkPoint a[3], SkScalar weight, const SkDLine& ray,
        SkIntersections* i) {
    SkDConic conic;
    conic.set(a, weight);
    i->intersectRay(conic, ray);
}

static void cubic_intersect_ray(const SkPoint a[4], SkScalar , const SkDLine& ray,
        SkIntersections* i) {
    SkDCubic cubic;
    cubic.set(a);
    i->intersectRay(cubic, ray);
}

static void (* const CurveIntersectRay[])(const SkPoint[] , SkScalar , const SkDLine& ,
        SkIntersections* ) = {
    nullptr,
    line_intersect_ray,
    quad_intersect_ray,
    conic_intersect_ray,
    cubic_intersect_ray
};

static void dline_intersect_ray(const SkDCurve& c, const SkDLine& ray,  SkIntersections* i) {
    i->intersectRay(c.fLine, ray);
}

static void dquad_intersect_ray(const SkDCurve& c, const SkDLine& ray, SkIntersections* i) {
    i->intersectRay(c.fQuad, ray);
}

static void dconic_intersect_ray(const SkDCurve& c, const SkDLine& ray, SkIntersections* i) {
    i->intersectRay(c.fConic, ray);
}

static void dcubic_intersect_ray(const SkDCurve& c, const SkDLine& ray, SkIntersections* i) {
    i->intersectRay(c.fCubic, ray);
}

static void (* const CurveDIntersectRay[])(const SkDCurve& , const SkDLine& , SkIntersections* ) = {
    nullptr,
    dline_intersect_ray,
    dquad_intersect_ray,
    dconic_intersect_ray,
    dcubic_intersect_ray
};

static int line_intercept_h(const SkPoint a[2], SkScalar , SkScalar y, double* roots) {
    if (a[0].fY == a[1].fY) {
        return false;
    }
    SkDLine line;
    roots[0] = SkIntersections::HorizontalIntercept(line.set(a), y);
    return between(0, roots[0], 1);
}

static int line_intercept_v(const SkPoint a[2], SkScalar , SkScalar x, double* roots) {
    if (a[0].fX == a[1].fX) {
        return false;
    }
    SkDLine line;
    roots[0] = SkIntersections::VerticalIntercept(line.set(a), x);
    return between(0, roots[0], 1);
}

static int quad_intercept_h(const SkPoint a[2], SkScalar , SkScalar y, double* roots) {
    SkDQuad quad;
    return SkIntersections::HorizontalIntercept(quad.set(a), y, roots);
}

static int quad_intercept_v(const SkPoint a[2], SkScalar , SkScalar x, double* roots) {
    SkDQuad quad;
    return SkIntersections::VerticalIntercept(quad.set(a), x, roots);
}

static int conic_intercept_h(const SkPoint a[2], SkScalar w, SkScalar y, double* roots) {
    SkDConic conic;
    return SkIntersections::HorizontalIntercept(conic.set(a, w), y, roots);
}

static int conic_intercept_v(const SkPoint a[2], SkScalar w, SkScalar x, double* roots) {
    SkDConic conic;
    return SkIntersections::VerticalIntercept(conic.set(a, w), x, roots);
}

static int cubic_intercept_h(const SkPoint a[3], SkScalar , SkScalar y, double* roots) {
    SkDCubic cubic;
    return cubic.set(a).horizontalIntersect(y, roots);
}

static int cubic_intercept_v(const SkPoint a[3], SkScalar , SkScalar x, double* roots) {
    SkDCubic cubic;
    return cubic.set(a).verticalIntersect(x, roots);
}

static int (* const CurveIntercept[])(const SkPoint[] , SkScalar , SkScalar , double* ) = {
    nullptr,
    nullptr,
    line_intercept_h,
    line_intercept_v,
    quad_intercept_h,
    quad_intercept_v,
    conic_intercept_h,
    conic_intercept_v,
    cubic_intercept_h,
    cubic_intercept_v,
};

#endif
