/***
 * libccd
 * ---------------------------------
 * Copyright (c)2010 Daniel Fiser <danfis@danfis.cz>
 *
 *
 *  This file is part of libccd.
 *
 *  Distributed under the OSI-approved BSD License (the "License");
 *  see accompanying file BDS-LICENSE for details or see
 *  <http://www.opensource.org/licenses/bsd-license.php>.
 *
 *  This software is distributed WITHOUT ANY WARRANTY; without even the
 *  implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 *  See the License for more information.
 */

#include <stdio.h>
#include "ccd/vec3.h"
#include "dbg.h"

static CCD_VEC3(__ccd_vec3_origin, CCD_ZERO, CCD_ZERO, CCD_ZERO);
ccd_vec3_t *ccd_vec3_origin = &__ccd_vec3_origin;

static ccd_vec3_t points_on_sphere[] = {
	CCD_VEC3_STATIC(CCD_REAL( 0.000000), CCD_REAL(-0.000000), CCD_REAL(-1.000000)),
	CCD_VEC3_STATIC(CCD_REAL( 0.723608), CCD_REAL(-0.525725), CCD_REAL(-0.447219)),
	CCD_VEC3_STATIC(CCD_REAL(-0.276388), CCD_REAL(-0.850649), CCD_REAL(-0.447219)),
	CCD_VEC3_STATIC(CCD_REAL(-0.894426), CCD_REAL(-0.000000), CCD_REAL(-0.447216)),
	CCD_VEC3_STATIC(CCD_REAL(-0.276388), CCD_REAL( 0.850649), CCD_REAL(-0.447220)),
	CCD_VEC3_STATIC(CCD_REAL( 0.723608), CCD_REAL( 0.525725), CCD_REAL(-0.447219)),
	CCD_VEC3_STATIC(CCD_REAL( 0.276388), CCD_REAL(-0.850649), CCD_REAL( 0.447220)),
	CCD_VEC3_STATIC(CCD_REAL(-0.723608), CCD_REAL(-0.525725), CCD_REAL( 0.447219)),
	CCD_VEC3_STATIC(CCD_REAL(-0.723608), CCD_REAL( 0.525725), CCD_REAL( 0.447219)),
	CCD_VEC3_STATIC(CCD_REAL( 0.276388), CCD_REAL( 0.850649), CCD_REAL( 0.447219)),
	CCD_VEC3_STATIC(CCD_REAL( 0.894426), CCD_REAL( 0.000000), CCD_REAL( 0.447216)),
	CCD_VEC3_STATIC(CCD_REAL(-0.000000), CCD_REAL( 0.000000), CCD_REAL( 1.000000)), 
	CCD_VEC3_STATIC(CCD_REAL( 0.425323), CCD_REAL(-0.309011), CCD_REAL(-0.850654)),
	CCD_VEC3_STATIC(CCD_REAL(-0.162456), CCD_REAL(-0.499995), CCD_REAL(-0.850654)),
	CCD_VEC3_STATIC(CCD_REAL( 0.262869), CCD_REAL(-0.809012), CCD_REAL(-0.525738)),
	CCD_VEC3_STATIC(CCD_REAL( 0.425323), CCD_REAL( 0.309011), CCD_REAL(-0.850654)),
	CCD_VEC3_STATIC(CCD_REAL( 0.850648), CCD_REAL(-0.000000), CCD_REAL(-0.525736)),
	CCD_VEC3_STATIC(CCD_REAL(-0.525730), CCD_REAL(-0.000000), CCD_REAL(-0.850652)),
	CCD_VEC3_STATIC(CCD_REAL(-0.688190), CCD_REAL(-0.499997), CCD_REAL(-0.525736)),
	CCD_VEC3_STATIC(CCD_REAL(-0.162456), CCD_REAL( 0.499995), CCD_REAL(-0.850654)),
	CCD_VEC3_STATIC(CCD_REAL(-0.688190), CCD_REAL( 0.499997), CCD_REAL(-0.525736)),
	CCD_VEC3_STATIC(CCD_REAL( 0.262869), CCD_REAL( 0.809012), CCD_REAL(-0.525738)),
	CCD_VEC3_STATIC(CCD_REAL( 0.951058), CCD_REAL( 0.309013), CCD_REAL( 0.000000)),
	CCD_VEC3_STATIC(CCD_REAL( 0.951058), CCD_REAL(-0.309013), CCD_REAL( 0.000000)),
	CCD_VEC3_STATIC(CCD_REAL( 0.587786), CCD_REAL(-0.809017), CCD_REAL( 0.000000)),
	CCD_VEC3_STATIC(CCD_REAL( 0.000000), CCD_REAL(-1.000000), CCD_REAL( 0.000000)),
	CCD_VEC3_STATIC(CCD_REAL(-0.587786), CCD_REAL(-0.809017), CCD_REAL( 0.000000)),
	CCD_VEC3_STATIC(CCD_REAL(-0.951058), CCD_REAL(-0.309013), CCD_REAL(-0.000000)),
	CCD_VEC3_STATIC(CCD_REAL(-0.951058), CCD_REAL( 0.309013), CCD_REAL(-0.000000)),
	CCD_VEC3_STATIC(CCD_REAL(-0.587786), CCD_REAL( 0.809017), CCD_REAL(-0.000000)),
	CCD_VEC3_STATIC(CCD_REAL(-0.000000), CCD_REAL( 1.000000), CCD_REAL(-0.000000)),
	CCD_VEC3_STATIC(CCD_REAL( 0.587786), CCD_REAL( 0.809017), CCD_REAL(-0.000000)),
	CCD_VEC3_STATIC(CCD_REAL( 0.688190), CCD_REAL(-0.499997), CCD_REAL( 0.525736)),
	CCD_VEC3_STATIC(CCD_REAL(-0.262869), CCD_REAL(-0.809012), CCD_REAL( 0.525738)),
	CCD_VEC3_STATIC(CCD_REAL(-0.850648), CCD_REAL( 0.000000), CCD_REAL( 0.525736)),
	CCD_VEC3_STATIC(CCD_REAL(-0.262869), CCD_REAL( 0.809012), CCD_REAL( 0.525738)),
	CCD_VEC3_STATIC(CCD_REAL( 0.688190), CCD_REAL( 0.499997), CCD_REAL( 0.525736)),
	CCD_VEC3_STATIC(CCD_REAL( 0.525730), CCD_REAL( 0.000000), CCD_REAL( 0.850652)),
	CCD_VEC3_STATIC(CCD_REAL( 0.162456), CCD_REAL(-0.499995), CCD_REAL( 0.850654)),
	CCD_VEC3_STATIC(CCD_REAL(-0.425323), CCD_REAL(-0.309011), CCD_REAL( 0.850654)),
	CCD_VEC3_STATIC(CCD_REAL(-0.425323), CCD_REAL( 0.309011), CCD_REAL( 0.850654)),
	CCD_VEC3_STATIC(CCD_REAL( 0.162456), CCD_REAL( 0.499995), CCD_REAL( 0.850654))
};
ccd_vec3_t *ccd_points_on_sphere = points_on_sphere;
size_t ccd_points_on_sphere_len = sizeof(points_on_sphere) / sizeof(ccd_vec3_t);


_ccd_inline ccd_real_t __ccdVec3PointSegmentDist2(const ccd_vec3_t *P,
                                                  const ccd_vec3_t *x0,
                                                  const ccd_vec3_t *b,
                                                  ccd_vec3_t *witness)
{
    // The computation comes from solving equation of segment:
    //      S(t) = x0 + t.d
    //          where - x0 is initial point of segment
    //                - d is direction of segment from x0 (|d| > 0)
    //                - t belongs to <0, 1> interval
    // 
    // Than, distance from a segment to some point P can be expressed:
    //      D(t) = |x0 + t.d - P|^2
    //          which is distance from any point on segment. Minimization
    //          of this function brings distance from P to segment.
    // Minimization of D(t) leads to simple quadratic equation that's
    // solving is straightforward.
    //
    // Bonus of this method is witness point for free.

    ccd_real_t dist, t;
    ccd_vec3_t d, a;

    // direction of segment
    ccdVec3Sub2(&d, b, x0);

    // precompute vector from P to x0
    ccdVec3Sub2(&a, x0, P);

    t  = -CCD_REAL(1.) * ccdVec3Dot(&a, &d);
    t /= ccdVec3Len2(&d);

    if (t < CCD_ZERO || ccdIsZero(t)){
        dist = ccdVec3Dist2(x0, P);
        if (witness)
            ccdVec3Copy(witness, x0);
    }else if (t > CCD_ONE || ccdEq(t, CCD_ONE)){
        dist = ccdVec3Dist2(b, P);
        if (witness)
            ccdVec3Copy(witness, b);
    }else{
        if (witness){
            ccdVec3Copy(witness, &d);
            ccdVec3Scale(witness, t);
            ccdVec3Add(witness, x0);
            dist = ccdVec3Dist2(witness, P);
        }else{
            // recycling variables
            ccdVec3Scale(&d, t);
            ccdVec3Add(&d, &a);
            dist = ccdVec3Len2(&d);
        }
    }

    return dist;
}

ccd_real_t ccdVec3PointSegmentDist2(const ccd_vec3_t *P,
                                    const ccd_vec3_t *x0, const ccd_vec3_t *b,
                                    ccd_vec3_t *witness)
{
    return __ccdVec3PointSegmentDist2(P, x0, b, witness);
}

ccd_real_t ccdVec3PointTriDist2(const ccd_vec3_t *P,
                                const ccd_vec3_t *x0, const ccd_vec3_t *B,
                                const ccd_vec3_t *C,
                                ccd_vec3_t *witness)
{
    // Computation comes from analytic expression for triangle (x0, B, C)
    //      T(s, t) = x0 + s.d1 + t.d2, where d1 = B - x0 and d2 = C - x0 and
    // Then equation for distance is:
    //      D(s, t) = | T(s, t) - P |^2
    // This leads to minimization of quadratic function of two variables.
    // The solution from is taken only if s is between 0 and 1, t is
    // between 0 and 1 and t + s < 1, otherwise distance from segment is
    // computed.

    ccd_vec3_t d1, d2, a;
    ccd_real_t u, v, w, p, q, r, d;
    ccd_real_t s, t, dist, dist2;
    ccd_vec3_t witness2;

    ccdVec3Sub2(&d1, B, x0);
    ccdVec3Sub2(&d2, C, x0);
    ccdVec3Sub2(&a, x0, P);

    u = ccdVec3Dot(&a, &a);
    v = ccdVec3Dot(&d1, &d1);
    w = ccdVec3Dot(&d2, &d2);
    p = ccdVec3Dot(&a, &d1);
    q = ccdVec3Dot(&a, &d2);
    r = ccdVec3Dot(&d1, &d2);

    d = w * v - r * r;
    if (ccdIsZero(d)){
        // To avoid division by zero for zero (or near zero) area triangles
        s = t = -1.;
    }else{
        s = (q * r - w * p) / d;
        t = (-s * r - q) / w;
    }

    if ((ccdIsZero(s) || s > CCD_ZERO)
            && (ccdEq(s, CCD_ONE) || s < CCD_ONE)
            && (ccdIsZero(t) || t > CCD_ZERO)
            && (ccdEq(t, CCD_ONE) || t < CCD_ONE)
            && (ccdEq(t + s, CCD_ONE) || t + s < CCD_ONE)){

        if (witness){
            ccdVec3Scale(&d1, s);
            ccdVec3Scale(&d2, t);
            ccdVec3Copy(witness, x0);
            ccdVec3Add(witness, &d1);
            ccdVec3Add(witness, &d2);

            dist = ccdVec3Dist2(witness, P);
        }else{
            dist  = s * s * v;
            dist += t * t * w;
            dist += CCD_REAL(2.) * s * t * r;
            dist += CCD_REAL(2.) * s * p;
            dist += CCD_REAL(2.) * t * q;
            dist += u;
        }
    }else{
        dist = __ccdVec3PointSegmentDist2(P, x0, B, witness);

        dist2 = __ccdVec3PointSegmentDist2(P, x0, C, &witness2);
        if (dist2 < dist){
            dist = dist2;
            if (witness)
                ccdVec3Copy(witness, &witness2);
        }

        dist2 = __ccdVec3PointSegmentDist2(P, B, C, &witness2);
        if (dist2 < dist){
            dist = dist2;
            if (witness)
                ccdVec3Copy(witness, &witness2);
        }
    }

    return dist;
}
