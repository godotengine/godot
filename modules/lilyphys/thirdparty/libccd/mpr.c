/***
 * libccd
 * ---------------------------------
 * Copyright (c)2010,2011 Daniel Fiser <danfis@danfis.cz>
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

#include <stdlib.h>
#include "ccd/ccd.h"
#include "simplex.h"
#include "dbg.h"

/** Finds origin (center) of Minkowski difference (actually it can be any
 *  interior point of Minkowski difference. */
_ccd_inline void findOrigin(const void *obj1, const void *obj2, const ccd_t *ccd,
                            ccd_support_t *center);

/** Discovers initial portal - that is tetrahedron that intersects with
 *  origin ray (ray from center of Minkowski diff to (0,0,0).
 *
 *  Returns -1 if already recognized that origin is outside Minkowski
 *  portal.
 *  Returns 1 if origin lies on v1 of simplex (only v0 and v1 are present
 *  in simplex).
 *  Returns 2 if origin lies on v0-v1 segment.
 *  Returns 0 if portal was built.
 */
static int discoverPortal(const void *obj1, const void *obj2,
                          const ccd_t *ccd, ccd_simplex_t *portal);


/** Expands portal towards origin and determine if objects intersect.
 *  Already established portal must be given as argument.
 *  If intersection is found 0 is returned, -1 otherwise */
static int refinePortal(const void *obj1, const void *obj2,
                        const ccd_t *ccd, ccd_simplex_t *portal);

/** Finds penetration info by expanding provided portal. */
static void findPenetr(const void *obj1, const void *obj2, const ccd_t *ccd,
                       ccd_simplex_t *portal,
                       ccd_real_t *depth, ccd_vec3_t *dir, ccd_vec3_t *pos);

/** Finds penetration info if origin lies on portal's v1 */
static void findPenetrTouch(const void *obj1, const void *obj2, const ccd_t *ccd,
                            ccd_simplex_t *portal,
                            ccd_real_t *depth, ccd_vec3_t *dir, ccd_vec3_t *pos);

/** Find penetration info if origin lies on portal's segment v0-v1 */
static void findPenetrSegment(const void *obj1, const void *obj2, const ccd_t *ccd,
                              ccd_simplex_t *portal,
                              ccd_real_t *depth, ccd_vec3_t *dir, ccd_vec3_t *pos);

/** Finds position vector from fully established portal */
static void findPos(const void *obj1, const void *obj2, const ccd_t *ccd,
                    const ccd_simplex_t *portal, ccd_vec3_t *pos);

/** Extends portal with new support point.
 *  Portal must have face v1-v2-v3 arranged to face outside portal. */
_ccd_inline void expandPortal(ccd_simplex_t *portal,
                              const ccd_support_t *v4);

/** Fill dir with direction outside portal. Portal's v1-v2-v3 face must be
 *  arranged in correct order! */
_ccd_inline void portalDir(const ccd_simplex_t *portal, ccd_vec3_t *dir);

/** Returns true if portal encapsules origin (0,0,0), dir is direction of
 *  v1-v2-v3 face. */
_ccd_inline int portalEncapsulesOrigin(const ccd_simplex_t *portal,
                                       const ccd_vec3_t *dir);

/** Returns true if portal with new point v4 would reach specified
 *  tolerance (i.e. returns true if portal can _not_ significantly expand
 *  within Minkowski difference).
 *
 *  v4 is candidate for new point in portal, dir is direction in which v4
 *  was obtained. */
_ccd_inline int portalReachTolerance(const ccd_simplex_t *portal,
                                     const ccd_support_t *v4,
                                     const ccd_vec3_t *dir,
                                     const ccd_t *ccd);

/** Returns true if portal expanded by new point v4 could possibly contain
 *  origin, dir is direction in which v4 was obtained. */
_ccd_inline int portalCanEncapsuleOrigin(const ccd_simplex_t *portal,   
                                         const ccd_support_t *v4,
                                         const ccd_vec3_t *dir);


int ccdMPRIntersect(const void *obj1, const void *obj2, const ccd_t *ccd)
{
    ccd_simplex_t portal;
    int res;

    // Phase 1: Portal discovery - find portal that intersects with origin
    // ray (ray from center of Minkowski diff to origin of coordinates)
    res = discoverPortal(obj1, obj2, ccd, &portal);
    if (res < 0)
        return 0;
    if (res > 0)
        return 1;

    // Phase 2: Portal refinement
    res = refinePortal(obj1, obj2, ccd, &portal);
    return (res == 0 ? 1 : 0);
}

int ccdMPRPenetration(const void *obj1, const void *obj2, const ccd_t *ccd,
                      ccd_real_t *depth, ccd_vec3_t *dir, ccd_vec3_t *pos)
{
    ccd_simplex_t portal;
    int res;

    // Phase 1: Portal discovery
    res = discoverPortal(obj1, obj2, ccd, &portal);
    if (res < 0){
        // Origin isn't inside portal - no collision.
        return -1;

    }else if (res == 1){
        // Touching contact on portal's v1.
        findPenetrTouch(obj1, obj2, ccd, &portal, depth, dir, pos);

    }else if (res == 2){
        // Origin lies on v0-v1 segment.
        findPenetrSegment(obj1, obj2, ccd, &portal, depth, dir, pos);

    }else if (res == 0){
        // Phase 2: Portal refinement
        res = refinePortal(obj1, obj2, ccd, &portal);
        if (res < 0)
            return -1;

        // Phase 3. Penetration info
        findPenetr(obj1, obj2, ccd, &portal, depth, dir, pos);
    }

    return 0;
}



_ccd_inline void findOrigin(const void *obj1, const void *obj2, const ccd_t *ccd,
                            ccd_support_t *center)
{
    ccd->center1(obj1, &center->v1);
    ccd->center2(obj2, &center->v2);
    ccdVec3Sub2(&center->v, &center->v1, &center->v2);
}

static int discoverPortal(const void *obj1, const void *obj2,
                          const ccd_t *ccd, ccd_simplex_t *portal)
{
    ccd_vec3_t dir, va, vb;
    ccd_real_t dot;
    int cont;

    // vertex 0 is center of portal
    findOrigin(obj1, obj2, ccd, ccdSimplexPointW(portal, 0));
    ccdSimplexSetSize(portal, 1);

    if (ccdVec3Eq(&ccdSimplexPoint(portal, 0)->v, ccd_vec3_origin)){
        // Portal's center lies on origin (0,0,0) => we know that objects
        // intersect but we would need to know penetration info.
        // So move center little bit...
        ccdVec3Set(&va, CCD_EPS * CCD_REAL(10.), CCD_ZERO, CCD_ZERO);
        ccdVec3Add(&ccdSimplexPointW(portal, 0)->v, &va);
    }


    // vertex 1 = support in direction of origin
    ccdVec3Copy(&dir, &ccdSimplexPoint(portal, 0)->v);
    ccdVec3Scale(&dir, CCD_REAL(-1.));
    ccdVec3Normalize(&dir);
    __ccdSupport(obj1, obj2, &dir, ccd, ccdSimplexPointW(portal, 1));
    ccdSimplexSetSize(portal, 2);

    // test if origin isn't outside of v1
    dot = ccdVec3Dot(&ccdSimplexPoint(portal, 1)->v, &dir);
    if (ccdIsZero(dot) || dot < CCD_ZERO)
        return -1;


    // vertex 2
    ccdVec3Cross(&dir, &ccdSimplexPoint(portal, 0)->v,
                       &ccdSimplexPoint(portal, 1)->v);
    if (ccdIsZero(ccdVec3Len2(&dir))){
        if (ccdVec3Eq(&ccdSimplexPoint(portal, 1)->v, ccd_vec3_origin)){
            // origin lies on v1
            return 1;
        }else{
            // origin lies on v0-v1 segment
            return 2;
        }
    }

    ccdVec3Normalize(&dir);
    __ccdSupport(obj1, obj2, &dir, ccd, ccdSimplexPointW(portal, 2));
    dot = ccdVec3Dot(&ccdSimplexPoint(portal, 2)->v, &dir);
    if (ccdIsZero(dot) || dot < CCD_ZERO)
        return -1;

    ccdSimplexSetSize(portal, 3);

    // vertex 3 direction
    ccdVec3Sub2(&va, &ccdSimplexPoint(portal, 1)->v,
                     &ccdSimplexPoint(portal, 0)->v);
    ccdVec3Sub2(&vb, &ccdSimplexPoint(portal, 2)->v,
                     &ccdSimplexPoint(portal, 0)->v);
    ccdVec3Cross(&dir, &va, &vb);
    ccdVec3Normalize(&dir);

    // it is better to form portal faces to be oriented "outside" origin
    dot = ccdVec3Dot(&dir, &ccdSimplexPoint(portal, 0)->v);
    if (dot > CCD_ZERO){
        ccdSimplexSwap(portal, 1, 2);
        ccdVec3Scale(&dir, CCD_REAL(-1.));
    }

    while (ccdSimplexSize(portal) < 4){
        __ccdSupport(obj1, obj2, &dir, ccd, ccdSimplexPointW(portal, 3));
        dot = ccdVec3Dot(&ccdSimplexPoint(portal, 3)->v, &dir);
        if (ccdIsZero(dot) || dot < CCD_ZERO)
            return -1;

        cont = 0;

        // test if origin is outside (v1, v0, v3) - set v2 as v3 and
        // continue
        ccdVec3Cross(&va, &ccdSimplexPoint(portal, 1)->v,
                          &ccdSimplexPoint(portal, 3)->v);
        dot = ccdVec3Dot(&va, &ccdSimplexPoint(portal, 0)->v);
        if (dot < CCD_ZERO && !ccdIsZero(dot)){
            ccdSimplexSet(portal, 2, ccdSimplexPoint(portal, 3));
            cont = 1;
        }

        if (!cont){
            // test if origin is outside (v3, v0, v2) - set v1 as v3 and
            // continue
            ccdVec3Cross(&va, &ccdSimplexPoint(portal, 3)->v,
                              &ccdSimplexPoint(portal, 2)->v);
            dot = ccdVec3Dot(&va, &ccdSimplexPoint(portal, 0)->v);
            if (dot < CCD_ZERO && !ccdIsZero(dot)){
                ccdSimplexSet(portal, 1, ccdSimplexPoint(portal, 3));
                cont = 1;
            }
        }

        if (cont){
            ccdVec3Sub2(&va, &ccdSimplexPoint(portal, 1)->v,
                             &ccdSimplexPoint(portal, 0)->v);
            ccdVec3Sub2(&vb, &ccdSimplexPoint(portal, 2)->v,
                             &ccdSimplexPoint(portal, 0)->v);
            ccdVec3Cross(&dir, &va, &vb);
            ccdVec3Normalize(&dir);
        }else{
            ccdSimplexSetSize(portal, 4);
        }
    }

    return 0;
}

static int refinePortal(const void *obj1, const void *obj2,
                        const ccd_t *ccd, ccd_simplex_t *portal)
{
    ccd_vec3_t dir;
    ccd_support_t v4;

    while (1){
        // compute direction outside the portal (from v0 throught v1,v2,v3
        // face)
        portalDir(portal, &dir);

        // test if origin is inside the portal
        if (portalEncapsulesOrigin(portal, &dir))
            return 0;

        // get next support point
        __ccdSupport(obj1, obj2, &dir, ccd, &v4);

        // test if v4 can expand portal to contain origin and if portal
        // expanding doesn't reach given tolerance
        if (!portalCanEncapsuleOrigin(portal, &v4, &dir)
                || portalReachTolerance(portal, &v4, &dir, ccd)){
            return -1;
        }

        // v1-v2-v3 triangle must be rearranged to face outside Minkowski
        // difference (direction from v0).
        expandPortal(portal, &v4);
    }

    return -1;
}


static void findPenetr(const void *obj1, const void *obj2, const ccd_t *ccd,
                       ccd_simplex_t *portal,
                       ccd_real_t *depth, ccd_vec3_t *pdir, ccd_vec3_t *pos)
{
    ccd_vec3_t dir;
    ccd_support_t v4;
    unsigned long iterations;

    iterations = 0UL;
    while (1){
        // compute portal direction and obtain next support point
        portalDir(portal, &dir);
        __ccdSupport(obj1, obj2, &dir, ccd, &v4);

        // reached tolerance -> find penetration info
        if (portalReachTolerance(portal, &v4, &dir, ccd)
                || iterations > ccd->max_iterations){
            *depth = ccdVec3PointTriDist2(ccd_vec3_origin,
                                          &ccdSimplexPoint(portal, 1)->v,
                                          &ccdSimplexPoint(portal, 2)->v,
                                          &ccdSimplexPoint(portal, 3)->v,
                                          pdir);
            *depth = CCD_SQRT(*depth);
            if (ccdIsZero(*depth)){
                // If depth is zero, then we have a touching contact.
                // So following findPenetrTouch(), we assign zero to
                // the direction vector (it can actually be anything
                // according to the decription of ccdMPRPenetration
                // function).
                ccdVec3Copy(pdir, ccd_vec3_origin);
            }else{
                ccdVec3Normalize(pdir);
            }

            // barycentric coordinates:
            findPos(obj1, obj2, ccd, portal, pos);

            return;
        }

        expandPortal(portal, &v4);

        iterations++;
    }
}

static void findPenetrTouch(const void *obj1, const void *obj2, const ccd_t *ccd,
                            ccd_simplex_t *portal,
                            ccd_real_t *depth, ccd_vec3_t *dir, ccd_vec3_t *pos)
{
    // Touching contact on portal's v1 - so depth is zero and direction
    // is unimportant and pos can be guessed
    *depth = CCD_REAL(0.);
    ccdVec3Copy(dir, ccd_vec3_origin);

    ccdVec3Copy(pos, &ccdSimplexPoint(portal, 1)->v1);
    ccdVec3Add(pos, &ccdSimplexPoint(portal, 1)->v2);
    ccdVec3Scale(pos, 0.5);
}

static void findPenetrSegment(const void *obj1, const void *obj2, const ccd_t *ccd,
                              ccd_simplex_t *portal,
                              ccd_real_t *depth, ccd_vec3_t *dir, ccd_vec3_t *pos)
{
    /*
    ccd_vec3_t vec;
    ccd_real_t k;
    */

    // Origin lies on v0-v1 segment.
    // Depth is distance to v1, direction also and position must be
    // computed

    ccdVec3Copy(pos, &ccdSimplexPoint(portal, 1)->v1);
    ccdVec3Add(pos, &ccdSimplexPoint(portal, 1)->v2);
    ccdVec3Scale(pos, CCD_REAL(0.5));

    /*
    ccdVec3Sub2(&vec, &ccdSimplexPoint(portal, 1)->v,
                      &ccdSimplexPoint(portal, 0)->v);
    k  = CCD_SQRT(ccdVec3Len2(&ccdSimplexPoint(portal, 0)->v));
    k /= CCD_SQRT(ccdVec3Len2(&vec));
    ccdVec3Scale(&vec, -k);
    ccdVec3Add(pos, &vec);
    */

    ccdVec3Copy(dir, &ccdSimplexPoint(portal, 1)->v);
    *depth = CCD_SQRT(ccdVec3Len2(dir));
    ccdVec3Normalize(dir);
}


static void findPos(const void *obj1, const void *obj2, const ccd_t *ccd,
                    const ccd_simplex_t *portal, ccd_vec3_t *pos)
{
    ccd_vec3_t dir;
    size_t i;
    ccd_real_t b[4], sum, inv;
    ccd_vec3_t vec, p1, p2;

    portalDir(portal, &dir);

    // use barycentric coordinates of tetrahedron to find origin
    ccdVec3Cross(&vec, &ccdSimplexPoint(portal, 1)->v,
                       &ccdSimplexPoint(portal, 2)->v);
    b[0] = ccdVec3Dot(&vec, &ccdSimplexPoint(portal, 3)->v);

    ccdVec3Cross(&vec, &ccdSimplexPoint(portal, 3)->v,
                       &ccdSimplexPoint(portal, 2)->v);
    b[1] = ccdVec3Dot(&vec, &ccdSimplexPoint(portal, 0)->v);

    ccdVec3Cross(&vec, &ccdSimplexPoint(portal, 0)->v,
                       &ccdSimplexPoint(portal, 1)->v);
    b[2] = ccdVec3Dot(&vec, &ccdSimplexPoint(portal, 3)->v);

    ccdVec3Cross(&vec, &ccdSimplexPoint(portal, 2)->v,
                       &ccdSimplexPoint(portal, 1)->v);
    b[3] = ccdVec3Dot(&vec, &ccdSimplexPoint(portal, 0)->v);

	sum = b[0] + b[1] + b[2] + b[3];

    if (ccdIsZero(sum) || sum < CCD_ZERO){
		b[0] = CCD_REAL(0.);

        ccdVec3Cross(&vec, &ccdSimplexPoint(portal, 2)->v,
                           &ccdSimplexPoint(portal, 3)->v);
        b[1] = ccdVec3Dot(&vec, &dir);
        ccdVec3Cross(&vec, &ccdSimplexPoint(portal, 3)->v,
                           &ccdSimplexPoint(portal, 1)->v);
        b[2] = ccdVec3Dot(&vec, &dir);
        ccdVec3Cross(&vec, &ccdSimplexPoint(portal, 1)->v,
                           &ccdSimplexPoint(portal, 2)->v);
        b[3] = ccdVec3Dot(&vec, &dir);

		sum = b[1] + b[2] + b[3];
	}

	inv = CCD_REAL(1.) / sum;

    ccdVec3Copy(&p1, ccd_vec3_origin);
    ccdVec3Copy(&p2, ccd_vec3_origin);
    for (i = 0; i < 4; i++){
        ccdVec3Copy(&vec, &ccdSimplexPoint(portal, i)->v1);
        ccdVec3Scale(&vec, b[i]);
        ccdVec3Add(&p1, &vec);

        ccdVec3Copy(&vec, &ccdSimplexPoint(portal, i)->v2);
        ccdVec3Scale(&vec, b[i]);
        ccdVec3Add(&p2, &vec);
    }
    ccdVec3Scale(&p1, inv);
    ccdVec3Scale(&p2, inv);

    ccdVec3Copy(pos, &p1);
    ccdVec3Add(pos, &p2);
    ccdVec3Scale(pos, 0.5);
}

_ccd_inline void expandPortal(ccd_simplex_t *portal,
                              const ccd_support_t *v4)
{
    ccd_real_t dot;
    ccd_vec3_t v4v0;

    ccdVec3Cross(&v4v0, &v4->v, &ccdSimplexPoint(portal, 0)->v);
    dot = ccdVec3Dot(&ccdSimplexPoint(portal, 1)->v, &v4v0);
    if (dot > CCD_ZERO){
        dot = ccdVec3Dot(&ccdSimplexPoint(portal, 2)->v, &v4v0);
        if (dot > CCD_ZERO){
            ccdSimplexSet(portal, 1, v4);
        }else{
            ccdSimplexSet(portal, 3, v4);
        }
    }else{
        dot = ccdVec3Dot(&ccdSimplexPoint(portal, 3)->v, &v4v0);
        if (dot > CCD_ZERO){
            ccdSimplexSet(portal, 2, v4);
        }else{
            ccdSimplexSet(portal, 1, v4);
        }
    }
}

_ccd_inline void portalDir(const ccd_simplex_t *portal, ccd_vec3_t *dir)
{
    ccd_vec3_t v2v1, v3v1;

    ccdVec3Sub2(&v2v1, &ccdSimplexPoint(portal, 2)->v,
                       &ccdSimplexPoint(portal, 1)->v);
    ccdVec3Sub2(&v3v1, &ccdSimplexPoint(portal, 3)->v,
                       &ccdSimplexPoint(portal, 1)->v);
    ccdVec3Cross(dir, &v2v1, &v3v1);
    ccdVec3Normalize(dir);
}

_ccd_inline int portalEncapsulesOrigin(const ccd_simplex_t *portal,
                                       const ccd_vec3_t *dir)
{
    ccd_real_t dot;
    dot = ccdVec3Dot(dir, &ccdSimplexPoint(portal, 1)->v);
    return ccdIsZero(dot) || dot > CCD_ZERO;
}

_ccd_inline int portalReachTolerance(const ccd_simplex_t *portal,
                                     const ccd_support_t *v4,
                                     const ccd_vec3_t *dir,
                                     const ccd_t *ccd)
{
    ccd_real_t dv1, dv2, dv3, dv4;
    ccd_real_t dot1, dot2, dot3;

    // find the smallest dot product of dir and {v1-v4, v2-v4, v3-v4}

    dv1 = ccdVec3Dot(&ccdSimplexPoint(portal, 1)->v, dir);
    dv2 = ccdVec3Dot(&ccdSimplexPoint(portal, 2)->v, dir);
    dv3 = ccdVec3Dot(&ccdSimplexPoint(portal, 3)->v, dir);
    dv4 = ccdVec3Dot(&v4->v, dir);

    dot1 = dv4 - dv1;
    dot2 = dv4 - dv2;
    dot3 = dv4 - dv3;

    dot1 = CCD_FMIN(dot1, dot2);
    dot1 = CCD_FMIN(dot1, dot3);

    return ccdEq(dot1, ccd->mpr_tolerance) || dot1 < ccd->mpr_tolerance;
}

_ccd_inline int portalCanEncapsuleOrigin(const ccd_simplex_t *portal,   
                                         const ccd_support_t *v4,
                                         const ccd_vec3_t *dir)
{
    ccd_real_t dot;
    dot = ccdVec3Dot(&v4->v, dir);
    return ccdIsZero(dot) || dot > CCD_ZERO;
}
