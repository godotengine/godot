/***
 * libccd
 * ---------------------------------
 * Copyright (c)2012 Daniel Fiser <danfis@danfis.cz>
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
#include <float.h>
#include "ccd/ccd.h"
#include "ccd/vec3.h"
#include "simplex.h"
#include "polytope.h"
#include "alloc.h"
#include "dbg.h"


/** Performs GJK algorithm. Returns 0 if intersection was found and simplex
 *  is filled with resulting polytope. */
static int __ccdGJK(const void *obj1, const void *obj2,
                    const ccd_t *ccd, ccd_simplex_t *simplex);

/** Performs GJK+EPA algorithm. Returns 0 if intersection was found and
 *  pt is filled with resulting polytope and nearest with pointer to
 *  nearest element (vertex, edge, face) of polytope to origin. */
static int __ccdGJKEPA(const void *obj1, const void *obj2,
                       const ccd_t *ccd,
                       ccd_pt_t *pt, ccd_pt_el_t **nearest);


/** Returns true if simplex contains origin.
 *  This function also alteres simplex and dir according to further
 *  processing of GJK algorithm. */
static int doSimplex(ccd_simplex_t *simplex, ccd_vec3_t *dir);
static int doSimplex2(ccd_simplex_t *simplex, ccd_vec3_t *dir);
static int doSimplex3(ccd_simplex_t *simplex, ccd_vec3_t *dir);
static int doSimplex4(ccd_simplex_t *simplex, ccd_vec3_t *dir);

/** d = a x b x c */
_ccd_inline void tripleCross(const ccd_vec3_t *a, const ccd_vec3_t *b,
                             const ccd_vec3_t *c, ccd_vec3_t *d);


/** Transforms simplex to polytope. It is assumed that simplex has 4
 *  vertices. */
static int simplexToPolytope4(const void *obj1, const void *obj2,
                              const ccd_t *ccd,
                              ccd_simplex_t *simplex,
                              ccd_pt_t *pt, ccd_pt_el_t **nearest);

/** Transforms simplex to polytope, three vertices required */
static int simplexToPolytope3(const void *obj1, const void *obj2,
                              const ccd_t *ccd,
                              const ccd_simplex_t *simplex,
                              ccd_pt_t *pt, ccd_pt_el_t **nearest);

/** Transforms simplex to polytope, two vertices required */
static int simplexToPolytope2(const void *obj1, const void *obj2,
                              const ccd_t *ccd,
                              const ccd_simplex_t *simplex,
                              ccd_pt_t *pt, ccd_pt_el_t **nearest);

/** Expands polytope using new vertex v.
 *  Return 0 on success, -2 on memory allocation failure.*/
static int expandPolytope(ccd_pt_t *pt, ccd_pt_el_t *el,
                          const ccd_support_t *newv);

/** Finds next support point (at stores it in out argument).
 *  Returns 0 on success, -1 otherwise */
static int nextSupport(const void *obj1, const void *obj2, const ccd_t *ccd,
                       const ccd_pt_el_t *el,
                       ccd_support_t *out);



void ccdFirstDirDefault(const void *o1, const void *o2, ccd_vec3_t *dir)
{
    ccdVec3Set(dir, CCD_ONE, CCD_ZERO, CCD_ZERO);
}

int ccdGJKIntersect(const void *obj1, const void *obj2, const ccd_t *ccd)
{
    ccd_simplex_t simplex;
    return __ccdGJK(obj1, obj2, ccd, &simplex) == 0;
}

int ccdGJKSeparate(const void *obj1, const void *obj2, const ccd_t *ccd,
                   ccd_vec3_t *sep)
{
    ccd_pt_t polytope;
    ccd_pt_el_t *nearest;
    int ret;

    ccdPtInit(&polytope);

    ret = __ccdGJKEPA(obj1, obj2, ccd, &polytope, &nearest);

    // set separation vector
    if (nearest)
        ccdVec3Copy(sep, &nearest->witness);

    ccdPtDestroy(&polytope);

    return ret;
}


static int penEPAPosCmp(const void *a, const void *b)
{
    ccd_pt_vertex_t *v1, *v2;
    v1 = *(ccd_pt_vertex_t **)a;
    v2 = *(ccd_pt_vertex_t **)b;

    if (ccdEq(v1->dist, v2->dist)){
        return 0;
    }else if (v1->dist < v2->dist){
        return -1;
    }else{
        return 1;
    }
}

static int penEPAPos(const ccd_pt_t *pt, const ccd_pt_el_t *nearest,
                     ccd_vec3_t *pos)
{
    ccd_pt_vertex_t *v;
    ccd_pt_vertex_t **vs;
    size_t i, len;
    ccd_real_t scale;

    // compute median
    len = 0;
    ccdListForEachEntry(&pt->vertices, v, ccd_pt_vertex_t, list){
        len++;
    }

    vs = CCD_ALLOC_ARR(ccd_pt_vertex_t *, len);
    if (vs == NULL)
        return -1;

    i = 0;
    ccdListForEachEntry(&pt->vertices, v, ccd_pt_vertex_t, list){
        vs[i++] = v;
    }

    qsort(vs, len, sizeof(ccd_pt_vertex_t *), penEPAPosCmp);

    ccdVec3Set(pos, CCD_ZERO, CCD_ZERO, CCD_ZERO);
    scale = CCD_ZERO;
    if (len % 2 == 1)
        len++;

    for (i = 0; i < len / 2; i++){
        ccdVec3Add(pos, &vs[i]->v.v1);
        ccdVec3Add(pos, &vs[i]->v.v2);
        scale += CCD_REAL(2.);
    }
    ccdVec3Scale(pos, CCD_ONE / scale);

    free(vs);

    return 0;
}

int ccdGJKPenetration(const void *obj1, const void *obj2, const ccd_t *ccd,
                      ccd_real_t *depth, ccd_vec3_t *dir, ccd_vec3_t *pos)
{
    ccd_pt_t polytope;
    ccd_pt_el_t *nearest;
    int ret;

    ccdPtInit(&polytope);

    ret = __ccdGJKEPA(obj1, obj2, ccd, &polytope, &nearest);

    // set separation vector
    if (ret == 0 && nearest){
        // compute depth of penetration
        *depth = CCD_SQRT(nearest->dist);

        // store normalized direction vector
        ccdVec3Copy(dir, &nearest->witness);
        ccdVec3Normalize(dir);

        // compute position
        if (penEPAPos(&polytope, nearest, pos) != 0){
            ccdPtDestroy(&polytope);
            return -2;
        }
    }

    ccdPtDestroy(&polytope);

    return ret;
}


static int __ccdGJK(const void *obj1, const void *obj2,
                    const ccd_t *ccd, ccd_simplex_t *simplex)
{
    unsigned long iterations;
    ccd_vec3_t dir; // direction vector
    ccd_support_t last; // last support point
    int do_simplex_res;

    // initialize simplex struct
    ccdSimplexInit(simplex);

    // get first direction
    ccd->first_dir(obj1, obj2, &dir);
    // get first support point
    __ccdSupport(obj1, obj2, &dir, ccd, &last);
    // and add this point to simplex as last one
    ccdSimplexAdd(simplex, &last);

    // set up direction vector to as (O - last) which is exactly -last
    ccdVec3Copy(&dir, &last.v);
    ccdVec3Scale(&dir, -CCD_ONE);

    // start iterations
    for (iterations = 0UL; iterations < ccd->max_iterations; ++iterations) {
        // obtain support point
        __ccdSupport(obj1, obj2, &dir, ccd, &last);

        // check if farthest point in Minkowski difference in direction dir
        // isn't somewhere before origin (the test on negative dot product)
        // - because if it is, objects are not intersecting at all.
        if (ccdVec3Dot(&last.v, &dir) < CCD_ZERO){
            return -1; // intersection not found
        }

        // add last support vector to simplex
        ccdSimplexAdd(simplex, &last);

        // if doSimplex returns 1 if objects intersect, -1 if objects don't
        // intersect and 0 if algorithm should continue
        do_simplex_res = doSimplex(simplex, &dir);
        if (do_simplex_res == 1){
            return 0; // intersection found
        }else if (do_simplex_res == -1){
            return -1; // intersection not found
        }

        if (ccdIsZero(ccdVec3Len2(&dir))){
            return -1; // intersection not found
        }
    }

    // intersection wasn't found
    return -1;
}

static int __ccdGJKEPA(const void *obj1, const void *obj2,
                       const ccd_t *ccd,
                       ccd_pt_t *polytope, ccd_pt_el_t **nearest)
{
    ccd_simplex_t simplex;
    ccd_support_t supp; // support point
    int ret, size;

    *nearest = NULL;

    // run GJK and obtain terminal simplex
    ret = __ccdGJK(obj1, obj2, ccd, &simplex);
    if (ret != 0)
        return -1;

    // transform simplex to polytope - simplex won't be used anymore
    size = ccdSimplexSize(&simplex);
    if (size == 4){
        ret = simplexToPolytope4(obj1, obj2, ccd, &simplex, polytope, nearest);
    }else if (size == 3){
        ret = simplexToPolytope3(obj1, obj2, ccd, &simplex, polytope, nearest);
    }else{ // size == 2
        ret = simplexToPolytope2(obj1, obj2, ccd, &simplex, polytope, nearest);
    }

    if (ret == -1){
        // touching contact
        return 0;
    }else if (ret == -2){
        // failed memory allocation
        return -2;
    }


    while (1){
        // get triangle nearest to origin
        *nearest = ccdPtNearest(polytope);

        // get next support point
        if (nextSupport(obj1, obj2, ccd, *nearest, &supp) != 0)
            break;

        // expand nearest triangle using new point - supp
        if (expandPolytope(polytope, *nearest, &supp) != 0)
            return -2;
    }

    return 0;
}



static int doSimplex2(ccd_simplex_t *simplex, ccd_vec3_t *dir)
{
    const ccd_support_t *A, *B;
    ccd_vec3_t AB, AO, tmp;
    ccd_real_t dot;

    // get last added as A
    A = ccdSimplexLast(simplex);
    // get the other point
    B = ccdSimplexPoint(simplex, 0);
    // compute AB oriented segment
    ccdVec3Sub2(&AB, &B->v, &A->v);
    // compute AO vector
    ccdVec3Copy(&AO, &A->v);
    ccdVec3Scale(&AO, -CCD_ONE);

    // dot product AB . AO
    dot = ccdVec3Dot(&AB, &AO);

    // check if origin doesn't lie on AB segment
    ccdVec3Cross(&tmp, &AB, &AO);
    if (ccdIsZero(ccdVec3Len2(&tmp)) && dot > CCD_ZERO){
        return 1;
    }

    // check if origin is in area where AB segment is
    if (ccdIsZero(dot) || dot < CCD_ZERO){
        // origin is in outside are of A
        ccdSimplexSet(simplex, 0, A);
        ccdSimplexSetSize(simplex, 1);
        ccdVec3Copy(dir, &AO);
    }else{
        // origin is in area where AB segment is

        // keep simplex untouched and set direction to
        // AB x AO x AB
        tripleCross(&AB, &AO, &AB, dir);
    }

    return 0;
}

static int doSimplex3(ccd_simplex_t *simplex, ccd_vec3_t *dir)
{
    const ccd_support_t *A, *B, *C;
    ccd_vec3_t AO, AB, AC, ABC, tmp;
    ccd_real_t dot, dist;

    // get last added as A
    A = ccdSimplexLast(simplex);
    // get the other points
    B = ccdSimplexPoint(simplex, 1);
    C = ccdSimplexPoint(simplex, 0);

    // check touching contact
    dist = ccdVec3PointTriDist2(ccd_vec3_origin, &A->v, &B->v, &C->v, NULL);
    if (ccdIsZero(dist)){
        return 1;
    }

    // check if triangle is really triangle (has area > 0)
    // if not simplex can't be expanded and thus no itersection is found
    if (ccdVec3Eq(&A->v, &B->v) || ccdVec3Eq(&A->v, &C->v)){
        return -1;
    }

    // compute AO vector
    ccdVec3Copy(&AO, &A->v);
    ccdVec3Scale(&AO, -CCD_ONE);

    // compute AB and AC segments and ABC vector (perpendircular to triangle)
    ccdVec3Sub2(&AB, &B->v, &A->v);
    ccdVec3Sub2(&AC, &C->v, &A->v);
    ccdVec3Cross(&ABC, &AB, &AC);

    ccdVec3Cross(&tmp, &ABC, &AC);
    dot = ccdVec3Dot(&tmp, &AO);
    if (ccdIsZero(dot) || dot > CCD_ZERO){
        dot = ccdVec3Dot(&AC, &AO);
        if (ccdIsZero(dot) || dot > CCD_ZERO){
            // C is already in place
            ccdSimplexSet(simplex, 1, A);
            ccdSimplexSetSize(simplex, 2);
            tripleCross(&AC, &AO, &AC, dir);
        }else{
ccd_do_simplex3_45:
            dot = ccdVec3Dot(&AB, &AO);
            if (ccdIsZero(dot) || dot > CCD_ZERO){
                ccdSimplexSet(simplex, 0, B);
                ccdSimplexSet(simplex, 1, A);
                ccdSimplexSetSize(simplex, 2);
                tripleCross(&AB, &AO, &AB, dir);
            }else{
                ccdSimplexSet(simplex, 0, A);
                ccdSimplexSetSize(simplex, 1);
                ccdVec3Copy(dir, &AO);
            }
        }
    }else{
        ccdVec3Cross(&tmp, &AB, &ABC);
        dot = ccdVec3Dot(&tmp, &AO);
        if (ccdIsZero(dot) || dot > CCD_ZERO){
            goto ccd_do_simplex3_45;
        }else{
            dot = ccdVec3Dot(&ABC, &AO);
            if (ccdIsZero(dot) || dot > CCD_ZERO){
                ccdVec3Copy(dir, &ABC);
            }else{
                ccd_support_t Ctmp;
                ccdSupportCopy(&Ctmp, C);
                ccdSimplexSet(simplex, 0, B);
                ccdSimplexSet(simplex, 1, &Ctmp);

                ccdVec3Copy(dir, &ABC);
                ccdVec3Scale(dir, -CCD_ONE);
            }
        }
    }

    return 0;
}

static int doSimplex4(ccd_simplex_t *simplex, ccd_vec3_t *dir)
{
    const ccd_support_t *A, *B, *C, *D;
    ccd_vec3_t AO, AB, AC, AD, ABC, ACD, ADB;
    int B_on_ACD, C_on_ADB, D_on_ABC;
    int AB_O, AC_O, AD_O;
    ccd_real_t dist;

    // get last added as A
    A = ccdSimplexLast(simplex);
    // get the other points
    B = ccdSimplexPoint(simplex, 2);
    C = ccdSimplexPoint(simplex, 1);
    D = ccdSimplexPoint(simplex, 0);

    // check if tetrahedron is really tetrahedron (has volume > 0)
    // if it is not simplex can't be expanded and thus no intersection is
    // found
    dist = ccdVec3PointTriDist2(&A->v, &B->v, &C->v, &D->v, NULL);
    if (ccdIsZero(dist)){
        return -1;
    }

    // check if origin lies on some of tetrahedron's face - if so objects
    // intersect
    dist = ccdVec3PointTriDist2(ccd_vec3_origin, &A->v, &B->v, &C->v, NULL);
    if (ccdIsZero(dist))
        return 1;
    dist = ccdVec3PointTriDist2(ccd_vec3_origin, &A->v, &C->v, &D->v, NULL);
    if (ccdIsZero(dist))
        return 1;
    dist = ccdVec3PointTriDist2(ccd_vec3_origin, &A->v, &B->v, &D->v, NULL);
    if (ccdIsZero(dist))
        return 1;
    dist = ccdVec3PointTriDist2(ccd_vec3_origin, &B->v, &C->v, &D->v, NULL);
    if (ccdIsZero(dist))
        return 1;

    // compute AO, AB, AC, AD segments and ABC, ACD, ADB normal vectors
    ccdVec3Copy(&AO, &A->v);
    ccdVec3Scale(&AO, -CCD_ONE);
    ccdVec3Sub2(&AB, &B->v, &A->v);
    ccdVec3Sub2(&AC, &C->v, &A->v);
    ccdVec3Sub2(&AD, &D->v, &A->v);
    ccdVec3Cross(&ABC, &AB, &AC);
    ccdVec3Cross(&ACD, &AC, &AD);
    ccdVec3Cross(&ADB, &AD, &AB);

    // side (positive or negative) of B, C, D relative to planes ACD, ADB
    // and ABC respectively
    B_on_ACD = ccdSign(ccdVec3Dot(&ACD, &AB));
    C_on_ADB = ccdSign(ccdVec3Dot(&ADB, &AC));
    D_on_ABC = ccdSign(ccdVec3Dot(&ABC, &AD));

    // whether origin is on same side of ACD, ADB, ABC as B, C, D
    // respectively
    AB_O = ccdSign(ccdVec3Dot(&ACD, &AO)) == B_on_ACD;
    AC_O = ccdSign(ccdVec3Dot(&ADB, &AO)) == C_on_ADB;
    AD_O = ccdSign(ccdVec3Dot(&ABC, &AO)) == D_on_ABC;

    if (AB_O && AC_O && AD_O){
        // origin is in tetrahedron
        return 1;

    // rearrange simplex to triangle and call doSimplex3()
    }else if (!AB_O){
        // B is farthest from the origin among all of the tetrahedron's
        // points, so remove it from the list and go on with the triangle
        // case

        // D and C are in place
        ccdSimplexSet(simplex, 2, A);
        ccdSimplexSetSize(simplex, 3);
    }else if (!AC_O){
        // C is farthest
        ccdSimplexSet(simplex, 1, D);
        ccdSimplexSet(simplex, 0, B);
        ccdSimplexSet(simplex, 2, A);
        ccdSimplexSetSize(simplex, 3);
    }else{ // (!AD_O)
        ccdSimplexSet(simplex, 0, C);
        ccdSimplexSet(simplex, 1, B);
        ccdSimplexSet(simplex, 2, A);
        ccdSimplexSetSize(simplex, 3);
    }

    return doSimplex3(simplex, dir);
}

static int doSimplex(ccd_simplex_t *simplex, ccd_vec3_t *dir)
{
    if (ccdSimplexSize(simplex) == 2){
        // simplex contains segment only one segment
        return doSimplex2(simplex, dir);
    }else if (ccdSimplexSize(simplex) == 3){
        // simplex contains triangle
        return doSimplex3(simplex, dir);
    }else{ // ccdSimplexSize(simplex) == 4
        // tetrahedron - this is the only shape which can encapsule origin
        // so doSimplex4() also contains test on it
        return doSimplex4(simplex, dir);
    }
}


_ccd_inline void tripleCross(const ccd_vec3_t *a, const ccd_vec3_t *b,
                             const ccd_vec3_t *c, ccd_vec3_t *d)
{
    ccd_vec3_t e;
    ccdVec3Cross(&e, a, b);
    ccdVec3Cross(d, &e, c);
}



/** Transforms simplex to polytope. It is assumed that simplex has 4
 *  vertices! */
static int simplexToPolytope4(const void *obj1, const void *obj2,
                              const ccd_t *ccd,
                              ccd_simplex_t *simplex,
                              ccd_pt_t *pt, ccd_pt_el_t **nearest)
{
    const ccd_support_t *a, *b, *c, *d;
    int use_polytope3;
    ccd_real_t dist;
    ccd_pt_vertex_t *v[4];
    ccd_pt_edge_t *e[6];
    size_t i;

    a = ccdSimplexPoint(simplex, 0);
    b = ccdSimplexPoint(simplex, 1);
    c = ccdSimplexPoint(simplex, 2);
    d = ccdSimplexPoint(simplex, 3);

    // check if origin lies on some of tetrahedron's face - if so use
    // simplexToPolytope3()
    use_polytope3 = 0;
    dist = ccdVec3PointTriDist2(ccd_vec3_origin, &a->v, &b->v, &c->v, NULL);
    if (ccdIsZero(dist)){
        use_polytope3 = 1;
    }
    dist = ccdVec3PointTriDist2(ccd_vec3_origin, &a->v, &c->v, &d->v, NULL);
    if (ccdIsZero(dist)){
        use_polytope3 = 1;
        ccdSimplexSet(simplex, 1, c);
        ccdSimplexSet(simplex, 2, d);
    }
    dist = ccdVec3PointTriDist2(ccd_vec3_origin, &a->v, &b->v, &d->v, NULL);
    if (ccdIsZero(dist)){
        use_polytope3 = 1;
        ccdSimplexSet(simplex, 2, d);
    }
    dist = ccdVec3PointTriDist2(ccd_vec3_origin, &b->v, &c->v, &d->v, NULL);
    if (ccdIsZero(dist)){
        use_polytope3 = 1;
        ccdSimplexSet(simplex, 0, b);
        ccdSimplexSet(simplex, 1, c);
        ccdSimplexSet(simplex, 2, d);
    }

    if (use_polytope3){
        ccdSimplexSetSize(simplex, 3);
        return simplexToPolytope3(obj1, obj2, ccd, simplex, pt, nearest);
    }

    // no touching contact - simply create tetrahedron
    for (i = 0; i < 4; i++){
        v[i] = ccdPtAddVertex(pt, ccdSimplexPoint(simplex, i));
    }
    
    e[0] = ccdPtAddEdge(pt, v[0], v[1]);
    e[1] = ccdPtAddEdge(pt, v[1], v[2]);
    e[2] = ccdPtAddEdge(pt, v[2], v[0]);
    e[3] = ccdPtAddEdge(pt, v[3], v[0]);
    e[4] = ccdPtAddEdge(pt, v[3], v[1]);
    e[5] = ccdPtAddEdge(pt, v[3], v[2]);

    // ccdPtAdd*() functions return NULL either if the memory allocation
    // failed of if any of the input pointers are NULL, so the bad
    // allocation can be checked by the last calls of ccdPtAddFace()
    // because the rest of the bad allocations eventually "bubble up" here
    if (ccdPtAddFace(pt, e[0], e[1], e[2]) == NULL
            || ccdPtAddFace(pt, e[3], e[4], e[0]) == NULL
            || ccdPtAddFace(pt, e[4], e[5], e[1]) == NULL
            || ccdPtAddFace(pt, e[5], e[3], e[2]) == NULL){
        return -2;
    }

    return 0;
}

/** Transforms simplex to polytope, three vertices required */
static int simplexToPolytope3(const void *obj1, const void *obj2,
                              const ccd_t *ccd,
                              const ccd_simplex_t *simplex,
                              ccd_pt_t *pt, ccd_pt_el_t **nearest)
{
    const ccd_support_t *a, *b, *c;
    ccd_support_t d, d2;
    ccd_vec3_t ab, ac, dir;
    ccd_pt_vertex_t *v[5];
    ccd_pt_edge_t *e[9];
    ccd_real_t dist, dist2;

    *nearest = NULL;

    a = ccdSimplexPoint(simplex, 0);
    b = ccdSimplexPoint(simplex, 1);
    c = ccdSimplexPoint(simplex, 2);

    // If only one triangle left from previous GJK run origin lies on this
    // triangle. So it is necessary to expand triangle into two
    // tetrahedrons connected with base (which is exactly abc triangle).

    // get next support point in direction of normal of triangle
    ccdVec3Sub2(&ab, &b->v, &a->v);
    ccdVec3Sub2(&ac, &c->v, &a->v);
    ccdVec3Cross(&dir, &ab, &ac);
    __ccdSupport(obj1, obj2, &dir, ccd, &d);
    dist = ccdVec3PointTriDist2(&d.v, &a->v, &b->v, &c->v, NULL);

    // and second one take in opposite direction
    ccdVec3Scale(&dir, -CCD_ONE);
    __ccdSupport(obj1, obj2, &dir, ccd, &d2);
    dist2 = ccdVec3PointTriDist2(&d2.v, &a->v, &b->v, &c->v, NULL);

    // check if face isn't already on edge of minkowski sum and thus we
    // have touching contact
    if (ccdIsZero(dist) || ccdIsZero(dist2)){
        v[0] = ccdPtAddVertex(pt, a);
        v[1] = ccdPtAddVertex(pt, b);
        v[2] = ccdPtAddVertex(pt, c);
        e[0] = ccdPtAddEdge(pt, v[0], v[1]);
        e[1] = ccdPtAddEdge(pt, v[1], v[2]);
        e[2] = ccdPtAddEdge(pt, v[2], v[0]);
        *nearest = (ccd_pt_el_t *)ccdPtAddFace(pt, e[0], e[1], e[2]);
        if (*nearest == NULL)
            return -2;

        return -1;
    }

    // form polyhedron
    v[0] = ccdPtAddVertex(pt, a);
    v[1] = ccdPtAddVertex(pt, b);
    v[2] = ccdPtAddVertex(pt, c);
    v[3] = ccdPtAddVertex(pt, &d);
    v[4] = ccdPtAddVertex(pt, &d2);

    e[0] = ccdPtAddEdge(pt, v[0], v[1]);
    e[1] = ccdPtAddEdge(pt, v[1], v[2]);
    e[2] = ccdPtAddEdge(pt, v[2], v[0]);

    e[3] = ccdPtAddEdge(pt, v[3], v[0]);
    e[4] = ccdPtAddEdge(pt, v[3], v[1]);
    e[5] = ccdPtAddEdge(pt, v[3], v[2]);

    e[6] = ccdPtAddEdge(pt, v[4], v[0]);
    e[7] = ccdPtAddEdge(pt, v[4], v[1]);
    e[8] = ccdPtAddEdge(pt, v[4], v[2]);

    if (ccdPtAddFace(pt, e[3], e[4], e[0]) == NULL
            || ccdPtAddFace(pt, e[4], e[5], e[1]) == NULL
            || ccdPtAddFace(pt, e[5], e[3], e[2]) == NULL

            || ccdPtAddFace(pt, e[6], e[7], e[0]) == NULL
            || ccdPtAddFace(pt, e[7], e[8], e[1]) == NULL
            || ccdPtAddFace(pt, e[8], e[6], e[2]) == NULL){
        return -2;
    }

    return 0;
}

/** Transforms simplex to polytope, two vertices required */
static int simplexToPolytope2(const void *obj1, const void *obj2,
                              const ccd_t *ccd,
                              const ccd_simplex_t *simplex,
                              ccd_pt_t *pt, ccd_pt_el_t **nearest)
{
    const ccd_support_t *a, *b;
    ccd_vec3_t ab, ac, dir;
    ccd_support_t supp[4];
    ccd_pt_vertex_t *v[6];
    ccd_pt_edge_t *e[12];
    size_t i;
    int found;

    a = ccdSimplexPoint(simplex, 0);
    b = ccdSimplexPoint(simplex, 1);

    // This situation is a bit tricky. If only one segment comes from
    // previous run of GJK - it means that either this segment is on
    // minkowski edge (and thus we have touch contact) or it it isn't and
    // therefore segment is somewhere *inside* minkowski sum and it *must*
    // be possible to fully enclose this segment with polyhedron formed by
    // at least 8 triangle faces.

    // get first support point (any)
    found = 0;
    for (i = 0; i < ccd_points_on_sphere_len; i++){
        __ccdSupport(obj1, obj2, &ccd_points_on_sphere[i], ccd, &supp[0]);
        if (!ccdVec3Eq(&a->v, &supp[0].v) && !ccdVec3Eq(&b->v, &supp[0].v)){
            found = 1;
            break;
        }
    }
    if (!found)
        goto simplexToPolytope2_touching_contact;

    // get second support point in opposite direction than supp[0]
    ccdVec3Copy(&dir, &supp[0].v);
    ccdVec3Scale(&dir, -CCD_ONE);
    __ccdSupport(obj1, obj2, &dir, ccd, &supp[1]);
    if (ccdVec3Eq(&a->v, &supp[1].v) || ccdVec3Eq(&b->v, &supp[1].v))
        goto simplexToPolytope2_touching_contact;

    // next will be in direction of normal of triangle a,supp[0],supp[1]
    ccdVec3Sub2(&ab, &supp[0].v, &a->v);
    ccdVec3Sub2(&ac, &supp[1].v, &a->v);
    ccdVec3Cross(&dir, &ab, &ac);
    __ccdSupport(obj1, obj2, &dir, ccd, &supp[2]);
    if (ccdVec3Eq(&a->v, &supp[2].v) || ccdVec3Eq(&b->v, &supp[2].v))
        goto simplexToPolytope2_touching_contact;

    // and last one will be in opposite direction
    ccdVec3Scale(&dir, -CCD_ONE);
    __ccdSupport(obj1, obj2, &dir, ccd, &supp[3]);
    if (ccdVec3Eq(&a->v, &supp[3].v) || ccdVec3Eq(&b->v, &supp[3].v))
        goto simplexToPolytope2_touching_contact;

    goto simplexToPolytope2_not_touching_contact;
simplexToPolytope2_touching_contact:
    v[0] = ccdPtAddVertex(pt, a);
    v[1] = ccdPtAddVertex(pt, b);
    *nearest = (ccd_pt_el_t *)ccdPtAddEdge(pt, v[0], v[1]);
    if (*nearest == NULL)
        return -2;

    return -1;

simplexToPolytope2_not_touching_contact:
    // form polyhedron
    v[0] = ccdPtAddVertex(pt, a);
    v[1] = ccdPtAddVertex(pt, &supp[0]);
    v[2] = ccdPtAddVertex(pt, b);
    v[3] = ccdPtAddVertex(pt, &supp[1]);
    v[4] = ccdPtAddVertex(pt, &supp[2]);
    v[5] = ccdPtAddVertex(pt, &supp[3]);

    e[0] = ccdPtAddEdge(pt, v[0], v[1]);
    e[1] = ccdPtAddEdge(pt, v[1], v[2]);
    e[2] = ccdPtAddEdge(pt, v[2], v[3]);
    e[3] = ccdPtAddEdge(pt, v[3], v[0]);

    e[4] = ccdPtAddEdge(pt, v[4], v[0]);
    e[5] = ccdPtAddEdge(pt, v[4], v[1]);
    e[6] = ccdPtAddEdge(pt, v[4], v[2]);
    e[7] = ccdPtAddEdge(pt, v[4], v[3]);

    e[8]  = ccdPtAddEdge(pt, v[5], v[0]);
    e[9]  = ccdPtAddEdge(pt, v[5], v[1]);
    e[10] = ccdPtAddEdge(pt, v[5], v[2]);
    e[11] = ccdPtAddEdge(pt, v[5], v[3]);

    if (ccdPtAddFace(pt, e[4], e[5], e[0]) == NULL
            || ccdPtAddFace(pt, e[5], e[6], e[1]) == NULL
            || ccdPtAddFace(pt, e[6], e[7], e[2]) == NULL
            || ccdPtAddFace(pt, e[7], e[4], e[3]) == NULL

            || ccdPtAddFace(pt, e[8],  e[9],  e[0]) == NULL
            || ccdPtAddFace(pt, e[9],  e[10], e[1]) == NULL
            || ccdPtAddFace(pt, e[10], e[11], e[2]) == NULL
            || ccdPtAddFace(pt, e[11], e[8],  e[3]) == NULL){
        return -2;
    }

    return 0;
}

/** Expands polytope's tri by new vertex v. Triangle tri is replaced by
 *  three triangles each with one vertex in v. */
static int expandPolytope(ccd_pt_t *pt, ccd_pt_el_t *el,
                          const ccd_support_t *newv)
{
    ccd_pt_vertex_t *v[5];
    ccd_pt_edge_t *e[8];
    ccd_pt_face_t *f[2];


    // element can be either segment or triangle
    if (el->type == CCD_PT_EDGE){
        // In this case, segment should be replaced by new point.
        // Simpliest case is when segment stands alone and in this case
        // this segment is replaced by two other segments both connected to
        // newv.
        // Segment can be also connected to max two faces and in that case
        // each face must be replaced by two other faces. To do this
        // correctly it is necessary to have correctly ordered edges and
        // vertices which is exactly what is done in following code.
        //

        ccdPtEdgeVertices((const ccd_pt_edge_t *)el, &v[0], &v[2]);

        ccdPtEdgeFaces((ccd_pt_edge_t *)el, &f[0], &f[1]);

        if (f[0]){
            ccdPtFaceEdges(f[0], &e[0], &e[1], &e[2]);
            if (e[0] == (ccd_pt_edge_t *)el){
                e[0] = e[2];
            }else if (e[1] == (ccd_pt_edge_t *)el){
                e[1] = e[2];
            }
            ccdPtEdgeVertices(e[0], &v[1], &v[3]);
            if (v[1] != v[0] && v[3] != v[0]){
                e[2] = e[0];
                e[0] = e[1];
                e[1] = e[2];
                if (v[1] == v[2])
                    v[1] = v[3];
            }else{
                if (v[1] == v[0])
                    v[1] = v[3];
            }

            if (f[1]){
                ccdPtFaceEdges(f[1], &e[2], &e[3], &e[4]);
                if (e[2] == (ccd_pt_edge_t *)el){
                    e[2] = e[4];
                }else if (e[3] == (ccd_pt_edge_t *)el){
                    e[3] = e[4];
                }
                ccdPtEdgeVertices(e[2], &v[3], &v[4]);
                if (v[3] != v[2] && v[4] != v[2]){
                    e[4] = e[2];
                    e[2] = e[3];
                    e[3] = e[4];
                    if (v[3] == v[0])
                        v[3] = v[4];
                }else{
                    if (v[3] == v[2])
                        v[3] = v[4];
                }
            }


            v[4] = ccdPtAddVertex(pt, newv);

            ccdPtDelFace(pt, f[0]);
            if (f[1]){
                ccdPtDelFace(pt, f[1]);
                ccdPtDelEdge(pt, (ccd_pt_edge_t *)el);
            }

            e[4] = ccdPtAddEdge(pt, v[4], v[2]);
            e[5] = ccdPtAddEdge(pt, v[4], v[0]);
            e[6] = ccdPtAddEdge(pt, v[4], v[1]);
            if (f[1])
                e[7] = ccdPtAddEdge(pt, v[4], v[3]);


            if (ccdPtAddFace(pt, e[1], e[4], e[6]) == NULL
                    || ccdPtAddFace(pt, e[0], e[6], e[5]) == NULL){
                return -2;
            }

            if (f[1]){
                if (ccdPtAddFace(pt, e[3], e[5], e[7]) == NULL
                        || ccdPtAddFace(pt, e[4], e[7], e[2]) == NULL){
                    return -2;
                }
            }else{
                if (ccdPtAddFace(pt, e[4], e[5], (ccd_pt_edge_t *)el) == NULL)
                    return -2;
            }
        }
    }else{ // el->type == CCD_PT_FACE
        // replace triangle by tetrahedron without base (base would be the
        // triangle that will be removed)

        // get triplet of surrounding edges and vertices of triangle face
        ccdPtFaceEdges((const ccd_pt_face_t *)el, &e[0], &e[1], &e[2]);
        ccdPtEdgeVertices(e[0], &v[0], &v[1]);
        ccdPtEdgeVertices(e[1], &v[2], &v[3]);

        // following code sorts edges to have e[0] between vertices 0-1,
        // e[1] between 1-2 and e[2] between 2-0
        if (v[2] != v[1] && v[3] != v[1]){
            // swap e[1] and e[2] 
            e[3] = e[1];
            e[1] = e[2];
            e[2] = e[3];
        }
        if (v[3] != v[0] && v[3] != v[1])
            v[2] = v[3];

        // remove triangle face
        ccdPtDelFace(pt, (ccd_pt_face_t *)el);

        // expand triangle to tetrahedron
        v[3] = ccdPtAddVertex(pt, newv);
        e[3] = ccdPtAddEdge(pt, v[3], v[0]);
        e[4] = ccdPtAddEdge(pt, v[3], v[1]);
        e[5] = ccdPtAddEdge(pt, v[3], v[2]);

        if (ccdPtAddFace(pt, e[3], e[4], e[0]) == NULL
                || ccdPtAddFace(pt, e[4], e[5], e[1]) == NULL
                || ccdPtAddFace(pt, e[5], e[3], e[2]) == NULL){
            return -2;
        }
    }

    return 0;
}

/** Finds next support point (and stores it in out argument).
 *  Returns 0 on success, -1 otherwise */
static int nextSupport(const void *obj1, const void *obj2, const ccd_t *ccd,
                       const ccd_pt_el_t *el,
                       ccd_support_t *out)
{
    ccd_vec3_t *a, *b, *c;
    ccd_real_t dist;

    if (el->type == CCD_PT_VERTEX)
        return -1;

    // touch contact
    if (ccdIsZero(el->dist))
        return -1;

    __ccdSupport(obj1, obj2, &el->witness, ccd, out);

    // Compute dist of support point along element witness point direction
    // so we can determine whether we expanded a polytope surrounding the
    // origin a bit.
    dist = ccdVec3Dot(&out->v, &el->witness);

    if (dist - el->dist < ccd->epa_tolerance)
        return -1;

    if (el->type == CCD_PT_EDGE){
        // fetch end points of edge
        ccdPtEdgeVec3((ccd_pt_edge_t *)el, &a, &b);

        // get distance from segment
        dist = ccdVec3PointSegmentDist2(&out->v, a, b, NULL);
    }else{ // el->type == CCD_PT_FACE
        // fetch vertices of triangle face
        ccdPtFaceVec3((ccd_pt_face_t *)el, &a, &b, &c);

        // check if new point can significantly expand polytope
        dist = ccdVec3PointTriDist2(&out->v, a, b, c, NULL);
    }

    if (dist < ccd->epa_tolerance)
        return -1;

    return 0;
}
