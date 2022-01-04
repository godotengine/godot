/***
 * libccd
 * ---------------------------------
 * Copyright (c)2010-2013 Daniel Fiser <danfis@danfis.cz>
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

#ifndef __CCD_VEC3_H__
#define __CCD_VEC3_H__

#include <math.h>
#include <float.h>
#include <stdlib.h>
#include "compiler.h"
#include "config.h"
#include "ccd_export.h"

#ifdef __cplusplus
extern "C" {
#endif /* __cplusplus */


#ifndef CCD_SINGLE
# ifndef CCD_DOUBLE
#  error You must define CCD_SINGLE or CCD_DOUBLE
# endif /* CCD_DOUBLE */
#endif /* CCD_SINGLE */

#ifdef WIN32
# define CCD_FMIN(x, y) ((x) < (y) ? (x) : (y))
#endif /* WIN32 */

#ifdef CCD_SINGLE
# ifdef CCD_DOUBLE
#  error You can define either CCD_SINGLE or CCD_DOUBLE, not both!
# endif /* CCD_DOUBLE */

typedef float ccd_real_t;

//# define CCD_EPS 1E-6
# define CCD_EPS FLT_EPSILON

# define CCD_REAL_MAX FLT_MAX

# define CCD_REAL(x) (x ## f)   /*!< form a constant */
# define CCD_SQRT(x) (sqrtf(x)) /*!< square root */
# define CCD_FABS(x) (fabsf(x)) /*!< absolute value */
# define CCD_FMAX(x, y) (fmaxf((x), (y))) /*!< maximum of two floats */

# ifndef CCD_FMIN
#  define CCD_FMIN(x, y) (fminf((x), (y))) /*!< minimum of two floats */
# endif /* CCD_FMIN */

#endif /* CCD_SINGLE */

#ifdef CCD_DOUBLE
typedef double ccd_real_t;

//# define CCD_EPS 1E-10
# define CCD_EPS DBL_EPSILON

# define CCD_REAL_MAX DBL_MAX

# define CCD_REAL(x) (x)       /*!< form a constant */
# define CCD_SQRT(x) (sqrt(x)) /*!< square root */
# define CCD_FABS(x) (fabs(x)) /*!< absolute value */
# define CCD_FMAX(x, y) (fmax((x), (y))) /*!< maximum of two floats */

# ifndef CCD_FMIN
#  define CCD_FMIN(x, y) (fmin((x), (y))) /*!< minimum of two floats */
# endif /* CCD_FMIN */

#endif /* CCD_DOUBLE */

#define CCD_ONE CCD_REAL(1.)
#define CCD_ZERO CCD_REAL(0.)

struct _ccd_vec3_t {
    ccd_real_t v[3];
};
typedef struct _ccd_vec3_t ccd_vec3_t;


/**
 * Holds origin (0,0,0) - this variable is meant to be read-only!
 */
CCD_EXPORT extern ccd_vec3_t *ccd_vec3_origin;

/**
 * Array of points uniformly distributed on unit sphere.
 */
CCD_EXPORT extern ccd_vec3_t *ccd_points_on_sphere;
CCD_EXPORT extern size_t ccd_points_on_sphere_len;

/** Returns sign of value. */
_ccd_inline int ccdSign(ccd_real_t val);
/** Returns true if val is zero. **/
_ccd_inline int ccdIsZero(ccd_real_t val);
/** Returns true if a and b equal. **/
_ccd_inline int ccdEq(ccd_real_t a, ccd_real_t b);


#define CCD_VEC3_STATIC(x, y, z) \
    { { (x), (y), (z) } }

#define CCD_VEC3(name, x, y, z) \
    ccd_vec3_t name = CCD_VEC3_STATIC((x), (y), (z))

_ccd_inline ccd_real_t ccdVec3X(const ccd_vec3_t *v);
_ccd_inline ccd_real_t ccdVec3Y(const ccd_vec3_t *v);
_ccd_inline ccd_real_t ccdVec3Z(const ccd_vec3_t *v);

/**
 * Returns true if a and b equal.
 */
_ccd_inline int ccdVec3Eq(const ccd_vec3_t *a, const ccd_vec3_t *b);

/**
 * Returns squared length of vector.
 */
_ccd_inline ccd_real_t ccdVec3Len2(const ccd_vec3_t *v);

/**
 * Returns distance between a and b.
 */
_ccd_inline ccd_real_t ccdVec3Dist2(const ccd_vec3_t *a, const ccd_vec3_t *b);


_ccd_inline void ccdVec3Set(ccd_vec3_t *v, ccd_real_t x, ccd_real_t y, ccd_real_t z);

/**
 * v = w
 */
_ccd_inline void ccdVec3Copy(ccd_vec3_t *v, const ccd_vec3_t *w);

/**
 * Substracts coordinates of vector w from vector v. v = v - w
 */
_ccd_inline void ccdVec3Sub(ccd_vec3_t *v, const ccd_vec3_t *w);

/**
 * Adds coordinates of vector w to vector v. v = v + w
 */
_ccd_inline void ccdVec3Add(ccd_vec3_t *v, const ccd_vec3_t *w);

/**
 * d = v - w
 */
_ccd_inline void ccdVec3Sub2(ccd_vec3_t *d, const ccd_vec3_t *v, const ccd_vec3_t *w);

/**
 * d = d * k;
 */
_ccd_inline void ccdVec3Scale(ccd_vec3_t *d, ccd_real_t k);

/**
 * Normalizes given vector to unit length.
 */
_ccd_inline void ccdVec3Normalize(ccd_vec3_t *d);


/**
 * Dot product of two vectors.
 */
_ccd_inline ccd_real_t ccdVec3Dot(const ccd_vec3_t *a, const ccd_vec3_t *b);

/**
 * Cross product: d = a x b.
 */
_ccd_inline void ccdVec3Cross(ccd_vec3_t *d, const ccd_vec3_t *a, const ccd_vec3_t *b);


/**
 * Returns distance^2 of point P to segment ab.
 * If witness is non-NULL it is filled with coordinates of point from which
 * was computed distance to point P.
 */
CCD_EXPORT ccd_real_t ccdVec3PointSegmentDist2(const ccd_vec3_t *P,
                                                const ccd_vec3_t *a,
                                                const ccd_vec3_t *b,
                                                ccd_vec3_t *witness);

/**
 * Returns distance^2 of point P from triangle formed by triplet a, b, c.
 * If witness vector is provided it is filled with coordinates of point
 * from which was computed distance to point P.
 */
CCD_EXPORT ccd_real_t ccdVec3PointTriDist2(const ccd_vec3_t *P,
                                            const ccd_vec3_t *a,
                                            const ccd_vec3_t *b,
                                            const ccd_vec3_t *c,
                                            ccd_vec3_t *witness);


/**** INLINES ****/
_ccd_inline int ccdSign(ccd_real_t val)
{
    if (ccdIsZero(val)){
        return 0;
    }else if (val < CCD_ZERO){
        return -1;
    }
    return 1;
}

_ccd_inline int ccdIsZero(ccd_real_t val)
{
    return CCD_FABS(val) < CCD_EPS;
}

_ccd_inline int ccdEq(ccd_real_t _a, ccd_real_t _b)
{
    ccd_real_t ab;
    ccd_real_t a, b;

    ab = CCD_FABS(_a - _b);
    if (CCD_FABS(ab) < CCD_EPS)
        return 1;

    a = CCD_FABS(_a);
    b = CCD_FABS(_b);
    if (b > a){
        return ab < CCD_EPS * b;
    }else{
        return ab < CCD_EPS * a;
    }
}


_ccd_inline ccd_real_t ccdVec3X(const ccd_vec3_t *v)
{
    return v->v[0];
}

_ccd_inline ccd_real_t ccdVec3Y(const ccd_vec3_t *v)
{
    return v->v[1];
}

_ccd_inline ccd_real_t ccdVec3Z(const ccd_vec3_t *v)
{
    return v->v[2];
}

_ccd_inline int ccdVec3Eq(const ccd_vec3_t *a, const ccd_vec3_t *b)
{
    return ccdEq(ccdVec3X(a), ccdVec3X(b))
            && ccdEq(ccdVec3Y(a), ccdVec3Y(b))
            && ccdEq(ccdVec3Z(a), ccdVec3Z(b));
}

_ccd_inline ccd_real_t ccdVec3Len2(const ccd_vec3_t *v)
{
    return ccdVec3Dot(v, v);
}

_ccd_inline ccd_real_t ccdVec3Dist2(const ccd_vec3_t *a, const ccd_vec3_t *b)
{
    ccd_vec3_t ab;
    ccdVec3Sub2(&ab, a, b);
    return ccdVec3Len2(&ab);
}

_ccd_inline void ccdVec3Set(ccd_vec3_t *v, ccd_real_t x, ccd_real_t y, ccd_real_t z)
{
    v->v[0] = x;
    v->v[1] = y;
    v->v[2] = z;
}

_ccd_inline void ccdVec3Copy(ccd_vec3_t *v, const ccd_vec3_t *w)
{
    *v = *w;
}

_ccd_inline void ccdVec3Sub(ccd_vec3_t *v, const ccd_vec3_t *w)
{
    v->v[0] -= w->v[0];
    v->v[1] -= w->v[1];
    v->v[2] -= w->v[2];
}
_ccd_inline void ccdVec3Sub2(ccd_vec3_t *d, const ccd_vec3_t *v, const ccd_vec3_t *w)
{
    d->v[0] = v->v[0] - w->v[0];
    d->v[1] = v->v[1] - w->v[1];
    d->v[2] = v->v[2] - w->v[2];
}

_ccd_inline void ccdVec3Add(ccd_vec3_t *v, const ccd_vec3_t *w)
{
    v->v[0] += w->v[0];
    v->v[1] += w->v[1];
    v->v[2] += w->v[2];
}

_ccd_inline void ccdVec3Scale(ccd_vec3_t *d, ccd_real_t k)
{
    d->v[0] *= k;
    d->v[1] *= k;
    d->v[2] *= k;
}

_ccd_inline void ccdVec3Normalize(ccd_vec3_t *d)
{
    ccd_real_t k = CCD_ONE / CCD_SQRT(ccdVec3Len2(d));
    ccdVec3Scale(d, k);
}

_ccd_inline ccd_real_t ccdVec3Dot(const ccd_vec3_t *a, const ccd_vec3_t *b)
{
    ccd_real_t dot;

    dot  = a->v[0] * b->v[0];
    dot += a->v[1] * b->v[1];
    dot += a->v[2] * b->v[2];
    return dot;
}

_ccd_inline void ccdVec3Cross(ccd_vec3_t *d, const ccd_vec3_t *a, const ccd_vec3_t *b)
{
    d->v[0] = (a->v[1] * b->v[2]) - (a->v[2] * b->v[1]);
    d->v[1] = (a->v[2] * b->v[0]) - (a->v[0] * b->v[2]);
    d->v[2] = (a->v[0] * b->v[1]) - (a->v[1] * b->v[0]);
}

#ifdef __cplusplus
} /* extern "C" */
#endif /* __cplusplus */

#endif /* __CCD_VEC3_H__ */
