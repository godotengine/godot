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

#ifndef __CCD_SIMPLEX_H__
#define __CCD_SIMPLEX_H__

#include "ccd/compiler.h"
#include "support.h"

#ifdef __cplusplus
extern "C" {
#endif /* __cplusplus */

struct _ccd_simplex_t {
    ccd_support_t ps[4];
    int last; //!< index of last added point
};
typedef struct _ccd_simplex_t ccd_simplex_t;


_ccd_inline void ccdSimplexInit(ccd_simplex_t *s);
_ccd_inline int ccdSimplexSize(const ccd_simplex_t *s);
_ccd_inline const ccd_support_t *ccdSimplexLast(const ccd_simplex_t *s);
_ccd_inline const ccd_support_t *ccdSimplexPoint(const ccd_simplex_t *s, int idx);
_ccd_inline ccd_support_t *ccdSimplexPointW(ccd_simplex_t *s, int idx);

_ccd_inline void ccdSimplexAdd(ccd_simplex_t *s, const ccd_support_t *v);
_ccd_inline void ccdSimplexSet(ccd_simplex_t *s, size_t pos, const ccd_support_t *a);
_ccd_inline void ccdSimplexSetSize(ccd_simplex_t *s, int size);
_ccd_inline void ccdSimplexSwap(ccd_simplex_t *s, size_t pos1, size_t pos2);


/**** INLINES ****/

_ccd_inline void ccdSimplexInit(ccd_simplex_t *s)
{
    s->last = -1;
}

_ccd_inline int ccdSimplexSize(const ccd_simplex_t *s)
{
    return s->last + 1;
}

_ccd_inline const ccd_support_t *ccdSimplexLast(const ccd_simplex_t *s)
{
    return ccdSimplexPoint(s, s->last);
}

_ccd_inline const ccd_support_t *ccdSimplexPoint(const ccd_simplex_t *s, int idx)
{
    // here is no check on boundaries
    return &s->ps[idx];
}
_ccd_inline ccd_support_t *ccdSimplexPointW(ccd_simplex_t *s, int idx)
{
    return &s->ps[idx];
}

_ccd_inline void ccdSimplexAdd(ccd_simplex_t *s, const ccd_support_t *v)
{
    // here is no check on boundaries in sake of speed
    ++s->last;
    ccdSupportCopy(s->ps + s->last, v);
}

_ccd_inline void ccdSimplexSet(ccd_simplex_t *s, size_t pos, const ccd_support_t *a)
{
    ccdSupportCopy(s->ps + pos, a);
}

_ccd_inline void ccdSimplexSetSize(ccd_simplex_t *s, int size)
{
    s->last = size - 1;
}

_ccd_inline void ccdSimplexSwap(ccd_simplex_t *s, size_t pos1, size_t pos2)
{
    ccd_support_t supp;

    ccdSupportCopy(&supp, &s->ps[pos1]);
    ccdSupportCopy(&s->ps[pos1], &s->ps[pos2]);
    ccdSupportCopy(&s->ps[pos2], &supp);
}

#ifdef __cplusplus
} /* extern "C" */
#endif /* __cplusplus */

#endif /* __CCD_SIMPLEX_H__ */
