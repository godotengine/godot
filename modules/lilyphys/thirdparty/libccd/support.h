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

#ifndef __CCD_SUPPORT_H__
#define __CCD_SUPPORT_H__

#include "ccd/ccd.h"

#ifdef __cplusplus
extern "C" {
#endif /* __cplusplus */

struct _ccd_support_t {
    ccd_vec3_t v;  //!< Support point in minkowski sum
    ccd_vec3_t v1; //!< Support point in obj1
    ccd_vec3_t v2; //!< Support point in obj2
};
typedef struct _ccd_support_t ccd_support_t;

_ccd_inline void ccdSupportCopy(ccd_support_t *, const ccd_support_t *s);

/**
 * Computes support point of obj1 and obj2 in direction dir.
 * Support point is returned via supp.
 */
CCD_EXPORT void __ccdSupport(const void *obj1, const void *obj2,
                  const ccd_vec3_t *dir, const ccd_t *ccd,
                  ccd_support_t *supp);


/**** INLINES ****/
_ccd_inline void ccdSupportCopy(ccd_support_t *d, const ccd_support_t *s)
{
    *d = *s;
}

#ifdef __cplusplus
} /* extern "C" */
#endif /* __cplusplus */

#endif /* __CCD_SUPPORT_H__ */
