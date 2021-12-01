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

#include "support.h"

void __ccdSupport(const void *obj1, const void *obj2,
                  const ccd_vec3_t *_dir, const ccd_t *ccd,
                  ccd_support_t *supp)
{
    ccd_vec3_t dir;

    ccdVec3Copy(&dir, _dir);

    ccd->support1(obj1, &dir, &supp->v1);

    ccdVec3Scale(&dir, -CCD_ONE);
    ccd->support2(obj2, &dir, &supp->v2);

    ccdVec3Sub2(&supp->v, &supp->v1, &supp->v2);
}
