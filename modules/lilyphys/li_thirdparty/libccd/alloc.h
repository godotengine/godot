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

#ifndef __CCD_ALLOC_H__
#define __CCD_ALLOC_H__

#include <stdlib.h>

#ifdef __cplusplus
extern "C" {
#endif /* __cplusplus */

/**
 * Functions and macros required for memory allocation.
 */

/* Memory allocation: */
#define __CCD_ALLOC_MEMORY(type, ptr_old, size) \
    (type *)realloc((void *)ptr_old, (size))

/** Allocate memory for one element of type.  */
#define CCD_ALLOC(type) \
    __CCD_ALLOC_MEMORY(type, NULL, sizeof(type))

/** Allocate memory for array of elements of type type.  */
#define CCD_ALLOC_ARR(type, num_elements) \
    __CCD_ALLOC_MEMORY(type, NULL, sizeof(type) * (num_elements))

#define CCD_REALLOC_ARR(ptr, type, num_elements) \
    __CCD_ALLOC_MEMORY(type, ptr, sizeof(type) * (num_elements))

#ifdef __cplusplus
} /* extern "C" */
#endif /* __cplusplus */

#endif /* __CCD_ALLOC_H__ */
