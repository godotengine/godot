/**
 * Copyright (c) 2016-present, Yann Collet, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree. An additional grant
 * of patent rights can be found in the PATENTS file in the same directory.
 */

 #ifndef ZSTDMT_COMPRESS_H
 #define ZSTDMT_COMPRESS_H

 #if defined (__cplusplus)
 extern "C" {
 #endif


/* Note : All prototypes defined in this file shall be considered experimental.
 *        There is no guarantee of API continuity (yet) on any of these prototypes */

/* ===   Dependencies   === */
#include <stddef.h>   /* size_t */
#define ZSTD_STATIC_LINKING_ONLY   /* ZSTD_parameters */
#include "zstd.h"     /* ZSTD_inBuffer, ZSTD_outBuffer, ZSTDLIB_API */


/* ===   Simple one-pass functions   === */

typedef struct ZSTDMT_CCtx_s ZSTDMT_CCtx;
ZSTDLIB_API ZSTDMT_CCtx* ZSTDMT_createCCtx(unsigned nbThreads);
ZSTDLIB_API size_t ZSTDMT_freeCCtx(ZSTDMT_CCtx* cctx);

ZSTDLIB_API size_t ZSTDMT_compressCCtx(ZSTDMT_CCtx* cctx,
                           void* dst, size_t dstCapacity,
                     const void* src, size_t srcSize,
                           int compressionLevel);


/* ===   Streaming functions   === */

ZSTDLIB_API size_t ZSTDMT_initCStream(ZSTDMT_CCtx* mtctx, int compressionLevel);
ZSTDLIB_API size_t ZSTDMT_resetCStream(ZSTDMT_CCtx* mtctx, unsigned long long pledgedSrcSize);    /**< pledgedSrcSize is optional and can be zero == unknown */

ZSTDLIB_API size_t ZSTDMT_compressStream(ZSTDMT_CCtx* mtctx, ZSTD_outBuffer* output, ZSTD_inBuffer* input);

ZSTDLIB_API size_t ZSTDMT_flushStream(ZSTDMT_CCtx* mtctx, ZSTD_outBuffer* output);   /**< @return : 0 == all flushed; >0 : still some data to be flushed; or an error code (ZSTD_isError()) */
ZSTDLIB_API size_t ZSTDMT_endStream(ZSTDMT_CCtx* mtctx, ZSTD_outBuffer* output);     /**< @return : 0 == all flushed; >0 : still some data to be flushed; or an error code (ZSTD_isError()) */


/* ===   Advanced functions and parameters  === */

#ifndef ZSTDMT_SECTION_SIZE_MIN
#  define ZSTDMT_SECTION_SIZE_MIN (1U << 20)   /* 1 MB - Minimum size of each compression job */
#endif

ZSTDLIB_API size_t ZSTDMT_initCStream_advanced(ZSTDMT_CCtx* mtctx, const void* dict, size_t dictSize,  /**< dict can be released after init, a local copy is preserved within zcs */
                                          ZSTD_parameters params, unsigned long long pledgedSrcSize);  /**< pledgedSrcSize is optional and can be zero == unknown */

/* ZSDTMT_parameter :
 * List of parameters that can be set using ZSTDMT_setMTCtxParameter() */
typedef enum {
    ZSTDMT_p_sectionSize,        /* size of input "section". Each section is compressed in parallel. 0 means default, which is dynamically determined within compression functions */
    ZSTDMT_p_overlapSectionLog   /* Log of overlapped section; 0 == no overlap, 6(default) == use 1/8th of window, >=9 == use full window */
} ZSDTMT_parameter;

/* ZSTDMT_setMTCtxParameter() :
 * allow setting individual parameters, one at a time, among a list of enums defined in ZSTDMT_parameter.
 * The function must be called typically after ZSTD_createCCtx().
 * Parameters not explicitly reset by ZSTDMT_init*() remain the same in consecutive compression sessions.
 * @return : 0, or an error code (which can be tested using ZSTD_isError()) */
ZSTDLIB_API size_t ZSTDMT_setMTCtxParameter(ZSTDMT_CCtx* mtctx, ZSDTMT_parameter parameter, unsigned value);


#if defined (__cplusplus)
}
#endif

#endif   /* ZSTDMT_COMPRESS_H */
