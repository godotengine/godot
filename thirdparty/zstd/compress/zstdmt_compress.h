/*
 * Copyright (c) 2016-present, Yann Collet, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under both the BSD-style license (found in the
 * LICENSE file in the root directory of this source tree) and the GPLv2 (found
 * in the COPYING file in the root directory of this source tree).
 * You may select, at your option, one of the above-listed licenses.
 */

 #ifndef ZSTDMT_COMPRESS_H
 #define ZSTDMT_COMPRESS_H

 #if defined (__cplusplus)
 extern "C" {
 #endif


/* Note : This is an internal API.
 *        Some methods are still exposed (ZSTDLIB_API),
 *        because it used to be the only way to invoke MT compression.
 *        Now, it's recommended to use ZSTD_compress_generic() instead.
 *        These methods will stop being exposed in a future version */

/* ===   Dependencies   === */
#include <stddef.h>                /* size_t */
#define ZSTD_STATIC_LINKING_ONLY   /* ZSTD_parameters */
#include "zstd.h"            /* ZSTD_inBuffer, ZSTD_outBuffer, ZSTDLIB_API */


/* ===   Memory management   === */
typedef struct ZSTDMT_CCtx_s ZSTDMT_CCtx;
ZSTDLIB_API ZSTDMT_CCtx* ZSTDMT_createCCtx(unsigned nbThreads);
ZSTDLIB_API ZSTDMT_CCtx* ZSTDMT_createCCtx_advanced(unsigned nbThreads,
                                                    ZSTD_customMem cMem);
ZSTDLIB_API size_t ZSTDMT_freeCCtx(ZSTDMT_CCtx* mtctx);

ZSTDLIB_API size_t ZSTDMT_sizeof_CCtx(ZSTDMT_CCtx* mtctx);


/* ===   Simple buffer-to-butter one-pass function   === */

ZSTDLIB_API size_t ZSTDMT_compressCCtx(ZSTDMT_CCtx* mtctx,
                                       void* dst, size_t dstCapacity,
                                 const void* src, size_t srcSize,
                                       int compressionLevel);



/* ===   Streaming functions   === */

ZSTDLIB_API size_t ZSTDMT_initCStream(ZSTDMT_CCtx* mtctx, int compressionLevel);
ZSTDLIB_API size_t ZSTDMT_resetCStream(ZSTDMT_CCtx* mtctx, unsigned long long pledgedSrcSize);  /**< if srcSize is not known at reset time, use ZSTD_CONTENTSIZE_UNKNOWN. Note: for compatibility with older programs, 0 means the same as ZSTD_CONTENTSIZE_UNKNOWN, but it may change in the future, to mean "empty" */

ZSTDLIB_API size_t ZSTDMT_compressStream(ZSTDMT_CCtx* mtctx, ZSTD_outBuffer* output, ZSTD_inBuffer* input);

ZSTDLIB_API size_t ZSTDMT_flushStream(ZSTDMT_CCtx* mtctx, ZSTD_outBuffer* output);   /**< @return : 0 == all flushed; >0 : still some data to be flushed; or an error code (ZSTD_isError()) */
ZSTDLIB_API size_t ZSTDMT_endStream(ZSTDMT_CCtx* mtctx, ZSTD_outBuffer* output);     /**< @return : 0 == all flushed; >0 : still some data to be flushed; or an error code (ZSTD_isError()) */


/* ===   Advanced functions and parameters  === */

#ifndef ZSTDMT_JOBSIZE_MIN
#  define ZSTDMT_JOBSIZE_MIN (1U << 20)   /* 1 MB - Minimum size of each compression job */
#endif

ZSTDLIB_API size_t ZSTDMT_compress_advanced(ZSTDMT_CCtx* mtctx,
                                           void* dst, size_t dstCapacity,
                                     const void* src, size_t srcSize,
                                     const ZSTD_CDict* cdict,
                                           ZSTD_parameters const params,
                                           unsigned overlapLog);

ZSTDLIB_API size_t ZSTDMT_initCStream_advanced(ZSTDMT_CCtx* mtctx,
                                        const void* dict, size_t dictSize,   /* dict can be released after init, a local copy is preserved within zcs */
                                        ZSTD_parameters params,
                                        unsigned long long pledgedSrcSize);  /* pledgedSrcSize is optional and can be zero == unknown */

ZSTDLIB_API size_t ZSTDMT_initCStream_usingCDict(ZSTDMT_CCtx* mtctx,
                                        const ZSTD_CDict* cdict,
                                        ZSTD_frameParameters fparams,
                                        unsigned long long pledgedSrcSize);  /* note : zero means empty */

/* ZSTDMT_parameter :
 * List of parameters that can be set using ZSTDMT_setMTCtxParameter() */
typedef enum {
    ZSTDMT_p_jobSize,           /* Each job is compressed in parallel. By default, this value is dynamically determined depending on compression parameters. Can be set explicitly here. */
    ZSTDMT_p_overlapSectionLog  /* Each job may reload a part of previous job to enhance compressionr ratio; 0 == no overlap, 6(default) == use 1/8th of window, >=9 == use full window */
} ZSTDMT_parameter;

/* ZSTDMT_setMTCtxParameter() :
 * allow setting individual parameters, one at a time, among a list of enums defined in ZSTDMT_parameter.
 * The function must be called typically after ZSTD_createCCtx() but __before ZSTDMT_init*() !__
 * Parameters not explicitly reset by ZSTDMT_init*() remain the same in consecutive compression sessions.
 * @return : 0, or an error code (which can be tested using ZSTD_isError()) */
ZSTDLIB_API size_t ZSTDMT_setMTCtxParameter(ZSTDMT_CCtx* mtctx, ZSTDMT_parameter parameter, unsigned value);


/*! ZSTDMT_compressStream_generic() :
 *  Combines ZSTDMT_compressStream() with ZSTDMT_flushStream() or ZSTDMT_endStream()
 *  depending on flush directive.
 * @return : minimum amount of data still to be flushed
 *           0 if fully flushed
 *           or an error code */
ZSTDLIB_API size_t ZSTDMT_compressStream_generic(ZSTDMT_CCtx* mtctx,
                                                ZSTD_outBuffer* output,
                                                ZSTD_inBuffer* input,
                                                ZSTD_EndDirective endOp);


/* ===   Private definitions; never ever use directly  === */

size_t ZSTDMT_CCtxParam_setMTCtxParameter(ZSTD_CCtx_params* params, ZSTDMT_parameter parameter, unsigned value);

/* ZSTDMT_CCtxParam_setNbThreads()
 * Set nbThreads, and clamp it correctly,
 * also reset jobSize and overlapLog */
size_t ZSTDMT_CCtxParam_setNbThreads(ZSTD_CCtx_params* params, unsigned nbThreads);

/* ZSTDMT_getNbThreads():
 * @return nb threads currently active in mtctx.
 * mtctx must be valid */
size_t ZSTDMT_getNbThreads(const ZSTDMT_CCtx* mtctx);

/*! ZSTDMT_initCStream_internal() :
 *  Private use only. Init streaming operation.
 *  expects params to be valid.
 *  must receive dict, or cdict, or none, but not both.
 *  @return : 0, or an error code */
size_t ZSTDMT_initCStream_internal(ZSTDMT_CCtx* zcs,
                    const void* dict, size_t dictSize, ZSTD_dictMode_e dictMode,
                    const ZSTD_CDict* cdict,
                    ZSTD_CCtx_params params, unsigned long long pledgedSrcSize);


#if defined (__cplusplus)
}
#endif

#endif   /* ZSTDMT_COMPRESS_H */
