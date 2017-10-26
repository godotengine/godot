/*
 * Copyright (c) 2016-present, Yann Collet, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under both the BSD-style license (found in the
 * LICENSE file in the root directory of this source tree) and the GPLv2 (found
 * in the COPYING file in the root directory of this source tree).
 * You may select, at your option, one of the above-listed licenses.
 */


/*-*************************************
*  Tuning parameters
***************************************/
#ifndef ZSTD_CLEVEL_DEFAULT
#  define ZSTD_CLEVEL_DEFAULT 3
#endif


/*-*************************************
*  Dependencies
***************************************/
#include <string.h>         /* memset */
#include "mem.h"
#define FSE_STATIC_LINKING_ONLY   /* FSE_encodeSymbol */
#include "fse.h"
#define HUF_STATIC_LINKING_ONLY
#include "huf.h"
#include "zstd_compress.h"
#include "zstd_fast.h"
#include "zstd_double_fast.h"
#include "zstd_lazy.h"
#include "zstd_opt.h"
#include "zstd_ldm.h"


/*-*************************************
*  Helper functions
***************************************/
size_t ZSTD_compressBound(size_t srcSize) {
    return ZSTD_COMPRESSBOUND(srcSize);
}


/*-*************************************
*  Sequence storage
***************************************/
static void ZSTD_resetSeqStore(seqStore_t* ssPtr)
{
    ssPtr->lit = ssPtr->litStart;
    ssPtr->sequences = ssPtr->sequencesStart;
    ssPtr->longLengthID = 0;
}


/*-*************************************
*  Context memory management
***************************************/
struct ZSTD_CDict_s {
    void* dictBuffer;
    const void* dictContent;
    size_t dictContentSize;
    ZSTD_CCtx* refContext;
};  /* typedef'd to ZSTD_CDict within "zstd.h" */

ZSTD_CCtx* ZSTD_createCCtx(void)
{
    return ZSTD_createCCtx_advanced(ZSTD_defaultCMem);
}

ZSTD_CCtx* ZSTD_createCCtx_advanced(ZSTD_customMem customMem)
{
    ZSTD_CCtx* cctx;

    if (!customMem.customAlloc ^ !customMem.customFree) return NULL;

    cctx = (ZSTD_CCtx*) ZSTD_calloc(sizeof(ZSTD_CCtx), customMem);
    if (!cctx) return NULL;
    cctx->customMem = customMem;
    cctx->requestedParams.compressionLevel = ZSTD_CLEVEL_DEFAULT;
    ZSTD_STATIC_ASSERT(zcss_init==0);
    ZSTD_STATIC_ASSERT(ZSTD_CONTENTSIZE_UNKNOWN==(0ULL - 1));
    return cctx;
}

ZSTD_CCtx* ZSTD_initStaticCCtx(void *workspace, size_t workspaceSize)
{
    ZSTD_CCtx* const cctx = (ZSTD_CCtx*) workspace;
    if (workspaceSize <= sizeof(ZSTD_CCtx)) return NULL;  /* minimum size */
    if ((size_t)workspace & 7) return NULL;  /* must be 8-aligned */
    memset(workspace, 0, workspaceSize);   /* may be a bit generous, could memset be smaller ? */
    cctx->staticSize = workspaceSize;
    cctx->workSpace = (void*)(cctx+1);
    cctx->workSpaceSize = workspaceSize - sizeof(ZSTD_CCtx);

    /* entropy space (never moves) */
    if (cctx->workSpaceSize < sizeof(ZSTD_entropyCTables_t)) return NULL;
    assert(((size_t)cctx->workSpace & (sizeof(void*)-1)) == 0);   /* ensure correct alignment */
    cctx->entropy = (ZSTD_entropyCTables_t*)cctx->workSpace;

    return cctx;
}

size_t ZSTD_freeCCtx(ZSTD_CCtx* cctx)
{
    if (cctx==NULL) return 0;   /* support free on NULL */
    if (cctx->staticSize) return ERROR(memory_allocation);   /* not compatible with static CCtx */
    ZSTD_free(cctx->workSpace, cctx->customMem);
    cctx->workSpace = NULL;
    ZSTD_freeCDict(cctx->cdictLocal);
    cctx->cdictLocal = NULL;
#ifdef ZSTD_MULTITHREAD
    ZSTDMT_freeCCtx(cctx->mtctx);
    cctx->mtctx = NULL;
#endif
    ZSTD_free(cctx, cctx->customMem);
    return 0;   /* reserved as a potential error code in the future */
}


static size_t ZSTD_sizeof_mtctx(const ZSTD_CCtx* cctx)
{
#ifdef ZSTD_MULTITHREAD
    return ZSTDMT_sizeof_CCtx(cctx->mtctx);
#else
    (void) cctx;
    return 0;
#endif
}


size_t ZSTD_sizeof_CCtx(const ZSTD_CCtx* cctx)
{
    if (cctx==NULL) return 0;   /* support sizeof on NULL */
    DEBUGLOG(3, "sizeof(*cctx) : %u", (U32)sizeof(*cctx));
    DEBUGLOG(3, "workSpaceSize (including streaming buffers): %u", (U32)cctx->workSpaceSize);
    DEBUGLOG(3, "inner cdict : %u", (U32)ZSTD_sizeof_CDict(cctx->cdictLocal));
    DEBUGLOG(3, "inner MTCTX : %u", (U32)ZSTD_sizeof_mtctx(cctx));
    return sizeof(*cctx) + cctx->workSpaceSize
           + ZSTD_sizeof_CDict(cctx->cdictLocal)
           + ZSTD_sizeof_mtctx(cctx);
}

size_t ZSTD_sizeof_CStream(const ZSTD_CStream* zcs)
{
    return ZSTD_sizeof_CCtx(zcs);  /* same object */
}

/* private API call, for dictBuilder only */
const seqStore_t* ZSTD_getSeqStore(const ZSTD_CCtx* ctx) { return &(ctx->seqStore); }

#define ZSTD_CLEVEL_CUSTOM 999

static ZSTD_compressionParameters ZSTD_getCParamsFromCCtxParams(
        ZSTD_CCtx_params params, U64 srcSizeHint, size_t dictSize)
{
    return (params.compressionLevel == ZSTD_CLEVEL_CUSTOM ?
                    params.cParams :
                    ZSTD_getCParams(params.compressionLevel, srcSizeHint, dictSize));
}

static void ZSTD_cLevelToCCtxParams_srcSize(ZSTD_CCtx_params* params, U64 srcSize)
{
    params->cParams = ZSTD_getCParamsFromCCtxParams(*params, srcSize, 0);
    params->compressionLevel = ZSTD_CLEVEL_CUSTOM;
}

static void ZSTD_cLevelToCParams(ZSTD_CCtx* cctx)
{
    ZSTD_cLevelToCCtxParams_srcSize(
            &cctx->requestedParams, cctx->pledgedSrcSizePlusOne-1);
}

static void ZSTD_cLevelToCCtxParams(ZSTD_CCtx_params* params)
{
    ZSTD_cLevelToCCtxParams_srcSize(params, 0);
}

static ZSTD_CCtx_params ZSTD_makeCCtxParamsFromCParams(
        ZSTD_compressionParameters cParams)
{
    ZSTD_CCtx_params cctxParams;
    memset(&cctxParams, 0, sizeof(cctxParams));
    cctxParams.cParams = cParams;
    cctxParams.compressionLevel = ZSTD_CLEVEL_CUSTOM;
    return cctxParams;
}

static ZSTD_CCtx_params* ZSTD_createCCtxParams_advanced(
        ZSTD_customMem customMem)
{
    ZSTD_CCtx_params* params;
    if (!customMem.customAlloc ^ !customMem.customFree) return NULL;
    params = (ZSTD_CCtx_params*)ZSTD_calloc(
            sizeof(ZSTD_CCtx_params), customMem);
    if (!params) { return NULL; }
    params->customMem = customMem;
    params->compressionLevel = ZSTD_CLEVEL_DEFAULT;
    return params;
}

ZSTD_CCtx_params* ZSTD_createCCtxParams(void)
{
    return ZSTD_createCCtxParams_advanced(ZSTD_defaultCMem);
}

size_t ZSTD_freeCCtxParams(ZSTD_CCtx_params* params)
{
    if (params == NULL) { return 0; }
    ZSTD_free(params, params->customMem);
    return 0;
}

size_t ZSTD_resetCCtxParams(ZSTD_CCtx_params* params)
{
    return ZSTD_initCCtxParams(params, ZSTD_CLEVEL_DEFAULT);
}

size_t ZSTD_initCCtxParams(ZSTD_CCtx_params* cctxParams, int compressionLevel) {
    if (!cctxParams) { return ERROR(GENERIC); }
    memset(cctxParams, 0, sizeof(*cctxParams));
    cctxParams->compressionLevel = compressionLevel;
    return 0;
}

size_t ZSTD_initCCtxParams_advanced(ZSTD_CCtx_params* cctxParams, ZSTD_parameters params)
{
    if (!cctxParams) { return ERROR(GENERIC); }
    CHECK_F( ZSTD_checkCParams(params.cParams) );
    memset(cctxParams, 0, sizeof(*cctxParams));
    cctxParams->cParams = params.cParams;
    cctxParams->fParams = params.fParams;
    cctxParams->compressionLevel = ZSTD_CLEVEL_CUSTOM;
    return 0;
}

static ZSTD_CCtx_params ZSTD_assignParamsToCCtxParams(
        ZSTD_CCtx_params cctxParams, ZSTD_parameters params)
{
    ZSTD_CCtx_params ret = cctxParams;
    ret.cParams = params.cParams;
    ret.fParams = params.fParams;
    ret.compressionLevel = ZSTD_CLEVEL_CUSTOM;
    return ret;
}

#define CLAMPCHECK(val,min,max) {            \
    if (((val)<(min)) | ((val)>(max))) {     \
        return ERROR(parameter_outOfBound);  \
}   }

size_t ZSTD_CCtx_setParameter(ZSTD_CCtx* cctx, ZSTD_cParameter param, unsigned value)
{
    if (cctx->streamStage != zcss_init) return ERROR(stage_wrong);

    switch(param)
    {
    case ZSTD_p_format :
        return ZSTD_CCtxParam_setParameter(&cctx->requestedParams, param, value);

    case ZSTD_p_compressionLevel:
        if (value == 0) return 0;  /* special value : 0 means "don't change anything" */
        if (cctx->cdict) return ERROR(stage_wrong);
        return ZSTD_CCtxParam_setParameter(&cctx->requestedParams, param, value);

    case ZSTD_p_windowLog:
    case ZSTD_p_hashLog:
    case ZSTD_p_chainLog:
    case ZSTD_p_searchLog:
    case ZSTD_p_minMatch:
    case ZSTD_p_targetLength:
    case ZSTD_p_compressionStrategy:
        if (value == 0) return 0;  /* special value : 0 means "don't change anything" */
        if (cctx->cdict) return ERROR(stage_wrong);
        ZSTD_cLevelToCParams(cctx);  /* Can optimize if srcSize is known */
        return ZSTD_CCtxParam_setParameter(&cctx->requestedParams, param, value);

    case ZSTD_p_contentSizeFlag:
    case ZSTD_p_checksumFlag:
    case ZSTD_p_dictIDFlag:
        return ZSTD_CCtxParam_setParameter(&cctx->requestedParams, param, value);

    case ZSTD_p_forceMaxWindow :  /* Force back-references to remain < windowSize,
                                   * even when referencing into Dictionary content
                                   * default : 0 when using a CDict, 1 when using a Prefix */
        cctx->loadedDictEnd = 0;
        return ZSTD_CCtxParam_setParameter(&cctx->requestedParams, param, value);

    case ZSTD_p_nbThreads:
        if (value==0) return 0;
        DEBUGLOG(5, " setting nbThreads : %u", value);
        if (value > 1 && cctx->staticSize) {
            return ERROR(parameter_unsupported);  /* MT not compatible with static alloc */
        }
        return ZSTD_CCtxParam_setParameter(&cctx->requestedParams, param, value);

    case ZSTD_p_jobSize:
        return ZSTD_CCtxParam_setParameter(&cctx->requestedParams, param, value);

    case ZSTD_p_overlapSizeLog:
        DEBUGLOG(5, " setting overlap with nbThreads == %u", cctx->requestedParams.nbThreads);
        return ZSTD_CCtxParam_setParameter(&cctx->requestedParams, param, value);

    case ZSTD_p_enableLongDistanceMatching:
        if (cctx->cdict) return ERROR(stage_wrong);
        if (value != 0) {
            ZSTD_cLevelToCParams(cctx);
        }
        return ZSTD_CCtxParam_setParameter(&cctx->requestedParams, param, value);

    case ZSTD_p_ldmHashLog:
    case ZSTD_p_ldmMinMatch:
        if (value == 0) return 0;  /* special value : 0 means "don't change anything" */
        if (cctx->cdict) return ERROR(stage_wrong);
        return ZSTD_CCtxParam_setParameter(&cctx->requestedParams, param, value);

    case ZSTD_p_ldmBucketSizeLog:
    case ZSTD_p_ldmHashEveryLog:
        if (cctx->cdict) return ERROR(stage_wrong);
        return ZSTD_CCtxParam_setParameter(&cctx->requestedParams, param, value);

    default: return ERROR(parameter_unsupported);
    }
}

size_t ZSTD_CCtxParam_setParameter(
        ZSTD_CCtx_params* params, ZSTD_cParameter param, unsigned value)
{
    switch(param)
    {
    case ZSTD_p_format :
        if (value > (unsigned)ZSTD_f_zstd1_magicless)
            return ERROR(parameter_unsupported);
        params->format = (ZSTD_format_e)value;
        return 0;

    case ZSTD_p_compressionLevel :
        if ((int)value > ZSTD_maxCLevel()) value = ZSTD_maxCLevel();
        if (value == 0) return 0;
        params->compressionLevel = value;
        return 0;

    case ZSTD_p_windowLog :
        if (value == 0) return 0;
        CLAMPCHECK(value, ZSTD_WINDOWLOG_MIN, ZSTD_WINDOWLOG_MAX);
        ZSTD_cLevelToCCtxParams(params);
        params->cParams.windowLog = value;
        return 0;

    case ZSTD_p_hashLog :
        if (value == 0) return 0;
        CLAMPCHECK(value, ZSTD_HASHLOG_MIN, ZSTD_HASHLOG_MAX);
        ZSTD_cLevelToCCtxParams(params);
        params->cParams.hashLog = value;
        return 0;

    case ZSTD_p_chainLog :
        if (value == 0) return 0;
        CLAMPCHECK(value, ZSTD_CHAINLOG_MIN, ZSTD_CHAINLOG_MAX);
        ZSTD_cLevelToCCtxParams(params);
        params->cParams.chainLog = value;
        return 0;

    case ZSTD_p_searchLog :
        if (value == 0) return 0;
        CLAMPCHECK(value, ZSTD_SEARCHLOG_MIN, ZSTD_SEARCHLOG_MAX);
        ZSTD_cLevelToCCtxParams(params);
        params->cParams.searchLog = value;
        return 0;

    case ZSTD_p_minMatch :
        if (value == 0) return 0;
        CLAMPCHECK(value, ZSTD_SEARCHLENGTH_MIN, ZSTD_SEARCHLENGTH_MAX);
        ZSTD_cLevelToCCtxParams(params);
        params->cParams.searchLength = value;
        return 0;

    case ZSTD_p_targetLength :
        if (value == 0) return 0;
        CLAMPCHECK(value, ZSTD_TARGETLENGTH_MIN, ZSTD_TARGETLENGTH_MAX);
        ZSTD_cLevelToCCtxParams(params);
        params->cParams.targetLength = value;
        return 0;

    case ZSTD_p_compressionStrategy :
        if (value == 0) return 0;
        CLAMPCHECK(value, (unsigned)ZSTD_fast, (unsigned)ZSTD_btultra);
        ZSTD_cLevelToCCtxParams(params);
        params->cParams.strategy = (ZSTD_strategy)value;
        return 0;

    case ZSTD_p_contentSizeFlag :
        /* Content size written in frame header _when known_ (default:1) */
        DEBUGLOG(5, "set content size flag = %u", (value>0));
        params->fParams.contentSizeFlag = value > 0;
        return 0;

    case ZSTD_p_checksumFlag :
        /* A 32-bits content checksum will be calculated and written at end of frame (default:0) */
        params->fParams.checksumFlag = value > 0;
        return 0;

    case ZSTD_p_dictIDFlag : /* When applicable, dictionary's dictID is provided in frame header (default:1) */
        DEBUGLOG(5, "set dictIDFlag = %u", (value>0));
        params->fParams.noDictIDFlag = (value == 0);
        return 0;

    case ZSTD_p_forceMaxWindow :
        params->forceWindow = value > 0;
        return 0;

    case ZSTD_p_nbThreads :
        if (value == 0) return 0;
#ifndef ZSTD_MULTITHREAD
        if (value > 1) return ERROR(parameter_unsupported);
        return 0;
#else
        return ZSTDMT_initializeCCtxParameters(params, value);
#endif

    case ZSTD_p_jobSize :
#ifndef ZSTD_MULTITHREAD
        return ERROR(parameter_unsupported);
#else
        if (params->nbThreads <= 1) return ERROR(parameter_unsupported);
        return ZSTDMT_CCtxParam_setMTCtxParameter(params, ZSTDMT_p_sectionSize, value);
#endif

    case ZSTD_p_overlapSizeLog :
#ifndef ZSTD_MULTITHREAD
        return ERROR(parameter_unsupported);
#else
        if (params->nbThreads <= 1) return ERROR(parameter_unsupported);
        return ZSTDMT_CCtxParam_setMTCtxParameter(params, ZSTDMT_p_overlapSectionLog, value);
#endif

    case ZSTD_p_enableLongDistanceMatching :
        if (value != 0) {
            ZSTD_cLevelToCCtxParams(params);
            params->cParams.windowLog = ZSTD_LDM_DEFAULT_WINDOW_LOG;
        }
        return ZSTD_ldm_initializeParameters(&params->ldmParams, value);

    case ZSTD_p_ldmHashLog :
        if (value == 0) return 0;
        CLAMPCHECK(value, ZSTD_HASHLOG_MIN, ZSTD_HASHLOG_MAX);
        params->ldmParams.hashLog = value;
        return 0;

    case ZSTD_p_ldmMinMatch :
        if (value == 0) return 0;
        CLAMPCHECK(value, ZSTD_LDM_MINMATCH_MIN, ZSTD_LDM_MINMATCH_MAX);
        params->ldmParams.minMatchLength = value;
        return 0;

    case ZSTD_p_ldmBucketSizeLog :
        if (value > ZSTD_LDM_BUCKETSIZELOG_MAX) {
            return ERROR(parameter_outOfBound);
        }
        params->ldmParams.bucketSizeLog = value;
        return 0;

    case ZSTD_p_ldmHashEveryLog :
        if (value > ZSTD_WINDOWLOG_MAX - ZSTD_HASHLOG_MIN) {
            return ERROR(parameter_outOfBound);
        }
        params->ldmParams.hashEveryLog = value;
        return 0;

    default: return ERROR(parameter_unsupported);
    }
}

/**
 * This function should be updated whenever ZSTD_CCtx_params is updated.
 * Parameters are copied manually before the dictionary is loaded.
 * The multithreading parameters jobSize and overlapSizeLog are set only if
 * nbThreads > 1.
 *
 * Pledged srcSize is treated as unknown.
 */
size_t ZSTD_CCtx_setParametersUsingCCtxParams(
        ZSTD_CCtx* cctx, const ZSTD_CCtx_params* params)
{
    if (cctx->streamStage != zcss_init) return ERROR(stage_wrong);
    if (cctx->cdict) return ERROR(stage_wrong);

    /* Assume the compression and frame parameters are validated */
    cctx->requestedParams.cParams = params->cParams;
    cctx->requestedParams.fParams = params->fParams;
    cctx->requestedParams.compressionLevel = params->compressionLevel;

    /* Set force window explicitly since it sets cctx->loadedDictEnd */
    CHECK_F( ZSTD_CCtx_setParameter(
                   cctx, ZSTD_p_forceMaxWindow, params->forceWindow) );

    /* Set multithreading parameters explicitly */
    CHECK_F( ZSTD_CCtx_setParameter(cctx, ZSTD_p_nbThreads, params->nbThreads) );
    if (params->nbThreads > 1) {
        CHECK_F( ZSTD_CCtx_setParameter(cctx, ZSTD_p_jobSize, params->jobSize) );
        CHECK_F( ZSTD_CCtx_setParameter(
                    cctx, ZSTD_p_overlapSizeLog, params->overlapSizeLog) );
    }

    /* Copy long distance matching parameters */
    cctx->requestedParams.ldmParams = params->ldmParams;

    /* customMem is used only for create/free params and can be ignored */
    return 0;
}

ZSTDLIB_API size_t ZSTD_CCtx_setPledgedSrcSize(ZSTD_CCtx* cctx, unsigned long long pledgedSrcSize)
{
    DEBUGLOG(4, " setting pledgedSrcSize to %u", (U32)pledgedSrcSize);
    if (cctx->streamStage != zcss_init) return ERROR(stage_wrong);
    cctx->pledgedSrcSizePlusOne = pledgedSrcSize+1;
    return 0;
}

size_t ZSTD_CCtx_loadDictionary_advanced(
        ZSTD_CCtx* cctx, const void* dict, size_t dictSize,
        ZSTD_dictLoadMethod_e dictLoadMethod, ZSTD_dictMode_e dictMode)
{
    if (cctx->streamStage != zcss_init) return ERROR(stage_wrong);
    if (cctx->staticSize) return ERROR(memory_allocation);  /* no malloc for static CCtx */
    DEBUGLOG(4, "load dictionary of size %u", (U32)dictSize);
    ZSTD_freeCDict(cctx->cdictLocal);  /* in case one already exists */
    if (dict==NULL || dictSize==0) {   /* no dictionary mode */
        cctx->cdictLocal = NULL;
        cctx->cdict = NULL;
    } else {
        ZSTD_compressionParameters const cParams =
                ZSTD_getCParamsFromCCtxParams(cctx->requestedParams, 0, dictSize);
        cctx->cdictLocal = ZSTD_createCDict_advanced(
                                dict, dictSize,
                                dictLoadMethod, dictMode,
                                cParams, cctx->customMem);
        cctx->cdict = cctx->cdictLocal;
        if (cctx->cdictLocal == NULL)
            return ERROR(memory_allocation);
    }
    return 0;
}

ZSTDLIB_API size_t ZSTD_CCtx_loadDictionary_byReference(
      ZSTD_CCtx* cctx, const void* dict, size_t dictSize)
{
    return ZSTD_CCtx_loadDictionary_advanced(
            cctx, dict, dictSize, ZSTD_dlm_byRef, ZSTD_dm_auto);
}

ZSTDLIB_API size_t ZSTD_CCtx_loadDictionary(ZSTD_CCtx* cctx, const void* dict, size_t dictSize)
{
    return ZSTD_CCtx_loadDictionary_advanced(
            cctx, dict, dictSize, ZSTD_dlm_byCopy, ZSTD_dm_auto);
}


size_t ZSTD_CCtx_refCDict(ZSTD_CCtx* cctx, const ZSTD_CDict* cdict)
{
    if (cctx->streamStage != zcss_init) return ERROR(stage_wrong);
    cctx->cdict = cdict;
    memset(&cctx->prefixDict, 0, sizeof(cctx->prefixDict));  /* exclusive */
    return 0;
}

size_t ZSTD_CCtx_refPrefix(ZSTD_CCtx* cctx, const void* prefix, size_t prefixSize)
{
    return ZSTD_CCtx_refPrefix_advanced(cctx, prefix, prefixSize, ZSTD_dm_rawContent);
}

size_t ZSTD_CCtx_refPrefix_advanced(
        ZSTD_CCtx* cctx, const void* prefix, size_t prefixSize, ZSTD_dictMode_e dictMode)
{
    if (cctx->streamStage != zcss_init) return ERROR(stage_wrong);
    cctx->cdict = NULL;   /* prefix discards any prior cdict */
    cctx->prefixDict.dict = prefix;
    cctx->prefixDict.dictSize = prefixSize;
    cctx->prefixDict.dictMode = dictMode;
    return 0;
}

static void ZSTD_startNewCompression(ZSTD_CCtx* cctx)
{
    cctx->streamStage = zcss_init;
    cctx->pledgedSrcSizePlusOne = 0;
}

/*! ZSTD_CCtx_reset() :
 *  Also dumps dictionary */
void ZSTD_CCtx_reset(ZSTD_CCtx* cctx)
{
    ZSTD_startNewCompression(cctx);
    cctx->cdict = NULL;
}

/** ZSTD_checkCParams() :
    control CParam values remain within authorized range.
    @return : 0, or an error code if one value is beyond authorized range */
size_t ZSTD_checkCParams(ZSTD_compressionParameters cParams)
{
    CLAMPCHECK(cParams.windowLog, ZSTD_WINDOWLOG_MIN, ZSTD_WINDOWLOG_MAX);
    CLAMPCHECK(cParams.chainLog, ZSTD_CHAINLOG_MIN, ZSTD_CHAINLOG_MAX);
    CLAMPCHECK(cParams.hashLog, ZSTD_HASHLOG_MIN, ZSTD_HASHLOG_MAX);
    CLAMPCHECK(cParams.searchLog, ZSTD_SEARCHLOG_MIN, ZSTD_SEARCHLOG_MAX);
    CLAMPCHECK(cParams.searchLength, ZSTD_SEARCHLENGTH_MIN, ZSTD_SEARCHLENGTH_MAX);
    CLAMPCHECK(cParams.targetLength, ZSTD_TARGETLENGTH_MIN, ZSTD_TARGETLENGTH_MAX);
    if ((U32)(cParams.strategy) > (U32)ZSTD_btultra)
        return ERROR(parameter_unsupported);
    return 0;
}

/** ZSTD_clampCParams() :
 *  make CParam values within valid range.
 *  @return : valid CParams */
static ZSTD_compressionParameters ZSTD_clampCParams(ZSTD_compressionParameters cParams)
{
#   define CLAMP(val,min,max) {      \
        if (val<min) val=min;        \
        else if (val>max) val=max;   \
    }
    CLAMP(cParams.windowLog, ZSTD_WINDOWLOG_MIN, ZSTD_WINDOWLOG_MAX);
    CLAMP(cParams.chainLog, ZSTD_CHAINLOG_MIN, ZSTD_CHAINLOG_MAX);
    CLAMP(cParams.hashLog, ZSTD_HASHLOG_MIN, ZSTD_HASHLOG_MAX);
    CLAMP(cParams.searchLog, ZSTD_SEARCHLOG_MIN, ZSTD_SEARCHLOG_MAX);
    CLAMP(cParams.searchLength, ZSTD_SEARCHLENGTH_MIN, ZSTD_SEARCHLENGTH_MAX);
    CLAMP(cParams.targetLength, ZSTD_TARGETLENGTH_MIN, ZSTD_TARGETLENGTH_MAX);
    if ((U32)(cParams.strategy) > (U32)ZSTD_btultra) cParams.strategy = ZSTD_btultra;
    return cParams;
}

/** ZSTD_cycleLog() :
 *  condition for correct operation : hashLog > 1 */
static U32 ZSTD_cycleLog(U32 hashLog, ZSTD_strategy strat)
{
    U32 const btScale = ((U32)strat >= (U32)ZSTD_btlazy2);
    return hashLog - btScale;
}

/** ZSTD_adjustCParams_internal() :
    optimize `cPar` for a given input (`srcSize` and `dictSize`).
    mostly downsizing to reduce memory consumption and initialization latency.
    Both `srcSize` and `dictSize` are optional (use 0 if unknown).
    Note : cPar is considered validated at this stage. Use ZSTD_checkCParams() to ensure that condition. */
ZSTD_compressionParameters ZSTD_adjustCParams_internal(ZSTD_compressionParameters cPar, unsigned long long srcSize, size_t dictSize)
{
    static const U64 minSrcSize = 513; /* (1<<9) + 1 */
    static const U64 maxWindowResize = 1ULL << (ZSTD_WINDOWLOG_MAX-1);
    assert(ZSTD_checkCParams(cPar)==0);

    if (dictSize && (srcSize+1<2) /* srcSize unknown */ )
        srcSize = minSrcSize;  /* presumed small when there is a dictionary */
    else if (srcSize == 0)
        srcSize = ZSTD_CONTENTSIZE_UNKNOWN;  /* 0 == unknown : presumed large */

    /* resize windowLog if input is small enough, to use less memory */
    if ( (srcSize < maxWindowResize)
      && (dictSize < maxWindowResize) )  {
        U32 const tSize = (U32)(srcSize + dictSize);
        static U32 const hashSizeMin = 1 << ZSTD_HASHLOG_MIN;
        U32 const srcLog = (tSize < hashSizeMin) ? ZSTD_HASHLOG_MIN :
                            ZSTD_highbit32(tSize-1) + 1;
        if (cPar.windowLog > srcLog) cPar.windowLog = srcLog;
    }
    if (cPar.hashLog > cPar.windowLog) cPar.hashLog = cPar.windowLog;
    {   U32 const cycleLog = ZSTD_cycleLog(cPar.chainLog, cPar.strategy);
        if (cycleLog > cPar.windowLog)
            cPar.chainLog -= (cycleLog - cPar.windowLog);
    }

    if (cPar.windowLog < ZSTD_WINDOWLOG_ABSOLUTEMIN)
        cPar.windowLog = ZSTD_WINDOWLOG_ABSOLUTEMIN;  /* required for frame header */

    return cPar;
}

ZSTD_compressionParameters ZSTD_adjustCParams(ZSTD_compressionParameters cPar, unsigned long long srcSize, size_t dictSize)
{
    cPar = ZSTD_clampCParams(cPar);
    return ZSTD_adjustCParams_internal(cPar, srcSize, dictSize);
}

size_t ZSTD_estimateCCtxSize_usingCCtxParams(const ZSTD_CCtx_params* params)
{
    /* Estimate CCtx size is supported for single-threaded compression only. */
    if (params->nbThreads > 1) { return ERROR(GENERIC); }
    {   ZSTD_compressionParameters const cParams =
                ZSTD_getCParamsFromCCtxParams(*params, 0, 0);
        size_t const blockSize = MIN(ZSTD_BLOCKSIZE_MAX, (size_t)1 << cParams.windowLog);
        U32    const divider = (cParams.searchLength==3) ? 3 : 4;
        size_t const maxNbSeq = blockSize / divider;
        size_t const tokenSpace = blockSize + 11*maxNbSeq;
        size_t const chainSize =
                (cParams.strategy == ZSTD_fast) ? 0 : ((size_t)1 << cParams.chainLog);
        size_t const hSize = ((size_t)1) << cParams.hashLog;
        U32    const hashLog3 = (cParams.searchLength>3) ?
                                0 : MIN(ZSTD_HASHLOG3_MAX, cParams.windowLog);
        size_t const h3Size = ((size_t)1) << hashLog3;
        size_t const entropySpace = sizeof(ZSTD_entropyCTables_t);
        size_t const tableSpace = (chainSize + hSize + h3Size) * sizeof(U32);

        size_t const optBudget =
                ((MaxML+1) + (MaxLL+1) + (MaxOff+1) + (1<<Litbits))*sizeof(U32)
                + (ZSTD_OPT_NUM+1)*(sizeof(ZSTD_match_t) + sizeof(ZSTD_optimal_t));
        size_t const optSpace = ((cParams.strategy == ZSTD_btopt) || (cParams.strategy == ZSTD_btultra)) ? optBudget : 0;

        size_t const ldmSpace = params->ldmParams.enableLdm ?
            ZSTD_ldm_getTableSize(params->ldmParams.hashLog,
                                  params->ldmParams.bucketSizeLog) : 0;

        size_t const neededSpace = entropySpace + tableSpace + tokenSpace +
                                   optSpace + ldmSpace;

        DEBUGLOG(5, "sizeof(ZSTD_CCtx) : %u", (U32)sizeof(ZSTD_CCtx));
        DEBUGLOG(5, "estimate workSpace : %u", (U32)neededSpace);
        return sizeof(ZSTD_CCtx) + neededSpace;
    }
}

size_t ZSTD_estimateCCtxSize_usingCParams(ZSTD_compressionParameters cParams)
{
    ZSTD_CCtx_params const params = ZSTD_makeCCtxParamsFromCParams(cParams);
    return ZSTD_estimateCCtxSize_usingCCtxParams(&params);
}

size_t ZSTD_estimateCCtxSize(int compressionLevel)
{
    ZSTD_compressionParameters const cParams = ZSTD_getCParams(compressionLevel, 0, 0);
    return ZSTD_estimateCCtxSize_usingCParams(cParams);
}

size_t ZSTD_estimateCStreamSize_usingCCtxParams(const ZSTD_CCtx_params* params)
{
    if (params->nbThreads > 1) { return ERROR(GENERIC); }
    {   size_t const CCtxSize = ZSTD_estimateCCtxSize_usingCCtxParams(params);
        size_t const blockSize = MIN(ZSTD_BLOCKSIZE_MAX, (size_t)1 << params->cParams.windowLog);
        size_t const inBuffSize = ((size_t)1 << params->cParams.windowLog) + blockSize;
        size_t const outBuffSize = ZSTD_compressBound(blockSize) + 1;
        size_t const streamingSize = inBuffSize + outBuffSize;

        return CCtxSize + streamingSize;
    }
}

size_t ZSTD_estimateCStreamSize_usingCParams(ZSTD_compressionParameters cParams)
{
    ZSTD_CCtx_params const params = ZSTD_makeCCtxParamsFromCParams(cParams);
    return ZSTD_estimateCStreamSize_usingCCtxParams(&params);
}

size_t ZSTD_estimateCStreamSize(int compressionLevel) {
    ZSTD_compressionParameters const cParams = ZSTD_getCParams(compressionLevel, 0, 0);
    return ZSTD_estimateCStreamSize_usingCParams(cParams);
}

static U32 ZSTD_equivalentCParams(ZSTD_compressionParameters cParams1,
                                  ZSTD_compressionParameters cParams2)
{
    U32 bslog1 = MIN(cParams1.windowLog, ZSTD_BLOCKSIZELOG_MAX);
    U32 bslog2 = MIN(cParams2.windowLog, ZSTD_BLOCKSIZELOG_MAX);
    return (bslog1 == bslog2)   /* same block size */
         & (cParams1.hashLog  == cParams2.hashLog)
         & (cParams1.chainLog == cParams2.chainLog)
         & (cParams1.strategy == cParams2.strategy)   /* opt parser space */
         & ((cParams1.searchLength==3) == (cParams2.searchLength==3));  /* hashlog3 space */
}

/** The parameters are equivalent if ldm is not enabled in both sets or
 *  all the parameters are equivalent. */
static U32 ZSTD_equivalentLdmParams(ldmParams_t ldmParams1,
                                    ldmParams_t ldmParams2)
{
    return (!ldmParams1.enableLdm && !ldmParams2.enableLdm) ||
           (ldmParams1.enableLdm == ldmParams2.enableLdm &&
            ldmParams1.hashLog == ldmParams2.hashLog &&
            ldmParams1.bucketSizeLog == ldmParams2.bucketSizeLog &&
            ldmParams1.minMatchLength == ldmParams2.minMatchLength &&
            ldmParams1.hashEveryLog == ldmParams2.hashEveryLog);
}

/** Equivalence for resetCCtx purposes */
static U32 ZSTD_equivalentParams(ZSTD_CCtx_params params1,
                                 ZSTD_CCtx_params params2)
{
    return ZSTD_equivalentCParams(params1.cParams, params2.cParams) &&
           ZSTD_equivalentLdmParams(params1.ldmParams, params2.ldmParams);
}

/*! ZSTD_continueCCtx() :
 *  reuse CCtx without reset (note : requires no dictionary) */
static size_t ZSTD_continueCCtx(ZSTD_CCtx* cctx, ZSTD_CCtx_params params, U64 pledgedSrcSize)
{
    U32 const end = (U32)(cctx->nextSrc - cctx->base);
    DEBUGLOG(4, "continue mode");
    cctx->appliedParams = params;
    cctx->pledgedSrcSizePlusOne = pledgedSrcSize+1;
    cctx->consumedSrcSize = 0;
    if (pledgedSrcSize == ZSTD_CONTENTSIZE_UNKNOWN)
        cctx->appliedParams.fParams.contentSizeFlag = 0;
    DEBUGLOG(4, "pledged content size : %u ; flag : %u",
        (U32)pledgedSrcSize, cctx->appliedParams.fParams.contentSizeFlag);
    cctx->lowLimit = end;
    cctx->dictLimit = end;
    cctx->nextToUpdate = end+1;
    cctx->stage = ZSTDcs_init;
    cctx->dictID = 0;
    cctx->loadedDictEnd = 0;
    { int i; for (i=0; i<ZSTD_REP_NUM; i++) cctx->seqStore.rep[i] = repStartValue[i]; }
    cctx->optState.litLengthSum = 0;  /* force reset of btopt stats */
    XXH64_reset(&cctx->xxhState, 0);
    return 0;
}

typedef enum { ZSTDcrp_continue, ZSTDcrp_noMemset } ZSTD_compResetPolicy_e;
typedef enum { ZSTDb_not_buffered, ZSTDb_buffered } ZSTD_buffered_policy_e;

/*! ZSTD_resetCCtx_internal() :
    note : `params` are assumed fully validated at this stage */
static size_t ZSTD_resetCCtx_internal(ZSTD_CCtx* zc,
                                      ZSTD_CCtx_params params, U64 pledgedSrcSize,
                                      ZSTD_compResetPolicy_e const crp,
                                      ZSTD_buffered_policy_e const zbuff)
{
    DEBUGLOG(4, "ZSTD_resetCCtx_internal");
    assert(!ZSTD_isError(ZSTD_checkCParams(params.cParams)));
    DEBUGLOG(4, "pledgedSrcSize: %u", (U32)pledgedSrcSize);

    if (crp == ZSTDcrp_continue) {
        if (ZSTD_equivalentParams(params, zc->appliedParams)) {
            DEBUGLOG(4, "ZSTD_equivalentParams()==1");
            assert(!(params.ldmParams.enableLdm &&
                     params.ldmParams.hashEveryLog == ZSTD_LDM_HASHEVERYLOG_NOTSET));
            zc->entropy->hufCTable_repeatMode = HUF_repeat_none;
            zc->entropy->offcode_repeatMode = FSE_repeat_none;
            zc->entropy->matchlength_repeatMode = FSE_repeat_none;
            zc->entropy->litlength_repeatMode = FSE_repeat_none;
            return ZSTD_continueCCtx(zc, params, pledgedSrcSize);
    }   }

    if (params.ldmParams.enableLdm) {
        /* Adjust long distance matching parameters */
        ZSTD_ldm_adjustParameters(&params.ldmParams, params.cParams.windowLog);
        assert(params.ldmParams.hashLog >= params.ldmParams.bucketSizeLog);
        assert(params.ldmParams.hashEveryLog < 32);
        zc->ldmState.hashPower =
                ZSTD_ldm_getHashPower(params.ldmParams.minMatchLength);
    }

    {   size_t const blockSize = MIN(ZSTD_BLOCKSIZE_MAX, (size_t)1 << params.cParams.windowLog);
        U32    const divider = (params.cParams.searchLength==3) ? 3 : 4;
        size_t const maxNbSeq = blockSize / divider;
        size_t const tokenSpace = blockSize + 11*maxNbSeq;
        size_t const chainSize = (params.cParams.strategy == ZSTD_fast) ?
                                0 : ((size_t)1 << params.cParams.chainLog);
        size_t const hSize = ((size_t)1) << params.cParams.hashLog;
        U32    const hashLog3 = (params.cParams.searchLength>3) ?
                                0 : MIN(ZSTD_HASHLOG3_MAX, params.cParams.windowLog);
        size_t const h3Size = ((size_t)1) << hashLog3;
        size_t const tableSpace = (chainSize + hSize + h3Size) * sizeof(U32);
        size_t const buffOutSize = (zbuff==ZSTDb_buffered) ? ZSTD_compressBound(blockSize)+1 : 0;
        size_t const buffInSize = (zbuff==ZSTDb_buffered) ? ((size_t)1 << params.cParams.windowLog) + blockSize : 0;
        void* ptr;

        /* Check if workSpace is large enough, alloc a new one if needed */
        {   size_t const entropySpace = sizeof(ZSTD_entropyCTables_t);
            size_t const optPotentialSpace = ((MaxML+1) + (MaxLL+1) + (MaxOff+1) + (1<<Litbits)) * sizeof(U32)
                                  + (ZSTD_OPT_NUM+1) * (sizeof(ZSTD_match_t)+sizeof(ZSTD_optimal_t));
            size_t const optSpace = ( (params.cParams.strategy == ZSTD_btopt)
                                    || (params.cParams.strategy == ZSTD_btultra)) ?
                                    optPotentialSpace : 0;
            size_t const bufferSpace = buffInSize + buffOutSize;
            size_t const ldmSpace = params.ldmParams.enableLdm
                ? ZSTD_ldm_getTableSize(params.ldmParams.hashLog, params.ldmParams.bucketSizeLog)
                : 0;
            size_t const neededSpace = entropySpace + optSpace + ldmSpace +
                                       tableSpace + tokenSpace + bufferSpace;

            if (zc->workSpaceSize < neededSpace) {  /* too small : resize */
                DEBUGLOG(5, "Need to update workSpaceSize from %uK to %uK \n",
                            (unsigned)zc->workSpaceSize>>10,
                            (unsigned)neededSpace>>10);
                /* static cctx : no resize, error out */
                if (zc->staticSize) return ERROR(memory_allocation);

                zc->workSpaceSize = 0;
                ZSTD_free(zc->workSpace, zc->customMem);
                zc->workSpace = ZSTD_malloc(neededSpace, zc->customMem);
                if (zc->workSpace == NULL) return ERROR(memory_allocation);
                zc->workSpaceSize = neededSpace;
                ptr = zc->workSpace;

                /* entropy space */
                assert(((size_t)zc->workSpace & 3) == 0);   /* ensure correct alignment */
                assert(zc->workSpaceSize >= sizeof(ZSTD_entropyCTables_t));
                zc->entropy = (ZSTD_entropyCTables_t*)zc->workSpace;
        }   }

        /* init params */
        zc->appliedParams = params;
        zc->pledgedSrcSizePlusOne = pledgedSrcSize+1;
        zc->consumedSrcSize = 0;
        if (pledgedSrcSize == ZSTD_CONTENTSIZE_UNKNOWN)
            zc->appliedParams.fParams.contentSizeFlag = 0;
        DEBUGLOG(5, "pledged content size : %u ; flag : %u",
            (U32)pledgedSrcSize, zc->appliedParams.fParams.contentSizeFlag);
        zc->blockSize = blockSize;

        XXH64_reset(&zc->xxhState, 0);
        zc->stage = ZSTDcs_init;
        zc->dictID = 0;
        zc->loadedDictEnd = 0;
        zc->entropy->hufCTable_repeatMode = HUF_repeat_none;
        zc->entropy->offcode_repeatMode = FSE_repeat_none;
        zc->entropy->matchlength_repeatMode = FSE_repeat_none;
        zc->entropy->litlength_repeatMode = FSE_repeat_none;
        zc->nextToUpdate = 1;
        zc->nextSrc = NULL;
        zc->base = NULL;
        zc->dictBase = NULL;
        zc->dictLimit = 0;
        zc->lowLimit = 0;
        { int i; for (i=0; i<ZSTD_REP_NUM; i++) zc->seqStore.rep[i] = repStartValue[i]; }
        zc->hashLog3 = hashLog3;
        zc->optState.litLengthSum = 0;

        ptr = zc->entropy + 1;

        /* opt parser space */
        if ((params.cParams.strategy == ZSTD_btopt) || (params.cParams.strategy == ZSTD_btultra)) {
            DEBUGLOG(5, "reserving optimal parser space");
            assert(((size_t)ptr & 3) == 0);  /* ensure ptr is properly aligned */
            zc->optState.litFreq = (U32*)ptr;
            zc->optState.litLengthFreq = zc->optState.litFreq + (1<<Litbits);
            zc->optState.matchLengthFreq = zc->optState.litLengthFreq + (MaxLL+1);
            zc->optState.offCodeFreq = zc->optState.matchLengthFreq + (MaxML+1);
            ptr = zc->optState.offCodeFreq + (MaxOff+1);
            zc->optState.matchTable = (ZSTD_match_t*)ptr;
            ptr = zc->optState.matchTable + ZSTD_OPT_NUM+1;
            zc->optState.priceTable = (ZSTD_optimal_t*)ptr;
            ptr = zc->optState.priceTable + ZSTD_OPT_NUM+1;
        }

        /* ldm hash table */
        /* initialize bucketOffsets table later for pointer alignment */
        if (params.ldmParams.enableLdm) {
            size_t const ldmHSize = ((size_t)1) << params.ldmParams.hashLog;
            memset(ptr, 0, ldmHSize * sizeof(ldmEntry_t));
            assert(((size_t)ptr & 3) == 0); /* ensure ptr is properly aligned */
            zc->ldmState.hashTable = (ldmEntry_t*)ptr;
            ptr = zc->ldmState.hashTable + ldmHSize;
        }

        /* table Space */
        if (crp!=ZSTDcrp_noMemset) memset(ptr, 0, tableSpace);   /* reset tables only */
        assert(((size_t)ptr & 3) == 0);  /* ensure ptr is properly aligned */
        zc->hashTable = (U32*)(ptr);
        zc->chainTable = zc->hashTable + hSize;
        zc->hashTable3 = zc->chainTable + chainSize;
        ptr = zc->hashTable3 + h3Size;

        /* sequences storage */
        zc->seqStore.sequencesStart = (seqDef*)ptr;
        ptr = zc->seqStore.sequencesStart + maxNbSeq;
        zc->seqStore.llCode = (BYTE*) ptr;
        zc->seqStore.mlCode = zc->seqStore.llCode + maxNbSeq;
        zc->seqStore.ofCode = zc->seqStore.mlCode + maxNbSeq;
        zc->seqStore.litStart = zc->seqStore.ofCode + maxNbSeq;
        ptr = zc->seqStore.litStart + blockSize;

        /* ldm bucketOffsets table */
        if (params.ldmParams.enableLdm) {
            size_t const ldmBucketSize =
                  ((size_t)1) << (params.ldmParams.hashLog -
                                  params.ldmParams.bucketSizeLog);
            memset(ptr, 0, ldmBucketSize);
            zc->ldmState.bucketOffsets = (BYTE*)ptr;
            ptr = zc->ldmState.bucketOffsets + ldmBucketSize;
        }

        /* buffers */
        zc->inBuffSize = buffInSize;
        zc->inBuff = (char*)ptr;
        zc->outBuffSize = buffOutSize;
        zc->outBuff = zc->inBuff + buffInSize;

        return 0;
    }
}

/* ZSTD_invalidateRepCodes() :
 * ensures next compression will not use repcodes from previous block.
 * Note : only works with regular variant;
 *        do not use with extDict variant ! */
void ZSTD_invalidateRepCodes(ZSTD_CCtx* cctx) {
    int i;
    for (i=0; i<ZSTD_REP_NUM; i++) cctx->seqStore.rep[i] = 0;
}


/*! ZSTD_copyCCtx_internal() :
 *  Duplicate an existing context `srcCCtx` into another one `dstCCtx`.
 *  The "context", in this case, refers to the hash and chain tables, entropy
 *  tables, and dictionary offsets.
 *  Only works during stage ZSTDcs_init (i.e. after creation, but before first call to ZSTD_compressContinue()).
 *  pledgedSrcSize=0 means "empty" if fParams.contentSizeFlag=1
 *  @return : 0, or an error code */
static size_t ZSTD_copyCCtx_internal(ZSTD_CCtx* dstCCtx,
                            const ZSTD_CCtx* srcCCtx,
                            ZSTD_frameParameters fParams,
                            unsigned long long pledgedSrcSize,
                            ZSTD_buffered_policy_e zbuff)
{
    DEBUGLOG(5, "ZSTD_copyCCtx_internal");
    if (srcCCtx->stage!=ZSTDcs_init) return ERROR(stage_wrong);

    memcpy(&dstCCtx->customMem, &srcCCtx->customMem, sizeof(ZSTD_customMem));
    {   ZSTD_CCtx_params params = dstCCtx->requestedParams;
        /* Copy only compression parameters related to tables. */
        params.cParams = srcCCtx->appliedParams.cParams;
        params.fParams = fParams;
        ZSTD_resetCCtx_internal(dstCCtx, params, pledgedSrcSize,
                                ZSTDcrp_noMemset, zbuff);
    }

    /* copy tables */
    {   size_t const chainSize = (srcCCtx->appliedParams.cParams.strategy == ZSTD_fast) ? 0 : ((size_t)1 << srcCCtx->appliedParams.cParams.chainLog);
        size_t const hSize =  (size_t)1 << srcCCtx->appliedParams.cParams.hashLog;
        size_t const h3Size = (size_t)1 << srcCCtx->hashLog3;
        size_t const tableSpace = (chainSize + hSize + h3Size) * sizeof(U32);
        assert((U32*)dstCCtx->chainTable == (U32*)dstCCtx->hashTable + hSize);  /* chainTable must follow hashTable */
        assert((U32*)dstCCtx->hashTable3 == (U32*)dstCCtx->chainTable + chainSize);
        memcpy(dstCCtx->hashTable, srcCCtx->hashTable, tableSpace);   /* presumes all tables follow each other */
    }

    /* copy dictionary offsets */
    dstCCtx->nextToUpdate = srcCCtx->nextToUpdate;
    dstCCtx->nextToUpdate3= srcCCtx->nextToUpdate3;
    dstCCtx->nextSrc      = srcCCtx->nextSrc;
    dstCCtx->base         = srcCCtx->base;
    dstCCtx->dictBase     = srcCCtx->dictBase;
    dstCCtx->dictLimit    = srcCCtx->dictLimit;
    dstCCtx->lowLimit     = srcCCtx->lowLimit;
    dstCCtx->loadedDictEnd= srcCCtx->loadedDictEnd;
    dstCCtx->dictID       = srcCCtx->dictID;

    /* copy entropy tables */
    memcpy(dstCCtx->entropy, srcCCtx->entropy, sizeof(ZSTD_entropyCTables_t));

    return 0;
}

/*! ZSTD_copyCCtx() :
 *  Duplicate an existing context `srcCCtx` into another one `dstCCtx`.
 *  Only works during stage ZSTDcs_init (i.e. after creation, but before first call to ZSTD_compressContinue()).
 *  pledgedSrcSize==0 means "unknown".
*   @return : 0, or an error code */
size_t ZSTD_copyCCtx(ZSTD_CCtx* dstCCtx, const ZSTD_CCtx* srcCCtx, unsigned long long pledgedSrcSize)
{
    ZSTD_frameParameters fParams = { 1 /*content*/, 0 /*checksum*/, 0 /*noDictID*/ };
    ZSTD_buffered_policy_e const zbuff = (ZSTD_buffered_policy_e)(srcCCtx->inBuffSize>0);
    ZSTD_STATIC_ASSERT((U32)ZSTDb_buffered==1);
    fParams.contentSizeFlag = pledgedSrcSize>0;

    return ZSTD_copyCCtx_internal(dstCCtx, srcCCtx, fParams, pledgedSrcSize, zbuff);
}


/*! ZSTD_reduceTable() :
 *  reduce table indexes by `reducerValue` */
static void ZSTD_reduceTable (U32* const table, U32 const size, U32 const reducerValue)
{
    U32 u;
    for (u=0 ; u < size ; u++) {
        if (table[u] < reducerValue) table[u] = 0;
        else table[u] -= reducerValue;
    }
}

/*! ZSTD_ldm_reduceTable() :
 *  reduce table indexes by `reducerValue` */
static void ZSTD_ldm_reduceTable(ldmEntry_t* const table, U32 const size,
                                 U32 const reducerValue)
{
    U32 u;
    for (u = 0; u < size; u++) {
        if (table[u].offset < reducerValue) table[u].offset = 0;
        else table[u].offset -= reducerValue;
    }
}

/*! ZSTD_reduceIndex() :
*   rescale all indexes to avoid future overflow (indexes are U32) */
static void ZSTD_reduceIndex (ZSTD_CCtx* zc, const U32 reducerValue)
{
    { U32 const hSize = (U32)1 << zc->appliedParams.cParams.hashLog;
      ZSTD_reduceTable(zc->hashTable, hSize, reducerValue); }

    { U32 const chainSize = (zc->appliedParams.cParams.strategy == ZSTD_fast) ? 0 : ((U32)1 << zc->appliedParams.cParams.chainLog);
      ZSTD_reduceTable(zc->chainTable, chainSize, reducerValue); }

    { U32 const h3Size = (zc->hashLog3) ? (U32)1 << zc->hashLog3 : 0;
      ZSTD_reduceTable(zc->hashTable3, h3Size, reducerValue); }

    { if (zc->appliedParams.ldmParams.enableLdm) {
          U32 const ldmHSize = (U32)1 << zc->appliedParams.ldmParams.hashLog;
          ZSTD_ldm_reduceTable(zc->ldmState.hashTable, ldmHSize, reducerValue);
      }
    }
}


/*-*******************************************************
*  Block entropic compression
*********************************************************/

/* See doc/zstd_compression_format.md for detailed format description */

size_t ZSTD_noCompressBlock (void* dst, size_t dstCapacity, const void* src, size_t srcSize)
{
    if (srcSize + ZSTD_blockHeaderSize > dstCapacity) return ERROR(dstSize_tooSmall);
    memcpy((BYTE*)dst + ZSTD_blockHeaderSize, src, srcSize);
    MEM_writeLE24(dst, (U32)(srcSize << 2) + (U32)bt_raw);
    return ZSTD_blockHeaderSize+srcSize;
}


static size_t ZSTD_noCompressLiterals (void* dst, size_t dstCapacity, const void* src, size_t srcSize)
{
    BYTE* const ostart = (BYTE* const)dst;
    U32   const flSize = 1 + (srcSize>31) + (srcSize>4095);

    if (srcSize + flSize > dstCapacity) return ERROR(dstSize_tooSmall);

    switch(flSize)
    {
        case 1: /* 2 - 1 - 5 */
            ostart[0] = (BYTE)((U32)set_basic + (srcSize<<3));
            break;
        case 2: /* 2 - 2 - 12 */
            MEM_writeLE16(ostart, (U16)((U32)set_basic + (1<<2) + (srcSize<<4)));
            break;
        case 3: /* 2 - 2 - 20 */
            MEM_writeLE32(ostart, (U32)((U32)set_basic + (3<<2) + (srcSize<<4)));
            break;
        default:   /* not necessary : flSize is {1,2,3} */
            assert(0);
    }

    memcpy(ostart + flSize, src, srcSize);
    return srcSize + flSize;
}

static size_t ZSTD_compressRleLiteralsBlock (void* dst, size_t dstCapacity, const void* src, size_t srcSize)
{
    BYTE* const ostart = (BYTE* const)dst;
    U32   const flSize = 1 + (srcSize>31) + (srcSize>4095);

    (void)dstCapacity;  /* dstCapacity already guaranteed to be >=4, hence large enough */

    switch(flSize)
    {
        case 1: /* 2 - 1 - 5 */
            ostart[0] = (BYTE)((U32)set_rle + (srcSize<<3));
            break;
        case 2: /* 2 - 2 - 12 */
            MEM_writeLE16(ostart, (U16)((U32)set_rle + (1<<2) + (srcSize<<4)));
            break;
        case 3: /* 2 - 2 - 20 */
            MEM_writeLE32(ostart, (U32)((U32)set_rle + (3<<2) + (srcSize<<4)));
            break;
        default:   /* not necessary : flSize is {1,2,3} */
            assert(0);
    }

    ostart[flSize] = *(const BYTE*)src;
    return flSize+1;
}


static size_t ZSTD_minGain(size_t srcSize) { return (srcSize >> 6) + 2; }

static size_t ZSTD_compressLiterals (ZSTD_entropyCTables_t * entropy,
                                     ZSTD_strategy strategy,
                                     void* dst, size_t dstCapacity,
                               const void* src, size_t srcSize)
{
    size_t const minGain = ZSTD_minGain(srcSize);
    size_t const lhSize = 3 + (srcSize >= 1 KB) + (srcSize >= 16 KB);
    BYTE*  const ostart = (BYTE*)dst;
    U32 singleStream = srcSize < 256;
    symbolEncodingType_e hType = set_compressed;
    size_t cLitSize;


    /* small ? don't even attempt compression (speed opt) */
#   define LITERAL_NOENTROPY 63
    {   size_t const minLitSize = entropy->hufCTable_repeatMode == HUF_repeat_valid ? 6 : LITERAL_NOENTROPY;
        if (srcSize <= minLitSize) return ZSTD_noCompressLiterals(dst, dstCapacity, src, srcSize);
    }

    if (dstCapacity < lhSize+1) return ERROR(dstSize_tooSmall);   /* not enough space for compression */
    {   HUF_repeat repeat = entropy->hufCTable_repeatMode;
        int const preferRepeat = strategy < ZSTD_lazy ? srcSize <= 1024 : 0;
        if (repeat == HUF_repeat_valid && lhSize == 3) singleStream = 1;
        cLitSize = singleStream ? HUF_compress1X_repeat(ostart+lhSize, dstCapacity-lhSize, src, srcSize, 255, 11,
                                      entropy->workspace, sizeof(entropy->workspace), (HUF_CElt*)entropy->hufCTable, &repeat, preferRepeat)
                                : HUF_compress4X_repeat(ostart+lhSize, dstCapacity-lhSize, src, srcSize, 255, 11,
                                      entropy->workspace, sizeof(entropy->workspace), (HUF_CElt*)entropy->hufCTable, &repeat, preferRepeat);
        if (repeat != HUF_repeat_none) { hType = set_repeat; }    /* reused the existing table */
        else { entropy->hufCTable_repeatMode = HUF_repeat_check; }       /* now have a table to reuse */
    }

    if ((cLitSize==0) | (cLitSize >= srcSize - minGain) | ERR_isError(cLitSize)) {
        entropy->hufCTable_repeatMode = HUF_repeat_none;
        return ZSTD_noCompressLiterals(dst, dstCapacity, src, srcSize);
    }
    if (cLitSize==1) {
        entropy->hufCTable_repeatMode = HUF_repeat_none;
        return ZSTD_compressRleLiteralsBlock(dst, dstCapacity, src, srcSize);
    }

    /* Build header */
    switch(lhSize)
    {
    case 3: /* 2 - 2 - 10 - 10 */
        {   U32 const lhc = hType + ((!singleStream) << 2) + ((U32)srcSize<<4) + ((U32)cLitSize<<14);
            MEM_writeLE24(ostart, lhc);
            break;
        }
    case 4: /* 2 - 2 - 14 - 14 */
        {   U32 const lhc = hType + (2 << 2) + ((U32)srcSize<<4) + ((U32)cLitSize<<18);
            MEM_writeLE32(ostart, lhc);
            break;
        }
    case 5: /* 2 - 2 - 18 - 18 */
        {   U32 const lhc = hType + (3 << 2) + ((U32)srcSize<<4) + ((U32)cLitSize<<22);
            MEM_writeLE32(ostart, lhc);
            ostart[4] = (BYTE)(cLitSize >> 10);
            break;
        }
    default:   /* not possible : lhSize is {3,4,5} */
        assert(0);
    }
    return lhSize+cLitSize;
}


void ZSTD_seqToCodes(const seqStore_t* seqStorePtr)
{
    BYTE const LL_deltaCode = 19;
    BYTE const ML_deltaCode = 36;
    const seqDef* const sequences = seqStorePtr->sequencesStart;
    BYTE* const llCodeTable = seqStorePtr->llCode;
    BYTE* const ofCodeTable = seqStorePtr->ofCode;
    BYTE* const mlCodeTable = seqStorePtr->mlCode;
    U32 const nbSeq = (U32)(seqStorePtr->sequences - seqStorePtr->sequencesStart);
    U32 u;
    for (u=0; u<nbSeq; u++) {
        U32 const llv = sequences[u].litLength;
        U32 const mlv = sequences[u].matchLength;
        llCodeTable[u] = (llv> 63) ? (BYTE)ZSTD_highbit32(llv) + LL_deltaCode : LL_Code[llv];
        ofCodeTable[u] = (BYTE)ZSTD_highbit32(sequences[u].offset);
        mlCodeTable[u] = (mlv>127) ? (BYTE)ZSTD_highbit32(mlv) + ML_deltaCode : ML_Code[mlv];
    }
    if (seqStorePtr->longLengthID==1)
        llCodeTable[seqStorePtr->longLengthPos] = MaxLL;
    if (seqStorePtr->longLengthID==2)
        mlCodeTable[seqStorePtr->longLengthPos] = MaxML;
}

typedef enum {
    ZSTD_defaultDisallowed = 0,
    ZSTD_defaultAllowed = 1
} ZSTD_defaultPolicy_e;

MEM_STATIC symbolEncodingType_e ZSTD_selectEncodingType(
        FSE_repeat* repeatMode, size_t const mostFrequent, size_t nbSeq,
        U32 defaultNormLog, ZSTD_defaultPolicy_e const isDefaultAllowed)
{
#define MIN_SEQ_FOR_DYNAMIC_FSE   64
#define MAX_SEQ_FOR_STATIC_FSE  1000
    ZSTD_STATIC_ASSERT(ZSTD_defaultDisallowed == 0 && ZSTD_defaultAllowed != 0);
    if ((mostFrequent == nbSeq) && (!isDefaultAllowed || nbSeq > 2)) {
        /* Prefer set_basic over set_rle when there are 2 or less symbols,
         * since RLE uses 1 byte, but set_basic uses 5-6 bits per symbol.
         * If basic encoding isn't possible, always choose RLE.
         */
        *repeatMode = FSE_repeat_check;
        return set_rle;
    }
    if (isDefaultAllowed && (*repeatMode == FSE_repeat_valid) && (nbSeq < MAX_SEQ_FOR_STATIC_FSE)) {
        return set_repeat;
    }
    if (isDefaultAllowed && ((nbSeq < MIN_SEQ_FOR_DYNAMIC_FSE) || (mostFrequent < (nbSeq >> (defaultNormLog-1))))) {
        *repeatMode = FSE_repeat_valid;
        return set_basic;
    }
    *repeatMode = FSE_repeat_check;
    return set_compressed;
}

MEM_STATIC size_t ZSTD_buildCTable(void* dst, size_t dstCapacity,
        FSE_CTable* CTable, U32 FSELog, symbolEncodingType_e type,
        U32* count, U32 max,
        BYTE const* codeTable, size_t nbSeq,
        S16 const* defaultNorm, U32 defaultNormLog, U32 defaultMax,
        void* workspace, size_t workspaceSize)
{
    BYTE* op = (BYTE*)dst;
    BYTE const* const oend = op + dstCapacity;

    switch (type) {
    case set_rle:
        *op = codeTable[0];
        CHECK_F(FSE_buildCTable_rle(CTable, (BYTE)max));
        return 1;
    case set_repeat:
        return 0;
    case set_basic:
        CHECK_F(FSE_buildCTable_wksp(CTable, defaultNorm, defaultMax, defaultNormLog, workspace, workspaceSize));
        return 0;
    case set_compressed: {
        S16 norm[MaxSeq + 1];
        size_t nbSeq_1 = nbSeq;
        const U32 tableLog = FSE_optimalTableLog(FSELog, nbSeq, max);
        if (count[codeTable[nbSeq-1]] > 1) {
            count[codeTable[nbSeq-1]]--;
            nbSeq_1--;
        }
        assert(nbSeq_1 > 1);
        CHECK_F(FSE_normalizeCount(norm, tableLog, count, nbSeq_1, max));
        {   size_t const NCountSize = FSE_writeNCount(op, oend - op, norm, max, tableLog);   /* overflow protected */
            if (FSE_isError(NCountSize)) return NCountSize;
            CHECK_F(FSE_buildCTable_wksp(CTable, norm, max, tableLog, workspace, workspaceSize));
            return NCountSize;
        }
    }
    default: return assert(0), ERROR(GENERIC);
    }
}

MEM_STATIC size_t ZSTD_encodeSequences(void* dst, size_t dstCapacity,
    FSE_CTable const* CTable_MatchLength, BYTE const* mlCodeTable,
    FSE_CTable const* CTable_OffsetBits, BYTE const* ofCodeTable,
    FSE_CTable const* CTable_LitLength, BYTE const* llCodeTable,
    seqDef const* sequences, size_t nbSeq, int longOffsets)
{
    BIT_CStream_t blockStream;
    FSE_CState_t  stateMatchLength;
    FSE_CState_t  stateOffsetBits;
    FSE_CState_t  stateLitLength;

    CHECK_E(BIT_initCStream(&blockStream, dst, dstCapacity), dstSize_tooSmall); /* not enough space remaining */

    /* first symbols */
    FSE_initCState2(&stateMatchLength, CTable_MatchLength, mlCodeTable[nbSeq-1]);
    FSE_initCState2(&stateOffsetBits,  CTable_OffsetBits,  ofCodeTable[nbSeq-1]);
    FSE_initCState2(&stateLitLength,   CTable_LitLength,   llCodeTable[nbSeq-1]);
    BIT_addBits(&blockStream, sequences[nbSeq-1].litLength, LL_bits[llCodeTable[nbSeq-1]]);
    if (MEM_32bits()) BIT_flushBits(&blockStream);
    BIT_addBits(&blockStream, sequences[nbSeq-1].matchLength, ML_bits[mlCodeTable[nbSeq-1]]);
    if (MEM_32bits()) BIT_flushBits(&blockStream);
    if (longOffsets) {
        U32 const ofBits = ofCodeTable[nbSeq-1];
        int const extraBits = ofBits - MIN(ofBits, STREAM_ACCUMULATOR_MIN-1);
        if (extraBits) {
            BIT_addBits(&blockStream, sequences[nbSeq-1].offset, extraBits);
            BIT_flushBits(&blockStream);
        }
        BIT_addBits(&blockStream, sequences[nbSeq-1].offset >> extraBits,
                    ofBits - extraBits);
    } else {
        BIT_addBits(&blockStream, sequences[nbSeq-1].offset, ofCodeTable[nbSeq-1]);
    }
    BIT_flushBits(&blockStream);

    {   size_t n;
        for (n=nbSeq-2 ; n<nbSeq ; n--) {      /* intentional underflow */
            BYTE const llCode = llCodeTable[n];
            BYTE const ofCode = ofCodeTable[n];
            BYTE const mlCode = mlCodeTable[n];
            U32  const llBits = LL_bits[llCode];
            U32  const ofBits = ofCode;                                     /* 32b*/  /* 64b*/
            U32  const mlBits = ML_bits[mlCode];
                                                                            /* (7)*/  /* (7)*/
            FSE_encodeSymbol(&blockStream, &stateOffsetBits, ofCode);       /* 15 */  /* 15 */
            FSE_encodeSymbol(&blockStream, &stateMatchLength, mlCode);      /* 24 */  /* 24 */
            if (MEM_32bits()) BIT_flushBits(&blockStream);                  /* (7)*/
            FSE_encodeSymbol(&blockStream, &stateLitLength, llCode);        /* 16 */  /* 33 */
            if (MEM_32bits() || (ofBits+mlBits+llBits >= 64-7-(LLFSELog+MLFSELog+OffFSELog)))
                BIT_flushBits(&blockStream);                                /* (7)*/
            BIT_addBits(&blockStream, sequences[n].litLength, llBits);
            if (MEM_32bits() && ((llBits+mlBits)>24)) BIT_flushBits(&blockStream);
            BIT_addBits(&blockStream, sequences[n].matchLength, mlBits);
            if (MEM_32bits() || (ofBits+mlBits+llBits > 56)) BIT_flushBits(&blockStream);
            if (longOffsets) {
                int const extraBits = ofBits - MIN(ofBits, STREAM_ACCUMULATOR_MIN-1);
                if (extraBits) {
                    BIT_addBits(&blockStream, sequences[n].offset, extraBits);
                    BIT_flushBits(&blockStream);                            /* (7)*/
                }
                BIT_addBits(&blockStream, sequences[n].offset >> extraBits,
                            ofBits - extraBits);                            /* 31 */
            } else {
                BIT_addBits(&blockStream, sequences[n].offset, ofBits);     /* 31 */
            }
            BIT_flushBits(&blockStream);                                    /* (7)*/
    }   }

    FSE_flushCState(&blockStream, &stateMatchLength);
    FSE_flushCState(&blockStream, &stateOffsetBits);
    FSE_flushCState(&blockStream, &stateLitLength);

    {   size_t const streamSize = BIT_closeCStream(&blockStream);
        if (streamSize==0) return ERROR(dstSize_tooSmall);   /* not enough space */
        return streamSize;
    }
}

MEM_STATIC size_t ZSTD_compressSequences_internal(seqStore_t* seqStorePtr,
                              ZSTD_entropyCTables_t* entropy,
                              ZSTD_compressionParameters const* cParams,
                              void* dst, size_t dstCapacity)
{
    const int longOffsets = cParams->windowLog > STREAM_ACCUMULATOR_MIN;
    U32 count[MaxSeq+1];
    FSE_CTable* CTable_LitLength = entropy->litlengthCTable;
    FSE_CTable* CTable_OffsetBits = entropy->offcodeCTable;
    FSE_CTable* CTable_MatchLength = entropy->matchlengthCTable;
    U32 LLtype, Offtype, MLtype;   /* compressed, raw or rle */
    const seqDef* const sequences = seqStorePtr->sequencesStart;
    const BYTE* const ofCodeTable = seqStorePtr->ofCode;
    const BYTE* const llCodeTable = seqStorePtr->llCode;
    const BYTE* const mlCodeTable = seqStorePtr->mlCode;
    BYTE* const ostart = (BYTE*)dst;
    BYTE* const oend = ostart + dstCapacity;
    BYTE* op = ostart;
    size_t const nbSeq = seqStorePtr->sequences - seqStorePtr->sequencesStart;
    BYTE* seqHead;

    ZSTD_STATIC_ASSERT(sizeof(entropy->workspace) >= (1<<MAX(MLFSELog,LLFSELog)));

    /* Compress literals */
    {   const BYTE* const literals = seqStorePtr->litStart;
        size_t const litSize = seqStorePtr->lit - literals;
        size_t const cSize = ZSTD_compressLiterals(
                entropy, cParams->strategy, op, dstCapacity, literals, litSize);
        if (ZSTD_isError(cSize))
          return cSize;
        op += cSize;
    }

    /* Sequences Header */
    if ((oend-op) < 3 /*max nbSeq Size*/ + 1 /*seqHead */) return ERROR(dstSize_tooSmall);
    if (nbSeq < 0x7F) *op++ = (BYTE)nbSeq;
    else if (nbSeq < LONGNBSEQ) op[0] = (BYTE)((nbSeq>>8) + 0x80), op[1] = (BYTE)nbSeq, op+=2;
    else op[0]=0xFF, MEM_writeLE16(op+1, (U16)(nbSeq - LONGNBSEQ)), op+=3;
    if (nbSeq==0) return op - ostart;

    /* seqHead : flags for FSE encoding type */
    seqHead = op++;

    /* convert length/distances into codes */
    ZSTD_seqToCodes(seqStorePtr);
    /* CTable for Literal Lengths */
    {   U32 max = MaxLL;
        size_t const mostFrequent = FSE_countFast_wksp(count, &max, llCodeTable, nbSeq, entropy->workspace);
        LLtype = ZSTD_selectEncodingType(&entropy->litlength_repeatMode, mostFrequent, nbSeq, LL_defaultNormLog, ZSTD_defaultAllowed);
        {   size_t const countSize = ZSTD_buildCTable(op, oend - op, CTable_LitLength, LLFSELog, (symbolEncodingType_e)LLtype,
                    count, max, llCodeTable, nbSeq, LL_defaultNorm, LL_defaultNormLog, MaxLL,
                    entropy->workspace, sizeof(entropy->workspace));
            if (ZSTD_isError(countSize)) return countSize;
            op += countSize;
    }   }
    /* CTable for Offsets */
    {   U32 max = MaxOff;
        size_t const mostFrequent = FSE_countFast_wksp(count, &max, ofCodeTable, nbSeq, entropy->workspace);
        /* We can only use the basic table if max <= DefaultMaxOff, otherwise the offsets are too large */
        ZSTD_defaultPolicy_e const defaultPolicy = max <= DefaultMaxOff ? ZSTD_defaultAllowed : ZSTD_defaultDisallowed;
        Offtype = ZSTD_selectEncodingType(&entropy->offcode_repeatMode, mostFrequent, nbSeq, OF_defaultNormLog, defaultPolicy);
        {   size_t const countSize = ZSTD_buildCTable(op, oend - op, CTable_OffsetBits, OffFSELog, (symbolEncodingType_e)Offtype,
                    count, max, ofCodeTable, nbSeq, OF_defaultNorm, OF_defaultNormLog, DefaultMaxOff,
                    entropy->workspace, sizeof(entropy->workspace));
            if (ZSTD_isError(countSize)) return countSize;
            op += countSize;
    }   }
    /* CTable for MatchLengths */
    {   U32 max = MaxML;
        size_t const mostFrequent = FSE_countFast_wksp(count, &max, mlCodeTable, nbSeq, entropy->workspace);
        MLtype = ZSTD_selectEncodingType(&entropy->matchlength_repeatMode, mostFrequent, nbSeq, ML_defaultNormLog, ZSTD_defaultAllowed);
        {   size_t const countSize = ZSTD_buildCTable(op, oend - op, CTable_MatchLength, MLFSELog, (symbolEncodingType_e)MLtype,
                    count, max, mlCodeTable, nbSeq, ML_defaultNorm, ML_defaultNormLog, MaxML,
                    entropy->workspace, sizeof(entropy->workspace));
            if (ZSTD_isError(countSize)) return countSize;
            op += countSize;
    }   }

    *seqHead = (BYTE)((LLtype<<6) + (Offtype<<4) + (MLtype<<2));

    {   size_t const streamSize = ZSTD_encodeSequences(op, oend - op,
                CTable_MatchLength, mlCodeTable,
                CTable_OffsetBits, ofCodeTable,
                CTable_LitLength, llCodeTable,
                sequences, nbSeq, longOffsets);
        if (ZSTD_isError(streamSize)) return streamSize;
        op += streamSize;
    }

    return op - ostart;
}

MEM_STATIC size_t ZSTD_compressSequences(seqStore_t* seqStorePtr,
                              ZSTD_entropyCTables_t* entropy,
                              ZSTD_compressionParameters const* cParams,
                              void* dst, size_t dstCapacity,
                              size_t srcSize)
{
    size_t const cSize = ZSTD_compressSequences_internal(seqStorePtr, entropy, cParams,
                                                         dst, dstCapacity);
    size_t const minGain = ZSTD_minGain(srcSize);
    size_t const maxCSize = srcSize - minGain;
    /* If the srcSize <= dstCapacity, then there is enough space to write a
     * raw uncompressed block. Since we ran out of space, the block must not
     * be compressible, so fall back to a raw uncompressed block.
     */
    int const uncompressibleError = cSize == ERROR(dstSize_tooSmall) && srcSize <= dstCapacity;

    if (ZSTD_isError(cSize) && !uncompressibleError)
        return cSize;
    /* Check compressibility */
    if (cSize >= maxCSize || uncompressibleError) {
        entropy->hufCTable_repeatMode = HUF_repeat_none;
        entropy->offcode_repeatMode = FSE_repeat_none;
        entropy->matchlength_repeatMode = FSE_repeat_none;
        entropy->litlength_repeatMode = FSE_repeat_none;
        return 0;
    }
    assert(!ZSTD_isError(cSize));

    /* confirm repcodes */
    { int i; for (i=0; i<ZSTD_REP_NUM; i++) seqStorePtr->rep[i] = seqStorePtr->repToConfirm[i]; }
    return cSize;
}

/* ZSTD_selectBlockCompressor() :
 * Not static, but internal use only (used by long distance matcher)
 * assumption : strat is a valid strategy */
typedef size_t (*ZSTD_blockCompressor) (ZSTD_CCtx* ctx, const void* src, size_t srcSize);
ZSTD_blockCompressor ZSTD_selectBlockCompressor(ZSTD_strategy strat, int extDict)
{
    static const ZSTD_blockCompressor blockCompressor[2][(unsigned)ZSTD_btultra+1] = {
        { ZSTD_compressBlock_fast  /* default for 0 */,
          ZSTD_compressBlock_fast, ZSTD_compressBlock_doubleFast, ZSTD_compressBlock_greedy,
          ZSTD_compressBlock_lazy, ZSTD_compressBlock_lazy2, ZSTD_compressBlock_btlazy2,
          ZSTD_compressBlock_btopt, ZSTD_compressBlock_btultra },
        { ZSTD_compressBlock_fast_extDict  /* default for 0 */,
          ZSTD_compressBlock_fast_extDict, ZSTD_compressBlock_doubleFast_extDict, ZSTD_compressBlock_greedy_extDict,
          ZSTD_compressBlock_lazy_extDict,ZSTD_compressBlock_lazy2_extDict, ZSTD_compressBlock_btlazy2_extDict,
          ZSTD_compressBlock_btopt_extDict, ZSTD_compressBlock_btultra_extDict }
    };
    ZSTD_STATIC_ASSERT((unsigned)ZSTD_fast == 1);
    assert((U32)strat >= (U32)ZSTD_fast);
    assert((U32)strat <= (U32)ZSTD_btultra);

    return blockCompressor[extDict!=0][(U32)strat];
}

static void ZSTD_storeLastLiterals(seqStore_t* seqStorePtr,
                                   const BYTE* anchor, size_t lastLLSize)
{
    memcpy(seqStorePtr->lit, anchor, lastLLSize);
    seqStorePtr->lit += lastLLSize;
}

static size_t ZSTD_compressBlock_internal(ZSTD_CCtx* zc, void* dst, size_t dstCapacity, const void* src, size_t srcSize)
{
    const BYTE* const base = zc->base;
    const BYTE* const istart = (const BYTE*)src;
    const U32 current = (U32)(istart-base);
    size_t lastLLSize;
    const BYTE* anchor;
    U32 const extDict = zc->lowLimit < zc->dictLimit;
    const ZSTD_blockCompressor blockCompressor =
        zc->appliedParams.ldmParams.enableLdm
            ? (extDict ? ZSTD_compressBlock_ldm_extDict : ZSTD_compressBlock_ldm)
            : ZSTD_selectBlockCompressor(zc->appliedParams.cParams.strategy, extDict);

    if (srcSize < MIN_CBLOCK_SIZE+ZSTD_blockHeaderSize+1) return 0;   /* don't even attempt compression below a certain srcSize */
    ZSTD_resetSeqStore(&(zc->seqStore));
    if (current > zc->nextToUpdate + 384)
        zc->nextToUpdate = current - MIN(192, (U32)(current - zc->nextToUpdate - 384));   /* limited update after finding a very long match */

    lastLLSize = blockCompressor(zc, src, srcSize);

    /* Last literals */
    anchor = (const BYTE*)src + srcSize - lastLLSize;
    ZSTD_storeLastLiterals(&zc->seqStore, anchor, lastLLSize);

    return ZSTD_compressSequences(&zc->seqStore, zc->entropy, &zc->appliedParams.cParams, dst, dstCapacity, srcSize);
}


/*! ZSTD_compress_frameChunk() :
*   Compress a chunk of data into one or multiple blocks.
*   All blocks will be terminated, all input will be consumed.
*   Function will issue an error if there is not enough `dstCapacity` to hold the compressed content.
*   Frame is supposed already started (header already produced)
*   @return : compressed size, or an error code
*/
static size_t ZSTD_compress_frameChunk (ZSTD_CCtx* cctx,
                                     void* dst, size_t dstCapacity,
                               const void* src, size_t srcSize,
                                     U32 lastFrameChunk)
{
    size_t blockSize = cctx->blockSize;
    size_t remaining = srcSize;
    const BYTE* ip = (const BYTE*)src;
    BYTE* const ostart = (BYTE*)dst;
    BYTE* op = ostart;
    U32 const maxDist = (U32)1 << cctx->appliedParams.cParams.windowLog;

    if (cctx->appliedParams.fParams.checksumFlag && srcSize)
        XXH64_update(&cctx->xxhState, src, srcSize);

    while (remaining) {
        U32 const lastBlock = lastFrameChunk & (blockSize >= remaining);
        size_t cSize;

        if (dstCapacity < ZSTD_blockHeaderSize + MIN_CBLOCK_SIZE)
            return ERROR(dstSize_tooSmall);   /* not enough space to store compressed block */
        if (remaining < blockSize) blockSize = remaining;

        /* preemptive overflow correction:
         * 1. correction is large enough:
         *    lowLimit > (3<<29) ==> current > 3<<29 + 1<<windowLog - blockSize
         *    1<<windowLog <= newCurrent < 1<<chainLog + 1<<windowLog
         *
         *    current - newCurrent
         *    > (3<<29 + 1<<windowLog - blockSize) - (1<<windowLog + 1<<chainLog)
         *    > (3<<29 - blockSize) - (1<<chainLog)
         *    > (3<<29 - blockSize) - (1<<30)             (NOTE: chainLog <= 30)
         *    > 1<<29 - 1<<17
         *
         * 2. (ip+blockSize - cctx->base) doesn't overflow:
         *    In 32 bit mode we limit windowLog to 30 so we don't get
         *    differences larger than 1<<31-1.
         * 3. cctx->lowLimit < 1<<32:
         *    windowLog <= 31 ==> 3<<29 + 1<<windowLog < 7<<29 < 1<<32.
         */
        if (cctx->lowLimit > (3U<<29)) {
            U32 const cycleMask = ((U32)1 << ZSTD_cycleLog(cctx->appliedParams.cParams.chainLog, cctx->appliedParams.cParams.strategy)) - 1;
            U32 const current = (U32)(ip - cctx->base);
            U32 const newCurrent = (current & cycleMask) + ((U32)1 << cctx->appliedParams.cParams.windowLog);
            U32 const correction = current - newCurrent;
            ZSTD_STATIC_ASSERT(ZSTD_CHAINLOG_MAX <= 30);
            ZSTD_STATIC_ASSERT(ZSTD_WINDOWLOG_MAX_32 <= 30);
            ZSTD_STATIC_ASSERT(ZSTD_WINDOWLOG_MAX <= 31);
            assert(current > newCurrent);
            assert(correction > 1<<28); /* Loose bound, should be about 1<<29 */
            ZSTD_reduceIndex(cctx, correction);
            cctx->base += correction;
            cctx->dictBase += correction;
            cctx->lowLimit -= correction;
            cctx->dictLimit -= correction;
            if (cctx->nextToUpdate < correction) cctx->nextToUpdate = 0;
            else cctx->nextToUpdate -= correction;
            DEBUGLOG(4, "Correction of 0x%x bytes to lowLimit=0x%x\n", correction, cctx->lowLimit);
        }

        if ((U32)(ip+blockSize - cctx->base) > cctx->loadedDictEnd + maxDist) {
            /* enforce maxDist */
            U32 const newLowLimit = (U32)(ip+blockSize - cctx->base) - maxDist;
            if (cctx->lowLimit < newLowLimit) cctx->lowLimit = newLowLimit;
            if (cctx->dictLimit < cctx->lowLimit) cctx->dictLimit = cctx->lowLimit;
        }

        cSize = ZSTD_compressBlock_internal(cctx, op+ZSTD_blockHeaderSize, dstCapacity-ZSTD_blockHeaderSize, ip, blockSize);
        if (ZSTD_isError(cSize)) return cSize;

        if (cSize == 0) {  /* block is not compressible */
            U32 const cBlockHeader24 = lastBlock + (((U32)bt_raw)<<1) + (U32)(blockSize << 3);
            if (blockSize + ZSTD_blockHeaderSize > dstCapacity) return ERROR(dstSize_tooSmall);
            MEM_writeLE32(op, cBlockHeader24);   /* no pb, 4th byte will be overwritten */
            memcpy(op + ZSTD_blockHeaderSize, ip, blockSize);
            cSize = ZSTD_blockHeaderSize+blockSize;
        } else {
            U32 const cBlockHeader24 = lastBlock + (((U32)bt_compressed)<<1) + (U32)(cSize << 3);
            MEM_writeLE24(op, cBlockHeader24);
            cSize += ZSTD_blockHeaderSize;
        }

        remaining -= blockSize;
        dstCapacity -= cSize;
        ip += blockSize;
        op += cSize;
    }

    if (lastFrameChunk && (op>ostart)) cctx->stage = ZSTDcs_ending;
    return op-ostart;
}


static size_t ZSTD_writeFrameHeader(void* dst, size_t dstCapacity,
                                    ZSTD_CCtx_params params, U64 pledgedSrcSize, U32 dictID)
{   BYTE* const op = (BYTE*)dst;
    U32   const dictIDSizeCodeLength = (dictID>0) + (dictID>=256) + (dictID>=65536);   /* 0-3 */
    U32   const dictIDSizeCode = params.fParams.noDictIDFlag ? 0 : dictIDSizeCodeLength;   /* 0-3 */
    U32   const checksumFlag = params.fParams.checksumFlag>0;
    U32   const windowSize = (U32)1 << params.cParams.windowLog;
    U32   const singleSegment = params.fParams.contentSizeFlag && (windowSize >= pledgedSrcSize);
    BYTE  const windowLogByte = (BYTE)((params.cParams.windowLog - ZSTD_WINDOWLOG_ABSOLUTEMIN) << 3);
    U32   const fcsCode = params.fParams.contentSizeFlag ?
                     (pledgedSrcSize>=256) + (pledgedSrcSize>=65536+256) + (pledgedSrcSize>=0xFFFFFFFFU) : 0;  /* 0-3 */
    BYTE  const frameHeaderDecriptionByte = (BYTE)(dictIDSizeCode + (checksumFlag<<2) + (singleSegment<<5) + (fcsCode<<6) );
    size_t pos=0;

    if (dstCapacity < ZSTD_frameHeaderSize_max) return ERROR(dstSize_tooSmall);
    DEBUGLOG(4, "ZSTD_writeFrameHeader : dictIDFlag : %u ; dictID : %u ; dictIDSizeCode : %u",
                !params.fParams.noDictIDFlag, dictID,  dictIDSizeCode);

    if (params.format == ZSTD_f_zstd1) {
        DEBUGLOG(4, "writing zstd magic number");
        MEM_writeLE32(dst, ZSTD_MAGICNUMBER);
        pos = 4;
    }
    op[pos++] = frameHeaderDecriptionByte;
    if (!singleSegment) op[pos++] = windowLogByte;
    switch(dictIDSizeCode)
    {
        default:  assert(0); /* impossible */
        case 0 : break;
        case 1 : op[pos] = (BYTE)(dictID); pos++; break;
        case 2 : MEM_writeLE16(op+pos, (U16)dictID); pos+=2; break;
        case 3 : MEM_writeLE32(op+pos, dictID); pos+=4; break;
    }
    switch(fcsCode)
    {
        default:  assert(0); /* impossible */
        case 0 : if (singleSegment) op[pos++] = (BYTE)(pledgedSrcSize); break;
        case 1 : MEM_writeLE16(op+pos, (U16)(pledgedSrcSize-256)); pos+=2; break;
        case 2 : MEM_writeLE32(op+pos, (U32)(pledgedSrcSize)); pos+=4; break;
        case 3 : MEM_writeLE64(op+pos, (U64)(pledgedSrcSize)); pos+=8; break;
    }
    return pos;
}


static size_t ZSTD_compressContinue_internal (ZSTD_CCtx* cctx,
                              void* dst, size_t dstCapacity,
                        const void* src, size_t srcSize,
                               U32 frame, U32 lastFrameChunk)
{
    const BYTE* const ip = (const BYTE*) src;
    size_t fhSize = 0;

    DEBUGLOG(5, "ZSTD_compressContinue_internal");
    DEBUGLOG(5, "stage: %u", cctx->stage);
    if (cctx->stage==ZSTDcs_created) return ERROR(stage_wrong);   /* missing init (ZSTD_compressBegin) */

    if (frame && (cctx->stage==ZSTDcs_init)) {
        fhSize = ZSTD_writeFrameHeader(dst, dstCapacity, cctx->appliedParams,
                                       cctx->pledgedSrcSizePlusOne-1, cctx->dictID);
        if (ZSTD_isError(fhSize)) return fhSize;
        dstCapacity -= fhSize;
        dst = (char*)dst + fhSize;
        cctx->stage = ZSTDcs_ongoing;
    }

    /* Check if blocks follow each other */
    if (src != cctx->nextSrc) {
        /* not contiguous */
        ptrdiff_t const delta = cctx->nextSrc - ip;
        cctx->lowLimit = cctx->dictLimit;
        cctx->dictLimit = (U32)(cctx->nextSrc - cctx->base);
        cctx->dictBase = cctx->base;
        cctx->base -= delta;
        cctx->nextToUpdate = cctx->dictLimit;
        if (cctx->dictLimit - cctx->lowLimit < HASH_READ_SIZE) cctx->lowLimit = cctx->dictLimit;   /* too small extDict */
    }

    /* if input and dictionary overlap : reduce dictionary (area presumed modified by input) */
    if ((ip+srcSize > cctx->dictBase + cctx->lowLimit) & (ip < cctx->dictBase + cctx->dictLimit)) {
        ptrdiff_t const highInputIdx = (ip + srcSize) - cctx->dictBase;
        U32 const lowLimitMax = (highInputIdx > (ptrdiff_t)cctx->dictLimit) ? cctx->dictLimit : (U32)highInputIdx;
        cctx->lowLimit = lowLimitMax;
    }

    cctx->nextSrc = ip + srcSize;

    if (srcSize) {
        size_t const cSize = frame ?
                             ZSTD_compress_frameChunk (cctx, dst, dstCapacity, src, srcSize, lastFrameChunk) :
                             ZSTD_compressBlock_internal (cctx, dst, dstCapacity, src, srcSize);
        if (ZSTD_isError(cSize)) return cSize;
        cctx->consumedSrcSize += srcSize;
        return cSize + fhSize;
    } else
        return fhSize;
}

size_t ZSTD_compressContinue (ZSTD_CCtx* cctx,
                              void* dst, size_t dstCapacity,
                        const void* src, size_t srcSize)
{
    return ZSTD_compressContinue_internal(cctx, dst, dstCapacity, src, srcSize, 1 /* frame mode */, 0 /* last chunk */);
}


size_t ZSTD_getBlockSize(const ZSTD_CCtx* cctx)
{
    ZSTD_compressionParameters const cParams =
            ZSTD_getCParamsFromCCtxParams(cctx->appliedParams, 0, 0);
    return MIN (ZSTD_BLOCKSIZE_MAX, (U32)1 << cParams.windowLog);
}

size_t ZSTD_compressBlock(ZSTD_CCtx* cctx, void* dst, size_t dstCapacity, const void* src, size_t srcSize)
{
    size_t const blockSizeMax = ZSTD_getBlockSize(cctx);
    if (srcSize > blockSizeMax) return ERROR(srcSize_wrong);
    return ZSTD_compressContinue_internal(cctx, dst, dstCapacity, src, srcSize, 0 /* frame mode */, 0 /* last chunk */);
}

/*! ZSTD_loadDictionaryContent() :
 *  @return : 0, or an error code
 */
static size_t ZSTD_loadDictionaryContent(ZSTD_CCtx* zc, const void* src, size_t srcSize)
{
    const BYTE* const ip = (const BYTE*) src;
    const BYTE* const iend = ip + srcSize;

    /* input becomes current prefix */
    zc->lowLimit = zc->dictLimit;
    zc->dictLimit = (U32)(zc->nextSrc - zc->base);
    zc->dictBase = zc->base;
    zc->base += ip - zc->nextSrc;
    zc->nextToUpdate = zc->dictLimit;
    zc->loadedDictEnd = zc->appliedParams.forceWindow ? 0 : (U32)(iend - zc->base);

    zc->nextSrc = iend;
    if (srcSize <= HASH_READ_SIZE) return 0;

    switch(zc->appliedParams.cParams.strategy)
    {
    case ZSTD_fast:
        ZSTD_fillHashTable (zc, iend, zc->appliedParams.cParams.searchLength);
        break;
    case ZSTD_dfast:
        ZSTD_fillDoubleHashTable (zc, iend, zc->appliedParams.cParams.searchLength);
        break;

    case ZSTD_greedy:
    case ZSTD_lazy:
    case ZSTD_lazy2:
        if (srcSize >= HASH_READ_SIZE)
            ZSTD_insertAndFindFirstIndex(zc, iend-HASH_READ_SIZE, zc->appliedParams.cParams.searchLength);
        break;

    case ZSTD_btlazy2:
    case ZSTD_btopt:
    case ZSTD_btultra:
        if (srcSize >= HASH_READ_SIZE)
            ZSTD_updateTree(zc, iend-HASH_READ_SIZE, iend, (U32)1 << zc->appliedParams.cParams.searchLog, zc->appliedParams.cParams.searchLength);
        break;

    default:
        assert(0);  /* not possible : not a valid strategy id */
    }

    zc->nextToUpdate = (U32)(iend - zc->base);
    return 0;
}


/* Dictionaries that assign zero probability to symbols that show up causes problems
   when FSE encoding.  Refuse dictionaries that assign zero probability to symbols
   that we may encounter during compression.
   NOTE: This behavior is not standard and could be improved in the future. */
static size_t ZSTD_checkDictNCount(short* normalizedCounter, unsigned dictMaxSymbolValue, unsigned maxSymbolValue) {
    U32 s;
    if (dictMaxSymbolValue < maxSymbolValue) return ERROR(dictionary_corrupted);
    for (s = 0; s <= maxSymbolValue; ++s) {
        if (normalizedCounter[s] == 0) return ERROR(dictionary_corrupted);
    }
    return 0;
}


/* Dictionary format :
 * See :
 * https://github.com/facebook/zstd/blob/master/doc/zstd_compression_format.md#dictionary-format
 */
/*! ZSTD_loadZstdDictionary() :
 * @return : 0, or an error code
 *  assumptions : magic number supposed already checked
 *                dictSize supposed > 8
 */
static size_t ZSTD_loadZstdDictionary(ZSTD_CCtx* cctx, const void* dict, size_t dictSize)
{
    const BYTE* dictPtr = (const BYTE*)dict;
    const BYTE* const dictEnd = dictPtr + dictSize;
    short offcodeNCount[MaxOff+1];
    unsigned offcodeMaxValue = MaxOff;

    ZSTD_STATIC_ASSERT(sizeof(cctx->entropy->workspace) >= (1<<MAX(MLFSELog,LLFSELog)));

    dictPtr += 4;   /* skip magic number */
    cctx->dictID = cctx->appliedParams.fParams.noDictIDFlag ? 0 :  MEM_readLE32(dictPtr);
    dictPtr += 4;

    {   unsigned maxSymbolValue = 255;
        size_t const hufHeaderSize = HUF_readCTable((HUF_CElt*)cctx->entropy->hufCTable, &maxSymbolValue, dictPtr, dictEnd-dictPtr);
        if (HUF_isError(hufHeaderSize)) return ERROR(dictionary_corrupted);
        if (maxSymbolValue < 255) return ERROR(dictionary_corrupted);
        dictPtr += hufHeaderSize;
    }

    {   unsigned offcodeLog;
        size_t const offcodeHeaderSize = FSE_readNCount(offcodeNCount, &offcodeMaxValue, &offcodeLog, dictPtr, dictEnd-dictPtr);
        if (FSE_isError(offcodeHeaderSize)) return ERROR(dictionary_corrupted);
        if (offcodeLog > OffFSELog) return ERROR(dictionary_corrupted);
        /* Defer checking offcodeMaxValue because we need to know the size of the dictionary content */
        CHECK_E( FSE_buildCTable_wksp(cctx->entropy->offcodeCTable, offcodeNCount, offcodeMaxValue, offcodeLog, cctx->entropy->workspace, sizeof(cctx->entropy->workspace)),
                 dictionary_corrupted);
        dictPtr += offcodeHeaderSize;
    }

    {   short matchlengthNCount[MaxML+1];
        unsigned matchlengthMaxValue = MaxML, matchlengthLog;
        size_t const matchlengthHeaderSize = FSE_readNCount(matchlengthNCount, &matchlengthMaxValue, &matchlengthLog, dictPtr, dictEnd-dictPtr);
        if (FSE_isError(matchlengthHeaderSize)) return ERROR(dictionary_corrupted);
        if (matchlengthLog > MLFSELog) return ERROR(dictionary_corrupted);
        /* Every match length code must have non-zero probability */
        CHECK_F( ZSTD_checkDictNCount(matchlengthNCount, matchlengthMaxValue, MaxML));
        CHECK_E( FSE_buildCTable_wksp(cctx->entropy->matchlengthCTable, matchlengthNCount, matchlengthMaxValue, matchlengthLog, cctx->entropy->workspace, sizeof(cctx->entropy->workspace)),
                 dictionary_corrupted);
        dictPtr += matchlengthHeaderSize;
    }

    {   short litlengthNCount[MaxLL+1];
        unsigned litlengthMaxValue = MaxLL, litlengthLog;
        size_t const litlengthHeaderSize = FSE_readNCount(litlengthNCount, &litlengthMaxValue, &litlengthLog, dictPtr, dictEnd-dictPtr);
        if (FSE_isError(litlengthHeaderSize)) return ERROR(dictionary_corrupted);
        if (litlengthLog > LLFSELog) return ERROR(dictionary_corrupted);
        /* Every literal length code must have non-zero probability */
        CHECK_F( ZSTD_checkDictNCount(litlengthNCount, litlengthMaxValue, MaxLL));
        CHECK_E( FSE_buildCTable_wksp(cctx->entropy->litlengthCTable, litlengthNCount, litlengthMaxValue, litlengthLog, cctx->entropy->workspace, sizeof(cctx->entropy->workspace)),
                 dictionary_corrupted);
        dictPtr += litlengthHeaderSize;
    }

    if (dictPtr+12 > dictEnd) return ERROR(dictionary_corrupted);
    cctx->seqStore.rep[0] = MEM_readLE32(dictPtr+0);
    cctx->seqStore.rep[1] = MEM_readLE32(dictPtr+4);
    cctx->seqStore.rep[2] = MEM_readLE32(dictPtr+8);
    dictPtr += 12;

    {   size_t const dictContentSize = (size_t)(dictEnd - dictPtr);
        U32 offcodeMax = MaxOff;
        if (dictContentSize <= ((U32)-1) - 128 KB) {
            U32 const maxOffset = (U32)dictContentSize + 128 KB; /* The maximum offset that must be supported */
            offcodeMax = ZSTD_highbit32(maxOffset); /* Calculate minimum offset code required to represent maxOffset */
        }
        /* All offset values <= dictContentSize + 128 KB must be representable */
        CHECK_F (ZSTD_checkDictNCount(offcodeNCount, offcodeMaxValue, MIN(offcodeMax, MaxOff)));
        /* All repCodes must be <= dictContentSize and != 0*/
        {   U32 u;
            for (u=0; u<3; u++) {
                if (cctx->seqStore.rep[u] == 0) return ERROR(dictionary_corrupted);
                if (cctx->seqStore.rep[u] > dictContentSize) return ERROR(dictionary_corrupted);
        }   }

        cctx->entropy->hufCTable_repeatMode = HUF_repeat_valid;
        cctx->entropy->offcode_repeatMode = FSE_repeat_valid;
        cctx->entropy->matchlength_repeatMode = FSE_repeat_valid;
        cctx->entropy->litlength_repeatMode = FSE_repeat_valid;
        return ZSTD_loadDictionaryContent(cctx, dictPtr, dictContentSize);
    }
}

/** ZSTD_compress_insertDictionary() :
*   @return : 0, or an error code */
static size_t ZSTD_compress_insertDictionary(ZSTD_CCtx* cctx,
                                       const void* dict, size_t dictSize,
                                             ZSTD_dictMode_e dictMode)
{
    DEBUGLOG(5, "ZSTD_compress_insertDictionary");
    if ((dict==NULL) || (dictSize<=8)) return 0;

    /* dict restricted modes */
    if (dictMode==ZSTD_dm_rawContent)
        return ZSTD_loadDictionaryContent(cctx, dict, dictSize);

    if (MEM_readLE32(dict) != ZSTD_MAGIC_DICTIONARY) {
        if (dictMode == ZSTD_dm_auto) {
            DEBUGLOG(5, "raw content dictionary detected");
            return ZSTD_loadDictionaryContent(cctx, dict, dictSize);
        }
        if (dictMode == ZSTD_dm_fullDict)
            return ERROR(dictionary_wrong);
        assert(0);   /* impossible */
    }

    /* dict as full zstd dictionary */
    return ZSTD_loadZstdDictionary(cctx, dict, dictSize);
}

/*! ZSTD_compressBegin_internal() :
 * @return : 0, or an error code */
static size_t ZSTD_compressBegin_internal(ZSTD_CCtx* cctx,
                             const void* dict, size_t dictSize,
                             ZSTD_dictMode_e dictMode,
                             const ZSTD_CDict* cdict,
                                   ZSTD_CCtx_params params, U64 pledgedSrcSize,
                                   ZSTD_buffered_policy_e zbuff)
{
    DEBUGLOG(4, "ZSTD_compressBegin_internal");
    /* params are supposed to be fully validated at this point */
    assert(!ZSTD_isError(ZSTD_checkCParams(params.cParams)));
    assert(!((dict) && (cdict)));  /* either dict or cdict, not both */

    if (cdict && cdict->dictContentSize>0) {
        return ZSTD_copyCCtx_internal(cctx, cdict->refContext,
                                      params.fParams, pledgedSrcSize,
                                      zbuff);
    }

    CHECK_F( ZSTD_resetCCtx_internal(cctx, params, pledgedSrcSize,
                                     ZSTDcrp_continue, zbuff) );
    return ZSTD_compress_insertDictionary(cctx, dict, dictSize, dictMode);
}

size_t ZSTD_compressBegin_advanced_internal(
                                    ZSTD_CCtx* cctx,
                                    const void* dict, size_t dictSize,
                                    ZSTD_dictMode_e dictMode,
                                    ZSTD_CCtx_params params,
                                    unsigned long long pledgedSrcSize)
{
    /* compression parameters verification and optimization */
    CHECK_F( ZSTD_checkCParams(params.cParams) );
    return ZSTD_compressBegin_internal(cctx, dict, dictSize, dictMode, NULL,
                                       params, pledgedSrcSize,
                                       ZSTDb_not_buffered);
}

/*! ZSTD_compressBegin_advanced() :
*   @return : 0, or an error code */
size_t ZSTD_compressBegin_advanced(ZSTD_CCtx* cctx,
                             const void* dict, size_t dictSize,
                                   ZSTD_parameters params, unsigned long long pledgedSrcSize)
{
    ZSTD_CCtx_params const cctxParams =
            ZSTD_assignParamsToCCtxParams(cctx->requestedParams, params);
    return ZSTD_compressBegin_advanced_internal(cctx, dict, dictSize, ZSTD_dm_auto,
                                                cctxParams,
                                                pledgedSrcSize);
}

size_t ZSTD_compressBegin_usingDict(ZSTD_CCtx* cctx, const void* dict, size_t dictSize, int compressionLevel)
{
    ZSTD_parameters const params = ZSTD_getParams(compressionLevel, 0, dictSize);
    ZSTD_CCtx_params const cctxParams =
            ZSTD_assignParamsToCCtxParams(cctx->requestedParams, params);
    return ZSTD_compressBegin_internal(cctx, dict, dictSize, ZSTD_dm_auto, NULL,
                                       cctxParams, 0, ZSTDb_not_buffered);
}

size_t ZSTD_compressBegin(ZSTD_CCtx* cctx, int compressionLevel)
{
    return ZSTD_compressBegin_usingDict(cctx, NULL, 0, compressionLevel);
}


/*! ZSTD_writeEpilogue() :
*   Ends a frame.
*   @return : nb of bytes written into dst (or an error code) */
static size_t ZSTD_writeEpilogue(ZSTD_CCtx* cctx, void* dst, size_t dstCapacity)
{
    BYTE* const ostart = (BYTE*)dst;
    BYTE* op = ostart;
    size_t fhSize = 0;

    DEBUGLOG(5, "ZSTD_writeEpilogue");
    if (cctx->stage == ZSTDcs_created) return ERROR(stage_wrong);  /* init missing */

    /* special case : empty frame */
    if (cctx->stage == ZSTDcs_init) {
        fhSize = ZSTD_writeFrameHeader(dst, dstCapacity, cctx->appliedParams, 0, 0);
        if (ZSTD_isError(fhSize)) return fhSize;
        dstCapacity -= fhSize;
        op += fhSize;
        cctx->stage = ZSTDcs_ongoing;
    }

    if (cctx->stage != ZSTDcs_ending) {
        /* write one last empty block, make it the "last" block */
        U32 const cBlockHeader24 = 1 /* last block */ + (((U32)bt_raw)<<1) + 0;
        if (dstCapacity<4) return ERROR(dstSize_tooSmall);
        MEM_writeLE32(op, cBlockHeader24);
        op += ZSTD_blockHeaderSize;
        dstCapacity -= ZSTD_blockHeaderSize;
    }

    if (cctx->appliedParams.fParams.checksumFlag) {
        U32 const checksum = (U32) XXH64_digest(&cctx->xxhState);
        if (dstCapacity<4) return ERROR(dstSize_tooSmall);
        MEM_writeLE32(op, checksum);
        op += 4;
    }

    cctx->stage = ZSTDcs_created;  /* return to "created but no init" status */
    return op-ostart;
}


size_t ZSTD_compressEnd (ZSTD_CCtx* cctx,
                         void* dst, size_t dstCapacity,
                   const void* src, size_t srcSize)
{
    size_t endResult;
    size_t const cSize = ZSTD_compressContinue_internal(cctx,
                                dst, dstCapacity, src, srcSize,
                                1 /* frame mode */, 1 /* last chunk */);
    if (ZSTD_isError(cSize)) return cSize;
    endResult = ZSTD_writeEpilogue(cctx, (char*)dst + cSize, dstCapacity-cSize);
    if (ZSTD_isError(endResult)) return endResult;
    if (cctx->appliedParams.fParams.contentSizeFlag) {  /* control src size */
        DEBUGLOG(4, "end of frame : controlling src size");
        if (cctx->pledgedSrcSizePlusOne != cctx->consumedSrcSize+1) {
            DEBUGLOG(4, "error : pledgedSrcSize = %u, while realSrcSize = %u",
                (U32)cctx->pledgedSrcSizePlusOne-1, (U32)cctx->consumedSrcSize);
            return ERROR(srcSize_wrong);
    }   }
    return cSize + endResult;
}


static size_t ZSTD_compress_internal (ZSTD_CCtx* cctx,
                               void* dst, size_t dstCapacity,
                         const void* src, size_t srcSize,
                         const void* dict,size_t dictSize,
                               ZSTD_parameters params)
{
    ZSTD_CCtx_params const cctxParams =
            ZSTD_assignParamsToCCtxParams(cctx->requestedParams, params);
    return ZSTD_compress_advanced_internal(cctx,
                                          dst, dstCapacity,
                                          src, srcSize,
                                          dict, dictSize,
                                          cctxParams);
}

size_t ZSTD_compress_advanced (ZSTD_CCtx* ctx,
                               void* dst, size_t dstCapacity,
                         const void* src, size_t srcSize,
                         const void* dict,size_t dictSize,
                               ZSTD_parameters params)
{
    CHECK_F(ZSTD_checkCParams(params.cParams));
    return ZSTD_compress_internal(ctx, dst, dstCapacity, src, srcSize, dict, dictSize, params);
}

/* Internal */
size_t ZSTD_compress_advanced_internal(
        ZSTD_CCtx* cctx,
        void* dst, size_t dstCapacity,
        const void* src, size_t srcSize,
        const void* dict,size_t dictSize,
        ZSTD_CCtx_params params)
{
    CHECK_F( ZSTD_compressBegin_internal(cctx, dict, dictSize, ZSTD_dm_auto, NULL,
                                         params, srcSize, ZSTDb_not_buffered) );
    return ZSTD_compressEnd(cctx, dst, dstCapacity, src, srcSize);
}

size_t ZSTD_compress_usingDict(ZSTD_CCtx* ctx, void* dst, size_t dstCapacity, const void* src, size_t srcSize,
                               const void* dict, size_t dictSize, int compressionLevel)
{
    ZSTD_parameters params = ZSTD_getParams(compressionLevel, srcSize, dict ? dictSize : 0);
    params.fParams.contentSizeFlag = 1;
    return ZSTD_compress_internal(ctx, dst, dstCapacity, src, srcSize, dict, dictSize, params);
}

size_t ZSTD_compressCCtx (ZSTD_CCtx* ctx, void* dst, size_t dstCapacity, const void* src, size_t srcSize, int compressionLevel)
{
    return ZSTD_compress_usingDict(ctx, dst, dstCapacity, src, srcSize, NULL, 0, compressionLevel);
}

size_t ZSTD_compress(void* dst, size_t dstCapacity, const void* src, size_t srcSize, int compressionLevel)
{
    size_t result;
    ZSTD_CCtx ctxBody;
    memset(&ctxBody, 0, sizeof(ctxBody));
    ctxBody.customMem = ZSTD_defaultCMem;
    result = ZSTD_compressCCtx(&ctxBody, dst, dstCapacity, src, srcSize, compressionLevel);
    ZSTD_free(ctxBody.workSpace, ZSTD_defaultCMem);  /* can't free ctxBody itself, as it's on stack; free only heap content */
    return result;
}


/* =====  Dictionary API  ===== */

/*! ZSTD_estimateCDictSize_advanced() :
 *  Estimate amount of memory that will be needed to create a dictionary with following arguments */
size_t ZSTD_estimateCDictSize_advanced(
        size_t dictSize, ZSTD_compressionParameters cParams,
        ZSTD_dictLoadMethod_e dictLoadMethod)
{
    DEBUGLOG(5, "sizeof(ZSTD_CDict) : %u", (U32)sizeof(ZSTD_CDict));
    DEBUGLOG(5, "CCtx estimate : %u",
             (U32)ZSTD_estimateCCtxSize_usingCParams(cParams));
    return sizeof(ZSTD_CDict) + ZSTD_estimateCCtxSize_usingCParams(cParams)
           + (dictLoadMethod == ZSTD_dlm_byRef ? 0 : dictSize);
}

size_t ZSTD_estimateCDictSize(size_t dictSize, int compressionLevel)
{
    ZSTD_compressionParameters const cParams = ZSTD_getCParams(compressionLevel, 0, dictSize);
    return ZSTD_estimateCDictSize_advanced(dictSize, cParams, ZSTD_dlm_byCopy);
}

size_t ZSTD_sizeof_CDict(const ZSTD_CDict* cdict)
{
    if (cdict==NULL) return 0;   /* support sizeof on NULL */
    DEBUGLOG(5, "sizeof(*cdict) : %u", (U32)sizeof(*cdict));
    DEBUGLOG(5, "ZSTD_sizeof_CCtx : %u", (U32)ZSTD_sizeof_CCtx(cdict->refContext));
    return ZSTD_sizeof_CCtx(cdict->refContext) + (cdict->dictBuffer ? cdict->dictContentSize : 0) + sizeof(*cdict);
}

static size_t ZSTD_initCDict_internal(
                    ZSTD_CDict* cdict,
              const void* dictBuffer, size_t dictSize,
                    ZSTD_dictLoadMethod_e dictLoadMethod,
                    ZSTD_dictMode_e dictMode,
                    ZSTD_compressionParameters cParams)
{
    DEBUGLOG(5, "ZSTD_initCDict_internal, mode %u", (U32)dictMode);
    if ((dictLoadMethod == ZSTD_dlm_byRef) || (!dictBuffer) || (!dictSize)) {
        cdict->dictBuffer = NULL;
        cdict->dictContent = dictBuffer;
    } else {
        void* const internalBuffer = ZSTD_malloc(dictSize, cdict->refContext->customMem);
        cdict->dictBuffer = internalBuffer;
        cdict->dictContent = internalBuffer;
        if (!internalBuffer) return ERROR(memory_allocation);
        memcpy(internalBuffer, dictBuffer, dictSize);
    }
    cdict->dictContentSize = dictSize;

    {   ZSTD_CCtx_params cctxParams = cdict->refContext->requestedParams;
        cctxParams.cParams = cParams;
        CHECK_F( ZSTD_compressBegin_internal(cdict->refContext,
                                        cdict->dictContent, dictSize, dictMode,
                                        NULL,
                                        cctxParams, ZSTD_CONTENTSIZE_UNKNOWN,
                                        ZSTDb_not_buffered) );
    }

    return 0;
}

ZSTD_CDict* ZSTD_createCDict_advanced(const void* dictBuffer, size_t dictSize,
                                      ZSTD_dictLoadMethod_e dictLoadMethod,
                                      ZSTD_dictMode_e dictMode,
                                      ZSTD_compressionParameters cParams, ZSTD_customMem customMem)
{
    DEBUGLOG(5, "ZSTD_createCDict_advanced, mode %u", (U32)dictMode);
    if (!customMem.customAlloc ^ !customMem.customFree) return NULL;

    {   ZSTD_CDict* const cdict = (ZSTD_CDict*)ZSTD_malloc(sizeof(ZSTD_CDict), customMem);
        ZSTD_CCtx* const cctx = ZSTD_createCCtx_advanced(customMem);

        if (!cdict || !cctx) {
            ZSTD_free(cdict, customMem);
            ZSTD_freeCCtx(cctx);
            return NULL;
        }
        cdict->refContext = cctx;
        if (ZSTD_isError( ZSTD_initCDict_internal(cdict,
                                        dictBuffer, dictSize,
                                        dictLoadMethod, dictMode,
                                        cParams) )) {
            ZSTD_freeCDict(cdict);
            return NULL;
        }

        return cdict;
    }
}

ZSTD_CDict* ZSTD_createCDict(const void* dict, size_t dictSize, int compressionLevel)
{
    ZSTD_compressionParameters cParams = ZSTD_getCParams(compressionLevel, 0, dictSize);
    return ZSTD_createCDict_advanced(dict, dictSize,
                                     ZSTD_dlm_byCopy, ZSTD_dm_auto,
                                     cParams, ZSTD_defaultCMem);
}

ZSTD_CDict* ZSTD_createCDict_byReference(const void* dict, size_t dictSize, int compressionLevel)
{
    ZSTD_compressionParameters cParams = ZSTD_getCParams(compressionLevel, 0, dictSize);
    return ZSTD_createCDict_advanced(dict, dictSize,
                                     ZSTD_dlm_byRef, ZSTD_dm_auto,
                                     cParams, ZSTD_defaultCMem);
}

size_t ZSTD_freeCDict(ZSTD_CDict* cdict)
{
    if (cdict==NULL) return 0;   /* support free on NULL */
    {   ZSTD_customMem const cMem = cdict->refContext->customMem;
        ZSTD_freeCCtx(cdict->refContext);
        ZSTD_free(cdict->dictBuffer, cMem);
        ZSTD_free(cdict, cMem);
        return 0;
    }
}

/*! ZSTD_initStaticCDict_advanced() :
 *  Generate a digested dictionary in provided memory area.
 *  workspace: The memory area to emplace the dictionary into.
 *             Provided pointer must 8-bytes aligned.
 *             It must outlive dictionary usage.
 *  workspaceSize: Use ZSTD_estimateCDictSize()
 *                 to determine how large workspace must be.
 *  cParams : use ZSTD_getCParams() to transform a compression level
 *            into its relevants cParams.
 * @return : pointer to ZSTD_CDict*, or NULL if error (size too small)
 *  Note : there is no corresponding "free" function.
 *         Since workspace was allocated externally, it must be freed externally.
 */
ZSTD_CDict* ZSTD_initStaticCDict(void* workspace, size_t workspaceSize,
                           const void* dict, size_t dictSize,
                                 ZSTD_dictLoadMethod_e dictLoadMethod,
                                 ZSTD_dictMode_e dictMode,
                                 ZSTD_compressionParameters cParams)
{
    size_t const cctxSize = ZSTD_estimateCCtxSize_usingCParams(cParams);
    size_t const neededSize = sizeof(ZSTD_CDict) + (dictLoadMethod == ZSTD_dlm_byRef ? 0 : dictSize)
                            + cctxSize;
    ZSTD_CDict* const cdict = (ZSTD_CDict*) workspace;
    void* ptr;
    DEBUGLOG(5, "(size_t)workspace & 7 : %u", (U32)(size_t)workspace & 7);
    if ((size_t)workspace & 7) return NULL;  /* 8-aligned */
    DEBUGLOG(5, "(workspaceSize < neededSize) : (%u < %u) => %u",
        (U32)workspaceSize, (U32)neededSize, (U32)(workspaceSize < neededSize));
    if (workspaceSize < neededSize) return NULL;

    if (dictLoadMethod == ZSTD_dlm_byCopy) {
        memcpy(cdict+1, dict, dictSize);
        dict = cdict+1;
        ptr = (char*)workspace + sizeof(ZSTD_CDict) + dictSize;
    } else {
        ptr = cdict+1;
    }
    cdict->refContext = ZSTD_initStaticCCtx(ptr, cctxSize);

    if (ZSTD_isError( ZSTD_initCDict_internal(cdict,
                                              dict, dictSize,
                                              ZSTD_dlm_byRef, dictMode,
                                              cParams) ))
        return NULL;

    return cdict;
}

ZSTD_compressionParameters ZSTD_getCParamsFromCDict(const ZSTD_CDict* cdict) {
    return cdict->refContext->appliedParams.cParams;
}

/* ZSTD_compressBegin_usingCDict_advanced() :
 * cdict must be != NULL */
size_t ZSTD_compressBegin_usingCDict_advanced(
    ZSTD_CCtx* const cctx, const ZSTD_CDict* const cdict,
    ZSTD_frameParameters const fParams, unsigned long long const pledgedSrcSize)
{
    if (cdict==NULL) return ERROR(dictionary_wrong);
    {   ZSTD_CCtx_params params = cctx->requestedParams;
        params.cParams = ZSTD_getCParamsFromCDict(cdict);
        params.fParams = fParams;
        DEBUGLOG(5, "ZSTD_compressBegin_usingCDict_advanced");
        return ZSTD_compressBegin_internal(cctx,
                                           NULL, 0, ZSTD_dm_auto,
                                           cdict,
                                           params, pledgedSrcSize,
                                           ZSTDb_not_buffered);
    }
}

/* ZSTD_compressBegin_usingCDict() :
 * pledgedSrcSize=0 means "unknown"
 * if pledgedSrcSize>0, it will enable contentSizeFlag */
size_t ZSTD_compressBegin_usingCDict(ZSTD_CCtx* cctx, const ZSTD_CDict* cdict)
{
    ZSTD_frameParameters const fParams = { 0 /*content*/, 0 /*checksum*/, 0 /*noDictID*/ };
    DEBUGLOG(5, "ZSTD_compressBegin_usingCDict : dictIDFlag == %u", !fParams.noDictIDFlag);
    return ZSTD_compressBegin_usingCDict_advanced(cctx, cdict, fParams, 0);
}

size_t ZSTD_compress_usingCDict_advanced(ZSTD_CCtx* cctx,
                                void* dst, size_t dstCapacity,
                                const void* src, size_t srcSize,
                                const ZSTD_CDict* cdict, ZSTD_frameParameters fParams)
{
    CHECK_F (ZSTD_compressBegin_usingCDict_advanced(cctx, cdict, fParams, srcSize));   /* will check if cdict != NULL */
    return ZSTD_compressEnd(cctx, dst, dstCapacity, src, srcSize);
}

/*! ZSTD_compress_usingCDict() :
 *  Compression using a digested Dictionary.
 *  Faster startup than ZSTD_compress_usingDict(), recommended when same dictionary is used multiple times.
 *  Note that compression parameters are decided at CDict creation time
 *  while frame parameters are hardcoded */
size_t ZSTD_compress_usingCDict(ZSTD_CCtx* cctx,
                                void* dst, size_t dstCapacity,
                                const void* src, size_t srcSize,
                                const ZSTD_CDict* cdict)
{
    ZSTD_frameParameters const fParams = { 1 /*content*/, 0 /*checksum*/, 0 /*noDictID*/ };
    return ZSTD_compress_usingCDict_advanced(cctx, dst, dstCapacity, src, srcSize, cdict, fParams);
}



/* ******************************************************************
*  Streaming
********************************************************************/

ZSTD_CStream* ZSTD_createCStream(void)
{
    return ZSTD_createCStream_advanced(ZSTD_defaultCMem);
}

ZSTD_CStream* ZSTD_initStaticCStream(void *workspace, size_t workspaceSize)
{
    return ZSTD_initStaticCCtx(workspace, workspaceSize);
}

ZSTD_CStream* ZSTD_createCStream_advanced(ZSTD_customMem customMem)
{   /* CStream and CCtx are now same object */
    return ZSTD_createCCtx_advanced(customMem);
}

size_t ZSTD_freeCStream(ZSTD_CStream* zcs)
{
    return ZSTD_freeCCtx(zcs);   /* same object */
}



/*======   Initialization   ======*/

size_t ZSTD_CStreamInSize(void)  { return ZSTD_BLOCKSIZE_MAX; }

size_t ZSTD_CStreamOutSize(void)
{
    return ZSTD_compressBound(ZSTD_BLOCKSIZE_MAX) + ZSTD_blockHeaderSize + 4 /* 32-bits hash */ ;
}

static size_t ZSTD_resetCStream_internal(ZSTD_CStream* zcs,
                    const void* dict, size_t dictSize, ZSTD_dictMode_e dictMode,
                    const ZSTD_CDict* cdict,
                    const ZSTD_CCtx_params params, unsigned long long pledgedSrcSize)
{
    DEBUGLOG(4, "ZSTD_resetCStream_internal");
    /* params are supposed to be fully validated at this point */
    assert(!ZSTD_isError(ZSTD_checkCParams(params.cParams)));
    assert(!((dict) && (cdict)));  /* either dict or cdict, not both */

    CHECK_F( ZSTD_compressBegin_internal(zcs,
                                        dict, dictSize, dictMode,
                                        cdict,
                                        params, pledgedSrcSize,
                                        ZSTDb_buffered) );

    zcs->inToCompress = 0;
    zcs->inBuffPos = 0;
    zcs->inBuffTarget = zcs->blockSize;
    zcs->outBuffContentSize = zcs->outBuffFlushedSize = 0;
    zcs->streamStage = zcss_load;
    zcs->frameEnded = 0;
    return 0;   /* ready to go */
}

size_t ZSTD_resetCStream(ZSTD_CStream* zcs, unsigned long long pledgedSrcSize)
{
    ZSTD_CCtx_params params = zcs->requestedParams;
    params.fParams.contentSizeFlag = (pledgedSrcSize > 0);
    params.cParams = ZSTD_getCParamsFromCCtxParams(params, pledgedSrcSize, 0);
    DEBUGLOG(4, "ZSTD_resetCStream");
    return ZSTD_resetCStream_internal(zcs, NULL, 0, ZSTD_dm_auto, zcs->cdict, params, pledgedSrcSize);
}

/*! ZSTD_initCStream_internal() :
 *  Note : not static, but hidden (not exposed). Used by zstdmt_compress.c
 *  Assumption 1 : params are valid
 *  Assumption 2 : either dict, or cdict, is defined, not both */
size_t ZSTD_initCStream_internal(ZSTD_CStream* zcs,
                    const void* dict, size_t dictSize, const ZSTD_CDict* cdict,
                    ZSTD_CCtx_params params, unsigned long long pledgedSrcSize)
{
    DEBUGLOG(4, "ZSTD_initCStream_internal");
    assert(!ZSTD_isError(ZSTD_checkCParams(params.cParams)));
    assert(!((dict) && (cdict)));  /* either dict or cdict, not both */

    if (dict && dictSize >= 8) {
        DEBUGLOG(5, "loading dictionary of size %u", (U32)dictSize);
        if (zcs->staticSize) {   /* static CCtx : never uses malloc */
            /* incompatible with internal cdict creation */
            return ERROR(memory_allocation);
        }
        ZSTD_freeCDict(zcs->cdictLocal);
        zcs->cdictLocal = ZSTD_createCDict_advanced(dict, dictSize,
                                            ZSTD_dlm_byCopy, ZSTD_dm_auto,
                                            params.cParams, zcs->customMem);
        zcs->cdict = zcs->cdictLocal;
        if (zcs->cdictLocal == NULL) return ERROR(memory_allocation);
    } else {
        if (cdict) {
            params.cParams = ZSTD_getCParamsFromCDict(cdict);  /* cParams are enforced from cdict */
        }
        ZSTD_freeCDict(zcs->cdictLocal);
        zcs->cdictLocal = NULL;
        zcs->cdict = cdict;
    }

    params.compressionLevel = ZSTD_CLEVEL_CUSTOM;
    zcs->requestedParams = params;

    return ZSTD_resetCStream_internal(zcs, NULL, 0, ZSTD_dm_auto, zcs->cdict, params, pledgedSrcSize);
}

/* ZSTD_initCStream_usingCDict_advanced() :
 * same as ZSTD_initCStream_usingCDict(), with control over frame parameters */
size_t ZSTD_initCStream_usingCDict_advanced(ZSTD_CStream* zcs,
                                            const ZSTD_CDict* cdict,
                                            ZSTD_frameParameters fParams,
                                            unsigned long long pledgedSrcSize)
{   /* cannot handle NULL cdict (does not know what to do) */
    if (!cdict) return ERROR(dictionary_wrong);
    {   ZSTD_CCtx_params params = zcs->requestedParams;
        params.cParams = ZSTD_getCParamsFromCDict(cdict);
        params.fParams = fParams;
        return ZSTD_initCStream_internal(zcs,
                                NULL, 0, cdict,
                                params, pledgedSrcSize);
    }
}

/* note : cdict must outlive compression session */
size_t ZSTD_initCStream_usingCDict(ZSTD_CStream* zcs, const ZSTD_CDict* cdict)
{
    ZSTD_frameParameters const fParams = { 0 /* contentSize */, 0 /* checksum */, 0 /* hideDictID */ };
    return ZSTD_initCStream_usingCDict_advanced(zcs, cdict, fParams, 0);  /* note : will check that cdict != NULL */
}

size_t ZSTD_initCStream_advanced(ZSTD_CStream* zcs,
                                 const void* dict, size_t dictSize,
                                 ZSTD_parameters params, unsigned long long pledgedSrcSize)
{
    ZSTD_CCtx_params const cctxParams =
            ZSTD_assignParamsToCCtxParams(zcs->requestedParams, params);
    CHECK_F( ZSTD_checkCParams(params.cParams) );
    return ZSTD_initCStream_internal(zcs, dict, dictSize, NULL, cctxParams, pledgedSrcSize);
}

size_t ZSTD_initCStream_usingDict(ZSTD_CStream* zcs, const void* dict, size_t dictSize, int compressionLevel)
{
    ZSTD_parameters const params = ZSTD_getParams(compressionLevel, 0, dictSize);
    ZSTD_CCtx_params const cctxParams =
            ZSTD_assignParamsToCCtxParams(zcs->requestedParams, params);
    return ZSTD_initCStream_internal(zcs, dict, dictSize, NULL, cctxParams, 0);
}

size_t ZSTD_initCStream_srcSize(ZSTD_CStream* zcs, int compressionLevel, unsigned long long pledgedSrcSize)
{
    ZSTD_CCtx_params cctxParams;
    ZSTD_parameters const params = ZSTD_getParams(compressionLevel, pledgedSrcSize, 0);
    cctxParams = ZSTD_assignParamsToCCtxParams(zcs->requestedParams, params);
    cctxParams.fParams.contentSizeFlag = (pledgedSrcSize>0);
    return ZSTD_initCStream_internal(zcs, NULL, 0, NULL, cctxParams, pledgedSrcSize);
}

size_t ZSTD_initCStream(ZSTD_CStream* zcs, int compressionLevel)
{
    return ZSTD_initCStream_srcSize(zcs, compressionLevel, 0);
}

/*======   Compression   ======*/

MEM_STATIC size_t ZSTD_limitCopy(void* dst, size_t dstCapacity,
                           const void* src, size_t srcSize)
{
    size_t const length = MIN(dstCapacity, srcSize);
    if (length) memcpy(dst, src, length);
    return length;
}

/** ZSTD_compressStream_generic():
 *  internal function for all *compressStream*() variants and *compress_generic()
 *  non-static, because can be called from zstdmt.c
 * @return : hint size for next input */
size_t ZSTD_compressStream_generic(ZSTD_CStream* zcs,
                                   ZSTD_outBuffer* output,
                                   ZSTD_inBuffer* input,
                                   ZSTD_EndDirective const flushMode)
{
    const char* const istart = (const char*)input->src;
    const char* const iend = istart + input->size;
    const char* ip = istart + input->pos;
    char* const ostart = (char*)output->dst;
    char* const oend = ostart + output->size;
    char* op = ostart + output->pos;
    U32 someMoreWork = 1;

    /* check expectations */
    DEBUGLOG(5, "ZSTD_compressStream_generic, flush=%u", (U32)flushMode);
    assert(zcs->inBuff != NULL);
    assert(zcs->inBuffSize>0);
    assert(zcs->outBuff!= NULL);
    assert(zcs->outBuffSize>0);
    assert(output->pos <= output->size);
    assert(input->pos <= input->size);

    while (someMoreWork) {
        switch(zcs->streamStage)
        {
        case zcss_init:
            /* call ZSTD_initCStream() first ! */
            return ERROR(init_missing);

        case zcss_load:
            if ( (flushMode == ZSTD_e_end)
              && ((size_t)(oend-op) >= ZSTD_compressBound(iend-ip))  /* enough dstCapacity */
              && (zcs->inBuffPos == 0) ) {
                /* shortcut to compression pass directly into output buffer */
                size_t const cSize = ZSTD_compressEnd(zcs,
                                                op, oend-op, ip, iend-ip);
                DEBUGLOG(4, "ZSTD_compressEnd : %u", (U32)cSize);
                if (ZSTD_isError(cSize)) return cSize;
                ip = iend;
                op += cSize;
                zcs->frameEnded = 1;
                ZSTD_startNewCompression(zcs);
                someMoreWork = 0; break;
            }
            /* complete loading into inBuffer */
            {   size_t const toLoad = zcs->inBuffTarget - zcs->inBuffPos;
                size_t const loaded = ZSTD_limitCopy(
                                        zcs->inBuff + zcs->inBuffPos, toLoad,
                                        ip, iend-ip);
                zcs->inBuffPos += loaded;
                ip += loaded;
                if ( (flushMode == ZSTD_e_continue)
                  && (zcs->inBuffPos < zcs->inBuffTarget) ) {
                    /* not enough input to fill full block : stop here */
                    someMoreWork = 0; break;
                }
                if ( (flushMode == ZSTD_e_flush)
                  && (zcs->inBuffPos == zcs->inToCompress) ) {
                    /* empty */
                    someMoreWork = 0; break;
                }
            }
            /* compress current block (note : this stage cannot be stopped in the middle) */
            DEBUGLOG(5, "stream compression stage (flushMode==%u)", flushMode);
            {   void* cDst;
                size_t cSize;
                size_t const iSize = zcs->inBuffPos - zcs->inToCompress;
                size_t oSize = oend-op;
                unsigned const lastBlock = (flushMode == ZSTD_e_end) && (ip==iend);
                if (oSize >= ZSTD_compressBound(iSize))
                    cDst = op;   /* compress into output buffer, to skip flush stage */
                else
                    cDst = zcs->outBuff, oSize = zcs->outBuffSize;
                cSize = lastBlock ?
                        ZSTD_compressEnd(zcs, cDst, oSize,
                                    zcs->inBuff + zcs->inToCompress, iSize) :
                        ZSTD_compressContinue(zcs, cDst, oSize,
                                    zcs->inBuff + zcs->inToCompress, iSize);
                if (ZSTD_isError(cSize)) return cSize;
                zcs->frameEnded = lastBlock;
                /* prepare next block */
                zcs->inBuffTarget = zcs->inBuffPos + zcs->blockSize;
                if (zcs->inBuffTarget > zcs->inBuffSize)
                    zcs->inBuffPos = 0, zcs->inBuffTarget = zcs->blockSize;
                DEBUGLOG(5, "inBuffTarget:%u / inBuffSize:%u",
                         (U32)zcs->inBuffTarget, (U32)zcs->inBuffSize);
                if (!lastBlock)
                    assert(zcs->inBuffTarget <= zcs->inBuffSize);
                zcs->inToCompress = zcs->inBuffPos;
                if (cDst == op) {  /* no need to flush */
                    op += cSize;
                    if (zcs->frameEnded) {
                        DEBUGLOG(5, "Frame completed directly in outBuffer");
                        someMoreWork = 0;
                        ZSTD_startNewCompression(zcs);
                    }
                    break;
                }
                zcs->outBuffContentSize = cSize;
                zcs->outBuffFlushedSize = 0;
                zcs->streamStage = zcss_flush; /* pass-through to flush stage */
            }
	    /* fall-through */
        case zcss_flush:
            DEBUGLOG(5, "flush stage");
            {   size_t const toFlush = zcs->outBuffContentSize - zcs->outBuffFlushedSize;
                size_t const flushed = ZSTD_limitCopy(op, oend-op,
                            zcs->outBuff + zcs->outBuffFlushedSize, toFlush);
                DEBUGLOG(5, "toFlush: %u into %u ==> flushed: %u",
                            (U32)toFlush, (U32)(oend-op), (U32)flushed);
                op += flushed;
                zcs->outBuffFlushedSize += flushed;
                if (toFlush!=flushed) {
                    /* flush not fully completed, presumably because dst is too small */
                    assert(op==oend);
                    someMoreWork = 0;
                    break;
                }
                zcs->outBuffContentSize = zcs->outBuffFlushedSize = 0;
                if (zcs->frameEnded) {
                    DEBUGLOG(5, "Frame completed on flush");
                    someMoreWork = 0;
                    ZSTD_startNewCompression(zcs);
                    break;
                }
                zcs->streamStage = zcss_load;
                break;
            }

        default: /* impossible */
            assert(0);
        }
    }

    input->pos = ip - istart;
    output->pos = op - ostart;
    if (zcs->frameEnded) return 0;
    {   size_t hintInSize = zcs->inBuffTarget - zcs->inBuffPos;
        if (hintInSize==0) hintInSize = zcs->blockSize;
        return hintInSize;
    }
}

size_t ZSTD_compressStream(ZSTD_CStream* zcs, ZSTD_outBuffer* output, ZSTD_inBuffer* input)
{
    /* check conditions */
    if (output->pos > output->size) return ERROR(GENERIC);
    if (input->pos  > input->size)  return ERROR(GENERIC);

    return ZSTD_compressStream_generic(zcs, output, input, ZSTD_e_continue);
}


size_t ZSTD_compress_generic (ZSTD_CCtx* cctx,
                              ZSTD_outBuffer* output,
                              ZSTD_inBuffer* input,
                              ZSTD_EndDirective endOp)
{
    DEBUGLOG(5, "ZSTD_compress_generic");
    /* check conditions */
    if (output->pos > output->size) return ERROR(GENERIC);
    if (input->pos  > input->size)  return ERROR(GENERIC);
    assert(cctx!=NULL);

    /* transparent initialization stage */
    if (cctx->streamStage == zcss_init) {
        ZSTD_prefixDict const prefixDict = cctx->prefixDict;
        ZSTD_CCtx_params params = cctx->requestedParams;
        params.cParams = ZSTD_getCParamsFromCCtxParams(
                cctx->requestedParams, cctx->pledgedSrcSizePlusOne-1, 0 /*dictSize*/);
        memset(&cctx->prefixDict, 0, sizeof(cctx->prefixDict));  /* single usage */
        assert(prefixDict.dict==NULL || cctx->cdict==NULL);   /* only one can be set */
        DEBUGLOG(4, "ZSTD_compress_generic : transparent init stage");

#ifdef ZSTD_MULTITHREAD
        if (params.nbThreads > 1) {
            if (cctx->mtctx == NULL || cctx->appliedParams.nbThreads != params.nbThreads) {
                ZSTDMT_freeCCtx(cctx->mtctx);
                cctx->mtctx = ZSTDMT_createCCtx_advanced(params.nbThreads, cctx->customMem);
                if (cctx->mtctx == NULL) return ERROR(memory_allocation);
            }
            DEBUGLOG(4, "call ZSTDMT_initCStream_internal as nbThreads=%u", params.nbThreads);
            CHECK_F( ZSTDMT_initCStream_internal(
                             cctx->mtctx,
                             prefixDict.dict, prefixDict.dictSize, ZSTD_dm_rawContent,
                             cctx->cdict, params, cctx->pledgedSrcSizePlusOne-1) );
            cctx->streamStage = zcss_load;
            cctx->appliedParams.nbThreads = params.nbThreads;
        } else
#endif
        {
            CHECK_F( ZSTD_resetCStream_internal(
                             cctx, prefixDict.dict, prefixDict.dictSize,
                             prefixDict.dictMode, cctx->cdict, params,
                             cctx->pledgedSrcSizePlusOne-1) );
    }   }

    /* compression stage */
#ifdef ZSTD_MULTITHREAD
    if (cctx->appliedParams.nbThreads > 1) {
        size_t const flushMin = ZSTDMT_compressStream_generic(cctx->mtctx, output, input, endOp);
        DEBUGLOG(5, "ZSTDMT_compressStream_generic result : %u", (U32)flushMin);
        if ( ZSTD_isError(flushMin)
          || (endOp == ZSTD_e_end && flushMin == 0) ) { /* compression completed */
            ZSTD_startNewCompression(cctx);
        }
        return flushMin;
    }
#endif
    CHECK_F( ZSTD_compressStream_generic(cctx, output, input, endOp) );
    DEBUGLOG(5, "completed ZSTD_compress_generic");
    return cctx->outBuffContentSize - cctx->outBuffFlushedSize; /* remaining to flush */
}

size_t ZSTD_compress_generic_simpleArgs (
                            ZSTD_CCtx* cctx,
                            void* dst, size_t dstCapacity, size_t* dstPos,
                      const void* src, size_t srcSize, size_t* srcPos,
                            ZSTD_EndDirective endOp)
{
    ZSTD_outBuffer output = { dst, dstCapacity, *dstPos };
    ZSTD_inBuffer  input  = { src, srcSize, *srcPos };
    /* ZSTD_compress_generic() will check validity of dstPos and srcPos */
    size_t const cErr = ZSTD_compress_generic(cctx, &output, &input, endOp);
    *dstPos = output.pos;
    *srcPos = input.pos;
    return cErr;
}


/*======   Finalize   ======*/

/*! ZSTD_flushStream() :
*   @return : amount of data remaining to flush */
size_t ZSTD_flushStream(ZSTD_CStream* zcs, ZSTD_outBuffer* output)
{
    ZSTD_inBuffer input = { NULL, 0, 0 };
    if (output->pos > output->size) return ERROR(GENERIC);
    CHECK_F( ZSTD_compressStream_generic(zcs, output, &input, ZSTD_e_flush) );
    return zcs->outBuffContentSize - zcs->outBuffFlushedSize;  /* remaining to flush */
}


size_t ZSTD_endStream(ZSTD_CStream* zcs, ZSTD_outBuffer* output)
{
    ZSTD_inBuffer input = { NULL, 0, 0 };
    if (output->pos > output->size) return ERROR(GENERIC);
    CHECK_F( ZSTD_compressStream_generic(zcs, output, &input, ZSTD_e_end) );
    {   size_t const lastBlockSize = zcs->frameEnded ? 0 : ZSTD_BLOCKHEADERSIZE;
        size_t const checksumSize = zcs->frameEnded ? 0 : zcs->appliedParams.fParams.checksumFlag * 4;
        size_t const toFlush = zcs->outBuffContentSize - zcs->outBuffFlushedSize + lastBlockSize + checksumSize;
        DEBUGLOG(5, "ZSTD_endStream : remaining to flush : %u",
                (unsigned)toFlush);
        return toFlush;
    }
}


/*-=====  Pre-defined compression levels  =====-*/

#define ZSTD_MAX_CLEVEL     22
int ZSTD_maxCLevel(void) { return ZSTD_MAX_CLEVEL; }

static const ZSTD_compressionParameters ZSTD_defaultCParameters[4][ZSTD_MAX_CLEVEL+1] = {
{   /* "default" - guarantees a monotonically increasing memory budget */
    /* W,  C,  H,  S,  L, TL, strat */
    { 18, 12, 12,  1,  7, 16, ZSTD_fast    },  /* level  0 - never used */
    { 19, 13, 14,  1,  7, 16, ZSTD_fast    },  /* level  1 */
    { 19, 15, 16,  1,  6, 16, ZSTD_fast    },  /* level  2 */
    { 20, 16, 17,  1,  5, 16, ZSTD_dfast   },  /* level  3 */
    { 20, 17, 18,  1,  5, 16, ZSTD_dfast   },  /* level  4 */
    { 20, 17, 18,  2,  5, 16, ZSTD_greedy  },  /* level  5 */
    { 21, 17, 19,  2,  5, 16, ZSTD_lazy    },  /* level  6 */
    { 21, 18, 19,  3,  5, 16, ZSTD_lazy    },  /* level  7 */
    { 21, 18, 20,  3,  5, 16, ZSTD_lazy2   },  /* level  8 */
    { 21, 19, 20,  3,  5, 16, ZSTD_lazy2   },  /* level  9 */
    { 21, 19, 21,  4,  5, 16, ZSTD_lazy2   },  /* level 10 */
    { 22, 20, 22,  4,  5, 16, ZSTD_lazy2   },  /* level 11 */
    { 22, 20, 22,  5,  5, 16, ZSTD_lazy2   },  /* level 12 */
    { 22, 21, 22,  5,  5, 16, ZSTD_lazy2   },  /* level 13 */
    { 22, 21, 22,  6,  5, 16, ZSTD_lazy2   },  /* level 14 */
    { 22, 21, 22,  5,  5, 16, ZSTD_btlazy2 },  /* level 15 */
    { 23, 22, 22,  5,  5, 16, ZSTD_btlazy2 },  /* level 16 */
    { 23, 22, 22,  4,  5, 24, ZSTD_btopt   },  /* level 17 */
    { 23, 22, 22,  5,  4, 32, ZSTD_btopt   },  /* level 18 */
    { 23, 23, 22,  6,  3, 48, ZSTD_btopt   },  /* level 19 */
    { 25, 25, 23,  7,  3, 64, ZSTD_btultra },  /* level 20 */
    { 26, 26, 24,  7,  3,256, ZSTD_btultra },  /* level 21 */
    { 27, 27, 25,  9,  3,512, ZSTD_btultra },  /* level 22 */
},
{   /* for srcSize <= 256 KB */
    /* W,  C,  H,  S,  L,  T, strat */
    {  0,  0,  0,  0,  0,  0, ZSTD_fast    },  /* level  0 - not used */
    { 18, 13, 14,  1,  6,  8, ZSTD_fast    },  /* level  1 */
    { 18, 14, 13,  1,  5,  8, ZSTD_dfast   },  /* level  2 */
    { 18, 16, 15,  1,  5,  8, ZSTD_dfast   },  /* level  3 */
    { 18, 15, 17,  1,  5,  8, ZSTD_greedy  },  /* level  4.*/
    { 18, 16, 17,  4,  5,  8, ZSTD_greedy  },  /* level  5.*/
    { 18, 16, 17,  3,  5,  8, ZSTD_lazy    },  /* level  6.*/
    { 18, 17, 17,  4,  4,  8, ZSTD_lazy    },  /* level  7 */
    { 18, 17, 17,  4,  4,  8, ZSTD_lazy2   },  /* level  8 */
    { 18, 17, 17,  5,  4,  8, ZSTD_lazy2   },  /* level  9 */
    { 18, 17, 17,  6,  4,  8, ZSTD_lazy2   },  /* level 10 */
    { 18, 18, 17,  6,  4,  8, ZSTD_lazy2   },  /* level 11.*/
    { 18, 18, 17,  7,  4,  8, ZSTD_lazy2   },  /* level 12.*/
    { 18, 19, 17,  6,  4,  8, ZSTD_btlazy2 },  /* level 13 */
    { 18, 18, 18,  4,  4, 16, ZSTD_btopt   },  /* level 14.*/
    { 18, 18, 18,  4,  3, 16, ZSTD_btopt   },  /* level 15.*/
    { 18, 19, 18,  6,  3, 32, ZSTD_btopt   },  /* level 16.*/
    { 18, 19, 18,  8,  3, 64, ZSTD_btopt   },  /* level 17.*/
    { 18, 19, 18,  9,  3,128, ZSTD_btopt   },  /* level 18.*/
    { 18, 19, 18, 10,  3,256, ZSTD_btopt   },  /* level 19.*/
    { 18, 19, 18, 11,  3,512, ZSTD_btultra },  /* level 20.*/
    { 18, 19, 18, 12,  3,512, ZSTD_btultra },  /* level 21.*/
    { 18, 19, 18, 13,  3,512, ZSTD_btultra },  /* level 22.*/
},
{   /* for srcSize <= 128 KB */
    /* W,  C,  H,  S,  L,  T, strat */
    { 17, 12, 12,  1,  7,  8, ZSTD_fast    },  /* level  0 - not used */
    { 17, 12, 13,  1,  6,  8, ZSTD_fast    },  /* level  1 */
    { 17, 13, 16,  1,  5,  8, ZSTD_fast    },  /* level  2 */
    { 17, 16, 16,  2,  5,  8, ZSTD_dfast   },  /* level  3 */
    { 17, 13, 15,  3,  4,  8, ZSTD_greedy  },  /* level  4 */
    { 17, 15, 17,  4,  4,  8, ZSTD_greedy  },  /* level  5 */
    { 17, 16, 17,  3,  4,  8, ZSTD_lazy    },  /* level  6 */
    { 17, 15, 17,  4,  4,  8, ZSTD_lazy2   },  /* level  7 */
    { 17, 17, 17,  4,  4,  8, ZSTD_lazy2   },  /* level  8 */
    { 17, 17, 17,  5,  4,  8, ZSTD_lazy2   },  /* level  9 */
    { 17, 17, 17,  6,  4,  8, ZSTD_lazy2   },  /* level 10 */
    { 17, 17, 17,  7,  4,  8, ZSTD_lazy2   },  /* level 11 */
    { 17, 17, 17,  8,  4,  8, ZSTD_lazy2   },  /* level 12 */
    { 17, 18, 17,  6,  4,  8, ZSTD_btlazy2 },  /* level 13.*/
    { 17, 17, 17,  7,  3,  8, ZSTD_btopt   },  /* level 14.*/
    { 17, 17, 17,  7,  3, 16, ZSTD_btopt   },  /* level 15.*/
    { 17, 18, 17,  7,  3, 32, ZSTD_btopt   },  /* level 16.*/
    { 17, 18, 17,  7,  3, 64, ZSTD_btopt   },  /* level 17.*/
    { 17, 18, 17,  7,  3,256, ZSTD_btopt   },  /* level 18.*/
    { 17, 18, 17,  8,  3,256, ZSTD_btopt   },  /* level 19.*/
    { 17, 18, 17,  9,  3,256, ZSTD_btultra },  /* level 20.*/
    { 17, 18, 17, 10,  3,256, ZSTD_btultra },  /* level 21.*/
    { 17, 18, 17, 11,  3,512, ZSTD_btultra },  /* level 22.*/
},
{   /* for srcSize <= 16 KB */
    /* W,  C,  H,  S,  L,  T, strat */
    { 14, 12, 12,  1,  7,  6, ZSTD_fast    },  /* level  0 - not used */
    { 14, 14, 14,  1,  6,  6, ZSTD_fast    },  /* level  1 */
    { 14, 14, 14,  1,  4,  6, ZSTD_fast    },  /* level  2 */
    { 14, 14, 14,  1,  4,  6, ZSTD_dfast   },  /* level  3.*/
    { 14, 14, 14,  4,  4,  6, ZSTD_greedy  },  /* level  4.*/
    { 14, 14, 14,  3,  4,  6, ZSTD_lazy    },  /* level  5.*/
    { 14, 14, 14,  4,  4,  6, ZSTD_lazy2   },  /* level  6 */
    { 14, 14, 14,  5,  4,  6, ZSTD_lazy2   },  /* level  7 */
    { 14, 14, 14,  6,  4,  6, ZSTD_lazy2   },  /* level  8.*/
    { 14, 15, 14,  6,  4,  6, ZSTD_btlazy2 },  /* level  9.*/
    { 14, 15, 14,  3,  3,  6, ZSTD_btopt   },  /* level 10.*/
    { 14, 15, 14,  6,  3,  8, ZSTD_btopt   },  /* level 11.*/
    { 14, 15, 14,  6,  3, 16, ZSTD_btopt   },  /* level 12.*/
    { 14, 15, 14,  6,  3, 24, ZSTD_btopt   },  /* level 13.*/
    { 14, 15, 15,  6,  3, 48, ZSTD_btopt   },  /* level 14.*/
    { 14, 15, 15,  6,  3, 64, ZSTD_btopt   },  /* level 15.*/
    { 14, 15, 15,  6,  3, 96, ZSTD_btopt   },  /* level 16.*/
    { 14, 15, 15,  6,  3,128, ZSTD_btopt   },  /* level 17.*/
    { 14, 15, 15,  6,  3,256, ZSTD_btopt   },  /* level 18.*/
    { 14, 15, 15,  7,  3,256, ZSTD_btopt   },  /* level 19.*/
    { 14, 15, 15,  8,  3,256, ZSTD_btultra },  /* level 20.*/
    { 14, 15, 15,  9,  3,256, ZSTD_btultra },  /* level 21.*/
    { 14, 15, 15, 10,  3,256, ZSTD_btultra },  /* level 22.*/
},
};

#if defined(ZSTD_DEBUG) && (ZSTD_DEBUG>=1)
/* This function just controls
 * the monotonic memory budget increase of ZSTD_defaultCParameters[0].
 * Run once, on first ZSTD_getCParams() usage, if ZSTD_DEBUG is enabled
 */
MEM_STATIC void ZSTD_check_compressionLevel_monotonicIncrease_memoryBudget(void)
{
    int level;
    for (level=1; level<ZSTD_maxCLevel(); level++) {
        ZSTD_compressionParameters const c1 = ZSTD_defaultCParameters[0][level];
        ZSTD_compressionParameters const c2 = ZSTD_defaultCParameters[0][level+1];
        assert(c1.windowLog <= c2.windowLog);
#       define ZSTD_TABLECOST(h,c) ((1<<(h)) + (1<<(c)))
        assert(ZSTD_TABLECOST(c1.hashLog, c1.chainLog) <= ZSTD_TABLECOST(c2.hashLog, c2.chainLog));
    }
}
#endif

/*! ZSTD_getCParams() :
*   @return ZSTD_compressionParameters structure for a selected compression level, `srcSize` and `dictSize`.
*   Size values are optional, provide 0 if not known or unused */
ZSTD_compressionParameters ZSTD_getCParams(int compressionLevel, unsigned long long srcSizeHint, size_t dictSize)
{
    size_t const addedSize = srcSizeHint ? 0 : 500;
    U64 const rSize = srcSizeHint+dictSize ? srcSizeHint+dictSize+addedSize : (U64)-1;
    U32 const tableID = (rSize <= 256 KB) + (rSize <= 128 KB) + (rSize <= 16 KB);   /* intentional underflow for srcSizeHint == 0 */

#if defined(ZSTD_DEBUG) && (ZSTD_DEBUG>=1)
    static int g_monotonicTest = 1;
    if (g_monotonicTest) {
        ZSTD_check_compressionLevel_monotonicIncrease_memoryBudget();
        g_monotonicTest=0;
    }
#endif

    if (compressionLevel <= 0) compressionLevel = ZSTD_CLEVEL_DEFAULT;   /* 0 == default; no negative compressionLevel yet */
    if (compressionLevel > ZSTD_MAX_CLEVEL) compressionLevel = ZSTD_MAX_CLEVEL;
    { ZSTD_compressionParameters const cp = ZSTD_defaultCParameters[tableID][compressionLevel];
      return ZSTD_adjustCParams_internal(cp, srcSizeHint, dictSize); }

}

/*! ZSTD_getParams() :
*   same as ZSTD_getCParams(), but @return a `ZSTD_parameters` object (instead of `ZSTD_compressionParameters`).
*   All fields of `ZSTD_frameParameters` are set to default (0) */
ZSTD_parameters ZSTD_getParams(int compressionLevel, unsigned long long srcSizeHint, size_t dictSize) {
    ZSTD_parameters params;
    ZSTD_compressionParameters const cParams = ZSTD_getCParams(compressionLevel, srcSizeHint, dictSize);
    memset(&params, 0, sizeof(params));
    params.cParams = cParams;
    return params;
}
