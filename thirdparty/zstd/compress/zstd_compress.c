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
#include "cpu.h"
#include "mem.h"
#define FSE_STATIC_LINKING_ONLY   /* FSE_encodeSymbol */
#include "fse.h"
#define HUF_STATIC_LINKING_ONLY
#include "huf.h"
#include "zstd_compress_internal.h"
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
*  Context memory management
***************************************/
struct ZSTD_CDict_s {
    void* dictBuffer;
    const void* dictContent;
    size_t dictContentSize;
    void* workspace;
    size_t workspaceSize;
    ZSTD_matchState_t matchState;
    ZSTD_compressedBlockState_t cBlockState;
    ZSTD_compressionParameters cParams;
    ZSTD_customMem customMem;
    U32 dictID;
};  /* typedef'd to ZSTD_CDict within "zstd.h" */

ZSTD_CCtx* ZSTD_createCCtx(void)
{
    return ZSTD_createCCtx_advanced(ZSTD_defaultCMem);
}

ZSTD_CCtx* ZSTD_createCCtx_advanced(ZSTD_customMem customMem)
{
    ZSTD_STATIC_ASSERT(zcss_init==0);
    ZSTD_STATIC_ASSERT(ZSTD_CONTENTSIZE_UNKNOWN==(0ULL - 1));
    if (!customMem.customAlloc ^ !customMem.customFree) return NULL;
    {   ZSTD_CCtx* const cctx = (ZSTD_CCtx*)ZSTD_calloc(sizeof(ZSTD_CCtx), customMem);
        if (!cctx) return NULL;
        cctx->customMem = customMem;
        cctx->requestedParams.compressionLevel = ZSTD_CLEVEL_DEFAULT;
        cctx->requestedParams.fParams.contentSizeFlag = 1;
        cctx->bmi2 = ZSTD_cpuid_bmi2(ZSTD_cpuid());
        return cctx;
    }
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

    /* statically sized space. entropyWorkspace never moves (but prev/next block swap places) */
    if (cctx->workSpaceSize < HUF_WORKSPACE_SIZE + 2 * sizeof(ZSTD_compressedBlockState_t)) return NULL;
    assert(((size_t)cctx->workSpace & (sizeof(void*)-1)) == 0);   /* ensure correct alignment */
    cctx->blockState.prevCBlock = (ZSTD_compressedBlockState_t*)cctx->workSpace;
    cctx->blockState.nextCBlock = cctx->blockState.prevCBlock + 1;
    {
        void* const ptr = cctx->blockState.nextCBlock + 1;
        cctx->entropyWorkspace = (U32*)ptr;
    }
    cctx->bmi2 = ZSTD_cpuid_bmi2(ZSTD_cpuid());
    return cctx;
}

size_t ZSTD_freeCCtx(ZSTD_CCtx* cctx)
{
    if (cctx==NULL) return 0;   /* support free on NULL */
    if (cctx->staticSize) return ERROR(memory_allocation);   /* not compatible with static CCtx */
    ZSTD_free(cctx->workSpace, cctx->customMem); cctx->workSpace = NULL;
    ZSTD_freeCDict(cctx->cdictLocal); cctx->cdictLocal = NULL;
#ifdef ZSTD_MULTITHREAD
    ZSTDMT_freeCCtx(cctx->mtctx); cctx->mtctx = NULL;
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

ZSTD_compressionParameters ZSTD_getCParamsFromCCtxParams(
        const ZSTD_CCtx_params* CCtxParams, U64 srcSizeHint, size_t dictSize)
{
    ZSTD_compressionParameters cParams = ZSTD_getCParams(CCtxParams->compressionLevel, srcSizeHint, dictSize);
    if (CCtxParams->ldmParams.enableLdm) cParams.windowLog = ZSTD_LDM_DEFAULT_WINDOW_LOG;
    if (CCtxParams->cParams.windowLog) cParams.windowLog = CCtxParams->cParams.windowLog;
    if (CCtxParams->cParams.hashLog) cParams.hashLog = CCtxParams->cParams.hashLog;
    if (CCtxParams->cParams.chainLog) cParams.chainLog = CCtxParams->cParams.chainLog;
    if (CCtxParams->cParams.searchLog) cParams.searchLog = CCtxParams->cParams.searchLog;
    if (CCtxParams->cParams.searchLength) cParams.searchLength = CCtxParams->cParams.searchLength;
    if (CCtxParams->cParams.targetLength) cParams.targetLength = CCtxParams->cParams.targetLength;
    if (CCtxParams->cParams.strategy) cParams.strategy = CCtxParams->cParams.strategy;
    return cParams;
}

static ZSTD_CCtx_params ZSTD_makeCCtxParamsFromCParams(
        ZSTD_compressionParameters cParams)
{
    ZSTD_CCtx_params cctxParams;
    memset(&cctxParams, 0, sizeof(cctxParams));
    cctxParams.cParams = cParams;
    cctxParams.compressionLevel = ZSTD_CLEVEL_DEFAULT;  /* should not matter, as all cParams are presumed properly defined */
    assert(!ZSTD_checkCParams(cParams));
    cctxParams.fParams.contentSizeFlag = 1;
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
    params->fParams.contentSizeFlag = 1;
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

size_t ZSTD_CCtxParams_reset(ZSTD_CCtx_params* params)
{
    return ZSTD_CCtxParams_init(params, ZSTD_CLEVEL_DEFAULT);
}

size_t ZSTD_CCtxParams_init(ZSTD_CCtx_params* cctxParams, int compressionLevel) {
    if (!cctxParams) { return ERROR(GENERIC); }
    memset(cctxParams, 0, sizeof(*cctxParams));
    cctxParams->compressionLevel = compressionLevel;
    cctxParams->fParams.contentSizeFlag = 1;
    return 0;
}

size_t ZSTD_CCtxParams_init_advanced(ZSTD_CCtx_params* cctxParams, ZSTD_parameters params)
{
    if (!cctxParams) { return ERROR(GENERIC); }
    CHECK_F( ZSTD_checkCParams(params.cParams) );
    memset(cctxParams, 0, sizeof(*cctxParams));
    cctxParams->cParams = params.cParams;
    cctxParams->fParams = params.fParams;
    cctxParams->compressionLevel = ZSTD_CLEVEL_DEFAULT;   /* should not matter, as all cParams are presumed properly defined */
    assert(!ZSTD_checkCParams(params.cParams));
    return 0;
}

/* ZSTD_assignParamsToCCtxParams() :
 * params is presumed valid at this stage */
static ZSTD_CCtx_params ZSTD_assignParamsToCCtxParams(
        ZSTD_CCtx_params cctxParams, ZSTD_parameters params)
{
    ZSTD_CCtx_params ret = cctxParams;
    ret.cParams = params.cParams;
    ret.fParams = params.fParams;
    ret.compressionLevel = ZSTD_CLEVEL_DEFAULT;   /* should not matter, as all cParams are presumed properly defined */
    assert(!ZSTD_checkCParams(params.cParams));
    return ret;
}

#define CLAMPCHECK(val,min,max) {            \
    if (((val)<(min)) | ((val)>(max))) {     \
        return ERROR(parameter_outOfBound);  \
}   }


static int ZSTD_isUpdateAuthorized(ZSTD_cParameter param)
{
    switch(param)
    {
    case ZSTD_p_compressionLevel:
    case ZSTD_p_hashLog:
    case ZSTD_p_chainLog:
    case ZSTD_p_searchLog:
    case ZSTD_p_minMatch:
    case ZSTD_p_targetLength:
    case ZSTD_p_compressionStrategy:
    case ZSTD_p_compressLiterals:
        return 1;

    case ZSTD_p_format:
    case ZSTD_p_windowLog:
    case ZSTD_p_contentSizeFlag:
    case ZSTD_p_checksumFlag:
    case ZSTD_p_dictIDFlag:
    case ZSTD_p_forceMaxWindow :
    case ZSTD_p_nbWorkers:
    case ZSTD_p_jobSize:
    case ZSTD_p_overlapSizeLog:
    case ZSTD_p_enableLongDistanceMatching:
    case ZSTD_p_ldmHashLog:
    case ZSTD_p_ldmMinMatch:
    case ZSTD_p_ldmBucketSizeLog:
    case ZSTD_p_ldmHashEveryLog:
    default:
        return 0;
    }
}

size_t ZSTD_CCtx_setParameter(ZSTD_CCtx* cctx, ZSTD_cParameter param, unsigned value)
{
    DEBUGLOG(4, "ZSTD_CCtx_setParameter (%u, %u)", (U32)param, value);
    if (cctx->streamStage != zcss_init) {
        if (ZSTD_isUpdateAuthorized(param)) {
            cctx->cParamsChanged = 1;
        } else {
            return ERROR(stage_wrong);
    }   }

    switch(param)
    {
    case ZSTD_p_format :
        return ZSTD_CCtxParam_setParameter(&cctx->requestedParams, param, value);

    case ZSTD_p_compressionLevel:
        if (cctx->cdict) return ERROR(stage_wrong);
        return ZSTD_CCtxParam_setParameter(&cctx->requestedParams, param, value);

    case ZSTD_p_windowLog:
    case ZSTD_p_hashLog:
    case ZSTD_p_chainLog:
    case ZSTD_p_searchLog:
    case ZSTD_p_minMatch:
    case ZSTD_p_targetLength:
    case ZSTD_p_compressionStrategy:
        if (cctx->cdict) return ERROR(stage_wrong);
        return ZSTD_CCtxParam_setParameter(&cctx->requestedParams, param, value);

    case ZSTD_p_compressLiterals:
    case ZSTD_p_contentSizeFlag:
    case ZSTD_p_checksumFlag:
    case ZSTD_p_dictIDFlag:
        return ZSTD_CCtxParam_setParameter(&cctx->requestedParams, param, value);

    case ZSTD_p_forceMaxWindow :  /* Force back-references to remain < windowSize,
                                   * even when referencing into Dictionary content.
                                   * default : 0 when using a CDict, 1 when using a Prefix */
        return ZSTD_CCtxParam_setParameter(&cctx->requestedParams, param, value);

    case ZSTD_p_nbWorkers:
        if ((value>0) && cctx->staticSize) {
            return ERROR(parameter_unsupported);  /* MT not compatible with static alloc */
        }
        return ZSTD_CCtxParam_setParameter(&cctx->requestedParams, param, value);

    case ZSTD_p_jobSize:
    case ZSTD_p_overlapSizeLog:
        return ZSTD_CCtxParam_setParameter(&cctx->requestedParams, param, value);

    case ZSTD_p_enableLongDistanceMatching:
    case ZSTD_p_ldmHashLog:
    case ZSTD_p_ldmMinMatch:
    case ZSTD_p_ldmBucketSizeLog:
    case ZSTD_p_ldmHashEveryLog:
        if (cctx->cdict) return ERROR(stage_wrong);
        return ZSTD_CCtxParam_setParameter(&cctx->requestedParams, param, value);

    default: return ERROR(parameter_unsupported);
    }
}

size_t ZSTD_CCtxParam_setParameter(
        ZSTD_CCtx_params* CCtxParams, ZSTD_cParameter param, unsigned value)
{
    DEBUGLOG(4, "ZSTD_CCtxParam_setParameter (%u, %u)", (U32)param, value);
    switch(param)
    {
    case ZSTD_p_format :
        if (value > (unsigned)ZSTD_f_zstd1_magicless)
            return ERROR(parameter_unsupported);
        CCtxParams->format = (ZSTD_format_e)value;
        return (size_t)CCtxParams->format;

    case ZSTD_p_compressionLevel : {
        int cLevel = (int)value;  /* cast expected to restore negative sign */
        if (cLevel > ZSTD_maxCLevel()) cLevel = ZSTD_maxCLevel();
        if (cLevel) {  /* 0 : does not change current level */
            CCtxParams->disableLiteralCompression = (cLevel<0);  /* negative levels disable huffman */
            CCtxParams->compressionLevel = cLevel;
        }
        if (CCtxParams->compressionLevel >= 0) return CCtxParams->compressionLevel;
        return 0;  /* return type (size_t) cannot represent negative values */
    }

    case ZSTD_p_windowLog :
        if (value>0)   /* 0 => use default */
            CLAMPCHECK(value, ZSTD_WINDOWLOG_MIN, ZSTD_WINDOWLOG_MAX);
        CCtxParams->cParams.windowLog = value;
        return CCtxParams->cParams.windowLog;

    case ZSTD_p_hashLog :
        if (value>0)   /* 0 => use default */
            CLAMPCHECK(value, ZSTD_HASHLOG_MIN, ZSTD_HASHLOG_MAX);
        CCtxParams->cParams.hashLog = value;
        return CCtxParams->cParams.hashLog;

    case ZSTD_p_chainLog :
        if (value>0)   /* 0 => use default */
            CLAMPCHECK(value, ZSTD_CHAINLOG_MIN, ZSTD_CHAINLOG_MAX);
        CCtxParams->cParams.chainLog = value;
        return CCtxParams->cParams.chainLog;

    case ZSTD_p_searchLog :
        if (value>0)   /* 0 => use default */
            CLAMPCHECK(value, ZSTD_SEARCHLOG_MIN, ZSTD_SEARCHLOG_MAX);
        CCtxParams->cParams.searchLog = value;
        return value;

    case ZSTD_p_minMatch :
        if (value>0)   /* 0 => use default */
            CLAMPCHECK(value, ZSTD_SEARCHLENGTH_MIN, ZSTD_SEARCHLENGTH_MAX);
        CCtxParams->cParams.searchLength = value;
        return CCtxParams->cParams.searchLength;

    case ZSTD_p_targetLength :
        /* all values are valid. 0 => use default */
        CCtxParams->cParams.targetLength = value;
        return CCtxParams->cParams.targetLength;

    case ZSTD_p_compressionStrategy :
        if (value>0)   /* 0 => use default */
            CLAMPCHECK(value, (unsigned)ZSTD_fast, (unsigned)ZSTD_btultra);
        CCtxParams->cParams.strategy = (ZSTD_strategy)value;
        return (size_t)CCtxParams->cParams.strategy;

    case ZSTD_p_compressLiterals:
        CCtxParams->disableLiteralCompression = !value;
        return !CCtxParams->disableLiteralCompression;

    case ZSTD_p_contentSizeFlag :
        /* Content size written in frame header _when known_ (default:1) */
        DEBUGLOG(4, "set content size flag = %u", (value>0));
        CCtxParams->fParams.contentSizeFlag = value > 0;
        return CCtxParams->fParams.contentSizeFlag;

    case ZSTD_p_checksumFlag :
        /* A 32-bits content checksum will be calculated and written at end of frame (default:0) */
        CCtxParams->fParams.checksumFlag = value > 0;
        return CCtxParams->fParams.checksumFlag;

    case ZSTD_p_dictIDFlag : /* When applicable, dictionary's dictID is provided in frame header (default:1) */
        DEBUGLOG(4, "set dictIDFlag = %u", (value>0));
        CCtxParams->fParams.noDictIDFlag = !value;
        return !CCtxParams->fParams.noDictIDFlag;

    case ZSTD_p_forceMaxWindow :
        CCtxParams->forceWindow = (value > 0);
        return CCtxParams->forceWindow;

    case ZSTD_p_nbWorkers :
#ifndef ZSTD_MULTITHREAD
        if (value>0) return ERROR(parameter_unsupported);
        return 0;
#else
        return ZSTDMT_CCtxParam_setNbWorkers(CCtxParams, value);
#endif

    case ZSTD_p_jobSize :
#ifndef ZSTD_MULTITHREAD
        return ERROR(parameter_unsupported);
#else
        return ZSTDMT_CCtxParam_setMTCtxParameter(CCtxParams, ZSTDMT_p_jobSize, value);
#endif

    case ZSTD_p_overlapSizeLog :
#ifndef ZSTD_MULTITHREAD
        return ERROR(parameter_unsupported);
#else
        return ZSTDMT_CCtxParam_setMTCtxParameter(CCtxParams, ZSTDMT_p_overlapSectionLog, value);
#endif

    case ZSTD_p_enableLongDistanceMatching :
        CCtxParams->ldmParams.enableLdm = (value>0);
        return CCtxParams->ldmParams.enableLdm;

    case ZSTD_p_ldmHashLog :
        if (value>0)   /* 0 ==> auto */
            CLAMPCHECK(value, ZSTD_HASHLOG_MIN, ZSTD_HASHLOG_MAX);
        CCtxParams->ldmParams.hashLog = value;
        return CCtxParams->ldmParams.hashLog;

    case ZSTD_p_ldmMinMatch :
        if (value>0)   /* 0 ==> default */
            CLAMPCHECK(value, ZSTD_LDM_MINMATCH_MIN, ZSTD_LDM_MINMATCH_MAX);
        CCtxParams->ldmParams.minMatchLength = value;
        return CCtxParams->ldmParams.minMatchLength;

    case ZSTD_p_ldmBucketSizeLog :
        if (value > ZSTD_LDM_BUCKETSIZELOG_MAX)
            return ERROR(parameter_outOfBound);
        CCtxParams->ldmParams.bucketSizeLog = value;
        return CCtxParams->ldmParams.bucketSizeLog;

    case ZSTD_p_ldmHashEveryLog :
        if (value > ZSTD_WINDOWLOG_MAX - ZSTD_HASHLOG_MIN)
            return ERROR(parameter_outOfBound);
        CCtxParams->ldmParams.hashEveryLog = value;
        return CCtxParams->ldmParams.hashEveryLog;

    default: return ERROR(parameter_unsupported);
    }
}

/** ZSTD_CCtx_setParametersUsingCCtxParams() :
 *  just applies `params` into `cctx`
 *  no action is performed, parameters are merely stored.
 *  If ZSTDMT is enabled, parameters are pushed to cctx->mtctx.
 *    This is possible even if a compression is ongoing.
 *    In which case, new parameters will be applied on the fly, starting with next compression job.
 */
size_t ZSTD_CCtx_setParametersUsingCCtxParams(
        ZSTD_CCtx* cctx, const ZSTD_CCtx_params* params)
{
    if (cctx->streamStage != zcss_init) return ERROR(stage_wrong);
    if (cctx->cdict) return ERROR(stage_wrong);

    cctx->requestedParams = *params;
    return 0;
}

ZSTDLIB_API size_t ZSTD_CCtx_setPledgedSrcSize(ZSTD_CCtx* cctx, unsigned long long pledgedSrcSize)
{
    DEBUGLOG(4, "ZSTD_CCtx_setPledgedSrcSize to %u bytes", (U32)pledgedSrcSize);
    if (cctx->streamStage != zcss_init) return ERROR(stage_wrong);
    cctx->pledgedSrcSizePlusOne = pledgedSrcSize+1;
    return 0;
}

size_t ZSTD_CCtx_loadDictionary_advanced(
        ZSTD_CCtx* cctx, const void* dict, size_t dictSize,
        ZSTD_dictLoadMethod_e dictLoadMethod, ZSTD_dictContentType_e dictContentType)
{
    if (cctx->streamStage != zcss_init) return ERROR(stage_wrong);
    if (cctx->staticSize) return ERROR(memory_allocation);  /* no malloc for static CCtx */
    DEBUGLOG(4, "ZSTD_CCtx_loadDictionary_advanced (size: %u)", (U32)dictSize);
    ZSTD_freeCDict(cctx->cdictLocal);  /* in case one already exists */
    if (dict==NULL || dictSize==0) {   /* no dictionary mode */
        cctx->cdictLocal = NULL;
        cctx->cdict = NULL;
    } else {
        ZSTD_compressionParameters const cParams =
                ZSTD_getCParamsFromCCtxParams(&cctx->requestedParams, cctx->pledgedSrcSizePlusOne-1, dictSize);
        cctx->cdictLocal = ZSTD_createCDict_advanced(
                                dict, dictSize,
                                dictLoadMethod, dictContentType,
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
            cctx, dict, dictSize, ZSTD_dlm_byRef, ZSTD_dct_auto);
}

ZSTDLIB_API size_t ZSTD_CCtx_loadDictionary(ZSTD_CCtx* cctx, const void* dict, size_t dictSize)
{
    return ZSTD_CCtx_loadDictionary_advanced(
            cctx, dict, dictSize, ZSTD_dlm_byCopy, ZSTD_dct_auto);
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
    return ZSTD_CCtx_refPrefix_advanced(cctx, prefix, prefixSize, ZSTD_dct_rawContent);
}

size_t ZSTD_CCtx_refPrefix_advanced(
        ZSTD_CCtx* cctx, const void* prefix, size_t prefixSize, ZSTD_dictContentType_e dictContentType)
{
    if (cctx->streamStage != zcss_init) return ERROR(stage_wrong);
    cctx->cdict = NULL;   /* prefix discards any prior cdict */
    cctx->prefixDict.dict = prefix;
    cctx->prefixDict.dictSize = prefixSize;
    cctx->prefixDict.dictContentType = dictContentType;
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
    if ((U32)(cParams.targetLength) < ZSTD_TARGETLENGTH_MIN)
        return ERROR(parameter_unsupported);
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
    if ((U32)(cParams.targetLength) < ZSTD_TARGETLENGTH_MIN) cParams.targetLength = ZSTD_TARGETLENGTH_MIN;
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

static size_t ZSTD_sizeof_matchState(ZSTD_compressionParameters const* cParams, const U32 forCCtx)
{
    size_t const chainSize = (cParams->strategy == ZSTD_fast) ? 0 : ((size_t)1 << cParams->chainLog);
    size_t const hSize = ((size_t)1) << cParams->hashLog;
    U32    const hashLog3 = (forCCtx && cParams->searchLength==3) ? MIN(ZSTD_HASHLOG3_MAX, cParams->windowLog) : 0;
    size_t const h3Size = ((size_t)1) << hashLog3;
    size_t const tableSpace = (chainSize + hSize + h3Size) * sizeof(U32);
    size_t const optPotentialSpace = ((MaxML+1) + (MaxLL+1) + (MaxOff+1) + (1<<Litbits)) * sizeof(U32)
                          + (ZSTD_OPT_NUM+1) * (sizeof(ZSTD_match_t)+sizeof(ZSTD_optimal_t));
    size_t const optSpace = (forCCtx && ((cParams->strategy == ZSTD_btopt) ||
                                         (cParams->strategy == ZSTD_btultra)))
                                ? optPotentialSpace
                                : 0;
    DEBUGLOG(4, "chainSize: %u - hSize: %u - h3Size: %u",
                (U32)chainSize, (U32)hSize, (U32)h3Size);
    return tableSpace + optSpace;
}

size_t ZSTD_estimateCCtxSize_usingCCtxParams(const ZSTD_CCtx_params* params)
{
    /* Estimate CCtx size is supported for single-threaded compression only. */
    if (params->nbWorkers > 0) { return ERROR(GENERIC); }
    {   ZSTD_compressionParameters const cParams =
                ZSTD_getCParamsFromCCtxParams(params, 0, 0);
        size_t const blockSize = MIN(ZSTD_BLOCKSIZE_MAX, (size_t)1 << cParams.windowLog);
        U32    const divider = (cParams.searchLength==3) ? 3 : 4;
        size_t const maxNbSeq = blockSize / divider;
        size_t const tokenSpace = blockSize + 11*maxNbSeq;
        size_t const entropySpace = HUF_WORKSPACE_SIZE;
        size_t const blockStateSpace = 2 * sizeof(ZSTD_compressedBlockState_t);
        size_t const matchStateSize = ZSTD_sizeof_matchState(&cParams, /* forCCtx */ 1);

        size_t const ldmSpace = ZSTD_ldm_getTableSize(params->ldmParams);
        size_t const ldmSeqSpace = ZSTD_ldm_getMaxNbSeq(params->ldmParams, blockSize) * sizeof(rawSeq);

        size_t const neededSpace = entropySpace + blockStateSpace + tokenSpace +
                                   matchStateSize + ldmSpace + ldmSeqSpace;

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

static size_t ZSTD_estimateCCtxSize_internal(int compressionLevel)
{
    ZSTD_compressionParameters const cParams = ZSTD_getCParams(compressionLevel, 0, 0);
    return ZSTD_estimateCCtxSize_usingCParams(cParams);
}

size_t ZSTD_estimateCCtxSize(int compressionLevel)
{
    int level;
    size_t memBudget = 0;
    for (level=1; level<=compressionLevel; level++) {
        size_t const newMB = ZSTD_estimateCCtxSize_internal(level);
        if (newMB > memBudget) memBudget = newMB;
    }
    return memBudget;
}

size_t ZSTD_estimateCStreamSize_usingCCtxParams(const ZSTD_CCtx_params* params)
{
    if (params->nbWorkers > 0) { return ERROR(GENERIC); }
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

static size_t ZSTD_estimateCStreamSize_internal(int compressionLevel) {
    ZSTD_compressionParameters const cParams = ZSTD_getCParams(compressionLevel, 0, 0);
    return ZSTD_estimateCStreamSize_usingCParams(cParams);
}

size_t ZSTD_estimateCStreamSize(int compressionLevel) {
    int level;
    size_t memBudget = 0;
    for (level=1; level<=compressionLevel; level++) {
        size_t const newMB = ZSTD_estimateCStreamSize_internal(level);
        if (newMB > memBudget) memBudget = newMB;
    }
    return memBudget;
}

/* ZSTD_getFrameProgression():
 * tells how much data has been consumed (input) and produced (output) for current frame.
 * able to count progression inside worker threads (non-blocking mode).
 */
ZSTD_frameProgression ZSTD_getFrameProgression(const ZSTD_CCtx* cctx)
{
#ifdef ZSTD_MULTITHREAD
    if (cctx->appliedParams.nbWorkers > 0) {
        return ZSTDMT_getFrameProgression(cctx->mtctx);
    }
#endif
    {   ZSTD_frameProgression fp;
        size_t const buffered = (cctx->inBuff == NULL) ? 0 :
                                cctx->inBuffPos - cctx->inToCompress;
        if (buffered) assert(cctx->inBuffPos >= cctx->inToCompress);
        assert(buffered <= ZSTD_BLOCKSIZE_MAX);
        fp.ingested = cctx->consumedSrcSize + buffered;
        fp.consumed = cctx->consumedSrcSize;
        fp.produced = cctx->producedCSize;
        return fp;
}   }


static U32 ZSTD_equivalentCParams(ZSTD_compressionParameters cParams1,
                                  ZSTD_compressionParameters cParams2)
{
    return (cParams1.hashLog  == cParams2.hashLog)
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

typedef enum { ZSTDb_not_buffered, ZSTDb_buffered } ZSTD_buffered_policy_e;

/* ZSTD_sufficientBuff() :
 * check internal buffers exist for streaming if buffPol == ZSTDb_buffered .
 * Note : they are assumed to be correctly sized if ZSTD_equivalentCParams()==1 */
static U32 ZSTD_sufficientBuff(size_t bufferSize1, size_t blockSize1,
                            ZSTD_buffered_policy_e buffPol2,
                            ZSTD_compressionParameters cParams2,
                            U64 pledgedSrcSize)
{
    size_t const windowSize2 = MAX(1, (size_t)MIN(((U64)1 << cParams2.windowLog), pledgedSrcSize));
    size_t const blockSize2 = MIN(ZSTD_BLOCKSIZE_MAX, windowSize2);
    size_t const neededBufferSize2 = (buffPol2==ZSTDb_buffered) ? windowSize2 + blockSize2 : 0;
    DEBUGLOG(4, "ZSTD_sufficientBuff: is windowSize2=%u <= wlog1=%u",
                (U32)windowSize2, cParams2.windowLog);
    DEBUGLOG(4, "ZSTD_sufficientBuff: is blockSize2=%u <= blockSize1=%u",
                (U32)blockSize2, (U32)blockSize1);
    return (blockSize2 <= blockSize1) /* seqStore space depends on blockSize */
         & (neededBufferSize2 <= bufferSize1);
}

/** Equivalence for resetCCtx purposes */
static U32 ZSTD_equivalentParams(ZSTD_CCtx_params params1,
                                 ZSTD_CCtx_params params2,
                                 size_t buffSize1, size_t blockSize1,
                                 ZSTD_buffered_policy_e buffPol2,
                                 U64 pledgedSrcSize)
{
    DEBUGLOG(4, "ZSTD_equivalentParams: pledgedSrcSize=%u", (U32)pledgedSrcSize);
    return ZSTD_equivalentCParams(params1.cParams, params2.cParams) &&
           ZSTD_equivalentLdmParams(params1.ldmParams, params2.ldmParams) &&
           ZSTD_sufficientBuff(buffSize1, blockSize1, buffPol2, params2.cParams, pledgedSrcSize);
}

static void ZSTD_reset_compressedBlockState(ZSTD_compressedBlockState_t* bs)
{
    int i;
    for (i = 0; i < ZSTD_REP_NUM; ++i)
        bs->rep[i] = repStartValue[i];
    bs->entropy.hufCTable_repeatMode = HUF_repeat_none;
    bs->entropy.offcode_repeatMode = FSE_repeat_none;
    bs->entropy.matchlength_repeatMode = FSE_repeat_none;
    bs->entropy.litlength_repeatMode = FSE_repeat_none;
}

/*! ZSTD_invalidateMatchState()
 * Invalidate all the matches in the match finder tables.
 * Requires nextSrc and base to be set (can be NULL).
 */
static void ZSTD_invalidateMatchState(ZSTD_matchState_t* ms)
{
    ZSTD_window_clear(&ms->window);

    ms->nextToUpdate = ms->window.dictLimit + 1;
    ms->loadedDictEnd = 0;
    ms->opt.litLengthSum = 0;  /* force reset of btopt stats */
}

/*! ZSTD_continueCCtx() :
 *  reuse CCtx without reset (note : requires no dictionary) */
static size_t ZSTD_continueCCtx(ZSTD_CCtx* cctx, ZSTD_CCtx_params params, U64 pledgedSrcSize)
{
    size_t const windowSize = MAX(1, (size_t)MIN(((U64)1 << params.cParams.windowLog), pledgedSrcSize));
    size_t const blockSize = MIN(ZSTD_BLOCKSIZE_MAX, windowSize);
    DEBUGLOG(4, "ZSTD_continueCCtx: re-use context in place");

    cctx->blockSize = blockSize;   /* previous block size could be different even for same windowLog, due to pledgedSrcSize */
    cctx->appliedParams = params;
    cctx->pledgedSrcSizePlusOne = pledgedSrcSize+1;
    cctx->consumedSrcSize = 0;
    cctx->producedCSize = 0;
    if (pledgedSrcSize == ZSTD_CONTENTSIZE_UNKNOWN)
        cctx->appliedParams.fParams.contentSizeFlag = 0;
    DEBUGLOG(4, "pledged content size : %u ; flag : %u",
        (U32)pledgedSrcSize, cctx->appliedParams.fParams.contentSizeFlag);
    cctx->stage = ZSTDcs_init;
    cctx->dictID = 0;
    if (params.ldmParams.enableLdm)
        ZSTD_window_clear(&cctx->ldmState.window);
    ZSTD_referenceExternalSequences(cctx, NULL, 0);
    ZSTD_invalidateMatchState(&cctx->blockState.matchState);
    ZSTD_reset_compressedBlockState(cctx->blockState.prevCBlock);
    XXH64_reset(&cctx->xxhState, 0);
    return 0;
}

typedef enum { ZSTDcrp_continue, ZSTDcrp_noMemset } ZSTD_compResetPolicy_e;

static void* ZSTD_reset_matchState(ZSTD_matchState_t* ms, void* ptr, ZSTD_compressionParameters const* cParams, ZSTD_compResetPolicy_e const crp, U32 const forCCtx)
{
    size_t const chainSize = (cParams->strategy == ZSTD_fast) ? 0 : ((size_t)1 << cParams->chainLog);
    size_t const hSize = ((size_t)1) << cParams->hashLog;
    U32    const hashLog3 = (forCCtx && cParams->searchLength==3) ? MIN(ZSTD_HASHLOG3_MAX, cParams->windowLog) : 0;
    size_t const h3Size = ((size_t)1) << hashLog3;
    size_t const tableSpace = (chainSize + hSize + h3Size) * sizeof(U32);

    assert(((size_t)ptr & 3) == 0);

    ms->hashLog3 = hashLog3;
    memset(&ms->window, 0, sizeof(ms->window));
    ZSTD_invalidateMatchState(ms);

    /* opt parser space */
    if (forCCtx && ((cParams->strategy == ZSTD_btopt) | (cParams->strategy == ZSTD_btultra))) {
        DEBUGLOG(4, "reserving optimal parser space");
        ms->opt.litFreq = (U32*)ptr;
        ms->opt.litLengthFreq = ms->opt.litFreq + (1<<Litbits);
        ms->opt.matchLengthFreq = ms->opt.litLengthFreq + (MaxLL+1);
        ms->opt.offCodeFreq = ms->opt.matchLengthFreq + (MaxML+1);
        ptr = ms->opt.offCodeFreq + (MaxOff+1);
        ms->opt.matchTable = (ZSTD_match_t*)ptr;
        ptr = ms->opt.matchTable + ZSTD_OPT_NUM+1;
        ms->opt.priceTable = (ZSTD_optimal_t*)ptr;
        ptr = ms->opt.priceTable + ZSTD_OPT_NUM+1;
    }

    /* table Space */
    DEBUGLOG(4, "reset table : %u", crp!=ZSTDcrp_noMemset);
    assert(((size_t)ptr & 3) == 0);  /* ensure ptr is properly aligned */
    if (crp!=ZSTDcrp_noMemset) memset(ptr, 0, tableSpace);   /* reset tables only */
    ms->hashTable = (U32*)(ptr);
    ms->chainTable = ms->hashTable + hSize;
    ms->hashTable3 = ms->chainTable + chainSize;
    ptr = ms->hashTable3 + h3Size;

    assert(((size_t)ptr & 3) == 0);
    return ptr;
}

/*! ZSTD_resetCCtx_internal() :
    note : `params` are assumed fully validated at this stage */
static size_t ZSTD_resetCCtx_internal(ZSTD_CCtx* zc,
                                      ZSTD_CCtx_params params, U64 pledgedSrcSize,
                                      ZSTD_compResetPolicy_e const crp,
                                      ZSTD_buffered_policy_e const zbuff)
{
    DEBUGLOG(4, "ZSTD_resetCCtx_internal: pledgedSrcSize=%u, wlog=%u",
                (U32)pledgedSrcSize, params.cParams.windowLog);
    assert(!ZSTD_isError(ZSTD_checkCParams(params.cParams)));

    if (crp == ZSTDcrp_continue) {
        if (ZSTD_equivalentParams(zc->appliedParams, params,
                                zc->inBuffSize, zc->blockSize,
                                zbuff, pledgedSrcSize)) {
            DEBUGLOG(4, "ZSTD_equivalentParams()==1 -> continue mode (wLog1=%u, blockSize1=%u)",
                        zc->appliedParams.cParams.windowLog, (U32)zc->blockSize);
            return ZSTD_continueCCtx(zc, params, pledgedSrcSize);
    }   }
    DEBUGLOG(4, "ZSTD_equivalentParams()==0 -> reset CCtx");

    if (params.ldmParams.enableLdm) {
        /* Adjust long distance matching parameters */
        params.ldmParams.windowLog = params.cParams.windowLog;
        ZSTD_ldm_adjustParameters(&params.ldmParams, &params.cParams);
        assert(params.ldmParams.hashLog >= params.ldmParams.bucketSizeLog);
        assert(params.ldmParams.hashEveryLog < 32);
        zc->ldmState.hashPower =
                ZSTD_ldm_getHashPower(params.ldmParams.minMatchLength);
    }

    {   size_t const windowSize = MAX(1, (size_t)MIN(((U64)1 << params.cParams.windowLog), pledgedSrcSize));
        size_t const blockSize = MIN(ZSTD_BLOCKSIZE_MAX, windowSize);
        U32    const divider = (params.cParams.searchLength==3) ? 3 : 4;
        size_t const maxNbSeq = blockSize / divider;
        size_t const tokenSpace = blockSize + 11*maxNbSeq;
        size_t const buffOutSize = (zbuff==ZSTDb_buffered) ? ZSTD_compressBound(blockSize)+1 : 0;
        size_t const buffInSize = (zbuff==ZSTDb_buffered) ? windowSize + blockSize : 0;
        size_t const matchStateSize = ZSTD_sizeof_matchState(&params.cParams, /* forCCtx */ 1);
        size_t const maxNbLdmSeq = ZSTD_ldm_getMaxNbSeq(params.ldmParams, blockSize);
        void* ptr;

        /* Check if workSpace is large enough, alloc a new one if needed */
        {   size_t const entropySpace = HUF_WORKSPACE_SIZE;
            size_t const blockStateSpace = 2 * sizeof(ZSTD_compressedBlockState_t);
            size_t const bufferSpace = buffInSize + buffOutSize;
            size_t const ldmSpace = ZSTD_ldm_getTableSize(params.ldmParams);
            size_t const ldmSeqSpace = maxNbLdmSeq * sizeof(rawSeq);

            size_t const neededSpace = entropySpace + blockStateSpace + ldmSpace +
                                       ldmSeqSpace + matchStateSize + tokenSpace +
                                       bufferSpace;
            DEBUGLOG(4, "Need %uKB workspace, including %uKB for match state, and %uKB for buffers",
                        (U32)(neededSpace>>10), (U32)(matchStateSize>>10), (U32)(bufferSpace>>10));
            DEBUGLOG(4, "windowSize: %u - blockSize: %u", (U32)windowSize, (U32)blockSize);

            if (zc->workSpaceSize < neededSpace) {  /* too small : resize */
                DEBUGLOG(4, "Need to update workSpaceSize from %uK to %uK",
                            (unsigned)(zc->workSpaceSize>>10),
                            (unsigned)(neededSpace>>10));
                /* static cctx : no resize, error out */
                if (zc->staticSize) return ERROR(memory_allocation);

                zc->workSpaceSize = 0;
                ZSTD_free(zc->workSpace, zc->customMem);
                zc->workSpace = ZSTD_malloc(neededSpace, zc->customMem);
                if (zc->workSpace == NULL) return ERROR(memory_allocation);
                zc->workSpaceSize = neededSpace;
                ptr = zc->workSpace;

                /* Statically sized space. entropyWorkspace never moves (but prev/next block swap places) */
                assert(((size_t)zc->workSpace & 3) == 0);   /* ensure correct alignment */
                assert(zc->workSpaceSize >= 2 * sizeof(ZSTD_compressedBlockState_t));
                zc->blockState.prevCBlock = (ZSTD_compressedBlockState_t*)zc->workSpace;
                zc->blockState.nextCBlock = zc->blockState.prevCBlock + 1;
                ptr = zc->blockState.nextCBlock + 1;
                zc->entropyWorkspace = (U32*)ptr;
        }   }

        /* init params */
        zc->appliedParams = params;
        zc->pledgedSrcSizePlusOne = pledgedSrcSize+1;
        zc->consumedSrcSize = 0;
        zc->producedCSize = 0;
        if (pledgedSrcSize == ZSTD_CONTENTSIZE_UNKNOWN)
            zc->appliedParams.fParams.contentSizeFlag = 0;
        DEBUGLOG(4, "pledged content size : %u ; flag : %u",
            (U32)pledgedSrcSize, zc->appliedParams.fParams.contentSizeFlag);
        zc->blockSize = blockSize;

        XXH64_reset(&zc->xxhState, 0);
        zc->stage = ZSTDcs_init;
        zc->dictID = 0;

        ZSTD_reset_compressedBlockState(zc->blockState.prevCBlock);

        ptr = zc->entropyWorkspace + HUF_WORKSPACE_SIZE_U32;

        /* ldm hash table */
        /* initialize bucketOffsets table later for pointer alignment */
        if (params.ldmParams.enableLdm) {
            size_t const ldmHSize = ((size_t)1) << params.ldmParams.hashLog;
            memset(ptr, 0, ldmHSize * sizeof(ldmEntry_t));
            assert(((size_t)ptr & 3) == 0); /* ensure ptr is properly aligned */
            zc->ldmState.hashTable = (ldmEntry_t*)ptr;
            ptr = zc->ldmState.hashTable + ldmHSize;
            zc->ldmSequences = (rawSeq*)ptr;
            ptr = zc->ldmSequences + maxNbLdmSeq;
            zc->maxNbLdmSequences = maxNbLdmSeq;

            memset(&zc->ldmState.window, 0, sizeof(zc->ldmState.window));
        }
        assert(((size_t)ptr & 3) == 0); /* ensure ptr is properly aligned */

        ptr = ZSTD_reset_matchState(&zc->blockState.matchState, ptr, &params.cParams, crp, /* forCCtx */ 1);

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
            ZSTD_window_clear(&zc->ldmState.window);
        }
        ZSTD_referenceExternalSequences(zc, NULL, 0);

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
    for (i=0; i<ZSTD_REP_NUM; i++) cctx->blockState.prevCBlock->rep[i] = 0;
    assert(!ZSTD_window_hasExtDict(cctx->blockState.matchState.window));
}

static size_t ZSTD_resetCCtx_usingCDict(ZSTD_CCtx* cctx,
                            const ZSTD_CDict* cdict,
                            unsigned windowLog,
                            ZSTD_frameParameters fParams,
                            U64 pledgedSrcSize,
                            ZSTD_buffered_policy_e zbuff)
{
    {   ZSTD_CCtx_params params = cctx->requestedParams;
        /* Copy only compression parameters related to tables. */
        params.cParams = cdict->cParams;
        if (windowLog) params.cParams.windowLog = windowLog;
        params.fParams = fParams;
        ZSTD_resetCCtx_internal(cctx, params, pledgedSrcSize,
                                ZSTDcrp_noMemset, zbuff);
        assert(cctx->appliedParams.cParams.strategy == cdict->cParams.strategy);
        assert(cctx->appliedParams.cParams.hashLog == cdict->cParams.hashLog);
        assert(cctx->appliedParams.cParams.chainLog == cdict->cParams.chainLog);
    }

    /* copy tables */
    {   size_t const chainSize = (cdict->cParams.strategy == ZSTD_fast) ? 0 : ((size_t)1 << cdict->cParams.chainLog);
        size_t const hSize =  (size_t)1 << cdict->cParams.hashLog;
        size_t const tableSpace = (chainSize + hSize) * sizeof(U32);
        assert((U32*)cctx->blockState.matchState.chainTable == (U32*)cctx->blockState.matchState.hashTable + hSize);  /* chainTable must follow hashTable */
        assert((U32*)cctx->blockState.matchState.hashTable3 == (U32*)cctx->blockState.matchState.chainTable + chainSize);
        assert((U32*)cdict->matchState.chainTable == (U32*)cdict->matchState.hashTable + hSize);  /* chainTable must follow hashTable */
        assert((U32*)cdict->matchState.hashTable3 == (U32*)cdict->matchState.chainTable + chainSize);
        memcpy(cctx->blockState.matchState.hashTable, cdict->matchState.hashTable, tableSpace);   /* presumes all tables follow each other */
    }
    /* Zero the hashTable3, since the cdict never fills it */
    {   size_t const h3Size = (size_t)1 << cctx->blockState.matchState.hashLog3;
        assert(cdict->matchState.hashLog3 == 0);
        memset(cctx->blockState.matchState.hashTable3, 0, h3Size * sizeof(U32));
    }

    /* copy dictionary offsets */
    {
        ZSTD_matchState_t const* srcMatchState = &cdict->matchState;
        ZSTD_matchState_t* dstMatchState = &cctx->blockState.matchState;
        dstMatchState->window       = srcMatchState->window;
        dstMatchState->nextToUpdate = srcMatchState->nextToUpdate;
        dstMatchState->nextToUpdate3= srcMatchState->nextToUpdate3;
        dstMatchState->loadedDictEnd= srcMatchState->loadedDictEnd;
    }
    cctx->dictID = cdict->dictID;

    /* copy block state */
    memcpy(cctx->blockState.prevCBlock, &cdict->cBlockState, sizeof(cdict->cBlockState));

    return 0;
}

/*! ZSTD_copyCCtx_internal() :
 *  Duplicate an existing context `srcCCtx` into another one `dstCCtx`.
 *  Only works during stage ZSTDcs_init (i.e. after creation, but before first call to ZSTD_compressContinue()).
 *  The "context", in this case, refers to the hash and chain tables,
 *  entropy tables, and dictionary references.
 * `windowLog` value is enforced if != 0, otherwise value is copied from srcCCtx.
 * @return : 0, or an error code */
static size_t ZSTD_copyCCtx_internal(ZSTD_CCtx* dstCCtx,
                            const ZSTD_CCtx* srcCCtx,
                            ZSTD_frameParameters fParams,
                            U64 pledgedSrcSize,
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
        assert(dstCCtx->appliedParams.cParams.windowLog == srcCCtx->appliedParams.cParams.windowLog);
        assert(dstCCtx->appliedParams.cParams.strategy == srcCCtx->appliedParams.cParams.strategy);
        assert(dstCCtx->appliedParams.cParams.hashLog == srcCCtx->appliedParams.cParams.hashLog);
        assert(dstCCtx->appliedParams.cParams.chainLog == srcCCtx->appliedParams.cParams.chainLog);
        assert(dstCCtx->blockState.matchState.hashLog3 == srcCCtx->blockState.matchState.hashLog3);
    }

    /* copy tables */
    {   size_t const chainSize = (srcCCtx->appliedParams.cParams.strategy == ZSTD_fast) ? 0 : ((size_t)1 << srcCCtx->appliedParams.cParams.chainLog);
        size_t const hSize =  (size_t)1 << srcCCtx->appliedParams.cParams.hashLog;
        size_t const h3Size = (size_t)1 << srcCCtx->blockState.matchState.hashLog3;
        size_t const tableSpace = (chainSize + hSize + h3Size) * sizeof(U32);
        assert((U32*)dstCCtx->blockState.matchState.chainTable == (U32*)dstCCtx->blockState.matchState.hashTable + hSize);  /* chainTable must follow hashTable */
        assert((U32*)dstCCtx->blockState.matchState.hashTable3 == (U32*)dstCCtx->blockState.matchState.chainTable + chainSize);
        memcpy(dstCCtx->blockState.matchState.hashTable, srcCCtx->blockState.matchState.hashTable, tableSpace);   /* presumes all tables follow each other */
    }

    /* copy dictionary offsets */
    {
        ZSTD_matchState_t const* srcMatchState = &srcCCtx->blockState.matchState;
        ZSTD_matchState_t* dstMatchState = &dstCCtx->blockState.matchState;
        dstMatchState->window       = srcMatchState->window;
        dstMatchState->nextToUpdate = srcMatchState->nextToUpdate;
        dstMatchState->nextToUpdate3= srcMatchState->nextToUpdate3;
        dstMatchState->loadedDictEnd= srcMatchState->loadedDictEnd;
    }
    dstCCtx->dictID = srcCCtx->dictID;

    /* copy block state */
    memcpy(dstCCtx->blockState.prevCBlock, srcCCtx->blockState.prevCBlock, sizeof(*srcCCtx->blockState.prevCBlock));

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
    if (pledgedSrcSize==0) pledgedSrcSize = ZSTD_CONTENTSIZE_UNKNOWN;
    fParams.contentSizeFlag = (pledgedSrcSize != ZSTD_CONTENTSIZE_UNKNOWN);

    return ZSTD_copyCCtx_internal(dstCCtx, srcCCtx,
                                fParams, pledgedSrcSize,
                                zbuff);
}


#define ZSTD_ROWSIZE 16
/*! ZSTD_reduceTable() :
 *  reduce table indexes by `reducerValue`, or squash to zero.
 *  PreserveMark preserves "unsorted mark" for btlazy2 strategy.
 *  It must be set to a clear 0/1 value, to remove branch during inlining.
 *  Presume table size is a multiple of ZSTD_ROWSIZE
 *  to help auto-vectorization */
FORCE_INLINE_TEMPLATE void
ZSTD_reduceTable_internal (U32* const table, U32 const size, U32 const reducerValue, int const preserveMark)
{
    int const nbRows = (int)size / ZSTD_ROWSIZE;
    int cellNb = 0;
    int rowNb;
    assert((size & (ZSTD_ROWSIZE-1)) == 0);  /* multiple of ZSTD_ROWSIZE */
    assert(size < (1U<<31));   /* can be casted to int */
    for (rowNb=0 ; rowNb < nbRows ; rowNb++) {
        int column;
        for (column=0; column<ZSTD_ROWSIZE; column++) {
            if (preserveMark) {
                U32 const adder = (table[cellNb] == ZSTD_DUBT_UNSORTED_MARK) ? reducerValue : 0;
                table[cellNb] += adder;
            }
            if (table[cellNb] < reducerValue) table[cellNb] = 0;
            else table[cellNb] -= reducerValue;
            cellNb++;
    }   }
}

static void ZSTD_reduceTable(U32* const table, U32 const size, U32 const reducerValue)
{
    ZSTD_reduceTable_internal(table, size, reducerValue, 0);
}

static void ZSTD_reduceTable_btlazy2(U32* const table, U32 const size, U32 const reducerValue)
{
    ZSTD_reduceTable_internal(table, size, reducerValue, 1);
}

/*! ZSTD_reduceIndex() :
*   rescale all indexes to avoid future overflow (indexes are U32) */
static void ZSTD_reduceIndex (ZSTD_CCtx* zc, const U32 reducerValue)
{
    ZSTD_matchState_t* const ms = &zc->blockState.matchState;
    {   U32 const hSize = (U32)1 << zc->appliedParams.cParams.hashLog;
        ZSTD_reduceTable(ms->hashTable, hSize, reducerValue);
    }

    if (zc->appliedParams.cParams.strategy != ZSTD_fast) {
        U32 const chainSize = (U32)1 << zc->appliedParams.cParams.chainLog;
        if (zc->appliedParams.cParams.strategy == ZSTD_btlazy2)
            ZSTD_reduceTable_btlazy2(ms->chainTable, chainSize, reducerValue);
        else
            ZSTD_reduceTable(ms->chainTable, chainSize, reducerValue);
    }

    if (ms->hashLog3) {
        U32 const h3Size = (U32)1 << ms->hashLog3;
        ZSTD_reduceTable(ms->hashTable3, h3Size, reducerValue);
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

static size_t ZSTD_compressLiterals (ZSTD_entropyCTables_t const* prevEntropy,
                                     ZSTD_entropyCTables_t* nextEntropy,
                                     ZSTD_strategy strategy, int disableLiteralCompression,
                                     void* dst, size_t dstCapacity,
                               const void* src, size_t srcSize,
                                     U32* workspace, const int bmi2)
{
    size_t const minGain = ZSTD_minGain(srcSize);
    size_t const lhSize = 3 + (srcSize >= 1 KB) + (srcSize >= 16 KB);
    BYTE*  const ostart = (BYTE*)dst;
    U32 singleStream = srcSize < 256;
    symbolEncodingType_e hType = set_compressed;
    size_t cLitSize;

    DEBUGLOG(5,"ZSTD_compressLiterals (disableLiteralCompression=%i)",
                disableLiteralCompression);

    /* Prepare nextEntropy assuming reusing the existing table */
    nextEntropy->hufCTable_repeatMode = prevEntropy->hufCTable_repeatMode;
    memcpy(nextEntropy->hufCTable, prevEntropy->hufCTable,
           sizeof(prevEntropy->hufCTable));

    if (disableLiteralCompression)
        return ZSTD_noCompressLiterals(dst, dstCapacity, src, srcSize);

    /* small ? don't even attempt compression (speed opt) */
#   define COMPRESS_LITERALS_SIZE_MIN 63
    {   size_t const minLitSize = (prevEntropy->hufCTable_repeatMode == HUF_repeat_valid) ? 6 : COMPRESS_LITERALS_SIZE_MIN;
        if (srcSize <= minLitSize) return ZSTD_noCompressLiterals(dst, dstCapacity, src, srcSize);
    }

    if (dstCapacity < lhSize+1) return ERROR(dstSize_tooSmall);   /* not enough space for compression */
    {   HUF_repeat repeat = prevEntropy->hufCTable_repeatMode;
        int const preferRepeat = strategy < ZSTD_lazy ? srcSize <= 1024 : 0;
        if (repeat == HUF_repeat_valid && lhSize == 3) singleStream = 1;
        cLitSize = singleStream ? HUF_compress1X_repeat(ostart+lhSize, dstCapacity-lhSize, src, srcSize, 255, 11,
                                      workspace, HUF_WORKSPACE_SIZE, (HUF_CElt*)nextEntropy->hufCTable, &repeat, preferRepeat, bmi2)
                                : HUF_compress4X_repeat(ostart+lhSize, dstCapacity-lhSize, src, srcSize, 255, 11,
                                      workspace, HUF_WORKSPACE_SIZE, (HUF_CElt*)nextEntropy->hufCTable, &repeat, preferRepeat, bmi2);
        if (repeat != HUF_repeat_none) {
            /* reused the existing table */
            hType = set_repeat;
        }
    }

    if ((cLitSize==0) | (cLitSize >= srcSize - minGain) | ERR_isError(cLitSize)) {
        memcpy(nextEntropy->hufCTable, prevEntropy->hufCTable, sizeof(prevEntropy->hufCTable));
        return ZSTD_noCompressLiterals(dst, dstCapacity, src, srcSize);
    }
    if (cLitSize==1) {
        memcpy(nextEntropy->hufCTable, prevEntropy->hufCTable, sizeof(prevEntropy->hufCTable));
        return ZSTD_compressRleLiteralsBlock(dst, dstCapacity, src, srcSize);
    }

    if (hType == set_compressed) {
        /* using a newly constructed table */
        nextEntropy->hufCTable_repeatMode = HUF_repeat_check;
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
    default:  /* not possible : lhSize is {3,4,5} */
        assert(0);
    }
    return lhSize+cLitSize;
}


void ZSTD_seqToCodes(const seqStore_t* seqStorePtr)
{
    const seqDef* const sequences = seqStorePtr->sequencesStart;
    BYTE* const llCodeTable = seqStorePtr->llCode;
    BYTE* const ofCodeTable = seqStorePtr->ofCode;
    BYTE* const mlCodeTable = seqStorePtr->mlCode;
    U32 const nbSeq = (U32)(seqStorePtr->sequences - seqStorePtr->sequencesStart);
    U32 u;
    for (u=0; u<nbSeq; u++) {
        U32 const llv = sequences[u].litLength;
        U32 const mlv = sequences[u].matchLength;
        llCodeTable[u] = (BYTE)ZSTD_LLcode(llv);
        ofCodeTable[u] = (BYTE)ZSTD_highbit32(sequences[u].offset);
        mlCodeTable[u] = (BYTE)ZSTD_MLcode(mlv);
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

MEM_STATIC
symbolEncodingType_e ZSTD_selectEncodingType(
        FSE_repeat* repeatMode, size_t const mostFrequent, size_t nbSeq,
        U32 defaultNormLog, ZSTD_defaultPolicy_e const isDefaultAllowed)
{
#define MIN_SEQ_FOR_DYNAMIC_FSE   64
#define MAX_SEQ_FOR_STATIC_FSE  1000
    ZSTD_STATIC_ASSERT(ZSTD_defaultDisallowed == 0 && ZSTD_defaultAllowed != 0);
    if ((mostFrequent == nbSeq) && (!isDefaultAllowed || nbSeq > 2)) {
        DEBUGLOG(5, "Selected set_rle");
        /* Prefer set_basic over set_rle when there are 2 or less symbols,
         * since RLE uses 1 byte, but set_basic uses 5-6 bits per symbol.
         * If basic encoding isn't possible, always choose RLE.
         */
        *repeatMode = FSE_repeat_check;
        return set_rle;
    }
    if ( isDefaultAllowed
      && (*repeatMode == FSE_repeat_valid) && (nbSeq < MAX_SEQ_FOR_STATIC_FSE)) {
        DEBUGLOG(5, "Selected set_repeat");
        return set_repeat;
    }
    if ( isDefaultAllowed
      && ((nbSeq < MIN_SEQ_FOR_DYNAMIC_FSE) || (mostFrequent < (nbSeq >> (defaultNormLog-1)))) ) {
        DEBUGLOG(5, "Selected set_basic");
        /* The format allows default tables to be repeated, but it isn't useful.
         * When using simple heuristics to select encoding type, we don't want
         * to confuse these tables with dictionaries. When running more careful
         * analysis, we don't need to waste time checking both repeating tables
         * and default tables.
         */
        *repeatMode = FSE_repeat_none;
        return set_basic;
    }
    DEBUGLOG(5, "Selected set_compressed");
    *repeatMode = FSE_repeat_check;
    return set_compressed;
}

MEM_STATIC
size_t ZSTD_buildCTable(void* dst, size_t dstCapacity,
        FSE_CTable* nextCTable, U32 FSELog, symbolEncodingType_e type,
        U32* count, U32 max,
        BYTE const* codeTable, size_t nbSeq,
        S16 const* defaultNorm, U32 defaultNormLog, U32 defaultMax,
        FSE_CTable const* prevCTable, size_t prevCTableSize,
        void* workspace, size_t workspaceSize)
{
    BYTE* op = (BYTE*)dst;
    BYTE const* const oend = op + dstCapacity;

    switch (type) {
    case set_rle:
        *op = codeTable[0];
        CHECK_F(FSE_buildCTable_rle(nextCTable, (BYTE)max));
        return 1;
    case set_repeat:
        memcpy(nextCTable, prevCTable, prevCTableSize);
        return 0;
    case set_basic:
        CHECK_F(FSE_buildCTable_wksp(nextCTable, defaultNorm, defaultMax, defaultNormLog, workspace, workspaceSize));  /* note : could be pre-calculated */
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
            CHECK_F(FSE_buildCTable_wksp(nextCTable, norm, max, tableLog, workspace, workspaceSize));
            return NCountSize;
        }
    }
    default: return assert(0), ERROR(GENERIC);
    }
}

FORCE_INLINE_TEMPLATE size_t
ZSTD_encodeSequences_body(
            void* dst, size_t dstCapacity,
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
            U32  const ofBits = ofCode;
            U32  const mlBits = ML_bits[mlCode];
            DEBUGLOG(6, "encoding: litlen:%2u - matchlen:%2u - offCode:%7u",
                        sequences[n].litLength,
                        sequences[n].matchLength + MINMATCH,
                        sequences[n].offset);
                                                                            /* 32b*/  /* 64b*/
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

    DEBUGLOG(6, "ZSTD_encodeSequences: flushing ML state with %u bits", stateMatchLength.stateLog);
    FSE_flushCState(&blockStream, &stateMatchLength);
    DEBUGLOG(6, "ZSTD_encodeSequences: flushing Off state with %u bits", stateOffsetBits.stateLog);
    FSE_flushCState(&blockStream, &stateOffsetBits);
    DEBUGLOG(6, "ZSTD_encodeSequences: flushing LL state with %u bits", stateLitLength.stateLog);
    FSE_flushCState(&blockStream, &stateLitLength);

    {   size_t const streamSize = BIT_closeCStream(&blockStream);
        if (streamSize==0) return ERROR(dstSize_tooSmall);   /* not enough space */
        return streamSize;
    }
}

static size_t
ZSTD_encodeSequences_default(
            void* dst, size_t dstCapacity,
            FSE_CTable const* CTable_MatchLength, BYTE const* mlCodeTable,
            FSE_CTable const* CTable_OffsetBits, BYTE const* ofCodeTable,
            FSE_CTable const* CTable_LitLength, BYTE const* llCodeTable,
            seqDef const* sequences, size_t nbSeq, int longOffsets)
{
    return ZSTD_encodeSequences_body(dst, dstCapacity,
                                    CTable_MatchLength, mlCodeTable,
                                    CTable_OffsetBits, ofCodeTable,
                                    CTable_LitLength, llCodeTable,
                                    sequences, nbSeq, longOffsets);
}


#if DYNAMIC_BMI2

static TARGET_ATTRIBUTE("bmi2") size_t
ZSTD_encodeSequences_bmi2(
            void* dst, size_t dstCapacity,
            FSE_CTable const* CTable_MatchLength, BYTE const* mlCodeTable,
            FSE_CTable const* CTable_OffsetBits, BYTE const* ofCodeTable,
            FSE_CTable const* CTable_LitLength, BYTE const* llCodeTable,
            seqDef const* sequences, size_t nbSeq, int longOffsets)
{
    return ZSTD_encodeSequences_body(dst, dstCapacity,
                                    CTable_MatchLength, mlCodeTable,
                                    CTable_OffsetBits, ofCodeTable,
                                    CTable_LitLength, llCodeTable,
                                    sequences, nbSeq, longOffsets);
}

#endif

size_t ZSTD_encodeSequences(
            void* dst, size_t dstCapacity,
            FSE_CTable const* CTable_MatchLength, BYTE const* mlCodeTable,
            FSE_CTable const* CTable_OffsetBits, BYTE const* ofCodeTable,
            FSE_CTable const* CTable_LitLength, BYTE const* llCodeTable,
            seqDef const* sequences, size_t nbSeq, int longOffsets, int bmi2)
{
#if DYNAMIC_BMI2
    if (bmi2) {
        return ZSTD_encodeSequences_bmi2(dst, dstCapacity,
                                         CTable_MatchLength, mlCodeTable,
                                         CTable_OffsetBits, ofCodeTable,
                                         CTable_LitLength, llCodeTable,
                                         sequences, nbSeq, longOffsets);
    }
#endif
    (void)bmi2;
    return ZSTD_encodeSequences_default(dst, dstCapacity,
                                        CTable_MatchLength, mlCodeTable,
                                        CTable_OffsetBits, ofCodeTable,
                                        CTable_LitLength, llCodeTable,
                                        sequences, nbSeq, longOffsets);
}

MEM_STATIC size_t ZSTD_compressSequences_internal(seqStore_t* seqStorePtr,
                              ZSTD_entropyCTables_t const* prevEntropy,
                              ZSTD_entropyCTables_t* nextEntropy,
                              ZSTD_CCtx_params const* cctxParams,
                              void* dst, size_t dstCapacity, U32* workspace,
                              const int bmi2)
{
    const int longOffsets = cctxParams->cParams.windowLog > STREAM_ACCUMULATOR_MIN;
    U32 count[MaxSeq+1];
    FSE_CTable* CTable_LitLength = nextEntropy->litlengthCTable;
    FSE_CTable* CTable_OffsetBits = nextEntropy->offcodeCTable;
    FSE_CTable* CTable_MatchLength = nextEntropy->matchlengthCTable;
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

    ZSTD_STATIC_ASSERT(HUF_WORKSPACE_SIZE >= (1<<MAX(MLFSELog,LLFSELog)));

    /* Compress literals */
    {   const BYTE* const literals = seqStorePtr->litStart;
        size_t const litSize = seqStorePtr->lit - literals;
        size_t const cSize = ZSTD_compressLiterals(
                                    prevEntropy, nextEntropy,
                                    cctxParams->cParams.strategy, cctxParams->disableLiteralCompression,
                                    op, dstCapacity,
                                    literals, litSize,
                                    workspace, bmi2);
        if (ZSTD_isError(cSize))
          return cSize;
        assert(cSize <= dstCapacity);
        op += cSize;
    }

    /* Sequences Header */
    if ((oend-op) < 3 /*max nbSeq Size*/ + 1 /*seqHead*/) return ERROR(dstSize_tooSmall);
    if (nbSeq < 0x7F)
        *op++ = (BYTE)nbSeq;
    else if (nbSeq < LONGNBSEQ)
        op[0] = (BYTE)((nbSeq>>8) + 0x80), op[1] = (BYTE)nbSeq, op+=2;
    else
        op[0]=0xFF, MEM_writeLE16(op+1, (U16)(nbSeq - LONGNBSEQ)), op+=3;
    if (nbSeq==0) {
      memcpy(nextEntropy->litlengthCTable, prevEntropy->litlengthCTable, sizeof(prevEntropy->litlengthCTable));
      nextEntropy->litlength_repeatMode = prevEntropy->litlength_repeatMode;
      memcpy(nextEntropy->offcodeCTable, prevEntropy->offcodeCTable, sizeof(prevEntropy->offcodeCTable));
      nextEntropy->offcode_repeatMode = prevEntropy->offcode_repeatMode;
      memcpy(nextEntropy->matchlengthCTable, prevEntropy->matchlengthCTable, sizeof(prevEntropy->matchlengthCTable));
      nextEntropy->matchlength_repeatMode = prevEntropy->matchlength_repeatMode;
      return op - ostart;
    }

    /* seqHead : flags for FSE encoding type */
    seqHead = op++;

    /* convert length/distances into codes */
    ZSTD_seqToCodes(seqStorePtr);
    /* build CTable for Literal Lengths */
    {   U32 max = MaxLL;
        size_t const mostFrequent = FSE_countFast_wksp(count, &max, llCodeTable, nbSeq, workspace);
        DEBUGLOG(5, "Building LL table");
        nextEntropy->litlength_repeatMode = prevEntropy->litlength_repeatMode;
        LLtype = ZSTD_selectEncodingType(&nextEntropy->litlength_repeatMode, mostFrequent, nbSeq, LL_defaultNormLog, ZSTD_defaultAllowed);
        {   size_t const countSize = ZSTD_buildCTable(op, oend - op, CTable_LitLength, LLFSELog, (symbolEncodingType_e)LLtype,
                    count, max, llCodeTable, nbSeq, LL_defaultNorm, LL_defaultNormLog, MaxLL,
                    prevEntropy->litlengthCTable, sizeof(prevEntropy->litlengthCTable),
                    workspace, HUF_WORKSPACE_SIZE);
            if (ZSTD_isError(countSize)) return countSize;
            op += countSize;
    }   }
    /* build CTable for Offsets */
    {   U32 max = MaxOff;
        size_t const mostFrequent = FSE_countFast_wksp(count, &max, ofCodeTable, nbSeq, workspace);
        /* We can only use the basic table if max <= DefaultMaxOff, otherwise the offsets are too large */
        ZSTD_defaultPolicy_e const defaultPolicy = (max <= DefaultMaxOff) ? ZSTD_defaultAllowed : ZSTD_defaultDisallowed;
        DEBUGLOG(5, "Building OF table");
        nextEntropy->offcode_repeatMode = prevEntropy->offcode_repeatMode;
        Offtype = ZSTD_selectEncodingType(&nextEntropy->offcode_repeatMode, mostFrequent, nbSeq, OF_defaultNormLog, defaultPolicy);
        {   size_t const countSize = ZSTD_buildCTable(op, oend - op, CTable_OffsetBits, OffFSELog, (symbolEncodingType_e)Offtype,
                    count, max, ofCodeTable, nbSeq, OF_defaultNorm, OF_defaultNormLog, DefaultMaxOff,
                    prevEntropy->offcodeCTable, sizeof(prevEntropy->offcodeCTable),
                    workspace, HUF_WORKSPACE_SIZE);
            if (ZSTD_isError(countSize)) return countSize;
            op += countSize;
    }   }
    /* build CTable for MatchLengths */
    {   U32 max = MaxML;
        size_t const mostFrequent = FSE_countFast_wksp(count, &max, mlCodeTable, nbSeq, workspace);
        DEBUGLOG(5, "Building ML table");
        nextEntropy->matchlength_repeatMode = prevEntropy->matchlength_repeatMode;
        MLtype = ZSTD_selectEncodingType(&nextEntropy->matchlength_repeatMode, mostFrequent, nbSeq, ML_defaultNormLog, ZSTD_defaultAllowed);
        {   size_t const countSize = ZSTD_buildCTable(op, oend - op, CTable_MatchLength, MLFSELog, (symbolEncodingType_e)MLtype,
                    count, max, mlCodeTable, nbSeq, ML_defaultNorm, ML_defaultNormLog, MaxML,
                    prevEntropy->matchlengthCTable, sizeof(prevEntropy->matchlengthCTable),
                    workspace, HUF_WORKSPACE_SIZE);
            if (ZSTD_isError(countSize)) return countSize;
            op += countSize;
    }   }

    *seqHead = (BYTE)((LLtype<<6) + (Offtype<<4) + (MLtype<<2));

    {   size_t const bitstreamSize = ZSTD_encodeSequences(
                                        op, oend - op,
                                        CTable_MatchLength, mlCodeTable,
                                        CTable_OffsetBits, ofCodeTable,
                                        CTable_LitLength, llCodeTable,
                                        sequences, nbSeq,
                                        longOffsets, bmi2);
        if (ZSTD_isError(bitstreamSize)) return bitstreamSize;
        op += bitstreamSize;
    }

    return op - ostart;
}

MEM_STATIC size_t ZSTD_compressSequences(seqStore_t* seqStorePtr,
                              ZSTD_entropyCTables_t const* prevEntropy,
                              ZSTD_entropyCTables_t* nextEntropy,
                              ZSTD_CCtx_params const* cctxParams,
                              void* dst, size_t dstCapacity,
                              size_t srcSize, U32* workspace, int bmi2)
{
    size_t const cSize = ZSTD_compressSequences_internal(
            seqStorePtr, prevEntropy, nextEntropy, cctxParams, dst, dstCapacity,
            workspace, bmi2);
    /* When srcSize <= dstCapacity, there is enough space to write a raw uncompressed block.
     * Since we ran out of space, block must be not compressible, so fall back to raw uncompressed block.
     */
    if ((cSize == ERROR(dstSize_tooSmall)) & (srcSize <= dstCapacity))
        return 0;  /* block not compressed */
    if (ZSTD_isError(cSize)) return cSize;

    /* Check compressibility */
    {   size_t const maxCSize = srcSize - ZSTD_minGain(srcSize);  /* note : fixed formula, maybe should depend on compression level, or strategy */
        if (cSize >= maxCSize) return 0;  /* block not compressed */
    }

    /* We check that dictionaries have offset codes available for the first
     * block. After the first block, the offcode table might not have large
     * enough codes to represent the offsets in the data.
     */
    if (nextEntropy->offcode_repeatMode == FSE_repeat_valid)
        nextEntropy->offcode_repeatMode = FSE_repeat_check;

    return cSize;
}

/* ZSTD_selectBlockCompressor() :
 * Not static, but internal use only (used by long distance matcher)
 * assumption : strat is a valid strategy */
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

static void ZSTD_resetSeqStore(seqStore_t* ssPtr)
{
    ssPtr->lit = ssPtr->litStart;
    ssPtr->sequences = ssPtr->sequencesStart;
    ssPtr->longLengthID = 0;
}

static size_t ZSTD_compressBlock_internal(ZSTD_CCtx* zc,
                                        void* dst, size_t dstCapacity,
                                        const void* src, size_t srcSize)
{
    ZSTD_matchState_t* const ms = &zc->blockState.matchState;
    DEBUGLOG(5, "ZSTD_compressBlock_internal (dstCapacity=%u, dictLimit=%u, nextToUpdate=%u)",
                (U32)dstCapacity, ms->window.dictLimit, ms->nextToUpdate);
    if (srcSize < MIN_CBLOCK_SIZE+ZSTD_blockHeaderSize+1) {
        ZSTD_ldm_skipSequences(&zc->externSeqStore, srcSize, zc->appliedParams.cParams.searchLength);
        return 0;   /* don't even attempt compression below a certain srcSize */
    }
    ZSTD_resetSeqStore(&(zc->seqStore));

    /* limited update after a very long match */
    {   const BYTE* const base = ms->window.base;
        const BYTE* const istart = (const BYTE*)src;
        const U32 current = (U32)(istart-base);
        if (current > ms->nextToUpdate + 384)
            ms->nextToUpdate = current - MIN(192, (U32)(current - ms->nextToUpdate - 384));
    }

    /* select and store sequences */
    {   U32 const extDict = ZSTD_window_hasExtDict(ms->window);
        size_t lastLLSize;
        {   int i;
            for (i = 0; i < ZSTD_REP_NUM; ++i)
                zc->blockState.nextCBlock->rep[i] = zc->blockState.prevCBlock->rep[i];
        }
        if (zc->externSeqStore.pos < zc->externSeqStore.size) {
            assert(!zc->appliedParams.ldmParams.enableLdm);
            /* Updates ldmSeqStore.pos */
            lastLLSize =
                ZSTD_ldm_blockCompress(&zc->externSeqStore,
                                       ms, &zc->seqStore,
                                       zc->blockState.nextCBlock->rep,
                                       &zc->appliedParams.cParams,
                                       src, srcSize, extDict);
            assert(zc->externSeqStore.pos <= zc->externSeqStore.size);
        } else if (zc->appliedParams.ldmParams.enableLdm) {
            rawSeqStore_t ldmSeqStore = {NULL, 0, 0, 0};

            ldmSeqStore.seq = zc->ldmSequences;
            ldmSeqStore.capacity = zc->maxNbLdmSequences;
            /* Updates ldmSeqStore.size */
            CHECK_F(ZSTD_ldm_generateSequences(&zc->ldmState, &ldmSeqStore,
                                               &zc->appliedParams.ldmParams,
                                               src, srcSize));
            /* Updates ldmSeqStore.pos */
            lastLLSize =
                ZSTD_ldm_blockCompress(&ldmSeqStore,
                                       ms, &zc->seqStore,
                                       zc->blockState.nextCBlock->rep,
                                       &zc->appliedParams.cParams,
                                       src, srcSize, extDict);
            assert(ldmSeqStore.pos == ldmSeqStore.size);
        } else {   /* not long range mode */
            ZSTD_blockCompressor const blockCompressor = ZSTD_selectBlockCompressor(zc->appliedParams.cParams.strategy, extDict);
            lastLLSize = blockCompressor(ms, &zc->seqStore, zc->blockState.nextCBlock->rep, &zc->appliedParams.cParams, src, srcSize);
        }
        {   const BYTE* const lastLiterals = (const BYTE*)src + srcSize - lastLLSize;
            ZSTD_storeLastLiterals(&zc->seqStore, lastLiterals, lastLLSize);
    }   }

    /* encode sequences and literals */
    {   size_t const cSize = ZSTD_compressSequences(&zc->seqStore,
                                &zc->blockState.prevCBlock->entropy, &zc->blockState.nextCBlock->entropy,
                                &zc->appliedParams,
                                dst, dstCapacity,
                                srcSize, zc->entropyWorkspace, zc->bmi2);
        if (ZSTD_isError(cSize) || cSize == 0) return cSize;
        /* confirm repcodes and entropy tables */
        {   ZSTD_compressedBlockState_t* const tmp = zc->blockState.prevCBlock;
            zc->blockState.prevCBlock = zc->blockState.nextCBlock;
            zc->blockState.nextCBlock = tmp;
        }
        return cSize;
    }
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
    assert(cctx->appliedParams.cParams.windowLog <= 31);

    DEBUGLOG(5, "ZSTD_compress_frameChunk (blockSize=%u)", (U32)blockSize);
    if (cctx->appliedParams.fParams.checksumFlag && srcSize)
        XXH64_update(&cctx->xxhState, src, srcSize);

    while (remaining) {
        ZSTD_matchState_t* const ms = &cctx->blockState.matchState;
        U32 const lastBlock = lastFrameChunk & (blockSize >= remaining);

        if (dstCapacity < ZSTD_blockHeaderSize + MIN_CBLOCK_SIZE)
            return ERROR(dstSize_tooSmall);   /* not enough space to store compressed block */
        if (remaining < blockSize) blockSize = remaining;

        if (ZSTD_window_needOverflowCorrection(ms->window, ip + blockSize)) {
            U32 const cycleLog = ZSTD_cycleLog(cctx->appliedParams.cParams.chainLog, cctx->appliedParams.cParams.strategy);
            U32 const correction = ZSTD_window_correctOverflow(&ms->window, cycleLog, maxDist, ip);
            ZSTD_STATIC_ASSERT(ZSTD_CHAINLOG_MAX <= 30);
            ZSTD_STATIC_ASSERT(ZSTD_WINDOWLOG_MAX_32 <= 30);
            ZSTD_STATIC_ASSERT(ZSTD_WINDOWLOG_MAX <= 31);

            ZSTD_reduceIndex(cctx, correction);
            if (ms->nextToUpdate < correction) ms->nextToUpdate = 0;
            else ms->nextToUpdate -= correction;
            ms->loadedDictEnd = 0;
        }
        ZSTD_window_enforceMaxDist(&ms->window, ip + blockSize, maxDist, &ms->loadedDictEnd);
        if (ms->nextToUpdate < ms->window.lowLimit) ms->nextToUpdate = ms->window.lowLimit;

        {   size_t cSize = ZSTD_compressBlock_internal(cctx,
                                op+ZSTD_blockHeaderSize, dstCapacity-ZSTD_blockHeaderSize,
                                ip, blockSize);
            if (ZSTD_isError(cSize)) return cSize;

            if (cSize == 0) {  /* block is not compressible */
                U32 const cBlockHeader24 = lastBlock + (((U32)bt_raw)<<1) + (U32)(blockSize << 3);
                if (blockSize + ZSTD_blockHeaderSize > dstCapacity) return ERROR(dstSize_tooSmall);
                MEM_writeLE32(op, cBlockHeader24);   /* 4th byte will be overwritten */
                memcpy(op + ZSTD_blockHeaderSize, ip, blockSize);
                cSize = ZSTD_blockHeaderSize + blockSize;
            } else {
                U32 const cBlockHeader24 = lastBlock + (((U32)bt_compressed)<<1) + (U32)(cSize << 3);
                MEM_writeLE24(op, cBlockHeader24);
                cSize += ZSTD_blockHeaderSize;
            }

            ip += blockSize;
            assert(remaining >= blockSize);
            remaining -= blockSize;
            op += cSize;
            assert(dstCapacity >= cSize);
            dstCapacity -= cSize;
            DEBUGLOG(5, "ZSTD_compress_frameChunk: adding a block of size %u",
                        (U32)cSize);
    }   }

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

/* ZSTD_writeLastEmptyBlock() :
 * output an empty Block with end-of-frame mark to complete a frame
 * @return : size of data written into `dst` (== ZSTD_blockHeaderSize (defined in zstd_internal.h))
 *           or an error code if `dstCapcity` is too small (<ZSTD_blockHeaderSize)
 */
size_t ZSTD_writeLastEmptyBlock(void* dst, size_t dstCapacity)
{
    if (dstCapacity < ZSTD_blockHeaderSize) return ERROR(dstSize_tooSmall);
    {   U32 const cBlockHeader24 = 1 /*lastBlock*/ + (((U32)bt_raw)<<1);  /* 0 size */
        MEM_writeLE24(dst, cBlockHeader24);
        return ZSTD_blockHeaderSize;
    }
}

size_t ZSTD_referenceExternalSequences(ZSTD_CCtx* cctx, rawSeq* seq, size_t nbSeq)
{
    if (cctx->stage != ZSTDcs_init)
        return ERROR(stage_wrong);
    if (cctx->appliedParams.ldmParams.enableLdm)
        return ERROR(parameter_unsupported);
    cctx->externSeqStore.seq = seq;
    cctx->externSeqStore.size = nbSeq;
    cctx->externSeqStore.capacity = nbSeq;
    cctx->externSeqStore.pos = 0;
    return 0;
}


static size_t ZSTD_compressContinue_internal (ZSTD_CCtx* cctx,
                              void* dst, size_t dstCapacity,
                        const void* src, size_t srcSize,
                               U32 frame, U32 lastFrameChunk)
{
    ZSTD_matchState_t* ms = &cctx->blockState.matchState;
    size_t fhSize = 0;

    DEBUGLOG(5, "ZSTD_compressContinue_internal, stage: %u, srcSize: %u",
                cctx->stage, (U32)srcSize);
    if (cctx->stage==ZSTDcs_created) return ERROR(stage_wrong);   /* missing init (ZSTD_compressBegin) */

    if (frame && (cctx->stage==ZSTDcs_init)) {
        fhSize = ZSTD_writeFrameHeader(dst, dstCapacity, cctx->appliedParams,
                                       cctx->pledgedSrcSizePlusOne-1, cctx->dictID);
        if (ZSTD_isError(fhSize)) return fhSize;
        dstCapacity -= fhSize;
        dst = (char*)dst + fhSize;
        cctx->stage = ZSTDcs_ongoing;
    }

    if (!srcSize) return fhSize;  /* do not generate an empty block if no input */

    if (!ZSTD_window_update(&ms->window, src, srcSize)) {
        ms->nextToUpdate = ms->window.dictLimit;
    }
    if (cctx->appliedParams.ldmParams.enableLdm)
        ZSTD_window_update(&cctx->ldmState.window, src, srcSize);

    DEBUGLOG(5, "ZSTD_compressContinue_internal (blockSize=%u)", (U32)cctx->blockSize);
    {   size_t const cSize = frame ?
                             ZSTD_compress_frameChunk (cctx, dst, dstCapacity, src, srcSize, lastFrameChunk) :
                             ZSTD_compressBlock_internal (cctx, dst, dstCapacity, src, srcSize);
        if (ZSTD_isError(cSize)) return cSize;
        cctx->consumedSrcSize += srcSize;
        cctx->producedCSize += (cSize + fhSize);
        if (cctx->appliedParams.fParams.contentSizeFlag) {  /* control src size */
            if (cctx->consumedSrcSize+1 > cctx->pledgedSrcSizePlusOne) {
                DEBUGLOG(4, "error : pledgedSrcSize = %u, while realSrcSize >= %u",
                    (U32)cctx->pledgedSrcSizePlusOne-1, (U32)cctx->consumedSrcSize);
                return ERROR(srcSize_wrong);
            }
        }
        return cSize + fhSize;
    }
}

size_t ZSTD_compressContinue (ZSTD_CCtx* cctx,
                              void* dst, size_t dstCapacity,
                        const void* src, size_t srcSize)
{
    DEBUGLOG(5, "ZSTD_compressContinue (srcSize=%u)", (U32)srcSize);
    return ZSTD_compressContinue_internal(cctx, dst, dstCapacity, src, srcSize, 1 /* frame mode */, 0 /* last chunk */);
}


size_t ZSTD_getBlockSize(const ZSTD_CCtx* cctx)
{
    ZSTD_compressionParameters const cParams = cctx->appliedParams.cParams;
    assert(!ZSTD_checkCParams(cParams));
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
static size_t ZSTD_loadDictionaryContent(ZSTD_matchState_t* ms, ZSTD_CCtx_params const* params, const void* src, size_t srcSize)
{
    const BYTE* const ip = (const BYTE*) src;
    const BYTE* const iend = ip + srcSize;
    ZSTD_compressionParameters const* cParams = &params->cParams;

    ZSTD_window_update(&ms->window, src, srcSize);
    ms->loadedDictEnd = params->forceWindow ? 0 : (U32)(iend - ms->window.base);

    if (srcSize <= HASH_READ_SIZE) return 0;

    switch(params->cParams.strategy)
    {
    case ZSTD_fast:
        ZSTD_fillHashTable(ms, cParams, iend);
        break;
    case ZSTD_dfast:
        ZSTD_fillDoubleHashTable(ms, cParams, iend);
        break;

    case ZSTD_greedy:
    case ZSTD_lazy:
    case ZSTD_lazy2:
        if (srcSize >= HASH_READ_SIZE)
            ZSTD_insertAndFindFirstIndex(ms, cParams, iend-HASH_READ_SIZE);
        break;

    case ZSTD_btlazy2:   /* we want the dictionary table fully sorted */
    case ZSTD_btopt:
    case ZSTD_btultra:
        if (srcSize >= HASH_READ_SIZE)
            ZSTD_updateTree(ms, cParams, iend-HASH_READ_SIZE, iend);
        break;

    default:
        assert(0);  /* not possible : not a valid strategy id */
    }

    ms->nextToUpdate = (U32)(iend - ms->window.base);
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
 * @return : dictID, or an error code
 *  assumptions : magic number supposed already checked
 *                dictSize supposed > 8
 */
static size_t ZSTD_loadZstdDictionary(ZSTD_compressedBlockState_t* bs, ZSTD_matchState_t* ms, ZSTD_CCtx_params const* params, const void* dict, size_t dictSize, void* workspace)
{
    const BYTE* dictPtr = (const BYTE*)dict;
    const BYTE* const dictEnd = dictPtr + dictSize;
    short offcodeNCount[MaxOff+1];
    unsigned offcodeMaxValue = MaxOff;
    size_t dictID;

    ZSTD_STATIC_ASSERT(HUF_WORKSPACE_SIZE >= (1<<MAX(MLFSELog,LLFSELog)));

    dictPtr += 4;   /* skip magic number */
    dictID = params->fParams.noDictIDFlag ? 0 :  MEM_readLE32(dictPtr);
    dictPtr += 4;

    {   unsigned maxSymbolValue = 255;
        size_t const hufHeaderSize = HUF_readCTable((HUF_CElt*)bs->entropy.hufCTable, &maxSymbolValue, dictPtr, dictEnd-dictPtr);
        if (HUF_isError(hufHeaderSize)) return ERROR(dictionary_corrupted);
        if (maxSymbolValue < 255) return ERROR(dictionary_corrupted);
        dictPtr += hufHeaderSize;
    }

    {   unsigned offcodeLog;
        size_t const offcodeHeaderSize = FSE_readNCount(offcodeNCount, &offcodeMaxValue, &offcodeLog, dictPtr, dictEnd-dictPtr);
        if (FSE_isError(offcodeHeaderSize)) return ERROR(dictionary_corrupted);
        if (offcodeLog > OffFSELog) return ERROR(dictionary_corrupted);
        /* Defer checking offcodeMaxValue because we need to know the size of the dictionary content */
        CHECK_E( FSE_buildCTable_wksp(bs->entropy.offcodeCTable, offcodeNCount, offcodeMaxValue, offcodeLog, workspace, HUF_WORKSPACE_SIZE),
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
        CHECK_E( FSE_buildCTable_wksp(bs->entropy.matchlengthCTable, matchlengthNCount, matchlengthMaxValue, matchlengthLog, workspace, HUF_WORKSPACE_SIZE),
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
        CHECK_E( FSE_buildCTable_wksp(bs->entropy.litlengthCTable, litlengthNCount, litlengthMaxValue, litlengthLog, workspace, HUF_WORKSPACE_SIZE),
                 dictionary_corrupted);
        dictPtr += litlengthHeaderSize;
    }

    if (dictPtr+12 > dictEnd) return ERROR(dictionary_corrupted);
    bs->rep[0] = MEM_readLE32(dictPtr+0);
    bs->rep[1] = MEM_readLE32(dictPtr+4);
    bs->rep[2] = MEM_readLE32(dictPtr+8);
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
                if (bs->rep[u] == 0) return ERROR(dictionary_corrupted);
                if (bs->rep[u] > dictContentSize) return ERROR(dictionary_corrupted);
        }   }

        bs->entropy.hufCTable_repeatMode = HUF_repeat_valid;
        bs->entropy.offcode_repeatMode = FSE_repeat_valid;
        bs->entropy.matchlength_repeatMode = FSE_repeat_valid;
        bs->entropy.litlength_repeatMode = FSE_repeat_valid;
        CHECK_F(ZSTD_loadDictionaryContent(ms, params, dictPtr, dictContentSize));
        return dictID;
    }
}

/** ZSTD_compress_insertDictionary() :
*   @return : dictID, or an error code */
static size_t ZSTD_compress_insertDictionary(ZSTD_compressedBlockState_t* bs, ZSTD_matchState_t* ms,
                                             ZSTD_CCtx_params const* params,
                                       const void* dict, size_t dictSize,
                                             ZSTD_dictContentType_e dictContentType,
                                             void* workspace)
{
    DEBUGLOG(4, "ZSTD_compress_insertDictionary (dictSize=%u)", (U32)dictSize);
    if ((dict==NULL) || (dictSize<=8)) return 0;

    ZSTD_reset_compressedBlockState(bs);

    /* dict restricted modes */
    if (dictContentType == ZSTD_dct_rawContent)
        return ZSTD_loadDictionaryContent(ms, params, dict, dictSize);

    if (MEM_readLE32(dict) != ZSTD_MAGIC_DICTIONARY) {
        if (dictContentType == ZSTD_dct_auto) {
            DEBUGLOG(4, "raw content dictionary detected");
            return ZSTD_loadDictionaryContent(ms, params, dict, dictSize);
        }
        if (dictContentType == ZSTD_dct_fullDict)
            return ERROR(dictionary_wrong);
        assert(0);   /* impossible */
    }

    /* dict as full zstd dictionary */
    return ZSTD_loadZstdDictionary(bs, ms, params, dict, dictSize, workspace);
}

/*! ZSTD_compressBegin_internal() :
 * @return : 0, or an error code */
size_t ZSTD_compressBegin_internal(ZSTD_CCtx* cctx,
                             const void* dict, size_t dictSize,
                             ZSTD_dictContentType_e dictContentType,
                             const ZSTD_CDict* cdict,
                             ZSTD_CCtx_params params, U64 pledgedSrcSize,
                             ZSTD_buffered_policy_e zbuff)
{
    DEBUGLOG(4, "ZSTD_compressBegin_internal: wlog=%u", params.cParams.windowLog);
    /* params are supposed to be fully validated at this point */
    assert(!ZSTD_isError(ZSTD_checkCParams(params.cParams)));
    assert(!((dict) && (cdict)));  /* either dict or cdict, not both */

    if (cdict && cdict->dictContentSize>0) {
        cctx->requestedParams = params;
        return ZSTD_resetCCtx_usingCDict(cctx, cdict, params.cParams.windowLog,
                                         params.fParams, pledgedSrcSize, zbuff);
    }

    CHECK_F( ZSTD_resetCCtx_internal(cctx, params, pledgedSrcSize,
                                     ZSTDcrp_continue, zbuff) );
    {
        size_t const dictID = ZSTD_compress_insertDictionary(
                cctx->blockState.prevCBlock, &cctx->blockState.matchState,
                &params, dict, dictSize, dictContentType, cctx->entropyWorkspace);
        if (ZSTD_isError(dictID)) return dictID;
        assert(dictID <= (size_t)(U32)-1);
        cctx->dictID = (U32)dictID;
    }
    return 0;
}

size_t ZSTD_compressBegin_advanced_internal(ZSTD_CCtx* cctx,
                                    const void* dict, size_t dictSize,
                                    ZSTD_dictContentType_e dictContentType,
                                    const ZSTD_CDict* cdict,
                                    ZSTD_CCtx_params params,
                                    unsigned long long pledgedSrcSize)
{
    DEBUGLOG(4, "ZSTD_compressBegin_advanced_internal: wlog=%u", params.cParams.windowLog);
    /* compression parameters verification and optimization */
    CHECK_F( ZSTD_checkCParams(params.cParams) );
    return ZSTD_compressBegin_internal(cctx,
                                       dict, dictSize, dictContentType,
                                       cdict,
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
    return ZSTD_compressBegin_advanced_internal(cctx,
                                            dict, dictSize, ZSTD_dct_auto,
                                            NULL /*cdict*/,
                                            cctxParams, pledgedSrcSize);
}

size_t ZSTD_compressBegin_usingDict(ZSTD_CCtx* cctx, const void* dict, size_t dictSize, int compressionLevel)
{
    ZSTD_parameters const params = ZSTD_getParams(compressionLevel, ZSTD_CONTENTSIZE_UNKNOWN, dictSize);
    ZSTD_CCtx_params const cctxParams =
            ZSTD_assignParamsToCCtxParams(cctx->requestedParams, params);
    DEBUGLOG(4, "ZSTD_compressBegin_usingDict (dictSize=%u)", (U32)dictSize);
    return ZSTD_compressBegin_internal(cctx, dict, dictSize, ZSTD_dct_auto, NULL,
                                       cctxParams, ZSTD_CONTENTSIZE_UNKNOWN, ZSTDb_not_buffered);
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

    DEBUGLOG(4, "ZSTD_writeEpilogue");
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
        DEBUGLOG(4, "ZSTD_writeEpilogue: write checksum : %08X", checksum);
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
    DEBUGLOG(4, "ZSTD_compress_internal");
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
    DEBUGLOG(4, "ZSTD_compress_advanced");
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
    DEBUGLOG(4, "ZSTD_compress_advanced_internal (srcSize:%u)",
                (U32)srcSize);
    CHECK_F( ZSTD_compressBegin_internal(cctx, dict, dictSize, ZSTD_dct_auto, NULL,
                                         params, srcSize, ZSTDb_not_buffered) );
    return ZSTD_compressEnd(cctx, dst, dstCapacity, src, srcSize);
}

size_t ZSTD_compress_usingDict(ZSTD_CCtx* cctx, void* dst, size_t dstCapacity, const void* src, size_t srcSize,
                               const void* dict, size_t dictSize, int compressionLevel)
{
    ZSTD_parameters const params = ZSTD_getParams(compressionLevel, srcSize ? srcSize : 1, dict ? dictSize : 0);
    ZSTD_CCtx_params cctxParams = ZSTD_assignParamsToCCtxParams(cctx->requestedParams, params);
    assert(params.fParams.contentSizeFlag == 1);
    ZSTD_CCtxParam_setParameter(&cctxParams, ZSTD_p_compressLiterals, compressionLevel>=0);
    return ZSTD_compress_advanced_internal(cctx, dst, dstCapacity, src, srcSize, dict, dictSize, cctxParams);
}

size_t ZSTD_compressCCtx (ZSTD_CCtx* cctx, void* dst, size_t dstCapacity, const void* src, size_t srcSize, int compressionLevel)
{
    DEBUGLOG(4, "ZSTD_compressCCtx (srcSize=%u)", (U32)srcSize);
    return ZSTD_compress_usingDict(cctx, dst, dstCapacity, src, srcSize, NULL, 0, compressionLevel);
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
    return sizeof(ZSTD_CDict) + HUF_WORKSPACE_SIZE + ZSTD_sizeof_matchState(&cParams, /* forCCtx */ 0)
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
    return cdict->workspaceSize + (cdict->dictBuffer ? cdict->dictContentSize : 0) + sizeof(*cdict);
}

static size_t ZSTD_initCDict_internal(
                    ZSTD_CDict* cdict,
              const void* dictBuffer, size_t dictSize,
                    ZSTD_dictLoadMethod_e dictLoadMethod,
                    ZSTD_dictContentType_e dictContentType,
                    ZSTD_compressionParameters cParams)
{
    DEBUGLOG(3, "ZSTD_initCDict_internal, dictContentType %u", (U32)dictContentType);
    assert(!ZSTD_checkCParams(cParams));
    cdict->cParams = cParams;
    if ((dictLoadMethod == ZSTD_dlm_byRef) || (!dictBuffer) || (!dictSize)) {
        cdict->dictBuffer = NULL;
        cdict->dictContent = dictBuffer;
    } else {
        void* const internalBuffer = ZSTD_malloc(dictSize, cdict->customMem);
        cdict->dictBuffer = internalBuffer;
        cdict->dictContent = internalBuffer;
        if (!internalBuffer) return ERROR(memory_allocation);
        memcpy(internalBuffer, dictBuffer, dictSize);
    }
    cdict->dictContentSize = dictSize;

    /* Reset the state to no dictionary */
    ZSTD_reset_compressedBlockState(&cdict->cBlockState);
    {   void* const end = ZSTD_reset_matchState(
                &cdict->matchState,
                (U32*)cdict->workspace + HUF_WORKSPACE_SIZE_U32,
                &cParams, ZSTDcrp_continue, /* forCCtx */ 0);
        assert(end == (char*)cdict->workspace + cdict->workspaceSize);
        (void)end;
    }
    /* (Maybe) load the dictionary
     * Skips loading the dictionary if it is <= 8 bytes.
     */
    {   ZSTD_CCtx_params params;
        memset(&params, 0, sizeof(params));
        params.compressionLevel = ZSTD_CLEVEL_DEFAULT;
        params.fParams.contentSizeFlag = 1;
        params.cParams = cParams;
        {   size_t const dictID = ZSTD_compress_insertDictionary(
                    &cdict->cBlockState, &cdict->matchState, &params,
                    cdict->dictContent, cdict->dictContentSize,
                    dictContentType, cdict->workspace);
            if (ZSTD_isError(dictID)) return dictID;
            assert(dictID <= (size_t)(U32)-1);
            cdict->dictID = (U32)dictID;
        }
    }

    return 0;
}

ZSTD_CDict* ZSTD_createCDict_advanced(const void* dictBuffer, size_t dictSize,
                                      ZSTD_dictLoadMethod_e dictLoadMethod,
                                      ZSTD_dictContentType_e dictContentType,
                                      ZSTD_compressionParameters cParams, ZSTD_customMem customMem)
{
    DEBUGLOG(3, "ZSTD_createCDict_advanced, mode %u", (U32)dictContentType);
    if (!customMem.customAlloc ^ !customMem.customFree) return NULL;

    {   ZSTD_CDict* const cdict = (ZSTD_CDict*)ZSTD_malloc(sizeof(ZSTD_CDict), customMem);
        size_t const workspaceSize = HUF_WORKSPACE_SIZE + ZSTD_sizeof_matchState(&cParams, /* forCCtx */ 0);
        void* const workspace = ZSTD_malloc(workspaceSize, customMem);

        if (!cdict || !workspace) {
            ZSTD_free(cdict, customMem);
            ZSTD_free(workspace, customMem);
            return NULL;
        }
        cdict->customMem = customMem;
        cdict->workspace = workspace;
        cdict->workspaceSize = workspaceSize;
        if (ZSTD_isError( ZSTD_initCDict_internal(cdict,
                                        dictBuffer, dictSize,
                                        dictLoadMethod, dictContentType,
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
                                     ZSTD_dlm_byCopy, ZSTD_dct_auto,
                                     cParams, ZSTD_defaultCMem);
}

ZSTD_CDict* ZSTD_createCDict_byReference(const void* dict, size_t dictSize, int compressionLevel)
{
    ZSTD_compressionParameters cParams = ZSTD_getCParams(compressionLevel, 0, dictSize);
    return ZSTD_createCDict_advanced(dict, dictSize,
                                     ZSTD_dlm_byRef, ZSTD_dct_auto,
                                     cParams, ZSTD_defaultCMem);
}

size_t ZSTD_freeCDict(ZSTD_CDict* cdict)
{
    if (cdict==NULL) return 0;   /* support free on NULL */
    {   ZSTD_customMem const cMem = cdict->customMem;
        ZSTD_free(cdict->workspace, cMem);
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
const ZSTD_CDict* ZSTD_initStaticCDict(
                                 void* workspace, size_t workspaceSize,
                           const void* dict, size_t dictSize,
                                 ZSTD_dictLoadMethod_e dictLoadMethod,
                                 ZSTD_dictContentType_e dictContentType,
                                 ZSTD_compressionParameters cParams)
{
    size_t const matchStateSize = ZSTD_sizeof_matchState(&cParams, /* forCCtx */ 0);
    size_t const neededSize = sizeof(ZSTD_CDict) + (dictLoadMethod == ZSTD_dlm_byRef ? 0 : dictSize)
                            + HUF_WORKSPACE_SIZE + matchStateSize;
    ZSTD_CDict* const cdict = (ZSTD_CDict*) workspace;
    void* ptr;
    if ((size_t)workspace & 7) return NULL;  /* 8-aligned */
    DEBUGLOG(4, "(workspaceSize < neededSize) : (%u < %u) => %u",
        (U32)workspaceSize, (U32)neededSize, (U32)(workspaceSize < neededSize));
    if (workspaceSize < neededSize) return NULL;

    if (dictLoadMethod == ZSTD_dlm_byCopy) {
        memcpy(cdict+1, dict, dictSize);
        dict = cdict+1;
        ptr = (char*)workspace + sizeof(ZSTD_CDict) + dictSize;
    } else {
        ptr = cdict+1;
    }
    cdict->workspace = ptr;
    cdict->workspaceSize = HUF_WORKSPACE_SIZE + matchStateSize;

    if (ZSTD_isError( ZSTD_initCDict_internal(cdict,
                                              dict, dictSize,
                                              ZSTD_dlm_byRef, dictContentType,
                                              cParams) ))
        return NULL;

    return cdict;
}

ZSTD_compressionParameters ZSTD_getCParamsFromCDict(const ZSTD_CDict* cdict)
{
    assert(cdict != NULL);
    return cdict->cParams;
}

/* ZSTD_compressBegin_usingCDict_advanced() :
 * cdict must be != NULL */
size_t ZSTD_compressBegin_usingCDict_advanced(
    ZSTD_CCtx* const cctx, const ZSTD_CDict* const cdict,
    ZSTD_frameParameters const fParams, unsigned long long const pledgedSrcSize)
{
    DEBUGLOG(4, "ZSTD_compressBegin_usingCDict_advanced");
    if (cdict==NULL) return ERROR(dictionary_wrong);
    {   ZSTD_CCtx_params params = cctx->requestedParams;
        params.cParams = ZSTD_getCParamsFromCDict(cdict);
        /* Increase window log to fit the entire dictionary and source if the
         * source size is known. Limit the increase to 19, which is the
         * window log for compression level 1 with the largest source size.
         */
        if (pledgedSrcSize != ZSTD_CONTENTSIZE_UNKNOWN) {
            U32 const limitedSrcSize = (U32)MIN(pledgedSrcSize, 1U << 19);
            U32 const limitedSrcLog = limitedSrcSize > 1 ? ZSTD_highbit32(limitedSrcSize - 1) + 1 : 1;
            params.cParams.windowLog = MAX(params.cParams.windowLog, limitedSrcLog);
        }
        params.fParams = fParams;
        return ZSTD_compressBegin_internal(cctx,
                                           NULL, 0, ZSTD_dct_auto,
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
    DEBUGLOG(4, "ZSTD_compressBegin_usingCDict : dictIDFlag == %u", !fParams.noDictIDFlag);
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
    DEBUGLOG(3, "ZSTD_createCStream");
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

static size_t ZSTD_resetCStream_internal(ZSTD_CStream* cctx,
                    const void* const dict, size_t const dictSize, ZSTD_dictContentType_e const dictContentType,
                    const ZSTD_CDict* const cdict,
                    ZSTD_CCtx_params const params, unsigned long long const pledgedSrcSize)
{
    DEBUGLOG(4, "ZSTD_resetCStream_internal (disableLiteralCompression=%i)",
                params.disableLiteralCompression);
    /* params are supposed to be fully validated at this point */
    assert(!ZSTD_isError(ZSTD_checkCParams(params.cParams)));
    assert(!((dict) && (cdict)));  /* either dict or cdict, not both */

    CHECK_F( ZSTD_compressBegin_internal(cctx,
                                         dict, dictSize, dictContentType,
                                         cdict,
                                         params, pledgedSrcSize,
                                         ZSTDb_buffered) );

    cctx->inToCompress = 0;
    cctx->inBuffPos = 0;
    cctx->inBuffTarget = cctx->blockSize
                      + (cctx->blockSize == pledgedSrcSize);   /* for small input: avoid automatic flush on reaching end of block, since it would require to add a 3-bytes null block to end frame */
    cctx->outBuffContentSize = cctx->outBuffFlushedSize = 0;
    cctx->streamStage = zcss_load;
    cctx->frameEnded = 0;
    return 0;   /* ready to go */
}

/* ZSTD_resetCStream():
 * pledgedSrcSize == 0 means "unknown" */
size_t ZSTD_resetCStream(ZSTD_CStream* zcs, unsigned long long pledgedSrcSize)
{
    ZSTD_CCtx_params params = zcs->requestedParams;
    DEBUGLOG(4, "ZSTD_resetCStream: pledgedSrcSize = %u", (U32)pledgedSrcSize);
    if (pledgedSrcSize==0) pledgedSrcSize = ZSTD_CONTENTSIZE_UNKNOWN;
    params.fParams.contentSizeFlag = 1;
    params.cParams = ZSTD_getCParamsFromCCtxParams(&params, pledgedSrcSize, 0);
    return ZSTD_resetCStream_internal(zcs, NULL, 0, ZSTD_dct_auto, zcs->cdict, params, pledgedSrcSize);
}

/*! ZSTD_initCStream_internal() :
 *  Note : for lib/compress only. Used by zstdmt_compress.c.
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
        DEBUGLOG(4, "loading dictionary of size %u", (U32)dictSize);
        if (zcs->staticSize) {   /* static CCtx : never uses malloc */
            /* incompatible with internal cdict creation */
            return ERROR(memory_allocation);
        }
        ZSTD_freeCDict(zcs->cdictLocal);
        zcs->cdictLocal = ZSTD_createCDict_advanced(dict, dictSize,
                                            ZSTD_dlm_byCopy, ZSTD_dct_auto,
                                            params.cParams, zcs->customMem);
        zcs->cdict = zcs->cdictLocal;
        if (zcs->cdictLocal == NULL) return ERROR(memory_allocation);
    } else {
        if (cdict) {
            params.cParams = ZSTD_getCParamsFromCDict(cdict);  /* cParams are enforced from cdict; it includes windowLog */
        }
        ZSTD_freeCDict(zcs->cdictLocal);
        zcs->cdictLocal = NULL;
        zcs->cdict = cdict;
    }

    return ZSTD_resetCStream_internal(zcs, NULL, 0, ZSTD_dct_auto, zcs->cdict, params, pledgedSrcSize);
}

/* ZSTD_initCStream_usingCDict_advanced() :
 * same as ZSTD_initCStream_usingCDict(), with control over frame parameters */
size_t ZSTD_initCStream_usingCDict_advanced(ZSTD_CStream* zcs,
                                            const ZSTD_CDict* cdict,
                                            ZSTD_frameParameters fParams,
                                            unsigned long long pledgedSrcSize)
{
    DEBUGLOG(4, "ZSTD_initCStream_usingCDict_advanced");
    if (!cdict) return ERROR(dictionary_wrong); /* cannot handle NULL cdict (does not know what to do) */
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
    ZSTD_frameParameters const fParams = { 0 /* contentSizeFlag */, 0 /* checksum */, 0 /* hideDictID */ };
    DEBUGLOG(4, "ZSTD_initCStream_usingCDict");
    return ZSTD_initCStream_usingCDict_advanced(zcs, cdict, fParams, ZSTD_CONTENTSIZE_UNKNOWN);  /* note : will check that cdict != NULL */
}


/* ZSTD_initCStream_advanced() :
 * pledgedSrcSize must be exact.
 * if srcSize is not known at init time, use value ZSTD_CONTENTSIZE_UNKNOWN.
 * dict is loaded with default parameters ZSTD_dm_auto and ZSTD_dlm_byCopy. */
size_t ZSTD_initCStream_advanced(ZSTD_CStream* zcs,
                                 const void* dict, size_t dictSize,
                                 ZSTD_parameters params, unsigned long long pledgedSrcSize)
{
    DEBUGLOG(4, "ZSTD_initCStream_advanced: pledgedSrcSize=%u, flag=%u",
                (U32)pledgedSrcSize, params.fParams.contentSizeFlag);
    CHECK_F( ZSTD_checkCParams(params.cParams) );
    if ((pledgedSrcSize==0) && (params.fParams.contentSizeFlag==0)) pledgedSrcSize = ZSTD_CONTENTSIZE_UNKNOWN;  /* for compatibility with older programs relying on this behavior. Users should now specify ZSTD_CONTENTSIZE_UNKNOWN. This line will be removed in the future. */
    {   ZSTD_CCtx_params const cctxParams = ZSTD_assignParamsToCCtxParams(zcs->requestedParams, params);
        return ZSTD_initCStream_internal(zcs, dict, dictSize, NULL /*cdict*/, cctxParams, pledgedSrcSize);
    }
}

size_t ZSTD_initCStream_usingDict(ZSTD_CStream* zcs, const void* dict, size_t dictSize, int compressionLevel)
{
    ZSTD_parameters const params = ZSTD_getParams(compressionLevel, 0, dictSize);
    ZSTD_CCtx_params const cctxParams =
            ZSTD_assignParamsToCCtxParams(zcs->requestedParams, params);
    return ZSTD_initCStream_internal(zcs, dict, dictSize, NULL, cctxParams, ZSTD_CONTENTSIZE_UNKNOWN);
}

size_t ZSTD_initCStream_srcSize(ZSTD_CStream* zcs, int compressionLevel, unsigned long long pss)
{
    U64 const pledgedSrcSize = (pss==0) ? ZSTD_CONTENTSIZE_UNKNOWN : pss;  /* temporary : 0 interpreted as "unknown" during transition period. Users willing to specify "unknown" **must** use ZSTD_CONTENTSIZE_UNKNOWN. `0` will be interpreted as "empty" in the future */
    ZSTD_parameters const params = ZSTD_getParams(compressionLevel, pledgedSrcSize, 0);
    ZSTD_CCtx_params const cctxParams = ZSTD_assignParamsToCCtxParams(zcs->requestedParams, params);
    return ZSTD_initCStream_internal(zcs, NULL, 0, NULL, cctxParams, pledgedSrcSize);
}

size_t ZSTD_initCStream(ZSTD_CStream* zcs, int compressionLevel)
{
    DEBUGLOG(4, "ZSTD_initCStream");
    return ZSTD_initCStream_srcSize(zcs, compressionLevel, ZSTD_CONTENTSIZE_UNKNOWN);
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
 *  non-static, because can be called from zstdmt_compress.c
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
    assert(zcs->inBuffSize > 0);
    assert(zcs->outBuff !=  NULL);
    assert(zcs->outBuffSize > 0);
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
    DEBUGLOG(5, "ZSTD_compress_generic, endOp=%u ", (U32)endOp);
    /* check conditions */
    if (output->pos > output->size) return ERROR(GENERIC);
    if (input->pos  > input->size)  return ERROR(GENERIC);
    assert(cctx!=NULL);

    /* transparent initialization stage */
    if (cctx->streamStage == zcss_init) {
        ZSTD_CCtx_params params = cctx->requestedParams;
        ZSTD_prefixDict const prefixDict = cctx->prefixDict;
        memset(&cctx->prefixDict, 0, sizeof(cctx->prefixDict));  /* single usage */
        assert(prefixDict.dict==NULL || cctx->cdict==NULL);   /* only one can be set */
        DEBUGLOG(4, "ZSTD_compress_generic : transparent init stage");
        if (endOp == ZSTD_e_end) cctx->pledgedSrcSizePlusOne = input->size + 1;  /* auto-fix pledgedSrcSize */
        params.cParams = ZSTD_getCParamsFromCCtxParams(
                &cctx->requestedParams, cctx->pledgedSrcSizePlusOne-1, 0 /*dictSize*/);

#ifdef ZSTD_MULTITHREAD
        if ((cctx->pledgedSrcSizePlusOne-1) <= ZSTDMT_JOBSIZE_MIN) {
            params.nbWorkers = 0; /* do not invoke multi-threading when src size is too small */
        }
        if (params.nbWorkers > 0) {
            /* mt context creation */
            if (cctx->mtctx == NULL || (params.nbWorkers != ZSTDMT_getNbWorkers(cctx->mtctx))) {
                DEBUGLOG(4, "ZSTD_compress_generic: creating new mtctx for nbWorkers=%u",
                            params.nbWorkers);
                if (cctx->mtctx != NULL)
                    DEBUGLOG(4, "ZSTD_compress_generic: previous nbWorkers was %u",
                                ZSTDMT_getNbWorkers(cctx->mtctx));
                ZSTDMT_freeCCtx(cctx->mtctx);
                cctx->mtctx = ZSTDMT_createCCtx_advanced(params.nbWorkers, cctx->customMem);
                if (cctx->mtctx == NULL) return ERROR(memory_allocation);
            }
            /* mt compression */
            DEBUGLOG(4, "call ZSTDMT_initCStream_internal as nbWorkers=%u", params.nbWorkers);
            CHECK_F( ZSTDMT_initCStream_internal(
                        cctx->mtctx,
                        prefixDict.dict, prefixDict.dictSize, ZSTD_dct_rawContent,
                        cctx->cdict, params, cctx->pledgedSrcSizePlusOne-1) );
            cctx->streamStage = zcss_load;
            cctx->appliedParams.nbWorkers = params.nbWorkers;
        } else
#endif
        {   CHECK_F( ZSTD_resetCStream_internal(cctx,
                            prefixDict.dict, prefixDict.dictSize, prefixDict.dictContentType,
                            cctx->cdict,
                            params, cctx->pledgedSrcSizePlusOne-1) );
            assert(cctx->streamStage == zcss_load);
            assert(cctx->appliedParams.nbWorkers == 0);
    }   }

    /* compression stage */
#ifdef ZSTD_MULTITHREAD
    if (cctx->appliedParams.nbWorkers > 0) {
        if (cctx->cParamsChanged) {
            ZSTDMT_updateCParams_whileCompressing(cctx->mtctx, &cctx->requestedParams);
            cctx->cParamsChanged = 0;
        }
        {   size_t const flushMin = ZSTDMT_compressStream_generic(cctx->mtctx, output, input, endOp);
            if ( ZSTD_isError(flushMin)
              || (endOp == ZSTD_e_end && flushMin == 0) ) { /* compression completed */
                ZSTD_startNewCompression(cctx);
            }
            return flushMin;
    }   }
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
 * @return : amount of data remaining to flush */
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
        DEBUGLOG(4, "ZSTD_endStream : remaining to flush : %u", (U32)toFlush);
        return toFlush;
    }
}


/*-=====  Pre-defined compression levels  =====-*/

#define ZSTD_MAX_CLEVEL     22
int ZSTD_maxCLevel(void) { return ZSTD_MAX_CLEVEL; }

static const ZSTD_compressionParameters ZSTD_defaultCParameters[4][ZSTD_MAX_CLEVEL+1] = {
{   /* "default" - guarantees a monotonically increasing memory budget */
    /* W,  C,  H,  S,  L, TL, strat */
    { 19, 12, 13,  1,  6,  1, ZSTD_fast    },  /* base for negative levels */
    { 19, 13, 14,  1,  7,  1, ZSTD_fast    },  /* level  1 */
    { 19, 15, 16,  1,  6,  1, ZSTD_fast    },  /* level  2 */
    { 20, 16, 17,  1,  5,  8, ZSTD_dfast   },  /* level  3 */
    { 20, 17, 18,  1,  5,  8, ZSTD_dfast   },  /* level  4 */
    { 20, 17, 18,  2,  5, 16, ZSTD_greedy  },  /* level  5 */
    { 21, 17, 19,  2,  5, 16, ZSTD_lazy    },  /* level  6 */
    { 21, 18, 19,  3,  5, 16, ZSTD_lazy    },  /* level  7 */
    { 21, 18, 20,  3,  5, 16, ZSTD_lazy2   },  /* level  8 */
    { 21, 19, 20,  3,  5, 16, ZSTD_lazy2   },  /* level  9 */
    { 21, 19, 21,  4,  5, 16, ZSTD_lazy2   },  /* level 10 */
    { 22, 20, 22,  4,  5, 16, ZSTD_lazy2   },  /* level 11 */
    { 22, 20, 22,  5,  5, 16, ZSTD_lazy2   },  /* level 12 */
    { 22, 21, 22,  4,  5, 32, ZSTD_btlazy2 },  /* level 13 */
    { 22, 21, 22,  5,  5, 32, ZSTD_btlazy2 },  /* level 14 */
    { 22, 22, 22,  6,  5, 32, ZSTD_btlazy2 },  /* level 15 */
    { 22, 21, 22,  4,  5, 48, ZSTD_btopt   },  /* level 16 */
    { 23, 22, 22,  4,  4, 48, ZSTD_btopt   },  /* level 17 */
    { 23, 22, 22,  5,  3, 64, ZSTD_btopt   },  /* level 18 */
    { 23, 23, 22,  7,  3,128, ZSTD_btopt   },  /* level 19 */
    { 25, 25, 23,  7,  3,128, ZSTD_btultra },  /* level 20 */
    { 26, 26, 24,  7,  3,256, ZSTD_btultra },  /* level 21 */
    { 27, 27, 25,  9,  3,512, ZSTD_btultra },  /* level 22 */
},
{   /* for srcSize <= 256 KB */
    /* W,  C,  H,  S,  L,  T, strat */
    { 18, 12, 13,  1,  5,  1, ZSTD_fast    },  /* base for negative levels */
    { 18, 13, 14,  1,  6,  1, ZSTD_fast    },  /* level  1 */
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
    { 18, 18, 17,  5,  4,  8, ZSTD_btlazy2 },  /* level 12.*/
    { 18, 19, 17,  7,  4,  8, ZSTD_btlazy2 },  /* level 13 */
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
    { 17, 12, 12,  1,  5,  1, ZSTD_fast    },  /* level  0 - not used */
    { 17, 12, 13,  1,  6,  1, ZSTD_fast    },  /* level  1 */
    { 17, 13, 16,  1,  5,  1, ZSTD_fast    },  /* level  2 */
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
    { 14, 12, 13,  1,  5,  1, ZSTD_fast    },  /* base for negative levels */
    { 14, 14, 14,  1,  6,  1, ZSTD_fast    },  /* level  1 */
    { 14, 14, 14,  1,  4,  1, ZSTD_fast    },  /* level  2 */
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

/*! ZSTD_getCParams() :
*  @return ZSTD_compressionParameters structure for a selected compression level, srcSize and dictSize.
*   Size values are optional, provide 0 if not known or unused */
ZSTD_compressionParameters ZSTD_getCParams(int compressionLevel, unsigned long long srcSizeHint, size_t dictSize)
{
    size_t const addedSize = srcSizeHint ? 0 : 500;
    U64 const rSize = srcSizeHint+dictSize ? srcSizeHint+dictSize+addedSize : (U64)-1;
    U32 const tableID = (rSize <= 256 KB) + (rSize <= 128 KB) + (rSize <= 16 KB);   /* intentional underflow for srcSizeHint == 0 */
    int row = compressionLevel;
    DEBUGLOG(5, "ZSTD_getCParams (cLevel=%i)", compressionLevel);
    if (compressionLevel == 0) row = ZSTD_CLEVEL_DEFAULT;   /* 0 == default */
    if (compressionLevel < 0) row = 0;   /* entry 0 is baseline for fast mode */
    if (compressionLevel > ZSTD_MAX_CLEVEL) row = ZSTD_MAX_CLEVEL;
    {   ZSTD_compressionParameters cp = ZSTD_defaultCParameters[tableID][row];
        if (compressionLevel < 0) cp.targetLength = (unsigned)(-compressionLevel);   /* acceleration factor */
        return ZSTD_adjustCParams_internal(cp, srcSizeHint, dictSize); }

}

/*! ZSTD_getParams() :
*   same as ZSTD_getCParams(), but @return a `ZSTD_parameters` object (instead of `ZSTD_compressionParameters`).
*   All fields of `ZSTD_frameParameters` are set to default (0) */
ZSTD_parameters ZSTD_getParams(int compressionLevel, unsigned long long srcSizeHint, size_t dictSize) {
    ZSTD_parameters params;
    ZSTD_compressionParameters const cParams = ZSTD_getCParams(compressionLevel, srcSizeHint, dictSize);
    DEBUGLOG(5, "ZSTD_getParams (cLevel=%i)", compressionLevel);
    memset(&params, 0, sizeof(params));
    params.cParams = cParams;
    params.fParams.contentSizeFlag = 1;
    return params;
}
