/**
 * Copyright (c) 2016-present, Yann Collet, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree. An additional grant
 * of patent rights can be found in the PATENTS file in the same directory.
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
#include "zstd_internal.h"  /* includes zstd.h */
#include "zstdmt_compress.h"


/*-*************************************
*  Constants
***************************************/
static const U32 g_searchStrength = 8;   /* control skip over incompressible data */
#define HASH_READ_SIZE 8
typedef enum { ZSTDcs_created=0, ZSTDcs_init, ZSTDcs_ongoing, ZSTDcs_ending } ZSTD_compressionStage_e;

/* entropy tables always have same size */
static size_t const hufCTable_size = HUF_CTABLE_SIZE(255);
static size_t const litlengthCTable_size = FSE_CTABLE_SIZE(LLFSELog, MaxLL);
static size_t const offcodeCTable_size = FSE_CTABLE_SIZE(OffFSELog, MaxOff);
static size_t const matchlengthCTable_size = FSE_CTABLE_SIZE(MLFSELog, MaxML);
static size_t const entropyScratchSpace_size = HUF_WORKSPACE_SIZE;


/*-*************************************
*  Helper functions
***************************************/
size_t ZSTD_compressBound(size_t srcSize) {
    size_t const lowLimit = 256 KB;
    size_t const margin = (srcSize < lowLimit) ? (lowLimit-srcSize) >> 12 : 0;  /* from 64 to 0 */
    return srcSize + (srcSize >> 8) + margin;
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
typedef enum { zcss_init=0, zcss_load, zcss_flush } ZSTD_cStreamStage;

struct ZSTD_CDict_s {
    void* dictBuffer;
    const void* dictContent;
    size_t dictContentSize;
    ZSTD_CCtx* refContext;
};  /* typedef'd to ZSTD_CDict within "zstd.h" */

struct ZSTD_CCtx_s {
    const BYTE* nextSrc;    /* next block here to continue on current prefix */
    const BYTE* base;       /* All regular indexes relative to this position */
    const BYTE* dictBase;   /* extDict indexes relative to this position */
    U32   dictLimit;        /* below that point, need extDict */
    U32   lowLimit;         /* below that point, no more data */
    U32   nextToUpdate;     /* index from which to continue dictionary update */
    U32   nextToUpdate3;    /* index from which to continue dictionary update */
    U32   hashLog3;         /* dispatch table : larger == faster, more memory */
    U32   loadedDictEnd;    /* index of end of dictionary */
    U32   forceWindow;      /* force back-references to respect limit of 1<<wLog, even for dictionary */
    ZSTD_compressionStage_e stage;
    U32   rep[ZSTD_REP_NUM];
    U32   repToConfirm[ZSTD_REP_NUM];
    U32   dictID;
    int   compressionLevel;
    ZSTD_parameters requestedParams;
    ZSTD_parameters appliedParams;
    void* workSpace;
    size_t workSpaceSize;
    size_t blockSize;
    U64 pledgedSrcSizePlusOne;  /* this way, 0 (default) == unknown */
    U64 consumedSrcSize;
    XXH64_state_t xxhState;
    ZSTD_customMem customMem;
    size_t staticSize;

    seqStore_t seqStore;    /* sequences storage ptrs */
    U32* hashTable;
    U32* hashTable3;
    U32* chainTable;
    HUF_repeat hufCTable_repeatMode;
    HUF_CElt* hufCTable;
    U32 fseCTables_ready;
    FSE_CTable* offcodeCTable;
    FSE_CTable* matchlengthCTable;
    FSE_CTable* litlengthCTable;
    unsigned* entropyScratchSpace;

    /* streaming */
    char*  inBuff;
    size_t inBuffSize;
    size_t inToCompress;
    size_t inBuffPos;
    size_t inBuffTarget;
    char*  outBuff;
    size_t outBuffSize;
    size_t outBuffContentSize;
    size_t outBuffFlushedSize;
    ZSTD_cStreamStage streamStage;
    U32    frameEnded;

    /* Dictionary */
    ZSTD_dictMode_e dictMode; /* select restricting dictionary to "rawContent" or "fullDict" only */
    U32 dictContentByRef;
    ZSTD_CDict* cdictLocal;
    const ZSTD_CDict* cdict;
    const void* prefix;
    size_t prefixSize;

    /* Multi-threading */
    U32 nbThreads;
    ZSTDMT_CCtx* mtctx;
};


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
    cctx->compressionLevel = ZSTD_CLEVEL_DEFAULT;
    ZSTD_STATIC_ASSERT(zcss_init==0);
    ZSTD_STATIC_ASSERT(ZSTD_CONTENTSIZE_UNKNOWN==(0ULL - 1));
    return cctx;
}

ZSTD_CCtx* ZSTD_initStaticCCtx(void *workspace, size_t workspaceSize)
{
    ZSTD_CCtx* cctx = (ZSTD_CCtx*) workspace;
    if (workspaceSize <= sizeof(ZSTD_CCtx)) return NULL;  /* minimum size */
    if ((size_t)workspace & 7) return NULL;  /* must be 8-aligned */
    memset(workspace, 0, workspaceSize);   /* may be a bit generous, could memset be smaller ? */
    cctx->staticSize = workspaceSize;
    cctx->workSpace = (void*)(cctx+1);
    cctx->workSpaceSize = workspaceSize - sizeof(ZSTD_CCtx);

    /* entropy space (never moves) */
    /* note : this code should be shared with resetCCtx, rather than copy/pasted */
    {   void* ptr = cctx->workSpace;
        cctx->hufCTable = (HUF_CElt*)ptr;
        ptr = (char*)cctx->hufCTable + hufCTable_size;
        cctx->offcodeCTable = (FSE_CTable*) ptr;
        ptr = (char*)ptr + offcodeCTable_size;
        cctx->matchlengthCTable = (FSE_CTable*) ptr;
        ptr = (char*)ptr + matchlengthCTable_size;
        cctx->litlengthCTable = (FSE_CTable*) ptr;
        ptr = (char*)ptr + litlengthCTable_size;
        assert(((size_t)ptr & 3) == 0);   /* ensure correct alignment */
        cctx->entropyScratchSpace = (unsigned*) ptr;
    }

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
    ZSTDMT_freeCCtx(cctx->mtctx);
    cctx->mtctx = NULL;
    ZSTD_free(cctx, cctx->customMem);
    return 0;   /* reserved as a potential error code in the future */
}

size_t ZSTD_sizeof_CCtx(const ZSTD_CCtx* cctx)
{
    if (cctx==NULL) return 0;   /* support sizeof on NULL */
    DEBUGLOG(5, "sizeof(*cctx) : %u", (U32)sizeof(*cctx));
    DEBUGLOG(5, "workSpaceSize : %u", (U32)cctx->workSpaceSize);
    DEBUGLOG(5, "streaming buffers : %u", (U32)(cctx->outBuffSize + cctx->inBuffSize));
    DEBUGLOG(5, "inner MTCTX : %u", (U32)ZSTDMT_sizeof_CCtx(cctx->mtctx));
    return sizeof(*cctx) + cctx->workSpaceSize
           + ZSTD_sizeof_CDict(cctx->cdictLocal)
           + cctx->outBuffSize + cctx->inBuffSize
           + ZSTDMT_sizeof_CCtx(cctx->mtctx);
}

size_t ZSTD_sizeof_CStream(const ZSTD_CStream* zcs)
{
    return ZSTD_sizeof_CCtx(zcs);  /* same object */
}

/* private API call, for dictBuilder only */
const seqStore_t* ZSTD_getSeqStore(const ZSTD_CCtx* ctx) { return &(ctx->seqStore); }

static ZSTD_parameters ZSTD_getParamsFromCCtx(const ZSTD_CCtx* cctx) { return cctx->appliedParams; }

/* older variant; will be deprecated */
size_t ZSTD_setCCtxParameter(ZSTD_CCtx* cctx, ZSTD_CCtxParameter param, unsigned value)
{
    switch(param)
    {
    case ZSTD_p_forceWindow : cctx->forceWindow = value>0; cctx->loadedDictEnd = 0; return 0;
    ZSTD_STATIC_ASSERT(ZSTD_dm_auto==0);
    ZSTD_STATIC_ASSERT(ZSTD_dm_rawContent==1);
    case ZSTD_p_forceRawDict : cctx->dictMode = (ZSTD_dictMode_e)(value>0); return 0;
    default: return ERROR(parameter_unknown);
    }
}


#define ZSTD_CLEVEL_CUSTOM 999
static void ZSTD_cLevelToCParams(ZSTD_CCtx* cctx)
{
    if (cctx->compressionLevel==ZSTD_CLEVEL_CUSTOM) return;
    cctx->requestedParams.cParams = ZSTD_getCParams(cctx->compressionLevel,
                                            cctx->pledgedSrcSizePlusOne-1, 0);
    cctx->compressionLevel = ZSTD_CLEVEL_CUSTOM;
}

#define CLAMPCHECK(val,min,max) {                       \
    if (((val)<(min)) | ((val)>(max))) {                \
        return ERROR(compressionParameter_outOfBound);  \
}   }

size_t ZSTD_CCtx_setParameter(ZSTD_CCtx* cctx, ZSTD_cParameter param, unsigned value)
{
    if (cctx->streamStage != zcss_init) return ERROR(stage_wrong);

    switch(param)
    {
    case ZSTD_p_compressionLevel :
        if ((int)value > ZSTD_maxCLevel()) value = ZSTD_maxCLevel();   /* cap max compression level */
        if (value == 0) return 0;  /* special value : 0 means "don't change anything" */
        if (cctx->cdict) return ERROR(stage_wrong);
        cctx->compressionLevel = value;
        return 0;

    case ZSTD_p_windowLog :
        DEBUGLOG(5, "setting ZSTD_p_windowLog = %u (cdict:%u)",
                    value, (cctx->cdict!=NULL));
        if (value == 0) return 0;  /* special value : 0 means "don't change anything" */
        if (cctx->cdict) return ERROR(stage_wrong);
        CLAMPCHECK(value, ZSTD_WINDOWLOG_MIN, ZSTD_WINDOWLOG_MAX);
        ZSTD_cLevelToCParams(cctx);
        cctx->requestedParams.cParams.windowLog = value;
        return 0;

    case ZSTD_p_hashLog :
        if (value == 0) return 0;  /* special value : 0 means "don't change anything" */
        if (cctx->cdict) return ERROR(stage_wrong);
        CLAMPCHECK(value, ZSTD_HASHLOG_MIN, ZSTD_HASHLOG_MAX);
        ZSTD_cLevelToCParams(cctx);
        cctx->requestedParams.cParams.hashLog = value;
        return 0;

    case ZSTD_p_chainLog :
        if (value == 0) return 0;  /* special value : 0 means "don't change anything" */
        if (cctx->cdict) return ERROR(stage_wrong);
        CLAMPCHECK(value, ZSTD_CHAINLOG_MIN, ZSTD_CHAINLOG_MAX);
        ZSTD_cLevelToCParams(cctx);
        cctx->requestedParams.cParams.chainLog = value;
        return 0;

    case ZSTD_p_searchLog :
        if (value == 0) return 0;  /* special value : 0 means "don't change anything" */
        if (cctx->cdict) return ERROR(stage_wrong);
        CLAMPCHECK(value, ZSTD_SEARCHLOG_MIN, ZSTD_SEARCHLOG_MAX);
        ZSTD_cLevelToCParams(cctx);
        cctx->requestedParams.cParams.searchLog = value;
        return 0;

    case ZSTD_p_minMatch :
        if (value == 0) return 0;  /* special value : 0 means "don't change anything" */
        if (cctx->cdict) return ERROR(stage_wrong);
        CLAMPCHECK(value, ZSTD_SEARCHLENGTH_MIN, ZSTD_SEARCHLENGTH_MAX);
        ZSTD_cLevelToCParams(cctx);
        cctx->requestedParams.cParams.searchLength = value;
        return 0;

    case ZSTD_p_targetLength :
        if (value == 0) return 0;  /* special value : 0 means "don't change anything" */
        if (cctx->cdict) return ERROR(stage_wrong);
        CLAMPCHECK(value, ZSTD_TARGETLENGTH_MIN, ZSTD_TARGETLENGTH_MAX);
        ZSTD_cLevelToCParams(cctx);
        cctx->requestedParams.cParams.targetLength = value;
        return 0;

    case ZSTD_p_compressionStrategy :
        if (value == 0) return 0;  /* special value : 0 means "don't change anything" */
        if (cctx->cdict) return ERROR(stage_wrong);
        CLAMPCHECK(value, (unsigned)ZSTD_fast, (unsigned)ZSTD_btultra);
        ZSTD_cLevelToCParams(cctx);
        cctx->requestedParams.cParams.strategy = (ZSTD_strategy)value;
        return 0;

    case ZSTD_p_contentSizeFlag :
        DEBUGLOG(5, "set content size flag = %u", (value>0));
        /* Content size written in frame header _when known_ (default:1) */
        cctx->requestedParams.fParams.contentSizeFlag = value>0;
        return 0;

    case ZSTD_p_checksumFlag :
        /* A 32-bits content checksum will be calculated and written at end of frame (default:0) */
        cctx->requestedParams.fParams.checksumFlag = value>0;
        return 0;

    case ZSTD_p_dictIDFlag : /* When applicable, dictionary's dictID is provided in frame header (default:1) */
        DEBUGLOG(5, "set dictIDFlag = %u", (value>0));
        cctx->requestedParams.fParams.noDictIDFlag = (value==0);
        return 0;

    /* Dictionary parameters */
    case ZSTD_p_dictMode :
        if (cctx->cdict) return ERROR(stage_wrong);  /* must be set before loading */
        /* restrict dictionary mode, to "rawContent" or "fullDict" only */
        ZSTD_STATIC_ASSERT((U32)ZSTD_dm_fullDict > (U32)ZSTD_dm_rawContent);
        if (value > (unsigned)ZSTD_dm_fullDict)
            return ERROR(compressionParameter_outOfBound);
        cctx->dictMode = (ZSTD_dictMode_e)value;
        return 0;

    case ZSTD_p_refDictContent :
        if (cctx->cdict) return ERROR(stage_wrong);  /* must be set before loading */
        /* dictionary content will be referenced, instead of copied */
        cctx->dictContentByRef = value>0;
        return 0;

    case ZSTD_p_forceMaxWindow :  /* Force back-references to remain < windowSize,
                                   * even when referencing into Dictionary content
                                   * default : 0 when using a CDict, 1 when using a Prefix */
        cctx->forceWindow = value>0;
        cctx->loadedDictEnd = 0;
        return 0;

    case ZSTD_p_nbThreads:
        if (value==0) return 0;
        DEBUGLOG(5, " setting nbThreads : %u", value);
#ifndef ZSTD_MULTITHREAD
        if (value > 1) return ERROR(compressionParameter_unsupported);
#endif
        if ((value>1) && (cctx->nbThreads != value)) {
            if (cctx->staticSize)  /* MT not compatible with static alloc */
                return ERROR(compressionParameter_unsupported);
            ZSTDMT_freeCCtx(cctx->mtctx);
            cctx->nbThreads = 1;
            cctx->mtctx = ZSTDMT_createCCtx(value);
            if (cctx->mtctx == NULL) return ERROR(memory_allocation);
        }
        cctx->nbThreads = value;
        return 0;

    case ZSTD_p_jobSize:
        if (cctx->nbThreads <= 1) return ERROR(compressionParameter_unsupported);
        assert(cctx->mtctx != NULL);
        return ZSTDMT_setMTCtxParameter(cctx->mtctx, ZSTDMT_p_sectionSize, value);

    case ZSTD_p_overlapSizeLog:
        DEBUGLOG(5, " setting overlap with nbThreads == %u", cctx->nbThreads);
        if (cctx->nbThreads <= 1) return ERROR(compressionParameter_unsupported);
        assert(cctx->mtctx != NULL);
        return ZSTDMT_setMTCtxParameter(cctx->mtctx, ZSTDMT_p_overlapSectionLog, value);

    default: return ERROR(parameter_unknown);
    }
}

ZSTDLIB_API size_t ZSTD_CCtx_setPledgedSrcSize(ZSTD_CCtx* cctx, unsigned long long pledgedSrcSize)
{
    DEBUGLOG(5, " setting pledgedSrcSize to %u", (U32)pledgedSrcSize);
    if (cctx->streamStage != zcss_init) return ERROR(stage_wrong);
    cctx->pledgedSrcSizePlusOne = pledgedSrcSize+1;
    return 0;
}

ZSTDLIB_API size_t ZSTD_CCtx_loadDictionary(ZSTD_CCtx* cctx, const void* dict, size_t dictSize)
{
    if (cctx->streamStage != zcss_init) return ERROR(stage_wrong);
    if (cctx->staticSize) return ERROR(memory_allocation);  /* no malloc for static CCtx */
    DEBUGLOG(5, "load dictionary of size %u", (U32)dictSize);
    ZSTD_freeCDict(cctx->cdictLocal);  /* in case one already exists */
    if (dict==NULL || dictSize==0) {   /* no dictionary mode */
        cctx->cdictLocal = NULL;
        cctx->cdict = NULL;
    } else {
        ZSTD_compressionParameters const cParams =
                cctx->compressionLevel == ZSTD_CLEVEL_CUSTOM ?
                cctx->requestedParams.cParams :
                ZSTD_getCParams(cctx->compressionLevel, 0, dictSize);
        cctx->cdictLocal = ZSTD_createCDict_advanced(
                                dict, dictSize,
                                cctx->dictContentByRef, cctx->dictMode,
                                cParams, cctx->customMem);
        cctx->cdict = cctx->cdictLocal;
        if (cctx->cdictLocal == NULL)
            return ERROR(memory_allocation);
    }
    return 0;
}

size_t ZSTD_CCtx_refCDict(ZSTD_CCtx* cctx, const ZSTD_CDict* cdict)
{
    if (cctx->streamStage != zcss_init) return ERROR(stage_wrong);
    cctx->cdict = cdict;
    cctx->prefix = NULL;   /* exclusive */
    cctx->prefixSize = 0;
    return 0;
}

size_t ZSTD_CCtx_refPrefix(ZSTD_CCtx* cctx, const void* prefix, size_t prefixSize)
{
    if (cctx->streamStage != zcss_init) return ERROR(stage_wrong);
    cctx->cdict = NULL;   /* prefix discards any prior cdict */
    cctx->prefix = prefix;
    cctx->prefixSize = prefixSize;
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
    if ((U32)(cParams.strategy) > (U32)ZSTD_btultra) return ERROR(compressionParameter_unsupported);
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
    mostly downsizing to reduce memory consumption and initialization.
    Both `srcSize` and `dictSize` are optional (use 0 if unknown),
    but if both are 0, no optimization can be done.
    Note : cPar is considered validated at this stage. Use ZSTD_checkParams() to ensure that. */
ZSTD_compressionParameters ZSTD_adjustCParams_internal(ZSTD_compressionParameters cPar, unsigned long long srcSize, size_t dictSize)
{
    assert(ZSTD_checkCParams(cPar)==0);
    if (srcSize+dictSize == 0) return cPar;   /* no size information available : no adjustment */

    /* resize params, to use less memory when necessary */
    {   U32 const minSrcSize = (srcSize==0) ? 500 : 0;
        U64 const rSize = srcSize + dictSize + minSrcSize;
        if (rSize < ((U64)1<<ZSTD_WINDOWLOG_MAX)) {
            U32 const srcLog = MAX(ZSTD_HASHLOG_MIN, ZSTD_highbit32((U32)(rSize)-1) + 1);
            if (cPar.windowLog > srcLog) cPar.windowLog = srcLog;
    }   }
    if (cPar.hashLog > cPar.windowLog) cPar.hashLog = cPar.windowLog;
    {   U32 const cycleLog = ZSTD_cycleLog(cPar.chainLog, cPar.strategy);
        if (cycleLog > cPar.windowLog) cPar.chainLog -= (cycleLog - cPar.windowLog);
    }

    if (cPar.windowLog < ZSTD_WINDOWLOG_ABSOLUTEMIN) cPar.windowLog = ZSTD_WINDOWLOG_ABSOLUTEMIN;  /* required for frame header */

    return cPar;
}

ZSTD_compressionParameters ZSTD_adjustCParams(ZSTD_compressionParameters cPar, unsigned long long srcSize, size_t dictSize)
{
    cPar = ZSTD_clampCParams(cPar);
    return ZSTD_adjustCParams_internal(cPar, srcSize, dictSize);
}


size_t ZSTD_estimateCCtxSize_advanced(ZSTD_compressionParameters cParams)
{
    size_t const blockSize = MIN(ZSTD_BLOCKSIZE_MAX, (size_t)1 << cParams.windowLog);
    U32    const divider = (cParams.searchLength==3) ? 3 : 4;
    size_t const maxNbSeq = blockSize / divider;
    size_t const tokenSpace = blockSize + 11*maxNbSeq;

    size_t const chainSize = (cParams.strategy == ZSTD_fast) ? 0 : (1 << cParams.chainLog);
    size_t const hSize = ((size_t)1) << cParams.hashLog;
    U32    const hashLog3 = (cParams.searchLength>3) ? 0 : MIN(ZSTD_HASHLOG3_MAX, cParams.windowLog);
    size_t const h3Size = ((size_t)1) << hashLog3;
    size_t const entropySpace = hufCTable_size + litlengthCTable_size
                              + offcodeCTable_size + matchlengthCTable_size
                              + entropyScratchSpace_size;
    size_t const tableSpace = (chainSize + hSize + h3Size) * sizeof(U32);

    size_t const optBudget = ((MaxML+1) + (MaxLL+1) + (MaxOff+1) + (1<<Litbits))*sizeof(U32)
                          + (ZSTD_OPT_NUM+1)*(sizeof(ZSTD_match_t) + sizeof(ZSTD_optimal_t));
    size_t const optSpace = ((cParams.strategy == ZSTD_btopt) || (cParams.strategy == ZSTD_btultra)) ? optBudget : 0;
    size_t const neededSpace = entropySpace + tableSpace + tokenSpace + optSpace;

    DEBUGLOG(5, "sizeof(ZSTD_CCtx) : %u", (U32)sizeof(ZSTD_CCtx));
    DEBUGLOG(5, "estimate workSpace : %u", (U32)neededSpace);
    return sizeof(ZSTD_CCtx) + neededSpace;
}

size_t ZSTD_estimateCCtxSize(int compressionLevel)
{
    ZSTD_compressionParameters const cParams = ZSTD_getCParams(compressionLevel, 0, 0);
    return ZSTD_estimateCCtxSize_advanced(cParams);
}

size_t ZSTD_estimateCStreamSize_advanced(ZSTD_compressionParameters cParams)
{
    size_t const CCtxSize = ZSTD_estimateCCtxSize_advanced(cParams);
    size_t const blockSize = MIN(ZSTD_BLOCKSIZE_MAX, (size_t)1 << cParams.windowLog);
    size_t const inBuffSize = ((size_t)1 << cParams.windowLog) + blockSize;
    size_t const outBuffSize = ZSTD_compressBound(blockSize) + 1;
    size_t const streamingSize = inBuffSize + outBuffSize;

    return CCtxSize + streamingSize;
}

size_t ZSTD_estimateCStreamSize(int compressionLevel) {
    ZSTD_compressionParameters const cParams = ZSTD_getCParams(compressionLevel, 0, 0);
    return ZSTD_estimateCStreamSize_advanced(cParams);
}


static U32 ZSTD_equivalentParams(ZSTD_compressionParameters cParams1,
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

/*! ZSTD_continueCCtx() :
 *  reuse CCtx without reset (note : requires no dictionary) */
static size_t ZSTD_continueCCtx(ZSTD_CCtx* cctx, ZSTD_parameters params, U64 pledgedSrcSize)
{
    U32 const end = (U32)(cctx->nextSrc - cctx->base);
    DEBUGLOG(5, "continue mode");
    cctx->appliedParams = params;
    cctx->pledgedSrcSizePlusOne = pledgedSrcSize+1;
    cctx->consumedSrcSize = 0;
    if (pledgedSrcSize == ZSTD_CONTENTSIZE_UNKNOWN)
        cctx->appliedParams.fParams.contentSizeFlag = 0;
    DEBUGLOG(5, "pledged content size : %u ; flag : %u",
        (U32)pledgedSrcSize, cctx->appliedParams.fParams.contentSizeFlag);
    cctx->lowLimit = end;
    cctx->dictLimit = end;
    cctx->nextToUpdate = end+1;
    cctx->stage = ZSTDcs_init;
    cctx->dictID = 0;
    cctx->loadedDictEnd = 0;
    { int i; for (i=0; i<ZSTD_REP_NUM; i++) cctx->rep[i] = repStartValue[i]; }
    cctx->seqStore.litLengthSum = 0;  /* force reset of btopt stats */
    XXH64_reset(&cctx->xxhState, 0);
    return 0;
}

typedef enum { ZSTDcrp_continue, ZSTDcrp_noMemset } ZSTD_compResetPolicy_e;
typedef enum { ZSTDb_not_buffered, ZSTDb_buffered } ZSTD_buffered_policy_e;

/*! ZSTD_resetCCtx_internal() :
    note : `params` are assumed fully validated at this stage */
static size_t ZSTD_resetCCtx_internal(ZSTD_CCtx* zc,
                                      ZSTD_parameters params, U64 pledgedSrcSize,
                                      ZSTD_compResetPolicy_e const crp,
                                      ZSTD_buffered_policy_e const zbuff)
{
    assert(!ZSTD_isError(ZSTD_checkCParams(params.cParams)));

    if (crp == ZSTDcrp_continue) {
        if (ZSTD_equivalentParams(params.cParams, zc->appliedParams.cParams)) {
            DEBUGLOG(5, "ZSTD_equivalentParams()==1");
            zc->fseCTables_ready = 0;
            zc->hufCTable_repeatMode = HUF_repeat_none;
            return ZSTD_continueCCtx(zc, params, pledgedSrcSize);
    }   }

    {   size_t const blockSize = MIN(ZSTD_BLOCKSIZE_MAX, (size_t)1 << params.cParams.windowLog);
        U32    const divider = (params.cParams.searchLength==3) ? 3 : 4;
        size_t const maxNbSeq = blockSize / divider;
        size_t const tokenSpace = blockSize + 11*maxNbSeq;
        size_t const chainSize = (params.cParams.strategy == ZSTD_fast) ?
                                0 : (1 << params.cParams.chainLog);
        size_t const hSize = ((size_t)1) << params.cParams.hashLog;
        U32    const hashLog3 = (params.cParams.searchLength>3) ?
                                0 : MIN(ZSTD_HASHLOG3_MAX, params.cParams.windowLog);
        size_t const h3Size = ((size_t)1) << hashLog3;
        size_t const tableSpace = (chainSize + hSize + h3Size) * sizeof(U32);
        size_t const buffOutSize = (zbuff==ZSTDb_buffered) ? ZSTD_compressBound(blockSize)+1 : 0;
        size_t const buffInSize = (zbuff==ZSTDb_buffered) ? ((size_t)1 << params.cParams.windowLog) + blockSize : 0;
        void* ptr;

        /* Check if workSpace is large enough, alloc a new one if needed */
        {   size_t const entropySpace = hufCTable_size + litlengthCTable_size
                                  + offcodeCTable_size + matchlengthCTable_size
                                  + entropyScratchSpace_size;
            size_t const optPotentialSpace = ((MaxML+1) + (MaxLL+1) + (MaxOff+1) + (1<<Litbits)) * sizeof(U32)
                                  + (ZSTD_OPT_NUM+1) * (sizeof(ZSTD_match_t)+sizeof(ZSTD_optimal_t));
            size_t const optSpace = ( (params.cParams.strategy == ZSTD_btopt)
                                    || (params.cParams.strategy == ZSTD_btultra)) ?
                                    optPotentialSpace : 0;
            size_t const bufferSpace = buffInSize + buffOutSize;
            size_t const neededSpace = entropySpace + optSpace + tableSpace
                                     + tokenSpace + bufferSpace;

            if (zc->workSpaceSize < neededSpace) {  /* too small : resize /*/
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
                zc->hufCTable = (HUF_CElt*)ptr;
                ptr = (char*)zc->hufCTable + hufCTable_size;  /* note : HUF_CElt* is incomplete type, size is estimated via macro */
                zc->offcodeCTable = (FSE_CTable*) ptr;
                ptr = (char*)ptr + offcodeCTable_size;
                zc->matchlengthCTable = (FSE_CTable*) ptr;
                ptr = (char*)ptr + matchlengthCTable_size;
                zc->litlengthCTable = (FSE_CTable*) ptr;
                ptr = (char*)ptr + litlengthCTable_size;
                assert(((size_t)ptr & 3) == 0);   /* ensure correct alignment */
                zc->entropyScratchSpace = (unsigned*) ptr;
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
        zc->fseCTables_ready = 0;
        zc->hufCTable_repeatMode = HUF_repeat_none;
        zc->nextToUpdate = 1;
        zc->nextSrc = NULL;
        zc->base = NULL;
        zc->dictBase = NULL;
        zc->dictLimit = 0;
        zc->lowLimit = 0;
        { int i; for (i=0; i<ZSTD_REP_NUM; i++) zc->rep[i] = repStartValue[i]; }
        zc->hashLog3 = hashLog3;
        zc->seqStore.litLengthSum = 0;

        /* ensure entropy tables are close together at the beginning */
        assert((void*)zc->hufCTable == zc->workSpace);
        assert((char*)zc->offcodeCTable == (char*)zc->hufCTable + hufCTable_size);
        assert((char*)zc->matchlengthCTable == (char*)zc->offcodeCTable + offcodeCTable_size);
        assert((char*)zc->litlengthCTable == (char*)zc->matchlengthCTable + matchlengthCTable_size);
        assert((char*)zc->entropyScratchSpace == (char*)zc->litlengthCTable + litlengthCTable_size);
        ptr = (char*)zc->entropyScratchSpace + entropyScratchSpace_size;

        /* opt parser space */
        if ((params.cParams.strategy == ZSTD_btopt) || (params.cParams.strategy == ZSTD_btultra)) {
            DEBUGLOG(5, "reserving optimal parser space");
            assert(((size_t)ptr & 3) == 0);  /* ensure ptr is properly aligned */
            zc->seqStore.litFreq = (U32*)ptr;
            zc->seqStore.litLengthFreq = zc->seqStore.litFreq + (1<<Litbits);
            zc->seqStore.matchLengthFreq = zc->seqStore.litLengthFreq + (MaxLL+1);
            zc->seqStore.offCodeFreq = zc->seqStore.matchLengthFreq + (MaxML+1);
            ptr = zc->seqStore.offCodeFreq + (MaxOff+1);
            zc->seqStore.matchTable = (ZSTD_match_t*)ptr;
            ptr = zc->seqStore.matchTable + ZSTD_OPT_NUM+1;
            zc->seqStore.priceTable = (ZSTD_optimal_t*)ptr;
            ptr = zc->seqStore.priceTable + ZSTD_OPT_NUM+1;
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
    for (i=0; i<ZSTD_REP_NUM; i++) cctx->rep[i] = 0;
}


/*! ZSTD_copyCCtx_internal() :
 *  Duplicate an existing context `srcCCtx` into another one `dstCCtx`.
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
    {   ZSTD_parameters params = srcCCtx->appliedParams;
        params.fParams = fParams;
        ZSTD_resetCCtx_internal(dstCCtx, params, pledgedSrcSize,
                                ZSTDcrp_noMemset, zbuff);
    }

    /* copy tables */
    {   size_t const chainSize = (srcCCtx->appliedParams.cParams.strategy == ZSTD_fast) ? 0 : (1 << srcCCtx->appliedParams.cParams.chainLog);
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
    dstCCtx->fseCTables_ready = srcCCtx->fseCTables_ready;
    if (srcCCtx->fseCTables_ready) {
        memcpy(dstCCtx->litlengthCTable, srcCCtx->litlengthCTable, litlengthCTable_size);
        memcpy(dstCCtx->matchlengthCTable, srcCCtx->matchlengthCTable, matchlengthCTable_size);
        memcpy(dstCCtx->offcodeCTable, srcCCtx->offcodeCTable, offcodeCTable_size);
    }
    dstCCtx->hufCTable_repeatMode = srcCCtx->hufCTable_repeatMode;
    if (srcCCtx->hufCTable_repeatMode) {
        memcpy(dstCCtx->hufCTable, srcCCtx->hufCTable, hufCTable_size);
    }

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

/*! ZSTD_reduceIndex() :
*   rescale all indexes to avoid future overflow (indexes are U32) */
static void ZSTD_reduceIndex (ZSTD_CCtx* zc, const U32 reducerValue)
{
    { U32 const hSize = 1 << zc->appliedParams.cParams.hashLog;
      ZSTD_reduceTable(zc->hashTable, hSize, reducerValue); }

    { U32 const chainSize = (zc->appliedParams.cParams.strategy == ZSTD_fast) ? 0 : (1 << zc->appliedParams.cParams.chainLog);
      ZSTD_reduceTable(zc->chainTable, chainSize, reducerValue); }

    { U32 const h3Size = (zc->hashLog3) ? 1 << zc->hashLog3 : 0;
      ZSTD_reduceTable(zc->hashTable3, h3Size, reducerValue); }
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

static size_t ZSTD_compressLiterals (ZSTD_CCtx* zc,
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
    {   size_t const minLitSize = zc->hufCTable_repeatMode == HUF_repeat_valid ? 6 : LITERAL_NOENTROPY;
        if (srcSize <= minLitSize) return ZSTD_noCompressLiterals(dst, dstCapacity, src, srcSize);
    }

    if (dstCapacity < lhSize+1) return ERROR(dstSize_tooSmall);   /* not enough space for compression */
    {   HUF_repeat repeat = zc->hufCTable_repeatMode;
        int const preferRepeat = zc->appliedParams.cParams.strategy < ZSTD_lazy ? srcSize <= 1024 : 0;
        if (repeat == HUF_repeat_valid && lhSize == 3) singleStream = 1;
        cLitSize = singleStream ? HUF_compress1X_repeat(ostart+lhSize, dstCapacity-lhSize, src, srcSize, 255, 11,
                                      zc->entropyScratchSpace, entropyScratchSpace_size, zc->hufCTable, &repeat, preferRepeat)
                                : HUF_compress4X_repeat(ostart+lhSize, dstCapacity-lhSize, src, srcSize, 255, 11,
                                      zc->entropyScratchSpace, entropyScratchSpace_size, zc->hufCTable, &repeat, preferRepeat);
        if (repeat != HUF_repeat_none) { hType = set_repeat; }    /* reused the existing table */
        else { zc->hufCTable_repeatMode = HUF_repeat_check; }       /* now have a table to reuse */
    }

    if ((cLitSize==0) | (cLitSize >= srcSize - minGain)) {
        zc->hufCTable_repeatMode = HUF_repeat_none;
        return ZSTD_noCompressLiterals(dst, dstCapacity, src, srcSize);
    }
    if (cLitSize==1) {
        zc->hufCTable_repeatMode = HUF_repeat_none;
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

static const BYTE LL_Code[64] = {  0,  1,  2,  3,  4,  5,  6,  7,
                                   8,  9, 10, 11, 12, 13, 14, 15,
                                  16, 16, 17, 17, 18, 18, 19, 19,
                                  20, 20, 20, 20, 21, 21, 21, 21,
                                  22, 22, 22, 22, 22, 22, 22, 22,
                                  23, 23, 23, 23, 23, 23, 23, 23,
                                  24, 24, 24, 24, 24, 24, 24, 24,
                                  24, 24, 24, 24, 24, 24, 24, 24 };

static const BYTE ML_Code[128] = { 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15,
                                  16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31,
                                  32, 32, 33, 33, 34, 34, 35, 35, 36, 36, 36, 36, 37, 37, 37, 37,
                                  38, 38, 38, 38, 38, 38, 38, 38, 39, 39, 39, 39, 39, 39, 39, 39,
                                  40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40,
                                  41, 41, 41, 41, 41, 41, 41, 41, 41, 41, 41, 41, 41, 41, 41, 41,
                                  42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42,
                                  42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42 };


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

MEM_STATIC size_t ZSTD_compressSequences (ZSTD_CCtx* zc,
                              void* dst, size_t dstCapacity,
                              size_t srcSize)
{
    const int longOffsets = zc->appliedParams.cParams.windowLog > STREAM_ACCUMULATOR_MIN;
    const seqStore_t* seqStorePtr = &(zc->seqStore);
    U32 count[MaxSeq+1];
    S16 norm[MaxSeq+1];
    FSE_CTable* CTable_LitLength = zc->litlengthCTable;
    FSE_CTable* CTable_OffsetBits = zc->offcodeCTable;
    FSE_CTable* CTable_MatchLength = zc->matchlengthCTable;
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
    BYTE scratchBuffer[1<<MAX(MLFSELog,LLFSELog)];

    /* Compress literals */
    {   const BYTE* const literals = seqStorePtr->litStart;
        size_t const litSize = seqStorePtr->lit - literals;
        size_t const cSize = ZSTD_compressLiterals(zc, op, dstCapacity, literals, litSize);
        if (ZSTD_isError(cSize)) return cSize;
        op += cSize;
    }

    /* Sequences Header */
    if ((oend-op) < 3 /*max nbSeq Size*/ + 1 /*seqHead */) return ERROR(dstSize_tooSmall);
    if (nbSeq < 0x7F) *op++ = (BYTE)nbSeq;
    else if (nbSeq < LONGNBSEQ) op[0] = (BYTE)((nbSeq>>8) + 0x80), op[1] = (BYTE)nbSeq, op+=2;
    else op[0]=0xFF, MEM_writeLE16(op+1, (U16)(nbSeq - LONGNBSEQ)), op+=3;
    if (nbSeq==0) goto _check_compressibility;

    /* seqHead : flags for FSE encoding type */
    seqHead = op++;

#define MIN_SEQ_FOR_DYNAMIC_FSE   64
#define MAX_SEQ_FOR_STATIC_FSE  1000

    /* convert length/distances into codes */
    ZSTD_seqToCodes(seqStorePtr);

    /* CTable for Literal Lengths */
    {   U32 max = MaxLL;
        size_t const mostFrequent = FSE_countFast_wksp(count, &max, llCodeTable, nbSeq, zc->entropyScratchSpace);
        if ((mostFrequent == nbSeq) && (nbSeq > 2)) {
            *op++ = llCodeTable[0];
            FSE_buildCTable_rle(CTable_LitLength, (BYTE)max);
            LLtype = set_rle;
        } else if ((zc->fseCTables_ready) && (nbSeq < MAX_SEQ_FOR_STATIC_FSE)) {
            LLtype = set_repeat;
        } else if ((nbSeq < MIN_SEQ_FOR_DYNAMIC_FSE) || (mostFrequent < (nbSeq >> (LL_defaultNormLog-1)))) {
            FSE_buildCTable_wksp(CTable_LitLength, LL_defaultNorm, MaxLL, LL_defaultNormLog, scratchBuffer, sizeof(scratchBuffer));
            LLtype = set_basic;
        } else {
            size_t nbSeq_1 = nbSeq;
            const U32 tableLog = FSE_optimalTableLog(LLFSELog, nbSeq, max);
            if (count[llCodeTable[nbSeq-1]]>1) { count[llCodeTable[nbSeq-1]]--; nbSeq_1--; }
            FSE_normalizeCount(norm, tableLog, count, nbSeq_1, max);
            { size_t const NCountSize = FSE_writeNCount(op, oend-op, norm, max, tableLog);   /* overflow protected */
              if (FSE_isError(NCountSize)) return NCountSize;
              op += NCountSize; }
            FSE_buildCTable_wksp(CTable_LitLength, norm, max, tableLog, scratchBuffer, sizeof(scratchBuffer));
            LLtype = set_compressed;
    }   }

    /* CTable for Offsets */
    {   U32 max = MaxOff;
        size_t const mostFrequent = FSE_countFast_wksp(count, &max, ofCodeTable, nbSeq, zc->entropyScratchSpace);
        if ((mostFrequent == nbSeq) && (nbSeq > 2)) {
            *op++ = ofCodeTable[0];
            FSE_buildCTable_rle(CTable_OffsetBits, (BYTE)max);
            Offtype = set_rle;
        } else if ((zc->fseCTables_ready) && (nbSeq < MAX_SEQ_FOR_STATIC_FSE)) {
            Offtype = set_repeat;
        } else if ((nbSeq < MIN_SEQ_FOR_DYNAMIC_FSE) || (mostFrequent < (nbSeq >> (OF_defaultNormLog-1)))) {
            FSE_buildCTable_wksp(CTable_OffsetBits, OF_defaultNorm, MaxOff, OF_defaultNormLog, scratchBuffer, sizeof(scratchBuffer));
            Offtype = set_basic;
        } else {
            size_t nbSeq_1 = nbSeq;
            const U32 tableLog = FSE_optimalTableLog(OffFSELog, nbSeq, max);
            if (count[ofCodeTable[nbSeq-1]]>1) { count[ofCodeTable[nbSeq-1]]--; nbSeq_1--; }
            FSE_normalizeCount(norm, tableLog, count, nbSeq_1, max);
            { size_t const NCountSize = FSE_writeNCount(op, oend-op, norm, max, tableLog);   /* overflow protected */
              if (FSE_isError(NCountSize)) return NCountSize;
              op += NCountSize; }
            FSE_buildCTable_wksp(CTable_OffsetBits, norm, max, tableLog, scratchBuffer, sizeof(scratchBuffer));
            Offtype = set_compressed;
    }   }

    /* CTable for MatchLengths */
    {   U32 max = MaxML;
        size_t const mostFrequent = FSE_countFast_wksp(count, &max, mlCodeTable, nbSeq, zc->entropyScratchSpace);
        if ((mostFrequent == nbSeq) && (nbSeq > 2)) {
            *op++ = *mlCodeTable;
            FSE_buildCTable_rle(CTable_MatchLength, (BYTE)max);
            MLtype = set_rle;
        } else if ((zc->fseCTables_ready) && (nbSeq < MAX_SEQ_FOR_STATIC_FSE)) {
            MLtype = set_repeat;
        } else if ((nbSeq < MIN_SEQ_FOR_DYNAMIC_FSE) || (mostFrequent < (nbSeq >> (ML_defaultNormLog-1)))) {
            FSE_buildCTable_wksp(CTable_MatchLength, ML_defaultNorm, MaxML, ML_defaultNormLog, scratchBuffer, sizeof(scratchBuffer));
            MLtype = set_basic;
        } else {
            size_t nbSeq_1 = nbSeq;
            const U32 tableLog = FSE_optimalTableLog(MLFSELog, nbSeq, max);
            if (count[mlCodeTable[nbSeq-1]]>1) { count[mlCodeTable[nbSeq-1]]--; nbSeq_1--; }
            FSE_normalizeCount(norm, tableLog, count, nbSeq_1, max);
            { size_t const NCountSize = FSE_writeNCount(op, oend-op, norm, max, tableLog);   /* overflow protected */
              if (FSE_isError(NCountSize)) return NCountSize;
              op += NCountSize; }
            FSE_buildCTable_wksp(CTable_MatchLength, norm, max, tableLog, scratchBuffer, sizeof(scratchBuffer));
            MLtype = set_compressed;
    }   }

    *seqHead = (BYTE)((LLtype<<6) + (Offtype<<4) + (MLtype<<2));
    zc->fseCTables_ready = 0;

    /* Encoding Sequences */
    {   BIT_CStream_t blockStream;
        FSE_CState_t  stateMatchLength;
        FSE_CState_t  stateOffsetBits;
        FSE_CState_t  stateLitLength;

        CHECK_E(BIT_initCStream(&blockStream, op, oend-op), dstSize_tooSmall); /* not enough space remaining */

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
                if (MEM_32bits()) BIT_flushBits(&blockStream);                  /* (7)*/
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
            op += streamSize;
    }   }

    /* check compressibility */
_check_compressibility:
    {   size_t const minGain = ZSTD_minGain(srcSize);
        size_t const maxCSize = srcSize - minGain;
        if ((size_t)(op-ostart) >= maxCSize) {
            zc->hufCTable_repeatMode = HUF_repeat_none;
            return 0;
    }   }

    /* confirm repcodes */
    { int i; for (i=0; i<ZSTD_REP_NUM; i++) zc->rep[i] = zc->repToConfirm[i]; }

    return op - ostart;
}


/*! ZSTD_storeSeq() :
    Store a sequence (literal length, literals, offset code and match length code) into seqStore_t.
    `offsetCode` : distance to match, or 0 == repCode.
    `matchCode` : matchLength - MINMATCH
*/
MEM_STATIC void ZSTD_storeSeq(seqStore_t* seqStorePtr, size_t litLength, const void* literals, U32 offsetCode, size_t matchCode)
{
#if defined(ZSTD_DEBUG) && (ZSTD_DEBUG >= 6)
    static const BYTE* g_start = NULL;
    U32 const pos = (U32)((const BYTE*)literals - g_start);
    if (g_start==NULL) g_start = (const BYTE*)literals;
    if ((pos > 0) && (pos < 1000000000))
        DEBUGLOG(6, "Cpos %6u :%5u literals & match %3u bytes at distance %6u",
               pos, (U32)litLength, (U32)matchCode+MINMATCH, (U32)offsetCode);
#endif
    /* copy Literals */
    assert(seqStorePtr->lit + litLength <= seqStorePtr->litStart + 128 KB);
    ZSTD_wildcopy(seqStorePtr->lit, literals, litLength);
    seqStorePtr->lit += litLength;

    /* literal Length */
    if (litLength>0xFFFF) {
        seqStorePtr->longLengthID = 1;
        seqStorePtr->longLengthPos = (U32)(seqStorePtr->sequences - seqStorePtr->sequencesStart);
    }
    seqStorePtr->sequences[0].litLength = (U16)litLength;

    /* match offset */
    seqStorePtr->sequences[0].offset = offsetCode + 1;

    /* match Length */
    if (matchCode>0xFFFF) {
        seqStorePtr->longLengthID = 2;
        seqStorePtr->longLengthPos = (U32)(seqStorePtr->sequences - seqStorePtr->sequencesStart);
    }
    seqStorePtr->sequences[0].matchLength = (U16)matchCode;

    seqStorePtr->sequences++;
}


/*-*************************************
*  Match length counter
***************************************/
static unsigned ZSTD_NbCommonBytes (register size_t val)
{
    if (MEM_isLittleEndian()) {
        if (MEM_64bits()) {
#       if defined(_MSC_VER) && defined(_WIN64)
            unsigned long r = 0;
            _BitScanForward64( &r, (U64)val );
            return (unsigned)(r>>3);
#       elif defined(__GNUC__) && (__GNUC__ >= 3)
            return (__builtin_ctzll((U64)val) >> 3);
#       else
            static const int DeBruijnBytePos[64] = { 0, 0, 0, 0, 0, 1, 1, 2,
                                                     0, 3, 1, 3, 1, 4, 2, 7,
                                                     0, 2, 3, 6, 1, 5, 3, 5,
                                                     1, 3, 4, 4, 2, 5, 6, 7,
                                                     7, 0, 1, 2, 3, 3, 4, 6,
                                                     2, 6, 5, 5, 3, 4, 5, 6,
                                                     7, 1, 2, 4, 6, 4, 4, 5,
                                                     7, 2, 6, 5, 7, 6, 7, 7 };
            return DeBruijnBytePos[((U64)((val & -(long long)val) * 0x0218A392CDABBD3FULL)) >> 58];
#       endif
        } else { /* 32 bits */
#       if defined(_MSC_VER)
            unsigned long r=0;
            _BitScanForward( &r, (U32)val );
            return (unsigned)(r>>3);
#       elif defined(__GNUC__) && (__GNUC__ >= 3)
            return (__builtin_ctz((U32)val) >> 3);
#       else
            static const int DeBruijnBytePos[32] = { 0, 0, 3, 0, 3, 1, 3, 0,
                                                     3, 2, 2, 1, 3, 2, 0, 1,
                                                     3, 3, 1, 2, 2, 2, 2, 0,
                                                     3, 1, 2, 0, 1, 0, 1, 1 };
            return DeBruijnBytePos[((U32)((val & -(S32)val) * 0x077CB531U)) >> 27];
#       endif
        }
    } else {  /* Big Endian CPU */
        if (MEM_64bits()) {
#       if defined(_MSC_VER) && defined(_WIN64)
            unsigned long r = 0;
            _BitScanReverse64( &r, val );
            return (unsigned)(r>>3);
#       elif defined(__GNUC__) && (__GNUC__ >= 3)
            return (__builtin_clzll(val) >> 3);
#       else
            unsigned r;
            const unsigned n32 = sizeof(size_t)*4;   /* calculate this way due to compiler complaining in 32-bits mode */
            if (!(val>>n32)) { r=4; } else { r=0; val>>=n32; }
            if (!(val>>16)) { r+=2; val>>=8; } else { val>>=24; }
            r += (!val);
            return r;
#       endif
        } else { /* 32 bits */
#       if defined(_MSC_VER)
            unsigned long r = 0;
            _BitScanReverse( &r, (unsigned long)val );
            return (unsigned)(r>>3);
#       elif defined(__GNUC__) && (__GNUC__ >= 3)
            return (__builtin_clz((U32)val) >> 3);
#       else
            unsigned r;
            if (!(val>>16)) { r=2; val>>=8; } else { r=0; val>>=24; }
            r += (!val);
            return r;
#       endif
    }   }
}


static size_t ZSTD_count(const BYTE* pIn, const BYTE* pMatch, const BYTE* const pInLimit)
{
    const BYTE* const pStart = pIn;
    const BYTE* const pInLoopLimit = pInLimit - (sizeof(size_t)-1);

    while (pIn < pInLoopLimit) {
        size_t const diff = MEM_readST(pMatch) ^ MEM_readST(pIn);
        if (!diff) { pIn+=sizeof(size_t); pMatch+=sizeof(size_t); continue; }
        pIn += ZSTD_NbCommonBytes(diff);
        return (size_t)(pIn - pStart);
    }
    if (MEM_64bits()) if ((pIn<(pInLimit-3)) && (MEM_read32(pMatch) == MEM_read32(pIn))) { pIn+=4; pMatch+=4; }
    if ((pIn<(pInLimit-1)) && (MEM_read16(pMatch) == MEM_read16(pIn))) { pIn+=2; pMatch+=2; }
    if ((pIn<pInLimit) && (*pMatch == *pIn)) pIn++;
    return (size_t)(pIn - pStart);
}

/** ZSTD_count_2segments() :
*   can count match length with `ip` & `match` in 2 different segments.
*   convention : on reaching mEnd, match count continue starting from iStart
*/
static size_t ZSTD_count_2segments(const BYTE* ip, const BYTE* match, const BYTE* iEnd, const BYTE* mEnd, const BYTE* iStart)
{
    const BYTE* const vEnd = MIN( ip + (mEnd - match), iEnd);
    size_t const matchLength = ZSTD_count(ip, match, vEnd);
    if (match + matchLength != mEnd) return matchLength;
    return matchLength + ZSTD_count(ip+matchLength, iStart, iEnd);
}


/*-*************************************
*  Hashes
***************************************/
static const U32 prime3bytes = 506832829U;
static U32    ZSTD_hash3(U32 u, U32 h) { return ((u << (32-24)) * prime3bytes)  >> (32-h) ; }
MEM_STATIC size_t ZSTD_hash3Ptr(const void* ptr, U32 h) { return ZSTD_hash3(MEM_readLE32(ptr), h); } /* only in zstd_opt.h */

static const U32 prime4bytes = 2654435761U;
static U32    ZSTD_hash4(U32 u, U32 h) { return (u * prime4bytes) >> (32-h) ; }
static size_t ZSTD_hash4Ptr(const void* ptr, U32 h) { return ZSTD_hash4(MEM_read32(ptr), h); }

static const U64 prime5bytes = 889523592379ULL;
static size_t ZSTD_hash5(U64 u, U32 h) { return (size_t)(((u  << (64-40)) * prime5bytes) >> (64-h)) ; }
static size_t ZSTD_hash5Ptr(const void* p, U32 h) { return ZSTD_hash5(MEM_readLE64(p), h); }

static const U64 prime6bytes = 227718039650203ULL;
static size_t ZSTD_hash6(U64 u, U32 h) { return (size_t)(((u  << (64-48)) * prime6bytes) >> (64-h)) ; }
static size_t ZSTD_hash6Ptr(const void* p, U32 h) { return ZSTD_hash6(MEM_readLE64(p), h); }

static const U64 prime7bytes = 58295818150454627ULL;
static size_t ZSTD_hash7(U64 u, U32 h) { return (size_t)(((u  << (64-56)) * prime7bytes) >> (64-h)) ; }
static size_t ZSTD_hash7Ptr(const void* p, U32 h) { return ZSTD_hash7(MEM_readLE64(p), h); }

static const U64 prime8bytes = 0xCF1BBCDCB7A56463ULL;
static size_t ZSTD_hash8(U64 u, U32 h) { return (size_t)(((u) * prime8bytes) >> (64-h)) ; }
static size_t ZSTD_hash8Ptr(const void* p, U32 h) { return ZSTD_hash8(MEM_readLE64(p), h); }

static size_t ZSTD_hashPtr(const void* p, U32 hBits, U32 mls)
{
    switch(mls)
    {
    default:
    case 4: return ZSTD_hash4Ptr(p, hBits);
    case 5: return ZSTD_hash5Ptr(p, hBits);
    case 6: return ZSTD_hash6Ptr(p, hBits);
    case 7: return ZSTD_hash7Ptr(p, hBits);
    case 8: return ZSTD_hash8Ptr(p, hBits);
    }
}


/*-*************************************
*  Fast Scan
***************************************/
static void ZSTD_fillHashTable (ZSTD_CCtx* zc, const void* end, const U32 mls)
{
    U32* const hashTable = zc->hashTable;
    U32  const hBits = zc->appliedParams.cParams.hashLog;
    const BYTE* const base = zc->base;
    const BYTE* ip = base + zc->nextToUpdate;
    const BYTE* const iend = ((const BYTE*)end) - HASH_READ_SIZE;
    const size_t fastHashFillStep = 3;

    while(ip <= iend) {
        hashTable[ZSTD_hashPtr(ip, hBits, mls)] = (U32)(ip - base);
        ip += fastHashFillStep;
    }
}


FORCE_INLINE
void ZSTD_compressBlock_fast_generic(ZSTD_CCtx* cctx,
                               const void* src, size_t srcSize,
                               const U32 mls)
{
    U32* const hashTable = cctx->hashTable;
    U32  const hBits = cctx->appliedParams.cParams.hashLog;
    seqStore_t* seqStorePtr = &(cctx->seqStore);
    const BYTE* const base = cctx->base;
    const BYTE* const istart = (const BYTE*)src;
    const BYTE* ip = istart;
    const BYTE* anchor = istart;
    const U32   lowestIndex = cctx->dictLimit;
    const BYTE* const lowest = base + lowestIndex;
    const BYTE* const iend = istart + srcSize;
    const BYTE* const ilimit = iend - HASH_READ_SIZE;
    U32 offset_1=cctx->rep[0], offset_2=cctx->rep[1];
    U32 offsetSaved = 0;

    /* init */
    ip += (ip==lowest);
    {   U32 const maxRep = (U32)(ip-lowest);
        if (offset_2 > maxRep) offsetSaved = offset_2, offset_2 = 0;
        if (offset_1 > maxRep) offsetSaved = offset_1, offset_1 = 0;
    }

    /* Main Search Loop */
    while (ip < ilimit) {   /* < instead of <=, because repcode check at (ip+1) */
        size_t mLength;
        size_t const h = ZSTD_hashPtr(ip, hBits, mls);
        U32 const current = (U32)(ip-base);
        U32 const matchIndex = hashTable[h];
        const BYTE* match = base + matchIndex;
        hashTable[h] = current;   /* update hash table */

        if ((offset_1 > 0) & (MEM_read32(ip+1-offset_1) == MEM_read32(ip+1))) {
            mLength = ZSTD_count(ip+1+4, ip+1+4-offset_1, iend) + 4;
            ip++;
            ZSTD_storeSeq(seqStorePtr, ip-anchor, anchor, 0, mLength-MINMATCH);
        } else {
            U32 offset;
            if ( (matchIndex <= lowestIndex) || (MEM_read32(match) != MEM_read32(ip)) ) {
                ip += ((ip-anchor) >> g_searchStrength) + 1;
                continue;
            }
            mLength = ZSTD_count(ip+4, match+4, iend) + 4;
            offset = (U32)(ip-match);
            while (((ip>anchor) & (match>lowest)) && (ip[-1] == match[-1])) { ip--; match--; mLength++; } /* catch up */
            offset_2 = offset_1;
            offset_1 = offset;

            ZSTD_storeSeq(seqStorePtr, ip-anchor, anchor, offset + ZSTD_REP_MOVE, mLength-MINMATCH);
        }

        /* match found */
        ip += mLength;
        anchor = ip;

        if (ip <= ilimit) {
            /* Fill Table */
            hashTable[ZSTD_hashPtr(base+current+2, hBits, mls)] = current+2;  /* here because current+2 could be > iend-8 */
            hashTable[ZSTD_hashPtr(ip-2, hBits, mls)] = (U32)(ip-2-base);
            /* check immediate repcode */
            while ( (ip <= ilimit)
                 && ( (offset_2>0)
                 & (MEM_read32(ip) == MEM_read32(ip - offset_2)) )) {
                /* store sequence */
                size_t const rLength = ZSTD_count(ip+4, ip+4-offset_2, iend) + 4;
                { U32 const tmpOff = offset_2; offset_2 = offset_1; offset_1 = tmpOff; }  /* swap offset_2 <=> offset_1 */
                hashTable[ZSTD_hashPtr(ip, hBits, mls)] = (U32)(ip-base);
                ZSTD_storeSeq(seqStorePtr, 0, anchor, 0, rLength-MINMATCH);
                ip += rLength;
                anchor = ip;
                continue;   /* faster when present ... (?) */
    }   }   }

    /* save reps for next block */
    cctx->repToConfirm[0] = offset_1 ? offset_1 : offsetSaved;
    cctx->repToConfirm[1] = offset_2 ? offset_2 : offsetSaved;

    /* Last Literals */
    {   size_t const lastLLSize = iend - anchor;
        memcpy(seqStorePtr->lit, anchor, lastLLSize);
        seqStorePtr->lit += lastLLSize;
    }
}


static void ZSTD_compressBlock_fast(ZSTD_CCtx* ctx,
                       const void* src, size_t srcSize)
{
    const U32 mls = ctx->appliedParams.cParams.searchLength;
    switch(mls)
    {
    default: /* includes case 3 */
    case 4 :
        ZSTD_compressBlock_fast_generic(ctx, src, srcSize, 4); return;
    case 5 :
        ZSTD_compressBlock_fast_generic(ctx, src, srcSize, 5); return;
    case 6 :
        ZSTD_compressBlock_fast_generic(ctx, src, srcSize, 6); return;
    case 7 :
        ZSTD_compressBlock_fast_generic(ctx, src, srcSize, 7); return;
    }
}


static void ZSTD_compressBlock_fast_extDict_generic(ZSTD_CCtx* ctx,
                                 const void* src, size_t srcSize,
                                 const U32 mls)
{
    U32* hashTable = ctx->hashTable;
    const U32 hBits = ctx->appliedParams.cParams.hashLog;
    seqStore_t* seqStorePtr = &(ctx->seqStore);
    const BYTE* const base = ctx->base;
    const BYTE* const dictBase = ctx->dictBase;
    const BYTE* const istart = (const BYTE*)src;
    const BYTE* ip = istart;
    const BYTE* anchor = istart;
    const U32   lowestIndex = ctx->lowLimit;
    const BYTE* const dictStart = dictBase + lowestIndex;
    const U32   dictLimit = ctx->dictLimit;
    const BYTE* const lowPrefixPtr = base + dictLimit;
    const BYTE* const dictEnd = dictBase + dictLimit;
    const BYTE* const iend = istart + srcSize;
    const BYTE* const ilimit = iend - 8;
    U32 offset_1=ctx->rep[0], offset_2=ctx->rep[1];

    /* Search Loop */
    while (ip < ilimit) {  /* < instead of <=, because (ip+1) */
        const size_t h = ZSTD_hashPtr(ip, hBits, mls);
        const U32 matchIndex = hashTable[h];
        const BYTE* matchBase = matchIndex < dictLimit ? dictBase : base;
        const BYTE* match = matchBase + matchIndex;
        const U32 current = (U32)(ip-base);
        const U32 repIndex = current + 1 - offset_1;   /* offset_1 expected <= current +1 */
        const BYTE* repBase = repIndex < dictLimit ? dictBase : base;
        const BYTE* repMatch = repBase + repIndex;
        size_t mLength;
        hashTable[h] = current;   /* update hash table */

        if ( (((U32)((dictLimit-1) - repIndex) >= 3) /* intentional underflow */ & (repIndex > lowestIndex))
           && (MEM_read32(repMatch) == MEM_read32(ip+1)) ) {
            const BYTE* repMatchEnd = repIndex < dictLimit ? dictEnd : iend;
            mLength = ZSTD_count_2segments(ip+1+4, repMatch+4, iend, repMatchEnd, lowPrefixPtr) + 4;
            ip++;
            ZSTD_storeSeq(seqStorePtr, ip-anchor, anchor, 0, mLength-MINMATCH);
        } else {
            if ( (matchIndex < lowestIndex) ||
                 (MEM_read32(match) != MEM_read32(ip)) ) {
                ip += ((ip-anchor) >> g_searchStrength) + 1;
                continue;
            }
            {   const BYTE* matchEnd = matchIndex < dictLimit ? dictEnd : iend;
                const BYTE* lowMatchPtr = matchIndex < dictLimit ? dictStart : lowPrefixPtr;
                U32 offset;
                mLength = ZSTD_count_2segments(ip+4, match+4, iend, matchEnd, lowPrefixPtr) + 4;
                while (((ip>anchor) & (match>lowMatchPtr)) && (ip[-1] == match[-1])) { ip--; match--; mLength++; }   /* catch up */
                offset = current - matchIndex;
                offset_2 = offset_1;
                offset_1 = offset;
                ZSTD_storeSeq(seqStorePtr, ip-anchor, anchor, offset + ZSTD_REP_MOVE, mLength-MINMATCH);
        }   }

        /* found a match : store it */
        ip += mLength;
        anchor = ip;

        if (ip <= ilimit) {
            /* Fill Table */
            hashTable[ZSTD_hashPtr(base+current+2, hBits, mls)] = current+2;
            hashTable[ZSTD_hashPtr(ip-2, hBits, mls)] = (U32)(ip-2-base);
            /* check immediate repcode */
            while (ip <= ilimit) {
                U32 const current2 = (U32)(ip-base);
                U32 const repIndex2 = current2 - offset_2;
                const BYTE* repMatch2 = repIndex2 < dictLimit ? dictBase + repIndex2 : base + repIndex2;
                if ( (((U32)((dictLimit-1) - repIndex2) >= 3) & (repIndex2 > lowestIndex))  /* intentional overflow */
                   && (MEM_read32(repMatch2) == MEM_read32(ip)) ) {
                    const BYTE* const repEnd2 = repIndex2 < dictLimit ? dictEnd : iend;
                    size_t const repLength2 = ZSTD_count_2segments(ip+4, repMatch2+4, iend, repEnd2, lowPrefixPtr) + 4;
                    U32 tmpOffset = offset_2; offset_2 = offset_1; offset_1 = tmpOffset;   /* swap offset_2 <=> offset_1 */
                    ZSTD_storeSeq(seqStorePtr, 0, anchor, 0, repLength2-MINMATCH);
                    hashTable[ZSTD_hashPtr(ip, hBits, mls)] = current2;
                    ip += repLength2;
                    anchor = ip;
                    continue;
                }
                break;
    }   }   }

    /* save reps for next block */
    ctx->repToConfirm[0] = offset_1; ctx->repToConfirm[1] = offset_2;

    /* Last Literals */
    {   size_t const lastLLSize = iend - anchor;
        memcpy(seqStorePtr->lit, anchor, lastLLSize);
        seqStorePtr->lit += lastLLSize;
    }
}


static void ZSTD_compressBlock_fast_extDict(ZSTD_CCtx* ctx,
                         const void* src, size_t srcSize)
{
    U32 const mls = ctx->appliedParams.cParams.searchLength;
    switch(mls)
    {
    default: /* includes case 3 */
    case 4 :
        ZSTD_compressBlock_fast_extDict_generic(ctx, src, srcSize, 4); return;
    case 5 :
        ZSTD_compressBlock_fast_extDict_generic(ctx, src, srcSize, 5); return;
    case 6 :
        ZSTD_compressBlock_fast_extDict_generic(ctx, src, srcSize, 6); return;
    case 7 :
        ZSTD_compressBlock_fast_extDict_generic(ctx, src, srcSize, 7); return;
    }
}


/*-*************************************
*  Double Fast
***************************************/
static void ZSTD_fillDoubleHashTable (ZSTD_CCtx* cctx, const void* end, const U32 mls)
{
    U32* const hashLarge = cctx->hashTable;
    U32  const hBitsL = cctx->appliedParams.cParams.hashLog;
    U32* const hashSmall = cctx->chainTable;
    U32  const hBitsS = cctx->appliedParams.cParams.chainLog;
    const BYTE* const base = cctx->base;
    const BYTE* ip = base + cctx->nextToUpdate;
    const BYTE* const iend = ((const BYTE*)end) - HASH_READ_SIZE;
    const size_t fastHashFillStep = 3;

    while(ip <= iend) {
        hashSmall[ZSTD_hashPtr(ip, hBitsS, mls)] = (U32)(ip - base);
        hashLarge[ZSTD_hashPtr(ip, hBitsL, 8)] = (U32)(ip - base);
        ip += fastHashFillStep;
    }
}


FORCE_INLINE
void ZSTD_compressBlock_doubleFast_generic(ZSTD_CCtx* cctx,
                                 const void* src, size_t srcSize,
                                 const U32 mls)
{
    U32* const hashLong = cctx->hashTable;
    const U32 hBitsL = cctx->appliedParams.cParams.hashLog;
    U32* const hashSmall = cctx->chainTable;
    const U32 hBitsS = cctx->appliedParams.cParams.chainLog;
    seqStore_t* seqStorePtr = &(cctx->seqStore);
    const BYTE* const base = cctx->base;
    const BYTE* const istart = (const BYTE*)src;
    const BYTE* ip = istart;
    const BYTE* anchor = istart;
    const U32 lowestIndex = cctx->dictLimit;
    const BYTE* const lowest = base + lowestIndex;
    const BYTE* const iend = istart + srcSize;
    const BYTE* const ilimit = iend - HASH_READ_SIZE;
    U32 offset_1=cctx->rep[0], offset_2=cctx->rep[1];
    U32 offsetSaved = 0;

    /* init */
    ip += (ip==lowest);
    {   U32 const maxRep = (U32)(ip-lowest);
        if (offset_2 > maxRep) offsetSaved = offset_2, offset_2 = 0;
        if (offset_1 > maxRep) offsetSaved = offset_1, offset_1 = 0;
    }

    /* Main Search Loop */
    while (ip < ilimit) {   /* < instead of <=, because repcode check at (ip+1) */
        size_t mLength;
        size_t const h2 = ZSTD_hashPtr(ip, hBitsL, 8);
        size_t const h = ZSTD_hashPtr(ip, hBitsS, mls);
        U32 const current = (U32)(ip-base);
        U32 const matchIndexL = hashLong[h2];
        U32 const matchIndexS = hashSmall[h];
        const BYTE* matchLong = base + matchIndexL;
        const BYTE* match = base + matchIndexS;
        hashLong[h2] = hashSmall[h] = current;   /* update hash tables */

        assert(offset_1 <= current);   /* supposed guaranteed by construction */
        if ((offset_1 > 0) & (MEM_read32(ip+1-offset_1) == MEM_read32(ip+1))) {
            /* favor repcode */
            mLength = ZSTD_count(ip+1+4, ip+1+4-offset_1, iend) + 4;
            ip++;
            ZSTD_storeSeq(seqStorePtr, ip-anchor, anchor, 0, mLength-MINMATCH);
        } else {
            U32 offset;
            if ( (matchIndexL > lowestIndex) && (MEM_read64(matchLong) == MEM_read64(ip)) ) {
                mLength = ZSTD_count(ip+8, matchLong+8, iend) + 8;
                offset = (U32)(ip-matchLong);
                while (((ip>anchor) & (matchLong>lowest)) && (ip[-1] == matchLong[-1])) { ip--; matchLong--; mLength++; } /* catch up */
            } else if ( (matchIndexS > lowestIndex) && (MEM_read32(match) == MEM_read32(ip)) ) {
                size_t const hl3 = ZSTD_hashPtr(ip+1, hBitsL, 8);
                U32 const matchIndexL3 = hashLong[hl3];
                const BYTE* matchL3 = base + matchIndexL3;
                hashLong[hl3] = current + 1;
                if ( (matchIndexL3 > lowestIndex) && (MEM_read64(matchL3) == MEM_read64(ip+1)) ) {
                    mLength = ZSTD_count(ip+9, matchL3+8, iend) + 8;
                    ip++;
                    offset = (U32)(ip-matchL3);
                    while (((ip>anchor) & (matchL3>lowest)) && (ip[-1] == matchL3[-1])) { ip--; matchL3--; mLength++; } /* catch up */
                } else {
                    mLength = ZSTD_count(ip+4, match+4, iend) + 4;
                    offset = (U32)(ip-match);
                    while (((ip>anchor) & (match>lowest)) && (ip[-1] == match[-1])) { ip--; match--; mLength++; } /* catch up */
                }
            } else {
                ip += ((ip-anchor) >> g_searchStrength) + 1;
                continue;
            }

            offset_2 = offset_1;
            offset_1 = offset;

            ZSTD_storeSeq(seqStorePtr, ip-anchor, anchor, offset + ZSTD_REP_MOVE, mLength-MINMATCH);
        }

        /* match found */
        ip += mLength;
        anchor = ip;

        if (ip <= ilimit) {
            /* Fill Table */
            hashLong[ZSTD_hashPtr(base+current+2, hBitsL, 8)] =
                hashSmall[ZSTD_hashPtr(base+current+2, hBitsS, mls)] = current+2;  /* here because current+2 could be > iend-8 */
            hashLong[ZSTD_hashPtr(ip-2, hBitsL, 8)] =
                hashSmall[ZSTD_hashPtr(ip-2, hBitsS, mls)] = (U32)(ip-2-base);

            /* check immediate repcode */
            while ( (ip <= ilimit)
                 && ( (offset_2>0)
                 & (MEM_read32(ip) == MEM_read32(ip - offset_2)) )) {
                /* store sequence */
                size_t const rLength = ZSTD_count(ip+4, ip+4-offset_2, iend) + 4;
                { U32 const tmpOff = offset_2; offset_2 = offset_1; offset_1 = tmpOff; } /* swap offset_2 <=> offset_1 */
                hashSmall[ZSTD_hashPtr(ip, hBitsS, mls)] = (U32)(ip-base);
                hashLong[ZSTD_hashPtr(ip, hBitsL, 8)] = (U32)(ip-base);
                ZSTD_storeSeq(seqStorePtr, 0, anchor, 0, rLength-MINMATCH);
                ip += rLength;
                anchor = ip;
                continue;   /* faster when present ... (?) */
    }   }   }

    /* save reps for next block */
    cctx->repToConfirm[0] = offset_1 ? offset_1 : offsetSaved;
    cctx->repToConfirm[1] = offset_2 ? offset_2 : offsetSaved;

    /* Last Literals */
    {   size_t const lastLLSize = iend - anchor;
        memcpy(seqStorePtr->lit, anchor, lastLLSize);
        seqStorePtr->lit += lastLLSize;
    }
}


static void ZSTD_compressBlock_doubleFast(ZSTD_CCtx* ctx, const void* src, size_t srcSize)
{
    const U32 mls = ctx->appliedParams.cParams.searchLength;
    switch(mls)
    {
    default: /* includes case 3 */
    case 4 :
        ZSTD_compressBlock_doubleFast_generic(ctx, src, srcSize, 4); return;
    case 5 :
        ZSTD_compressBlock_doubleFast_generic(ctx, src, srcSize, 5); return;
    case 6 :
        ZSTD_compressBlock_doubleFast_generic(ctx, src, srcSize, 6); return;
    case 7 :
        ZSTD_compressBlock_doubleFast_generic(ctx, src, srcSize, 7); return;
    }
}


static void ZSTD_compressBlock_doubleFast_extDict_generic(ZSTD_CCtx* ctx,
                                 const void* src, size_t srcSize,
                                 const U32 mls)
{
    U32* const hashLong = ctx->hashTable;
    U32  const hBitsL = ctx->appliedParams.cParams.hashLog;
    U32* const hashSmall = ctx->chainTable;
    U32  const hBitsS = ctx->appliedParams.cParams.chainLog;
    seqStore_t* seqStorePtr = &(ctx->seqStore);
    const BYTE* const base = ctx->base;
    const BYTE* const dictBase = ctx->dictBase;
    const BYTE* const istart = (const BYTE*)src;
    const BYTE* ip = istart;
    const BYTE* anchor = istart;
    const U32   lowestIndex = ctx->lowLimit;
    const BYTE* const dictStart = dictBase + lowestIndex;
    const U32   dictLimit = ctx->dictLimit;
    const BYTE* const lowPrefixPtr = base + dictLimit;
    const BYTE* const dictEnd = dictBase + dictLimit;
    const BYTE* const iend = istart + srcSize;
    const BYTE* const ilimit = iend - 8;
    U32 offset_1=ctx->rep[0], offset_2=ctx->rep[1];

    /* Search Loop */
    while (ip < ilimit) {  /* < instead of <=, because (ip+1) */
        const size_t hSmall = ZSTD_hashPtr(ip, hBitsS, mls);
        const U32 matchIndex = hashSmall[hSmall];
        const BYTE* matchBase = matchIndex < dictLimit ? dictBase : base;
        const BYTE* match = matchBase + matchIndex;

        const size_t hLong = ZSTD_hashPtr(ip, hBitsL, 8);
        const U32 matchLongIndex = hashLong[hLong];
        const BYTE* matchLongBase = matchLongIndex < dictLimit ? dictBase : base;
        const BYTE* matchLong = matchLongBase + matchLongIndex;

        const U32 current = (U32)(ip-base);
        const U32 repIndex = current + 1 - offset_1;   /* offset_1 expected <= current +1 */
        const BYTE* repBase = repIndex < dictLimit ? dictBase : base;
        const BYTE* repMatch = repBase + repIndex;
        size_t mLength;
        hashSmall[hSmall] = hashLong[hLong] = current;   /* update hash table */

        if ( (((U32)((dictLimit-1) - repIndex) >= 3) /* intentional underflow */ & (repIndex > lowestIndex))
           && (MEM_read32(repMatch) == MEM_read32(ip+1)) ) {
            const BYTE* repMatchEnd = repIndex < dictLimit ? dictEnd : iend;
            mLength = ZSTD_count_2segments(ip+1+4, repMatch+4, iend, repMatchEnd, lowPrefixPtr) + 4;
            ip++;
            ZSTD_storeSeq(seqStorePtr, ip-anchor, anchor, 0, mLength-MINMATCH);
        } else {
            if ((matchLongIndex > lowestIndex) && (MEM_read64(matchLong) == MEM_read64(ip))) {
                const BYTE* matchEnd = matchLongIndex < dictLimit ? dictEnd : iend;
                const BYTE* lowMatchPtr = matchLongIndex < dictLimit ? dictStart : lowPrefixPtr;
                U32 offset;
                mLength = ZSTD_count_2segments(ip+8, matchLong+8, iend, matchEnd, lowPrefixPtr) + 8;
                offset = current - matchLongIndex;
                while (((ip>anchor) & (matchLong>lowMatchPtr)) && (ip[-1] == matchLong[-1])) { ip--; matchLong--; mLength++; }   /* catch up */
                offset_2 = offset_1;
                offset_1 = offset;
                ZSTD_storeSeq(seqStorePtr, ip-anchor, anchor, offset + ZSTD_REP_MOVE, mLength-MINMATCH);

            } else if ((matchIndex > lowestIndex) && (MEM_read32(match) == MEM_read32(ip))) {
                size_t const h3 = ZSTD_hashPtr(ip+1, hBitsL, 8);
                U32 const matchIndex3 = hashLong[h3];
                const BYTE* const match3Base = matchIndex3 < dictLimit ? dictBase : base;
                const BYTE* match3 = match3Base + matchIndex3;
                U32 offset;
                hashLong[h3] = current + 1;
                if ( (matchIndex3 > lowestIndex) && (MEM_read64(match3) == MEM_read64(ip+1)) ) {
                    const BYTE* matchEnd = matchIndex3 < dictLimit ? dictEnd : iend;
                    const BYTE* lowMatchPtr = matchIndex3 < dictLimit ? dictStart : lowPrefixPtr;
                    mLength = ZSTD_count_2segments(ip+9, match3+8, iend, matchEnd, lowPrefixPtr) + 8;
                    ip++;
                    offset = current+1 - matchIndex3;
                    while (((ip>anchor) & (match3>lowMatchPtr)) && (ip[-1] == match3[-1])) { ip--; match3--; mLength++; } /* catch up */
                } else {
                    const BYTE* matchEnd = matchIndex < dictLimit ? dictEnd : iend;
                    const BYTE* lowMatchPtr = matchIndex < dictLimit ? dictStart : lowPrefixPtr;
                    mLength = ZSTD_count_2segments(ip+4, match+4, iend, matchEnd, lowPrefixPtr) + 4;
                    offset = current - matchIndex;
                    while (((ip>anchor) & (match>lowMatchPtr)) && (ip[-1] == match[-1])) { ip--; match--; mLength++; }   /* catch up */
                }
                offset_2 = offset_1;
                offset_1 = offset;
                ZSTD_storeSeq(seqStorePtr, ip-anchor, anchor, offset + ZSTD_REP_MOVE, mLength-MINMATCH);

            } else {
                ip += ((ip-anchor) >> g_searchStrength) + 1;
                continue;
        }   }

        /* found a match : store it */
        ip += mLength;
        anchor = ip;

        if (ip <= ilimit) {
            /* Fill Table */
            hashSmall[ZSTD_hashPtr(base+current+2, hBitsS, mls)] = current+2;
            hashLong[ZSTD_hashPtr(base+current+2, hBitsL, 8)] = current+2;
            hashSmall[ZSTD_hashPtr(ip-2, hBitsS, mls)] = (U32)(ip-2-base);
            hashLong[ZSTD_hashPtr(ip-2, hBitsL, 8)] = (U32)(ip-2-base);
            /* check immediate repcode */
            while (ip <= ilimit) {
                U32 const current2 = (U32)(ip-base);
                U32 const repIndex2 = current2 - offset_2;
                const BYTE* repMatch2 = repIndex2 < dictLimit ? dictBase + repIndex2 : base + repIndex2;
                if ( (((U32)((dictLimit-1) - repIndex2) >= 3) & (repIndex2 > lowestIndex))  /* intentional overflow */
                   && (MEM_read32(repMatch2) == MEM_read32(ip)) ) {
                    const BYTE* const repEnd2 = repIndex2 < dictLimit ? dictEnd : iend;
                    size_t const repLength2 = ZSTD_count_2segments(ip+4, repMatch2+4, iend, repEnd2, lowPrefixPtr) + 4;
                    U32 tmpOffset = offset_2; offset_2 = offset_1; offset_1 = tmpOffset;   /* swap offset_2 <=> offset_1 */
                    ZSTD_storeSeq(seqStorePtr, 0, anchor, 0, repLength2-MINMATCH);
                    hashSmall[ZSTD_hashPtr(ip, hBitsS, mls)] = current2;
                    hashLong[ZSTD_hashPtr(ip, hBitsL, 8)] = current2;
                    ip += repLength2;
                    anchor = ip;
                    continue;
                }
                break;
    }   }   }

    /* save reps for next block */
    ctx->repToConfirm[0] = offset_1; ctx->repToConfirm[1] = offset_2;

    /* Last Literals */
    {   size_t const lastLLSize = iend - anchor;
        memcpy(seqStorePtr->lit, anchor, lastLLSize);
        seqStorePtr->lit += lastLLSize;
    }
}


static void ZSTD_compressBlock_doubleFast_extDict(ZSTD_CCtx* ctx,
                         const void* src, size_t srcSize)
{
    U32 const mls = ctx->appliedParams.cParams.searchLength;
    switch(mls)
    {
    default: /* includes case 3 */
    case 4 :
        ZSTD_compressBlock_doubleFast_extDict_generic(ctx, src, srcSize, 4); return;
    case 5 :
        ZSTD_compressBlock_doubleFast_extDict_generic(ctx, src, srcSize, 5); return;
    case 6 :
        ZSTD_compressBlock_doubleFast_extDict_generic(ctx, src, srcSize, 6); return;
    case 7 :
        ZSTD_compressBlock_doubleFast_extDict_generic(ctx, src, srcSize, 7); return;
    }
}


/*-*************************************
*  Binary Tree search
***************************************/
/** ZSTD_insertBt1() : add one or multiple positions to tree.
*   ip : assumed <= iend-8 .
*   @return : nb of positions added */
static U32 ZSTD_insertBt1(ZSTD_CCtx* zc, const BYTE* const ip, const U32 mls, const BYTE* const iend, U32 nbCompares,
                          U32 extDict)
{
    U32*   const hashTable = zc->hashTable;
    U32    const hashLog = zc->appliedParams.cParams.hashLog;
    size_t const h  = ZSTD_hashPtr(ip, hashLog, mls);
    U32*   const bt = zc->chainTable;
    U32    const btLog  = zc->appliedParams.cParams.chainLog - 1;
    U32    const btMask = (1 << btLog) - 1;
    U32 matchIndex = hashTable[h];
    size_t commonLengthSmaller=0, commonLengthLarger=0;
    const BYTE* const base = zc->base;
    const BYTE* const dictBase = zc->dictBase;
    const U32 dictLimit = zc->dictLimit;
    const BYTE* const dictEnd = dictBase + dictLimit;
    const BYTE* const prefixStart = base + dictLimit;
    const BYTE* match;
    const U32 current = (U32)(ip-base);
    const U32 btLow = btMask >= current ? 0 : current - btMask;
    U32* smallerPtr = bt + 2*(current&btMask);
    U32* largerPtr  = smallerPtr + 1;
    U32 dummy32;   /* to be nullified at the end */
    U32 const windowLow = zc->lowLimit;
    U32 matchEndIdx = current+8;
    size_t bestLength = 8;
#ifdef ZSTD_C_PREDICT
    U32 predictedSmall = *(bt + 2*((current-1)&btMask) + 0);
    U32 predictedLarge = *(bt + 2*((current-1)&btMask) + 1);
    predictedSmall += (predictedSmall>0);
    predictedLarge += (predictedLarge>0);
#endif /* ZSTD_C_PREDICT */

    hashTable[h] = current;   /* Update Hash Table */

    while (nbCompares-- && (matchIndex > windowLow)) {
        U32* const nextPtr = bt + 2*(matchIndex & btMask);
        size_t matchLength = MIN(commonLengthSmaller, commonLengthLarger);   /* guaranteed minimum nb of common bytes */

#ifdef ZSTD_C_PREDICT   /* note : can create issues when hlog small <= 11 */
        const U32* predictPtr = bt + 2*((matchIndex-1) & btMask);   /* written this way, as bt is a roll buffer */
        if (matchIndex == predictedSmall) {
            /* no need to check length, result known */
            *smallerPtr = matchIndex;
            if (matchIndex <= btLow) { smallerPtr=&dummy32; break; }   /* beyond tree size, stop the search */
            smallerPtr = nextPtr+1;               /* new "smaller" => larger of match */
            matchIndex = nextPtr[1];              /* new matchIndex larger than previous (closer to current) */
            predictedSmall = predictPtr[1] + (predictPtr[1]>0);
            continue;
        }
        if (matchIndex == predictedLarge) {
            *largerPtr = matchIndex;
            if (matchIndex <= btLow) { largerPtr=&dummy32; break; }   /* beyond tree size, stop the search */
            largerPtr = nextPtr;
            matchIndex = nextPtr[0];
            predictedLarge = predictPtr[0] + (predictPtr[0]>0);
            continue;
        }
#endif
        if ((!extDict) || (matchIndex+matchLength >= dictLimit)) {
            match = base + matchIndex;
            if (match[matchLength] == ip[matchLength])
                matchLength += ZSTD_count(ip+matchLength+1, match+matchLength+1, iend) +1;
        } else {
            match = dictBase + matchIndex;
            matchLength += ZSTD_count_2segments(ip+matchLength, match+matchLength, iend, dictEnd, prefixStart);
            if (matchIndex+matchLength >= dictLimit)
                match = base + matchIndex;   /* to prepare for next usage of match[matchLength] */
        }

        if (matchLength > bestLength) {
            bestLength = matchLength;
            if (matchLength > matchEndIdx - matchIndex)
                matchEndIdx = matchIndex + (U32)matchLength;
        }

        if (ip+matchLength == iend)   /* equal : no way to know if inf or sup */
            break;   /* drop , to guarantee consistency ; miss a bit of compression, but other solutions can corrupt the tree */

        if (match[matchLength] < ip[matchLength]) {  /* necessarily within correct buffer */
            /* match is smaller than current */
            *smallerPtr = matchIndex;             /* update smaller idx */
            commonLengthSmaller = matchLength;    /* all smaller will now have at least this guaranteed common length */
            if (matchIndex <= btLow) { smallerPtr=&dummy32; break; }   /* beyond tree size, stop the search */
            smallerPtr = nextPtr+1;               /* new "smaller" => larger of match */
            matchIndex = nextPtr[1];              /* new matchIndex larger than previous (closer to current) */
        } else {
            /* match is larger than current */
            *largerPtr = matchIndex;
            commonLengthLarger = matchLength;
            if (matchIndex <= btLow) { largerPtr=&dummy32; break; }   /* beyond tree size, stop the search */
            largerPtr = nextPtr;
            matchIndex = nextPtr[0];
    }   }

    *smallerPtr = *largerPtr = 0;
    if (bestLength > 384) return MIN(192, (U32)(bestLength - 384));   /* speed optimization */
    if (matchEndIdx > current + 8) return matchEndIdx - current - 8;
    return 1;
}


static size_t ZSTD_insertBtAndFindBestMatch (
                        ZSTD_CCtx* zc,
                        const BYTE* const ip, const BYTE* const iend,
                        size_t* offsetPtr,
                        U32 nbCompares, const U32 mls,
                        U32 extDict)
{
    U32*   const hashTable = zc->hashTable;
    U32    const hashLog = zc->appliedParams.cParams.hashLog;
    size_t const h  = ZSTD_hashPtr(ip, hashLog, mls);
    U32*   const bt = zc->chainTable;
    U32    const btLog  = zc->appliedParams.cParams.chainLog - 1;
    U32    const btMask = (1 << btLog) - 1;
    U32 matchIndex  = hashTable[h];
    size_t commonLengthSmaller=0, commonLengthLarger=0;
    const BYTE* const base = zc->base;
    const BYTE* const dictBase = zc->dictBase;
    const U32 dictLimit = zc->dictLimit;
    const BYTE* const dictEnd = dictBase + dictLimit;
    const BYTE* const prefixStart = base + dictLimit;
    const U32 current = (U32)(ip-base);
    const U32 btLow = btMask >= current ? 0 : current - btMask;
    const U32 windowLow = zc->lowLimit;
    U32* smallerPtr = bt + 2*(current&btMask);
    U32* largerPtr  = bt + 2*(current&btMask) + 1;
    U32 matchEndIdx = current+8;
    U32 dummy32;   /* to be nullified at the end */
    size_t bestLength = 0;

    hashTable[h] = current;   /* Update Hash Table */

    while (nbCompares-- && (matchIndex > windowLow)) {
        U32* const nextPtr = bt + 2*(matchIndex & btMask);
        size_t matchLength = MIN(commonLengthSmaller, commonLengthLarger);   /* guaranteed minimum nb of common bytes */
        const BYTE* match;

        if ((!extDict) || (matchIndex+matchLength >= dictLimit)) {
            match = base + matchIndex;
            if (match[matchLength] == ip[matchLength])
                matchLength += ZSTD_count(ip+matchLength+1, match+matchLength+1, iend) +1;
        } else {
            match = dictBase + matchIndex;
            matchLength += ZSTD_count_2segments(ip+matchLength, match+matchLength, iend, dictEnd, prefixStart);
            if (matchIndex+matchLength >= dictLimit)
                match = base + matchIndex;   /* to prepare for next usage of match[matchLength] */
        }

        if (matchLength > bestLength) {
            if (matchLength > matchEndIdx - matchIndex)
                matchEndIdx = matchIndex + (U32)matchLength;
            if ( (4*(int)(matchLength-bestLength)) > (int)(ZSTD_highbit32(current-matchIndex+1) - ZSTD_highbit32((U32)offsetPtr[0]+1)) )
                bestLength = matchLength, *offsetPtr = ZSTD_REP_MOVE + current - matchIndex;
            if (ip+matchLength == iend)   /* equal : no way to know if inf or sup */
                break;   /* drop, to guarantee consistency (miss a little bit of compression) */
        }

        if (match[matchLength] < ip[matchLength]) {
            /* match is smaller than current */
            *smallerPtr = matchIndex;             /* update smaller idx */
            commonLengthSmaller = matchLength;    /* all smaller will now have at least this guaranteed common length */
            if (matchIndex <= btLow) { smallerPtr=&dummy32; break; }   /* beyond tree size, stop the search */
            smallerPtr = nextPtr+1;               /* new "smaller" => larger of match */
            matchIndex = nextPtr[1];              /* new matchIndex larger than previous (closer to current) */
        } else {
            /* match is larger than current */
            *largerPtr = matchIndex;
            commonLengthLarger = matchLength;
            if (matchIndex <= btLow) { largerPtr=&dummy32; break; }   /* beyond tree size, stop the search */
            largerPtr = nextPtr;
            matchIndex = nextPtr[0];
    }   }

    *smallerPtr = *largerPtr = 0;

    zc->nextToUpdate = (matchEndIdx > current + 8) ? matchEndIdx - 8 : current+1;
    return bestLength;
}


static void ZSTD_updateTree(ZSTD_CCtx* zc, const BYTE* const ip, const BYTE* const iend, const U32 nbCompares, const U32 mls)
{
    const BYTE* const base = zc->base;
    const U32 target = (U32)(ip - base);
    U32 idx = zc->nextToUpdate;

    while(idx < target)
        idx += ZSTD_insertBt1(zc, base+idx, mls, iend, nbCompares, 0);
}

/** ZSTD_BtFindBestMatch() : Tree updater, providing best match */
static size_t ZSTD_BtFindBestMatch (
                        ZSTD_CCtx* zc,
                        const BYTE* const ip, const BYTE* const iLimit,
                        size_t* offsetPtr,
                        const U32 maxNbAttempts, const U32 mls)
{
    if (ip < zc->base + zc->nextToUpdate) return 0;   /* skipped area */
    ZSTD_updateTree(zc, ip, iLimit, maxNbAttempts, mls);
    return ZSTD_insertBtAndFindBestMatch(zc, ip, iLimit, offsetPtr, maxNbAttempts, mls, 0);
}


static size_t ZSTD_BtFindBestMatch_selectMLS (
                        ZSTD_CCtx* zc,   /* Index table will be updated */
                        const BYTE* ip, const BYTE* const iLimit,
                        size_t* offsetPtr,
                        const U32 maxNbAttempts, const U32 matchLengthSearch)
{
    switch(matchLengthSearch)
    {
    default : /* includes case 3 */
    case 4 : return ZSTD_BtFindBestMatch(zc, ip, iLimit, offsetPtr, maxNbAttempts, 4);
    case 5 : return ZSTD_BtFindBestMatch(zc, ip, iLimit, offsetPtr, maxNbAttempts, 5);
    case 7 :
    case 6 : return ZSTD_BtFindBestMatch(zc, ip, iLimit, offsetPtr, maxNbAttempts, 6);
    }
}


static void ZSTD_updateTree_extDict(ZSTD_CCtx* zc, const BYTE* const ip, const BYTE* const iend, const U32 nbCompares, const U32 mls)
{
    const BYTE* const base = zc->base;
    const U32 target = (U32)(ip - base);
    U32 idx = zc->nextToUpdate;

    while (idx < target) idx += ZSTD_insertBt1(zc, base+idx, mls, iend, nbCompares, 1);
}


/** Tree updater, providing best match */
static size_t ZSTD_BtFindBestMatch_extDict (
                        ZSTD_CCtx* zc,
                        const BYTE* const ip, const BYTE* const iLimit,
                        size_t* offsetPtr,
                        const U32 maxNbAttempts, const U32 mls)
{
    if (ip < zc->base + zc->nextToUpdate) return 0;   /* skipped area */
    ZSTD_updateTree_extDict(zc, ip, iLimit, maxNbAttempts, mls);
    return ZSTD_insertBtAndFindBestMatch(zc, ip, iLimit, offsetPtr, maxNbAttempts, mls, 1);
}


static size_t ZSTD_BtFindBestMatch_selectMLS_extDict (
                        ZSTD_CCtx* zc,   /* Index table will be updated */
                        const BYTE* ip, const BYTE* const iLimit,
                        size_t* offsetPtr,
                        const U32 maxNbAttempts, const U32 matchLengthSearch)
{
    switch(matchLengthSearch)
    {
    default : /* includes case 3 */
    case 4 : return ZSTD_BtFindBestMatch_extDict(zc, ip, iLimit, offsetPtr, maxNbAttempts, 4);
    case 5 : return ZSTD_BtFindBestMatch_extDict(zc, ip, iLimit, offsetPtr, maxNbAttempts, 5);
    case 7 :
    case 6 : return ZSTD_BtFindBestMatch_extDict(zc, ip, iLimit, offsetPtr, maxNbAttempts, 6);
    }
}



/* *********************************
*  Hash Chain
***********************************/
#define NEXT_IN_CHAIN(d, mask)   chainTable[(d) & mask]

/* Update chains up to ip (excluded)
   Assumption : always within prefix (i.e. not within extDict) */
FORCE_INLINE
U32 ZSTD_insertAndFindFirstIndex (ZSTD_CCtx* zc, const BYTE* ip, U32 mls)
{
    U32* const hashTable  = zc->hashTable;
    const U32 hashLog = zc->appliedParams.cParams.hashLog;
    U32* const chainTable = zc->chainTable;
    const U32 chainMask = (1 << zc->appliedParams.cParams.chainLog) - 1;
    const BYTE* const base = zc->base;
    const U32 target = (U32)(ip - base);
    U32 idx = zc->nextToUpdate;

    while(idx < target) { /* catch up */
        size_t const h = ZSTD_hashPtr(base+idx, hashLog, mls);
        NEXT_IN_CHAIN(idx, chainMask) = hashTable[h];
        hashTable[h] = idx;
        idx++;
    }

    zc->nextToUpdate = target;
    return hashTable[ZSTD_hashPtr(ip, hashLog, mls)];
}


/* inlining is important to hardwire a hot branch (template emulation) */
FORCE_INLINE
size_t ZSTD_HcFindBestMatch_generic (
                        ZSTD_CCtx* zc,   /* Index table will be updated */
                        const BYTE* const ip, const BYTE* const iLimit,
                        size_t* offsetPtr,
                        const U32 maxNbAttempts, const U32 mls, const U32 extDict)
{
    U32* const chainTable = zc->chainTable;
    const U32 chainSize = (1 << zc->appliedParams.cParams.chainLog);
    const U32 chainMask = chainSize-1;
    const BYTE* const base = zc->base;
    const BYTE* const dictBase = zc->dictBase;
    const U32 dictLimit = zc->dictLimit;
    const BYTE* const prefixStart = base + dictLimit;
    const BYTE* const dictEnd = dictBase + dictLimit;
    const U32 lowLimit = zc->lowLimit;
    const U32 current = (U32)(ip-base);
    const U32 minChain = current > chainSize ? current - chainSize : 0;
    int nbAttempts=maxNbAttempts;
    size_t ml=4-1;

    /* HC4 match finder */
    U32 matchIndex = ZSTD_insertAndFindFirstIndex (zc, ip, mls);

    for ( ; (matchIndex>lowLimit) & (nbAttempts>0) ; nbAttempts--) {
        const BYTE* match;
        size_t currentMl=0;
        if ((!extDict) || matchIndex >= dictLimit) {
            match = base + matchIndex;
            if (match[ml] == ip[ml])   /* potentially better */
                currentMl = ZSTD_count(ip, match, iLimit);
        } else {
            match = dictBase + matchIndex;
            if (MEM_read32(match) == MEM_read32(ip))   /* assumption : matchIndex <= dictLimit-4 (by table construction) */
                currentMl = ZSTD_count_2segments(ip+4, match+4, iLimit, dictEnd, prefixStart) + 4;
        }

        /* save best solution */
        if (currentMl > ml) {
            ml = currentMl;
            *offsetPtr = current - matchIndex + ZSTD_REP_MOVE;
            if (ip+currentMl == iLimit) break; /* best possible, avoids read overflow on next attempt */
        }

        if (matchIndex <= minChain) break;
        matchIndex = NEXT_IN_CHAIN(matchIndex, chainMask);
    }

    return ml;
}


FORCE_INLINE size_t ZSTD_HcFindBestMatch_selectMLS (
                        ZSTD_CCtx* zc,
                        const BYTE* ip, const BYTE* const iLimit,
                        size_t* offsetPtr,
                        const U32 maxNbAttempts, const U32 matchLengthSearch)
{
    switch(matchLengthSearch)
    {
    default : /* includes case 3 */
    case 4 : return ZSTD_HcFindBestMatch_generic(zc, ip, iLimit, offsetPtr, maxNbAttempts, 4, 0);
    case 5 : return ZSTD_HcFindBestMatch_generic(zc, ip, iLimit, offsetPtr, maxNbAttempts, 5, 0);
    case 7 :
    case 6 : return ZSTD_HcFindBestMatch_generic(zc, ip, iLimit, offsetPtr, maxNbAttempts, 6, 0);
    }
}


FORCE_INLINE size_t ZSTD_HcFindBestMatch_extDict_selectMLS (
                        ZSTD_CCtx* zc,
                        const BYTE* ip, const BYTE* const iLimit,
                        size_t* offsetPtr,
                        const U32 maxNbAttempts, const U32 matchLengthSearch)
{
    switch(matchLengthSearch)
    {
    default : /* includes case 3 */
    case 4 : return ZSTD_HcFindBestMatch_generic(zc, ip, iLimit, offsetPtr, maxNbAttempts, 4, 1);
    case 5 : return ZSTD_HcFindBestMatch_generic(zc, ip, iLimit, offsetPtr, maxNbAttempts, 5, 1);
    case 7 :
    case 6 : return ZSTD_HcFindBestMatch_generic(zc, ip, iLimit, offsetPtr, maxNbAttempts, 6, 1);
    }
}


/* *******************************
*  Common parser - lazy strategy
*********************************/
FORCE_INLINE
void ZSTD_compressBlock_lazy_generic(ZSTD_CCtx* ctx,
                                     const void* src, size_t srcSize,
                                     const U32 searchMethod, const U32 depth)
{
    seqStore_t* seqStorePtr = &(ctx->seqStore);
    const BYTE* const istart = (const BYTE*)src;
    const BYTE* ip = istart;
    const BYTE* anchor = istart;
    const BYTE* const iend = istart + srcSize;
    const BYTE* const ilimit = iend - 8;
    const BYTE* const base = ctx->base + ctx->dictLimit;

    U32 const maxSearches = 1 << ctx->appliedParams.cParams.searchLog;
    U32 const mls = ctx->appliedParams.cParams.searchLength;

    typedef size_t (*searchMax_f)(ZSTD_CCtx* zc, const BYTE* ip, const BYTE* iLimit,
                        size_t* offsetPtr,
                        U32 maxNbAttempts, U32 matchLengthSearch);
    searchMax_f const searchMax = searchMethod ? ZSTD_BtFindBestMatch_selectMLS : ZSTD_HcFindBestMatch_selectMLS;
    U32 offset_1 = ctx->rep[0], offset_2 = ctx->rep[1], savedOffset=0;

    /* init */
    ip += (ip==base);
    ctx->nextToUpdate3 = ctx->nextToUpdate;
    {   U32 const maxRep = (U32)(ip-base);
        if (offset_2 > maxRep) savedOffset = offset_2, offset_2 = 0;
        if (offset_1 > maxRep) savedOffset = offset_1, offset_1 = 0;
    }

    /* Match Loop */
    while (ip < ilimit) {
        size_t matchLength=0;
        size_t offset=0;
        const BYTE* start=ip+1;

        /* check repCode */
        if ((offset_1>0) & (MEM_read32(ip+1) == MEM_read32(ip+1 - offset_1))) {
            /* repcode : we take it */
            matchLength = ZSTD_count(ip+1+4, ip+1+4-offset_1, iend) + 4;
            if (depth==0) goto _storeSequence;
        }

        /* first search (depth 0) */
        {   size_t offsetFound = 99999999;
            size_t const ml2 = searchMax(ctx, ip, iend, &offsetFound, maxSearches, mls);
            if (ml2 > matchLength)
                matchLength = ml2, start = ip, offset=offsetFound;
        }

        if (matchLength < 4) {
            ip += ((ip-anchor) >> g_searchStrength) + 1;   /* jump faster over incompressible sections */
            continue;
        }

        /* let's try to find a better solution */
        if (depth>=1)
        while (ip<ilimit) {
            ip ++;
            if ((offset) && ((offset_1>0) & (MEM_read32(ip) == MEM_read32(ip - offset_1)))) {
                size_t const mlRep = ZSTD_count(ip+4, ip+4-offset_1, iend) + 4;
                int const gain2 = (int)(mlRep * 3);
                int const gain1 = (int)(matchLength*3 - ZSTD_highbit32((U32)offset+1) + 1);
                if ((mlRep >= 4) && (gain2 > gain1))
                    matchLength = mlRep, offset = 0, start = ip;
            }
            {   size_t offset2=99999999;
                size_t const ml2 = searchMax(ctx, ip, iend, &offset2, maxSearches, mls);
                int const gain2 = (int)(ml2*4 - ZSTD_highbit32((U32)offset2+1));   /* raw approx */
                int const gain1 = (int)(matchLength*4 - ZSTD_highbit32((U32)offset+1) + 4);
                if ((ml2 >= 4) && (gain2 > gain1)) {
                    matchLength = ml2, offset = offset2, start = ip;
                    continue;   /* search a better one */
            }   }

            /* let's find an even better one */
            if ((depth==2) && (ip<ilimit)) {
                ip ++;
                if ((offset) && ((offset_1>0) & (MEM_read32(ip) == MEM_read32(ip - offset_1)))) {
                    size_t const ml2 = ZSTD_count(ip+4, ip+4-offset_1, iend) + 4;
                    int const gain2 = (int)(ml2 * 4);
                    int const gain1 = (int)(matchLength*4 - ZSTD_highbit32((U32)offset+1) + 1);
                    if ((ml2 >= 4) && (gain2 > gain1))
                        matchLength = ml2, offset = 0, start = ip;
                }
                {   size_t offset2=99999999;
                    size_t const ml2 = searchMax(ctx, ip, iend, &offset2, maxSearches, mls);
                    int const gain2 = (int)(ml2*4 - ZSTD_highbit32((U32)offset2+1));   /* raw approx */
                    int const gain1 = (int)(matchLength*4 - ZSTD_highbit32((U32)offset+1) + 7);
                    if ((ml2 >= 4) && (gain2 > gain1)) {
                        matchLength = ml2, offset = offset2, start = ip;
                        continue;
            }   }   }
            break;  /* nothing found : store previous solution */
        }

        /* NOTE:
         * start[-offset+ZSTD_REP_MOVE-1] is undefined behavior.
         * (-offset+ZSTD_REP_MOVE-1) is unsigned, and is added to start, which
         * overflows the pointer, which is undefined behavior.
         */
        /* catch up */
        if (offset) {
            while ( (start > anchor)
                 && (start > base+offset-ZSTD_REP_MOVE)
                 && (start[-1] == (start-offset+ZSTD_REP_MOVE)[-1]) )  /* only search for offset within prefix */
                { start--; matchLength++; }
            offset_2 = offset_1; offset_1 = (U32)(offset - ZSTD_REP_MOVE);
        }
        /* store sequence */
_storeSequence:
        {   size_t const litLength = start - anchor;
            ZSTD_storeSeq(seqStorePtr, litLength, anchor, (U32)offset, matchLength-MINMATCH);
            anchor = ip = start + matchLength;
        }

        /* check immediate repcode */
        while ( (ip <= ilimit)
             && ((offset_2>0)
             & (MEM_read32(ip) == MEM_read32(ip - offset_2)) )) {
            /* store sequence */
            matchLength = ZSTD_count(ip+4, ip+4-offset_2, iend) + 4;
            offset = offset_2; offset_2 = offset_1; offset_1 = (U32)offset; /* swap repcodes */
            ZSTD_storeSeq(seqStorePtr, 0, anchor, 0, matchLength-MINMATCH);
            ip += matchLength;
            anchor = ip;
            continue;   /* faster when present ... (?) */
    }   }

    /* Save reps for next block */
    ctx->repToConfirm[0] = offset_1 ? offset_1 : savedOffset;
    ctx->repToConfirm[1] = offset_2 ? offset_2 : savedOffset;

    /* Last Literals */
    {   size_t const lastLLSize = iend - anchor;
        memcpy(seqStorePtr->lit, anchor, lastLLSize);
        seqStorePtr->lit += lastLLSize;
    }
}


static void ZSTD_compressBlock_btlazy2(ZSTD_CCtx* ctx, const void* src, size_t srcSize)
{
    ZSTD_compressBlock_lazy_generic(ctx, src, srcSize, 1, 2);
}

static void ZSTD_compressBlock_lazy2(ZSTD_CCtx* ctx, const void* src, size_t srcSize)
{
    ZSTD_compressBlock_lazy_generic(ctx, src, srcSize, 0, 2);
}

static void ZSTD_compressBlock_lazy(ZSTD_CCtx* ctx, const void* src, size_t srcSize)
{
    ZSTD_compressBlock_lazy_generic(ctx, src, srcSize, 0, 1);
}

static void ZSTD_compressBlock_greedy(ZSTD_CCtx* ctx, const void* src, size_t srcSize)
{
    ZSTD_compressBlock_lazy_generic(ctx, src, srcSize, 0, 0);
}


FORCE_INLINE
void ZSTD_compressBlock_lazy_extDict_generic(ZSTD_CCtx* ctx,
                                     const void* src, size_t srcSize,
                                     const U32 searchMethod, const U32 depth)
{
    seqStore_t* seqStorePtr = &(ctx->seqStore);
    const BYTE* const istart = (const BYTE*)src;
    const BYTE* ip = istart;
    const BYTE* anchor = istart;
    const BYTE* const iend = istart + srcSize;
    const BYTE* const ilimit = iend - 8;
    const BYTE* const base = ctx->base;
    const U32 dictLimit = ctx->dictLimit;
    const U32 lowestIndex = ctx->lowLimit;
    const BYTE* const prefixStart = base + dictLimit;
    const BYTE* const dictBase = ctx->dictBase;
    const BYTE* const dictEnd  = dictBase + dictLimit;
    const BYTE* const dictStart  = dictBase + ctx->lowLimit;

    const U32 maxSearches = 1 << ctx->appliedParams.cParams.searchLog;
    const U32 mls = ctx->appliedParams.cParams.searchLength;

    typedef size_t (*searchMax_f)(ZSTD_CCtx* zc, const BYTE* ip, const BYTE* iLimit,
                        size_t* offsetPtr,
                        U32 maxNbAttempts, U32 matchLengthSearch);
    searchMax_f searchMax = searchMethod ? ZSTD_BtFindBestMatch_selectMLS_extDict : ZSTD_HcFindBestMatch_extDict_selectMLS;

    U32 offset_1 = ctx->rep[0], offset_2 = ctx->rep[1];

    /* init */
    ctx->nextToUpdate3 = ctx->nextToUpdate;
    ip += (ip == prefixStart);

    /* Match Loop */
    while (ip < ilimit) {
        size_t matchLength=0;
        size_t offset=0;
        const BYTE* start=ip+1;
        U32 current = (U32)(ip-base);

        /* check repCode */
        {   const U32 repIndex = (U32)(current+1 - offset_1);
            const BYTE* const repBase = repIndex < dictLimit ? dictBase : base;
            const BYTE* const repMatch = repBase + repIndex;
            if (((U32)((dictLimit-1) - repIndex) >= 3) & (repIndex > lowestIndex))   /* intentional overflow */
            if (MEM_read32(ip+1) == MEM_read32(repMatch)) {
                /* repcode detected we should take it */
                const BYTE* const repEnd = repIndex < dictLimit ? dictEnd : iend;
                matchLength = ZSTD_count_2segments(ip+1+4, repMatch+4, iend, repEnd, prefixStart) + 4;
                if (depth==0) goto _storeSequence;
        }   }

        /* first search (depth 0) */
        {   size_t offsetFound = 99999999;
            size_t const ml2 = searchMax(ctx, ip, iend, &offsetFound, maxSearches, mls);
            if (ml2 > matchLength)
                matchLength = ml2, start = ip, offset=offsetFound;
        }

         if (matchLength < 4) {
            ip += ((ip-anchor) >> g_searchStrength) + 1;   /* jump faster over incompressible sections */
            continue;
        }

        /* let's try to find a better solution */
        if (depth>=1)
        while (ip<ilimit) {
            ip ++;
            current++;
            /* check repCode */
            if (offset) {
                const U32 repIndex = (U32)(current - offset_1);
                const BYTE* const repBase = repIndex < dictLimit ? dictBase : base;
                const BYTE* const repMatch = repBase + repIndex;
                if (((U32)((dictLimit-1) - repIndex) >= 3) & (repIndex > lowestIndex))  /* intentional overflow */
                if (MEM_read32(ip) == MEM_read32(repMatch)) {
                    /* repcode detected */
                    const BYTE* const repEnd = repIndex < dictLimit ? dictEnd : iend;
                    size_t const repLength = ZSTD_count_2segments(ip+4, repMatch+4, iend, repEnd, prefixStart) + 4;
                    int const gain2 = (int)(repLength * 3);
                    int const gain1 = (int)(matchLength*3 - ZSTD_highbit32((U32)offset+1) + 1);
                    if ((repLength >= 4) && (gain2 > gain1))
                        matchLength = repLength, offset = 0, start = ip;
            }   }

            /* search match, depth 1 */
            {   size_t offset2=99999999;
                size_t const ml2 = searchMax(ctx, ip, iend, &offset2, maxSearches, mls);
                int const gain2 = (int)(ml2*4 - ZSTD_highbit32((U32)offset2+1));   /* raw approx */
                int const gain1 = (int)(matchLength*4 - ZSTD_highbit32((U32)offset+1) + 4);
                if ((ml2 >= 4) && (gain2 > gain1)) {
                    matchLength = ml2, offset = offset2, start = ip;
                    continue;   /* search a better one */
            }   }

            /* let's find an even better one */
            if ((depth==2) && (ip<ilimit)) {
                ip ++;
                current++;
                /* check repCode */
                if (offset) {
                    const U32 repIndex = (U32)(current - offset_1);
                    const BYTE* const repBase = repIndex < dictLimit ? dictBase : base;
                    const BYTE* const repMatch = repBase + repIndex;
                    if (((U32)((dictLimit-1) - repIndex) >= 3) & (repIndex > lowestIndex))  /* intentional overflow */
                    if (MEM_read32(ip) == MEM_read32(repMatch)) {
                        /* repcode detected */
                        const BYTE* const repEnd = repIndex < dictLimit ? dictEnd : iend;
                        size_t const repLength = ZSTD_count_2segments(ip+4, repMatch+4, iend, repEnd, prefixStart) + 4;
                        int const gain2 = (int)(repLength * 4);
                        int const gain1 = (int)(matchLength*4 - ZSTD_highbit32((U32)offset+1) + 1);
                        if ((repLength >= 4) && (gain2 > gain1))
                            matchLength = repLength, offset = 0, start = ip;
                }   }

                /* search match, depth 2 */
                {   size_t offset2=99999999;
                    size_t const ml2 = searchMax(ctx, ip, iend, &offset2, maxSearches, mls);
                    int const gain2 = (int)(ml2*4 - ZSTD_highbit32((U32)offset2+1));   /* raw approx */
                    int const gain1 = (int)(matchLength*4 - ZSTD_highbit32((U32)offset+1) + 7);
                    if ((ml2 >= 4) && (gain2 > gain1)) {
                        matchLength = ml2, offset = offset2, start = ip;
                        continue;
            }   }   }
            break;  /* nothing found : store previous solution */
        }

        /* catch up */
        if (offset) {
            U32 const matchIndex = (U32)((start-base) - (offset - ZSTD_REP_MOVE));
            const BYTE* match = (matchIndex < dictLimit) ? dictBase + matchIndex : base + matchIndex;
            const BYTE* const mStart = (matchIndex < dictLimit) ? dictStart : prefixStart;
            while ((start>anchor) && (match>mStart) && (start[-1] == match[-1])) { start--; match--; matchLength++; }  /* catch up */
            offset_2 = offset_1; offset_1 = (U32)(offset - ZSTD_REP_MOVE);
        }

        /* store sequence */
_storeSequence:
        {   size_t const litLength = start - anchor;
            ZSTD_storeSeq(seqStorePtr, litLength, anchor, (U32)offset, matchLength-MINMATCH);
            anchor = ip = start + matchLength;
        }

        /* check immediate repcode */
        while (ip <= ilimit) {
            const U32 repIndex = (U32)((ip-base) - offset_2);
            const BYTE* const repBase = repIndex < dictLimit ? dictBase : base;
            const BYTE* const repMatch = repBase + repIndex;
            if (((U32)((dictLimit-1) - repIndex) >= 3) & (repIndex > lowestIndex))  /* intentional overflow */
            if (MEM_read32(ip) == MEM_read32(repMatch)) {
                /* repcode detected we should take it */
                const BYTE* const repEnd = repIndex < dictLimit ? dictEnd : iend;
                matchLength = ZSTD_count_2segments(ip+4, repMatch+4, iend, repEnd, prefixStart) + 4;
                offset = offset_2; offset_2 = offset_1; offset_1 = (U32)offset;   /* swap offset history */
                ZSTD_storeSeq(seqStorePtr, 0, anchor, 0, matchLength-MINMATCH);
                ip += matchLength;
                anchor = ip;
                continue;   /* faster when present ... (?) */
            }
            break;
    }   }

    /* Save reps for next block */
    ctx->repToConfirm[0] = offset_1; ctx->repToConfirm[1] = offset_2;

    /* Last Literals */
    {   size_t const lastLLSize = iend - anchor;
        memcpy(seqStorePtr->lit, anchor, lastLLSize);
        seqStorePtr->lit += lastLLSize;
    }
}


void ZSTD_compressBlock_greedy_extDict(ZSTD_CCtx* ctx, const void* src, size_t srcSize)
{
    ZSTD_compressBlock_lazy_extDict_generic(ctx, src, srcSize, 0, 0);
}

static void ZSTD_compressBlock_lazy_extDict(ZSTD_CCtx* ctx, const void* src, size_t srcSize)
{
    ZSTD_compressBlock_lazy_extDict_generic(ctx, src, srcSize, 0, 1);
}

static void ZSTD_compressBlock_lazy2_extDict(ZSTD_CCtx* ctx, const void* src, size_t srcSize)
{
    ZSTD_compressBlock_lazy_extDict_generic(ctx, src, srcSize, 0, 2);
}

static void ZSTD_compressBlock_btlazy2_extDict(ZSTD_CCtx* ctx, const void* src, size_t srcSize)
{
    ZSTD_compressBlock_lazy_extDict_generic(ctx, src, srcSize, 1, 2);
}


/* The optimal parser */
#include "zstd_opt.h"

static void ZSTD_compressBlock_btopt(ZSTD_CCtx* ctx, const void* src, size_t srcSize)
{
#ifdef ZSTD_OPT_H_91842398743
    ZSTD_compressBlock_opt_generic(ctx, src, srcSize, 0);
#else
    (void)ctx; (void)src; (void)srcSize;
    return;
#endif
}

static void ZSTD_compressBlock_btultra(ZSTD_CCtx* ctx, const void* src, size_t srcSize)
{
#ifdef ZSTD_OPT_H_91842398743
    ZSTD_compressBlock_opt_generic(ctx, src, srcSize, 1);
#else
    (void)ctx; (void)src; (void)srcSize;
    return;
#endif
}

static void ZSTD_compressBlock_btopt_extDict(ZSTD_CCtx* ctx, const void* src, size_t srcSize)
{
#ifdef ZSTD_OPT_H_91842398743
    ZSTD_compressBlock_opt_extDict_generic(ctx, src, srcSize, 0);
#else
    (void)ctx; (void)src; (void)srcSize;
    return;
#endif
}

static void ZSTD_compressBlock_btultra_extDict(ZSTD_CCtx* ctx, const void* src, size_t srcSize)
{
#ifdef ZSTD_OPT_H_91842398743
    ZSTD_compressBlock_opt_extDict_generic(ctx, src, srcSize, 1);
#else
    (void)ctx; (void)src; (void)srcSize;
    return;
#endif
}


/* ZSTD_selectBlockCompressor() :
 * assumption : strat is a valid strategy */
typedef void (*ZSTD_blockCompressor) (ZSTD_CCtx* ctx, const void* src, size_t srcSize);
static ZSTD_blockCompressor ZSTD_selectBlockCompressor(ZSTD_strategy strat, int extDict)
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


static size_t ZSTD_compressBlock_internal(ZSTD_CCtx* zc, void* dst, size_t dstCapacity, const void* src, size_t srcSize)
{
    ZSTD_blockCompressor const blockCompressor = ZSTD_selectBlockCompressor(zc->appliedParams.cParams.strategy, zc->lowLimit < zc->dictLimit);
    const BYTE* const base = zc->base;
    const BYTE* const istart = (const BYTE*)src;
    const U32 current = (U32)(istart-base);
    if (srcSize < MIN_CBLOCK_SIZE+ZSTD_blockHeaderSize+1) return 0;   /* don't even attempt compression below a certain srcSize */
    ZSTD_resetSeqStore(&(zc->seqStore));
    if (current > zc->nextToUpdate + 384)
        zc->nextToUpdate = current - MIN(192, (U32)(current - zc->nextToUpdate - 384));   /* limited update after finding a very long match */
    blockCompressor(zc, src, srcSize);
    return ZSTD_compressSequences(zc, dst, dstCapacity, srcSize);
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
    U32 const maxDist = 1 << cctx->appliedParams.cParams.windowLog;

    if (cctx->appliedParams.fParams.checksumFlag && srcSize)
        XXH64_update(&cctx->xxhState, src, srcSize);

    while (remaining) {
        U32 const lastBlock = lastFrameChunk & (blockSize >= remaining);
        size_t cSize;

        if (dstCapacity < ZSTD_blockHeaderSize + MIN_CBLOCK_SIZE)
            return ERROR(dstSize_tooSmall);   /* not enough space to store compressed block */
        if (remaining < blockSize) blockSize = remaining;

        /* preemptive overflow correction */
        if (cctx->lowLimit > (3U<<29)) {
            U32 const cycleMask = (1 << ZSTD_cycleLog(cctx->appliedParams.cParams.hashLog, cctx->appliedParams.cParams.strategy)) - 1;
            U32 const current = (U32)(ip - cctx->base);
            U32 const newCurrent = (current & cycleMask) + (1 << cctx->appliedParams.cParams.windowLog);
            U32 const correction = current - newCurrent;
            ZSTD_STATIC_ASSERT(ZSTD_WINDOWLOG_MAX_64 <= 30);
            ZSTD_reduceIndex(cctx, correction);
            cctx->base += correction;
            cctx->dictBase += correction;
            cctx->lowLimit -= correction;
            cctx->dictLimit -= correction;
            if (cctx->nextToUpdate < correction) cctx->nextToUpdate = 0;
            else cctx->nextToUpdate -= correction;
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
                                    ZSTD_parameters params, U64 pledgedSrcSize, U32 dictID)
{   BYTE* const op = (BYTE*)dst;
    U32   const dictIDSizeCodeLength = (dictID>0) + (dictID>=256) + (dictID>=65536);   /* 0-3 */
    U32   const dictIDSizeCode = params.fParams.noDictIDFlag ? 0 : dictIDSizeCodeLength;   /* 0-3 */
    U32   const checksumFlag = params.fParams.checksumFlag>0;
    U32   const windowSize = 1U << params.cParams.windowLog;
    U32   const singleSegment = params.fParams.contentSizeFlag && (windowSize >= pledgedSrcSize);
    BYTE  const windowLogByte = (BYTE)((params.cParams.windowLog - ZSTD_WINDOWLOG_ABSOLUTEMIN) << 3);
    U32   const fcsCode = params.fParams.contentSizeFlag ?
                     (pledgedSrcSize>=256) + (pledgedSrcSize>=65536+256) + (pledgedSrcSize>=0xFFFFFFFFU) : 0;  /* 0-3 */
    BYTE  const frameHeaderDecriptionByte = (BYTE)(dictIDSizeCode + (checksumFlag<<2) + (singleSegment<<5) + (fcsCode<<6) );
    size_t pos;

    if (dstCapacity < ZSTD_frameHeaderSize_max) return ERROR(dstSize_tooSmall);
    DEBUGLOG(5, "ZSTD_writeFrameHeader : dictIDFlag : %u ; dictID : %u ; dictIDSizeCode : %u",
                !params.fParams.noDictIDFlag, dictID,  dictIDSizeCode);

    MEM_writeLE32(dst, ZSTD_MAGICNUMBER);
    op[4] = frameHeaderDecriptionByte; pos=5;
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
    U32 const cLevel = cctx->compressionLevel;
    ZSTD_compressionParameters cParams = (cLevel == ZSTD_CLEVEL_CUSTOM) ?
                                        cctx->appliedParams.cParams :
                                        ZSTD_getCParams(cLevel, 0, 0);
    return MIN (ZSTD_BLOCKSIZE_MAX, 1 << cParams.windowLog);
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
    zc->loadedDictEnd = zc->forceWindow ? 0 : (U32)(iend - zc->base);

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
            ZSTD_updateTree(zc, iend-HASH_READ_SIZE, iend, 1 << zc->appliedParams.cParams.searchLog, zc->appliedParams.cParams.searchLength);
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
    BYTE scratchBuffer[1<<MAX(MLFSELog,LLFSELog)];

    dictPtr += 4;   /* skip magic number */
    cctx->dictID = cctx->appliedParams.fParams.noDictIDFlag ? 0 :  MEM_readLE32(dictPtr);
    dictPtr += 4;

    {   size_t const hufHeaderSize = HUF_readCTable(cctx->hufCTable, 255, dictPtr, dictEnd-dictPtr);
        if (HUF_isError(hufHeaderSize)) return ERROR(dictionary_corrupted);
        dictPtr += hufHeaderSize;
    }

    {   unsigned offcodeLog;
        size_t const offcodeHeaderSize = FSE_readNCount(offcodeNCount, &offcodeMaxValue, &offcodeLog, dictPtr, dictEnd-dictPtr);
        if (FSE_isError(offcodeHeaderSize)) return ERROR(dictionary_corrupted);
        if (offcodeLog > OffFSELog) return ERROR(dictionary_corrupted);
        /* Defer checking offcodeMaxValue because we need to know the size of the dictionary content */
        CHECK_E( FSE_buildCTable_wksp(cctx->offcodeCTable, offcodeNCount, offcodeMaxValue, offcodeLog, scratchBuffer, sizeof(scratchBuffer)),
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
        CHECK_E( FSE_buildCTable_wksp(cctx->matchlengthCTable, matchlengthNCount, matchlengthMaxValue, matchlengthLog, scratchBuffer, sizeof(scratchBuffer)),
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
        CHECK_E( FSE_buildCTable_wksp(cctx->litlengthCTable, litlengthNCount, litlengthMaxValue, litlengthLog, scratchBuffer, sizeof(scratchBuffer)),
                 dictionary_corrupted);
        dictPtr += litlengthHeaderSize;
    }

    if (dictPtr+12 > dictEnd) return ERROR(dictionary_corrupted);
    cctx->rep[0] = MEM_readLE32(dictPtr+0);
    cctx->rep[1] = MEM_readLE32(dictPtr+4);
    cctx->rep[2] = MEM_readLE32(dictPtr+8);
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
                if (cctx->rep[u] == 0) return ERROR(dictionary_corrupted);
                if (cctx->rep[u] > dictContentSize) return ERROR(dictionary_corrupted);
        }   }

        cctx->fseCTables_ready = 1;
        cctx->hufCTable_repeatMode = HUF_repeat_valid;
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
                                   ZSTD_parameters params, U64 pledgedSrcSize,
                                   ZSTD_buffered_policy_e zbuff)
{
    DEBUGLOG(4, "ZSTD_compressBegin_internal");
    DEBUGLOG(4, "dict ? %s", dict ? "dict" : (cdict ? "cdict" : "none"));
    DEBUGLOG(4, "dictMode : %u", (U32)dictMode);
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


/*! ZSTD_compressBegin_advanced() :
*   @return : 0, or an error code */
size_t ZSTD_compressBegin_advanced(ZSTD_CCtx* cctx,
                             const void* dict, size_t dictSize,
                                   ZSTD_parameters params, unsigned long long pledgedSrcSize)
{
    /* compression parameters verification and optimization */
    CHECK_F(ZSTD_checkCParams(params.cParams));
    return ZSTD_compressBegin_internal(cctx, dict, dictSize, ZSTD_dm_auto, NULL,
                                    params, pledgedSrcSize, ZSTDb_not_buffered);
}


size_t ZSTD_compressBegin_usingDict(ZSTD_CCtx* cctx, const void* dict, size_t dictSize, int compressionLevel)
{
    ZSTD_parameters const params = ZSTD_getParams(compressionLevel, 0, dictSize);
    return ZSTD_compressBegin_internal(cctx, dict, dictSize, ZSTD_dm_auto, NULL,
                                       params, 0, ZSTDb_not_buffered);
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
        DEBUGLOG(5, "end of frame : controlling src size");
        if (cctx->pledgedSrcSizePlusOne != cctx->consumedSrcSize+1) {
            DEBUGLOG(5, "error : pledgedSrcSize = %u, while realSrcSize = %u",
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
    CHECK_F( ZSTD_compressBegin_internal(cctx, dict, dictSize, ZSTD_dm_auto, NULL,
                                         params, srcSize, ZSTDb_not_buffered) );
    return ZSTD_compressEnd(cctx, dst,  dstCapacity, src, srcSize);
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
size_t ZSTD_estimateCDictSize_advanced(size_t dictSize, ZSTD_compressionParameters cParams, unsigned byReference)
{
    DEBUGLOG(5, "sizeof(ZSTD_CDict) : %u", (U32)sizeof(ZSTD_CDict));
    DEBUGLOG(5, "CCtx estimate : %u", (U32)ZSTD_estimateCCtxSize_advanced(cParams));
    return sizeof(ZSTD_CDict) + ZSTD_estimateCCtxSize_advanced(cParams)
           + (byReference ? 0 : dictSize);
}

size_t ZSTD_estimateCDictSize(size_t dictSize, int compressionLevel)
{
    ZSTD_compressionParameters const cParams = ZSTD_getCParams(compressionLevel, 0, dictSize);
    return ZSTD_estimateCDictSize_advanced(dictSize, cParams, 0);
}

size_t ZSTD_sizeof_CDict(const ZSTD_CDict* cdict)
{
    if (cdict==NULL) return 0;   /* support sizeof on NULL */
    DEBUGLOG(5, "sizeof(*cdict) : %u", (U32)sizeof(*cdict));
    DEBUGLOG(5, "ZSTD_sizeof_CCtx : %u", (U32)ZSTD_sizeof_CCtx(cdict->refContext));
    return ZSTD_sizeof_CCtx(cdict->refContext) + (cdict->dictBuffer ? cdict->dictContentSize : 0) + sizeof(*cdict);
}

static ZSTD_parameters ZSTD_makeParams(ZSTD_compressionParameters cParams, ZSTD_frameParameters fParams)
{
    ZSTD_parameters params;
    params.cParams = cParams;
    params.fParams = fParams;
    return params;
}

static size_t ZSTD_initCDict_internal(
                    ZSTD_CDict* cdict,
              const void* dictBuffer, size_t dictSize,
                    unsigned byReference, ZSTD_dictMode_e dictMode,
                    ZSTD_compressionParameters cParams)
{
    DEBUGLOG(5, "ZSTD_initCDict_internal, mode %u", (U32)dictMode);
    if ((byReference) || (!dictBuffer) || (!dictSize)) {
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

    {   ZSTD_frameParameters const fParams = { 0 /* contentSizeFlag */,
                    0 /* checksumFlag */, 0 /* noDictIDFlag */ };  /* dummy */
        ZSTD_parameters const params = ZSTD_makeParams(cParams, fParams);
        CHECK_F( ZSTD_compressBegin_internal(cdict->refContext,
                                        cdict->dictContent, dictSize, dictMode,
                                        NULL,
                                        params, ZSTD_CONTENTSIZE_UNKNOWN,
                                        ZSTDb_not_buffered) );
    }

    return 0;
}

ZSTD_CDict* ZSTD_createCDict_advanced(const void* dictBuffer, size_t dictSize,
                                      unsigned byReference, ZSTD_dictMode_e dictMode,
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
                                        byReference, dictMode,
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
                                     0 /* byReference */, ZSTD_dm_auto,
                                     cParams, ZSTD_defaultCMem);
}

ZSTD_CDict* ZSTD_createCDict_byReference(const void* dict, size_t dictSize, int compressionLevel)
{
    ZSTD_compressionParameters cParams = ZSTD_getCParams(compressionLevel, 0, dictSize);
    return ZSTD_createCDict_advanced(dict, dictSize,
                                     1 /* byReference */, ZSTD_dm_auto,
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
                                 unsigned byReference, ZSTD_dictMode_e dictMode,
                                 ZSTD_compressionParameters cParams)
{
    size_t const cctxSize = ZSTD_estimateCCtxSize_advanced(cParams);
    size_t const neededSize = sizeof(ZSTD_CDict) + (byReference ? 0 : dictSize)
                            + cctxSize;
    ZSTD_CDict* const cdict = (ZSTD_CDict*) workspace;
    void* ptr;
    DEBUGLOG(5, "(size_t)workspace & 7 : %u", (U32)(size_t)workspace & 7);
    if ((size_t)workspace & 7) return NULL;  /* 8-aligned */
    DEBUGLOG(5, "(workspaceSize < neededSize) : (%u < %u) => %u",
        (U32)workspaceSize, (U32)neededSize, (U32)(workspaceSize < neededSize));
    if (workspaceSize < neededSize) return NULL;

    if (!byReference) {
        memcpy(cdict+1, dict, dictSize);
        dict = cdict+1;
        ptr = (char*)workspace + sizeof(ZSTD_CDict) + dictSize;
    } else {
        ptr = cdict+1;
    }
    cdict->refContext = ZSTD_initStaticCCtx(ptr, cctxSize);

    if (ZSTD_isError( ZSTD_initCDict_internal(cdict,
                                              dict, dictSize,
                                              1 /* byReference */, dictMode,
                                              cParams) ))
        return NULL;

    return cdict;
}

ZSTD_parameters ZSTD_getParamsFromCDict(const ZSTD_CDict* cdict) {
    return ZSTD_getParamsFromCCtx(cdict->refContext);
}

/* ZSTD_compressBegin_usingCDict_advanced() :
 * cdict must be != NULL */
size_t ZSTD_compressBegin_usingCDict_advanced(
    ZSTD_CCtx* const cctx, const ZSTD_CDict* const cdict,
    ZSTD_frameParameters const fParams, unsigned long long const pledgedSrcSize)
{
    if (cdict==NULL) return ERROR(dictionary_wrong);
    {   ZSTD_parameters params = cdict->refContext->appliedParams;
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
                    ZSTD_parameters params, unsigned long long pledgedSrcSize)
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
    ZSTD_parameters params = zcs->requestedParams;
    params.fParams.contentSizeFlag = (pledgedSrcSize > 0);
    DEBUGLOG(5, "ZSTD_resetCStream");
    if (zcs->compressionLevel != ZSTD_CLEVEL_CUSTOM) {
        params.cParams = ZSTD_getCParams(zcs->compressionLevel, pledgedSrcSize, 0 /* dictSize */);
    }
    return ZSTD_resetCStream_internal(zcs, NULL, 0, zcs->dictMode, zcs->cdict, params, pledgedSrcSize);
}

/*! ZSTD_initCStream_internal() :
 *  Note : not static, but hidden (not exposed). Used by zstdmt_compress.c
 *  Assumption 1 : params are valid
 *  Assumption 2 : either dict, or cdict, is defined, not both */
size_t ZSTD_initCStream_internal(ZSTD_CStream* zcs,
                    const void* dict, size_t dictSize, const ZSTD_CDict* cdict,
                    ZSTD_parameters params, unsigned long long pledgedSrcSize)
{
    DEBUGLOG(5, "ZSTD_initCStream_internal");
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
                                            zcs->dictContentByRef, zcs->dictMode,
                                            params.cParams, zcs->customMem);
        zcs->cdict = zcs->cdictLocal;
        if (zcs->cdictLocal == NULL) return ERROR(memory_allocation);
    } else {
        if (cdict) {
            ZSTD_parameters const cdictParams = ZSTD_getParamsFromCDict(cdict);
            params.cParams = cdictParams.cParams;  /* cParams are enforced from cdict */
        }
        ZSTD_freeCDict(zcs->cdictLocal);
        zcs->cdictLocal = NULL;
        zcs->cdict = cdict;
    }

    zcs->requestedParams = params;
    zcs->compressionLevel = ZSTD_CLEVEL_CUSTOM;
    return ZSTD_resetCStream_internal(zcs, NULL, 0, zcs->dictMode, zcs->cdict, params, pledgedSrcSize);
}

/* ZSTD_initCStream_usingCDict_advanced() :
 * same as ZSTD_initCStream_usingCDict(), with control over frame parameters */
size_t ZSTD_initCStream_usingCDict_advanced(ZSTD_CStream* zcs,
                                            const ZSTD_CDict* cdict,
                                            ZSTD_frameParameters fParams,
                                            unsigned long long pledgedSrcSize)
{   /* cannot handle NULL cdict (does not know what to do) */
    if (!cdict) return ERROR(dictionary_wrong);
    {   ZSTD_parameters params = ZSTD_getParamsFromCDict(cdict);
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
    CHECK_F( ZSTD_checkCParams(params.cParams) );
    zcs->requestedParams = params;
    zcs->compressionLevel = ZSTD_CLEVEL_CUSTOM;
    return ZSTD_initCStream_internal(zcs, dict, dictSize, NULL, params, pledgedSrcSize);
}

size_t ZSTD_initCStream_usingDict(ZSTD_CStream* zcs, const void* dict, size_t dictSize, int compressionLevel)
{
    ZSTD_parameters const params = ZSTD_getParams(compressionLevel, 0, dictSize);
    zcs->compressionLevel = compressionLevel;
    return ZSTD_initCStream_internal(zcs, dict, dictSize, NULL, params, 0);
}

size_t ZSTD_initCStream_srcSize(ZSTD_CStream* zcs, int compressionLevel, unsigned long long pledgedSrcSize)
{
    ZSTD_parameters params = ZSTD_getParams(compressionLevel, pledgedSrcSize, 0);
    params.fParams.contentSizeFlag = (pledgedSrcSize>0);
    return ZSTD_initCStream_internal(zcs, NULL, 0, NULL, params, pledgedSrcSize);
}

size_t ZSTD_initCStream(ZSTD_CStream* zcs, int compressionLevel)
{
    ZSTD_parameters const params = ZSTD_getParams(compressionLevel, 0, 0);
    return ZSTD_initCStream_internal(zcs, NULL, 0, NULL, params, 0);
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

/*! ZSTDMT_initCStream_internal() :
 *  Private use only. Init streaming operation.
 *  expects params to be valid.
 *  must receive dict, or cdict, or none, but not both.
 *  @return : 0, or an error code */
size_t ZSTDMT_initCStream_internal(ZSTDMT_CCtx* zcs,
                    const void* dict, size_t dictSize, const ZSTD_CDict* cdict,
                    ZSTD_parameters params, unsigned long long pledgedSrcSize);


size_t ZSTD_compress_generic (ZSTD_CCtx* cctx,
                              ZSTD_outBuffer* output,
                              ZSTD_inBuffer* input,
                              ZSTD_EndDirective endOp)
{
    /* check conditions */
    if (output->pos > output->size) return ERROR(GENERIC);
    if (input->pos  > input->size)  return ERROR(GENERIC);
    assert(cctx!=NULL);

    /* transparent initialization stage */
    if (cctx->streamStage == zcss_init) {
        const void* const prefix = cctx->prefix;
        size_t const prefixSize = cctx->prefixSize;
        ZSTD_parameters params = cctx->requestedParams;
        if (cctx->compressionLevel != ZSTD_CLEVEL_CUSTOM)
            params.cParams = ZSTD_getCParams(cctx->compressionLevel,
                                cctx->pledgedSrcSizePlusOne-1, 0 /*dictSize*/);
        cctx->prefix = NULL; cctx->prefixSize = 0;   /* single usage */
        assert(prefix==NULL || cctx->cdict==NULL);   /* only one can be set */

#ifdef ZSTD_MULTITHREAD
        if (cctx->nbThreads > 1) {
            DEBUGLOG(4, "call ZSTDMT_initCStream_internal as nbThreads=%u", cctx->nbThreads);
            CHECK_F( ZSTDMT_initCStream_internal(cctx->mtctx, prefix, prefixSize, cctx->cdict, params, cctx->pledgedSrcSizePlusOne-1) );
            cctx->streamStage = zcss_load;
        } else
#endif
        {
            CHECK_F( ZSTD_resetCStream_internal(cctx, prefix, prefixSize, cctx->dictMode, cctx->cdict, params, cctx->pledgedSrcSizePlusOne-1) );
    }   }

    /* compression stage */
#ifdef ZSTD_MULTITHREAD
    if (cctx->nbThreads > 1) {
        size_t const flushMin = ZSTDMT_compressStream_generic(cctx->mtctx, output, input, endOp);
        DEBUGLOG(5, "ZSTDMT_compressStream_generic : %u", (U32)flushMin);
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
