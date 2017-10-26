/*
 * Copyright (c) 2016-present, Yann Collet, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under both the BSD-style license (found in the
 * LICENSE file in the root directory of this source tree) and the GPLv2 (found
 * in the COPYING file in the root directory of this source tree).
 * You may select, at your option, one of the above-listed licenses.
 */

#ifndef ZSTD_CCOMMON_H_MODULE
#define ZSTD_CCOMMON_H_MODULE


/*-*************************************
*  Dependencies
***************************************/
#include "compiler.h"
#include "mem.h"
#include "error_private.h"
#define ZSTD_STATIC_LINKING_ONLY
#include "zstd.h"
#define FSE_STATIC_LINKING_ONLY
#include "fse.h"
#define HUF_STATIC_LINKING_ONLY
#include "huf.h"
#ifndef XXH_STATIC_LINKING_ONLY
#  define XXH_STATIC_LINKING_ONLY  /* XXH64_state_t */
#endif
#include "xxhash.h"                /* XXH_reset, update, digest */


#if defined (__cplusplus)
extern "C" {
#endif


/*-*************************************
*  Debug
***************************************/
#if defined(ZSTD_DEBUG) && (ZSTD_DEBUG>=1)
#  include <assert.h>
#else
#  ifndef assert
#    define assert(condition) ((void)0)
#  endif
#endif

#define ZSTD_STATIC_ASSERT(c) { enum { ZSTD_static_assert = 1/(int)(!!(c)) }; }

#if defined(ZSTD_DEBUG) && (ZSTD_DEBUG>=2)
#  include <stdio.h>
/* recommended values for ZSTD_DEBUG display levels :
 * 1 : no display, enables assert() only
 * 2 : reserved for currently active debugging path
 * 3 : events once per object lifetime (CCtx, CDict)
 * 4 : events once per frame
 * 5 : events once per block
 * 6 : events once per sequence (*very* verbose) */
#  define DEBUGLOG(l, ...) {                         \
                if (l<=ZSTD_DEBUG) {                 \
                    fprintf(stderr, __FILE__ ": ");  \
                    fprintf(stderr, __VA_ARGS__);    \
                    fprintf(stderr, " \n");          \
            }   }
#else
#  define DEBUGLOG(l, ...)      {}    /* disabled */
#endif


/*-*************************************
*  shared macros
***************************************/
#undef MIN
#undef MAX
#define MIN(a,b) ((a)<(b) ? (a) : (b))
#define MAX(a,b) ((a)>(b) ? (a) : (b))
#define CHECK_F(f) { size_t const errcod = f; if (ERR_isError(errcod)) return errcod; }  /* check and Forward error code */
#define CHECK_E(f, e) { size_t const errcod = f; if (ERR_isError(errcod)) return ERROR(e); }  /* check and send Error code */


/*-*************************************
*  Common constants
***************************************/
#define ZSTD_OPT_NUM    (1<<12)

#define ZSTD_REP_NUM      3                 /* number of repcodes */
#define ZSTD_REP_CHECK    (ZSTD_REP_NUM)    /* number of repcodes to check by the optimal parser */
#define ZSTD_REP_MOVE     (ZSTD_REP_NUM-1)
#define ZSTD_REP_MOVE_OPT (ZSTD_REP_NUM)
static const U32 repStartValue[ZSTD_REP_NUM] = { 1, 4, 8 };

#define KB *(1 <<10)
#define MB *(1 <<20)
#define GB *(1U<<30)

#define BIT7 128
#define BIT6  64
#define BIT5  32
#define BIT4  16
#define BIT1   2
#define BIT0   1

#define ZSTD_WINDOWLOG_ABSOLUTEMIN 10
#define ZSTD_WINDOWLOG_DEFAULTMAX 27 /* Default maximum allowed window log */
static const size_t ZSTD_fcs_fieldSize[4] = { 0, 2, 4, 8 };
static const size_t ZSTD_did_fieldSize[4] = { 0, 1, 2, 4 };

#define ZSTD_FRAMEIDSIZE 4
static const size_t ZSTD_frameIdSize = ZSTD_FRAMEIDSIZE;  /* magic number size */

#define ZSTD_BLOCKHEADERSIZE 3   /* C standard doesn't allow `static const` variable to be init using another `static const` variable */
static const size_t ZSTD_blockHeaderSize = ZSTD_BLOCKHEADERSIZE;
typedef enum { bt_raw, bt_rle, bt_compressed, bt_reserved } blockType_e;

#define MIN_SEQUENCES_SIZE 1 /* nbSeq==0 */
#define MIN_CBLOCK_SIZE (1 /*litCSize*/ + 1 /* RLE or RAW */ + MIN_SEQUENCES_SIZE /* nbSeq==0 */)   /* for a non-null block */

#define HufLog 12
typedef enum { set_basic, set_rle, set_compressed, set_repeat } symbolEncodingType_e;

#define LONGNBSEQ 0x7F00

#define MINMATCH 3

#define Litbits  8
#define MaxLit ((1<<Litbits) - 1)
#define MaxML  52
#define MaxLL  35
#define DefaultMaxOff 28
#define MaxOff 31
#define MaxSeq MAX(MaxLL, MaxML)   /* Assumption : MaxOff < MaxLL,MaxML */
#define MLFSELog    9
#define LLFSELog    9
#define OffFSELog   8

static const U32 LL_bits[MaxLL+1] = { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                      1, 1, 1, 1, 2, 2, 3, 3, 4, 6, 7, 8, 9,10,11,12,
                                     13,14,15,16 };
static const S16 LL_defaultNorm[MaxLL+1] = { 4, 3, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1,
                                             2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 2, 1, 1, 1, 1, 1,
                                            -1,-1,-1,-1 };
#define LL_DEFAULTNORMLOG 6  /* for static allocation */
static const U32 LL_defaultNormLog = LL_DEFAULTNORMLOG;

static const U32 ML_bits[MaxML+1] = { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                      1, 1, 1, 1, 2, 2, 3, 3, 4, 4, 5, 7, 8, 9,10,11,
                                     12,13,14,15,16 };
static const S16 ML_defaultNorm[MaxML+1] = { 1, 4, 3, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1,
                                             1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                             1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,-1,-1,
                                            -1,-1,-1,-1,-1 };
#define ML_DEFAULTNORMLOG 6  /* for static allocation */
static const U32 ML_defaultNormLog = ML_DEFAULTNORMLOG;

static const S16 OF_defaultNorm[DefaultMaxOff+1] = { 1, 1, 1, 1, 1, 1, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1,
                                                     1, 1, 1, 1, 1, 1, 1, 1,-1,-1,-1,-1,-1 };
#define OF_DEFAULTNORMLOG 5  /* for static allocation */
static const U32 OF_defaultNormLog = OF_DEFAULTNORMLOG;


/*-*******************************************
*  Shared functions to include for inlining
*********************************************/
static void ZSTD_copy8(void* dst, const void* src) { memcpy(dst, src, 8); }
#define COPY8(d,s) { ZSTD_copy8(d,s); d+=8; s+=8; }

/*! ZSTD_wildcopy() :
*   custom version of memcpy(), can copy up to 7 bytes too many (8 bytes if length==0) */
#define WILDCOPY_OVERLENGTH 8
MEM_STATIC void ZSTD_wildcopy(void* dst, const void* src, ptrdiff_t length)
{
    const BYTE* ip = (const BYTE*)src;
    BYTE* op = (BYTE*)dst;
    BYTE* const oend = op + length;
    do
        COPY8(op, ip)
    while (op < oend);
}

MEM_STATIC void ZSTD_wildcopy_e(void* dst, const void* src, void* dstEnd)   /* should be faster for decoding, but strangely, not verified on all platform */
{
    const BYTE* ip = (const BYTE*)src;
    BYTE* op = (BYTE*)dst;
    BYTE* const oend = (BYTE*)dstEnd;
    do
        COPY8(op, ip)
    while (op < oend);
}


/*-*******************************************
*  Private interfaces
*********************************************/
typedef struct ZSTD_stats_s ZSTD_stats_t;

typedef struct seqDef_s {
    U32 offset;
    U16 litLength;
    U16 matchLength;
} seqDef;


typedef struct {
    seqDef* sequencesStart;
    seqDef* sequences;
    BYTE* litStart;
    BYTE* lit;
    BYTE* llCode;
    BYTE* mlCode;
    BYTE* ofCode;
    U32   longLengthID;   /* 0 == no longLength; 1 == Lit.longLength; 2 == Match.longLength; */
    U32   longLengthPos;
    U32   rep[ZSTD_REP_NUM];
    U32   repToConfirm[ZSTD_REP_NUM];
} seqStore_t;

typedef struct {
    U32 off;
    U32 len;
} ZSTD_match_t;

typedef struct {
    U32 price;
    U32 off;
    U32 mlen;
    U32 litlen;
    U32 rep[ZSTD_REP_NUM];
} ZSTD_optimal_t;

typedef struct {
    U32* litFreq;
    U32* litLengthFreq;
    U32* matchLengthFreq;
    U32* offCodeFreq;
    ZSTD_match_t* matchTable;
    ZSTD_optimal_t* priceTable;

    U32  matchLengthSum;
    U32  matchSum;
    U32  litLengthSum;
    U32  litSum;
    U32  offCodeSum;
    U32  log2matchLengthSum;
    U32  log2matchSum;
    U32  log2litLengthSum;
    U32  log2litSum;
    U32  log2offCodeSum;
    U32  factor;
    U32  staticPrices;
    U32  cachedPrice;
    U32  cachedLitLength;
    const BYTE* cachedLiterals;
} optState_t;

typedef struct {
    U32 offset;
    U32 checksum;
} ldmEntry_t;

typedef struct {
    ldmEntry_t* hashTable;
    BYTE* bucketOffsets;    /* Next position in bucket to insert entry */
    U64 hashPower;          /* Used to compute the rolling hash.
                             * Depends on ldmParams.minMatchLength */
} ldmState_t;

typedef struct {
    U32 enableLdm;          /* 1 if enable long distance matching */
    U32 hashLog;            /* Log size of hashTable */
    U32 bucketSizeLog;      /* Log bucket size for collision resolution, at most 8 */
    U32 minMatchLength;     /* Minimum match length */
    U32 hashEveryLog;       /* Log number of entries to skip */
} ldmParams_t;

typedef struct {
    U32 hufCTable[HUF_CTABLE_SIZE_U32(255)];
    FSE_CTable offcodeCTable[FSE_CTABLE_SIZE_U32(OffFSELog, MaxOff)];
    FSE_CTable matchlengthCTable[FSE_CTABLE_SIZE_U32(MLFSELog, MaxML)];
    FSE_CTable litlengthCTable[FSE_CTABLE_SIZE_U32(LLFSELog, MaxLL)];
    U32 workspace[HUF_WORKSPACE_SIZE_U32];
    HUF_repeat hufCTable_repeatMode;
    FSE_repeat offcode_repeatMode;
    FSE_repeat matchlength_repeatMode;
    FSE_repeat litlength_repeatMode;
} ZSTD_entropyCTables_t;

struct ZSTD_CCtx_params_s {
    ZSTD_format_e format;
    ZSTD_compressionParameters cParams;
    ZSTD_frameParameters fParams;

    int compressionLevel;
    U32 forceWindow;           /* force back-references to respect limit of
                                * 1<<wLog, even for dictionary */

    /* Multithreading: used to pass parameters to mtctx */
    U32 nbThreads;
    unsigned jobSize;
    unsigned overlapSizeLog;

    /* Long distance matching parameters */
    ldmParams_t ldmParams;

    /* For use with createCCtxParams() and freeCCtxParams() only */
    ZSTD_customMem customMem;

};  /* typedef'd to ZSTD_CCtx_params within "zstd.h" */

const seqStore_t* ZSTD_getSeqStore(const ZSTD_CCtx* ctx);
void ZSTD_seqToCodes(const seqStore_t* seqStorePtr);

/* custom memory allocation functions */
void* ZSTD_malloc(size_t size, ZSTD_customMem customMem);
void* ZSTD_calloc(size_t size, ZSTD_customMem customMem);
void ZSTD_free(void* ptr, ZSTD_customMem customMem);


/*======  common function  ======*/

MEM_STATIC U32 ZSTD_highbit32(U32 val)
{
    assert(val != 0);
    {
#   if defined(_MSC_VER)   /* Visual */
        unsigned long r=0;
        _BitScanReverse(&r, val);
        return (unsigned)r;
#   elif defined(__GNUC__) && (__GNUC__ >= 3)   /* GCC Intrinsic */
        return 31 - __builtin_clz(val);
#   else   /* Software version */
        static const int DeBruijnClz[32] = { 0, 9, 1, 10, 13, 21, 2, 29, 11, 14, 16, 18, 22, 25, 3, 30, 8, 12, 20, 28, 15, 17, 24, 7, 19, 27, 23, 6, 26, 5, 4, 31 };
        U32 v = val;
        int r;
        v |= v >> 1;
        v |= v >> 2;
        v |= v >> 4;
        v |= v >> 8;
        v |= v >> 16;
        r = DeBruijnClz[(U32)(v * 0x07C4ACDDU) >> 27];
        return r;
#   endif
    }
}


/* hidden functions */

/* ZSTD_invalidateRepCodes() :
 * ensures next compression will not use repcodes from previous block.
 * Note : only works with regular variant;
 *        do not use with extDict variant ! */
void ZSTD_invalidateRepCodes(ZSTD_CCtx* cctx);


/*! ZSTD_initCStream_internal() :
 *  Private use only. Init streaming operation.
 *  expects params to be valid.
 *  must receive dict, or cdict, or none, but not both.
 *  @return : 0, or an error code */
size_t ZSTD_initCStream_internal(ZSTD_CStream* zcs,
                     const void* dict, size_t dictSize,
                     const ZSTD_CDict* cdict,
                     ZSTD_CCtx_params  params, unsigned long long pledgedSrcSize);

/*! ZSTD_compressStream_generic() :
 *  Private use only. To be called from zstdmt_compress.c in single-thread mode. */
size_t ZSTD_compressStream_generic(ZSTD_CStream* zcs,
                                   ZSTD_outBuffer* output,
                                   ZSTD_inBuffer* input,
                                   ZSTD_EndDirective const flushMode);

/*! ZSTD_getCParamsFromCDict() :
 *  as the name implies */
ZSTD_compressionParameters ZSTD_getCParamsFromCDict(const ZSTD_CDict* cdict);

/* ZSTD_compressBegin_advanced_internal() :
 * Private use only. To be called from zstdmt_compress.c. */
size_t ZSTD_compressBegin_advanced_internal(ZSTD_CCtx* cctx,
                                    const void* dict, size_t dictSize,
                                    ZSTD_dictMode_e dictMode,
                                    ZSTD_CCtx_params params,
                                    unsigned long long pledgedSrcSize);

/* ZSTD_compress_advanced_internal() :
 * Private use only. To be called from zstdmt_compress.c. */
size_t ZSTD_compress_advanced_internal(ZSTD_CCtx* cctx,
                                       void* dst, size_t dstCapacity,
                                 const void* src, size_t srcSize,
                                 const void* dict,size_t dictSize,
                                 ZSTD_CCtx_params params);

typedef struct {
    blockType_e blockType;
    U32 lastBlock;
    U32 origSize;
} blockProperties_t;

/*! ZSTD_getcBlockSize() :
*   Provides the size of compressed block from block header `src` */
size_t ZSTD_getcBlockSize(const void* src, size_t srcSize,
                          blockProperties_t* bpPtr);

#if defined (__cplusplus)
}
#endif

#endif   /* ZSTD_CCOMMON_H_MODULE */
