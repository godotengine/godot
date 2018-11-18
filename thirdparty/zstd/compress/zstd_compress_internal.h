/*
 * Copyright (c) 2016-present, Yann Collet, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under both the BSD-style license (found in the
 * LICENSE file in the root directory of this source tree) and the GPLv2 (found
 * in the COPYING file in the root directory of this source tree).
 * You may select, at your option, one of the above-listed licenses.
 */

/* This header contains definitions
 * that shall **only** be used by modules within lib/compress.
 */

#ifndef ZSTD_COMPRESS_H
#define ZSTD_COMPRESS_H

/*-*************************************
*  Dependencies
***************************************/
#include "zstd_internal.h"
#ifdef ZSTD_MULTITHREAD
#  include "zstdmt_compress.h"
#endif

#if defined (__cplusplus)
extern "C" {
#endif

/*-*************************************
*  Constants
***************************************/
#define kSearchStrength      8
#define HASH_READ_SIZE       8
#define ZSTD_DUBT_UNSORTED_MARK 1   /* For btlazy2 strategy, index 1 now means "unsorted".
                                       It could be confused for a real successor at index "1", if sorted as larger than its predecessor.
                                       It's not a big deal though : candidate will just be sorted again.
                                       Additionnally, candidate position 1 will be lost.
                                       But candidate 1 cannot hide a large tree of candidates, so it's a minimal loss.
                                       The benefit is that ZSTD_DUBT_UNSORTED_MARK cannot be misdhandled after table re-use with a different strategy */


/*-*************************************
*  Context memory management
***************************************/
typedef enum { ZSTDcs_created=0, ZSTDcs_init, ZSTDcs_ongoing, ZSTDcs_ending } ZSTD_compressionStage_e;
typedef enum { zcss_init=0, zcss_load, zcss_flush } ZSTD_cStreamStage;

typedef struct ZSTD_prefixDict_s {
    const void* dict;
    size_t dictSize;
    ZSTD_dictContentType_e dictContentType;
} ZSTD_prefixDict;

typedef struct {
    U32 hufCTable[HUF_CTABLE_SIZE_U32(255)];
    FSE_CTable offcodeCTable[FSE_CTABLE_SIZE_U32(OffFSELog, MaxOff)];
    FSE_CTable matchlengthCTable[FSE_CTABLE_SIZE_U32(MLFSELog, MaxML)];
    FSE_CTable litlengthCTable[FSE_CTABLE_SIZE_U32(LLFSELog, MaxLL)];
    HUF_repeat hufCTable_repeatMode;
    FSE_repeat offcode_repeatMode;
    FSE_repeat matchlength_repeatMode;
    FSE_repeat litlength_repeatMode;
} ZSTD_entropyCTables_t;

typedef struct {
    U32 off;
    U32 len;
} ZSTD_match_t;

typedef struct {
    int price;
    U32 off;
    U32 mlen;
    U32 litlen;
    U32 rep[ZSTD_REP_NUM];
} ZSTD_optimal_t;

typedef struct {
    /* All tables are allocated inside cctx->workspace by ZSTD_resetCCtx_internal() */
    U32* litFreq;               /* table of literals statistics, of size 256 */
    U32* litLengthFreq;         /* table of litLength statistics, of size (MaxLL+1) */
    U32* matchLengthFreq;       /* table of matchLength statistics, of size (MaxML+1) */
    U32* offCodeFreq;           /* table of offCode statistics, of size (MaxOff+1) */
    ZSTD_match_t* matchTable;   /* list of found matches, of size ZSTD_OPT_NUM+1 */
    ZSTD_optimal_t* priceTable; /* All positions tracked by optimal parser, of size ZSTD_OPT_NUM+1 */

    U32  litSum;                 /* nb of literals */
    U32  litLengthSum;           /* nb of litLength codes */
    U32  matchLengthSum;         /* nb of matchLength codes */
    U32  offCodeSum;             /* nb of offset codes */
    /* begin updated by ZSTD_setLog2Prices */
    U32  log2litSum;             /* pow2 to compare log2(litfreq) to */
    U32  log2litLengthSum;       /* pow2 to compare log2(llfreq) to */
    U32  log2matchLengthSum;     /* pow2 to compare log2(mlfreq) to */
    U32  log2offCodeSum;         /* pow2 to compare log2(offreq) to */
    /* end : updated by ZSTD_setLog2Prices */
    U32  staticPrices;           /* prices follow a pre-defined cost structure, statistics are irrelevant */
} optState_t;

typedef struct {
  ZSTD_entropyCTables_t entropy;
  U32 rep[ZSTD_REP_NUM];
} ZSTD_compressedBlockState_t;

typedef struct {
    BYTE const* nextSrc;    /* next block here to continue on current prefix */
    BYTE const* base;       /* All regular indexes relative to this position */
    BYTE const* dictBase;   /* extDict indexes relative to this position */
    U32 dictLimit;          /* below that point, need extDict */
    U32 lowLimit;           /* below that point, no more data */
} ZSTD_window_t;

typedef struct {
    ZSTD_window_t window;      /* State for window round buffer management */
    U32 loadedDictEnd;         /* index of end of dictionary */
    U32 nextToUpdate;          /* index from which to continue table update */
    U32 nextToUpdate3;         /* index from which to continue table update */
    U32 hashLog3;              /* dispatch table : larger == faster, more memory */
    U32* hashTable;
    U32* hashTable3;
    U32* chainTable;
    optState_t opt;         /* optimal parser state */
} ZSTD_matchState_t;

typedef struct {
    ZSTD_compressedBlockState_t* prevCBlock;
    ZSTD_compressedBlockState_t* nextCBlock;
    ZSTD_matchState_t matchState;
} ZSTD_blockState_t;

typedef struct {
    U32 offset;
    U32 checksum;
} ldmEntry_t;

typedef struct {
    ZSTD_window_t window;   /* State for the window round buffer management */
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
    U32 windowLog;          /* Window log for the LDM */
} ldmParams_t;

typedef struct {
    U32 offset;
    U32 litLength;
    U32 matchLength;
} rawSeq;

typedef struct {
  rawSeq* seq;     /* The start of the sequences */
  size_t pos;      /* The position where reading stopped. <= size. */
  size_t size;     /* The number of sequences. <= capacity. */
  size_t capacity; /* The capacity of the `seq` pointer */
} rawSeqStore_t;

struct ZSTD_CCtx_params_s {
    ZSTD_format_e format;
    ZSTD_compressionParameters cParams;
    ZSTD_frameParameters fParams;

    int compressionLevel;
    int disableLiteralCompression;
    int forceWindow;           /* force back-references to respect limit of
                                * 1<<wLog, even for dictionary */

    /* Multithreading: used to pass parameters to mtctx */
    unsigned nbWorkers;
    unsigned jobSize;
    unsigned overlapSizeLog;

    /* Long distance matching parameters */
    ldmParams_t ldmParams;

    /* Internal use, for createCCtxParams() and freeCCtxParams() only */
    ZSTD_customMem customMem;
};  /* typedef'd to ZSTD_CCtx_params within "zstd.h" */

struct ZSTD_CCtx_s {
    ZSTD_compressionStage_e stage;
    int cParamsChanged;                  /* == 1 if cParams(except wlog) or compression level are changed in requestedParams. Triggers transmission of new params to ZSTDMT (if available) then reset to 0. */
    int bmi2;                            /* == 1 if the CPU supports BMI2 and 0 otherwise. CPU support is determined dynamically once per context lifetime. */
    ZSTD_CCtx_params requestedParams;
    ZSTD_CCtx_params appliedParams;
    U32   dictID;
    void* workSpace;
    size_t workSpaceSize;
    size_t blockSize;
    unsigned long long pledgedSrcSizePlusOne;  /* this way, 0 (default) == unknown */
    unsigned long long consumedSrcSize;
    unsigned long long producedCSize;
    XXH64_state_t xxhState;
    ZSTD_customMem customMem;
    size_t staticSize;

    seqStore_t seqStore;      /* sequences storage ptrs */
    ldmState_t ldmState;      /* long distance matching state */
    rawSeq* ldmSequences;     /* Storage for the ldm output sequences */
    size_t maxNbLdmSequences;
    rawSeqStore_t externSeqStore; /* Mutable reference to external sequences */
    ZSTD_blockState_t blockState;
    U32* entropyWorkspace;  /* entropy workspace of HUF_WORKSPACE_SIZE bytes */

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
    ZSTD_CDict* cdictLocal;
    const ZSTD_CDict* cdict;
    ZSTD_prefixDict prefixDict;   /* single-usage dictionary */

    /* Multi-threading */
#ifdef ZSTD_MULTITHREAD
    ZSTDMT_CCtx* mtctx;
#endif
};


typedef size_t (*ZSTD_blockCompressor) (
        ZSTD_matchState_t* bs, seqStore_t* seqStore, U32 rep[ZSTD_REP_NUM],
        ZSTD_compressionParameters const* cParams, void const* src, size_t srcSize);
ZSTD_blockCompressor ZSTD_selectBlockCompressor(ZSTD_strategy strat, int extDict);


MEM_STATIC U32 ZSTD_LLcode(U32 litLength)
{
    static const BYTE LL_Code[64] = {  0,  1,  2,  3,  4,  5,  6,  7,
                                       8,  9, 10, 11, 12, 13, 14, 15,
                                      16, 16, 17, 17, 18, 18, 19, 19,
                                      20, 20, 20, 20, 21, 21, 21, 21,
                                      22, 22, 22, 22, 22, 22, 22, 22,
                                      23, 23, 23, 23, 23, 23, 23, 23,
                                      24, 24, 24, 24, 24, 24, 24, 24,
                                      24, 24, 24, 24, 24, 24, 24, 24 };
    static const U32 LL_deltaCode = 19;
    return (litLength > 63) ? ZSTD_highbit32(litLength) + LL_deltaCode : LL_Code[litLength];
}

/* ZSTD_MLcode() :
 * note : mlBase = matchLength - MINMATCH;
 *        because it's the format it's stored in seqStore->sequences */
MEM_STATIC U32 ZSTD_MLcode(U32 mlBase)
{
    static const BYTE ML_Code[128] = { 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15,
                                      16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31,
                                      32, 32, 33, 33, 34, 34, 35, 35, 36, 36, 36, 36, 37, 37, 37, 37,
                                      38, 38, 38, 38, 38, 38, 38, 38, 39, 39, 39, 39, 39, 39, 39, 39,
                                      40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40,
                                      41, 41, 41, 41, 41, 41, 41, 41, 41, 41, 41, 41, 41, 41, 41, 41,
                                      42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42,
                                      42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42 };
    static const U32 ML_deltaCode = 36;
    return (mlBase > 127) ? ZSTD_highbit32(mlBase) + ML_deltaCode : ML_Code[mlBase];
}

/*! ZSTD_storeSeq() :
 *  Store a sequence (literal length, literals, offset code and match length code) into seqStore_t.
 *  `offsetCode` : distance to match + 3 (values 1-3 are repCodes).
 *  `mlBase` : matchLength - MINMATCH
*/
MEM_STATIC void ZSTD_storeSeq(seqStore_t* seqStorePtr, size_t litLength, const void* literals, U32 offsetCode, size_t mlBase)
{
#if defined(ZSTD_DEBUG) && (ZSTD_DEBUG >= 6)
    static const BYTE* g_start = NULL;
    if (g_start==NULL) g_start = (const BYTE*)literals;  /* note : index only works for compression within a single segment */
    {   U32 const pos = (U32)((const BYTE*)literals - g_start);
        DEBUGLOG(6, "Cpos%7u :%3u literals, match%3u bytes at dist.code%7u",
               pos, (U32)litLength, (U32)mlBase+MINMATCH, (U32)offsetCode);
    }
#endif
    /* copy Literals */
    assert(seqStorePtr->lit + litLength <= seqStorePtr->litStart + 128 KB);
    ZSTD_wildcopy(seqStorePtr->lit, literals, litLength);
    seqStorePtr->lit += litLength;

    /* literal Length */
    if (litLength>0xFFFF) {
        assert(seqStorePtr->longLengthID == 0); /* there can only be a single long length */
        seqStorePtr->longLengthID = 1;
        seqStorePtr->longLengthPos = (U32)(seqStorePtr->sequences - seqStorePtr->sequencesStart);
    }
    seqStorePtr->sequences[0].litLength = (U16)litLength;

    /* match offset */
    seqStorePtr->sequences[0].offset = offsetCode + 1;

    /* match Length */
    if (mlBase>0xFFFF) {
        assert(seqStorePtr->longLengthID == 0); /* there can only be a single long length */
        seqStorePtr->longLengthID = 2;
        seqStorePtr->longLengthPos = (U32)(seqStorePtr->sequences - seqStorePtr->sequencesStart);
    }
    seqStorePtr->sequences[0].matchLength = (U16)mlBase;

    seqStorePtr->sequences++;
}


/*-*************************************
*  Match length counter
***************************************/
static unsigned ZSTD_NbCommonBytes (size_t val)
{
    if (MEM_isLittleEndian()) {
        if (MEM_64bits()) {
#       if defined(_MSC_VER) && defined(_WIN64)
            unsigned long r = 0;
            _BitScanForward64( &r, (U64)val );
            return (unsigned)(r>>3);
#       elif defined(__GNUC__) && (__GNUC__ >= 4)
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
#       elif defined(__GNUC__) && (__GNUC__ >= 4)
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


MEM_STATIC size_t ZSTD_count(const BYTE* pIn, const BYTE* pMatch, const BYTE* const pInLimit)
{
    const BYTE* const pStart = pIn;
    const BYTE* const pInLoopLimit = pInLimit - (sizeof(size_t)-1);

    if (pIn < pInLoopLimit) {
        { size_t const diff = MEM_readST(pMatch) ^ MEM_readST(pIn);
          if (diff) return ZSTD_NbCommonBytes(diff); }
        pIn+=sizeof(size_t); pMatch+=sizeof(size_t);
        while (pIn < pInLoopLimit) {
            size_t const diff = MEM_readST(pMatch) ^ MEM_readST(pIn);
            if (!diff) { pIn+=sizeof(size_t); pMatch+=sizeof(size_t); continue; }
            pIn += ZSTD_NbCommonBytes(diff);
            return (size_t)(pIn - pStart);
    }   }
    if (MEM_64bits() && (pIn<(pInLimit-3)) && (MEM_read32(pMatch) == MEM_read32(pIn))) { pIn+=4; pMatch+=4; }
    if ((pIn<(pInLimit-1)) && (MEM_read16(pMatch) == MEM_read16(pIn))) { pIn+=2; pMatch+=2; }
    if ((pIn<pInLimit) && (*pMatch == *pIn)) pIn++;
    return (size_t)(pIn - pStart);
}

/** ZSTD_count_2segments() :
 *  can count match length with `ip` & `match` in 2 different segments.
 *  convention : on reaching mEnd, match count continue starting from iStart
 */
MEM_STATIC size_t
ZSTD_count_2segments(const BYTE* ip, const BYTE* match,
                     const BYTE* iEnd, const BYTE* mEnd, const BYTE* iStart)
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

MEM_STATIC size_t ZSTD_hashPtr(const void* p, U32 hBits, U32 mls)
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
*  Round buffer management
***************************************/
/* Max current allowed */
#define ZSTD_CURRENT_MAX ((3U << 29) + (1U << ZSTD_WINDOWLOG_MAX))
/* Maximum chunk size before overflow correction needs to be called again */
#define ZSTD_CHUNKSIZE_MAX                                                     \
    ( ((U32)-1)                  /* Maximum ending current index */            \
    - ZSTD_CURRENT_MAX)          /* Maximum beginning lowLimit */

/**
 * ZSTD_window_clear():
 * Clears the window containing the history by simply setting it to empty.
 */
MEM_STATIC void ZSTD_window_clear(ZSTD_window_t* window)
{
    size_t const endT = (size_t)(window->nextSrc - window->base);
    U32 const end = (U32)endT;

    window->lowLimit = end;
    window->dictLimit = end;
}

/**
 * ZSTD_window_hasExtDict():
 * Returns non-zero if the window has a non-empty extDict.
 */
MEM_STATIC U32 ZSTD_window_hasExtDict(ZSTD_window_t const window)
{
    return window.lowLimit < window.dictLimit;
}

/**
 * ZSTD_window_needOverflowCorrection():
 * Returns non-zero if the indices are getting too large and need overflow
 * protection.
 */
MEM_STATIC U32 ZSTD_window_needOverflowCorrection(ZSTD_window_t const window,
                                                  void const* srcEnd)
{
    U32 const current = (U32)((BYTE const*)srcEnd - window.base);
    return current > ZSTD_CURRENT_MAX;
}

/**
 * ZSTD_window_correctOverflow():
 * Reduces the indices to protect from index overflow.
 * Returns the correction made to the indices, which must be applied to every
 * stored index.
 *
 * The least significant cycleLog bits of the indices must remain the same,
 * which may be 0. Every index up to maxDist in the past must be valid.
 * NOTE: (maxDist & cycleMask) must be zero.
 */
MEM_STATIC U32 ZSTD_window_correctOverflow(ZSTD_window_t* window, U32 cycleLog,
                                           U32 maxDist, void const* src)
{
    /* preemptive overflow correction:
     * 1. correction is large enough:
     *    lowLimit > (3<<29) ==> current > 3<<29 + 1<<windowLog
     *    1<<windowLog <= newCurrent < 1<<chainLog + 1<<windowLog
     *
     *    current - newCurrent
     *    > (3<<29 + 1<<windowLog) - (1<<windowLog + 1<<chainLog)
     *    > (3<<29) - (1<<chainLog)
     *    > (3<<29) - (1<<30)             (NOTE: chainLog <= 30)
     *    > 1<<29
     *
     * 2. (ip+ZSTD_CHUNKSIZE_MAX - cctx->base) doesn't overflow:
     *    After correction, current is less than (1<<chainLog + 1<<windowLog).
     *    In 64-bit mode we are safe, because we have 64-bit ptrdiff_t.
     *    In 32-bit mode we are safe, because (chainLog <= 29), so
     *    ip+ZSTD_CHUNKSIZE_MAX - cctx->base < 1<<32.
     * 3. (cctx->lowLimit + 1<<windowLog) < 1<<32:
     *    windowLog <= 31 ==> 3<<29 + 1<<windowLog < 7<<29 < 1<<32.
     */
    U32 const cycleMask = (1U << cycleLog) - 1;
    U32 const current = (U32)((BYTE const*)src - window->base);
    U32 const newCurrent = (current & cycleMask) + maxDist;
    U32 const correction = current - newCurrent;
    assert((maxDist & cycleMask) == 0);
    assert(current > newCurrent);
    /* Loose bound, should be around 1<<29 (see above) */
    assert(correction > 1<<28);

    window->base += correction;
    window->dictBase += correction;
    window->lowLimit -= correction;
    window->dictLimit -= correction;

    DEBUGLOG(4, "Correction of 0x%x bytes to lowLimit=0x%x", correction,
             window->lowLimit);
    return correction;
}

/**
 * ZSTD_window_enforceMaxDist():
 * Updates lowLimit so that:
 *    (srcEnd - base) - lowLimit == maxDist + loadedDictEnd
 * This allows a simple check that index >= lowLimit to see if index is valid.
 * This must be called before a block compression call, with srcEnd as the block
 * source end.
 * If loadedDictEndPtr is not NULL, we set it to zero once we update lowLimit.
 * This is because dictionaries are allowed to be referenced as long as the last
 * byte of the dictionary is in the window, but once they are out of range,
 * they cannot be referenced. If loadedDictEndPtr is NULL, we use
 * loadedDictEnd == 0.
 */
MEM_STATIC void ZSTD_window_enforceMaxDist(ZSTD_window_t* window,
                                           void const* srcEnd, U32 maxDist,
                                           U32* loadedDictEndPtr)
{
    U32 const current = (U32)((BYTE const*)srcEnd - window->base);
    U32 loadedDictEnd = loadedDictEndPtr != NULL ? *loadedDictEndPtr : 0;
    if (current > maxDist + loadedDictEnd) {
        U32 const newLowLimit = current - maxDist;
        if (window->lowLimit < newLowLimit) window->lowLimit = newLowLimit;
        if (window->dictLimit < window->lowLimit) {
            DEBUGLOG(5, "Update dictLimit from %u to %u", window->dictLimit,
                     window->lowLimit);
            window->dictLimit = window->lowLimit;
        }
        if (loadedDictEndPtr)
            *loadedDictEndPtr = 0;
    }
}

/**
 * ZSTD_window_update():
 * Updates the window by appending [src, src + srcSize) to the window.
 * If it is not contiguous, the current prefix becomes the extDict, and we
 * forget about the extDict. Handles overlap of the prefix and extDict.
 * Returns non-zero if the segment is contiguous.
 */
MEM_STATIC U32 ZSTD_window_update(ZSTD_window_t* window,
                                  void const* src, size_t srcSize)
{
    BYTE const* const ip = (BYTE const*)src;
    U32 contiguous = 1;
    /* Check if blocks follow each other */
    if (src != window->nextSrc) {
        /* not contiguous */
        size_t const distanceFromBase = (size_t)(window->nextSrc - window->base);
        DEBUGLOG(5, "Non contiguous blocks, new segment starts at %u",
                 window->dictLimit);
        window->lowLimit = window->dictLimit;
        assert(distanceFromBase == (size_t)(U32)distanceFromBase);  /* should never overflow */
        window->dictLimit = (U32)distanceFromBase;
        window->dictBase = window->base;
        window->base = ip - distanceFromBase;
        // ms->nextToUpdate = window->dictLimit;
        if (window->dictLimit - window->lowLimit < HASH_READ_SIZE) window->lowLimit = window->dictLimit;   /* too small extDict */
        contiguous = 0;
    }
    window->nextSrc = ip + srcSize;
    /* if input and dictionary overlap : reduce dictionary (area presumed modified by input) */
    if ( (ip+srcSize > window->dictBase + window->lowLimit)
       & (ip < window->dictBase + window->dictLimit)) {
        ptrdiff_t const highInputIdx = (ip + srcSize) - window->dictBase;
        U32 const lowLimitMax = (highInputIdx > (ptrdiff_t)window->dictLimit) ? window->dictLimit : (U32)highInputIdx;
        window->lowLimit = lowLimitMax;
    }
    return contiguous;
}

#if defined (__cplusplus)
}
#endif


/* ==============================================================
 * Private declarations
 * These prototypes shall only be called from within lib/compress
 * ============================================================== */

/* ZSTD_getCParamsFromCCtxParams() :
 * cParams are built depending on compressionLevel, src size hints, 
 * LDM and manually set compression parameters.
 */
ZSTD_compressionParameters ZSTD_getCParamsFromCCtxParams(
        const ZSTD_CCtx_params* CCtxParams, U64 srcSizeHint, size_t dictSize);

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
                                    ZSTD_dictContentType_e dictContentType,
                                    const ZSTD_CDict* cdict,
                                    ZSTD_CCtx_params params,
                                    unsigned long long pledgedSrcSize);

/* ZSTD_compress_advanced_internal() :
 * Private use only. To be called from zstdmt_compress.c. */
size_t ZSTD_compress_advanced_internal(ZSTD_CCtx* cctx,
                                       void* dst, size_t dstCapacity,
                                 const void* src, size_t srcSize,
                                 const void* dict,size_t dictSize,
                                 ZSTD_CCtx_params params);


/* ZSTD_writeLastEmptyBlock() :
 * output an empty Block with end-of-frame mark to complete a frame
 * @return : size of data written into `dst` (== ZSTD_blockHeaderSize (defined in zstd_internal.h))
 *           or an error code if `dstCapcity` is too small (<ZSTD_blockHeaderSize)
 */
size_t ZSTD_writeLastEmptyBlock(void* dst, size_t dstCapacity);


/* ZSTD_referenceExternalSequences() :
 * Must be called before starting a compression operation.
 * seqs must parse a prefix of the source.
 * This cannot be used when long range matching is enabled.
 * Zstd will use these sequences, and pass the literals to a secondary block
 * compressor.
 * @return : An error code on failure.
 * NOTE: seqs are not verified! Invalid sequences can cause out-of-bounds memory
 * access and data corruption.
 */
size_t ZSTD_referenceExternalSequences(ZSTD_CCtx* cctx, rawSeq* seq, size_t nbSeq);


#endif /* ZSTD_COMPRESS_H */
