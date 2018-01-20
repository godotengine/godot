/*
 * Copyright (c) 2016-present, Przemyslaw Skibinski, Yann Collet, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under both the BSD-style license (found in the
 * LICENSE file in the root directory of this source tree) and the GPLv2 (found
 * in the COPYING file in the root directory of this source tree).
 * You may select, at your option, one of the above-listed licenses.
 */

#include "zstd_compress_internal.h"
#include "zstd_opt.h"
#include "zstd_lazy.h"   /* ZSTD_updateTree, ZSTD_updateTree_extDict */


#define ZSTD_LITFREQ_ADD    2   /* scaling factor for litFreq, so that frequencies adapt faster to new stats. Also used for matchSum (?) */
#define ZSTD_FREQ_DIV       4   /* log factor when using previous stats to init next stats */
#define ZSTD_MAX_PRICE     (1<<30)


/*-*************************************
*  Price functions for optimal parser
***************************************/
static void ZSTD_setLog2Prices(optState_t* optPtr)
{
    optPtr->log2litSum = ZSTD_highbit32(optPtr->litSum+1);
    optPtr->log2litLengthSum = ZSTD_highbit32(optPtr->litLengthSum+1);
    optPtr->log2matchLengthSum = ZSTD_highbit32(optPtr->matchLengthSum+1);
    optPtr->log2offCodeSum = ZSTD_highbit32(optPtr->offCodeSum+1);
}


static void ZSTD_rescaleFreqs(optState_t* const optPtr,
                              const BYTE* const src, size_t const srcSize)
{
    optPtr->staticPrices = 0;

    if (optPtr->litLengthSum == 0) {  /* first init */
        unsigned u;
        if (srcSize <= 1024) optPtr->staticPrices = 1;

        assert(optPtr->litFreq!=NULL);
        for (u=0; u<=MaxLit; u++)
            optPtr->litFreq[u] = 0;
        for (u=0; u<srcSize; u++)
            optPtr->litFreq[src[u]]++;
        optPtr->litSum = 0;
        for (u=0; u<=MaxLit; u++) {
            optPtr->litFreq[u] = 1 + (optPtr->litFreq[u] >> ZSTD_FREQ_DIV);
            optPtr->litSum += optPtr->litFreq[u];
        }

        for (u=0; u<=MaxLL; u++)
            optPtr->litLengthFreq[u] = 1;
        optPtr->litLengthSum = MaxLL+1;
        for (u=0; u<=MaxML; u++)
            optPtr->matchLengthFreq[u] = 1;
        optPtr->matchLengthSum = MaxML+1;
        for (u=0; u<=MaxOff; u++)
            optPtr->offCodeFreq[u] = 1;
        optPtr->offCodeSum = (MaxOff+1);

    } else {
        unsigned u;

        optPtr->litSum = 0;
        for (u=0; u<=MaxLit; u++) {
            optPtr->litFreq[u] = 1 + (optPtr->litFreq[u] >> (ZSTD_FREQ_DIV+1));
            optPtr->litSum += optPtr->litFreq[u];
        }
        optPtr->litLengthSum = 0;
        for (u=0; u<=MaxLL; u++) {
            optPtr->litLengthFreq[u] = 1 + (optPtr->litLengthFreq[u]>>(ZSTD_FREQ_DIV+1));
            optPtr->litLengthSum += optPtr->litLengthFreq[u];
        }
        optPtr->matchLengthSum = 0;
        for (u=0; u<=MaxML; u++) {
            optPtr->matchLengthFreq[u] = 1 + (optPtr->matchLengthFreq[u]>>ZSTD_FREQ_DIV);
            optPtr->matchLengthSum += optPtr->matchLengthFreq[u];
        }
        optPtr->offCodeSum = 0;
        for (u=0; u<=MaxOff; u++) {
            optPtr->offCodeFreq[u] = 1 + (optPtr->offCodeFreq[u]>>ZSTD_FREQ_DIV);
            optPtr->offCodeSum += optPtr->offCodeFreq[u];
        }
    }

    ZSTD_setLog2Prices(optPtr);
}


/* ZSTD_rawLiteralsCost() :
 * cost of literals (only) in given segment (which length can be null)
 * does not include cost of literalLength symbol */
static U32 ZSTD_rawLiteralsCost(const BYTE* const literals, U32 const litLength,
                                const optState_t* const optPtr)
{
    if (optPtr->staticPrices) return (litLength*6);  /* 6 bit per literal - no statistic used */
    if (litLength == 0) return 0;

    /* literals */
    {   U32 u;
        U32 cost = litLength * optPtr->log2litSum;
        for (u=0; u < litLength; u++)
            cost -= ZSTD_highbit32(optPtr->litFreq[literals[u]]+1);
        return cost;
    }
}

/* ZSTD_litLengthPrice() :
 * cost of literalLength symbol */
static U32 ZSTD_litLengthPrice(U32 const litLength, const optState_t* const optPtr)
{
    if (optPtr->staticPrices) return ZSTD_highbit32((U32)litLength+1);

    /* literal Length */
    {   U32 const llCode = ZSTD_LLcode(litLength);
        U32 const price = LL_bits[llCode] + optPtr->log2litLengthSum - ZSTD_highbit32(optPtr->litLengthFreq[llCode]+1);
        return price;
    }
}

/* ZSTD_litLengthPrice() :
 * cost of the literal part of a sequence,
 * including literals themselves, and literalLength symbol */
static U32 ZSTD_fullLiteralsCost(const BYTE* const literals, U32 const litLength,
                                 const optState_t* const optPtr)
{
    return ZSTD_rawLiteralsCost(literals, litLength, optPtr)
         + ZSTD_litLengthPrice(litLength, optPtr);
}

/* ZSTD_litLengthContribution() :
 * @return ( cost(litlength) - cost(0) )
 * this value can then be added to rawLiteralsCost()
 * to provide a cost which is directly comparable to a match ending at same position */
static int ZSTD_litLengthContribution(U32 const litLength, const optState_t* const optPtr)
{
    if (optPtr->staticPrices) return ZSTD_highbit32(litLength+1);

    /* literal Length */
    {   U32 const llCode = ZSTD_LLcode(litLength);
        int const contribution = LL_bits[llCode]
                        + ZSTD_highbit32(optPtr->litLengthFreq[0]+1)
                        - ZSTD_highbit32(optPtr->litLengthFreq[llCode]+1);
#if 1
        return contribution;
#else
        return MAX(0, contribution); /* sometimes better, sometimes not ... */
#endif
    }
}

/* ZSTD_literalsContribution() :
 * creates a fake cost for the literals part of a sequence
 * which can be compared to the ending cost of a match
 * should a new match start at this position */
static int ZSTD_literalsContribution(const BYTE* const literals, U32 const litLength,
                                     const optState_t* const optPtr)
{
    int const contribution = ZSTD_rawLiteralsCost(literals, litLength, optPtr)
                           + ZSTD_litLengthContribution(litLength, optPtr);
    return contribution;
}

/* ZSTD_getMatchPrice() :
 * Provides the cost of the match part (offset + matchLength) of a sequence
 * Must be combined with ZSTD_fullLiteralsCost() to get the full cost of a sequence.
 * optLevel: when <2, favors small offset for decompression speed (improved cache efficiency) */
FORCE_INLINE_TEMPLATE U32 ZSTD_getMatchPrice(
                                    U32 const offset, U32 const matchLength,
                                    const optState_t* const optPtr,
                                    int const optLevel)
{
    U32 price;
    U32 const offCode = ZSTD_highbit32(offset+1);
    U32 const mlBase = matchLength - MINMATCH;
    assert(matchLength >= MINMATCH);

    if (optPtr->staticPrices)  /* fixed scheme, do not use statistics */
        return ZSTD_highbit32((U32)mlBase+1) + 16 + offCode;

    price = offCode + optPtr->log2offCodeSum - ZSTD_highbit32(optPtr->offCodeFreq[offCode]+1);
    if ((optLevel<2) /*static*/ && offCode >= 20) price += (offCode-19)*2; /* handicap for long distance offsets, favor decompression speed */

    /* match Length */
    {   U32 const mlCode = ZSTD_MLcode(mlBase);
        price += ML_bits[mlCode] + optPtr->log2matchLengthSum - ZSTD_highbit32(optPtr->matchLengthFreq[mlCode]+1);
    }

    DEBUGLOG(8, "ZSTD_getMatchPrice(ml:%u) = %u", matchLength, price);
    return price;
}

static void ZSTD_updateStats(optState_t* const optPtr,
                             U32 litLength, const BYTE* literals,
                             U32 offsetCode, U32 matchLength)
{
    /* literals */
    {   U32 u;
        for (u=0; u < litLength; u++)
            optPtr->litFreq[literals[u]] += ZSTD_LITFREQ_ADD;
        optPtr->litSum += litLength*ZSTD_LITFREQ_ADD;
    }

    /* literal Length */
    {   U32 const llCode = ZSTD_LLcode(litLength);
        optPtr->litLengthFreq[llCode]++;
        optPtr->litLengthSum++;
    }

    /* match offset code (0-2=>repCode; 3+=>offset+2) */
    {   U32 const offCode = ZSTD_highbit32(offsetCode+1);
        assert(offCode <= MaxOff);
        optPtr->offCodeFreq[offCode]++;
        optPtr->offCodeSum++;
    }

    /* match Length */
    {   U32 const mlBase = matchLength - MINMATCH;
        U32 const mlCode = ZSTD_MLcode(mlBase);
        optPtr->matchLengthFreq[mlCode]++;
        optPtr->matchLengthSum++;
    }
}


/* ZSTD_readMINMATCH() :
 * function safe only for comparisons
 * assumption : memPtr must be at least 4 bytes before end of buffer */
MEM_STATIC U32 ZSTD_readMINMATCH(const void* memPtr, U32 length)
{
    switch (length)
    {
    default :
    case 4 : return MEM_read32(memPtr);
    case 3 : if (MEM_isLittleEndian())
                return MEM_read32(memPtr)<<8;
             else
                return MEM_read32(memPtr)>>8;
    }
}


/* Update hashTable3 up to ip (excluded)
   Assumption : always within prefix (i.e. not within extDict) */
static U32 ZSTD_insertAndFindFirstIndexHash3 (ZSTD_CCtx* const cctx, const BYTE* const ip)
{
    U32* const hashTable3  = cctx->hashTable3;
    U32 const hashLog3  = cctx->hashLog3;
    const BYTE* const base = cctx->base;
    U32 idx = cctx->nextToUpdate3;
    U32 const target = cctx->nextToUpdate3 = (U32)(ip - base);
    size_t const hash3 = ZSTD_hash3Ptr(ip, hashLog3);

    while(idx < target) {
        hashTable3[ZSTD_hash3Ptr(base+idx, hashLog3)] = idx;
        idx++;
    }

    return hashTable3[hash3];
}


/*-*************************************
*  Binary Tree search
***************************************/
FORCE_INLINE_TEMPLATE
U32 ZSTD_insertBtAndGetAllMatches (
                    ZSTD_CCtx* zc,
                    const BYTE* const ip, const BYTE* const iLimit, int const extDict,
                    U32 nbCompares, U32 const mls, U32 const sufficient_len,
                    U32 rep[ZSTD_REP_NUM], U32 const ll0,
                    ZSTD_match_t* matches, const U32 lengthToBeat)
{
    const BYTE* const base = zc->base;
    U32 const current = (U32)(ip-base);
    U32 const hashLog = zc->appliedParams.cParams.hashLog;
    U32 const minMatch = (mls==3) ? 3 : 4;
    U32* const hashTable = zc->hashTable;
    size_t const h  = ZSTD_hashPtr(ip, hashLog, mls);
    U32 matchIndex  = hashTable[h];
    U32* const bt   = zc->chainTable;
    U32 const btLog = zc->appliedParams.cParams.chainLog - 1;
    U32 const btMask= (1U << btLog) - 1;
    size_t commonLengthSmaller=0, commonLengthLarger=0;
    const BYTE* const dictBase = zc->dictBase;
    U32 const dictLimit = zc->dictLimit;
    const BYTE* const dictEnd = dictBase + dictLimit;
    const BYTE* const prefixStart = base + dictLimit;
    U32 const btLow = btMask >= current ? 0 : current - btMask;
    U32 const windowLow = zc->lowLimit;
    U32* smallerPtr = bt + 2*(current&btMask);
    U32* largerPtr  = bt + 2*(current&btMask) + 1;
    U32 matchEndIdx = current+8+1;   /* farthest referenced position of any match => detects repetitive patterns */
    U32 dummy32;   /* to be nullified at the end */
    U32 mnum = 0;

    size_t bestLength = lengthToBeat-1;
    DEBUGLOG(7, "ZSTD_insertBtAndGetAllMatches");

    /* check repCode */
    {   U32 const lastR = ZSTD_REP_NUM + ll0;
        U32 repCode;
        for (repCode = ll0; repCode < lastR; repCode++) {
            U32 const repOffset = (repCode==ZSTD_REP_NUM) ? (rep[0] - 1) : rep[repCode];
            U32 const repIndex = current - repOffset;
            U32 repLen = 0;
            assert(current >= dictLimit);
            if (repOffset-1 /* intentional overflow, discards 0 and -1 */ < current-dictLimit) {  /* equivalent to `current > repIndex >= dictLimit` */
                if (ZSTD_readMINMATCH(ip, minMatch) == ZSTD_readMINMATCH(ip - repOffset, minMatch)) {
                    repLen = (U32)ZSTD_count(ip+minMatch, ip+minMatch-repOffset, iLimit) + minMatch;
                }
            } else {  /* repIndex < dictLimit || repIndex >= current */
                const BYTE* const repMatch = dictBase + repIndex;
                assert(current >= windowLow);
                if ( extDict /* this case only valid in extDict mode */
                  && ( ((repOffset-1) /*intentional overflow*/ < current - windowLow)  /* equivalent to `current > repIndex >= windowLow` */
                     & (((U32)((dictLimit-1) - repIndex) >= 3) ) /* intentional overflow : do not test positions overlapping 2 memory segments */)
                  && (ZSTD_readMINMATCH(ip, minMatch) == ZSTD_readMINMATCH(repMatch, minMatch)) ) {
                    repLen = (U32)ZSTD_count_2segments(ip+minMatch, repMatch+minMatch, iLimit, dictEnd, prefixStart) + minMatch;
            }   }
            /* save longer solution */
            if (repLen > bestLength) {
                DEBUGLOG(8, "found rep-match %u of length %u",
                            repCode - ll0, (U32)repLen);
                bestLength = repLen;
                matches[mnum].off = repCode - ll0;
                matches[mnum].len = (U32)repLen;
                mnum++;
                if ( (repLen > sufficient_len)
                   | (ip+repLen == iLimit) ) {  /* best possible */
                    return mnum;
    }   }   }   }

    /* HC3 match finder */
    if ((mls == 3) /*static*/ && (bestLength < mls)) {
        U32 const matchIndex3 = ZSTD_insertAndFindFirstIndexHash3 (zc, ip);
        if ((matchIndex3 > windowLow)
          & (current - matchIndex3 < (1<<18)) /*heuristic : longer distance likely too expensive*/ ) {
            size_t mlen;
            if ((!extDict) /*static*/ || (matchIndex3 >= dictLimit)) {
                const BYTE* const match = base + matchIndex3;
                mlen = ZSTD_count(ip, match, iLimit);
            } else {
                const BYTE* const match = dictBase + matchIndex3;
                mlen = ZSTD_count_2segments(ip, match, iLimit, dictEnd, prefixStart);
            }

            /* save best solution */
            if (mlen >= mls /* == 3 > bestLength */) {
                DEBUGLOG(8, "found small match with hlog3, of length %u",
                            (U32)mlen);
                bestLength = mlen;
                assert(current > matchIndex3);
                assert(mnum==0);  /* no prior solution */
                matches[0].off = (current - matchIndex3) + ZSTD_REP_MOVE;
                matches[0].len = (U32)mlen;
                mnum = 1;
                if ( (mlen > sufficient_len) |
                     (ip+mlen == iLimit) ) {  /* best possible length */
                    zc->nextToUpdate = current+1;  /* skip insertion */
                    return 1;
    }   }   }   }

    hashTable[h] = current;   /* Update Hash Table */

    while (nbCompares-- && (matchIndex > windowLow)) {
        U32* const nextPtr = bt + 2*(matchIndex & btMask);
        size_t matchLength = MIN(commonLengthSmaller, commonLengthLarger);   /* guaranteed minimum nb of common bytes */
        const BYTE* match;
        assert(current > matchIndex);

        if ((!extDict) || (matchIndex+matchLength >= dictLimit)) {
            assert(matchIndex+matchLength >= dictLimit);  /* ensure the condition is correct when !extDict */
            match = base + matchIndex;
            matchLength += ZSTD_count(ip+matchLength, match+matchLength, iLimit);
        } else {
            match = dictBase + matchIndex;
            matchLength += ZSTD_count_2segments(ip+matchLength, match+matchLength, iLimit, dictEnd, prefixStart);
            if (matchIndex+matchLength >= dictLimit)
                match = base + matchIndex;   /* prepare for match[matchLength] */
        }

        if (matchLength > bestLength) {
            DEBUGLOG(8, "found match of length %u at distance %u",
                    (U32)matchLength, current - matchIndex);
            assert(matchEndIdx > matchIndex);
            if (matchLength > matchEndIdx - matchIndex)
                matchEndIdx = matchIndex + (U32)matchLength;
            bestLength = matchLength;
            matches[mnum].off = (current - matchIndex) + ZSTD_REP_MOVE;
            matches[mnum].len = (U32)matchLength;
            mnum++;
            if (matchLength > ZSTD_OPT_NUM) break;
            if (ip+matchLength == iLimit) {  /* equal : no way to know if inf or sup */
                break;   /* drop, to preserve bt consistency (miss a little bit of compression) */
            }
        }

        if (match[matchLength] < ip[matchLength]) {
            /* match smaller than current */
            *smallerPtr = matchIndex;             /* update smaller idx */
            commonLengthSmaller = matchLength;    /* all smaller will now have at least this guaranteed common length */
            if (matchIndex <= btLow) { smallerPtr=&dummy32; break; }   /* beyond tree size, stop the search */
            smallerPtr = nextPtr+1;               /* new candidate => larger than match, which was smaller than current */
            matchIndex = nextPtr[1];              /* new matchIndex, larger than previous, closer to current */
        } else {
            *largerPtr = matchIndex;
            commonLengthLarger = matchLength;
            if (matchIndex <= btLow) { largerPtr=&dummy32; break; }   /* beyond tree size, stop the search */
            largerPtr = nextPtr;
            matchIndex = nextPtr[0];
    }   }

    *smallerPtr = *largerPtr = 0;

    assert(matchEndIdx > current+8);
    zc->nextToUpdate = matchEndIdx - 8;  /* skip repetitive patterns */
    return mnum;
}


FORCE_INLINE_TEMPLATE U32 ZSTD_BtGetAllMatches (
                        ZSTD_CCtx* zc,   /* Index table will be updated */
                        const BYTE* ip, const BYTE* const iHighLimit, int const extDict,
                        U32 const maxNbAttempts, U32 const matchLengthSearch, U32 const sufficient_len,
                        U32 rep[ZSTD_REP_NUM], U32 const ll0,
                        ZSTD_match_t* matches, U32 const lengthToBeat)
{
    DEBUGLOG(7, "ZSTD_BtGetAllMatches");
    if (ip < zc->base + zc->nextToUpdate) return 0;   /* skipped area */
    if (extDict) ZSTD_updateTree_extDict(zc, ip, iHighLimit, maxNbAttempts, matchLengthSearch);
    else ZSTD_updateTree(zc, ip, iHighLimit, maxNbAttempts, matchLengthSearch);
    switch(matchLengthSearch)
    {
    case 3 : return ZSTD_insertBtAndGetAllMatches(zc, ip, iHighLimit, extDict, maxNbAttempts, 3, sufficient_len, rep, ll0, matches, lengthToBeat);
    default :
    case 4 : return ZSTD_insertBtAndGetAllMatches(zc, ip, iHighLimit, extDict, maxNbAttempts, 4, sufficient_len, rep, ll0, matches, lengthToBeat);
    case 5 : return ZSTD_insertBtAndGetAllMatches(zc, ip, iHighLimit, extDict, maxNbAttempts, 5, sufficient_len, rep, ll0, matches, lengthToBeat);
    case 7 :
    case 6 : return ZSTD_insertBtAndGetAllMatches(zc, ip, iHighLimit, extDict, maxNbAttempts, 6, sufficient_len, rep, ll0, matches, lengthToBeat);
    }
}


/*-*******************************
*  Optimal parser
*********************************/
typedef struct repcodes_s {
    U32 rep[3];
} repcodes_t;

repcodes_t ZSTD_updateRep(U32 const rep[3], U32 const offset, U32 const ll0)
{
    repcodes_t newReps;
    if (offset >= ZSTD_REP_NUM) {  /* full offset */
        newReps.rep[2] = rep[1];
        newReps.rep[1] = rep[0];
        newReps.rep[0] = offset - ZSTD_REP_MOVE;
    } else {   /* repcode */
        U32 const repCode = offset + ll0;
        if (repCode > 0) {  /* note : if repCode==0, no change */
            U32 const currentOffset = (repCode==ZSTD_REP_NUM) ? (rep[0] - 1) : rep[repCode];
            newReps.rep[2] = (repCode >= 2) ? rep[1] : rep[2];
            newReps.rep[1] = rep[0];
            newReps.rep[0] = currentOffset;
        } else {   /* repCode == 0 */
            memcpy(&newReps, rep, sizeof(newReps));
        }
    }
    return newReps;
}


typedef struct {
    const BYTE* anchor;
    U32 litlen;
    U32 rawLitCost;
} cachedLiteralPrice_t;

static U32 ZSTD_rawLiteralsCost_cached(
                            cachedLiteralPrice_t* const cachedLitPrice,
                            const BYTE* const anchor, U32 const litlen,
                            const optState_t* const optStatePtr)
{
    U32 startCost;
    U32 remainingLength;
    const BYTE* startPosition;

    if (anchor == cachedLitPrice->anchor) {
        startCost = cachedLitPrice->rawLitCost;
        startPosition = anchor + cachedLitPrice->litlen;
        assert(litlen >= cachedLitPrice->litlen);
        remainingLength = litlen - cachedLitPrice->litlen;
    } else {
        startCost = 0;
        startPosition = anchor;
        remainingLength = litlen;
    }

    {   U32 const rawLitCost = startCost + ZSTD_rawLiteralsCost(startPosition, remainingLength, optStatePtr);
        cachedLitPrice->anchor = anchor;
        cachedLitPrice->litlen = litlen;
        cachedLitPrice->rawLitCost = rawLitCost;
        return rawLitCost;
    }
}

static U32 ZSTD_fullLiteralsCost_cached(
                            cachedLiteralPrice_t* const cachedLitPrice,
                            const BYTE* const anchor, U32 const litlen,
                            const optState_t* const optStatePtr)
{
    return ZSTD_rawLiteralsCost_cached(cachedLitPrice, anchor, litlen, optStatePtr)
         + ZSTD_litLengthPrice(litlen, optStatePtr);
}

static int ZSTD_literalsContribution_cached(
                            cachedLiteralPrice_t* const cachedLitPrice,
                            const BYTE* const anchor, U32 const litlen,
                            const optState_t* const optStatePtr)
{
    int const contribution = ZSTD_rawLiteralsCost_cached(cachedLitPrice, anchor, litlen, optStatePtr)
                           + ZSTD_litLengthContribution(litlen, optStatePtr);
    return contribution;
}

FORCE_INLINE_TEMPLATE
size_t ZSTD_compressBlock_opt_generic(ZSTD_CCtx* ctx,
                                      const void* src, size_t srcSize,
                                      const int optLevel, const int extDict)
{
    seqStore_t* const seqStorePtr = &(ctx->seqStore);
    optState_t* const optStatePtr = &(ctx->optState);
    const BYTE* const istart = (const BYTE*)src;
    const BYTE* ip = istart;
    const BYTE* anchor = istart;
    const BYTE* const iend = istart + srcSize;
    const BYTE* const ilimit = iend - 8;
    const BYTE* const base = ctx->base;
    const BYTE* const prefixStart = base + ctx->dictLimit;

    U32 const maxSearches = 1U << ctx->appliedParams.cParams.searchLog;
    U32 const sufficient_len = MIN(ctx->appliedParams.cParams.targetLength, ZSTD_OPT_NUM -1);
    U32 const mls = ctx->appliedParams.cParams.searchLength;
    U32 const minMatch = (ctx->appliedParams.cParams.searchLength == 3) ? 3 : 4;

    ZSTD_optimal_t* const opt = optStatePtr->priceTable;
    ZSTD_match_t* const matches = optStatePtr->matchTable;
    cachedLiteralPrice_t cachedLitPrice;
    U32 rep[ZSTD_REP_NUM];

    /* init */
    DEBUGLOG(5, "ZSTD_compressBlock_opt_generic");
    ctx->nextToUpdate3 = ctx->nextToUpdate;
    ZSTD_rescaleFreqs(optStatePtr, (const BYTE*)src, srcSize);
    ip += (ip==prefixStart);
    { int i; for (i=0; i<ZSTD_REP_NUM; i++) rep[i]=seqStorePtr->rep[i]; }
    memset(&cachedLitPrice, 0, sizeof(cachedLitPrice));

    /* Match Loop */
    while (ip < ilimit) {
        U32 cur, last_pos = 0;
        U32 best_mlen, best_off;

        /* find first match */
        {   U32 const litlen = (U32)(ip - anchor);
            U32 const ll0 = !litlen;
            U32 const nbMatches = ZSTD_BtGetAllMatches(ctx, ip, iend, extDict, maxSearches, mls, sufficient_len, rep, ll0, matches, minMatch);
            if (!nbMatches) { ip++; continue; }

            /* initialize opt[0] */
            { U32 i ; for (i=0; i<ZSTD_REP_NUM; i++) opt[0].rep[i] = rep[i]; }
            opt[0].mlen = 1;
            opt[0].litlen = litlen;

            /* large match -> immediate encoding */
            {   U32 const maxML = matches[nbMatches-1].len;
                DEBUGLOG(7, "found %u matches of maxLength=%u and offset=%u at cPos=%u => start new serie",
                            nbMatches, maxML, matches[nbMatches-1].off, (U32)(ip-prefixStart));

                if (maxML > sufficient_len) {
                    best_mlen = maxML;
                    best_off = matches[nbMatches-1].off;
                    DEBUGLOG(7, "large match (%u>%u), immediate encoding",
                                best_mlen, sufficient_len);
                    cur = 0;
                    last_pos = 1;
                    goto _shortestPath;
            }   }

            /* set prices for first matches starting position == 0 */
            {   U32 const literalsPrice = ZSTD_fullLiteralsCost_cached(&cachedLitPrice, anchor, litlen, optStatePtr);
                U32 pos;
                U32 matchNb;
                for (pos = 0; pos < minMatch; pos++) {
                    opt[pos].mlen = 1;
                    opt[pos].price = ZSTD_MAX_PRICE;
                }
                for (matchNb = 0; matchNb < nbMatches; matchNb++) {
                    U32 const offset = matches[matchNb].off;
                    U32 const end = matches[matchNb].len;
                    repcodes_t const repHistory = ZSTD_updateRep(rep, offset, ll0);
                    for ( ; pos <= end ; pos++ ) {
                        U32 const matchPrice = literalsPrice + ZSTD_getMatchPrice(offset, pos, optStatePtr, optLevel);
                        DEBUGLOG(7, "rPos:%u => set initial price : %u",
                                    pos, matchPrice);
                        opt[pos].mlen = pos;
                        opt[pos].off = offset;
                        opt[pos].litlen = litlen;
                        opt[pos].price = matchPrice;
                        memcpy(opt[pos].rep, &repHistory, sizeof(repHistory));
                }   }
                last_pos = pos-1;
            }
        }

        /* check further positions */
        for (cur = 1; cur <= last_pos; cur++) {
            const BYTE* const inr = ip + cur;
            assert(cur < ZSTD_OPT_NUM);

            /* Fix current position with one literal if cheaper */
            {   U32 const litlen = (opt[cur-1].mlen == 1) ? opt[cur-1].litlen + 1 : 1;
                int price;  /* note : contribution can be negative */
                if (cur > litlen) {
                    price = opt[cur - litlen].price + ZSTD_literalsContribution(inr-litlen, litlen, optStatePtr);
                } else {
                    price = ZSTD_literalsContribution_cached(&cachedLitPrice, anchor, litlen, optStatePtr);
                }
                assert(price < 1000000000); /* overflow check */
                if (price <= opt[cur].price) {
                    DEBUGLOG(7, "rPos:%u : better price (%u<%u) using literal",
                                cur, price, opt[cur].price);
                    opt[cur].mlen = 1;
                    opt[cur].off = 0;
                    opt[cur].litlen = litlen;
                    opt[cur].price = price;
                    memcpy(opt[cur].rep, opt[cur-1].rep, sizeof(opt[cur].rep));
            }   }

            /* last match must start at a minimum distance of 8 from oend */
            if (inr > ilimit) continue;

            if (cur == last_pos) break;

             if ( (optLevel==0) /*static*/
               && (opt[cur+1].price <= opt[cur].price) )
                continue;  /* skip unpromising positions; about ~+6% speed, -0.01 ratio */

            {   U32 const ll0 = (opt[cur].mlen != 1);
                U32 const litlen = (opt[cur].mlen == 1) ? opt[cur].litlen : 0;
                U32 const previousPrice = (cur > litlen) ? opt[cur-litlen].price : 0;
                U32 const basePrice = previousPrice + ZSTD_fullLiteralsCost(inr-litlen, litlen, optStatePtr);
                U32 const nbMatches = ZSTD_BtGetAllMatches(ctx, inr, iend, extDict, maxSearches, mls, sufficient_len, opt[cur].rep, ll0, matches, minMatch);
                U32 matchNb;
                if (!nbMatches) continue;

                {   U32 const maxML = matches[nbMatches-1].len;
                    DEBUGLOG(7, "rPos:%u, found %u matches, of maxLength=%u",
                                cur, nbMatches, maxML);

                    if ( (maxML > sufficient_len)
                       | (cur + maxML >= ZSTD_OPT_NUM) ) {
                        best_mlen = maxML;
                        best_off = matches[nbMatches-1].off;
                        last_pos = cur + 1;
                        goto _shortestPath;
                    }
                }

                /* set prices using matches found at position == cur */
                for (matchNb = 0; matchNb < nbMatches; matchNb++) {
                    U32 const offset = matches[matchNb].off;
                    repcodes_t const repHistory = ZSTD_updateRep(opt[cur].rep, offset, ll0);
                    U32 const lastML = matches[matchNb].len;
                    U32 const startML = (matchNb>0) ? matches[matchNb-1].len+1 : minMatch;
                    U32 mlen;

                    DEBUGLOG(7, "testing match %u => offCode=%u, mlen=%u, llen=%u",
                                matchNb, matches[matchNb].off, lastML, litlen);

                    for (mlen = lastML; mlen >= startML; mlen--) {
                        U32 const pos = cur + mlen;
                        int const price = basePrice + ZSTD_getMatchPrice(offset, mlen, optStatePtr, optLevel);

                        if ((pos > last_pos) || (price < opt[pos].price)) {
                            DEBUGLOG(7, "rPos:%u => new better price (%u<%u)",
                                        pos, price, opt[pos].price);
                            while (last_pos < pos) { opt[last_pos+1].price = ZSTD_MAX_PRICE; last_pos++; }
                            opt[pos].mlen = mlen;
                            opt[pos].off = offset;
                            opt[pos].litlen = litlen;
                            opt[pos].price = price;
                            memcpy(opt[pos].rep, &repHistory, sizeof(repHistory));
                        } else {
                            if (optLevel==0) break;  /* gets ~+10% speed for about -0.01 ratio loss */
                        }
            }   }   }
        }  /* for (cur = 1; cur <= last_pos; cur++) */

        best_mlen = opt[last_pos].mlen;
        best_off = opt[last_pos].off;
        cur = last_pos - best_mlen;

_shortestPath:   /* cur, last_pos, best_mlen, best_off have to be set */
        assert(opt[0].mlen == 1);

        /* reverse traversal */
        DEBUGLOG(7, "start reverse traversal (last_pos:%u, cur:%u)",
                    last_pos, cur);
        {   U32 selectedMatchLength = best_mlen;
            U32 selectedOffset = best_off;
            U32 pos = cur;
            while (1) {
                U32 const mlen = opt[pos].mlen;
                U32 const off = opt[pos].off;
                opt[pos].mlen = selectedMatchLength;
                opt[pos].off = selectedOffset;
                selectedMatchLength = mlen;
                selectedOffset = off;
                if (mlen > pos) break;
                pos -= mlen;
        }   }

        /* save sequences */
        {   U32 pos;
            for (pos=0; pos < last_pos; ) {
                U32 const llen = (U32)(ip - anchor);
                U32 const mlen = opt[pos].mlen;
                U32 const offset = opt[pos].off;
                if (mlen == 1) { ip++; pos++; continue; }  /* literal position => move on */
                pos += mlen; ip += mlen;

                /* repcodes update : like ZSTD_updateRep(), but update in place */
                if (offset >= ZSTD_REP_NUM) {  /* full offset */
                    rep[2] = rep[1];
                    rep[1] = rep[0];
                    rep[0] = offset - ZSTD_REP_MOVE;
                } else {   /* repcode */
                    U32 const repCode = offset + (llen==0);
                    if (repCode) {  /* note : if repCode==0, no change */
                        U32 const currentOffset = (repCode==ZSTD_REP_NUM) ? (rep[0] - 1) : rep[repCode];
                        if (repCode >= 2) rep[2] = rep[1];
                        rep[1] = rep[0];
                        rep[0] = currentOffset;
                    }
                }

                ZSTD_updateStats(optStatePtr, llen, anchor, offset, mlen);
                ZSTD_storeSeq(seqStorePtr, llen, anchor, offset, mlen-MINMATCH);
                anchor = ip;
        }   }
        ZSTD_setLog2Prices(optStatePtr);
    }   /* while (ip < ilimit) */

    /* Save reps for next block */
    { int i; for (i=0; i<ZSTD_REP_NUM; i++) seqStorePtr->repToConfirm[i] = rep[i]; }

    /* Return the last literals size */
    return iend - anchor;
}


size_t ZSTD_compressBlock_btopt(ZSTD_CCtx* ctx, const void* src, size_t srcSize)
{
    DEBUGLOG(5, "ZSTD_compressBlock_btopt");
    return ZSTD_compressBlock_opt_generic(ctx, src, srcSize, 0 /*optLevel*/, 0 /*extDict*/);
}

size_t ZSTD_compressBlock_btultra(ZSTD_CCtx* ctx, const void* src, size_t srcSize)
{
    return ZSTD_compressBlock_opt_generic(ctx, src, srcSize, 2 /*optLevel*/, 0 /*extDict*/);
}

size_t ZSTD_compressBlock_btopt_extDict(ZSTD_CCtx* ctx, const void* src, size_t srcSize)
{
    return ZSTD_compressBlock_opt_generic(ctx, src, srcSize, 0 /*optLevel*/, 1 /*extDict*/);
}

size_t ZSTD_compressBlock_btultra_extDict(ZSTD_CCtx* ctx, const void* src, size_t srcSize)
{
    return ZSTD_compressBlock_opt_generic(ctx, src, srcSize, 2 /*optLevel*/, 1 /*extDict*/);
}
