/*
 * Copyright (c) 2016-present, Przemyslaw Skibinski, Yann Collet, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under both the BSD-style license (found in the
 * LICENSE file in the root directory of this source tree) and the GPLv2 (found
 * in the COPYING file in the root directory of this source tree).
 * You may select, at your option, one of the above-listed licenses.
 */

#include "zstd_opt.h"
#include "zstd_lazy.h"


#define ZSTD_LITFREQ_ADD    2
#define ZSTD_FREQ_DIV       4
#define ZSTD_MAX_PRICE      (1<<30)

/*-*************************************
*  Price functions for optimal parser
***************************************/
static void ZSTD_setLog2Prices(optState_t* optPtr)
{
    optPtr->log2matchLengthSum = ZSTD_highbit32(optPtr->matchLengthSum+1);
    optPtr->log2litLengthSum = ZSTD_highbit32(optPtr->litLengthSum+1);
    optPtr->log2litSum = ZSTD_highbit32(optPtr->litSum+1);
    optPtr->log2offCodeSum = ZSTD_highbit32(optPtr->offCodeSum+1);
    optPtr->factor = 1 + ((optPtr->litSum>>5) / optPtr->litLengthSum) + ((optPtr->litSum<<1) / (optPtr->litSum + optPtr->matchSum));
}


static void ZSTD_rescaleFreqs(optState_t* optPtr, const BYTE* src, size_t srcSize)
{
    unsigned u;

    optPtr->cachedLiterals = NULL;
    optPtr->cachedPrice = optPtr->cachedLitLength = 0;
    optPtr->staticPrices = 0;

    if (optPtr->litLengthSum == 0) {
        if (srcSize <= 1024) optPtr->staticPrices = 1;

        assert(optPtr->litFreq!=NULL);
        for (u=0; u<=MaxLit; u++)
            optPtr->litFreq[u] = 0;
        for (u=0; u<srcSize; u++)
            optPtr->litFreq[src[u]]++;

        optPtr->litSum = 0;
        optPtr->litLengthSum = MaxLL+1;
        optPtr->matchLengthSum = MaxML+1;
        optPtr->offCodeSum = (MaxOff+1);
        optPtr->matchSum = (ZSTD_LITFREQ_ADD<<Litbits);

        for (u=0; u<=MaxLit; u++) {
            optPtr->litFreq[u] = 1 + (optPtr->litFreq[u]>>ZSTD_FREQ_DIV);
            optPtr->litSum += optPtr->litFreq[u];
        }
        for (u=0; u<=MaxLL; u++)
            optPtr->litLengthFreq[u] = 1;
        for (u=0; u<=MaxML; u++)
            optPtr->matchLengthFreq[u] = 1;
        for (u=0; u<=MaxOff; u++)
            optPtr->offCodeFreq[u] = 1;
    } else {
        optPtr->matchLengthSum = 0;
        optPtr->litLengthSum = 0;
        optPtr->offCodeSum = 0;
        optPtr->matchSum = 0;
        optPtr->litSum = 0;

        for (u=0; u<=MaxLit; u++) {
            optPtr->litFreq[u] = 1 + (optPtr->litFreq[u]>>(ZSTD_FREQ_DIV+1));
            optPtr->litSum += optPtr->litFreq[u];
        }
        for (u=0; u<=MaxLL; u++) {
            optPtr->litLengthFreq[u] = 1 + (optPtr->litLengthFreq[u]>>(ZSTD_FREQ_DIV+1));
            optPtr->litLengthSum += optPtr->litLengthFreq[u];
        }
        for (u=0; u<=MaxML; u++) {
            optPtr->matchLengthFreq[u] = 1 + (optPtr->matchLengthFreq[u]>>ZSTD_FREQ_DIV);
            optPtr->matchLengthSum += optPtr->matchLengthFreq[u];
            optPtr->matchSum += optPtr->matchLengthFreq[u] * (u + 3);
        }
        optPtr->matchSum *= ZSTD_LITFREQ_ADD;
        for (u=0; u<=MaxOff; u++) {
            optPtr->offCodeFreq[u] = 1 + (optPtr->offCodeFreq[u]>>ZSTD_FREQ_DIV);
            optPtr->offCodeSum += optPtr->offCodeFreq[u];
        }
    }

    ZSTD_setLog2Prices(optPtr);
}


static U32 ZSTD_getLiteralPrice(optState_t* optPtr, U32 litLength, const BYTE* literals)
{
    U32 price, u;

    if (optPtr->staticPrices)
        return ZSTD_highbit32((U32)litLength+1) + (litLength*6);

    if (litLength == 0)
        return optPtr->log2litLengthSum - ZSTD_highbit32(optPtr->litLengthFreq[0]+1);

    /* literals */
    if (optPtr->cachedLiterals == literals) {
        U32 const additional = litLength - optPtr->cachedLitLength;
        const BYTE* literals2 = optPtr->cachedLiterals + optPtr->cachedLitLength;
        price = optPtr->cachedPrice + additional * optPtr->log2litSum;
        for (u=0; u < additional; u++)
            price -= ZSTD_highbit32(optPtr->litFreq[literals2[u]]+1);
        optPtr->cachedPrice = price;
        optPtr->cachedLitLength = litLength;
    } else {
        price = litLength * optPtr->log2litSum;
        for (u=0; u < litLength; u++)
            price -= ZSTD_highbit32(optPtr->litFreq[literals[u]]+1);

        if (litLength >= 12) {
            optPtr->cachedLiterals = literals;
            optPtr->cachedPrice = price;
            optPtr->cachedLitLength = litLength;
        }
    }

    /* literal Length */
    {   const BYTE LL_deltaCode = 19;
        const BYTE llCode = (litLength>63) ? (BYTE)ZSTD_highbit32(litLength) + LL_deltaCode : LL_Code[litLength];
        price += LL_bits[llCode] + optPtr->log2litLengthSum - ZSTD_highbit32(optPtr->litLengthFreq[llCode]+1);
    }

    return price;
}


FORCE_INLINE_TEMPLATE U32 ZSTD_getPrice(optState_t* optPtr, U32 litLength, const BYTE* literals, U32 offset, U32 matchLength, const int ultra)
{
    /* offset */
    U32 price;
    BYTE const offCode = (BYTE)ZSTD_highbit32(offset+1);

    if (optPtr->staticPrices)
        return ZSTD_getLiteralPrice(optPtr, litLength, literals) + ZSTD_highbit32((U32)matchLength+1) + 16 + offCode;

    price = offCode + optPtr->log2offCodeSum - ZSTD_highbit32(optPtr->offCodeFreq[offCode]+1);
    if (!ultra && offCode >= 20) price += (offCode-19)*2;

    /* match Length */
    {   const BYTE ML_deltaCode = 36;
        const BYTE mlCode = (matchLength>127) ? (BYTE)ZSTD_highbit32(matchLength) + ML_deltaCode : ML_Code[matchLength];
        price += ML_bits[mlCode] + optPtr->log2matchLengthSum - ZSTD_highbit32(optPtr->matchLengthFreq[mlCode]+1);
    }

    return price + ZSTD_getLiteralPrice(optPtr, litLength, literals) + optPtr->factor;
}


static void ZSTD_updatePrice(optState_t* optPtr, U32 litLength, const BYTE* literals, U32 offset, U32 matchLength)
{
    U32 u;

    /* literals */
    optPtr->litSum += litLength*ZSTD_LITFREQ_ADD;
    for (u=0; u < litLength; u++)
        optPtr->litFreq[literals[u]] += ZSTD_LITFREQ_ADD;

    /* literal Length */
    {   const BYTE LL_deltaCode = 19;
        const BYTE llCode = (litLength>63) ? (BYTE)ZSTD_highbit32(litLength) + LL_deltaCode : LL_Code[litLength];
        optPtr->litLengthFreq[llCode]++;
        optPtr->litLengthSum++;
    }

    /* match offset */
    {   BYTE const offCode = (BYTE)ZSTD_highbit32(offset+1);
        optPtr->offCodeSum++;
        optPtr->offCodeFreq[offCode]++;
    }

    /* match Length */
    {   const BYTE ML_deltaCode = 36;
        const BYTE mlCode = (matchLength>127) ? (BYTE)ZSTD_highbit32(matchLength) + ML_deltaCode : ML_Code[matchLength];
        optPtr->matchLengthFreq[mlCode]++;
        optPtr->matchLengthSum++;
    }

    ZSTD_setLog2Prices(optPtr);
}


#define SET_PRICE(pos, mlen_, offset_, litlen_, price_)   \
    {                                                 \
        while (last_pos < pos)  { opt[last_pos+1].price = ZSTD_MAX_PRICE; last_pos++; } \
        opt[pos].mlen = mlen_;                         \
        opt[pos].off = offset_;                        \
        opt[pos].litlen = litlen_;                     \
        opt[pos].price = price_;                       \
    }


/* function safe only for comparisons */
static U32 ZSTD_readMINMATCH(const void* memPtr, U32 length)
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
static
U32 ZSTD_insertAndFindFirstIndexHash3 (ZSTD_CCtx* zc, const BYTE* ip)
{
    U32* const hashTable3  = zc->hashTable3;
    U32 const hashLog3  = zc->hashLog3;
    const BYTE* const base = zc->base;
    U32 idx = zc->nextToUpdate3;
    const U32 target = zc->nextToUpdate3 = (U32)(ip - base);
    const size_t hash3 = ZSTD_hash3Ptr(ip, hashLog3);

    while(idx < target) {
        hashTable3[ZSTD_hash3Ptr(base+idx, hashLog3)] = idx;
        idx++;
    }

    return hashTable3[hash3];
}


/*-*************************************
*  Binary Tree search
***************************************/
static U32 ZSTD_insertBtAndGetAllMatches (
                        ZSTD_CCtx* zc,
                        const BYTE* const ip, const BYTE* const iLimit,
                        U32 nbCompares, const U32 mls,
                        U32 extDict, ZSTD_match_t* matches, const U32 minMatchLen)
{
    const BYTE* const base = zc->base;
    const U32 current = (U32)(ip-base);
    const U32 hashLog = zc->appliedParams.cParams.hashLog;
    const size_t h  = ZSTD_hashPtr(ip, hashLog, mls);
    U32* const hashTable = zc->hashTable;
    U32 matchIndex  = hashTable[h];
    U32* const bt   = zc->chainTable;
    const U32 btLog = zc->appliedParams.cParams.chainLog - 1;
    const U32 btMask= (1U << btLog) - 1;
    size_t commonLengthSmaller=0, commonLengthLarger=0;
    const BYTE* const dictBase = zc->dictBase;
    const U32 dictLimit = zc->dictLimit;
    const BYTE* const dictEnd = dictBase + dictLimit;
    const BYTE* const prefixStart = base + dictLimit;
    const U32 btLow = btMask >= current ? 0 : current - btMask;
    const U32 windowLow = zc->lowLimit;
    U32* smallerPtr = bt + 2*(current&btMask);
    U32* largerPtr  = bt + 2*(current&btMask) + 1;
    U32 matchEndIdx = current+8;
    U32 dummy32;   /* to be nullified at the end */
    U32 mnum = 0;

    const U32 minMatch = (mls == 3) ? 3 : 4;
    size_t bestLength = minMatchLen-1;

    if (minMatch == 3) { /* HC3 match finder */
        U32 const matchIndex3 = ZSTD_insertAndFindFirstIndexHash3 (zc, ip);
        if (matchIndex3>windowLow && (current - matchIndex3 < (1<<18))) {
            const BYTE* match;
            size_t currentMl=0;
            if ((!extDict) || matchIndex3 >= dictLimit) {
                match = base + matchIndex3;
                if (match[bestLength] == ip[bestLength]) currentMl = ZSTD_count(ip, match, iLimit);
            } else {
                match = dictBase + matchIndex3;
                if (ZSTD_readMINMATCH(match, MINMATCH) == ZSTD_readMINMATCH(ip, MINMATCH))    /* assumption : matchIndex3 <= dictLimit-4 (by table construction) */
                    currentMl = ZSTD_count_2segments(ip+MINMATCH, match+MINMATCH, iLimit, dictEnd, prefixStart) + MINMATCH;
            }

            /* save best solution */
            if (currentMl > bestLength) {
                bestLength = currentMl;
                matches[mnum].off = ZSTD_REP_MOVE_OPT + current - matchIndex3;
                matches[mnum].len = (U32)currentMl;
                mnum++;
                if (currentMl > ZSTD_OPT_NUM) goto update;
                if (ip+currentMl == iLimit) goto update; /* best possible, and avoid read overflow*/
            }
        }
    }

    hashTable[h] = current;   /* Update Hash Table */

    while (nbCompares-- && (matchIndex > windowLow)) {
        U32* nextPtr = bt + 2*(matchIndex & btMask);
        size_t matchLength = MIN(commonLengthSmaller, commonLengthLarger);   /* guaranteed minimum nb of common bytes */
        const BYTE* match;

        if ((!extDict) || (matchIndex+matchLength >= dictLimit)) {
            match = base + matchIndex;
            if (match[matchLength] == ip[matchLength]) {
                matchLength += ZSTD_count(ip+matchLength+1, match+matchLength+1, iLimit) +1;
            }
        } else {
            match = dictBase + matchIndex;
            matchLength += ZSTD_count_2segments(ip+matchLength, match+matchLength, iLimit, dictEnd, prefixStart);
            if (matchIndex+matchLength >= dictLimit)
                match = base + matchIndex;   /* to prepare for next usage of match[matchLength] */
        }

        if (matchLength > bestLength) {
            if (matchLength > matchEndIdx - matchIndex) matchEndIdx = matchIndex + (U32)matchLength;
            bestLength = matchLength;
            matches[mnum].off = ZSTD_REP_MOVE_OPT + current - matchIndex;
            matches[mnum].len = (U32)matchLength;
            mnum++;
            if (matchLength > ZSTD_OPT_NUM) break;
            if (ip+matchLength == iLimit)   /* equal : no way to know if inf or sup */
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

update:
    zc->nextToUpdate = (matchEndIdx > current + 8) ? matchEndIdx - 8 : current+1;
    return mnum;
}


/** Tree updater, providing best match */
static U32 ZSTD_BtGetAllMatches (
                        ZSTD_CCtx* zc,
                        const BYTE* const ip, const BYTE* const iLimit,
                        const U32 maxNbAttempts, const U32 mls, ZSTD_match_t* matches, const U32 minMatchLen)
{
    if (ip < zc->base + zc->nextToUpdate) return 0;   /* skipped area */
    ZSTD_updateTree(zc, ip, iLimit, maxNbAttempts, mls);
    return ZSTD_insertBtAndGetAllMatches(zc, ip, iLimit, maxNbAttempts, mls, 0, matches, minMatchLen);
}


static U32 ZSTD_BtGetAllMatches_selectMLS (
                        ZSTD_CCtx* zc,   /* Index table will be updated */
                        const BYTE* ip, const BYTE* const iHighLimit,
                        const U32 maxNbAttempts, const U32 matchLengthSearch, ZSTD_match_t* matches, const U32 minMatchLen)
{
    switch(matchLengthSearch)
    {
    case 3 : return ZSTD_BtGetAllMatches(zc, ip, iHighLimit, maxNbAttempts, 3, matches, minMatchLen);
    default :
    case 4 : return ZSTD_BtGetAllMatches(zc, ip, iHighLimit, maxNbAttempts, 4, matches, minMatchLen);
    case 5 : return ZSTD_BtGetAllMatches(zc, ip, iHighLimit, maxNbAttempts, 5, matches, minMatchLen);
    case 7 :
    case 6 : return ZSTD_BtGetAllMatches(zc, ip, iHighLimit, maxNbAttempts, 6, matches, minMatchLen);
    }
}

/** Tree updater, providing best match */
static U32 ZSTD_BtGetAllMatches_extDict (
                        ZSTD_CCtx* zc,
                        const BYTE* const ip, const BYTE* const iLimit,
                        const U32 maxNbAttempts, const U32 mls, ZSTD_match_t* matches, const U32 minMatchLen)
{
    if (ip < zc->base + zc->nextToUpdate) return 0;   /* skipped area */
    ZSTD_updateTree_extDict(zc, ip, iLimit, maxNbAttempts, mls);
    return ZSTD_insertBtAndGetAllMatches(zc, ip, iLimit, maxNbAttempts, mls, 1, matches, minMatchLen);
}


static U32 ZSTD_BtGetAllMatches_selectMLS_extDict (
                        ZSTD_CCtx* zc,   /* Index table will be updated */
                        const BYTE* ip, const BYTE* const iHighLimit,
                        const U32 maxNbAttempts, const U32 matchLengthSearch, ZSTD_match_t* matches, const U32 minMatchLen)
{
    switch(matchLengthSearch)
    {
    case 3 : return ZSTD_BtGetAllMatches_extDict(zc, ip, iHighLimit, maxNbAttempts, 3, matches, minMatchLen);
    default :
    case 4 : return ZSTD_BtGetAllMatches_extDict(zc, ip, iHighLimit, maxNbAttempts, 4, matches, minMatchLen);
    case 5 : return ZSTD_BtGetAllMatches_extDict(zc, ip, iHighLimit, maxNbAttempts, 5, matches, minMatchLen);
    case 7 :
    case 6 : return ZSTD_BtGetAllMatches_extDict(zc, ip, iHighLimit, maxNbAttempts, 6, matches, minMatchLen);
    }
}


/*-*******************************
*  Optimal parser
*********************************/
FORCE_INLINE_TEMPLATE
size_t ZSTD_compressBlock_opt_generic(ZSTD_CCtx* ctx,
                                      const void* src, size_t srcSize, const int ultra)
{
    seqStore_t* seqStorePtr = &(ctx->seqStore);
    optState_t* optStatePtr = &(ctx->optState);
    const BYTE* const istart = (const BYTE*)src;
    const BYTE* ip = istart;
    const BYTE* anchor = istart;
    const BYTE* const iend = istart + srcSize;
    const BYTE* const ilimit = iend - 8;
    const BYTE* const base = ctx->base;
    const BYTE* const prefixStart = base + ctx->dictLimit;

    const U32 maxSearches = 1U << ctx->appliedParams.cParams.searchLog;
    const U32 sufficient_len = ctx->appliedParams.cParams.targetLength;
    const U32 mls = ctx->appliedParams.cParams.searchLength;
    const U32 minMatch = (ctx->appliedParams.cParams.searchLength == 3) ? 3 : 4;

    ZSTD_optimal_t* opt = optStatePtr->priceTable;
    ZSTD_match_t* matches = optStatePtr->matchTable;
    const BYTE* inr;
    U32 offset, rep[ZSTD_REP_NUM];

    /* init */
    ctx->nextToUpdate3 = ctx->nextToUpdate;
    ZSTD_rescaleFreqs(optStatePtr, (const BYTE*)src, srcSize);
    ip += (ip==prefixStart);
    { U32 i; for (i=0; i<ZSTD_REP_NUM; i++) rep[i]=seqStorePtr->rep[i]; }

    /* Match Loop */
    while (ip < ilimit) {
        U32 cur, match_num, last_pos, litlen, price;
        U32 u, mlen, best_mlen, best_off, litLength;
        memset(opt, 0, sizeof(ZSTD_optimal_t));
        last_pos = 0;
        litlen = (U32)(ip - anchor);

        /* check repCode */
        {   U32 i, last_i = ZSTD_REP_CHECK + (ip==anchor);
            for (i=(ip == anchor); i<last_i; i++) {
                const S32 repCur = (i==ZSTD_REP_MOVE_OPT) ? (rep[0] - 1) : rep[i];
                if ( (repCur > 0) && (repCur < (S32)(ip-prefixStart))
                    && (ZSTD_readMINMATCH(ip, minMatch) == ZSTD_readMINMATCH(ip - repCur, minMatch))) {
                    mlen = (U32)ZSTD_count(ip+minMatch, ip+minMatch-repCur, iend) + minMatch;
                    if (mlen > sufficient_len || mlen >= ZSTD_OPT_NUM) {
                        best_mlen = mlen; best_off = i; cur = 0; last_pos = 1;
                        goto _storeSequence;
                    }
                    best_off = i - (ip == anchor);
                    do {
                        price = ZSTD_getPrice(optStatePtr, litlen, anchor, best_off, mlen - MINMATCH, ultra);
                        if (mlen > last_pos || price < opt[mlen].price)
                            SET_PRICE(mlen, mlen, i, litlen, price);   /* note : macro modifies last_pos */
                        mlen--;
                    } while (mlen >= minMatch);
        }   }   }

        match_num = ZSTD_BtGetAllMatches_selectMLS(ctx, ip, iend, maxSearches, mls, matches, minMatch);

        if (!last_pos && !match_num) { ip++; continue; }

        if (match_num && (matches[match_num-1].len > sufficient_len || matches[match_num-1].len >= ZSTD_OPT_NUM)) {
            best_mlen = matches[match_num-1].len;
            best_off = matches[match_num-1].off;
            cur = 0;
            last_pos = 1;
            goto _storeSequence;
        }

        /* set prices using matches at position = 0 */
        best_mlen = (last_pos) ? last_pos : minMatch;
        for (u = 0; u < match_num; u++) {
            mlen = (u>0) ? matches[u-1].len+1 : best_mlen;
            best_mlen = matches[u].len;
            while (mlen <= best_mlen) {
                price = ZSTD_getPrice(optStatePtr, litlen, anchor, matches[u].off-1, mlen - MINMATCH, ultra);
                if (mlen > last_pos || price < opt[mlen].price)
                    SET_PRICE(mlen, mlen, matches[u].off, litlen, price);   /* note : macro modifies last_pos */
                mlen++;
        }   }

        if (last_pos < minMatch) { ip++; continue; }

        /* initialize opt[0] */
        { U32 i ; for (i=0; i<ZSTD_REP_NUM; i++) opt[0].rep[i] = rep[i]; }
        opt[0].mlen = 1;
        opt[0].litlen = litlen;

         /* check further positions */
        for (cur = 1; cur <= last_pos; cur++) {
           inr = ip + cur;

           if (opt[cur-1].mlen == 1) {
                litlen = opt[cur-1].litlen + 1;
                if (cur > litlen) {
                    price = opt[cur - litlen].price + ZSTD_getLiteralPrice(optStatePtr, litlen, inr-litlen);
                } else
                    price = ZSTD_getLiteralPrice(optStatePtr, litlen, anchor);
           } else {
                litlen = 1;
                price = opt[cur - 1].price + ZSTD_getLiteralPrice(optStatePtr, litlen, inr-1);
           }

           if (cur > last_pos || price <= opt[cur].price)
                SET_PRICE(cur, 1, 0, litlen, price);

           if (cur == last_pos) break;

           if (inr > ilimit)  /* last match must start at a minimum distance of 8 from oend */
               continue;

           mlen = opt[cur].mlen;
           if (opt[cur].off > ZSTD_REP_MOVE_OPT) {
                opt[cur].rep[2] = opt[cur-mlen].rep[1];
                opt[cur].rep[1] = opt[cur-mlen].rep[0];
                opt[cur].rep[0] = opt[cur].off - ZSTD_REP_MOVE_OPT;
           } else {
                opt[cur].rep[2] = (opt[cur].off > 1) ? opt[cur-mlen].rep[1] : opt[cur-mlen].rep[2];
                opt[cur].rep[1] = (opt[cur].off > 0) ? opt[cur-mlen].rep[0] : opt[cur-mlen].rep[1];
                /* If opt[cur].off == ZSTD_REP_MOVE_OPT, then mlen != 1.
                 * offset ZSTD_REP_MOVE_OPT is used for the special case
                 * litLength == 0, where offset 0 means something special.
                 * mlen == 1 means the previous byte was stored as a literal,
                 * so they are mutually exclusive.
                 */
                assert(!(opt[cur].off == ZSTD_REP_MOVE_OPT && mlen == 1));
                opt[cur].rep[0] = (opt[cur].off == ZSTD_REP_MOVE_OPT) ? (opt[cur-mlen].rep[0] - 1) : (opt[cur-mlen].rep[opt[cur].off]);
           }

            best_mlen = minMatch;
            {   U32 i, last_i = ZSTD_REP_CHECK + (mlen != 1);
                for (i=(opt[cur].mlen != 1); i<last_i; i++) {  /* check rep */
                    const S32 repCur = (i==ZSTD_REP_MOVE_OPT) ? (opt[cur].rep[0] - 1) : opt[cur].rep[i];
                    if ( (repCur > 0) && (repCur < (S32)(inr-prefixStart))
                       && (ZSTD_readMINMATCH(inr, minMatch) == ZSTD_readMINMATCH(inr - repCur, minMatch))) {
                       mlen = (U32)ZSTD_count(inr+minMatch, inr+minMatch - repCur, iend) + minMatch;

                       if (mlen > sufficient_len || cur + mlen >= ZSTD_OPT_NUM) {
                            best_mlen = mlen; best_off = i; last_pos = cur + 1;
                            goto _storeSequence;
                       }

                       best_off = i - (opt[cur].mlen != 1);
                       if (mlen > best_mlen) best_mlen = mlen;

                       do {
                           if (opt[cur].mlen == 1) {
                                litlen = opt[cur].litlen;
                                if (cur > litlen) {
                                    price = opt[cur - litlen].price + ZSTD_getPrice(optStatePtr, litlen, inr-litlen, best_off, mlen - MINMATCH, ultra);
                                } else
                                    price = ZSTD_getPrice(optStatePtr, litlen, anchor, best_off, mlen - MINMATCH, ultra);
                            } else {
                                litlen = 0;
                                price = opt[cur].price + ZSTD_getPrice(optStatePtr, 0, NULL, best_off, mlen - MINMATCH, ultra);
                            }

                            if (cur + mlen > last_pos || price <= opt[cur + mlen].price)
                                SET_PRICE(cur + mlen, mlen, i, litlen, price);
                            mlen--;
                        } while (mlen >= minMatch);
            }   }   }

            match_num = ZSTD_BtGetAllMatches_selectMLS(ctx, inr, iend, maxSearches, mls, matches, best_mlen);

            if (match_num > 0 && (matches[match_num-1].len > sufficient_len || cur + matches[match_num-1].len >= ZSTD_OPT_NUM)) {
                best_mlen = matches[match_num-1].len;
                best_off = matches[match_num-1].off;
                last_pos = cur + 1;
                goto _storeSequence;
            }

            /* set prices using matches at position = cur */
            for (u = 0; u < match_num; u++) {
                mlen = (u>0) ? matches[u-1].len+1 : best_mlen;
                best_mlen = matches[u].len;

                while (mlen <= best_mlen) {
                    if (opt[cur].mlen == 1) {
                        litlen = opt[cur].litlen;
                        if (cur > litlen)
                            price = opt[cur - litlen].price + ZSTD_getPrice(optStatePtr, litlen, ip+cur-litlen, matches[u].off-1, mlen - MINMATCH, ultra);
                        else
                            price = ZSTD_getPrice(optStatePtr, litlen, anchor, matches[u].off-1, mlen - MINMATCH, ultra);
                    } else {
                        litlen = 0;
                        price = opt[cur].price + ZSTD_getPrice(optStatePtr, 0, NULL, matches[u].off-1, mlen - MINMATCH, ultra);
                    }

                    if (cur + mlen > last_pos || (price < opt[cur + mlen].price))
                        SET_PRICE(cur + mlen, mlen, matches[u].off, litlen, price);

                    mlen++;
        }   }   }

        best_mlen = opt[last_pos].mlen;
        best_off = opt[last_pos].off;
        cur = last_pos - best_mlen;

        /* store sequence */
_storeSequence:   /* cur, last_pos, best_mlen, best_off have to be set */
        opt[0].mlen = 1;

        while (1) {
            mlen = opt[cur].mlen;
            offset = opt[cur].off;
            opt[cur].mlen = best_mlen;
            opt[cur].off = best_off;
            best_mlen = mlen;
            best_off = offset;
            if (mlen > cur) break;
            cur -= mlen;
        }

        for (u = 0; u <= last_pos;) {
            u += opt[u].mlen;
        }

        for (cur=0; cur < last_pos; ) {
            mlen = opt[cur].mlen;
            if (mlen == 1) { ip++; cur++; continue; }
            offset = opt[cur].off;
            cur += mlen;
            litLength = (U32)(ip - anchor);

            if (offset > ZSTD_REP_MOVE_OPT) {
                rep[2] = rep[1];
                rep[1] = rep[0];
                rep[0] = offset - ZSTD_REP_MOVE_OPT;
                offset--;
            } else {
                if (offset != 0) {
                    best_off = (offset==ZSTD_REP_MOVE_OPT) ? (rep[0] - 1) : (rep[offset]);
                    if (offset != 1) rep[2] = rep[1];
                    rep[1] = rep[0];
                    rep[0] = best_off;
                }
                if (litLength==0) offset--;
            }

            ZSTD_updatePrice(optStatePtr, litLength, anchor, offset, mlen-MINMATCH);
            ZSTD_storeSeq(seqStorePtr, litLength, anchor, offset, mlen-MINMATCH);
            anchor = ip = ip + mlen;
    }    }   /* for (cur=0; cur < last_pos; ) */

    /* Save reps for next block */
    { int i; for (i=0; i<ZSTD_REP_NUM; i++) seqStorePtr->repToConfirm[i] = rep[i]; }

    /* Return the last literals size */
    return iend - anchor;
}


size_t ZSTD_compressBlock_btopt(ZSTD_CCtx* ctx, const void* src, size_t srcSize)
{
    return ZSTD_compressBlock_opt_generic(ctx, src, srcSize, 0);
}

size_t ZSTD_compressBlock_btultra(ZSTD_CCtx* ctx, const void* src, size_t srcSize)
{
    return ZSTD_compressBlock_opt_generic(ctx, src, srcSize, 1);
}


FORCE_INLINE_TEMPLATE
size_t ZSTD_compressBlock_opt_extDict_generic(ZSTD_CCtx* ctx,
                                     const void* src, size_t srcSize, const int ultra)
{
    seqStore_t* seqStorePtr = &(ctx->seqStore);
    optState_t* optStatePtr = &(ctx->optState);
    const BYTE* const istart = (const BYTE*)src;
    const BYTE* ip = istart;
    const BYTE* anchor = istart;
    const BYTE* const iend = istart + srcSize;
    const BYTE* const ilimit = iend - 8;
    const BYTE* const base = ctx->base;
    const U32 lowestIndex = ctx->lowLimit;
    const U32 dictLimit = ctx->dictLimit;
    const BYTE* const prefixStart = base + dictLimit;
    const BYTE* const dictBase = ctx->dictBase;
    const BYTE* const dictEnd  = dictBase + dictLimit;

    const U32 maxSearches = 1U << ctx->appliedParams.cParams.searchLog;
    const U32 sufficient_len = ctx->appliedParams.cParams.targetLength;
    const U32 mls = ctx->appliedParams.cParams.searchLength;
    const U32 minMatch = (ctx->appliedParams.cParams.searchLength == 3) ? 3 : 4;

    ZSTD_optimal_t* opt = optStatePtr->priceTable;
    ZSTD_match_t* matches = optStatePtr->matchTable;
    const BYTE* inr;

    /* init */
    U32 offset, rep[ZSTD_REP_NUM];
    { U32 i; for (i=0; i<ZSTD_REP_NUM; i++) rep[i]=seqStorePtr->rep[i]; }

    ctx->nextToUpdate3 = ctx->nextToUpdate;
    ZSTD_rescaleFreqs(optStatePtr, (const BYTE*)src, srcSize);
    ip += (ip==prefixStart);

    /* Match Loop */
    while (ip < ilimit) {
        U32 cur, match_num, last_pos, litlen, price;
        U32 u, mlen, best_mlen, best_off, litLength;
        U32 current = (U32)(ip-base);
        memset(opt, 0, sizeof(ZSTD_optimal_t));
        last_pos = 0;
        opt[0].litlen = (U32)(ip - anchor);

        /* check repCode */
        {   U32 i, last_i = ZSTD_REP_CHECK + (ip==anchor);
            for (i = (ip==anchor); i<last_i; i++) {
                const S32 repCur = (i==ZSTD_REP_MOVE_OPT) ? (rep[0] - 1) : rep[i];
                const U32 repIndex = (U32)(current - repCur);
                const BYTE* const repBase = repIndex < dictLimit ? dictBase : base;
                const BYTE* const repMatch = repBase + repIndex;
                if ( (repCur > 0 && repCur <= (S32)current)
                   && (((U32)((dictLimit-1) - repIndex) >= 3) & (repIndex>lowestIndex))  /* intentional overflow */
                   && (ZSTD_readMINMATCH(ip, minMatch) == ZSTD_readMINMATCH(repMatch, minMatch)) ) {
                    /* repcode detected we should take it */
                    const BYTE* const repEnd = repIndex < dictLimit ? dictEnd : iend;
                    mlen = (U32)ZSTD_count_2segments(ip+minMatch, repMatch+minMatch, iend, repEnd, prefixStart) + minMatch;

                    if (mlen > sufficient_len || mlen >= ZSTD_OPT_NUM) {
                        best_mlen = mlen; best_off = i; cur = 0; last_pos = 1;
                        goto _storeSequence;
                    }

                    best_off = i - (ip==anchor);
                    litlen = opt[0].litlen;
                    do {
                        price = ZSTD_getPrice(optStatePtr, litlen, anchor, best_off, mlen - MINMATCH, ultra);
                        if (mlen > last_pos || price < opt[mlen].price)
                            SET_PRICE(mlen, mlen, i, litlen, price);   /* note : macro modifies last_pos */
                        mlen--;
                    } while (mlen >= minMatch);
        }   }   }

        match_num = ZSTD_BtGetAllMatches_selectMLS_extDict(ctx, ip, iend, maxSearches, mls, matches, minMatch);  /* first search (depth 0) */

        if (!last_pos && !match_num) { ip++; continue; }

        { U32 i; for (i=0; i<ZSTD_REP_NUM; i++) opt[0].rep[i] = rep[i]; }
        opt[0].mlen = 1;

        if (match_num && (matches[match_num-1].len > sufficient_len || matches[match_num-1].len >= ZSTD_OPT_NUM)) {
            best_mlen = matches[match_num-1].len;
            best_off = matches[match_num-1].off;
            cur = 0;
            last_pos = 1;
            goto _storeSequence;
        }

        best_mlen = (last_pos) ? last_pos : minMatch;

        /* set prices using matches at position = 0 */
        for (u = 0; u < match_num; u++) {
            mlen = (u>0) ? matches[u-1].len+1 : best_mlen;
            best_mlen = matches[u].len;
            litlen = opt[0].litlen;
            while (mlen <= best_mlen) {
                price = ZSTD_getPrice(optStatePtr, litlen, anchor, matches[u].off-1, mlen - MINMATCH, ultra);
                if (mlen > last_pos || price < opt[mlen].price)
                    SET_PRICE(mlen, mlen, matches[u].off, litlen, price);
                mlen++;
        }   }

        if (last_pos < minMatch) {
            ip++; continue;
        }

        /* check further positions */
        for (cur = 1; cur <= last_pos; cur++) {
            inr = ip + cur;

            if (opt[cur-1].mlen == 1) {
                litlen = opt[cur-1].litlen + 1;
                if (cur > litlen) {
                    price = opt[cur - litlen].price + ZSTD_getLiteralPrice(optStatePtr, litlen, inr-litlen);
                } else
                    price = ZSTD_getLiteralPrice(optStatePtr, litlen, anchor);
            } else {
                litlen = 1;
                price = opt[cur - 1].price + ZSTD_getLiteralPrice(optStatePtr, litlen, inr-1);
            }

            if (cur > last_pos || price <= opt[cur].price)
                SET_PRICE(cur, 1, 0, litlen, price);

            if (cur == last_pos) break;

            if (inr > ilimit)  /* last match must start at a minimum distance of 8 from oend */
                continue;

            mlen = opt[cur].mlen;
            if (opt[cur].off > ZSTD_REP_MOVE_OPT) {
                opt[cur].rep[2] = opt[cur-mlen].rep[1];
                opt[cur].rep[1] = opt[cur-mlen].rep[0];
                opt[cur].rep[0] = opt[cur].off - ZSTD_REP_MOVE_OPT;
            } else {
                opt[cur].rep[2] = (opt[cur].off > 1) ? opt[cur-mlen].rep[1] : opt[cur-mlen].rep[2];
                opt[cur].rep[1] = (opt[cur].off > 0) ? opt[cur-mlen].rep[0] : opt[cur-mlen].rep[1];
                assert(!(opt[cur].off == ZSTD_REP_MOVE_OPT && mlen == 1));
                opt[cur].rep[0] = (opt[cur].off == ZSTD_REP_MOVE_OPT) ? (opt[cur-mlen].rep[0] - 1) : (opt[cur-mlen].rep[opt[cur].off]);
            }

            best_mlen = minMatch;
            {   U32 i, last_i = ZSTD_REP_CHECK + (mlen != 1);
                for (i = (mlen != 1); i<last_i; i++) {
                    const S32 repCur = (i==ZSTD_REP_MOVE_OPT) ? (opt[cur].rep[0] - 1) : opt[cur].rep[i];
                    const U32 repIndex = (U32)(current+cur - repCur);
                    const BYTE* const repBase = repIndex < dictLimit ? dictBase : base;
                    const BYTE* const repMatch = repBase + repIndex;
                    if ( (repCur > 0 && repCur <= (S32)(current+cur))
                      && (((U32)((dictLimit-1) - repIndex) >= 3) & (repIndex>lowestIndex))  /* intentional overflow */
                      && (ZSTD_readMINMATCH(inr, minMatch) == ZSTD_readMINMATCH(repMatch, minMatch)) ) {
                        /* repcode detected */
                        const BYTE* const repEnd = repIndex < dictLimit ? dictEnd : iend;
                        mlen = (U32)ZSTD_count_2segments(inr+minMatch, repMatch+minMatch, iend, repEnd, prefixStart) + minMatch;

                        if (mlen > sufficient_len || cur + mlen >= ZSTD_OPT_NUM) {
                            best_mlen = mlen; best_off = i; last_pos = cur + 1;
                            goto _storeSequence;
                        }

                        best_off = i - (opt[cur].mlen != 1);
                        if (mlen > best_mlen) best_mlen = mlen;

                        do {
                            if (opt[cur].mlen == 1) {
                                litlen = opt[cur].litlen;
                                if (cur > litlen) {
                                    price = opt[cur - litlen].price + ZSTD_getPrice(optStatePtr, litlen, inr-litlen, best_off, mlen - MINMATCH, ultra);
                                } else
                                    price = ZSTD_getPrice(optStatePtr, litlen, anchor, best_off, mlen - MINMATCH, ultra);
                            } else {
                                litlen = 0;
                                price = opt[cur].price + ZSTD_getPrice(optStatePtr, 0, NULL, best_off, mlen - MINMATCH, ultra);
                            }

                            if (cur + mlen > last_pos || price <= opt[cur + mlen].price)
                                SET_PRICE(cur + mlen, mlen, i, litlen, price);
                            mlen--;
                        } while (mlen >= minMatch);
            }   }   }

            match_num = ZSTD_BtGetAllMatches_selectMLS_extDict(ctx, inr, iend, maxSearches, mls, matches, minMatch);

            if (match_num > 0 && (matches[match_num-1].len > sufficient_len || cur + matches[match_num-1].len >= ZSTD_OPT_NUM)) {
                best_mlen = matches[match_num-1].len;
                best_off = matches[match_num-1].off;
                last_pos = cur + 1;
                goto _storeSequence;
            }

            /* set prices using matches at position = cur */
            for (u = 0; u < match_num; u++) {
                mlen = (u>0) ? matches[u-1].len+1 : best_mlen;
                best_mlen = matches[u].len;

                while (mlen <= best_mlen) {
                    if (opt[cur].mlen == 1) {
                        litlen = opt[cur].litlen;
                        if (cur > litlen)
                            price = opt[cur - litlen].price + ZSTD_getPrice(optStatePtr, litlen, ip+cur-litlen, matches[u].off-1, mlen - MINMATCH, ultra);
                        else
                            price = ZSTD_getPrice(optStatePtr, litlen, anchor, matches[u].off-1, mlen - MINMATCH, ultra);
                    } else {
                        litlen = 0;
                        price = opt[cur].price + ZSTD_getPrice(optStatePtr, 0, NULL, matches[u].off-1, mlen - MINMATCH, ultra);
                    }

                    if (cur + mlen > last_pos || (price < opt[cur + mlen].price))
                        SET_PRICE(cur + mlen, mlen, matches[u].off, litlen, price);

                    mlen++;
        }   }   }   /* for (cur = 1; cur <= last_pos; cur++) */

        best_mlen = opt[last_pos].mlen;
        best_off = opt[last_pos].off;
        cur = last_pos - best_mlen;

        /* store sequence */
_storeSequence:   /* cur, last_pos, best_mlen, best_off have to be set */
        opt[0].mlen = 1;

        while (1) {
            mlen = opt[cur].mlen;
            offset = opt[cur].off;
            opt[cur].mlen = best_mlen;
            opt[cur].off = best_off;
            best_mlen = mlen;
            best_off = offset;
            if (mlen > cur) break;
            cur -= mlen;
        }

        for (u = 0; u <= last_pos; ) {
            u += opt[u].mlen;
        }

        for (cur=0; cur < last_pos; ) {
            mlen = opt[cur].mlen;
            if (mlen == 1) { ip++; cur++; continue; }
            offset = opt[cur].off;
            cur += mlen;
            litLength = (U32)(ip - anchor);

            if (offset > ZSTD_REP_MOVE_OPT) {
                rep[2] = rep[1];
                rep[1] = rep[0];
                rep[0] = offset - ZSTD_REP_MOVE_OPT;
                offset--;
            } else {
                if (offset != 0) {
                    best_off = (offset==ZSTD_REP_MOVE_OPT) ? (rep[0] - 1) : (rep[offset]);
                    if (offset != 1) rep[2] = rep[1];
                    rep[1] = rep[0];
                    rep[0] = best_off;
                }

                if (litLength==0) offset--;
            }

            ZSTD_updatePrice(optStatePtr, litLength, anchor, offset, mlen-MINMATCH);
            ZSTD_storeSeq(seqStorePtr, litLength, anchor, offset, mlen-MINMATCH);
            anchor = ip = ip + mlen;
    }    }   /* for (cur=0; cur < last_pos; ) */

    /* Save reps for next block */
    { int i; for (i=0; i<ZSTD_REP_NUM; i++) seqStorePtr->repToConfirm[i] = rep[i]; }

    /* Return the last literals size */
    return iend - anchor;
}


size_t ZSTD_compressBlock_btopt_extDict(ZSTD_CCtx* ctx, const void* src, size_t srcSize)
{
    return ZSTD_compressBlock_opt_extDict_generic(ctx, src, srcSize, 0);
}

size_t ZSTD_compressBlock_btultra_extDict(ZSTD_CCtx* ctx, const void* src, size_t srcSize)
{
    return ZSTD_compressBlock_opt_extDict_generic(ctx, src, srcSize, 1);
}
