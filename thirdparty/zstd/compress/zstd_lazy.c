/*
 * Copyright (c) 2016-present, Yann Collet, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under both the BSD-style license (found in the
 * LICENSE file in the root directory of this source tree) and the GPLv2 (found
 * in the COPYING file in the root directory of this source tree).
 * You may select, at your option, one of the above-listed licenses.
 */

#include "zstd_compress_internal.h"
#include "zstd_lazy.h"


/*-*************************************
*  Binary Tree search
***************************************/

void ZSTD_updateDUBT(
                ZSTD_matchState_t* ms, ZSTD_compressionParameters const* cParams,
                const BYTE* ip, const BYTE* iend,
                U32 mls)
{
    U32* const hashTable = ms->hashTable;
    U32  const hashLog = cParams->hashLog;

    U32* const bt = ms->chainTable;
    U32  const btLog  = cParams->chainLog - 1;
    U32  const btMask = (1 << btLog) - 1;

    const BYTE* const base = ms->window.base;
    U32 const target = (U32)(ip - base);
    U32 idx = ms->nextToUpdate;

    if (idx != target)
        DEBUGLOG(7, "ZSTD_updateDUBT, from %u to %u (dictLimit:%u)",
                    idx, target, ms->window.dictLimit);
    assert(ip + 8 <= iend);   /* condition for ZSTD_hashPtr */
    (void)iend;

    assert(idx >= ms->window.dictLimit);   /* condition for valid base+idx */
    for ( ; idx < target ; idx++) {
        size_t const h  = ZSTD_hashPtr(base + idx, hashLog, mls);   /* assumption : ip + 8 <= iend */
        U32    const matchIndex = hashTable[h];

        U32*   const nextCandidatePtr = bt + 2*(idx&btMask);
        U32*   const sortMarkPtr  = nextCandidatePtr + 1;

        DEBUGLOG(8, "ZSTD_updateDUBT: insert %u", idx);
        hashTable[h] = idx;   /* Update Hash Table */
        *nextCandidatePtr = matchIndex;   /* update BT like a chain */
        *sortMarkPtr = ZSTD_DUBT_UNSORTED_MARK;
    }
    ms->nextToUpdate = target;
}


/** ZSTD_insertDUBT1() :
 *  sort one already inserted but unsorted position
 *  assumption : current >= btlow == (current - btmask)
 *  doesn't fail */
static void ZSTD_insertDUBT1(
                 ZSTD_matchState_t* ms, ZSTD_compressionParameters const* cParams,
                 U32 current, const BYTE* inputEnd,
                 U32 nbCompares, U32 btLow, int extDict)
{
    U32*   const bt = ms->chainTable;
    U32    const btLog  = cParams->chainLog - 1;
    U32    const btMask = (1 << btLog) - 1;
    size_t commonLengthSmaller=0, commonLengthLarger=0;
    const BYTE* const base = ms->window.base;
    const BYTE* const dictBase = ms->window.dictBase;
    const U32 dictLimit = ms->window.dictLimit;
    const BYTE* const ip = (current>=dictLimit) ? base + current : dictBase + current;
    const BYTE* const iend = (current>=dictLimit) ? inputEnd : dictBase + dictLimit;
    const BYTE* const dictEnd = dictBase + dictLimit;
    const BYTE* const prefixStart = base + dictLimit;
    const BYTE* match;
    U32* smallerPtr = bt + 2*(current&btMask);
    U32* largerPtr  = smallerPtr + 1;
    U32 matchIndex = *smallerPtr;
    U32 dummy32;   /* to be nullified at the end */
    U32 const windowLow = ms->window.lowLimit;

    DEBUGLOG(8, "ZSTD_insertDUBT1(%u) (dictLimit=%u, lowLimit=%u)",
                current, dictLimit, windowLow);
    assert(current >= btLow);
    assert(ip < iend);   /* condition for ZSTD_count */

    while (nbCompares-- && (matchIndex > windowLow)) {
        U32* const nextPtr = bt + 2*(matchIndex & btMask);
        size_t matchLength = MIN(commonLengthSmaller, commonLengthLarger);   /* guaranteed minimum nb of common bytes */
        assert(matchIndex < current);

        if ( (!extDict)
          || (matchIndex+matchLength >= dictLimit)  /* both in current segment*/
          || (current < dictLimit) /* both in extDict */) {
            const BYTE* const mBase = !extDict || ((matchIndex+matchLength) >= dictLimit) ? base : dictBase;
            assert( (matchIndex+matchLength >= dictLimit)   /* might be wrong if extDict is incorrectly set to 0 */
                 || (current < dictLimit) );
            match = mBase + matchIndex;
            matchLength += ZSTD_count(ip+matchLength, match+matchLength, iend);
        } else {
            match = dictBase + matchIndex;
            matchLength += ZSTD_count_2segments(ip+matchLength, match+matchLength, iend, dictEnd, prefixStart);
            if (matchIndex+matchLength >= dictLimit)
                match = base + matchIndex;   /* to prepare for next usage of match[matchLength] */
        }

        DEBUGLOG(8, "ZSTD_insertDUBT1: comparing %u with %u : found %u common bytes ",
                    current, matchIndex, (U32)matchLength);

        if (ip+matchLength == iend) {   /* equal : no way to know if inf or sup */
            break;   /* drop , to guarantee consistency ; miss a bit of compression, but other solutions can corrupt tree */
        }

        if (match[matchLength] < ip[matchLength]) {  /* necessarily within buffer */
            /* match is smaller than current */
            *smallerPtr = matchIndex;             /* update smaller idx */
            commonLengthSmaller = matchLength;    /* all smaller will now have at least this guaranteed common length */
            if (matchIndex <= btLow) { smallerPtr=&dummy32; break; }   /* beyond tree size, stop searching */
            DEBUGLOG(8, "ZSTD_insertDUBT1: %u (>btLow=%u) is smaller : next => %u",
                        matchIndex, btLow, nextPtr[1]);
            smallerPtr = nextPtr+1;               /* new "candidate" => larger than match, which was smaller than target */
            matchIndex = nextPtr[1];              /* new matchIndex, larger than previous and closer to current */
        } else {
            /* match is larger than current */
            *largerPtr = matchIndex;
            commonLengthLarger = matchLength;
            if (matchIndex <= btLow) { largerPtr=&dummy32; break; }   /* beyond tree size, stop searching */
            DEBUGLOG(8, "ZSTD_insertDUBT1: %u (>btLow=%u) is larger => %u",
                        matchIndex, btLow, nextPtr[0]);
            largerPtr = nextPtr;
            matchIndex = nextPtr[0];
    }   }

    *smallerPtr = *largerPtr = 0;
}


static size_t ZSTD_DUBT_findBestMatch (
                            ZSTD_matchState_t* ms, ZSTD_compressionParameters const* cParams,
                            const BYTE* const ip, const BYTE* const iend,
                            size_t* offsetPtr,
                            U32 const mls,
                            U32 const extDict)
{
    U32*   const hashTable = ms->hashTable;
    U32    const hashLog = cParams->hashLog;
    size_t const h  = ZSTD_hashPtr(ip, hashLog, mls);
    U32          matchIndex  = hashTable[h];

    const BYTE* const base = ms->window.base;
    U32    const current = (U32)(ip-base);
    U32    const windowLow = ms->window.lowLimit;

    U32*   const bt = ms->chainTable;
    U32    const btLog  = cParams->chainLog - 1;
    U32    const btMask = (1 << btLog) - 1;
    U32    const btLow = (btMask >= current) ? 0 : current - btMask;
    U32    const unsortLimit = MAX(btLow, windowLow);

    U32*         nextCandidate = bt + 2*(matchIndex&btMask);
    U32*         unsortedMark = bt + 2*(matchIndex&btMask) + 1;
    U32          nbCompares = 1U << cParams->searchLog;
    U32          nbCandidates = nbCompares;
    U32          previousCandidate = 0;

    DEBUGLOG(7, "ZSTD_DUBT_findBestMatch (%u) ", current);
    assert(ip <= iend-8);   /* required for h calculation */

    /* reach end of unsorted candidates list */
    while ( (matchIndex > unsortLimit)
         && (*unsortedMark == ZSTD_DUBT_UNSORTED_MARK)
         && (nbCandidates > 1) ) {
        DEBUGLOG(8, "ZSTD_DUBT_findBestMatch: candidate %u is unsorted",
                    matchIndex);
        *unsortedMark = previousCandidate;
        previousCandidate = matchIndex;
        matchIndex = *nextCandidate;
        nextCandidate = bt + 2*(matchIndex&btMask);
        unsortedMark = bt + 2*(matchIndex&btMask) + 1;
        nbCandidates --;
    }

    if ( (matchIndex > unsortLimit)
      && (*unsortedMark==ZSTD_DUBT_UNSORTED_MARK) ) {
        DEBUGLOG(7, "ZSTD_DUBT_findBestMatch: nullify last unsorted candidate %u",
                    matchIndex);
        *nextCandidate = *unsortedMark = 0;   /* nullify next candidate if it's still unsorted (note : simplification, detrimental to compression ratio, beneficial for speed) */
    }

    /* batch sort stacked candidates */
    matchIndex = previousCandidate;
    while (matchIndex) {  /* will end on matchIndex == 0 */
        U32* const nextCandidateIdxPtr = bt + 2*(matchIndex&btMask) + 1;
        U32 const nextCandidateIdx = *nextCandidateIdxPtr;
        ZSTD_insertDUBT1(ms, cParams, matchIndex, iend,
                         nbCandidates, unsortLimit, extDict);
        matchIndex = nextCandidateIdx;
        nbCandidates++;
    }

    /* find longest match */
    {   size_t commonLengthSmaller=0, commonLengthLarger=0;
        const BYTE* const dictBase = ms->window.dictBase;
        const U32 dictLimit = ms->window.dictLimit;
        const BYTE* const dictEnd = dictBase + dictLimit;
        const BYTE* const prefixStart = base + dictLimit;
        U32* smallerPtr = bt + 2*(current&btMask);
        U32* largerPtr  = bt + 2*(current&btMask) + 1;
        U32 matchEndIdx = current+8+1;
        U32 dummy32;   /* to be nullified at the end */
        size_t bestLength = 0;

        matchIndex  = hashTable[h];
        hashTable[h] = current;   /* Update Hash Table */

        while (nbCompares-- && (matchIndex > windowLow)) {
            U32* const nextPtr = bt + 2*(matchIndex & btMask);
            size_t matchLength = MIN(commonLengthSmaller, commonLengthLarger);   /* guaranteed minimum nb of common bytes */
            const BYTE* match;

            if ((!extDict) || (matchIndex+matchLength >= dictLimit)) {
                match = base + matchIndex;
                matchLength += ZSTD_count(ip+matchLength, match+matchLength, iend);
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
                if (ip+matchLength == iend) {   /* equal : no way to know if inf or sup */
                    break;   /* drop, to guarantee consistency (miss a little bit of compression) */
                }
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

        assert(matchEndIdx > current+8); /* ensure nextToUpdate is increased */
        ms->nextToUpdate = matchEndIdx - 8;   /* skip repetitive patterns */
        if (bestLength >= MINMATCH) {
            U32 const mIndex = current - ((U32)*offsetPtr - ZSTD_REP_MOVE); (void)mIndex;
            DEBUGLOG(8, "ZSTD_DUBT_findBestMatch(%u) : found match of length %u and offsetCode %u (pos %u)",
                        current, (U32)bestLength, (U32)*offsetPtr, mIndex);
        }
        return bestLength;
    }
}


/** ZSTD_BtFindBestMatch() : Tree updater, providing best match */
static size_t ZSTD_BtFindBestMatch (
                        ZSTD_matchState_t* ms, ZSTD_compressionParameters const* cParams,
                        const BYTE* const ip, const BYTE* const iLimit,
                        size_t* offsetPtr,
                        const U32 mls /* template */)
{
    DEBUGLOG(7, "ZSTD_BtFindBestMatch");
    if (ip < ms->window.base + ms->nextToUpdate) return 0;   /* skipped area */
    ZSTD_updateDUBT(ms, cParams, ip, iLimit, mls);
    return ZSTD_DUBT_findBestMatch(ms, cParams, ip, iLimit, offsetPtr, mls, 0);
}


static size_t ZSTD_BtFindBestMatch_selectMLS (
                        ZSTD_matchState_t* ms, ZSTD_compressionParameters const* cParams,
                        const BYTE* ip, const BYTE* const iLimit,
                        size_t* offsetPtr)
{
    switch(cParams->searchLength)
    {
    default : /* includes case 3 */
    case 4 : return ZSTD_BtFindBestMatch(ms, cParams, ip, iLimit, offsetPtr, 4);
    case 5 : return ZSTD_BtFindBestMatch(ms, cParams, ip, iLimit, offsetPtr, 5);
    case 7 :
    case 6 : return ZSTD_BtFindBestMatch(ms, cParams, ip, iLimit, offsetPtr, 6);
    }
}


/** Tree updater, providing best match */
static size_t ZSTD_BtFindBestMatch_extDict (
                        ZSTD_matchState_t* ms, ZSTD_compressionParameters const* cParams,
                        const BYTE* const ip, const BYTE* const iLimit,
                        size_t* offsetPtr,
                        const U32 mls)
{
    DEBUGLOG(7, "ZSTD_BtFindBestMatch_extDict");
    if (ip < ms->window.base + ms->nextToUpdate) return 0;   /* skipped area */
    ZSTD_updateDUBT(ms, cParams, ip, iLimit, mls);
    return ZSTD_DUBT_findBestMatch(ms, cParams, ip, iLimit, offsetPtr, mls, 1);
}


static size_t ZSTD_BtFindBestMatch_selectMLS_extDict (
                        ZSTD_matchState_t* ms, ZSTD_compressionParameters const* cParams,
                        const BYTE* ip, const BYTE* const iLimit,
                        size_t* offsetPtr)
{
    switch(cParams->searchLength)
    {
    default : /* includes case 3 */
    case 4 : return ZSTD_BtFindBestMatch_extDict(ms, cParams, ip, iLimit, offsetPtr, 4);
    case 5 : return ZSTD_BtFindBestMatch_extDict(ms, cParams, ip, iLimit, offsetPtr, 5);
    case 7 :
    case 6 : return ZSTD_BtFindBestMatch_extDict(ms, cParams, ip, iLimit, offsetPtr, 6);
    }
}



/* *********************************
*  Hash Chain
***********************************/
#define NEXT_IN_CHAIN(d, mask)   chainTable[(d) & mask]

/* Update chains up to ip (excluded)
   Assumption : always within prefix (i.e. not within extDict) */
static U32 ZSTD_insertAndFindFirstIndex_internal(
                        ZSTD_matchState_t* ms, ZSTD_compressionParameters const* cParams,
                        const BYTE* ip, U32 const mls)
{
    U32* const hashTable  = ms->hashTable;
    const U32 hashLog = cParams->hashLog;
    U32* const chainTable = ms->chainTable;
    const U32 chainMask = (1 << cParams->chainLog) - 1;
    const BYTE* const base = ms->window.base;
    const U32 target = (U32)(ip - base);
    U32 idx = ms->nextToUpdate;

    while(idx < target) { /* catch up */
        size_t const h = ZSTD_hashPtr(base+idx, hashLog, mls);
        NEXT_IN_CHAIN(idx, chainMask) = hashTable[h];
        hashTable[h] = idx;
        idx++;
    }

    ms->nextToUpdate = target;
    return hashTable[ZSTD_hashPtr(ip, hashLog, mls)];
}

U32 ZSTD_insertAndFindFirstIndex(
                        ZSTD_matchState_t* ms, ZSTD_compressionParameters const* cParams,
                        const BYTE* ip)
{
    return ZSTD_insertAndFindFirstIndex_internal(ms, cParams, ip, cParams->searchLength);
}


/* inlining is important to hardwire a hot branch (template emulation) */
FORCE_INLINE_TEMPLATE
size_t ZSTD_HcFindBestMatch_generic (
                        ZSTD_matchState_t* ms, ZSTD_compressionParameters const* cParams,
                        const BYTE* const ip, const BYTE* const iLimit,
                        size_t* offsetPtr,
                        const U32 mls, const U32 extDict)
{
    U32* const chainTable = ms->chainTable;
    const U32 chainSize = (1 << cParams->chainLog);
    const U32 chainMask = chainSize-1;
    const BYTE* const base = ms->window.base;
    const BYTE* const dictBase = ms->window.dictBase;
    const U32 dictLimit = ms->window.dictLimit;
    const BYTE* const prefixStart = base + dictLimit;
    const BYTE* const dictEnd = dictBase + dictLimit;
    const U32 lowLimit = ms->window.lowLimit;
    const U32 current = (U32)(ip-base);
    const U32 minChain = current > chainSize ? current - chainSize : 0;
    U32 nbAttempts = 1U << cParams->searchLog;
    size_t ml=4-1;

    /* HC4 match finder */
    U32 matchIndex = ZSTD_insertAndFindFirstIndex_internal(ms, cParams, ip, mls);

    for ( ; (matchIndex>lowLimit) & (nbAttempts>0) ; nbAttempts--) {
        size_t currentMl=0;
        if ((!extDict) || matchIndex >= dictLimit) {
            const BYTE* const match = base + matchIndex;
            if (match[ml] == ip[ml])   /* potentially better */
                currentMl = ZSTD_count(ip, match, iLimit);
        } else {
            const BYTE* const match = dictBase + matchIndex;
            assert(match+4 <= dictEnd);
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


FORCE_INLINE_TEMPLATE size_t ZSTD_HcFindBestMatch_selectMLS (
                        ZSTD_matchState_t* ms, ZSTD_compressionParameters const* cParams,
                        const BYTE* ip, const BYTE* const iLimit,
                        size_t* offsetPtr)
{
    switch(cParams->searchLength)
    {
    default : /* includes case 3 */
    case 4 : return ZSTD_HcFindBestMatch_generic(ms, cParams, ip, iLimit, offsetPtr, 4, 0);
    case 5 : return ZSTD_HcFindBestMatch_generic(ms, cParams, ip, iLimit, offsetPtr, 5, 0);
    case 7 :
    case 6 : return ZSTD_HcFindBestMatch_generic(ms, cParams, ip, iLimit, offsetPtr, 6, 0);
    }
}


FORCE_INLINE_TEMPLATE size_t ZSTD_HcFindBestMatch_extDict_selectMLS (
                        ZSTD_matchState_t* ms, ZSTD_compressionParameters const* cParams,
                        const BYTE* ip, const BYTE* const iLimit,
                        size_t* const offsetPtr)
{
    switch(cParams->searchLength)
    {
    default : /* includes case 3 */
    case 4 : return ZSTD_HcFindBestMatch_generic(ms, cParams, ip, iLimit, offsetPtr, 4, 1);
    case 5 : return ZSTD_HcFindBestMatch_generic(ms, cParams, ip, iLimit, offsetPtr, 5, 1);
    case 7 :
    case 6 : return ZSTD_HcFindBestMatch_generic(ms, cParams, ip, iLimit, offsetPtr, 6, 1);
    }
}


/* *******************************
*  Common parser - lazy strategy
*********************************/
FORCE_INLINE_TEMPLATE
size_t ZSTD_compressBlock_lazy_generic(
                        ZSTD_matchState_t* ms, seqStore_t* seqStore,
                        U32 rep[ZSTD_REP_NUM],
                        ZSTD_compressionParameters const* cParams,
                        const void* src, size_t srcSize,
                        const U32 searchMethod, const U32 depth)
{
    const BYTE* const istart = (const BYTE*)src;
    const BYTE* ip = istart;
    const BYTE* anchor = istart;
    const BYTE* const iend = istart + srcSize;
    const BYTE* const ilimit = iend - 8;
    const BYTE* const base = ms->window.base + ms->window.dictLimit;

    typedef size_t (*searchMax_f)(
                        ZSTD_matchState_t* ms, ZSTD_compressionParameters const* cParams,
                        const BYTE* ip, const BYTE* iLimit, size_t* offsetPtr);
    searchMax_f const searchMax = searchMethod ? ZSTD_BtFindBestMatch_selectMLS : ZSTD_HcFindBestMatch_selectMLS;
    U32 offset_1 = rep[0], offset_2 = rep[1], savedOffset=0;

    /* init */
    ip += (ip==base);
    ms->nextToUpdate3 = ms->nextToUpdate;
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
            size_t const ml2 = searchMax(ms, cParams, ip, iend, &offsetFound);
            if (ml2 > matchLength)
                matchLength = ml2, start = ip, offset=offsetFound;
        }

        if (matchLength < 4) {
            ip += ((ip-anchor) >> kSearchStrength) + 1;   /* jump faster over incompressible sections */
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
                size_t const ml2 = searchMax(ms, cParams, ip, iend, &offset2);
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
                    size_t const ml2 = searchMax(ms, cParams, ip, iend, &offset2);
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
            while ( ((start > anchor) & (start - (offset-ZSTD_REP_MOVE) > base))
                 && (start[-1] == (start-(offset-ZSTD_REP_MOVE))[-1]) )  /* only search for offset within prefix */
                { start--; matchLength++; }
            offset_2 = offset_1; offset_1 = (U32)(offset - ZSTD_REP_MOVE);
        }
        /* store sequence */
_storeSequence:
        {   size_t const litLength = start - anchor;
            ZSTD_storeSeq(seqStore, litLength, anchor, (U32)offset, matchLength-MINMATCH);
            anchor = ip = start + matchLength;
        }

        /* check immediate repcode */
        while ( ((ip <= ilimit) & (offset_2>0))
             && (MEM_read32(ip) == MEM_read32(ip - offset_2)) ) {
            /* store sequence */
            matchLength = ZSTD_count(ip+4, ip+4-offset_2, iend) + 4;
            offset = offset_2; offset_2 = offset_1; offset_1 = (U32)offset; /* swap repcodes */
            ZSTD_storeSeq(seqStore, 0, anchor, 0, matchLength-MINMATCH);
            ip += matchLength;
            anchor = ip;
            continue;   /* faster when present ... (?) */
    }   }

    /* Save reps for next block */
    rep[0] = offset_1 ? offset_1 : savedOffset;
    rep[1] = offset_2 ? offset_2 : savedOffset;

    /* Return the last literals size */
    return iend - anchor;
}


size_t ZSTD_compressBlock_btlazy2(
        ZSTD_matchState_t* ms, seqStore_t* seqStore, U32 rep[ZSTD_REP_NUM],
        ZSTD_compressionParameters const* cParams, void const* src, size_t srcSize)
{
    return ZSTD_compressBlock_lazy_generic(ms, seqStore, rep, cParams, src, srcSize, 1, 2);
}

size_t ZSTD_compressBlock_lazy2(
        ZSTD_matchState_t* ms, seqStore_t* seqStore, U32 rep[ZSTD_REP_NUM],
        ZSTD_compressionParameters const* cParams, void const* src, size_t srcSize)
{
    return ZSTD_compressBlock_lazy_generic(ms, seqStore, rep, cParams, src, srcSize, 0, 2);
}

size_t ZSTD_compressBlock_lazy(
        ZSTD_matchState_t* ms, seqStore_t* seqStore, U32 rep[ZSTD_REP_NUM],
        ZSTD_compressionParameters const* cParams, void const* src, size_t srcSize)
{
    return ZSTD_compressBlock_lazy_generic(ms, seqStore, rep, cParams, src, srcSize, 0, 1);
}

size_t ZSTD_compressBlock_greedy(
        ZSTD_matchState_t* ms, seqStore_t* seqStore, U32 rep[ZSTD_REP_NUM],
        ZSTD_compressionParameters const* cParams, void const* src, size_t srcSize)
{
    return ZSTD_compressBlock_lazy_generic(ms, seqStore, rep, cParams, src, srcSize, 0, 0);
}


FORCE_INLINE_TEMPLATE
size_t ZSTD_compressBlock_lazy_extDict_generic(
                        ZSTD_matchState_t* ms, seqStore_t* seqStore,
                        U32 rep[ZSTD_REP_NUM],
                        ZSTD_compressionParameters const* cParams,
                        const void* src, size_t srcSize,
                        const U32 searchMethod, const U32 depth)
{
    const BYTE* const istart = (const BYTE*)src;
    const BYTE* ip = istart;
    const BYTE* anchor = istart;
    const BYTE* const iend = istart + srcSize;
    const BYTE* const ilimit = iend - 8;
    const BYTE* const base = ms->window.base;
    const U32 dictLimit = ms->window.dictLimit;
    const U32 lowestIndex = ms->window.lowLimit;
    const BYTE* const prefixStart = base + dictLimit;
    const BYTE* const dictBase = ms->window.dictBase;
    const BYTE* const dictEnd  = dictBase + dictLimit;
    const BYTE* const dictStart  = dictBase + lowestIndex;

    typedef size_t (*searchMax_f)(
                        ZSTD_matchState_t* ms, ZSTD_compressionParameters const* cParams,
                        const BYTE* ip, const BYTE* iLimit, size_t* offsetPtr);
    searchMax_f searchMax = searchMethod ? ZSTD_BtFindBestMatch_selectMLS_extDict : ZSTD_HcFindBestMatch_extDict_selectMLS;

    U32 offset_1 = rep[0], offset_2 = rep[1];

    /* init */
    ms->nextToUpdate3 = ms->nextToUpdate;
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
            size_t const ml2 = searchMax(ms, cParams, ip, iend, &offsetFound);
            if (ml2 > matchLength)
                matchLength = ml2, start = ip, offset=offsetFound;
        }

         if (matchLength < 4) {
            ip += ((ip-anchor) >> kSearchStrength) + 1;   /* jump faster over incompressible sections */
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
                size_t const ml2 = searchMax(ms, cParams, ip, iend, &offset2);
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
                    size_t const ml2 = searchMax(ms, cParams, ip, iend, &offset2);
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
            ZSTD_storeSeq(seqStore, litLength, anchor, (U32)offset, matchLength-MINMATCH);
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
                ZSTD_storeSeq(seqStore, 0, anchor, 0, matchLength-MINMATCH);
                ip += matchLength;
                anchor = ip;
                continue;   /* faster when present ... (?) */
            }
            break;
    }   }

    /* Save reps for next block */
    rep[0] = offset_1;
    rep[1] = offset_2;

    /* Return the last literals size */
    return iend - anchor;
}


size_t ZSTD_compressBlock_greedy_extDict(
        ZSTD_matchState_t* ms, seqStore_t* seqStore, U32 rep[ZSTD_REP_NUM],
        ZSTD_compressionParameters const* cParams, void const* src, size_t srcSize)
{
    return ZSTD_compressBlock_lazy_extDict_generic(ms, seqStore, rep, cParams, src, srcSize, 0, 0);
}

size_t ZSTD_compressBlock_lazy_extDict(
        ZSTD_matchState_t* ms, seqStore_t* seqStore, U32 rep[ZSTD_REP_NUM],
        ZSTD_compressionParameters const* cParams, void const* src, size_t srcSize)

{
    return ZSTD_compressBlock_lazy_extDict_generic(ms, seqStore, rep, cParams, src, srcSize, 0, 1);
}

size_t ZSTD_compressBlock_lazy2_extDict(
        ZSTD_matchState_t* ms, seqStore_t* seqStore, U32 rep[ZSTD_REP_NUM],
        ZSTD_compressionParameters const* cParams, void const* src, size_t srcSize)

{
    return ZSTD_compressBlock_lazy_extDict_generic(ms, seqStore, rep, cParams, src, srcSize, 0, 2);
}

size_t ZSTD_compressBlock_btlazy2_extDict(
        ZSTD_matchState_t* ms, seqStore_t* seqStore, U32 rep[ZSTD_REP_NUM],
        ZSTD_compressionParameters const* cParams, void const* src, size_t srcSize)

{
    return ZSTD_compressBlock_lazy_extDict_generic(ms, seqStore, rep, cParams, src, srcSize, 1, 2);
}
