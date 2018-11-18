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
#include "zstd_fast.h"


void ZSTD_fillHashTable(ZSTD_matchState_t* ms,
                        ZSTD_compressionParameters const* cParams,
                        void const* end)
{
    U32* const hashTable = ms->hashTable;
    U32  const hBits = cParams->hashLog;
    U32  const mls = cParams->searchLength;
    const BYTE* const base = ms->window.base;
    const BYTE* ip = base + ms->nextToUpdate;
    const BYTE* const iend = ((const BYTE*)end) - HASH_READ_SIZE;
    const U32 fastHashFillStep = 3;

    /* Always insert every fastHashFillStep position into the hash table.
     * Insert the other positions if their hash entry is empty.
     */
    for (; ip + fastHashFillStep - 1 <= iend; ip += fastHashFillStep) {
        U32 const current = (U32)(ip - base);
        U32 i;
        for (i = 0; i < fastHashFillStep; ++i) {
            size_t const hash = ZSTD_hashPtr(ip + i, hBits, mls);
            if (i == 0 || hashTable[hash] == 0)
                hashTable[hash] = current + i;
        }
    }
}

FORCE_INLINE_TEMPLATE
size_t ZSTD_compressBlock_fast_generic(
        ZSTD_matchState_t* ms, seqStore_t* seqStore, U32 rep[ZSTD_REP_NUM],
        void const* src, size_t srcSize,
        U32 const hlog, U32 const stepSize, U32 const mls)
{
    U32* const hashTable = ms->hashTable;
    const BYTE* const base = ms->window.base;
    const BYTE* const istart = (const BYTE*)src;
    const BYTE* ip = istart;
    const BYTE* anchor = istart;
    const U32   lowestIndex = ms->window.dictLimit;
    const BYTE* const lowest = base + lowestIndex;
    const BYTE* const iend = istart + srcSize;
    const BYTE* const ilimit = iend - HASH_READ_SIZE;
    U32 offset_1=rep[0], offset_2=rep[1];
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
        size_t const h = ZSTD_hashPtr(ip, hlog, mls);
        U32 const current = (U32)(ip-base);
        U32 const matchIndex = hashTable[h];
        const BYTE* match = base + matchIndex;
        hashTable[h] = current;   /* update hash table */

        if ((offset_1 > 0) & (MEM_read32(ip+1-offset_1) == MEM_read32(ip+1))) {
            mLength = ZSTD_count(ip+1+4, ip+1+4-offset_1, iend) + 4;
            ip++;
            ZSTD_storeSeq(seqStore, ip-anchor, anchor, 0, mLength-MINMATCH);
        } else {
            if ( (matchIndex <= lowestIndex)
              || (MEM_read32(match) != MEM_read32(ip)) ) {
                assert(stepSize >= 1);
                ip += ((ip-anchor) >> kSearchStrength) + stepSize;
                continue;
            }
            mLength = ZSTD_count(ip+4, match+4, iend) + 4;
            {   U32 const offset = (U32)(ip-match);
                while (((ip>anchor) & (match>lowest)) && (ip[-1] == match[-1])) { ip--; match--; mLength++; } /* catch up */
                offset_2 = offset_1;
                offset_1 = offset;
                ZSTD_storeSeq(seqStore, ip-anchor, anchor, offset + ZSTD_REP_MOVE, mLength-MINMATCH);
        }   }

        /* match found */
        ip += mLength;
        anchor = ip;

        if (ip <= ilimit) {
            /* Fill Table */
            hashTable[ZSTD_hashPtr(base+current+2, hlog, mls)] = current+2;  /* here because current+2 could be > iend-8 */
            hashTable[ZSTD_hashPtr(ip-2, hlog, mls)] = (U32)(ip-2-base);
            /* check immediate repcode */
            while ( (ip <= ilimit)
                 && ( (offset_2>0)
                 & (MEM_read32(ip) == MEM_read32(ip - offset_2)) )) {
                /* store sequence */
                size_t const rLength = ZSTD_count(ip+4, ip+4-offset_2, iend) + 4;
                { U32 const tmpOff = offset_2; offset_2 = offset_1; offset_1 = tmpOff; }  /* swap offset_2 <=> offset_1 */
                hashTable[ZSTD_hashPtr(ip, hlog, mls)] = (U32)(ip-base);
                ZSTD_storeSeq(seqStore, 0, anchor, 0, rLength-MINMATCH);
                ip += rLength;
                anchor = ip;
                continue;   /* faster when present ... (?) */
    }   }   }

    /* save reps for next block */
    rep[0] = offset_1 ? offset_1 : offsetSaved;
    rep[1] = offset_2 ? offset_2 : offsetSaved;

    /* Return the last literals size */
    return iend - anchor;
}


size_t ZSTD_compressBlock_fast(
        ZSTD_matchState_t* ms, seqStore_t* seqStore, U32 rep[ZSTD_REP_NUM],
        ZSTD_compressionParameters const* cParams, void const* src, size_t srcSize)
{
    U32 const hlog = cParams->hashLog;
    U32 const mls = cParams->searchLength;
    U32 const stepSize = cParams->targetLength;
    switch(mls)
    {
    default: /* includes case 3 */
    case 4 :
        return ZSTD_compressBlock_fast_generic(ms, seqStore, rep, src, srcSize, hlog, stepSize, 4);
    case 5 :
        return ZSTD_compressBlock_fast_generic(ms, seqStore, rep, src, srcSize, hlog, stepSize, 5);
    case 6 :
        return ZSTD_compressBlock_fast_generic(ms, seqStore, rep, src, srcSize, hlog, stepSize, 6);
    case 7 :
        return ZSTD_compressBlock_fast_generic(ms, seqStore, rep, src, srcSize, hlog, stepSize, 7);
    }
}


static size_t ZSTD_compressBlock_fast_extDict_generic(
        ZSTD_matchState_t* ms, seqStore_t* seqStore, U32 rep[ZSTD_REP_NUM],
        void const* src, size_t srcSize,
        U32 const hlog, U32 const stepSize, U32 const mls)
{
    U32* hashTable = ms->hashTable;
    const BYTE* const base = ms->window.base;
    const BYTE* const dictBase = ms->window.dictBase;
    const BYTE* const istart = (const BYTE*)src;
    const BYTE* ip = istart;
    const BYTE* anchor = istart;
    const U32   lowestIndex = ms->window.lowLimit;
    const BYTE* const dictStart = dictBase + lowestIndex;
    const U32   dictLimit = ms->window.dictLimit;
    const BYTE* const lowPrefixPtr = base + dictLimit;
    const BYTE* const dictEnd = dictBase + dictLimit;
    const BYTE* const iend = istart + srcSize;
    const BYTE* const ilimit = iend - 8;
    U32 offset_1=rep[0], offset_2=rep[1];

    /* Search Loop */
    while (ip < ilimit) {  /* < instead of <=, because (ip+1) */
        const size_t h = ZSTD_hashPtr(ip, hlog, mls);
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
            ZSTD_storeSeq(seqStore, ip-anchor, anchor, 0, mLength-MINMATCH);
        } else {
            if ( (matchIndex < lowestIndex) ||
                 (MEM_read32(match) != MEM_read32(ip)) ) {
                assert(stepSize >= 1);
                ip += ((ip-anchor) >> kSearchStrength) + stepSize;
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
                ZSTD_storeSeq(seqStore, ip-anchor, anchor, offset + ZSTD_REP_MOVE, mLength-MINMATCH);
        }   }

        /* found a match : store it */
        ip += mLength;
        anchor = ip;

        if (ip <= ilimit) {
            /* Fill Table */
            hashTable[ZSTD_hashPtr(base+current+2, hlog, mls)] = current+2;
            hashTable[ZSTD_hashPtr(ip-2, hlog, mls)] = (U32)(ip-2-base);
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
                    ZSTD_storeSeq(seqStore, 0, anchor, 0, repLength2-MINMATCH);
                    hashTable[ZSTD_hashPtr(ip, hlog, mls)] = current2;
                    ip += repLength2;
                    anchor = ip;
                    continue;
                }
                break;
    }   }   }

    /* save reps for next block */
    rep[0] = offset_1;
    rep[1] = offset_2;

    /* Return the last literals size */
    return iend - anchor;
}


size_t ZSTD_compressBlock_fast_extDict(
        ZSTD_matchState_t* ms, seqStore_t* seqStore, U32 rep[ZSTD_REP_NUM],
        ZSTD_compressionParameters const* cParams, void const* src, size_t srcSize)
{
    U32 const hlog = cParams->hashLog;
    U32 const mls = cParams->searchLength;
    U32 const stepSize = cParams->targetLength;
    switch(mls)
    {
    default: /* includes case 3 */
    case 4 :
        return ZSTD_compressBlock_fast_extDict_generic(ms, seqStore, rep, src, srcSize, hlog, stepSize, 4);
    case 5 :
        return ZSTD_compressBlock_fast_extDict_generic(ms, seqStore, rep, src, srcSize, hlog, stepSize, 5);
    case 6 :
        return ZSTD_compressBlock_fast_extDict_generic(ms, seqStore, rep, src, srcSize, hlog, stepSize, 6);
    case 7 :
        return ZSTD_compressBlock_fast_extDict_generic(ms, seqStore, rep, src, srcSize, hlog, stepSize, 7);
    }
}
