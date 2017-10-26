/*
 * Copyright (c) 2016-present, Yann Collet, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under both the BSD-style license (found in the
 * LICENSE file in the root directory of this source tree) and the GPLv2 (found
 * in the COPYING file in the root directory of this source tree).
 */

#include "zstd_ldm.h"

#include "zstd_fast.h"          /* ZSTD_fillHashTable() */
#include "zstd_double_fast.h"   /* ZSTD_fillDoubleHashTable() */

#define LDM_BUCKET_SIZE_LOG 3
#define LDM_MIN_MATCH_LENGTH 64
#define LDM_HASH_RLOG 7
#define LDM_HASH_CHAR_OFFSET 10

size_t ZSTD_ldm_initializeParameters(ldmParams_t* params, U32 enableLdm)
{
    ZSTD_STATIC_ASSERT(LDM_BUCKET_SIZE_LOG <= ZSTD_LDM_BUCKETSIZELOG_MAX);
    params->enableLdm = enableLdm>0;
    params->hashLog = 0;
    params->bucketSizeLog = LDM_BUCKET_SIZE_LOG;
    params->minMatchLength = LDM_MIN_MATCH_LENGTH;
    params->hashEveryLog = ZSTD_LDM_HASHEVERYLOG_NOTSET;
    return 0;
}

void ZSTD_ldm_adjustParameters(ldmParams_t* params, U32 windowLog)
{
    if (params->hashLog == 0) {
        params->hashLog = MAX(ZSTD_HASHLOG_MIN, windowLog - LDM_HASH_RLOG);
        assert(params->hashLog <= ZSTD_HASHLOG_MAX);
    }
    if (params->hashEveryLog == ZSTD_LDM_HASHEVERYLOG_NOTSET) {
        params->hashEveryLog =
                windowLog < params->hashLog ? 0 : windowLog - params->hashLog;
    }
    params->bucketSizeLog = MIN(params->bucketSizeLog, params->hashLog);
}

size_t ZSTD_ldm_getTableSize(U32 hashLog, U32 bucketSizeLog) {
    size_t const ldmHSize = ((size_t)1) << hashLog;
    size_t const ldmBucketSizeLog = MIN(bucketSizeLog, hashLog);
    size_t const ldmBucketSize =
        ((size_t)1) << (hashLog - ldmBucketSizeLog);
    return ldmBucketSize + (ldmHSize * (sizeof(ldmEntry_t)));
}

/** ZSTD_ldm_getSmallHash() :
 *  numBits should be <= 32
 *  If numBits==0, returns 0.
 *  @return : the most significant numBits of value. */
static U32 ZSTD_ldm_getSmallHash(U64 value, U32 numBits)
{
    assert(numBits <= 32);
    return numBits == 0 ? 0 : (U32)(value >> (64 - numBits));
}

/** ZSTD_ldm_getChecksum() :
 *  numBitsToDiscard should be <= 32
 *  @return : the next most significant 32 bits after numBitsToDiscard */
static U32 ZSTD_ldm_getChecksum(U64 hash, U32 numBitsToDiscard)
{
    assert(numBitsToDiscard <= 32);
    return (hash >> (64 - 32 - numBitsToDiscard)) & 0xFFFFFFFF;
}

/** ZSTD_ldm_getTag() ;
 *  Given the hash, returns the most significant numTagBits bits
 *  after (32 + hbits) bits.
 *
 *  If there are not enough bits remaining, return the last
 *  numTagBits bits. */
static U32 ZSTD_ldm_getTag(U64 hash, U32 hbits, U32 numTagBits)
{
    assert(numTagBits < 32 && hbits <= 32);
    if (32 - hbits < numTagBits) {
        return hash & (((U32)1 << numTagBits) - 1);
    } else {
        return (hash >> (32 - hbits - numTagBits)) & (((U32)1 << numTagBits) - 1);
    }
}

/** ZSTD_ldm_getBucket() :
 *  Returns a pointer to the start of the bucket associated with hash. */
static ldmEntry_t* ZSTD_ldm_getBucket(
        ldmState_t* ldmState, size_t hash, ldmParams_t const ldmParams)
{
    return ldmState->hashTable + (hash << ldmParams.bucketSizeLog);
}

/** ZSTD_ldm_insertEntry() :
 *  Insert the entry with corresponding hash into the hash table */
static void ZSTD_ldm_insertEntry(ldmState_t* ldmState,
                                 size_t const hash, const ldmEntry_t entry,
                                 ldmParams_t const ldmParams)
{
    BYTE* const bucketOffsets = ldmState->bucketOffsets;
    *(ZSTD_ldm_getBucket(ldmState, hash, ldmParams) + bucketOffsets[hash]) = entry;
    bucketOffsets[hash]++;
    bucketOffsets[hash] &= ((U32)1 << ldmParams.bucketSizeLog) - 1;
}

/** ZSTD_ldm_makeEntryAndInsertByTag() :
 *
 *  Gets the small hash, checksum, and tag from the rollingHash.
 *
 *  If the tag matches (1 << ldmParams.hashEveryLog)-1, then
 *  creates an ldmEntry from the offset, and inserts it into the hash table.
 *
 *  hBits is the length of the small hash, which is the most significant hBits
 *  of rollingHash. The checksum is the next 32 most significant bits, followed
 *  by ldmParams.hashEveryLog bits that make up the tag. */
static void ZSTD_ldm_makeEntryAndInsertByTag(ldmState_t* ldmState,
                                             U64 const rollingHash,
                                             U32 const hBits,
                                             U32 const offset,
                                             ldmParams_t const ldmParams)
{
    U32 const tag = ZSTD_ldm_getTag(rollingHash, hBits, ldmParams.hashEveryLog);
    U32 const tagMask = ((U32)1 << ldmParams.hashEveryLog) - 1;
    if (tag == tagMask) {
        U32 const hash = ZSTD_ldm_getSmallHash(rollingHash, hBits);
        U32 const checksum = ZSTD_ldm_getChecksum(rollingHash, hBits);
        ldmEntry_t entry;
        entry.offset = offset;
        entry.checksum = checksum;
        ZSTD_ldm_insertEntry(ldmState, hash, entry, ldmParams);
    }
}

/** ZSTD_ldm_getRollingHash() :
 *  Get a 64-bit hash using the first len bytes from buf.
 *
 *  Giving bytes s = s_1, s_2, ... s_k, the hash is defined to be
 *  H(s) = s_1*(a^(k-1)) + s_2*(a^(k-2)) + ... + s_k*(a^0)
 *
 *  where the constant a is defined to be prime8bytes.
 *
 *  The implementation adds an offset to each byte, so
 *  H(s) = (s_1 + HASH_CHAR_OFFSET)*(a^(k-1)) + ... */
static U64 ZSTD_ldm_getRollingHash(const BYTE* buf, U32 len)
{
    U64 ret = 0;
    U32 i;
    for (i = 0; i < len; i++) {
        ret *= prime8bytes;
        ret += buf[i] + LDM_HASH_CHAR_OFFSET;
    }
    return ret;
}

/** ZSTD_ldm_ipow() :
 *  Return base^exp. */
static U64 ZSTD_ldm_ipow(U64 base, U64 exp)
{
    U64 ret = 1;
    while (exp) {
        if (exp & 1) { ret *= base; }
        exp >>= 1;
        base *= base;
    }
    return ret;
}

U64 ZSTD_ldm_getHashPower(U32 minMatchLength) {
    assert(minMatchLength >= ZSTD_LDM_MINMATCH_MIN);
    return ZSTD_ldm_ipow(prime8bytes, minMatchLength - 1);
}

/** ZSTD_ldm_updateHash() :
 *  Updates hash by removing toRemove and adding toAdd. */
static U64 ZSTD_ldm_updateHash(U64 hash, BYTE toRemove, BYTE toAdd, U64 hashPower)
{
    hash -= ((toRemove + LDM_HASH_CHAR_OFFSET) * hashPower);
    hash *= prime8bytes;
    hash += toAdd + LDM_HASH_CHAR_OFFSET;
    return hash;
}

/** ZSTD_ldm_countBackwardsMatch() :
 *  Returns the number of bytes that match backwards before pIn and pMatch.
 *
 *  We count only bytes where pMatch >= pBase and pIn >= pAnchor. */
static size_t ZSTD_ldm_countBackwardsMatch(
            const BYTE* pIn, const BYTE* pAnchor,
            const BYTE* pMatch, const BYTE* pBase)
{
    size_t matchLength = 0;
    while (pIn > pAnchor && pMatch > pBase && pIn[-1] == pMatch[-1]) {
        pIn--;
        pMatch--;
        matchLength++;
    }
    return matchLength;
}

/** ZSTD_ldm_fillFastTables() :
 *
 *  Fills the relevant tables for the ZSTD_fast and ZSTD_dfast strategies.
 *  This is similar to ZSTD_loadDictionaryContent.
 *
 *  The tables for the other strategies are filled within their
 *  block compressors. */
static size_t ZSTD_ldm_fillFastTables(ZSTD_CCtx* zc, const void* end)
{
    const BYTE* const iend = (const BYTE*)end;
    const U32 mls = zc->appliedParams.cParams.searchLength;

    switch(zc->appliedParams.cParams.strategy)
    {
    case ZSTD_fast:
        ZSTD_fillHashTable(zc, iend, mls);
        zc->nextToUpdate = (U32)(iend - zc->base);
        break;

    case ZSTD_dfast:
        ZSTD_fillDoubleHashTable(zc, iend, mls);
        zc->nextToUpdate = (U32)(iend - zc->base);
        break;

    case ZSTD_greedy:
    case ZSTD_lazy:
    case ZSTD_lazy2:
    case ZSTD_btlazy2:
    case ZSTD_btopt:
    case ZSTD_btultra:
        break;
    default:
        assert(0);  /* not possible : not a valid strategy id */
    }

    return 0;
}

/** ZSTD_ldm_fillLdmHashTable() :
 *
 *  Fills hashTable from (lastHashed + 1) to iend (non-inclusive).
 *  lastHash is the rolling hash that corresponds to lastHashed.
 *
 *  Returns the rolling hash corresponding to position iend-1. */
static U64 ZSTD_ldm_fillLdmHashTable(ldmState_t* state,
                                     U64 lastHash, const BYTE* lastHashed,
                                     const BYTE* iend, const BYTE* base,
                                     U32 hBits, ldmParams_t const ldmParams)
{
    U64 rollingHash = lastHash;
    const BYTE* cur = lastHashed + 1;

    while (cur < iend) {
        rollingHash = ZSTD_ldm_updateHash(rollingHash, cur[-1],
                                          cur[ldmParams.minMatchLength-1],
                                          state->hashPower);
        ZSTD_ldm_makeEntryAndInsertByTag(state,
                                         rollingHash, hBits,
                                         (U32)(cur - base), ldmParams);
        ++cur;
    }
    return rollingHash;
}


/** ZSTD_ldm_limitTableUpdate() :
 *
 *  Sets cctx->nextToUpdate to a position corresponding closer to anchor
 *  if it is far way
 *  (after a long match, only update tables a limited amount). */
static void ZSTD_ldm_limitTableUpdate(ZSTD_CCtx* cctx, const BYTE* anchor)
{
    U32 const current = (U32)(anchor - cctx->base);
    if (current > cctx->nextToUpdate + 1024) {
        cctx->nextToUpdate =
            current - MIN(512, current - cctx->nextToUpdate - 1024);
    }
}

typedef size_t (*ZSTD_blockCompressor) (ZSTD_CCtx* ctx, const void* src, size_t srcSize);
/* defined in zstd_compress.c */
ZSTD_blockCompressor ZSTD_selectBlockCompressor(ZSTD_strategy strat, int extDict);

FORCE_INLINE_TEMPLATE
size_t ZSTD_compressBlock_ldm_generic(ZSTD_CCtx* cctx,
                                      const void* src, size_t srcSize)
{
    ldmState_t* const ldmState = &(cctx->ldmState);
    const ldmParams_t ldmParams = cctx->appliedParams.ldmParams;
    const U64 hashPower = ldmState->hashPower;
    const U32 hBits = ldmParams.hashLog - ldmParams.bucketSizeLog;
    const U32 ldmBucketSize = ((U32)1 << ldmParams.bucketSizeLog);
    const U32 ldmTagMask = ((U32)1 << ldmParams.hashEveryLog) - 1;
    seqStore_t* const seqStorePtr = &(cctx->seqStore);
    const BYTE* const base = cctx->base;
    const BYTE* const istart = (const BYTE*)src;
    const BYTE* ip = istart;
    const BYTE* anchor = istart;
    const U32   lowestIndex = cctx->dictLimit;
    const BYTE* const lowest = base + lowestIndex;
    const BYTE* const iend = istart + srcSize;
    const BYTE* const ilimit = iend - MAX(ldmParams.minMatchLength, HASH_READ_SIZE);

    const ZSTD_blockCompressor blockCompressor =
        ZSTD_selectBlockCompressor(cctx->appliedParams.cParams.strategy, 0);
    U32* const repToConfirm = seqStorePtr->repToConfirm;
    U32 savedRep[ZSTD_REP_NUM];
    U64 rollingHash = 0;
    const BYTE* lastHashed = NULL;
    size_t i, lastLiterals;

    /* Save seqStorePtr->rep and copy repToConfirm */
    for (i = 0; i < ZSTD_REP_NUM; i++)
        savedRep[i] = repToConfirm[i] = seqStorePtr->rep[i];

    /* Main Search Loop */
    while (ip < ilimit) {   /* < instead of <=, because repcode check at (ip+1) */
        size_t mLength;
        U32 const current = (U32)(ip - base);
        size_t forwardMatchLength = 0, backwardMatchLength = 0;
        ldmEntry_t* bestEntry = NULL;
        if (ip != istart) {
            rollingHash = ZSTD_ldm_updateHash(rollingHash, lastHashed[0],
                                              lastHashed[ldmParams.minMatchLength],
                                              hashPower);
        } else {
            rollingHash = ZSTD_ldm_getRollingHash(ip, ldmParams.minMatchLength);
        }
        lastHashed = ip;

        /* Do not insert and do not look for a match */
        if (ZSTD_ldm_getTag(rollingHash, hBits, ldmParams.hashEveryLog) !=
                ldmTagMask) {
           ip++;
           continue;
        }

        /* Get the best entry and compute the match lengths */
        {
            ldmEntry_t* const bucket =
                ZSTD_ldm_getBucket(ldmState,
                                   ZSTD_ldm_getSmallHash(rollingHash, hBits),
                                   ldmParams);
            ldmEntry_t* cur;
            size_t bestMatchLength = 0;
            U32 const checksum = ZSTD_ldm_getChecksum(rollingHash, hBits);

            for (cur = bucket; cur < bucket + ldmBucketSize; ++cur) {
                const BYTE* const pMatch = cur->offset + base;
                size_t curForwardMatchLength, curBackwardMatchLength,
                       curTotalMatchLength;
                if (cur->checksum != checksum || cur->offset <= lowestIndex) {
                    continue;
                }

                curForwardMatchLength = ZSTD_count(ip, pMatch, iend);
                if (curForwardMatchLength < ldmParams.minMatchLength) {
                    continue;
                }
                curBackwardMatchLength = ZSTD_ldm_countBackwardsMatch(
                                             ip, anchor, pMatch, lowest);
                curTotalMatchLength = curForwardMatchLength +
                                      curBackwardMatchLength;

                if (curTotalMatchLength > bestMatchLength) {
                    bestMatchLength = curTotalMatchLength;
                    forwardMatchLength = curForwardMatchLength;
                    backwardMatchLength = curBackwardMatchLength;
                    bestEntry = cur;
                }
            }
        }

        /* No match found -- continue searching */
        if (bestEntry == NULL) {
            ZSTD_ldm_makeEntryAndInsertByTag(ldmState, rollingHash,
                                             hBits, current,
                                             ldmParams);
            ip++;
            continue;
        }

        /* Match found */
        mLength = forwardMatchLength + backwardMatchLength;
        ip -= backwardMatchLength;

        /* Call the block compressor on the remaining literals */
        {
            U32 const matchIndex = bestEntry->offset;
            const BYTE* const match = base + matchIndex - backwardMatchLength;
            U32 const offset = (U32)(ip - match);

            /* Overwrite rep codes */
            for (i = 0; i < ZSTD_REP_NUM; i++)
                seqStorePtr->rep[i] = repToConfirm[i];

            /* Fill tables for block compressor */
            ZSTD_ldm_limitTableUpdate(cctx, anchor);
            ZSTD_ldm_fillFastTables(cctx, anchor);

            /* Call block compressor and get remaining literals */
            lastLiterals = blockCompressor(cctx, anchor, ip - anchor);
            cctx->nextToUpdate = (U32)(ip - base);

            /* Update repToConfirm with the new offset */
            for (i = ZSTD_REP_NUM - 1; i > 0; i--)
                repToConfirm[i] = repToConfirm[i-1];
            repToConfirm[0] = offset;

            /* Store the sequence with the leftover literals */
            ZSTD_storeSeq(seqStorePtr, lastLiterals, ip - lastLiterals,
                          offset + ZSTD_REP_MOVE, mLength - MINMATCH);
        }

        /* Insert the current entry into the hash table */
        ZSTD_ldm_makeEntryAndInsertByTag(ldmState, rollingHash, hBits,
                                         (U32)(lastHashed - base),
                                         ldmParams);

        assert(ip + backwardMatchLength == lastHashed);

        /* Fill the hash table from lastHashed+1 to ip+mLength*/
        /* Heuristic: don't need to fill the entire table at end of block */
        if (ip + mLength < ilimit) {
            rollingHash = ZSTD_ldm_fillLdmHashTable(
                              ldmState, rollingHash, lastHashed,
                              ip + mLength, base, hBits, ldmParams);
            lastHashed = ip + mLength - 1;
        }
        ip += mLength;
        anchor = ip;
        /* Check immediate repcode */
        while ( (ip < ilimit)
             && ( (repToConfirm[1] > 0) && (repToConfirm[1] <= (U32)(ip-lowest))
             && (MEM_read32(ip) == MEM_read32(ip - repToConfirm[1])) )) {

            size_t const rLength = ZSTD_count(ip+4, ip+4-repToConfirm[1],
                                              iend) + 4;
            /* Swap repToConfirm[1] <=> repToConfirm[0] */
            {
                U32 const tmpOff = repToConfirm[1];
                repToConfirm[1] = repToConfirm[0];
                repToConfirm[0] = tmpOff;
            }

            ZSTD_storeSeq(seqStorePtr, 0, anchor, 0, rLength-MINMATCH);

            /* Fill the  hash table from lastHashed+1 to ip+rLength*/
            if (ip + rLength < ilimit) {
                rollingHash = ZSTD_ldm_fillLdmHashTable(
                                ldmState, rollingHash, lastHashed,
                                ip + rLength, base, hBits, ldmParams);
                lastHashed = ip + rLength - 1;
            }
            ip += rLength;
            anchor = ip;
        }
    }

    /* Overwrite rep */
    for (i = 0; i < ZSTD_REP_NUM; i++)
        seqStorePtr->rep[i] = repToConfirm[i];

    ZSTD_ldm_limitTableUpdate(cctx, anchor);
    ZSTD_ldm_fillFastTables(cctx, anchor);

    lastLiterals = blockCompressor(cctx, anchor, iend - anchor);
    cctx->nextToUpdate = (U32)(iend - base);

    /* Restore seqStorePtr->rep */
    for (i = 0; i < ZSTD_REP_NUM; i++)
        seqStorePtr->rep[i] = savedRep[i];

    /* Return the last literals size */
    return lastLiterals;
}

size_t ZSTD_compressBlock_ldm(ZSTD_CCtx* ctx,
                              const void* src, size_t srcSize)
{
    return ZSTD_compressBlock_ldm_generic(ctx, src, srcSize);
}

static size_t ZSTD_compressBlock_ldm_extDict_generic(
                                 ZSTD_CCtx* ctx,
                                 const void* src, size_t srcSize)
{
    ldmState_t* const ldmState = &(ctx->ldmState);
    const ldmParams_t ldmParams = ctx->appliedParams.ldmParams;
    const U64 hashPower = ldmState->hashPower;
    const U32 hBits = ldmParams.hashLog - ldmParams.bucketSizeLog;
    const U32 ldmBucketSize = ((U32)1 << ldmParams.bucketSizeLog);
    const U32 ldmTagMask = ((U32)1 << ldmParams.hashEveryLog) - 1;
    seqStore_t* const seqStorePtr = &(ctx->seqStore);
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
    const BYTE* const ilimit = iend - MAX(ldmParams.minMatchLength, HASH_READ_SIZE);

    const ZSTD_blockCompressor blockCompressor =
        ZSTD_selectBlockCompressor(ctx->appliedParams.cParams.strategy, 1);
    U32* const repToConfirm = seqStorePtr->repToConfirm;
    U32 savedRep[ZSTD_REP_NUM];
    U64 rollingHash = 0;
    const BYTE* lastHashed = NULL;
    size_t i, lastLiterals;

    /* Save seqStorePtr->rep and copy repToConfirm */
    for (i = 0; i < ZSTD_REP_NUM; i++) {
        savedRep[i] = repToConfirm[i] = seqStorePtr->rep[i];
    }

    /* Search Loop */
    while (ip < ilimit) {  /* < instead of <=, because (ip+1) */
        size_t mLength;
        const U32 current = (U32)(ip-base);
        size_t forwardMatchLength = 0, backwardMatchLength = 0;
        ldmEntry_t* bestEntry = NULL;
        if (ip != istart) {
          rollingHash = ZSTD_ldm_updateHash(rollingHash, lastHashed[0],
                                       lastHashed[ldmParams.minMatchLength],
                                       hashPower);
        } else {
            rollingHash = ZSTD_ldm_getRollingHash(ip, ldmParams.minMatchLength);
        }
        lastHashed = ip;

        if (ZSTD_ldm_getTag(rollingHash, hBits, ldmParams.hashEveryLog) !=
                ldmTagMask) {
            /* Don't insert and don't look for a match */
           ip++;
           continue;
        }

        /* Get the best entry and compute the match lengths */
        {
            ldmEntry_t* const bucket =
                ZSTD_ldm_getBucket(ldmState,
                                   ZSTD_ldm_getSmallHash(rollingHash, hBits),
                                   ldmParams);
            ldmEntry_t* cur;
            size_t bestMatchLength = 0;
            U32 const checksum = ZSTD_ldm_getChecksum(rollingHash, hBits);

            for (cur = bucket; cur < bucket + ldmBucketSize; ++cur) {
                const BYTE* const curMatchBase =
                    cur->offset < dictLimit ? dictBase : base;
                const BYTE* const pMatch = curMatchBase + cur->offset;
                const BYTE* const matchEnd =
                    cur->offset < dictLimit ? dictEnd : iend;
                const BYTE* const lowMatchPtr =
                    cur->offset < dictLimit ? dictStart : lowPrefixPtr;
                size_t curForwardMatchLength, curBackwardMatchLength,
                       curTotalMatchLength;

                if (cur->checksum != checksum || cur->offset <= lowestIndex) {
                    continue;
                }

                curForwardMatchLength = ZSTD_count_2segments(
                                            ip, pMatch, iend,
                                            matchEnd, lowPrefixPtr);
                if (curForwardMatchLength < ldmParams.minMatchLength) {
                    continue;
                }
                curBackwardMatchLength = ZSTD_ldm_countBackwardsMatch(
                                             ip, anchor, pMatch, lowMatchPtr);
                curTotalMatchLength = curForwardMatchLength +
                                      curBackwardMatchLength;

                if (curTotalMatchLength > bestMatchLength) {
                    bestMatchLength = curTotalMatchLength;
                    forwardMatchLength = curForwardMatchLength;
                    backwardMatchLength = curBackwardMatchLength;
                    bestEntry = cur;
                }
            }
        }

        /* No match found -- continue searching */
        if (bestEntry == NULL) {
            ZSTD_ldm_makeEntryAndInsertByTag(ldmState, rollingHash, hBits,
                                             (U32)(lastHashed - base),
                                             ldmParams);
            ip++;
            continue;
        }

        /* Match found */
        mLength = forwardMatchLength + backwardMatchLength;
        ip -= backwardMatchLength;

        /* Call the block compressor on the remaining literals */
        {
            /* ip = current - backwardMatchLength
             * The match is at (bestEntry->offset - backwardMatchLength) */
            U32 const matchIndex = bestEntry->offset;
            U32 const offset = current - matchIndex;

            /* Overwrite rep codes */
            for (i = 0; i < ZSTD_REP_NUM; i++)
                seqStorePtr->rep[i] = repToConfirm[i];

            /* Fill the hash table for the block compressor */
            ZSTD_ldm_limitTableUpdate(ctx, anchor);
            ZSTD_ldm_fillFastTables(ctx, anchor);

            /* Call block compressor and get remaining literals  */
            lastLiterals = blockCompressor(ctx, anchor, ip - anchor);
            ctx->nextToUpdate = (U32)(ip - base);

            /* Update repToConfirm with the new offset */
            for (i = ZSTD_REP_NUM - 1; i > 0; i--)
                repToConfirm[i] = repToConfirm[i-1];
            repToConfirm[0] = offset;

            /* Store the sequence with the leftover literals */
            ZSTD_storeSeq(seqStorePtr, lastLiterals, ip - lastLiterals,
                          offset + ZSTD_REP_MOVE, mLength - MINMATCH);
        }

        /* Insert the current entry into the hash table */
        ZSTD_ldm_makeEntryAndInsertByTag(ldmState, rollingHash, hBits,
                                         (U32)(lastHashed - base),
                                         ldmParams);

        /* Fill the hash table from lastHashed+1 to ip+mLength */
        assert(ip + backwardMatchLength == lastHashed);
        if (ip + mLength < ilimit) {
            rollingHash = ZSTD_ldm_fillLdmHashTable(
                              ldmState, rollingHash, lastHashed,
                              ip + mLength, base, hBits,
                              ldmParams);
            lastHashed = ip + mLength - 1;
        }
        ip += mLength;
        anchor = ip;

        /* check immediate repcode */
        while (ip < ilimit) {
            U32 const current2 = (U32)(ip-base);
            U32 const repIndex2 = current2 - repToConfirm[1];
            const BYTE* repMatch2 = repIndex2 < dictLimit ?
                                    dictBase + repIndex2 : base + repIndex2;
            if ( (((U32)((dictLimit-1) - repIndex2) >= 3) &
                        (repIndex2 > lowestIndex))  /* intentional overflow */
               && (MEM_read32(repMatch2) == MEM_read32(ip)) ) {
                const BYTE* const repEnd2 = repIndex2 < dictLimit ?
                                            dictEnd : iend;
                size_t const repLength2 =
                        ZSTD_count_2segments(ip+4, repMatch2+4, iend,
                                             repEnd2, lowPrefixPtr) + 4;

                U32 tmpOffset = repToConfirm[1];
                repToConfirm[1] = repToConfirm[0];
                repToConfirm[0] = tmpOffset;

                ZSTD_storeSeq(seqStorePtr, 0, anchor, 0, repLength2-MINMATCH);

                /* Fill the  hash table from lastHashed+1 to ip+repLength2*/
                if (ip + repLength2 < ilimit) {
                    rollingHash = ZSTD_ldm_fillLdmHashTable(
                                      ldmState, rollingHash, lastHashed,
                                      ip + repLength2, base, hBits,
                                      ldmParams);
                    lastHashed = ip + repLength2 - 1;
                }
                ip += repLength2;
                anchor = ip;
                continue;
            }
            break;
        }
    }

    /* Overwrite rep */
    for (i = 0; i < ZSTD_REP_NUM; i++)
        seqStorePtr->rep[i] = repToConfirm[i];

    ZSTD_ldm_limitTableUpdate(ctx, anchor);
    ZSTD_ldm_fillFastTables(ctx, anchor);

    /* Call the block compressor one last time on the last literals */
    lastLiterals = blockCompressor(ctx, anchor, iend - anchor);
    ctx->nextToUpdate = (U32)(iend - base);

    /* Restore seqStorePtr->rep */
    for (i = 0; i < ZSTD_REP_NUM; i++)
        seqStorePtr->rep[i] = savedRep[i];

    /* Return the last literals size */
    return lastLiterals;
}

size_t ZSTD_compressBlock_ldm_extDict(ZSTD_CCtx* ctx,
                                      const void* src, size_t srcSize)
{
    return ZSTD_compressBlock_ldm_extDict_generic(ctx, src, srcSize);
}
