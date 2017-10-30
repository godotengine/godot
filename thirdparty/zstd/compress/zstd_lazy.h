/*
 * Copyright (c) 2016-present, Yann Collet, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under both the BSD-style license (found in the
 * LICENSE file in the root directory of this source tree) and the GPLv2 (found
 * in the COPYING file in the root directory of this source tree).
 * You may select, at your option, one of the above-listed licenses.
 */

#ifndef ZSTD_LAZY_H
#define ZSTD_LAZY_H

#include "zstd_compress.h"

#if defined (__cplusplus)
extern "C" {
#endif

U32 ZSTD_insertAndFindFirstIndex (ZSTD_CCtx* zc, const BYTE* ip, U32 mls);
void ZSTD_updateTree(ZSTD_CCtx* zc, const BYTE* const ip, const BYTE* const iend, const U32 nbCompares, const U32 mls);
void ZSTD_updateTree_extDict(ZSTD_CCtx* zc, const BYTE* const ip, const BYTE* const iend, const U32 nbCompares, const U32 mls);

size_t ZSTD_compressBlock_btlazy2(ZSTD_CCtx* ctx, const void* src, size_t srcSize);
size_t ZSTD_compressBlock_lazy2(ZSTD_CCtx* ctx, const void* src, size_t srcSize);
size_t ZSTD_compressBlock_lazy(ZSTD_CCtx* ctx, const void* src, size_t srcSize);
size_t ZSTD_compressBlock_greedy(ZSTD_CCtx* ctx, const void* src, size_t srcSize);

size_t ZSTD_compressBlock_greedy_extDict(ZSTD_CCtx* ctx, const void* src, size_t srcSize);
size_t ZSTD_compressBlock_lazy_extDict(ZSTD_CCtx* ctx, const void* src, size_t srcSize);
size_t ZSTD_compressBlock_lazy2_extDict(ZSTD_CCtx* ctx, const void* src, size_t srcSize);
size_t ZSTD_compressBlock_btlazy2_extDict(ZSTD_CCtx* ctx, const void* src, size_t srcSize);

#if defined (__cplusplus)
}
#endif

#endif /* ZSTD_LAZY_H */
