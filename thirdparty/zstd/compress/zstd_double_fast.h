/*
 * Copyright (c) 2016-present, Yann Collet, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under both the BSD-style license (found in the
 * LICENSE file in the root directory of this source tree) and the GPLv2 (found
 * in the COPYING file in the root directory of this source tree).
 * You may select, at your option, one of the above-listed licenses.
 */

#ifndef ZSTD_DOUBLE_FAST_H
#define ZSTD_DOUBLE_FAST_H

#if defined (__cplusplus)
extern "C" {
#endif

#include "mem.h"      /* U32 */
#include "zstd.h"     /* ZSTD_CCtx, size_t */

void ZSTD_fillDoubleHashTable(ZSTD_CCtx* cctx, const void* end, const U32 mls);
size_t ZSTD_compressBlock_doubleFast(ZSTD_CCtx* ctx, const void* src, size_t srcSize);
size_t ZSTD_compressBlock_doubleFast_extDict(ZSTD_CCtx* ctx, const void* src, size_t srcSize);

#if defined (__cplusplus)
}
#endif

#endif /* ZSTD_DOUBLE_FAST_H */
