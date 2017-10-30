/*
 * Copyright (c) 2016-present, Yann Collet, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under both the BSD-style license (found in the
 * LICENSE file in the root directory of this source tree) and the GPLv2 (found
 * in the COPYING file in the root directory of this source tree).
 * You may select, at your option, one of the above-listed licenses.
 */

#ifndef ZSTD_OPT_H
#define ZSTD_OPT_H

#include "zstd_compress.h"

#if defined (__cplusplus)
extern "C" {
#endif

size_t ZSTD_compressBlock_btopt(ZSTD_CCtx* ctx, const void* src, size_t srcSize);
size_t ZSTD_compressBlock_btultra(ZSTD_CCtx* ctx, const void* src, size_t srcSize);

size_t ZSTD_compressBlock_btopt_extDict(ZSTD_CCtx* ctx, const void* src, size_t srcSize);
size_t ZSTD_compressBlock_btultra_extDict(ZSTD_CCtx* ctx, const void* src, size_t srcSize);

#if defined (__cplusplus)
}
#endif

#endif /* ZSTD_OPT_H */
