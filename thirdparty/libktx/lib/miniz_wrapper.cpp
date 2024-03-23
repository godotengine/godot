/* -*- tab-width: 4; -*- */
/* vi: set sw=2 ts=4 expandtab: */

/*
 * Copyright 2023-2023 The Khronos Group Inc.
 * Copyright 2023-2023 RasterGrid Kft.
 * SPDX-License-Identifier: Apache-2.0
 */

/**
 * @internal
 * @file miniz_wrapper.c
 * @~English
 *
 * @brief Wrapper functions for ZLIB compression/decompression using miniz.
 *
 * @author Daniel Rakos, RasterGrid
 */

#include "ktx.h"
#include "ktxint.h"

#include <assert.h>

#if !KTX_FEATURE_WRITE
// The reader does not link with the basisu components that already include a
// definition of miniz so we include it here explicitly.
#ifdef __GNUC__
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wextra"
#pragma GCC diagnostic ignored "-Wmisleading-indentation"
#endif
#include "encoder/basisu_miniz.h"
#ifdef __GNUC__
#pragma GCC diagnostic pop
#endif
#else
// Otherwise we only declare the interfaces and link with the basisu version.
// This is needed because while miniz is defined as a header in basisu it's
// not declaring the functions as static or inline, hence causing multiple
// conflicting definitions at link-time.
namespace buminiz {
    typedef unsigned long mz_ulong;
    enum { MZ_OK = 0, MZ_STREAM_END = 1, MZ_NEED_DICT = 2, MZ_ERRNO = -1, MZ_STREAM_ERROR = -2, MZ_DATA_ERROR = -3, MZ_MEM_ERROR = -4, MZ_BUF_ERROR = -5, MZ_VERSION_ERROR = -6, MZ_PARAM_ERROR = -10000 };
    mz_ulong mz_compressBound(mz_ulong source_len);
    int mz_compress2(unsigned char *pDest, mz_ulong *pDest_len, const unsigned char *pSource, mz_ulong source_len, int level);
    int mz_uncompress(unsigned char *pDest, mz_ulong *pDest_len, const unsigned char *pSource, mz_ulong source_len);
}
#endif

using namespace buminiz;

extern "C" {

/**
 * @internal
 * @~English
 * @brief Returns upper bound for compresses data using miniz (ZLIB).
 *
 * @param srcLength source data length
 *
 * @author Daniel Rakos, RasterGrid
 */
ktx_size_t ktxCompressZLIBBounds(ktx_size_t srcLength) {
    return mz_compressBound((mz_ulong)srcLength);
}

/**
 * @internal
 * @~English
 * @brief Compresses data using miniz (ZLIB)
 *
 * @param pDest         destination data buffer
 * @param pDestLength   destination data buffer size
 *                      (filled with written byte count on success)
 * @param pSrc          source data buffer
 * @param srcLength     source data size
 * @param level         compression level (between 1 and 9)
 *
 * @author Daniel Rakos, RasterGrid
 */
KTX_error_code ktxCompressZLIBInt(unsigned char* pDest,
                                  ktx_size_t* pDestLength,
                                  const unsigned char* pSrc,
                                  ktx_size_t srcLength,
                                  ktx_uint32_t level) {
    if ((srcLength | *pDestLength) > 0xFFFFFFFFU) return KTX_INVALID_VALUE;
    mz_ulong mzCompressedSize = (mz_ulong)*pDestLength;
    int status = mz_compress2(pDest, &mzCompressedSize, pSrc, (mz_ulong)srcLength, level);
    switch (status) {
    case MZ_OK:
        *pDestLength = mzCompressedSize;
        return KTX_SUCCESS;
    case MZ_PARAM_ERROR:
        return KTX_INVALID_VALUE;
    case MZ_BUF_ERROR:
#ifdef DEBUG
        assert(false && "Deflate dstSize too small.");
#endif
        return KTX_OUT_OF_MEMORY;
    case MZ_MEM_ERROR:
#ifdef DEBUG
        assert(false && "Deflate workspace too small.");
#endif
        return KTX_OUT_OF_MEMORY;
    default:
        // The remaining errors look like they should only
        // occur during decompression but just in case.
#ifdef DEBUG
        assert(true);
#endif
        return KTX_INVALID_OPERATION;
    }
}

/**
 * @internal
 * @~English
 * @brief Uncompresses data using miniz (ZLIB)
 *
 * @param pDest         destination data buffer
 * @param pDestLength   destination data buffer size
 *                      (filled with written byte count on success)
 * @param pSrc          source data buffer
 * @param srcLength     source data size
 *
 * @author Daniel Rakos, RasterGrid
 */
KTX_error_code ktxUncompressZLIBInt(unsigned char* pDest,
                                    ktx_size_t* pDestLength,
                                    const unsigned char* pSrc,
                                    ktx_size_t srcLength) {
    if ((srcLength | *pDestLength) > 0xFFFFFFFFU) return KTX_INVALID_VALUE;
    mz_ulong mzUncompressedSize = (mz_ulong)*pDestLength;
    int status = mz_uncompress(pDest, &mzUncompressedSize, pSrc, (mz_ulong)srcLength);
    switch (status) {
    case MZ_OK:
        *pDestLength = mzUncompressedSize;
        return KTX_SUCCESS;
    case MZ_BUF_ERROR:
        return KTX_DECOMPRESS_LENGTH_ERROR; // buffer too small
    case MZ_MEM_ERROR:
        return KTX_OUT_OF_MEMORY;
    default:
        return KTX_FILE_DATA_ERROR;
    }
}

}
