/* -*- tab-width: 4; -*- */
/* vi: set sw=2 ts=4 expandtab: */

/* $Id$ */

/*
 * Copyright 2010-2020 The Khronos Group Inc.
 * SPDX-License-Identifier: Apache-2.0
 */


/*
 * Author: Mark Callow from original code by Georg Kolling
 */

#ifndef KTXINT_H
#define KTXINT_H

#include <math.h>

/* Define this to include the ETC unpack software in the library. */
#ifndef SUPPORT_SOFTWARE_ETC_UNPACK
  /* Include for all GL versions because have seen OpenGL ES 3
   * implementaions that do not support ETC1 (ARM Mali emulator v1.0)!
   */
  #define SUPPORT_SOFTWARE_ETC_UNPACK 1
#endif

#ifndef MAX
#define MAX(x, y) (((x) > (y)) ? (x) : (y))
#endif
#ifndef MIN
#define MIN(x, y) (((x) < (y)) ? (x) : (y))
#endif

#define QUOTE(x) #x
#define STR(x) QUOTE(x)

#define KTX2_IDENTIFIER_REF  { 0xAB, 0x4B, 0x54, 0x58, 0x20, 0x32, 0x30, 0xBB, 0x0D, 0x0A, 0x1A, 0x0A }
#define KTX2_HEADER_SIZE     (80)

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @internal
 * @brief used to pass GL context capabilites to subroutines.
 */
#define _KTX_NO_R16_FORMATS     0x0
#define _KTX_R16_FORMATS_NORM   0x1
#define _KTX_R16_FORMATS_SNORM  0x2
#define _KTX_ALL_R16_FORMATS (_KTX_R16_FORMATS_NORM | _KTX_R16_FORMATS_SNORM)
extern GLint _ktxR16Formats;
extern GLboolean _ktxSupportsSRGB;

/**
 * @internal
 * @~English
 * @brief KTX file header.
 *
 * See the KTX specification for descriptions.
 */
typedef struct KTX_header {
    ktx_uint8_t  identifier[12];
    ktx_uint32_t endianness;
    ktx_uint32_t glType;
    ktx_uint32_t glTypeSize;
    ktx_uint32_t glFormat;
    ktx_uint32_t glInternalformat;
    ktx_uint32_t glBaseInternalformat;
    ktx_uint32_t pixelWidth;
    ktx_uint32_t pixelHeight;
    ktx_uint32_t pixelDepth;
    ktx_uint32_t numberOfArrayElements;
    ktx_uint32_t numberOfFaces;
    ktx_uint32_t numberOfMipLevels;
    ktx_uint32_t bytesOfKeyValueData;
} KTX_header;

/* This will cause compilation to fail if the struct size doesn't match */
typedef int KTX_header_SIZE_ASSERT [sizeof(KTX_header) == KTX_HEADER_SIZE];

/**
 * @internal
 * @~English
 * @brief 32-bit KTX 2 index entry.
 */
typedef struct ktxIndexEntry32 {
    ktx_uint32_t byteOffset; /*!< Offset of item from start of file. */
    ktx_uint32_t byteLength; /*!< Number of bytes of data in the item. */
} ktxIndexEntry32;
/**
 * @internal
 * @~English
 * @brief 64-bit KTX 2 index entry.
 */
typedef struct ktxIndexEntry64 {
    ktx_uint64_t byteOffset; /*!< Offset of item from start of file. */
    ktx_uint64_t byteLength; /*!< Number of bytes of data in the item. */
} ktxIndexEntry64;

/**
 * @internal
 * @~English
 * @brief KTX 2 file header.
 *
 * See the KTX 2 specification for descriptions.
 */
typedef struct KTX_header2 {
    ktx_uint8_t  identifier[12];
    ktx_uint32_t vkFormat;
    ktx_uint32_t typeSize;
    ktx_uint32_t pixelWidth;
    ktx_uint32_t pixelHeight;
    ktx_uint32_t pixelDepth;
    ktx_uint32_t layerCount;
    ktx_uint32_t faceCount;
    ktx_uint32_t levelCount;
    ktx_uint32_t supercompressionScheme;
    ktxIndexEntry32 dataFormatDescriptor;
    ktxIndexEntry32 keyValueData;
    ktxIndexEntry64 supercompressionGlobalData;
} KTX_header2;

/* This will cause compilation to fail if the struct size doesn't match */
typedef int KTX_header2_SIZE_ASSERT [sizeof(KTX_header2) == KTX2_HEADER_SIZE];

/**
 * @internal
 * @~English
 * @brief KTX 2 level index entry.
 */
typedef struct ktxLevelIndexEntry {
    ktx_uint64_t byteOffset; /*!< Offset of level from start of file. */
    ktx_uint64_t byteLength;
                /*!< Number of bytes of compressed image data in the level. */
    ktx_uint64_t uncompressedByteLength;
                /*!< Number of bytes of uncompressed image data in the level. */
} ktxLevelIndexEntry;

/**
 * @internal
 * @~English
 * @brief Structure for supplemental information about the texture.
 *
 * _ktxCheckHeader returns supplemental information about the texture in this
 * structure that is derived during checking of the file header.
 */
typedef struct KTX_supplemental_info
{
    ktx_uint8_t compressed;
    ktx_uint8_t generateMipmaps;
    ktx_uint16_t textureDimension;
} KTX_supplemental_info;
/**
 * @internal
 * @var ktx_uint8_t KTX_supplemental_info::compressed
 * @~English
 * @brief KTX_TRUE, if this a compressed texture, KTX_FALSE otherwise?
 */
/**
 * @internal
 * @var ktx_uint8_t KTX_supplemental_info::generateMipmaps
 * @~English
 * @brief KTX_TRUE, if mipmap generation is required, KTX_FALSE otherwise.
 */
/**
 * @internal
 * @var ktx_uint16_t KTX_supplemental_info::textureDimension
 * @~English
 * @brief The number of dimensions, 1, 2 or 3, of data in the texture image.
 */

/*
 * @internal
 * CheckHeader1
 *
 * Reads the KTX file header and performs some sanity checking on the values
 */
KTX_error_code ktxCheckHeader1_(KTX_header* pHeader,
                                KTX_supplemental_info* pSuppInfo);

/*
 * @internal
 * CheckHeader2
 *
 * Reads the KTX 2 file header and performs some sanity checking on the values
 */
KTX_error_code ktxCheckHeader2_(KTX_header2* pHeader,
                                KTX_supplemental_info* pSuppInfo);

/*
 * SwapEndian16: Swaps endianness in an array of 16-bit values
 */
void _ktxSwapEndian16(ktx_uint16_t* pData16, ktx_size_t count);

/*
 * SwapEndian32: Swaps endianness in an array of 32-bit values
 */
void _ktxSwapEndian32(ktx_uint32_t* pData32, ktx_size_t count);

/*
 * SwapEndian32: Swaps endianness in an array of 64-bit values
 */
void _ktxSwapEndian64(ktx_uint64_t* pData64, ktx_size_t count);

/*
 * UnpackETC: uncompresses an ETC compressed texture image
 */
KTX_error_code _ktxUnpackETC(const GLubyte* srcETC, const GLenum srcFormat,
                             ktx_uint32_t active_width, ktx_uint32_t active_height,
                             GLubyte** dstImage,
                             GLenum* format, GLenum* internalFormat, GLenum* type,
                             GLint R16Formats, GLboolean supportsSRGB);

/*
 * @internal
 * ktxCompressZLIBBounds
 *
 * Returns upper bound for compresses data using miniz (ZLIB)
 */
ktx_size_t ktxCompressZLIBBounds(ktx_size_t srcLength);

/*
 * @internal
 * ktxCompressZLIBInt
 *
 * Compresses data using miniz (ZLIB)
 */
KTX_error_code ktxCompressZLIBInt(unsigned char* pDest,
                                  ktx_size_t* pDestLength,
                                  const unsigned char* pSrc,
                                  ktx_size_t srcLength,
                                  ktx_uint32_t level);

/*
 * @internal
 * ktxUncompressZLIBInt
 *
 * Uncompresses data using miniz (ZLIB)
 */
KTX_error_code ktxUncompressZLIBInt(unsigned char* pDest,
                                    ktx_size_t* pDestLength,
                                    const unsigned char* pSrc,
                                    ktx_size_t srcLength);

/*
 * Pad nbytes to next multiple of n
 */
#define _KTX_PADN(n, nbytes) (ktx_uint32_t)(n * ceilf((float)(nbytes) / n))
/*
 * Calculate bytes of of padding needed to reach next multiple of n.
 */
/* Equivalent to (n * ceil(nbytes / n)) - nbytes */
#define _KTX_PADN_LEN(n, nbytes) \
    (ktx_uint32_t)((n * ceilf((float)(nbytes) / n)) - (nbytes))

/*
 * Pad nbytes to next multiple of 4
 */
#define _KTX_PAD4(nbytes) _KTX_PADN(4, nbytes)
/*
 * Calculate bytes of of padding needed to reach next multiple of 4.
 */
#define _KTX_PAD4_LEN(nbytes) _KTX_PADN_LEN(4, nbytes)

/*
 * Pad nbytes to next multiple of 8
 */
#define _KTX_PAD8(nbytes) _KTX_PADN(8, nbytes)
/*
 * Calculate bytes of of padding needed to reach next multiple of 8.
 */
#define _KTX_PAD8_LEN(nbytes) _KTX_PADN_LEN(8, nbytes)

/*
 * Pad nbytes to KTX_GL_UNPACK_ALIGNMENT
 */
#define _KTX_PAD_UNPACK_ALIGN(nbytes)  \
        _KTX_PADN(KTX_GL_UNPACK_ALIGNMENT, nbytes)
/*
 * Calculate bytes of of padding needed to reach KTX_GL_UNPACK_ALIGNMENT.
 */
#define _KTX_PAD_UNPACK_ALIGN_LEN(nbytes)  \
        _KTX_PADN_LEN(KTX_GL_UNPACK_ALIGNMENT, nbytes)

/*
 ======================================
     Internal utility functions
 ======================================
*/

KTX_error_code printKTX2Info2(ktxStream* src, KTX_header2* header);

/*
 * fopen a file identified by a UTF-8 path.
 */
#if defined(_WIN32)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <assert.h>
#include <windows.h>
#include <shellapi.h>
#include <stdlib.h>

// For Windows, we convert the UTF-8 path and mode to UTF-16 path and use
// _wfopen which correctly handles unicode characters.
static inline FILE* ktxFOpenUTF8(char const* path, char const* mode) {
    int wpLen = MultiByteToWideChar(CP_UTF8, 0, path, -1, NULL, 0);
    int wmLen = MultiByteToWideChar(CP_UTF8, 0, mode, -1, NULL, 0);
    FILE* fp = NULL;
    if (wpLen > 0 && wmLen > 0)
    {
        wchar_t* wpath = (wchar_t*)malloc(wpLen * sizeof(wchar_t));
        wchar_t* wmode = (wchar_t*)malloc(wmLen * sizeof(wchar_t));
        MultiByteToWideChar(CP_UTF8, 0, path, -1, wpath, wpLen);
        MultiByteToWideChar(CP_UTF8, 0, mode, -1, wmode, wmLen);
        // Returned errno_t value is also set in the global errno.
        // Apps use that for error detail as libktx only returns
        // KTX_FILE_OPEN_FAILED.
        (void)_wfopen_s(&fp, wpath, wmode);
        free(wpath);
        free(wmode);
        return fp;
    } else {
        assert(KTX_FALSE
               && "ktxFOpenUTF8 called with zero length path or mode.");
        return NULL;
    }
}
#else
// For other platforms there is no need for any conversion, they
// support UTF-8 natively.
static inline FILE* ktxFOpenUTF8(char const* path, char const* mode) {
    return fopen(path, mode);
}
#endif

#ifdef __cplusplus
}
#endif

#endif /* KTXINT_H */
