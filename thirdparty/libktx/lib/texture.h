/* -*- tab-width: 4; -*- */
/* vi: set sw=2 ts=4 expandtab textwidth=70: */

/*
 * Copyright 2019-2020 The Khronos Group Inc.
 * SPDX-License-Identifier: Apache-2.0
 */

/**
 * @internal
 * @file texture.h
 * @~English
 *
 * @brief Declare internal ktxTexture functions for sharing between
 *        compilation units.
 *
 * These functions are private and should not be used outside the library.
 */

#ifndef _TEXTURE_H_
#define _TEXTURE_H_

#include "ktx.h"
#include "formatsize.h"

#define DECLARE_PRIVATE(class) class ## _private* private = This->_private
#define DECLARE_PROTECTED(class) class ## _protected* prtctd = This->_protected;

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    KTX_FORMAT_VERSION_ONE = 1,
    KTX_FORMAT_VERSION_TWO = 2
} ktxFormatVersionEnum;

typedef ktx_size_t (* PFNCALCDATASIZELEVELS)(ktxTexture* This,
                                             ktx_uint32_t levels);
typedef ktx_size_t (* PFNCALCFACELODSIZE)(ktxTexture* This, ktx_uint32_t level);
typedef ktx_size_t (* PFNCALCLEVELOFFSET)(ktxTexture* This, ktx_uint32_t level);
typedef struct ktxTexture_vtblInt {
    PFNCALCDATASIZELEVELS calcDataSizeLevels;
    PFNCALCFACELODSIZE calcFaceLodSize;
    PFNCALCLEVELOFFSET calcLevelOffset;
} ktxTexture_vtblInt;

#define ktxTexture_calcDataSizeLevels(This, levels) \
            This->_protected->_vtbl.calcDataSizeLevels(This, levels);
#define ktxTexture_calcFaceLodSize(This, level) \
            This->_protected->_vtbl.calcFaceLodSize(This, level);
#define ktxTexture_calcLevelOffset(This, level) \
            This->_protected->_vtbl.calcLevelOffset(This, level);

/**
 * @memberof ktxTexture
 * @~English
 *
 * @brief protected members of ktxTexture.
 */
typedef struct ktxTexture_protected {
    ktxTexture_vtblInt _vtbl;
    ktxFormatSize _formatSize;
    ktx_uint32_t _typeSize;
    ktxStream _stream;
} ktxTexture_protected;

#define ktxTexture_getStream(t) ((ktxStream*)(&(t)->_protected->_stream))
#define ktxTexture1_getStream(t1) ktxTexture_getStream((ktxTexture*)t1)
#define ktxTexture2_getStream(t2) ktxTexture_getStream((ktxTexture*)t2)

KTX_error_code
ktxTexture_iterateLoadedImages(ktxTexture* This, PFNKTXITERCB iterCb,
                               void* userdata);
KTX_error_code
ktxTexture_iterateSourceImages(ktxTexture* This, PFNKTXITERCB iterCb,
                               void* userdata);

ktx_size_t ktxTexture_calcDataSizeTexture(ktxTexture* This);
ktx_size_t ktxTexture_calcImageSize(ktxTexture* This, ktx_uint32_t level,
                                    ktxFormatVersionEnum fv);
ktx_bool_t ktxTexture_isActiveStream(ktxTexture* This);
ktx_size_t ktxTexture_calcLevelSize(ktxTexture* This, ktx_uint32_t level,
                                    ktxFormatVersionEnum fv);
ktx_size_t ktxTexture_doCalcFaceLodSize(ktxTexture* This, ktx_uint32_t level,
                                        ktxFormatVersionEnum fv);
ktx_size_t ktxTexture_layerSize(ktxTexture* This, ktx_uint32_t level,
                                ktxFormatVersionEnum fv);
void ktxTexture_rowInfo(ktxTexture* This, ktx_uint32_t level,
                        ktx_uint32_t* numRows, ktx_uint32_t* rowBytes,
                        ktx_uint32_t* rowPadding);
KTX_error_code
ktxTexture_construct(ktxTexture* This, ktxTextureCreateInfo* createInfo,
                     ktxFormatSize* formatSize);

KTX_error_code
ktxTexture_constructFromStream(ktxTexture* This, ktxStream* pStream,
                               ktxTextureCreateFlags createFlags);

void
ktxTexture_destruct(ktxTexture* This);

#ifdef __cplusplus
}
#endif

#endif /* _TEXTURE_H_ */
