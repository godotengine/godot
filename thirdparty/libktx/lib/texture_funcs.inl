/* -*- tab-width: 4; -*- */
/* vi: set sw=2 ts=4 expandtab textwidth=70: */

/*
 * Copyright 2019-2020 The Khronos Group Inc.
 * SPDX-License-Identifier: Apache-2.0
 */

/**
 * @internal
 * @file texture_funcs.h
 * @~English
 *
 * @brief Templates for functions common to base & derived ktxTexture classes.
 *
 * Define CLASS before including this file.
 */

#define CAT(c, n) PRIMITIVE_CAT(c, n)
#define PRIMITIVE_CAT(c, n) c ## _ ## n

#define CLASS_FUNC(name) CAT(CLASS, name)

/*
 ======================================
     Virtual ktxTexture functions
 ======================================
*/


void CLASS_FUNC(Destroy)(CLASS* This);
KTX_error_code CLASS_FUNC(GetImageOffset)(CLASS* This, ktx_uint32_t level,
                                          ktx_uint32_t layer,
                                          ktx_uint32_t faceSlice,
                                          ktx_size_t* pOffset);
ktx_size_t CLASS_FUNC(GetImageSize)(CLASS* This, ktx_uint32_t level);
KTX_error_code CLASS_FUNC(GLUpload)(CLASS* This, GLuint* pTexture,
                                    GLenum* pTarget, GLenum* pGlerror);
KTX_error_code CLASS_FUNC(IterateLevels)(CLASS* This,
                                         PFNKTXITERCB iterCb,
                                         void* userdata);
KTX_error_code CLASS_FUNC(IterateLevelFaces)(CLASS* This,
                                             PFNKTXITERCB iterCb,
                                             void* userdata);
KTX_error_code CLASS_FUNC(IterateLoadLevelFaces)(CLASS* This,
                                                 PFNKTXITERCB iterCb,
                                                 void* userdata);
KTX_error_code CLASS_FUNC(LoadImageData)(CLASS* This,
                                         ktx_uint8_t* pBuffer,
                                         ktx_size_t bufSize);
KTX_error_code CLASS_FUNC(SetImageFromStdioStream)(CLASS* This,
                                    ktx_uint32_t level,ktx_uint32_t layer,
                                    ktx_uint32_t faceSlice,
                                    FILE* src, ktx_size_t srcSize);
KTX_error_code CLASS_FUNC(SetImageFromMemory)(CLASS* This,
                               ktx_uint32_t level, ktx_uint32_t layer,
                               ktx_uint32_t faceSlice,
                               const ktx_uint8_t* src, ktx_size_t srcSize);

KTX_error_code CLASS_FUNC(WriteToStdioStream)(CLASS* This, FILE* dstsstr);
KTX_error_code CLASS_FUNC(WriteToNamedFile)(CLASS* This,
                                             const char* const dstname);
KTX_error_code CLASS_FUNC(WriteToMemory)(CLASS* This,
                          ktx_uint8_t** ppDstBytes, ktx_size_t* pSize);
KTX_error_code CLASS_FUNC(WriteToStream)(CLASS* This,
                          ktxStream* dststr);

/*
 ======================================
     Internal ktxTexture functions
 ======================================
*/


void CLASS_FUNC(destruct)(CLASS* This);

