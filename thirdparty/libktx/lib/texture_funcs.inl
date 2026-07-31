/* -*- tab-width: 4; -*- */
/* vi: set sw=2 ts=4 expandtab textwidth=70: */

/*
 * Copyright 2019-2020 The Khronos Group Inc.
 * SPDX-License-Identifier: Apache-2.0
 */

/**
 * @internal
 * @file
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
KTX_error_code CLASS_FUNC(SetImageFromStdioStream)(CLASS* This,
                                    ktx_uint32_t level,ktx_uint32_t layer,
                                    ktx_uint32_t faceSlice,
                                    FILE* src, ktx_size_t srcSize);
KTX_error_code CLASS_FUNC(SetImageFromMemory)(CLASS* This,
                               ktx_uint32_t level, ktx_uint32_t layer,
                               ktx_uint32_t faceSlice,
                               const ktx_uint8_t* src, ktx_size_t srcSize);


/*
 ======================================
     Internal ktxTexture functions
 ======================================
*/


void CLASS_FUNC(destruct)(CLASS* This);

