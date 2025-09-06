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
 * @brief Declare internal ktxTexture1 functions for sharing between
 *        compilation units.
 *
 * These functions are private and should not be used outside the library.
 */

#ifndef _TEXTURE1_H_
#define _TEXTURE1_H_

#include "texture.h"

#ifdef __cplusplus
extern "C" {
#endif

#define CLASS ktxTexture1
#include "texture_funcs.inl"
#undef CLASS

KTX_error_code
ktxTexture1_constructFromStreamAndHeader(ktxTexture1* This, ktxStream* pStream,
                                         KTX_header* pHeader,
                                         ktxTextureCreateFlags createFlags);

ktx_uint64_t ktxTexture1_calcDataSizeTexture(ktxTexture1* This);
ktx_size_t ktxTexture1_calcLevelOffset(ktxTexture1* This, ktx_uint32_t level);
ktx_uint32_t ktxTexture1_glTypeSize(ktxTexture1* This);

#ifdef __cplusplus
}
#endif

#endif /* _TEXTURE1_H_ */
