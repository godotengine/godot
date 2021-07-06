//
// Copyright 2013 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//

// copyvertex.h: Defines vertex buffer copying and conversion functions

#ifndef LIBANGLE_RENDERER_COPYVERTEX_H_
#define LIBANGLE_RENDERER_COPYVERTEX_H_

#include "common/mathutil.h"

namespace rx
{

using VertexCopyFunction = void (*)(const uint8_t *input,
                                    size_t stride,
                                    size_t count,
                                    uint8_t *output);

// 'alphaDefaultValueBits' gives the default value for the alpha channel (4th component)
template <typename T,
          size_t inputComponentCount,
          size_t outputComponentCount,
          uint32_t alphaDefaultValueBits>
void CopyNativeVertexData(const uint8_t *input, size_t stride, size_t count, uint8_t *output);

template <size_t inputComponentCount, size_t outputComponentCount>
void Copy8SintTo16SintVertexData(const uint8_t *input,
                                 size_t stride,
                                 size_t count,
                                 uint8_t *output);

template <size_t componentCount>
void Copy8SnormTo16SnormVertexData(const uint8_t *input,
                                   size_t stride,
                                   size_t count,
                                   uint8_t *output);

template <size_t inputComponentCount, size_t outputComponentCount>
void Copy32FixedTo32FVertexData(const uint8_t *input, size_t stride, size_t count, uint8_t *output);

template <typename T, size_t inputComponentCount, size_t outputComponentCount, bool normalized>
void CopyTo32FVertexData(const uint8_t *input, size_t stride, size_t count, uint8_t *output);

template <bool isSigned, bool normalized, bool toFloat>
void CopyXYZ10W2ToXYZW32FVertexData(const uint8_t *input,
                                    size_t stride,
                                    size_t count,
                                    uint8_t *output);

template <bool isSigned, bool normalized>
void CopyXYZ10ToXYZW32FVertexData(const uint8_t *input,
                                  size_t stride,
                                  size_t count,
                                  uint8_t *output);

template <bool isSigned, bool normalized>
void CopyW2XYZ10ToXYZW32FVertexData(const uint8_t *input,
                                    size_t stride,
                                    size_t count,
                                    uint8_t *output);

}  // namespace rx

#include "copyvertex.inc.h"

#endif  // LIBANGLE_RENDERER_COPYVERTEX_H_
