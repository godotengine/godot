//
// Copyright 2013 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//

// copyimage.cpp: Defines image copying functions

#include "image_util/copyimage.h"

namespace angle
{

void CopyBGRA8ToRGBA8(const uint8_t *source, uint8_t *dest)
{
    uint32_t argb                       = *reinterpret_cast<const uint32_t *>(source);
    *reinterpret_cast<uint32_t *>(dest) = (argb & 0xFF00FF00) |        // Keep alpha and green
                                          (argb & 0x00FF0000) >> 16 |  // Move red to blue
                                          (argb & 0x000000FF) << 16;   // Move blue to red
}

}  // namespace angle
