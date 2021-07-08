//
// Copyright 2002 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//

// generatemip.h: Defines the GenerateMip function, templated on the format
// type of the image for which mip levels are being generated.

#ifndef IMAGEUTIL_GENERATEMIP_H_
#define IMAGEUTIL_GENERATEMIP_H_

#include <stddef.h>
#include <stdint.h>

namespace angle
{

template <typename T>
inline void GenerateMip(size_t sourceWidth,
                        size_t sourceHeight,
                        size_t sourceDepth,
                        const uint8_t *sourceData,
                        size_t sourceRowPitch,
                        size_t sourceDepthPitch,
                        uint8_t *destData,
                        size_t destRowPitch,
                        size_t destDepthPitch);

}  // namespace angle

#include "generatemip.inc"

#endif  // IMAGEUTIL_GENERATEMIP_H_
