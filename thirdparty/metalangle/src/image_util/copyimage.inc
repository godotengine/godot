//
// Copyright 2014 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//

// copyimage.inc: Defines image copying functions

namespace angle
{
template <typename sourceType, typename colorDataType>
void ReadColor(const uint8_t *source, uint8_t *dest)
{
    sourceType::readColor(reinterpret_cast<Color<colorDataType>*>(dest),
                          reinterpret_cast<const sourceType*>(source));
}

template <typename destType, typename colorDataType>
void WriteColor(const uint8_t *source, uint8_t *dest)
{
    destType::writeColor(reinterpret_cast<destType*>(dest),
                         reinterpret_cast<const Color<colorDataType>*>(source));
}

template <typename SourceType>
void ReadDepthStencil(const uint8_t *source, uint8_t *dest)
{
    SourceType::ReadDepthStencil(reinterpret_cast<DepthStencil *>(dest),
                                 reinterpret_cast<const SourceType *>(source));
}

template <typename DestType>
void WriteDepthStencil(const uint8_t *source, uint8_t *dest)
{
    DestType::WriteDepthStencil(reinterpret_cast<DestType *>(dest),
                                reinterpret_cast<const DepthStencil *>(source));
}
}  // namespace angle
