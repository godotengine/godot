//
// Copyright 2013 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//

// loadimage.h: Defines image loading functions

#ifndef IMAGEUTIL_LOADIMAGE_H_
#define IMAGEUTIL_LOADIMAGE_H_

#include <stddef.h>
#include <stdint.h>

namespace angle
{

void LoadA8ToRGBA8(size_t width,
                   size_t height,
                   size_t depth,
                   const uint8_t *input,
                   size_t inputRowPitch,
                   size_t inputDepthPitch,
                   uint8_t *output,
                   size_t outputRowPitch,
                   size_t outputDepthPitch);

void LoadA8ToBGRA8(size_t width,
                   size_t height,
                   size_t depth,
                   const uint8_t *input,
                   size_t inputRowPitch,
                   size_t inputDepthPitch,
                   uint8_t *output,
                   size_t outputRowPitch,
                   size_t outputDepthPitch);

void LoadA32FToRGBA32F(size_t width,
                       size_t height,
                       size_t depth,
                       const uint8_t *input,
                       size_t inputRowPitch,
                       size_t inputDepthPitch,
                       uint8_t *output,
                       size_t outputRowPitch,
                       size_t outputDepthPitch);

void LoadA16FToRGBA16F(size_t width,
                       size_t height,
                       size_t depth,
                       const uint8_t *input,
                       size_t inputRowPitch,
                       size_t inputDepthPitch,
                       uint8_t *output,
                       size_t outputRowPitch,
                       size_t outputDepthPitch);

void LoadL8ToRGBA8(size_t width,
                   size_t height,
                   size_t depth,
                   const uint8_t *input,
                   size_t inputRowPitch,
                   size_t inputDepthPitch,
                   uint8_t *output,
                   size_t outputRowPitch,
                   size_t outputDepthPitch);

void LoadL8ToBGRA8(size_t width,
                   size_t height,
                   size_t depth,
                   const uint8_t *input,
                   size_t inputRowPitch,
                   size_t inputDepthPitch,
                   uint8_t *output,
                   size_t outputRowPitch,
                   size_t outputDepthPitch);

void LoadL32FToRGBA32F(size_t width,
                       size_t height,
                       size_t depth,
                       const uint8_t *input,
                       size_t inputRowPitch,
                       size_t inputDepthPitch,
                       uint8_t *output,
                       size_t outputRowPitch,
                       size_t outputDepthPitch);

void LoadL16FToRGBA16F(size_t width,
                       size_t height,
                       size_t depth,
                       const uint8_t *input,
                       size_t inputRowPitch,
                       size_t inputDepthPitch,
                       uint8_t *output,
                       size_t outputRowPitch,
                       size_t outputDepthPitch);

void LoadLA8ToRGBA8(size_t width,
                    size_t height,
                    size_t depth,
                    const uint8_t *input,
                    size_t inputRowPitch,
                    size_t inputDepthPitch,
                    uint8_t *output,
                    size_t outputRowPitch,
                    size_t outputDepthPitch);

void LoadLA8ToBGRA8(size_t width,
                    size_t height,
                    size_t depth,
                    const uint8_t *input,
                    size_t inputRowPitch,
                    size_t inputDepthPitch,
                    uint8_t *output,
                    size_t outputRowPitch,
                    size_t outputDepthPitch);

void LoadLA32FToRGBA32F(size_t width,
                        size_t height,
                        size_t depth,
                        const uint8_t *input,
                        size_t inputRowPitch,
                        size_t inputDepthPitch,
                        uint8_t *output,
                        size_t outputRowPitch,
                        size_t outputDepthPitch);

void LoadLA16FToRGBA16F(size_t width,
                        size_t height,
                        size_t depth,
                        const uint8_t *input,
                        size_t inputRowPitch,
                        size_t inputDepthPitch,
                        uint8_t *output,
                        size_t outputRowPitch,
                        size_t outputDepthPitch);

void LoadRGB8ToBGR565(size_t width,
                      size_t height,
                      size_t depth,
                      const uint8_t *input,
                      size_t inputRowPitch,
                      size_t inputDepthPitch,
                      uint8_t *output,
                      size_t outputRowPitch,
                      size_t outputDepthPitch);

void LoadRGB565ToBGR565(size_t width,
                        size_t height,
                        size_t depth,
                        const uint8_t *input,
                        size_t inputRowPitch,
                        size_t inputDepthPitch,
                        uint8_t *output,
                        size_t outputRowPitch,
                        size_t outputDepthPitch);

void LoadRGB8ToBGRX8(size_t width,
                     size_t height,
                     size_t depth,
                     const uint8_t *input,
                     size_t inputRowPitch,
                     size_t inputDepthPitch,
                     uint8_t *output,
                     size_t outputRowPitch,
                     size_t outputDepthPitch);

void LoadRG8ToBGRX8(size_t width,
                    size_t height,
                    size_t depth,
                    const uint8_t *input,
                    size_t inputRowPitch,
                    size_t inputDepthPitch,
                    uint8_t *output,
                    size_t outputRowPitch,
                    size_t outputDepthPitch);

void LoadR8ToBGRX8(size_t width,
                   size_t height,
                   size_t depth,
                   const uint8_t *input,
                   size_t inputRowPitch,
                   size_t inputDepthPitch,
                   uint8_t *output,
                   size_t outputRowPitch,
                   size_t outputDepthPitch);

void LoadR5G6B5ToBGRA8(size_t width,
                       size_t height,
                       size_t depth,
                       const uint8_t *input,
                       size_t inputRowPitch,
                       size_t inputDepthPitch,
                       uint8_t *output,
                       size_t outputRowPitch,
                       size_t outputDepthPitch);

void LoadR5G6B5ToRGBA8(size_t width,
                       size_t height,
                       size_t depth,
                       const uint8_t *input,
                       size_t inputRowPitch,
                       size_t inputDepthPitch,
                       uint8_t *output,
                       size_t outputRowPitch,
                       size_t outputDepthPitch);

void LoadRGBA8ToBGRA8(size_t width,
                      size_t height,
                      size_t depth,
                      const uint8_t *input,
                      size_t inputRowPitch,
                      size_t inputDepthPitch,
                      uint8_t *output,
                      size_t outputRowPitch,
                      size_t outputDepthPitch);

void LoadRGBA8ToBGRA4(size_t width,
                      size_t height,
                      size_t depth,
                      const uint8_t *input,
                      size_t inputRowPitch,
                      size_t inputDepthPitch,
                      uint8_t *output,
                      size_t outputRowPitch,
                      size_t outputDepthPitch);

void LoadRGBA4ToARGB4(size_t width,
                      size_t height,
                      size_t depth,
                      const uint8_t *input,
                      size_t inputRowPitch,
                      size_t inputDepthPitch,
                      uint8_t *output,
                      size_t outputRowPitch,
                      size_t outputDepthPitch);

void LoadRGBA4ToBGRA8(size_t width,
                      size_t height,
                      size_t depth,
                      const uint8_t *input,
                      size_t inputRowPitch,
                      size_t inputDepthPitch,
                      uint8_t *output,
                      size_t outputRowPitch,
                      size_t outputDepthPitch);

void LoadRGBA4ToRGBA8(size_t width,
                      size_t height,
                      size_t depth,
                      const uint8_t *input,
                      size_t inputRowPitch,
                      size_t inputDepthPitch,
                      uint8_t *output,
                      size_t outputRowPitch,
                      size_t outputDepthPitch);

void LoadBGRA4ToBGRA8(size_t width,
                      size_t height,
                      size_t depth,
                      const uint8_t *input,
                      size_t inputRowPitch,
                      size_t inputDepthPitch,
                      uint8_t *output,
                      size_t outputRowPitch,
                      size_t outputDepthPitch);

void LoadRGBA8ToBGR5A1(size_t width,
                       size_t height,
                       size_t depth,
                       const uint8_t *input,
                       size_t inputRowPitch,
                       size_t inputDepthPitch,
                       uint8_t *output,
                       size_t outputRowPitch,
                       size_t outputDepthPitch);

void LoadRGB10A2ToBGR5A1(size_t width,
                         size_t height,
                         size_t depth,
                         const uint8_t *input,
                         size_t inputRowPitch,
                         size_t inputDepthPitch,
                         uint8_t *output,
                         size_t outputRowPitch,
                         size_t outputDepthPitch);

void LoadRGB5A1ToA1RGB5(size_t width,
                        size_t height,
                        size_t depth,
                        const uint8_t *input,
                        size_t inputRowPitch,
                        size_t inputDepthPitch,
                        uint8_t *output,
                        size_t outputRowPitch,
                        size_t outputDepthPitch);

void LoadRGB5A1ToBGRA8(size_t width,
                       size_t height,
                       size_t depth,
                       const uint8_t *input,
                       size_t inputRowPitch,
                       size_t inputDepthPitch,
                       uint8_t *output,
                       size_t outputRowPitch,
                       size_t outputDepthPitch);

void LoadRGB5A1ToRGBA8(size_t width,
                       size_t height,
                       size_t depth,
                       const uint8_t *input,
                       size_t inputRowPitch,
                       size_t inputDepthPitch,
                       uint8_t *output,
                       size_t outputRowPitch,
                       size_t outputDepthPitch);

void LoadBGR5A1ToBGRA8(size_t width,
                       size_t height,
                       size_t depth,
                       const uint8_t *input,
                       size_t inputRowPitch,
                       size_t inputDepthPitch,
                       uint8_t *output,
                       size_t outputRowPitch,
                       size_t outputDepthPitch);

void LoadRGB10A2ToRGBA8(size_t width,
                        size_t height,
                        size_t depth,
                        const uint8_t *input,
                        size_t inputRowPitch,
                        size_t inputDepthPitch,
                        uint8_t *output,
                        size_t outputRowPitch,
                        size_t outputDepthPitch);

void LoadRGB10A2ToRGB10X2(size_t width,
                          size_t height,
                          size_t depth,
                          const uint8_t *input,
                          size_t inputRowPitch,
                          size_t inputDepthPitch,
                          uint8_t *output,
                          size_t outputRowPitch,
                          size_t outputDepthPitch);

void LoadRGB16FToRGB9E5(size_t width,
                        size_t height,
                        size_t depth,
                        const uint8_t *input,
                        size_t inputRowPitch,
                        size_t inputDepthPitch,
                        uint8_t *output,
                        size_t outputRowPitch,
                        size_t outputDepthPitch);

void LoadRGB32FToRGB9E5(size_t width,
                        size_t height,
                        size_t depth,
                        const uint8_t *input,
                        size_t inputRowPitch,
                        size_t inputDepthPitch,
                        uint8_t *output,
                        size_t outputRowPitch,
                        size_t outputDepthPitch);

void LoadRGB16FToRG11B10F(size_t width,
                          size_t height,
                          size_t depth,
                          const uint8_t *input,
                          size_t inputRowPitch,
                          size_t inputDepthPitch,
                          uint8_t *output,
                          size_t outputRowPitch,
                          size_t outputDepthPitch);

void LoadRGB32FToRG11B10F(size_t width,
                          size_t height,
                          size_t depth,
                          const uint8_t *input,
                          size_t inputRowPitch,
                          size_t inputDepthPitch,
                          uint8_t *output,
                          size_t outputRowPitch,
                          size_t outputDepthPitch);

void LoadG8R24ToR24G8(size_t width,
                      size_t height,
                      size_t depth,
                      const uint8_t *input,
                      size_t inputRowPitch,
                      size_t inputDepthPitch,
                      uint8_t *output,
                      size_t outputRowPitch,
                      size_t outputDepthPitch);

void LoadD24S8ToD32FS8X24(size_t width,
                          size_t height,
                          size_t depth,
                          const uint8_t *input,
                          size_t inputRowPitch,
                          size_t inputDepthPitch,
                          uint8_t *output,
                          size_t outputRowPitch,
                          size_t outputDepthPitch);

void LoadD24S8ToD32F(size_t width,
                     size_t height,
                     size_t depth,
                     const uint8_t *input,
                     size_t inputRowPitch,
                     size_t inputDepthPitch,
                     uint8_t *output,
                     size_t outputRowPitch,
                     size_t outputDepthPitch);

void LoadD32ToD32FX32(size_t width,
                      size_t height,
                      size_t depth,
                      const uint8_t *input,
                      size_t inputRowPitch,
                      size_t inputDepthPitch,
                      uint8_t *output,
                      size_t outputRowPitch,
                      size_t outputDepthPitch);

void LoadD32ToD32F(size_t width,
                   size_t height,
                   size_t depth,
                   const uint8_t *input,
                   size_t inputRowPitch,
                   size_t inputDepthPitch,
                   uint8_t *output,
                   size_t outputRowPitch,
                   size_t outputDepthPitch);

void LoadD32FToD32F(size_t width,
                    size_t height,
                    size_t depth,
                    const uint8_t *input,
                    size_t inputRowPitch,
                    size_t inputDepthPitch,
                    uint8_t *output,
                    size_t outputRowPitch,
                    size_t outputDepthPitch);

void LoadD32FS8X24ToD24S8(size_t width,
                          size_t height,
                          size_t depth,
                          const uint8_t *input,
                          size_t inputRowPitch,
                          size_t inputDepthPitch,
                          uint8_t *output,
                          size_t outputRowPitch,
                          size_t outputDepthPitch);

void LoadX24S8ToS8(size_t width,
                   size_t height,
                   size_t depth,
                   const uint8_t *input,
                   size_t inputRowPitch,
                   size_t inputDepthPitch,
                   uint8_t *output,
                   size_t outputRowPitch,
                   size_t outputDepthPitch);

void LoadX32S8ToS8(size_t width,
                   size_t height,
                   size_t depth,
                   const uint8_t *input,
                   size_t inputRowPitch,
                   size_t inputDepthPitch,
                   uint8_t *output,
                   size_t outputRowPitch,
                   size_t outputDepthPitch);

void LoadD32FS8X24ToD32F(size_t width,
                         size_t height,
                         size_t depth,
                         const uint8_t *input,
                         size_t inputRowPitch,
                         size_t inputDepthPitch,
                         uint8_t *output,
                         size_t outputRowPitch,
                         size_t outputDepthPitch);

void LoadD32FS8X24ToD32FS8X24(size_t width,
                              size_t height,
                              size_t depth,
                              const uint8_t *input,
                              size_t inputRowPitch,
                              size_t inputDepthPitch,
                              uint8_t *output,
                              size_t outputRowPitch,
                              size_t outputDepthPitch);

template <typename type, size_t componentCount>
inline void LoadToNative(size_t width,
                         size_t height,
                         size_t depth,
                         const uint8_t *input,
                         size_t inputRowPitch,
                         size_t inputDepthPitch,
                         uint8_t *output,
                         size_t outputRowPitch,
                         size_t outputDepthPitch);

template <typename type, uint32_t fourthComponentBits>
inline void LoadToNative3To4(size_t width,
                             size_t height,
                             size_t depth,
                             const uint8_t *input,
                             size_t inputRowPitch,
                             size_t inputDepthPitch,
                             uint8_t *output,
                             size_t outputRowPitch,
                             size_t outputDepthPitch);

template <size_t componentCount>
inline void Load32FTo16F(size_t width,
                         size_t height,
                         size_t depth,
                         const uint8_t *input,
                         size_t inputRowPitch,
                         size_t inputDepthPitch,
                         uint8_t *output,
                         size_t outputRowPitch,
                         size_t outputDepthPitch);

void LoadRGB32FToRGBA16F(size_t width,
                         size_t height,
                         size_t depth,
                         const uint8_t *input,
                         size_t inputRowPitch,
                         size_t inputDepthPitch,
                         uint8_t *output,
                         size_t outputRowPitch,
                         size_t outputDepthPitch);

void LoadRGB32FToRGB16F(size_t width,
                        size_t height,
                        size_t depth,
                        const uint8_t *input,
                        size_t inputRowPitch,
                        size_t inputDepthPitch,
                        uint8_t *output,
                        size_t outputRowPitch,
                        size_t outputDepthPitch);

template <size_t blockWidth, size_t blockHeight, size_t blockDepth, size_t blockSize>
inline void LoadCompressedToNative(size_t width,
                                   size_t height,
                                   size_t depth,
                                   const uint8_t *input,
                                   size_t inputRowPitch,
                                   size_t inputDepthPitch,
                                   uint8_t *output,
                                   size_t outputRowPitch,
                                   size_t outputDepthPitch);

void LoadRGBDXT1ToRGBADXT3(size_t width,
                           size_t height,
                           size_t depth,
                           const uint8_t *input,
                           size_t inputRowPitch,
                           size_t inputDepthPitch,
                           uint8_t *output,
                           size_t outputRowPitch,
                           size_t outputDepthPitch);

void LoadR32ToR16(size_t width,
                  size_t height,
                  size_t depth,
                  const uint8_t *input,
                  size_t inputRowPitch,
                  size_t inputDepthPitch,
                  uint8_t *output,
                  size_t outputRowPitch,
                  size_t outputDepthPitch);

template <typename type,
          uint32_t firstBits,
          uint32_t secondBits,
          uint32_t thirdBits,
          uint32_t fourthBits>
inline void Initialize4ComponentData(size_t width,
                                     size_t height,
                                     size_t depth,
                                     uint8_t *output,
                                     size_t outputRowPitch,
                                     size_t outputDepthPitch);

void LoadR32ToR24G8(size_t width,
                    size_t height,
                    size_t depth,
                    const uint8_t *input,
                    size_t inputRowPitch,
                    size_t inputDepthPitch,
                    uint8_t *output,
                    size_t outputRowPitch,
                    size_t outputDepthPitch);

void LoadETC1RGB8ToRGBA8(size_t width,
                         size_t height,
                         size_t depth,
                         const uint8_t *input,
                         size_t inputRowPitch,
                         size_t inputDepthPitch,
                         uint8_t *output,
                         size_t outputRowPitch,
                         size_t outputDepthPitch);

void LoadETC1RGB8ToBC1(size_t width,
                       size_t height,
                       size_t depth,
                       const uint8_t *input,
                       size_t inputRowPitch,
                       size_t inputDepthPitch,
                       uint8_t *output,
                       size_t outputRowPitch,
                       size_t outputDepthPitch);

void LoadEACR11ToR8(size_t width,
                    size_t height,
                    size_t depth,
                    const uint8_t *input,
                    size_t inputRowPitch,
                    size_t inputDepthPitch,
                    uint8_t *output,
                    size_t outputRowPitch,
                    size_t outputDepthPitch);

void LoadEACR11SToR8(size_t width,
                     size_t height,
                     size_t depth,
                     const uint8_t *input,
                     size_t inputRowPitch,
                     size_t inputDepthPitch,
                     uint8_t *output,
                     size_t outputRowPitch,
                     size_t outputDepthPitch);

void LoadEACRG11ToRG8(size_t width,
                      size_t height,
                      size_t depth,
                      const uint8_t *input,
                      size_t inputRowPitch,
                      size_t inputDepthPitch,
                      uint8_t *output,
                      size_t outputRowPitch,
                      size_t outputDepthPitch);

void LoadEACRG11SToRG8(size_t width,
                       size_t height,
                       size_t depth,
                       const uint8_t *input,
                       size_t inputRowPitch,
                       size_t inputDepthPitch,
                       uint8_t *output,
                       size_t outputRowPitch,
                       size_t outputDepthPitch);

void LoadEACR11ToR16(size_t width,
                     size_t height,
                     size_t depth,
                     const uint8_t *input,
                     size_t inputRowPitch,
                     size_t inputDepthPitch,
                     uint8_t *output,
                     size_t outputRowPitch,
                     size_t outputDepthPitch);

void LoadEACR11SToR16(size_t width,
                      size_t height,
                      size_t depth,
                      const uint8_t *input,
                      size_t inputRowPitch,
                      size_t inputDepthPitch,
                      uint8_t *output,
                      size_t outputRowPitch,
                      size_t outputDepthPitch);

void LoadEACRG11ToRG16(size_t width,
                       size_t height,
                       size_t depth,
                       const uint8_t *input,
                       size_t inputRowPitch,
                       size_t inputDepthPitch,
                       uint8_t *output,
                       size_t outputRowPitch,
                       size_t outputDepthPitch);

void LoadEACRG11SToRG16(size_t width,
                        size_t height,
                        size_t depth,
                        const uint8_t *input,
                        size_t inputRowPitch,
                        size_t inputDepthPitch,
                        uint8_t *output,
                        size_t outputRowPitch,
                        size_t outputDepthPitch);

void LoadEACR11ToR16F(size_t width,
                      size_t height,
                      size_t depth,
                      const uint8_t *input,
                      size_t inputRowPitch,
                      size_t inputDepthPitch,
                      uint8_t *output,
                      size_t outputRowPitch,
                      size_t outputDepthPitch);

void LoadEACR11SToR16F(size_t width,
                       size_t height,
                       size_t depth,
                       const uint8_t *input,
                       size_t inputRowPitch,
                       size_t inputDepthPitch,
                       uint8_t *output,
                       size_t outputRowPitch,
                       size_t outputDepthPitch);

void LoadEACRG11ToRG16F(size_t width,
                        size_t height,
                        size_t depth,
                        const uint8_t *input,
                        size_t inputRowPitch,
                        size_t inputDepthPitch,
                        uint8_t *output,
                        size_t outputRowPitch,
                        size_t outputDepthPitch);

void LoadEACRG11SToRG16F(size_t width,
                         size_t height,
                         size_t depth,
                         const uint8_t *input,
                         size_t inputRowPitch,
                         size_t inputDepthPitch,
                         uint8_t *output,
                         size_t outputRowPitch,
                         size_t outputDepthPitch);

void LoadETC2RGB8ToRGBA8(size_t width,
                         size_t height,
                         size_t depth,
                         const uint8_t *input,
                         size_t inputRowPitch,
                         size_t inputDepthPitch,
                         uint8_t *output,
                         size_t outputRowPitch,
                         size_t outputDepthPitch);

void LoadETC2RGB8ToBC1(size_t width,
                       size_t height,
                       size_t depth,
                       const uint8_t *input,
                       size_t inputRowPitch,
                       size_t inputDepthPitch,
                       uint8_t *output,
                       size_t outputRowPitch,
                       size_t outputDepthPitch);

void LoadETC2SRGB8ToRGBA8(size_t width,
                          size_t height,
                          size_t depth,
                          const uint8_t *input,
                          size_t inputRowPitch,
                          size_t inputDepthPitch,
                          uint8_t *output,
                          size_t outputRowPitch,
                          size_t outputDepthPitch);

void LoadETC2SRGB8ToBC1(size_t width,
                        size_t height,
                        size_t depth,
                        const uint8_t *input,
                        size_t inputRowPitch,
                        size_t inputDepthPitch,
                        uint8_t *output,
                        size_t outputRowPitch,
                        size_t outputDepthPitch);

void LoadETC2RGB8A1ToRGBA8(size_t width,
                           size_t height,
                           size_t depth,
                           const uint8_t *input,
                           size_t inputRowPitch,
                           size_t inputDepthPitch,
                           uint8_t *output,
                           size_t outputRowPitch,
                           size_t outputDepthPitch);

void LoadETC2RGB8A1ToBC1(size_t width,
                         size_t height,
                         size_t depth,
                         const uint8_t *input,
                         size_t inputRowPitch,
                         size_t inputDepthPitch,
                         uint8_t *output,
                         size_t outputRowPitch,
                         size_t outputDepthPitch);

void LoadETC2SRGB8A1ToRGBA8(size_t width,
                            size_t height,
                            size_t depth,
                            const uint8_t *input,
                            size_t inputRowPitch,
                            size_t inputDepthPitch,
                            uint8_t *output,
                            size_t outputRowPitch,
                            size_t outputDepthPitch);

void LoadETC2SRGB8A1ToBC1(size_t width,
                          size_t height,
                          size_t depth,
                          const uint8_t *input,
                          size_t inputRowPitch,
                          size_t inputDepthPitch,
                          uint8_t *output,
                          size_t outputRowPitch,
                          size_t outputDepthPitch);

void LoadETC2RGBA8ToRGBA8(size_t width,
                          size_t height,
                          size_t depth,
                          const uint8_t *input,
                          size_t inputRowPitch,
                          size_t inputDepthPitch,
                          uint8_t *output,
                          size_t outputRowPitch,
                          size_t outputDepthPitch);

void LoadETC2SRGBA8ToSRGBA8(size_t width,
                            size_t height,
                            size_t depth,
                            const uint8_t *input,
                            size_t inputRowPitch,
                            size_t inputDepthPitch,
                            uint8_t *output,
                            size_t outputRowPitch,
                            size_t outputDepthPitch);

}  // namespace angle

#include "loadimage.inc"

#endif  // IMAGEUTIL_LOADIMAGE_H_
