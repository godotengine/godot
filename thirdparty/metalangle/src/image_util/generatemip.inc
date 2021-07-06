//
// Copyright 2015 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//

// generatemip.inc: Defines the GenerateMip function, templated on the format
// type of the image for which mip levels are being generated.

#include "common/mathutil.h"

#include "image_util/imageformats.h"

namespace angle
{

namespace priv
{

template <typename T>
static inline T *GetPixel(uint8_t *data, size_t x, size_t y, size_t z, size_t rowPitch, size_t depthPitch)
{
    return reinterpret_cast<T*>(data + (x * sizeof(T)) + (y * rowPitch) + (z * depthPitch));
}

template <typename T>
static inline const T *GetPixel(const uint8_t *data, size_t x, size_t y, size_t z, size_t rowPitch, size_t depthPitch)
{
    return reinterpret_cast<const T*>(data + (x * sizeof(T)) + (y * rowPitch) + (z * depthPitch));
}

template <typename T>
static void GenerateMip_Y(size_t sourceWidth, size_t sourceHeight, size_t sourceDepth,
                          const uint8_t *sourceData, size_t sourceRowPitch, size_t sourceDepthPitch,
                          size_t destWidth, size_t destHeight, size_t destDepth,
                          uint8_t *destData, size_t destRowPitch, size_t destDepthPitch)
{
    ASSERT(sourceWidth == 1);
    ASSERT(sourceHeight > 1);
    ASSERT(sourceDepth == 1);

    for (size_t y = 0; y < destHeight; y++)
    {
        const T *src0 = GetPixel<T>(sourceData, 0, y * 2, 0, sourceRowPitch, sourceDepthPitch);
        const T *src1 = GetPixel<T>(sourceData, 0, y * 2 + 1, 0, sourceRowPitch, sourceDepthPitch);
        T *dst = GetPixel<T>(destData, 0, y, 0, destRowPitch, destDepthPitch);

        T::average(dst, src0, src1);
    }
}

template <typename T>
static void GenerateMip_X(size_t sourceWidth, size_t sourceHeight, size_t sourceDepth,
                          const uint8_t *sourceData, size_t sourceRowPitch, size_t sourceDepthPitch,
                          size_t destWidth, size_t destHeight, size_t destDepth,
                          uint8_t *destData, size_t destRowPitch, size_t destDepthPitch)
{
    ASSERT(sourceWidth > 1);
    ASSERT(sourceHeight == 1);
    ASSERT(sourceDepth == 1);

    for (size_t x = 0; x < destWidth; x++)
    {
        const T *src0 = GetPixel<T>(sourceData, x * 2, 0, 0, sourceRowPitch, sourceDepthPitch);
        const T *src1 = GetPixel<T>(sourceData, x * 2 + 1, 0, 0, sourceRowPitch, sourceDepthPitch);
        T *dst = GetPixel<T>(destData, x, 0, 0, destRowPitch, destDepthPitch);

        T::average(dst, src0, src1);
    }
}

template <typename T>
static void GenerateMip_Z(size_t sourceWidth, size_t sourceHeight, size_t sourceDepth,
                          const uint8_t *sourceData, size_t sourceRowPitch, size_t sourceDepthPitch,
                          size_t destWidth, size_t destHeight, size_t destDepth,
                          uint8_t *destData, size_t destRowPitch, size_t destDepthPitch)
{
    ASSERT(sourceWidth == 1);
    ASSERT(sourceHeight == 1);
    ASSERT(sourceDepth > 1);

    for (size_t z = 0; z < destDepth; z++)
    {
        const T *src0 = GetPixel<T>(sourceData, 0, 0, z * 2, sourceRowPitch, sourceDepthPitch);
        const T *src1 = GetPixel<T>(sourceData, 0, 0, z * 2 + 1, sourceRowPitch, sourceDepthPitch);
        T *dst = GetPixel<T>(destData, 0, 0, z, destRowPitch, destDepthPitch);

        T::average(dst, src0, src1);
    }
}

template <typename T>
static void GenerateMip_XY(size_t sourceWidth, size_t sourceHeight, size_t sourceDepth,
                           const uint8_t *sourceData, size_t sourceRowPitch, size_t sourceDepthPitch,
                           size_t destWidth, size_t destHeight, size_t destDepth,
                           uint8_t *destData, size_t destRowPitch, size_t destDepthPitch)
{
    ASSERT(sourceWidth > 1);
    ASSERT(sourceHeight > 1);
    ASSERT(sourceDepth == 1);

    for (size_t y = 0; y < destHeight; y++)
    {
        for (size_t x = 0; x < destWidth; x++)
        {
            const T *src0 = GetPixel<T>(sourceData, x * 2, y * 2, 0, sourceRowPitch, sourceDepthPitch);
            const T *src1 = GetPixel<T>(sourceData, x * 2, y * 2 + 1, 0, sourceRowPitch, sourceDepthPitch);
            const T *src2 = GetPixel<T>(sourceData, x * 2 + 1, y * 2, 0, sourceRowPitch, sourceDepthPitch);
            const T *src3 = GetPixel<T>(sourceData, x * 2 + 1, y * 2 + 1, 0, sourceRowPitch, sourceDepthPitch);
            T *dst = GetPixel<T>(destData, x, y, 0, destRowPitch, destDepthPitch);

            T tmp0, tmp1;

            T::average(&tmp0, src0, src1);
            T::average(&tmp1, src2, src3);
            T::average(dst, &tmp0, &tmp1);
        }
    }
}

template <typename T>
static void GenerateMip_YZ(size_t sourceWidth, size_t sourceHeight, size_t sourceDepth,
                           const uint8_t *sourceData, size_t sourceRowPitch, size_t sourceDepthPitch,
                           size_t destWidth, size_t destHeight, size_t destDepth,
                           uint8_t *destData, size_t destRowPitch, size_t destDepthPitch)
{
    ASSERT(sourceWidth == 1);
    ASSERT(sourceHeight > 1);
    ASSERT(sourceDepth > 1);

    for (size_t z = 0; z < destDepth; z++)
    {
        for (size_t y = 0; y < destHeight; y++)
        {
            const T *src0 = GetPixel<T>(sourceData, 0, y * 2, z * 2, sourceRowPitch, sourceDepthPitch);
            const T *src1 = GetPixel<T>(sourceData, 0, y * 2, z * 2 + 1, sourceRowPitch, sourceDepthPitch);
            const T *src2 = GetPixel<T>(sourceData, 0, y * 2 + 1, z * 2, sourceRowPitch, sourceDepthPitch);
            const T *src3 = GetPixel<T>(sourceData, 0, y * 2 + 1, z * 2 + 1, sourceRowPitch, sourceDepthPitch);
            T *dst = GetPixel<T>(destData, 0, y, z, destRowPitch, destDepthPitch);

            T tmp0, tmp1;

            T::average(&tmp0, src0, src1);
            T::average(&tmp1, src2, src3);
            T::average(dst, &tmp0, &tmp1);
        }
    }
}

template <typename T>
static void GenerateMip_XZ(size_t sourceWidth, size_t sourceHeight, size_t sourceDepth,
                           const uint8_t *sourceData, size_t sourceRowPitch, size_t sourceDepthPitch,
                           size_t destWidth, size_t destHeight, size_t destDepth,
                           uint8_t *destData, size_t destRowPitch, size_t destDepthPitch)
{
    ASSERT(sourceWidth > 1);
    ASSERT(sourceHeight == 1);
    ASSERT(sourceDepth > 1);

    for (size_t z = 0; z < destDepth; z++)
    {
        for (size_t x = 0; x < destWidth; x++)
        {
            const T *src0 = GetPixel<T>(sourceData, x * 2, 0, z * 2, sourceRowPitch, sourceDepthPitch);
            const T *src1 = GetPixel<T>(sourceData, x * 2, 0, z * 2 + 1, sourceRowPitch, sourceDepthPitch);
            const T *src2 = GetPixel<T>(sourceData, x * 2 + 1, 0, z * 2, sourceRowPitch, sourceDepthPitch);
            const T *src3 = GetPixel<T>(sourceData, x * 2 + 1, 0, z * 2 + 1, sourceRowPitch, sourceDepthPitch);
            T *dst = GetPixel<T>(destData, x, 0, z, destRowPitch, destDepthPitch);

            T tmp0, tmp1;

            T::average(&tmp0, src0, src1);
            T::average(&tmp1, src2, src3);
            T::average(dst, &tmp0, &tmp1);
        }
    }
}

template <typename T>
static void GenerateMip_XYZ(size_t sourceWidth, size_t sourceHeight, size_t sourceDepth,
                            const uint8_t *sourceData, size_t sourceRowPitch, size_t sourceDepthPitch,
                            size_t destWidth, size_t destHeight, size_t destDepth,
                            uint8_t *destData, size_t destRowPitch, size_t destDepthPitch)
{
    ASSERT(sourceWidth > 1);
    ASSERT(sourceHeight > 1);
    ASSERT(sourceDepth > 1);

    for (size_t z = 0; z < destDepth; z++)
    {
        for (size_t y = 0; y < destHeight; y++)
        {
            for (size_t x = 0; x < destWidth; x++)
            {
                const T *src0 = GetPixel<T>(sourceData, x * 2, y * 2, z * 2, sourceRowPitch, sourceDepthPitch);
                const T *src1 = GetPixel<T>(sourceData, x * 2, y * 2, z * 2 + 1, sourceRowPitch, sourceDepthPitch);
                const T *src2 = GetPixel<T>(sourceData, x * 2, y * 2 + 1, z * 2, sourceRowPitch, sourceDepthPitch);
                const T *src3 = GetPixel<T>(sourceData, x * 2, y * 2 + 1, z * 2 + 1, sourceRowPitch, sourceDepthPitch);
                const T *src4 = GetPixel<T>(sourceData, x * 2 + 1, y * 2, z * 2, sourceRowPitch, sourceDepthPitch);
                const T *src5 = GetPixel<T>(sourceData, x * 2 + 1, y * 2, z * 2 + 1, sourceRowPitch, sourceDepthPitch);
                const T *src6 = GetPixel<T>(sourceData, x * 2 + 1, y * 2 + 1, z * 2, sourceRowPitch, sourceDepthPitch);
                const T *src7 = GetPixel<T>(sourceData, x * 2 + 1, y * 2 + 1, z * 2 + 1, sourceRowPitch, sourceDepthPitch);
                T *dst = GetPixel<T>(destData, x, y, z, destRowPitch, destDepthPitch);

                T tmp0, tmp1, tmp2, tmp3, tmp4, tmp5;

                T::average(&tmp0, src0, src1);
                T::average(&tmp1, src2, src3);
                T::average(&tmp2, src4, src5);
                T::average(&tmp3, src6, src7);

                T::average(&tmp4, &tmp0, &tmp1);
                T::average(&tmp5, &tmp2, &tmp3);

                T::average(dst, &tmp4, &tmp5);
            }
        }
    }
}


typedef void (*MipGenerationFunction)(size_t sourceWidth, size_t sourceHeight, size_t sourceDepth,
                                      const uint8_t *sourceData, size_t sourceRowPitch, size_t sourceDepthPitch,
                                      size_t destWidth, size_t destHeight, size_t destDepth,
                                      uint8_t *destData, size_t destRowPitch, size_t destDepthPitch);

template <typename T>
static MipGenerationFunction GetMipGenerationFunction(size_t sourceWidth, size_t sourceHeight, size_t sourceDepth)
{
    uint8_t index = ((sourceWidth > 1)  ? 1 : 0) |
                    ((sourceHeight > 1) ? 2 : 0) |
                    ((sourceDepth > 1)  ? 4 : 0);

    switch (index)
    {
      case 0: return nullptr;
      case 1: return GenerateMip_X<T>;   // W x 1 x 1
      case 2: return GenerateMip_Y<T>;   // 1 x H x 1
      case 3: return GenerateMip_XY<T>;  // W x H x 1
      case 4: return GenerateMip_Z<T>;   // 1 x 1 x D
      case 5: return GenerateMip_XZ<T>;  // W x 1 x D
      case 6: return GenerateMip_YZ<T>;  // 1 x H x D
      case 7: return GenerateMip_XYZ<T>; // W x H x D
    }

    UNREACHABLE();
    return nullptr;
}

}  // namespace priv

template <typename T>
inline void GenerateMip(size_t sourceWidth, size_t sourceHeight, size_t sourceDepth,
                        const uint8_t *sourceData, size_t sourceRowPitch, size_t sourceDepthPitch,
                        uint8_t *destData, size_t destRowPitch, size_t destDepthPitch)
{
    size_t mipWidth = std::max<size_t>(1, sourceWidth >> 1);
    size_t mipHeight = std::max<size_t>(1, sourceHeight >> 1);
    size_t mipDepth = std::max<size_t>(1, sourceDepth >> 1);

    priv::MipGenerationFunction generationFunction = priv::GetMipGenerationFunction<T>(sourceWidth, sourceHeight, sourceDepth);
    ASSERT(generationFunction != nullptr);

    generationFunction(sourceWidth, sourceHeight, sourceDepth, sourceData, sourceRowPitch, sourceDepthPitch,
                       mipWidth, mipHeight, mipDepth, destData, destRowPitch, destDepthPitch);
}

}  // namespace angle
