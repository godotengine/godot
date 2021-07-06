//
// Copyright 2013 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//

// angle_loadimage.cpp: Defines image loading functions.

#include "image_util/loadimage.h"

#include "common/mathutil.h"
#include "common/platform.h"
#include "image_util/imageformats.h"

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
                   size_t outputDepthPitch)
{
#if defined(ANGLE_USE_SSE)
    if (gl::supportsSSE2())
    {
        __m128i zeroWide = _mm_setzero_si128();

        for (size_t z = 0; z < depth; z++)
        {
            for (size_t y = 0; y < height; y++)
            {
                const uint8_t *source =
                    priv::OffsetDataPointer<uint8_t>(input, y, z, inputRowPitch, inputDepthPitch);
                uint32_t *dest = priv::OffsetDataPointer<uint32_t>(output, y, z, outputRowPitch,
                                                                   outputDepthPitch);

                size_t x = 0;

                // Make output writes aligned
                for (; ((reinterpret_cast<intptr_t>(&dest[x]) & 0xF) != 0 && x < width); x++)
                {
                    dest[x] = static_cast<uint32_t>(source[x]) << 24;
                }

                for (; x + 7 < width; x += 8)
                {
                    __m128i sourceData =
                        _mm_loadl_epi64(reinterpret_cast<const __m128i *>(&source[x]));
                    // Interleave each byte to 16bit, make the lower byte to zero
                    sourceData = _mm_unpacklo_epi8(zeroWide, sourceData);
                    // Interleave each 16bit to 32bit, make the lower 16bit to zero
                    __m128i lo = _mm_unpacklo_epi16(zeroWide, sourceData);
                    __m128i hi = _mm_unpackhi_epi16(zeroWide, sourceData);

                    _mm_store_si128(reinterpret_cast<__m128i *>(&dest[x]), lo);
                    _mm_store_si128(reinterpret_cast<__m128i *>(&dest[x + 4]), hi);
                }

                // Handle the remainder
                for (; x < width; x++)
                {
                    dest[x] = static_cast<uint32_t>(source[x]) << 24;
                }
            }
        }

        return;
    }
#endif

    for (size_t z = 0; z < depth; z++)
    {
        for (size_t y = 0; y < height; y++)
        {
            const uint8_t *source =
                priv::OffsetDataPointer<uint8_t>(input, y, z, inputRowPitch, inputDepthPitch);
            uint32_t *dest =
                priv::OffsetDataPointer<uint32_t>(output, y, z, outputRowPitch, outputDepthPitch);
            for (size_t x = 0; x < width; x++)
            {
                dest[x] = static_cast<uint32_t>(source[x]) << 24;
            }
        }
    }
}

void LoadA8ToBGRA8(size_t width,
                   size_t height,
                   size_t depth,
                   const uint8_t *input,
                   size_t inputRowPitch,
                   size_t inputDepthPitch,
                   uint8_t *output,
                   size_t outputRowPitch,
                   size_t outputDepthPitch)
{
    // Same as loading to RGBA
    LoadA8ToRGBA8(width, height, depth, input, inputRowPitch, inputDepthPitch, output,
                  outputRowPitch, outputDepthPitch);
}

void LoadA32FToRGBA32F(size_t width,
                       size_t height,
                       size_t depth,
                       const uint8_t *input,
                       size_t inputRowPitch,
                       size_t inputDepthPitch,
                       uint8_t *output,
                       size_t outputRowPitch,
                       size_t outputDepthPitch)
{
    for (size_t z = 0; z < depth; z++)
    {
        for (size_t y = 0; y < height; y++)
        {
            const float *source =
                priv::OffsetDataPointer<float>(input, y, z, inputRowPitch, inputDepthPitch);
            float *dest =
                priv::OffsetDataPointer<float>(output, y, z, outputRowPitch, outputDepthPitch);
            for (size_t x = 0; x < width; x++)
            {
                dest[4 * x + 0] = 0.0f;
                dest[4 * x + 1] = 0.0f;
                dest[4 * x + 2] = 0.0f;
                dest[4 * x + 3] = source[x];
            }
        }
    }
}

void LoadA16FToRGBA16F(size_t width,
                       size_t height,
                       size_t depth,
                       const uint8_t *input,
                       size_t inputRowPitch,
                       size_t inputDepthPitch,
                       uint8_t *output,
                       size_t outputRowPitch,
                       size_t outputDepthPitch)
{
    for (size_t z = 0; z < depth; z++)
    {
        for (size_t y = 0; y < height; y++)
        {
            const uint16_t *source =
                priv::OffsetDataPointer<uint16_t>(input, y, z, inputRowPitch, inputDepthPitch);
            uint16_t *dest =
                priv::OffsetDataPointer<uint16_t>(output, y, z, outputRowPitch, outputDepthPitch);
            for (size_t x = 0; x < width; x++)
            {
                dest[4 * x + 0] = 0;
                dest[4 * x + 1] = 0;
                dest[4 * x + 2] = 0;
                dest[4 * x + 3] = source[x];
            }
        }
    }
}

void LoadL8ToRGBA8(size_t width,
                   size_t height,
                   size_t depth,
                   const uint8_t *input,
                   size_t inputRowPitch,
                   size_t inputDepthPitch,
                   uint8_t *output,
                   size_t outputRowPitch,
                   size_t outputDepthPitch)
{
    for (size_t z = 0; z < depth; z++)
    {
        for (size_t y = 0; y < height; y++)
        {
            const uint8_t *source =
                priv::OffsetDataPointer<uint8_t>(input, y, z, inputRowPitch, inputDepthPitch);
            uint8_t *dest =
                priv::OffsetDataPointer<uint8_t>(output, y, z, outputRowPitch, outputDepthPitch);
            for (size_t x = 0; x < width; x++)
            {
                uint8_t sourceVal = source[x];
                dest[4 * x + 0]   = sourceVal;
                dest[4 * x + 1]   = sourceVal;
                dest[4 * x + 2]   = sourceVal;
                dest[4 * x + 3]   = 0xFF;
            }
        }
    }
}

void LoadL8ToBGRA8(size_t width,
                   size_t height,
                   size_t depth,
                   const uint8_t *input,
                   size_t inputRowPitch,
                   size_t inputDepthPitch,
                   uint8_t *output,
                   size_t outputRowPitch,
                   size_t outputDepthPitch)
{
    // Same as loading to RGBA
    LoadL8ToRGBA8(width, height, depth, input, inputRowPitch, inputDepthPitch, output,
                  outputRowPitch, outputDepthPitch);
}

void LoadL32FToRGBA32F(size_t width,
                       size_t height,
                       size_t depth,
                       const uint8_t *input,
                       size_t inputRowPitch,
                       size_t inputDepthPitch,
                       uint8_t *output,
                       size_t outputRowPitch,
                       size_t outputDepthPitch)
{
    for (size_t z = 0; z < depth; z++)
    {
        for (size_t y = 0; y < height; y++)
        {
            const float *source =
                priv::OffsetDataPointer<float>(input, y, z, inputRowPitch, inputDepthPitch);
            float *dest =
                priv::OffsetDataPointer<float>(output, y, z, outputRowPitch, outputDepthPitch);
            for (size_t x = 0; x < width; x++)
            {
                dest[4 * x + 0] = source[x];
                dest[4 * x + 1] = source[x];
                dest[4 * x + 2] = source[x];
                dest[4 * x + 3] = 1.0f;
            }
        }
    }
}

void LoadL16FToRGBA16F(size_t width,
                       size_t height,
                       size_t depth,
                       const uint8_t *input,
                       size_t inputRowPitch,
                       size_t inputDepthPitch,
                       uint8_t *output,
                       size_t outputRowPitch,
                       size_t outputDepthPitch)
{
    for (size_t z = 0; z < depth; z++)
    {
        for (size_t y = 0; y < height; y++)
        {
            const uint16_t *source =
                priv::OffsetDataPointer<uint16_t>(input, y, z, inputRowPitch, inputDepthPitch);
            uint16_t *dest =
                priv::OffsetDataPointer<uint16_t>(output, y, z, outputRowPitch, outputDepthPitch);
            for (size_t x = 0; x < width; x++)
            {
                dest[4 * x + 0] = source[x];
                dest[4 * x + 1] = source[x];
                dest[4 * x + 2] = source[x];
                dest[4 * x + 3] = gl::Float16One;
            }
        }
    }
}

void LoadLA8ToRGBA8(size_t width,
                    size_t height,
                    size_t depth,
                    const uint8_t *input,
                    size_t inputRowPitch,
                    size_t inputDepthPitch,
                    uint8_t *output,
                    size_t outputRowPitch,
                    size_t outputDepthPitch)
{
    for (size_t z = 0; z < depth; z++)
    {
        for (size_t y = 0; y < height; y++)
        {
            const uint8_t *source =
                priv::OffsetDataPointer<uint8_t>(input, y, z, inputRowPitch, inputDepthPitch);
            uint8_t *dest =
                priv::OffsetDataPointer<uint8_t>(output, y, z, outputRowPitch, outputDepthPitch);
            for (size_t x = 0; x < width; x++)
            {
                dest[4 * x + 0] = source[2 * x + 0];
                dest[4 * x + 1] = source[2 * x + 0];
                dest[4 * x + 2] = source[2 * x + 0];
                dest[4 * x + 3] = source[2 * x + 1];
            }
        }
    }
}

void LoadLA8ToBGRA8(size_t width,
                    size_t height,
                    size_t depth,
                    const uint8_t *input,
                    size_t inputRowPitch,
                    size_t inputDepthPitch,
                    uint8_t *output,
                    size_t outputRowPitch,
                    size_t outputDepthPitch)
{
    // Same as loading to RGBA
    LoadLA8ToRGBA8(width, height, depth, input, inputRowPitch, inputDepthPitch, output,
                   outputRowPitch, outputDepthPitch);
}

void LoadLA32FToRGBA32F(size_t width,
                        size_t height,
                        size_t depth,
                        const uint8_t *input,
                        size_t inputRowPitch,
                        size_t inputDepthPitch,
                        uint8_t *output,
                        size_t outputRowPitch,
                        size_t outputDepthPitch)
{
    for (size_t z = 0; z < depth; z++)
    {
        for (size_t y = 0; y < height; y++)
        {
            const float *source =
                priv::OffsetDataPointer<float>(input, y, z, inputRowPitch, inputDepthPitch);
            float *dest =
                priv::OffsetDataPointer<float>(output, y, z, outputRowPitch, outputDepthPitch);
            for (size_t x = 0; x < width; x++)
            {
                dest[4 * x + 0] = source[2 * x + 0];
                dest[4 * x + 1] = source[2 * x + 0];
                dest[4 * x + 2] = source[2 * x + 0];
                dest[4 * x + 3] = source[2 * x + 1];
            }
        }
    }
}

void LoadLA16FToRGBA16F(size_t width,
                        size_t height,
                        size_t depth,
                        const uint8_t *input,
                        size_t inputRowPitch,
                        size_t inputDepthPitch,
                        uint8_t *output,
                        size_t outputRowPitch,
                        size_t outputDepthPitch)
{
    for (size_t z = 0; z < depth; z++)
    {
        for (size_t y = 0; y < height; y++)
        {
            const uint16_t *source =
                priv::OffsetDataPointer<uint16_t>(input, y, z, inputRowPitch, inputDepthPitch);
            uint16_t *dest =
                priv::OffsetDataPointer<uint16_t>(output, y, z, outputRowPitch, outputDepthPitch);
            for (size_t x = 0; x < width; x++)
            {
                dest[4 * x + 0] = source[2 * x + 0];
                dest[4 * x + 1] = source[2 * x + 0];
                dest[4 * x + 2] = source[2 * x + 0];
                dest[4 * x + 3] = source[2 * x + 1];
            }
        }
    }
}

void LoadRGB8ToBGR565(size_t width,
                      size_t height,
                      size_t depth,
                      const uint8_t *input,
                      size_t inputRowPitch,
                      size_t inputDepthPitch,
                      uint8_t *output,
                      size_t outputRowPitch,
                      size_t outputDepthPitch)
{
    for (size_t z = 0; z < depth; z++)
    {
        for (size_t y = 0; y < height; y++)
        {
            const uint8_t *source =
                priv::OffsetDataPointer<uint8_t>(input, y, z, inputRowPitch, inputDepthPitch);
            uint16_t *dest =
                priv::OffsetDataPointer<uint16_t>(output, y, z, outputRowPitch, outputDepthPitch);
            for (size_t x = 0; x < width; x++)
            {
                uint8_t r8 = source[x * 3 + 0];
                uint8_t g8 = source[x * 3 + 1];
                uint8_t b8 = source[x * 3 + 2];
                auto r5    = static_cast<uint16_t>(r8 >> 3);
                auto g6    = static_cast<uint16_t>(g8 >> 2);
                auto b5    = static_cast<uint16_t>(b8 >> 3);
                dest[x]    = (r5 << 11) | (g6 << 5) | b5;
            }
        }
    }
}

void LoadRGB565ToBGR565(size_t width,
                        size_t height,
                        size_t depth,
                        const uint8_t *input,
                        size_t inputRowPitch,
                        size_t inputDepthPitch,
                        uint8_t *output,
                        size_t outputRowPitch,
                        size_t outputDepthPitch)
{
    for (size_t z = 0; z < depth; z++)
    {
        for (size_t y = 0; y < height; y++)
        {
            const uint16_t *source =
                priv::OffsetDataPointer<uint16_t>(input, y, z, inputRowPitch, inputDepthPitch);
            uint16_t *dest =
                priv::OffsetDataPointer<uint16_t>(output, y, z, outputRowPitch, outputDepthPitch);
            for (size_t x = 0; x < width; x++)
            {
                // The GL type RGB is packed with with red in the MSB, while the D3D11 type BGR
                // is packed with red in the LSB
                auto rgb    = source[x];
                uint16_t r5 = gl::getShiftedData<5, 11>(rgb);
                uint16_t g6 = gl::getShiftedData<6, 5>(rgb);
                uint16_t b5 = gl::getShiftedData<5, 0>(rgb);
                dest[x]     = (r5 << 11) | (g6 << 5) | b5;
            }
        }
    }
}

void LoadRGB8ToBGRX8(size_t width,
                     size_t height,
                     size_t depth,
                     const uint8_t *input,
                     size_t inputRowPitch,
                     size_t inputDepthPitch,
                     uint8_t *output,
                     size_t outputRowPitch,
                     size_t outputDepthPitch)
{
    for (size_t z = 0; z < depth; z++)
    {
        for (size_t y = 0; y < height; y++)
        {
            const uint8_t *source =
                priv::OffsetDataPointer<uint8_t>(input, y, z, inputRowPitch, inputDepthPitch);
            uint8_t *dest =
                priv::OffsetDataPointer<uint8_t>(output, y, z, outputRowPitch, outputDepthPitch);
            for (size_t x = 0; x < width; x++)
            {
                dest[4 * x + 0] = source[x * 3 + 2];
                dest[4 * x + 1] = source[x * 3 + 1];
                dest[4 * x + 2] = source[x * 3 + 0];
                dest[4 * x + 3] = 0xFF;
            }
        }
    }
}

void LoadRG8ToBGRX8(size_t width,
                    size_t height,
                    size_t depth,
                    const uint8_t *input,
                    size_t inputRowPitch,
                    size_t inputDepthPitch,
                    uint8_t *output,
                    size_t outputRowPitch,
                    size_t outputDepthPitch)
{
    for (size_t z = 0; z < depth; z++)
    {
        for (size_t y = 0; y < height; y++)
        {
            const uint8_t *source =
                priv::OffsetDataPointer<uint8_t>(input, y, z, inputRowPitch, inputDepthPitch);
            uint8_t *dest =
                priv::OffsetDataPointer<uint8_t>(output, y, z, outputRowPitch, outputDepthPitch);
            for (size_t x = 0; x < width; x++)
            {
                dest[4 * x + 0] = 0x00;
                dest[4 * x + 1] = source[x * 2 + 1];
                dest[4 * x + 2] = source[x * 2 + 0];
                dest[4 * x + 3] = 0xFF;
            }
        }
    }
}

void LoadR8ToBGRX8(size_t width,
                   size_t height,
                   size_t depth,
                   const uint8_t *input,
                   size_t inputRowPitch,
                   size_t inputDepthPitch,
                   uint8_t *output,
                   size_t outputRowPitch,
                   size_t outputDepthPitch)
{
    for (size_t z = 0; z < depth; z++)
    {
        for (size_t y = 0; y < height; y++)
        {
            const uint8_t *source =
                priv::OffsetDataPointer<uint8_t>(input, y, z, inputRowPitch, inputDepthPitch);
            uint8_t *dest =
                priv::OffsetDataPointer<uint8_t>(output, y, z, outputRowPitch, outputDepthPitch);
            for (size_t x = 0; x < width; x++)
            {
                dest[4 * x + 0] = 0x00;
                dest[4 * x + 1] = 0x00;
                dest[4 * x + 2] = source[x];
                dest[4 * x + 3] = 0xFF;
            }
        }
    }
}

void LoadR5G6B5ToBGRA8(size_t width,
                       size_t height,
                       size_t depth,
                       const uint8_t *input,
                       size_t inputRowPitch,
                       size_t inputDepthPitch,
                       uint8_t *output,
                       size_t outputRowPitch,
                       size_t outputDepthPitch)
{
    for (size_t z = 0; z < depth; z++)
    {
        for (size_t y = 0; y < height; y++)
        {
            const uint16_t *source =
                priv::OffsetDataPointer<uint16_t>(input, y, z, inputRowPitch, inputDepthPitch);
            uint8_t *dest =
                priv::OffsetDataPointer<uint8_t>(output, y, z, outputRowPitch, outputDepthPitch);
            for (size_t x = 0; x < width; x++)
            {
                uint16_t rgb = source[x];
                dest[4 * x + 0] =
                    static_cast<uint8_t>(((rgb & 0x001F) << 3) | ((rgb & 0x001F) >> 2));
                dest[4 * x + 1] =
                    static_cast<uint8_t>(((rgb & 0x07E0) >> 3) | ((rgb & 0x07E0) >> 9));
                dest[4 * x + 2] =
                    static_cast<uint8_t>(((rgb & 0xF800) >> 8) | ((rgb & 0xF800) >> 13));
                dest[4 * x + 3] = 0xFF;
            }
        }
    }
}

void LoadR5G6B5ToRGBA8(size_t width,
                       size_t height,
                       size_t depth,
                       const uint8_t *input,
                       size_t inputRowPitch,
                       size_t inputDepthPitch,
                       uint8_t *output,
                       size_t outputRowPitch,
                       size_t outputDepthPitch)
{
    for (size_t z = 0; z < depth; z++)
    {
        for (size_t y = 0; y < height; y++)
        {
            const uint16_t *source =
                priv::OffsetDataPointer<uint16_t>(input, y, z, inputRowPitch, inputDepthPitch);
            uint8_t *dest =
                priv::OffsetDataPointer<uint8_t>(output, y, z, outputRowPitch, outputDepthPitch);
            for (size_t x = 0; x < width; x++)
            {
                uint16_t rgb = source[x];
                dest[4 * x + 0] =
                    static_cast<uint8_t>(((rgb & 0xF800) >> 8) | ((rgb & 0xF800) >> 13));
                dest[4 * x + 1] =
                    static_cast<uint8_t>(((rgb & 0x07E0) >> 3) | ((rgb & 0x07E0) >> 9));
                dest[4 * x + 2] =
                    static_cast<uint8_t>(((rgb & 0x001F) << 3) | ((rgb & 0x001F) >> 2));
                dest[4 * x + 3] = 0xFF;
            }
        }
    }
}

void LoadRGBA8ToBGRA8(size_t width,
                      size_t height,
                      size_t depth,
                      const uint8_t *input,
                      size_t inputRowPitch,
                      size_t inputDepthPitch,
                      uint8_t *output,
                      size_t outputRowPitch,
                      size_t outputDepthPitch)
{
#if defined(ANGLE_USE_SSE)
    if (gl::supportsSSE2())
    {
        __m128i brMask = _mm_set1_epi32(0x00ff00ff);

        for (size_t z = 0; z < depth; z++)
        {
            for (size_t y = 0; y < height; y++)
            {
                const uint32_t *source =
                    priv::OffsetDataPointer<uint32_t>(input, y, z, inputRowPitch, inputDepthPitch);
                uint32_t *dest = priv::OffsetDataPointer<uint32_t>(output, y, z, outputRowPitch,
                                                                   outputDepthPitch);

                size_t x = 0;

                // Make output writes aligned
                for (; ((reinterpret_cast<intptr_t>(&dest[x]) & 15) != 0) && x < width; x++)
                {
                    uint32_t rgba = source[x];
                    dest[x]       = (ANGLE_ROTL(rgba, 16) & 0x00ff00ff) | (rgba & 0xff00ff00);
                }

                for (; x + 3 < width; x += 4)
                {
                    __m128i sourceData =
                        _mm_loadu_si128(reinterpret_cast<const __m128i *>(&source[x]));
                    // Mask out g and a, which don't change
                    __m128i gaComponents = _mm_andnot_si128(brMask, sourceData);
                    // Mask out b and r
                    __m128i brComponents = _mm_and_si128(sourceData, brMask);
                    // Swap b and r
                    __m128i brSwapped = _mm_shufflehi_epi16(
                        _mm_shufflelo_epi16(brComponents, _MM_SHUFFLE(2, 3, 0, 1)),
                        _MM_SHUFFLE(2, 3, 0, 1));
                    __m128i result = _mm_or_si128(gaComponents, brSwapped);
                    _mm_store_si128(reinterpret_cast<__m128i *>(&dest[x]), result);
                }

                // Perform leftover writes
                for (; x < width; x++)
                {
                    uint32_t rgba = source[x];
                    dest[x]       = (ANGLE_ROTL(rgba, 16) & 0x00ff00ff) | (rgba & 0xff00ff00);
                }
            }
        }

        return;
    }
#endif

    for (size_t z = 0; z < depth; z++)
    {
        for (size_t y = 0; y < height; y++)
        {
            const uint32_t *source =
                priv::OffsetDataPointer<uint32_t>(input, y, z, inputRowPitch, inputDepthPitch);
            uint32_t *dest =
                priv::OffsetDataPointer<uint32_t>(output, y, z, outputRowPitch, outputDepthPitch);
            for (size_t x = 0; x < width; x++)
            {
                uint32_t rgba = source[x];
                dest[x]       = (ANGLE_ROTL(rgba, 16) & 0x00ff00ff) | (rgba & 0xff00ff00);
            }
        }
    }
}

void LoadRGBA8ToBGRA4(size_t width,
                      size_t height,
                      size_t depth,
                      const uint8_t *input,
                      size_t inputRowPitch,
                      size_t inputDepthPitch,
                      uint8_t *output,
                      size_t outputRowPitch,
                      size_t outputDepthPitch)
{
    for (size_t z = 0; z < depth; z++)
    {
        for (size_t y = 0; y < height; y++)
        {
            const uint32_t *source =
                priv::OffsetDataPointer<uint32_t>(input, y, z, inputRowPitch, inputDepthPitch);
            uint16_t *dest =
                priv::OffsetDataPointer<uint16_t>(output, y, z, outputRowPitch, outputDepthPitch);
            for (size_t x = 0; x < width; x++)
            {
                uint32_t rgba8 = source[x];
                auto r4        = static_cast<uint16_t>((rgba8 & 0x000000FF) >> 4);
                auto g4        = static_cast<uint16_t>((rgba8 & 0x0000FF00) >> 12);
                auto b4        = static_cast<uint16_t>((rgba8 & 0x00FF0000) >> 20);
                auto a4        = static_cast<uint16_t>((rgba8 & 0xFF000000) >> 28);
                dest[x]        = (a4 << 12) | (r4 << 8) | (g4 << 4) | b4;
            }
        }
    }
}

void LoadRGBA4ToARGB4(size_t width,
                      size_t height,
                      size_t depth,
                      const uint8_t *input,
                      size_t inputRowPitch,
                      size_t inputDepthPitch,
                      uint8_t *output,
                      size_t outputRowPitch,
                      size_t outputDepthPitch)
{
    for (size_t z = 0; z < depth; z++)
    {
        for (size_t y = 0; y < height; y++)
        {
            const uint16_t *source =
                priv::OffsetDataPointer<uint16_t>(input, y, z, inputRowPitch, inputDepthPitch);
            uint16_t *dest =
                priv::OffsetDataPointer<uint16_t>(output, y, z, outputRowPitch, outputDepthPitch);
            for (size_t x = 0; x < width; x++)
            {
                dest[x] = ANGLE_ROTR16(source[x], 4);
            }
        }
    }
}

void LoadRGBA4ToBGRA8(size_t width,
                      size_t height,
                      size_t depth,
                      const uint8_t *input,
                      size_t inputRowPitch,
                      size_t inputDepthPitch,
                      uint8_t *output,
                      size_t outputRowPitch,
                      size_t outputDepthPitch)
{
    for (size_t z = 0; z < depth; z++)
    {
        for (size_t y = 0; y < height; y++)
        {
            const uint16_t *source =
                priv::OffsetDataPointer<uint16_t>(input, y, z, inputRowPitch, inputDepthPitch);
            uint8_t *dest =
                priv::OffsetDataPointer<uint8_t>(output, y, z, outputRowPitch, outputDepthPitch);
            for (size_t x = 0; x < width; x++)
            {
                uint16_t rgba = source[x];
                dest[4 * x + 0] =
                    static_cast<uint8_t>(((rgba & 0x00F0) << 0) | ((rgba & 0x00F0) >> 4));
                dest[4 * x + 1] =
                    static_cast<uint8_t>(((rgba & 0x0F00) >> 4) | ((rgba & 0x0F00) >> 8));
                dest[4 * x + 2] =
                    static_cast<uint8_t>(((rgba & 0xF000) >> 8) | ((rgba & 0xF000) >> 12));
                dest[4 * x + 3] =
                    static_cast<uint8_t>(((rgba & 0x000F) << 4) | ((rgba & 0x000F) >> 0));
            }
        }
    }
}

void LoadRGBA4ToRGBA8(size_t width,
                      size_t height,
                      size_t depth,
                      const uint8_t *input,
                      size_t inputRowPitch,
                      size_t inputDepthPitch,
                      uint8_t *output,
                      size_t outputRowPitch,
                      size_t outputDepthPitch)
{
    for (size_t z = 0; z < depth; z++)
    {
        for (size_t y = 0; y < height; y++)
        {
            const uint16_t *source =
                priv::OffsetDataPointer<uint16_t>(input, y, z, inputRowPitch, inputDepthPitch);
            uint8_t *dest =
                priv::OffsetDataPointer<uint8_t>(output, y, z, outputRowPitch, outputDepthPitch);
            for (size_t x = 0; x < width; x++)
            {
                uint16_t rgba = source[x];
                dest[4 * x + 0] =
                    static_cast<uint8_t>(((rgba & 0xF000) >> 8) | ((rgba & 0xF000) >> 12));
                dest[4 * x + 1] =
                    static_cast<uint8_t>(((rgba & 0x0F00) >> 4) | ((rgba & 0x0F00) >> 8));
                dest[4 * x + 2] =
                    static_cast<uint8_t>(((rgba & 0x00F0) << 0) | ((rgba & 0x00F0) >> 4));
                dest[4 * x + 3] =
                    static_cast<uint8_t>(((rgba & 0x000F) << 4) | ((rgba & 0x000F) >> 0));
            }
        }
    }
}

void LoadBGRA4ToBGRA8(size_t width,
                      size_t height,
                      size_t depth,
                      const uint8_t *input,
                      size_t inputRowPitch,
                      size_t inputDepthPitch,
                      uint8_t *output,
                      size_t outputRowPitch,
                      size_t outputDepthPitch)
{
    for (size_t z = 0; z < depth; z++)
    {
        for (size_t y = 0; y < height; y++)
        {
            const uint16_t *source =
                priv::OffsetDataPointer<uint16_t>(input, y, z, inputRowPitch, inputDepthPitch);
            uint8_t *dest =
                priv::OffsetDataPointer<uint8_t>(output, y, z, outputRowPitch, outputDepthPitch);
            for (size_t x = 0; x < width; x++)
            {
                uint16_t bgra = source[x];
                dest[4 * x + 0] =
                    static_cast<uint8_t>(((bgra & 0xF000) >> 8) | ((bgra & 0xF000) >> 12));
                dest[4 * x + 1] =
                    static_cast<uint8_t>(((bgra & 0x0F00) >> 4) | ((bgra & 0x0F00) >> 8));
                dest[4 * x + 2] =
                    static_cast<uint8_t>(((bgra & 0x00F0) << 0) | ((bgra & 0x00F0) >> 4));
                dest[4 * x + 3] =
                    static_cast<uint8_t>(((bgra & 0x000F) << 4) | ((bgra & 0x000F) >> 0));
            }
        }
    }
}

void LoadRGBA8ToBGR5A1(size_t width,
                       size_t height,
                       size_t depth,
                       const uint8_t *input,
                       size_t inputRowPitch,
                       size_t inputDepthPitch,
                       uint8_t *output,
                       size_t outputRowPitch,
                       size_t outputDepthPitch)
{
    for (size_t z = 0; z < depth; z++)
    {
        for (size_t y = 0; y < height; y++)
        {
            const uint32_t *source =
                priv::OffsetDataPointer<uint32_t>(input, y, z, inputRowPitch, inputDepthPitch);
            uint16_t *dest =
                priv::OffsetDataPointer<uint16_t>(output, y, z, outputRowPitch, outputDepthPitch);
            for (size_t x = 0; x < width; x++)
            {
                uint32_t rgba8 = source[x];
                auto r5        = static_cast<uint16_t>((rgba8 & 0x000000FF) >> 3);
                auto g5        = static_cast<uint16_t>((rgba8 & 0x0000FF00) >> 11);
                auto b5        = static_cast<uint16_t>((rgba8 & 0x00FF0000) >> 19);
                auto a1        = static_cast<uint16_t>((rgba8 & 0xFF000000) >> 31);
                dest[x]        = (a1 << 15) | (r5 << 10) | (g5 << 5) | b5;
            }
        }
    }
}

void LoadRGB10A2ToBGR5A1(size_t width,
                         size_t height,
                         size_t depth,
                         const uint8_t *input,
                         size_t inputRowPitch,
                         size_t inputDepthPitch,
                         uint8_t *output,
                         size_t outputRowPitch,
                         size_t outputDepthPitch)
{
    for (size_t z = 0; z < depth; z++)
    {
        for (size_t y = 0; y < height; y++)
        {
            const R10G10B10A2 *source =
                priv::OffsetDataPointer<R10G10B10A2>(input, y, z, inputRowPitch, inputDepthPitch);
            uint16_t *dest =
                priv::OffsetDataPointer<uint16_t>(output, y, z, outputRowPitch, outputDepthPitch);
            for (size_t x = 0; x < width; x++)
            {
                R10G10B10A2 rgb10a2 = source[x];

                uint16_t r5 = static_cast<uint16_t>(rgb10a2.R >> 5u);
                uint16_t g5 = static_cast<uint16_t>(rgb10a2.G >> 5u);
                uint16_t b5 = static_cast<uint16_t>(rgb10a2.B >> 5u);
                uint16_t a1 = static_cast<uint16_t>(rgb10a2.A >> 1u);

                dest[x] = (a1 << 15) | (r5 << 10) | (g5 << 5) | b5;
            }
        }
    }
}

void LoadRGB5A1ToA1RGB5(size_t width,
                        size_t height,
                        size_t depth,
                        const uint8_t *input,
                        size_t inputRowPitch,
                        size_t inputDepthPitch,
                        uint8_t *output,
                        size_t outputRowPitch,
                        size_t outputDepthPitch)
{
    for (size_t z = 0; z < depth; z++)
    {
        for (size_t y = 0; y < height; y++)
        {
            const uint16_t *source =
                priv::OffsetDataPointer<uint16_t>(input, y, z, inputRowPitch, inputDepthPitch);
            uint16_t *dest =
                priv::OffsetDataPointer<uint16_t>(output, y, z, outputRowPitch, outputDepthPitch);
            for (size_t x = 0; x < width; x++)
            {
                dest[x] = ANGLE_ROTR16(source[x], 1);
            }
        }
    }
}

void LoadRGB5A1ToBGRA8(size_t width,
                       size_t height,
                       size_t depth,
                       const uint8_t *input,
                       size_t inputRowPitch,
                       size_t inputDepthPitch,
                       uint8_t *output,
                       size_t outputRowPitch,
                       size_t outputDepthPitch)
{
    for (size_t z = 0; z < depth; z++)
    {
        for (size_t y = 0; y < height; y++)
        {
            const uint16_t *source =
                priv::OffsetDataPointer<uint16_t>(input, y, z, inputRowPitch, inputDepthPitch);
            uint8_t *dest =
                priv::OffsetDataPointer<uint8_t>(output, y, z, outputRowPitch, outputDepthPitch);
            for (size_t x = 0; x < width; x++)
            {
                uint16_t rgba = source[x];
                dest[4 * x + 0] =
                    static_cast<uint8_t>(((rgba & 0x003E) << 2) | ((rgba & 0x003E) >> 3));
                dest[4 * x + 1] =
                    static_cast<uint8_t>(((rgba & 0x07C0) >> 3) | ((rgba & 0x07C0) >> 8));
                dest[4 * x + 2] =
                    static_cast<uint8_t>(((rgba & 0xF800) >> 8) | ((rgba & 0xF800) >> 13));
                dest[4 * x + 3] = static_cast<uint8_t>((rgba & 0x0001) ? 0xFF : 0);
            }
        }
    }
}

void LoadRGB5A1ToRGBA8(size_t width,
                       size_t height,
                       size_t depth,
                       const uint8_t *input,
                       size_t inputRowPitch,
                       size_t inputDepthPitch,
                       uint8_t *output,
                       size_t outputRowPitch,
                       size_t outputDepthPitch)
{
    for (size_t z = 0; z < depth; z++)
    {
        for (size_t y = 0; y < height; y++)
        {
            const uint16_t *source =
                priv::OffsetDataPointer<uint16_t>(input, y, z, inputRowPitch, inputDepthPitch);
            uint8_t *dest =
                priv::OffsetDataPointer<uint8_t>(output, y, z, outputRowPitch, outputDepthPitch);
            for (size_t x = 0; x < width; x++)
            {
                uint16_t rgba = source[x];
                dest[4 * x + 0] =
                    static_cast<uint8_t>(((rgba & 0xF800) >> 8) | ((rgba & 0xF800) >> 13));
                dest[4 * x + 1] =
                    static_cast<uint8_t>(((rgba & 0x07C0) >> 3) | ((rgba & 0x07C0) >> 8));
                dest[4 * x + 2] =
                    static_cast<uint8_t>(((rgba & 0x003E) << 2) | ((rgba & 0x003E) >> 3));
                dest[4 * x + 3] = static_cast<uint8_t>((rgba & 0x0001) ? 0xFF : 0);
            }
        }
    }
}

void LoadBGR5A1ToBGRA8(size_t width,
                       size_t height,
                       size_t depth,
                       const uint8_t *input,
                       size_t inputRowPitch,
                       size_t inputDepthPitch,
                       uint8_t *output,
                       size_t outputRowPitch,
                       size_t outputDepthPitch)
{
    for (size_t z = 0; z < depth; z++)
    {
        for (size_t y = 0; y < height; y++)
        {
            const uint16_t *source =
                priv::OffsetDataPointer<uint16_t>(input, y, z, inputRowPitch, inputDepthPitch);
            uint8_t *dest =
                priv::OffsetDataPointer<uint8_t>(output, y, z, outputRowPitch, outputDepthPitch);
            for (size_t x = 0; x < width; x++)
            {
                uint16_t bgra = source[x];
                dest[4 * x + 0] =
                    static_cast<uint8_t>(((bgra & 0xF800) >> 8) | ((bgra & 0xF800) >> 13));
                dest[4 * x + 1] =
                    static_cast<uint8_t>(((bgra & 0x07C0) >> 3) | ((bgra & 0x07C0) >> 8));
                dest[4 * x + 2] =
                    static_cast<uint8_t>(((bgra & 0x003E) << 2) | ((bgra & 0x003E) >> 3));
                dest[4 * x + 3] = static_cast<uint8_t>((bgra & 0x0001) ? 0xFF : 0);
            }
        }
    }
}

void LoadRGB10A2ToRGBA8(size_t width,
                        size_t height,
                        size_t depth,
                        const uint8_t *input,
                        size_t inputRowPitch,
                        size_t inputDepthPitch,
                        uint8_t *output,
                        size_t outputRowPitch,
                        size_t outputDepthPitch)
{
    for (size_t z = 0; z < depth; z++)
    {
        for (size_t y = 0; y < height; y++)
        {
            const uint32_t *source =
                priv::OffsetDataPointer<uint32_t>(input, y, z, inputRowPitch, inputDepthPitch);
            uint8_t *dest =
                priv::OffsetDataPointer<uint8_t>(output, y, z, outputRowPitch, outputDepthPitch);
            for (size_t x = 0; x < width; x++)
            {
                uint32_t rgba   = source[x];
                dest[4 * x + 0] = static_cast<uint8_t>((rgba & 0x000003FF) >> 2);
                dest[4 * x + 1] = static_cast<uint8_t>((rgba & 0x000FFC00) >> 12);
                dest[4 * x + 2] = static_cast<uint8_t>((rgba & 0x3FF00000) >> 22);
                dest[4 * x + 3] = static_cast<uint8_t>(((rgba & 0xC0000000) >> 30) * 0x55);
            }
        }
    }
}

void LoadRGB10A2ToRGB10X2(size_t width,
                          size_t height,
                          size_t depth,
                          const uint8_t *input,
                          size_t inputRowPitch,
                          size_t inputDepthPitch,
                          uint8_t *output,
                          size_t outputRowPitch,
                          size_t outputDepthPitch)
{
    for (size_t z = 0; z < depth; z++)
    {
        for (size_t y = 0; y < height; y++)
        {
            const uint32_t *source =
                priv::OffsetDataPointer<uint32_t>(input, y, z, inputRowPitch, inputDepthPitch);
            uint32_t *dest =
                priv::OffsetDataPointer<uint32_t>(output, y, z, outputRowPitch, outputDepthPitch);
            for (size_t x = 0; x < width; x++)
            {
                dest[x] = source[x] | 0xC0000000;
            }
        }
    }
}

void LoadRGB16FToRGB9E5(size_t width,
                        size_t height,
                        size_t depth,
                        const uint8_t *input,
                        size_t inputRowPitch,
                        size_t inputDepthPitch,
                        uint8_t *output,
                        size_t outputRowPitch,
                        size_t outputDepthPitch)
{
    for (size_t z = 0; z < depth; z++)
    {
        for (size_t y = 0; y < height; y++)
        {
            const uint16_t *source =
                priv::OffsetDataPointer<uint16_t>(input, y, z, inputRowPitch, inputDepthPitch);
            uint32_t *dest =
                priv::OffsetDataPointer<uint32_t>(output, y, z, outputRowPitch, outputDepthPitch);
            for (size_t x = 0; x < width; x++)
            {
                dest[x] = gl::convertRGBFloatsTo999E5(gl::float16ToFloat32(source[x * 3 + 0]),
                                                      gl::float16ToFloat32(source[x * 3 + 1]),
                                                      gl::float16ToFloat32(source[x * 3 + 2]));
            }
        }
    }
}

void LoadRGB32FToRGB9E5(size_t width,
                        size_t height,
                        size_t depth,
                        const uint8_t *input,
                        size_t inputRowPitch,
                        size_t inputDepthPitch,
                        uint8_t *output,
                        size_t outputRowPitch,
                        size_t outputDepthPitch)
{
    for (size_t z = 0; z < depth; z++)
    {
        for (size_t y = 0; y < height; y++)
        {
            const float *source =
                priv::OffsetDataPointer<float>(input, y, z, inputRowPitch, inputDepthPitch);
            uint32_t *dest =
                priv::OffsetDataPointer<uint32_t>(output, y, z, outputRowPitch, outputDepthPitch);
            for (size_t x = 0; x < width; x++)
            {
                dest[x] = gl::convertRGBFloatsTo999E5(source[x * 3 + 0], source[x * 3 + 1],
                                                      source[x * 3 + 2]);
            }
        }
    }
}

void LoadRGB16FToRG11B10F(size_t width,
                          size_t height,
                          size_t depth,
                          const uint8_t *input,
                          size_t inputRowPitch,
                          size_t inputDepthPitch,
                          uint8_t *output,
                          size_t outputRowPitch,
                          size_t outputDepthPitch)
{
    for (size_t z = 0; z < depth; z++)
    {
        for (size_t y = 0; y < height; y++)
        {
            const uint16_t *source =
                priv::OffsetDataPointer<uint16_t>(input, y, z, inputRowPitch, inputDepthPitch);
            uint32_t *dest =
                priv::OffsetDataPointer<uint32_t>(output, y, z, outputRowPitch, outputDepthPitch);
            for (size_t x = 0; x < width; x++)
            {
                dest[x] = (gl::float32ToFloat11(gl::float16ToFloat32(source[x * 3 + 0])) << 0) |
                          (gl::float32ToFloat11(gl::float16ToFloat32(source[x * 3 + 1])) << 11) |
                          (gl::float32ToFloat10(gl::float16ToFloat32(source[x * 3 + 2])) << 22);
            }
        }
    }
}

void LoadRGB32FToRG11B10F(size_t width,
                          size_t height,
                          size_t depth,
                          const uint8_t *input,
                          size_t inputRowPitch,
                          size_t inputDepthPitch,
                          uint8_t *output,
                          size_t outputRowPitch,
                          size_t outputDepthPitch)
{
    for (size_t z = 0; z < depth; z++)
    {
        for (size_t y = 0; y < height; y++)
        {
            const float *source =
                priv::OffsetDataPointer<float>(input, y, z, inputRowPitch, inputDepthPitch);
            uint32_t *dest =
                priv::OffsetDataPointer<uint32_t>(output, y, z, outputRowPitch, outputDepthPitch);
            for (size_t x = 0; x < width; x++)
            {
                dest[x] = (gl::float32ToFloat11(source[x * 3 + 0]) << 0) |
                          (gl::float32ToFloat11(source[x * 3 + 1]) << 11) |
                          (gl::float32ToFloat10(source[x * 3 + 2]) << 22);
            }
        }
    }
}

void LoadG8R24ToR24G8(size_t width,
                      size_t height,
                      size_t depth,
                      const uint8_t *input,
                      size_t inputRowPitch,
                      size_t inputDepthPitch,
                      uint8_t *output,
                      size_t outputRowPitch,
                      size_t outputDepthPitch)
{
    for (size_t z = 0; z < depth; z++)
    {
        for (size_t y = 0; y < height; y++)
        {
            const uint32_t *source =
                priv::OffsetDataPointer<uint32_t>(input, y, z, inputRowPitch, inputDepthPitch);
            uint32_t *dest =
                priv::OffsetDataPointer<uint32_t>(output, y, z, outputRowPitch, outputDepthPitch);
            for (size_t x = 0; x < width; x++)
            {
                uint32_t d = source[x] >> 8;
                uint8_t s  = source[x] & 0xFF;
                dest[x]    = d | (s << 24);
            }
        }
    }
}

void LoadD24S8ToD32FS8X24(size_t width,
                          size_t height,
                          size_t depth,
                          const uint8_t *input,
                          size_t inputRowPitch,
                          size_t inputDepthPitch,
                          uint8_t *output,
                          size_t outputRowPitch,
                          size_t outputDepthPitch)
{
    for (size_t z = 0; z < depth; z++)
    {
        for (size_t y = 0; y < height; y++)
        {
            const uint32_t *source =
                priv::OffsetDataPointer<uint32_t>(input, y, z, inputRowPitch, inputDepthPitch);
            float *destDepth =
                priv::OffsetDataPointer<float>(output, y, z, outputRowPitch, outputDepthPitch);
            uint32_t *destStencil =
                priv::OffsetDataPointer<uint32_t>(output, y, z, outputRowPitch, outputDepthPitch) +
                1;
            for (size_t x = 0; x < width; x++)
            {
                destDepth[x * 2]   = (source[x] & 0xFFFFFF) / static_cast<float>(0xFFFFFF);
                destStencil[x * 2] = source[x] & 0xFF000000;
            }
        }
    }
}

void LoadD24S8ToD32F(size_t width,
                     size_t height,
                     size_t depth,
                     const uint8_t *input,
                     size_t inputRowPitch,
                     size_t inputDepthPitch,
                     uint8_t *output,
                     size_t outputRowPitch,
                     size_t outputDepthPitch)
{
    for (size_t z = 0; z < depth; z++)
    {
        for (size_t y = 0; y < height; y++)
        {
            const uint32_t *source =
                priv::OffsetDataPointer<uint32_t>(input, y, z, inputRowPitch, inputDepthPitch);
            float *destDepth =
                priv::OffsetDataPointer<float>(output, y, z, outputRowPitch, outputDepthPitch);
            for (size_t x = 0; x < width; x++)
            {
                uint32_t sourcePixel = (source[x] >> 8) & 0xFFFFFF;
                destDepth[x]         = sourcePixel / static_cast<float>(0xFFFFFF);
            }
        }
    }
}

void LoadD32ToD32FX32(size_t width,
                      size_t height,
                      size_t depth,
                      const uint8_t *input,
                      size_t inputRowPitch,
                      size_t inputDepthPitch,
                      uint8_t *output,
                      size_t outputRowPitch,
                      size_t outputDepthPitch)
{
    for (size_t z = 0; z < depth; z++)
    {
        for (size_t y = 0; y < height; y++)
        {
            const uint32_t *source =
                priv::OffsetDataPointer<uint32_t>(input, y, z, inputRowPitch, inputDepthPitch);
            float *destDepth =
                priv::OffsetDataPointer<float>(output, y, z, outputRowPitch, outputDepthPitch);
            for (size_t x = 0; x < width; x++)
            {
                destDepth[x * 2] = source[x] / static_cast<float>(0xFFFFFFFF);
            }
        }
    }
}

void LoadD32ToD32F(size_t width,
                   size_t height,
                   size_t depth,
                   const uint8_t *input,
                   size_t inputRowPitch,
                   size_t inputDepthPitch,
                   uint8_t *output,
                   size_t outputRowPitch,
                   size_t outputDepthPitch)
{
    for (size_t z = 0; z < depth; z++)
    {
        for (size_t y = 0; y < height; y++)
        {
            const uint32_t *source =
                priv::OffsetDataPointer<uint32_t>(input, y, z, inputRowPitch, inputDepthPitch);
            float *destDepth =
                priv::OffsetDataPointer<float>(output, y, z, outputRowPitch, outputDepthPitch);
            for (size_t x = 0; x < width; x++)
            {
                uint32_t sourcePixel = source[x];
                destDepth[x]         = sourcePixel / static_cast<float>(0xFFFFFFFF);
            }
        }
    }
}

void LoadD32FToD32F(size_t width,
                    size_t height,
                    size_t depth,
                    const uint8_t *input,
                    size_t inputRowPitch,
                    size_t inputDepthPitch,
                    uint8_t *output,
                    size_t outputRowPitch,
                    size_t outputDepthPitch)
{
    for (size_t z = 0; z < depth; z++)
    {
        for (size_t y = 0; y < height; y++)
        {
            const float *source =
                priv::OffsetDataPointer<float>(input, y, z, inputRowPitch, inputDepthPitch);
            float *dest =
                priv::OffsetDataPointer<float>(output, y, z, outputRowPitch, outputDepthPitch);
            for (size_t x = 0; x < width; x++)
            {
                dest[x] = gl::clamp01(source[x]);
            }
        }
    }
}

void LoadD32FS8X24ToD24S8(size_t width,
                          size_t height,
                          size_t depth,
                          const uint8_t *input,
                          size_t inputRowPitch,
                          size_t inputDepthPitch,
                          uint8_t *output,
                          size_t outputRowPitch,
                          size_t outputDepthPitch)
{
    for (size_t z = 0; z < depth; z++)
    {
        for (size_t y = 0; y < height; y++)
        {
            const float *sourceDepth =
                priv::OffsetDataPointer<float>(input, y, z, inputRowPitch, inputDepthPitch);
            const uint32_t *sourceStencil =
                priv::OffsetDataPointer<uint32_t>(input, y, z, inputRowPitch, inputDepthPitch) + 1;
            uint32_t *dest =
                priv::OffsetDataPointer<uint32_t>(output, y, z, outputRowPitch, outputDepthPitch);
            for (size_t x = 0; x < width; x++)
            {
                uint32_t d = static_cast<uint32_t>(gl::clamp01(sourceDepth[x * 2]) * 0xFFFFFF);
                uint32_t s = sourceStencil[x * 2] & 0xFF000000;
                dest[x]    = d | s;
            }
        }
    }
}

void LoadX24S8ToS8(size_t width,
                   size_t height,
                   size_t depth,
                   const uint8_t *input,
                   size_t inputRowPitch,
                   size_t inputDepthPitch,
                   uint8_t *output,
                   size_t outputRowPitch,
                   size_t outputDepthPitch)
{
    for (size_t z = 0; z < depth; z++)
    {
        for (size_t y = 0; y < height; y++)
        {
            const uint32_t *source = reinterpret_cast<const uint32_t *>(
                input + (y * inputRowPitch) + (z * inputDepthPitch));
            uint8_t *destStencil =
                reinterpret_cast<uint8_t *>(output + (y * outputRowPitch) + (z * outputDepthPitch));
            for (size_t x = 0; x < width; x++)
            {
                destStencil[x] = (source[x] & 0xFF);
            }
        }
    }
}

void LoadX32S8ToS8(size_t width,
                   size_t height,
                   size_t depth,
                   const uint8_t *input,
                   size_t inputRowPitch,
                   size_t inputDepthPitch,
                   uint8_t *output,
                   size_t outputRowPitch,
                   size_t outputDepthPitch)
{
    for (size_t z = 0; z < depth; z++)
    {
        for (size_t y = 0; y < height; y++)
        {
            const uint32_t *source = reinterpret_cast<const uint32_t *>(
                input + (y * inputRowPitch) + (z * inputDepthPitch));
            uint8_t *destStencil =
                reinterpret_cast<uint8_t *>(output + (y * outputRowPitch) + (z * outputDepthPitch));
            for (size_t x = 0; x < width; x++)
            {
                destStencil[x] = (source[(x * 2) + 1] & 0xFF);
            }
        }
    }
}

void LoadD32FS8X24ToD32F(size_t width,
                         size_t height,
                         size_t depth,
                         const uint8_t *input,
                         size_t inputRowPitch,
                         size_t inputDepthPitch,
                         uint8_t *output,
                         size_t outputRowPitch,
                         size_t outputDepthPitch)
{
    for (size_t z = 0; z < depth; z++)
    {
        for (size_t y = 0; y < height; y++)
        {
            const float *sourceDepth =
                priv::OffsetDataPointer<float>(input, y, z, inputRowPitch, inputDepthPitch);
            float *destDepth =
                priv::OffsetDataPointer<float>(output, y, z, outputRowPitch, outputDepthPitch);
            for (size_t x = 0; x < width; x++)
            {
                destDepth[x] = gl::clamp01(sourceDepth[x * 2]);
            }
        }
    }
}

void LoadD32FS8X24ToD32FS8X24(size_t width,
                              size_t height,
                              size_t depth,
                              const uint8_t *input,
                              size_t inputRowPitch,
                              size_t inputDepthPitch,
                              uint8_t *output,
                              size_t outputRowPitch,
                              size_t outputDepthPitch)
{
    for (size_t z = 0; z < depth; z++)
    {
        for (size_t y = 0; y < height; y++)
        {
            const float *sourceDepth =
                priv::OffsetDataPointer<float>(input, y, z, inputRowPitch, inputDepthPitch);
            const uint32_t *sourceStencil =
                priv::OffsetDataPointer<uint32_t>(input, y, z, inputRowPitch, inputDepthPitch) + 1;
            float *destDepth =
                priv::OffsetDataPointer<float>(output, y, z, outputRowPitch, outputDepthPitch);
            uint32_t *destStencil =
                priv::OffsetDataPointer<uint32_t>(output, y, z, outputRowPitch, outputDepthPitch) +
                1;
            for (size_t x = 0; x < width; x++)
            {
                destDepth[x * 2]   = gl::clamp01(sourceDepth[x * 2]);
                destStencil[x * 2] = sourceStencil[x * 2] & 0xFF000000;
            }
        }
    }
}

void LoadRGB32FToRGBA16F(size_t width,
                         size_t height,
                         size_t depth,
                         const uint8_t *input,
                         size_t inputRowPitch,
                         size_t inputDepthPitch,
                         uint8_t *output,
                         size_t outputRowPitch,
                         size_t outputDepthPitch)
{
    for (size_t z = 0; z < depth; z++)
    {
        for (size_t y = 0; y < height; y++)
        {
            const float *source =
                priv::OffsetDataPointer<float>(input, y, z, inputRowPitch, inputDepthPitch);
            uint16_t *dest =
                priv::OffsetDataPointer<uint16_t>(output, y, z, outputRowPitch, outputDepthPitch);
            for (size_t x = 0; x < width; x++)
            {
                dest[x * 4 + 0] = gl::float32ToFloat16(source[x * 3 + 0]);
                dest[x * 4 + 1] = gl::float32ToFloat16(source[x * 3 + 1]);
                dest[x * 4 + 2] = gl::float32ToFloat16(source[x * 3 + 2]);
                dest[x * 4 + 3] = gl::Float16One;
            }
        }
    }
}

void LoadRGB32FToRGB16F(size_t width,
                        size_t height,
                        size_t depth,
                        const uint8_t *input,
                        size_t inputRowPitch,
                        size_t inputDepthPitch,
                        uint8_t *output,
                        size_t outputRowPitch,
                        size_t outputDepthPitch)
{
    for (size_t z = 0; z < depth; z++)
    {
        for (size_t y = 0; y < height; y++)
        {
            const float *source =
                priv::OffsetDataPointer<float>(input, y, z, inputRowPitch, inputDepthPitch);
            uint16_t *dest =
                priv::OffsetDataPointer<uint16_t>(output, y, z, outputRowPitch, outputDepthPitch);
            for (size_t x = 0; x < width; x++)
            {
                dest[x * 3 + 0] = gl::float32ToFloat16(source[x * 3 + 0]);
                dest[x * 3 + 1] = gl::float32ToFloat16(source[x * 3 + 1]);
                dest[x * 3 + 2] = gl::float32ToFloat16(source[x * 3 + 2]);
            }
        }
    }
}

void LoadRGBDXT1ToRGBADXT3(size_t width,
                           size_t height,
                           size_t depth,
                           const uint8_t *input,
                           size_t inputRowPitch,
                           size_t inputDepthPitch,
                           uint8_t *output,
                           size_t outputRowPitch,
                           size_t outputDepthPitch)
{
    constexpr size_t kBlockWidth  = 4;
    constexpr size_t kBlockHeight = 4;
    constexpr size_t kBlockDepth  = 1;
    constexpr size_t kBlockSize   = 8;

    const size_t columns = (width + (kBlockWidth - 1)) / kBlockWidth;
    const size_t rows    = (height + (kBlockHeight - 1)) / kBlockHeight;
    const size_t layers  = (depth + (kBlockDepth - 1)) / kBlockDepth;

    for (size_t z = 0; z < layers; ++z)
    {
        for (size_t y = 0; y < rows; ++y)
        {
            for (size_t x = 0; x < columns; x++)
            {
                const uint8_t *source =
                    priv::OffsetDataPointer<uint8_t>(input, y, z, inputRowPitch, inputDepthPitch) +
                    (x * kBlockSize);
                uint8_t *dest = priv::OffsetDataPointer<uint8_t>(output, y, z, outputRowPitch,
                                                                 outputDepthPitch) +
                                (x * kBlockSize * 2);

                // DXT3 has a block for alpha then a block for color
                memset(dest, 0xFF, kBlockSize);
                memcpy(dest + kBlockSize, source, kBlockSize);
            }
        }
    }
}

void LoadR32ToR16(size_t width,
                  size_t height,
                  size_t depth,
                  const uint8_t *input,
                  size_t inputRowPitch,
                  size_t inputDepthPitch,
                  uint8_t *output,
                  size_t outputRowPitch,
                  size_t outputDepthPitch)
{
    for (size_t z = 0; z < depth; z++)
    {
        for (size_t y = 0; y < height; y++)
        {
            const uint32_t *source =
                priv::OffsetDataPointer<uint32_t>(input, y, z, inputRowPitch, inputDepthPitch);
            uint16_t *dest =
                priv::OffsetDataPointer<uint16_t>(output, y, z, outputRowPitch, outputDepthPitch);
            for (size_t x = 0; x < width; x++)
            {
                dest[x] = source[x] >> 16;
            }
        }
    }
}

void LoadR32ToR24G8(size_t width,
                    size_t height,
                    size_t depth,
                    const uint8_t *input,
                    size_t inputRowPitch,
                    size_t inputDepthPitch,
                    uint8_t *output,
                    size_t outputRowPitch,
                    size_t outputDepthPitch)
{
    for (size_t z = 0; z < depth; z++)
    {
        for (size_t y = 0; y < height; y++)
        {
            const uint32_t *source =
                priv::OffsetDataPointer<uint32_t>(input, y, z, inputRowPitch, inputDepthPitch);
            uint32_t *dest =
                priv::OffsetDataPointer<uint32_t>(output, y, z, outputRowPitch, outputDepthPitch);

            for (size_t x = 0; x < width; x++)
            {
                dest[x] = source[x] >> 8;
            }
        }
    }
}

}  // namespace angle
