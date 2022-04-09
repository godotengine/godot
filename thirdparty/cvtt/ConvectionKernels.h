/*
Convection Texture Tools
Copyright (c) 2018 Eric Lasota

Permission is hereby granted, free of charge, to any person obtaining
a copy of this software and associated documentation files (the
"Software"), to deal in the Software without restriction, including
without limitation the rights to use, copy, modify, merge, publish,
distribute, sublicense, and/or sell copies of the Software, and to
permit persons to whom the Software is furnished to do so, subject
to the following conditions:

The above copyright notice and this permission notice shall be included
in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
*/
#pragma once
#ifndef __CVTT_CONVECTION_KERNELS__
#define __CVTT_CONVECTION_KERNELS__

#include <stddef.h>
#include <stdint.h>

namespace cvtt
{
    namespace Flags
    {
        // Use fast indexing in BC7 encoding (about 2x faster, slightly worse quality)
        const uint32_t BC7_FastIndexing         = 0x008;

        // Try precomputed single-color lookups where applicable (slightly slower, small quality increase on specific blocks)
        const uint32_t BC7_TrySingleColor       = 0x010;

        // Don't allow non-zero or non-max alpha values in blocks that only contain one or the other
        const uint32_t BC7_RespectPunchThrough  = 0x020;

        // Use fast indexing in HDR formats (faster, worse quality)
        const uint32_t BC6H_FastIndexing        = 0x040;

        // Exhaustive search RGB orderings when encoding BC1-BC3 (much slower, better quality)
        const uint32_t S3TC_Exhaustive          = 0x080;

        // Penalize distant endpoints, improving quality on inaccurate GPU decoders
        const uint32_t S3TC_Paranoid            = 0x100;

        // Uniform color channel importance
        const uint32_t Uniform                  = 0x200;

        // Use fake BT.709 color space for etc2comp compatibility (slower)
        const uint32_t ETC_UseFakeBT709         = 0x400;

        // Use accurate quantization functions when quantizing fake BT.709 (much slower, marginal improvement on specific blocks)
        const uint32_t ETC_FakeBT709Accurate    = 0x800;

        // Misc useful default flag combinations
        const uint32_t Fastest = (BC6H_FastIndexing | BC7_FastIndexing | S3TC_Paranoid);
        const uint32_t Faster = (BC6H_FastIndexing | BC7_FastIndexing | S3TC_Paranoid);
        const uint32_t Fast = (BC7_FastIndexing | S3TC_Paranoid);
        const uint32_t Default = (BC7_FastIndexing | S3TC_Paranoid);
        const uint32_t Better = (S3TC_Paranoid | S3TC_Exhaustive);
        const uint32_t Ultra = (BC7_TrySingleColor | S3TC_Paranoid | S3TC_Exhaustive | ETC_FakeBT709Accurate);
    }

    const unsigned int NumParallelBlocks = 8;

    struct Options
    {
        uint32_t flags;         // Bitmask of cvtt::Flags values
        float threshold;        // Alpha test threshold for BC1
        float redWeight;        // Red channel importance
        float greenWeight;      // Green channel importance
        float blueWeight;       // Blue channel importance
        float alphaWeight;      // Alpha channel importance

        int refineRoundsBC7;   // Number of refine rounds for BC7
        int refineRoundsBC6H;   // Number of refine rounds for BC6H (max 3)
        int refineRoundsIIC;    // Number of refine rounds for independent interpolated channels (BC3 alpha, BC4, BC5)
        int refineRoundsS3TC;   // Number of refine rounds for S3TC RGB

        int seedPoints;         // Number of seed points (min 1, max 4)

        Options()
            : flags(Flags::Default)
            , threshold(0.5f)
            , redWeight(0.2125f / 0.7154f)
            , greenWeight(1.0f)
            , blueWeight(0.0721f / 0.7154f)
            , alphaWeight(1.0f)
            , refineRoundsBC7(2)
            , refineRoundsBC6H(3)
            , refineRoundsIIC(8)
            , refineRoundsS3TC(2)
            , seedPoints(4)
        {
        }
    };

    struct BC7FineTuningParams
    {
        // Seed point counts for each mode+configuration combination
        uint8_t mode0SP[16];
        uint8_t mode1SP[64];
        uint8_t mode2SP[64];
        uint8_t mode3SP[64];
        uint8_t mode4SP[4][2];
        uint8_t mode5SP[4];
        uint8_t mode6SP;
        uint8_t mode7SP[64];

        BC7FineTuningParams()
        {
            for (int i = 0; i < 16; i++)
                this->mode0SP[i] = 4;

            for (int i = 0; i < 64; i++)
            {
                this->mode1SP[i] = 4;
                this->mode2SP[i] = 4;
                this->mode3SP[i] = 4;
                this->mode7SP[i] = 4;
            }

            for (int i = 0; i < 4; i++)
            {
                for (int j = 0; j < 2; j++)
                    this->mode4SP[i][j] = 4;

                this->mode5SP[i] = 4;
            }

            this->mode6SP = 4;
        }
    };

    struct BC7EncodingPlan
    {
        static const int kNumRGBAShapes = 129;
        static const int kNumRGBShapes = 243;

        uint64_t mode1PartitionEnabled;
        uint64_t mode2PartitionEnabled;
        uint64_t mode3PartitionEnabled;
        uint16_t mode0PartitionEnabled;
        uint64_t mode7RGBAPartitionEnabled;
        uint64_t mode7RGBPartitionEnabled;
        uint8_t mode4SP[4][2];
        uint8_t mode5SP[4];
        bool mode6Enabled;

        uint8_t seedPointsForShapeRGB[kNumRGBShapes];
        uint8_t seedPointsForShapeRGBA[kNumRGBAShapes];

        uint8_t rgbaShapeList[kNumRGBAShapes];
        uint8_t rgbaNumShapesToEvaluate;

        uint8_t rgbShapeList[kNumRGBShapes];
        uint8_t rgbNumShapesToEvaluate;

        BC7EncodingPlan()
        {
            for (int i = 0; i < kNumRGBShapes; i++)
            {
                this->rgbShapeList[i] = i;
                this->seedPointsForShapeRGB[i] = 4;
            }
            this->rgbNumShapesToEvaluate = kNumRGBShapes;

            for (int i = 0; i < kNumRGBAShapes; i++)
            {
                this->rgbaShapeList[i] = i;
                this->seedPointsForShapeRGBA[i] = 4;
            }
            this->rgbaNumShapesToEvaluate = kNumRGBAShapes;


            this->mode0PartitionEnabled = 0xffff;
            this->mode1PartitionEnabled = 0xffffffffffffffffULL;
            this->mode2PartitionEnabled = 0xffffffffffffffffULL;
            this->mode3PartitionEnabled = 0xffffffffffffffffULL;
            this->mode6Enabled = true;
            this->mode7RGBPartitionEnabled = 0xffffffffffffffffULL;
            this->mode7RGBAPartitionEnabled = 0xffffffffffffffffULL;

            for (int i = 0; i < 4; i++)
            {
                for (int j = 0; j < 2; j++)
                    this->mode4SP[i][j] = 4;

                this->mode5SP[i] = 4;
            }
        }
    };

    // RGBA input block for unsigned 8-bit formats
    struct PixelBlockU8
    {
        uint8_t m_pixels[16][4];
    };

    // RGBA input block for signed 8-bit formats
    struct PixelBlockS8
    {
        int8_t m_pixels[16][4];
    };

    struct PixelBlockScalarS16
    {
        int16_t m_pixels[16];
    };

    // RGBA input block for half-precision float formats (bit-cast to int16_t)
    struct PixelBlockF16
    {
        int16_t m_pixels[16][4];
    };

    class ETC2CompressionData
    {
    protected:
        ETC2CompressionData() {}
    };

    class ETC1CompressionData
    {
    protected:
        ETC1CompressionData() {}
    };

    namespace Kernels
    {
        typedef void* allocFunc_t(void *context, size_t size);
        typedef void freeFunc_t(void *context, void* ptr, size_t size);

        // NOTE: All functions accept and output NumParallelBlocks blocks at once
        void EncodeBC1(uint8_t *pBC, const PixelBlockU8 *pBlocks, const Options &options);
        void EncodeBC2(uint8_t *pBC, const PixelBlockU8 *pBlocks, const Options &options);
        void EncodeBC3(uint8_t *pBC, const PixelBlockU8 *pBlocks, const Options &options);
        void EncodeBC4U(uint8_t *pBC, const PixelBlockU8 *pBlocks, const Options &options);
        void EncodeBC4S(uint8_t *pBC, const PixelBlockS8 *pBlocks, const Options &options);
        void EncodeBC5U(uint8_t *pBC, const PixelBlockU8 *pBlocks, const Options &options);
        void EncodeBC5S(uint8_t *pBC, const PixelBlockS8 *pBlocks, const Options &options);
        void EncodeBC6HU(uint8_t *pBC, const PixelBlockF16 *pBlocks, const Options &options);
        void EncodeBC6HS(uint8_t *pBC, const PixelBlockF16 *pBlocks, const Options &options);
        void EncodeBC7(uint8_t *pBC, const PixelBlockU8 *pBlocks, const Options &options, const BC7EncodingPlan &encodingPlan);
        void EncodeETC1(uint8_t *pBC, const PixelBlockU8 *pBlocks, const Options &options, ETC1CompressionData *compressionData);
        void EncodeETC2(uint8_t *pBC, const PixelBlockU8 *pBlocks, const Options &options, ETC2CompressionData *compressionData);
        void EncodeETC2RGBA(uint8_t *pBC, const PixelBlockU8 *pBlocks, const cvtt::Options &options, cvtt::ETC2CompressionData *compressionData);
        void EncodeETC2PunchthroughAlpha(uint8_t *pBC, const PixelBlockU8 *pBlocks, const cvtt::Options &options, cvtt::ETC2CompressionData *compressionData);

        void EncodeETC2Alpha(uint8_t *pBC, const PixelBlockU8 *pBlocks, const cvtt::Options &options);
        void EncodeETC2Alpha11(uint8_t *pBC, const PixelBlockScalarS16 *pBlocks, bool isSigned, const cvtt::Options &options);

        // Generates a BC7 encoding plan from a quality parameter that ranges from 1 (fastest) to 100 (best)
        void ConfigureBC7EncodingPlanFromQuality(BC7EncodingPlan &encodingPlan, int quality);

        // Generates a BC7 encoding plan from fine-tuning parameters.
        bool ConfigureBC7EncodingPlanFromFineTuningParams(BC7EncodingPlan &encodingPlan, const BC7FineTuningParams &params);

        // ETC compression requires temporary storage that normally consumes a large amount of stack space.
        // To allocate and release it, use one of these functions.
        ETC2CompressionData *AllocETC2Data(allocFunc_t allocFunc, void *context, const cvtt::Options &options);
        void ReleaseETC2Data(ETC2CompressionData *compressionData, freeFunc_t freeFunc);

        ETC1CompressionData *AllocETC1Data(allocFunc_t allocFunc, void *context);
        void ReleaseETC1Data(ETC1CompressionData *compressionData, freeFunc_t freeFunc);

        void DecodeBC6HU(PixelBlockF16 *pBlocks, const uint8_t *pBC);
        void DecodeBC6HS(PixelBlockF16 *pBlocks, const uint8_t *pBC);
        void DecodeBC7(PixelBlockU8 *pBlocks, const uint8_t *pBC);
    }
}

#endif
