/*
Convection Texture Tools
Copyright (c) 2018-2019 Eric Lasota

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
#include "ConvectionKernels_Config.h"

#if !defined(CVTT_SINGLE_FILE) || defined(CVTT_SINGLE_FILE_IMPL)

#include <stdint.h>
#include "ConvectionKernels.h"
#include "ConvectionKernels_Util.h"
#include "ConvectionKernels_BC67.h"
#include "ConvectionKernels_ETC.h"
#include "ConvectionKernels_S3TC.h"

#include <assert.h>

namespace cvtt
{
    namespace Kernels
    {
        void EncodeBC7(uint8_t *pBC, const PixelBlockU8 *pBlocks, const cvtt::Options &options, const BC7EncodingPlan &encodingPlan)
        {
            assert(pBlocks);
            assert(pBC);

            float channelWeights[4];
            Util::FillWeights(options, channelWeights);

            for (size_t blockBase = 0; blockBase < cvtt::NumParallelBlocks; blockBase += ParallelMath::ParallelSize)
            {
                Internal::BC7Computer::Pack(options.flags, pBlocks + blockBase, pBC, channelWeights, encodingPlan, options.refineRoundsBC7);
                pBC += ParallelMath::ParallelSize * 16;
            }
        }

        void EncodeBC6HU(uint8_t *pBC, const PixelBlockF16 *pBlocks, const cvtt::Options &options)
        {
            assert(pBlocks);
            assert(pBC);

            float channelWeights[4];
            Util::FillWeights(options, channelWeights);

            for (size_t blockBase = 0; blockBase < cvtt::NumParallelBlocks; blockBase += ParallelMath::ParallelSize)
            {
                Internal::BC6HComputer::Pack(options.flags, pBlocks + blockBase, pBC, channelWeights, false, options.seedPoints, options.refineRoundsBC6H);
                pBC += ParallelMath::ParallelSize * 16;
            }
        }

        void EncodeBC6HS(uint8_t *pBC, const PixelBlockF16 *pBlocks, const cvtt::Options &options)
        {
            assert(pBlocks);
            assert(pBC);

            float channelWeights[4];
            Util::FillWeights(options, channelWeights);

            for (size_t blockBase = 0; blockBase < cvtt::NumParallelBlocks; blockBase += ParallelMath::ParallelSize)
            {
                Internal::BC6HComputer::Pack(options.flags, pBlocks + blockBase, pBC, channelWeights, true, options.seedPoints, options.refineRoundsBC6H);
                pBC += ParallelMath::ParallelSize * 16;
            }
        }

        void EncodeBC1(uint8_t *pBC, const PixelBlockU8 *pBlocks, const cvtt::Options &options)
        {
            assert(pBlocks);
            assert(pBC);

            float channelWeights[4];
            Util::FillWeights(options, channelWeights);

            for (size_t blockBase = 0; blockBase < cvtt::NumParallelBlocks; blockBase += ParallelMath::ParallelSize)
            {
                Internal::S3TCComputer::PackRGB(options.flags, pBlocks + blockBase, pBC, 8, channelWeights, true, options.threshold, (options.flags & Flags::S3TC_Exhaustive) != 0, options.seedPoints, options.refineRoundsS3TC);
                pBC += ParallelMath::ParallelSize * 8;
            }
        }

        void EncodeBC2(uint8_t *pBC, const PixelBlockU8 *pBlocks, const Options &options)
        {
            assert(pBlocks);
            assert(pBC);

            float channelWeights[4];
            Util::FillWeights(options, channelWeights);

            for (size_t blockBase = 0; blockBase < NumParallelBlocks; blockBase += ParallelMath::ParallelSize)
            {
                Internal::S3TCComputer::PackRGB(options.flags, pBlocks + blockBase, pBC + 8, 16, channelWeights, false, 1.0f, (options.flags & Flags::S3TC_Exhaustive) != 0, options.seedPoints, options.refineRoundsS3TC);
                Internal::S3TCComputer::PackExplicitAlpha(options.flags, pBlocks + blockBase, 3, pBC, 16);
                pBC += ParallelMath::ParallelSize * 16;
            }
        }

        void EncodeBC3(uint8_t *pBC, const PixelBlockU8 *pBlocks, const Options &options)
        {
            assert(pBlocks);
            assert(pBC);

            float channelWeights[4];
            Util::FillWeights(options, channelWeights);

            for (size_t blockBase = 0; blockBase < NumParallelBlocks; blockBase += ParallelMath::ParallelSize)
            {
                Internal::S3TCComputer::PackRGB(options.flags, pBlocks + blockBase, pBC + 8, 16, channelWeights, false, 1.0f, (options.flags & Flags::S3TC_Exhaustive) != 0, options.seedPoints, options.refineRoundsS3TC);
                Internal::S3TCComputer::PackInterpolatedAlpha(options.flags, pBlocks + blockBase, 3, pBC, 16, false, options.seedPoints, options.refineRoundsIIC);
                pBC += ParallelMath::ParallelSize * 16;
            }
        }

        void EncodeBC4U(uint8_t *pBC, const PixelBlockU8 *pBlocks, const Options &options)
        {
            assert(pBlocks);
            assert(pBC);

            float channelWeights[4];
            Util::FillWeights(options, channelWeights);

            for (size_t blockBase = 0; blockBase < NumParallelBlocks; blockBase += ParallelMath::ParallelSize)
            {
                Internal::S3TCComputer::PackInterpolatedAlpha(options.flags, pBlocks + blockBase, 0, pBC, 8, false, options.seedPoints, options.refineRoundsIIC);
                pBC += ParallelMath::ParallelSize * 8;
            }
        }

        void EncodeBC4S(uint8_t *pBC, const PixelBlockS8 *pBlocks, const Options &options)
        {
            assert(pBlocks);
            assert(pBC);

            float channelWeights[4];
            Util::FillWeights(options, channelWeights);

            for (size_t blockBase = 0; blockBase < NumParallelBlocks; blockBase += ParallelMath::ParallelSize)
            {
                PixelBlockU8 inputBlocks[ParallelMath::ParallelSize];
                Util::BiasSignedInput(inputBlocks, pBlocks + blockBase);

                Internal::S3TCComputer::PackInterpolatedAlpha(options.flags, inputBlocks, 0, pBC, 8, true, options.seedPoints, options.refineRoundsIIC);
                pBC += ParallelMath::ParallelSize * 8;
            }
        }

        void EncodeBC5U(uint8_t *pBC, const PixelBlockU8 *pBlocks, const Options &options)
        {
            assert(pBlocks);
            assert(pBC);

            float channelWeights[4];
            Util::FillWeights(options, channelWeights);

            for (size_t blockBase = 0; blockBase < NumParallelBlocks; blockBase += ParallelMath::ParallelSize)
            {
                Internal::S3TCComputer::PackInterpolatedAlpha(options.flags, pBlocks + blockBase, 0, pBC, 16, false, options.seedPoints, options.refineRoundsIIC);
                Internal::S3TCComputer::PackInterpolatedAlpha(options.flags, pBlocks + blockBase, 1, pBC + 8, 16, false, options.seedPoints, options.refineRoundsIIC);
                pBC += ParallelMath::ParallelSize * 16;
            }
        }

        void EncodeBC5S(uint8_t *pBC, const PixelBlockS8 *pBlocks, const Options &options)
        {
            assert(pBlocks);
            assert(pBC);

            float channelWeights[4];
            Util::FillWeights(options, channelWeights);

            for (size_t blockBase = 0; blockBase < NumParallelBlocks; blockBase += ParallelMath::ParallelSize)
            {
                PixelBlockU8 inputBlocks[ParallelMath::ParallelSize];
                Util::BiasSignedInput(inputBlocks, pBlocks + blockBase);

                Internal::S3TCComputer::PackInterpolatedAlpha(options.flags, inputBlocks, 0, pBC, 16, true, options.seedPoints, options.refineRoundsIIC);
                Internal::S3TCComputer::PackInterpolatedAlpha(options.flags, inputBlocks, 1, pBC + 8, 16, true, options.seedPoints, options.refineRoundsIIC);
                pBC += ParallelMath::ParallelSize * 16;
            }
        }

        void EncodeETC1(uint8_t *pBC, const PixelBlockU8 *pBlocks, const cvtt::Options &options, cvtt::ETC1CompressionData *compressionData)
        {
            assert(pBlocks);
            assert(pBC);

            float channelWeights[4];
            Util::FillWeights(options, channelWeights);

            for (size_t blockBase = 0; blockBase < cvtt::NumParallelBlocks; blockBase += ParallelMath::ParallelSize)
            {
                Internal::ETCComputer::CompressETC1Block(pBC, pBlocks + blockBase, compressionData, options);
                pBC += ParallelMath::ParallelSize * 8;
            }
        }

        void EncodeETC2(uint8_t *pBC, const PixelBlockU8 *pBlocks, const cvtt::Options &options, cvtt::ETC2CompressionData *compressionData)
        {
            assert(pBlocks);
            assert(pBC);

            float channelWeights[4];
            Util::FillWeights(options, channelWeights);

            for (size_t blockBase = 0; blockBase < cvtt::NumParallelBlocks; blockBase += ParallelMath::ParallelSize)
            {
                Internal::ETCComputer::CompressETC2Block(pBC, pBlocks + blockBase, compressionData, options, false);
                pBC += ParallelMath::ParallelSize * 8;
            }
        }

        void EncodeETC2PunchthroughAlpha(uint8_t *pBC, const PixelBlockU8 *pBlocks, const cvtt::Options &options, cvtt::ETC2CompressionData *compressionData)
        {
            assert(pBlocks);
            assert(pBC);

            float channelWeights[4];
            Util::FillWeights(options, channelWeights);

            for (size_t blockBase = 0; blockBase < cvtt::NumParallelBlocks; blockBase += ParallelMath::ParallelSize)
            {
                Internal::ETCComputer::CompressETC2Block(pBC, pBlocks + blockBase, compressionData, options, true);
                pBC += ParallelMath::ParallelSize * 8;
            }
        }

        void EncodeETC2Alpha(uint8_t *pBC, const PixelBlockU8 *pBlocks, const cvtt::Options &options)
        {
            assert(pBlocks);
            assert(pBC);

            for (size_t blockBase = 0; blockBase < cvtt::NumParallelBlocks; blockBase += ParallelMath::ParallelSize)
            {
                Internal::ETCComputer::CompressETC2AlphaBlock(pBC, pBlocks + blockBase, options);
                pBC += ParallelMath::ParallelSize * 8;
            }
        }

        void EncodeETC2Alpha11(uint8_t *pBC, const PixelBlockScalarS16 *pBlocks, bool isSigned, const cvtt::Options &options)
        {
            assert(pBlocks);
            assert(pBC);

            for (size_t blockBase = 0; blockBase < cvtt::NumParallelBlocks; blockBase += ParallelMath::ParallelSize)
            {
                Internal::ETCComputer::CompressEACBlock(pBC, pBlocks + blockBase, isSigned, options);
                pBC += ParallelMath::ParallelSize * 8;
            }
        }

        void EncodeETC2RGBA(uint8_t *pBC, const PixelBlockU8 *pBlocks, const cvtt::Options &options, cvtt::ETC2CompressionData *compressionData)
        {
            uint8_t alphaBlockData[cvtt::NumParallelBlocks * 8];
            uint8_t colorBlockData[cvtt::NumParallelBlocks * 8];

            EncodeETC2(colorBlockData, pBlocks, options, compressionData);
            EncodeETC2Alpha(alphaBlockData, pBlocks, options);

            for (size_t blockBase = 0; blockBase < cvtt::NumParallelBlocks; blockBase++)
            {
                for (size_t blockData = 0; blockData < 8; blockData++)
                    pBC[blockBase * 16 + blockData] = alphaBlockData[blockBase * 8 + blockData];

                for (size_t blockData = 0; blockData < 8; blockData++)
                    pBC[blockBase * 16 + 8 + blockData] = colorBlockData[blockBase * 8 + blockData];
            }
        }

        void DecodeBC7(PixelBlockU8 *pBlocks, const uint8_t *pBC)
        {
            assert(pBlocks);
            assert(pBC);

            for (size_t blockBase = 0; blockBase < cvtt::NumParallelBlocks; blockBase++)
            {
                Internal::BC7Computer::UnpackOne(pBlocks[blockBase], pBC);
                pBC += 16;
            }
        }

        void DecodeBC6HU(PixelBlockF16 *pBlocks, const uint8_t *pBC)
        {
            assert(pBlocks);
            assert(pBC);

            for (size_t blockBase = 0; blockBase < cvtt::NumParallelBlocks; blockBase++)
            {
                Internal::BC6HComputer::UnpackOne(pBlocks[blockBase], pBC, false);
                pBC += 16;
            }
        }

        void DecodeBC6HS(PixelBlockF16 *pBlocks, const uint8_t *pBC)
        {
            assert(pBlocks);
            assert(pBC);

            for (size_t blockBase = 0; blockBase < cvtt::NumParallelBlocks; blockBase++)
            {
                Internal::BC6HComputer::UnpackOne(pBlocks[blockBase], pBC, true);
                pBC += 16;
            }
        }

        ETC1CompressionData *AllocETC1Data(allocFunc_t allocFunc, void *context)
        {
            return cvtt::Internal::ETCComputer::AllocETC1Data(allocFunc, context);
        }

        void ReleaseETC1Data(ETC1CompressionData *compressionData, freeFunc_t freeFunc)
        {
            cvtt::Internal::ETCComputer::ReleaseETC1Data(compressionData, freeFunc);
        }

        ETC2CompressionData *AllocETC2Data(allocFunc_t allocFunc, void *context, const cvtt::Options &options)
        {
            return cvtt::Internal::ETCComputer::AllocETC2Data(allocFunc, context, options);
        }

        void ReleaseETC2Data(ETC2CompressionData *compressionData, freeFunc_t freeFunc)
        {
            cvtt::Internal::ETCComputer::ReleaseETC2Data(compressionData, freeFunc);
        }
    }
}

#endif
