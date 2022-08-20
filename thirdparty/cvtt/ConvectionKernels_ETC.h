#pragma once
#ifndef __CVTT_CONVECTIONKERNELS_ETC_H__
#define __CVTT_CONVECTIONKERNELS_ETC_H__

#include "ConvectionKernels.h"
#include "ConvectionKernels_ParallelMath.h"

namespace cvtt
{
    struct Options;

    namespace Internal
    {
        class ETCComputer
        {
        public:
            static void CompressETC1Block(uint8_t *outputBuffer, const PixelBlockU8 *inputBlocks, ETC1CompressionData *compressionData, const Options &options);
            static void CompressETC2Block(uint8_t *outputBuffer, const PixelBlockU8 *inputBlocks, ETC2CompressionData *compressionData, const Options &options, bool punchthroughAlpha);
            static void CompressETC2AlphaBlock(uint8_t *outputBuffer, const PixelBlockU8 *inputBlocks, const Options &options);
            static void CompressEACBlock(uint8_t *outputBuffer, const PixelBlockScalarS16 *inputBlocks, bool isSigned, const Options &options);

            static ETC2CompressionData *AllocETC2Data(cvtt::Kernels::allocFunc_t allocFunc, void *context, const cvtt::Options &options);
            static void ReleaseETC2Data(ETC2CompressionData *compressionData, cvtt::Kernels::freeFunc_t freeFunc);

            static ETC1CompressionData *AllocETC1Data(cvtt::Kernels::allocFunc_t allocFunc, void *context);
            static void ReleaseETC1Data(ETC1CompressionData *compressionData, cvtt::Kernels::freeFunc_t freeFunc);

        private:
            typedef ParallelMath::Float MFloat;
            typedef ParallelMath::SInt16 MSInt16;
            typedef ParallelMath::UInt15 MUInt15;
            typedef ParallelMath::UInt16 MUInt16;
            typedef ParallelMath::SInt32 MSInt32;
            typedef ParallelMath::UInt31 MUInt31;

            struct DifferentialResolveStorage
            {
                static const unsigned int MaxAttemptsPerSector = 57 + 81 + 81 + 81 + 81 + 81 + 81 + 81;

                MUInt15 diffNumAttempts[2];
                MFloat diffErrors[2][MaxAttemptsPerSector];
                MUInt16 diffSelectors[2][MaxAttemptsPerSector];
                MUInt15 diffColors[2][MaxAttemptsPerSector];
                MUInt15 diffTables[2][MaxAttemptsPerSector];

                uint16_t attemptSortIndexes[2][MaxAttemptsPerSector];
            };

            struct HModeEval
            {
                MFloat errors[62][16];
                MUInt16 signBits[62];
                MUInt15 uniqueQuantizedColors[62];
                MUInt15 numUniqueColors[2];
            };

            struct ETC1CompressionDataInternal : public cvtt::ETC1CompressionData
            {
                explicit ETC1CompressionDataInternal(void *context)
                    : m_context(context)
                {
                }

                DifferentialResolveStorage m_drs;
                void *m_context;
            };

            struct ETC2CompressionDataInternal : public cvtt::ETC2CompressionData
            {
                explicit ETC2CompressionDataInternal(void *context, const cvtt::Options &options);

                HModeEval m_h;
                DifferentialResolveStorage m_drs;

                void *m_context;
                float m_chromaSideAxis0[3];
                float m_chromaSideAxis1[3];
            };

            static MFloat ComputeErrorUniform(const MUInt15 pixelA[3], const MUInt15 pixelB[3]);
            static MFloat ComputeErrorWeighted(const MUInt15 reconstructed[3], const MFloat pixelB[3], const Options options);
            static MFloat ComputeErrorFakeBT709(const MUInt15 reconstructed[3], const MFloat pixelB[3]);

            static void TestHalfBlock(MFloat &outError, MUInt16 &outSelectors, MUInt15 quantizedPackedColor, const MUInt15 pixels[8][3], const MFloat preWeightedPixels[8][3], const MSInt16 modifiers[4], bool isDifferential, const Options &options);
            static void TestHalfBlockPunchthrough(MFloat &outError, MUInt16 &outSelectors, MUInt15 quantizedPackedColor, const MUInt15 pixels[8][3], const MFloat preWeightedPixels[8][3], const ParallelMath::Int16CompFlag isTransparent[8], const MUInt15 modifier, const Options &options);
            static void FindBestDifferentialCombination(int flip, int d, const ParallelMath::Int16CompFlag canIgnoreSector[2], ParallelMath::Int16CompFlag& bestIsThisMode, MFloat& bestTotalError, MUInt15& bestFlip, MUInt15& bestD, MUInt15 bestColors[2], MUInt16 bestSelectors[2], MUInt15 bestTables[2], DifferentialResolveStorage &drs);

            static ParallelMath::Int16CompFlag ETCDifferentialIsLegalForChannel(const MUInt15 &a, const MUInt15 &b);
            static ParallelMath::Int16CompFlag ETCDifferentialIsLegal(const MUInt15 &a, const MUInt15 &b);
            static bool ETCDifferentialIsLegalForChannelScalar(const uint16_t &a, const uint16_t &b);
            static bool ETCDifferentialIsLegalScalar(const uint16_t &a, const uint16_t &b);

            static void EncodeTMode(uint8_t *outputBuffer, MFloat &bestError, const ParallelMath::Int16CompFlag isIsolated[16], const MUInt15 pixels[16][3], const MFloat preWeightedPixels[16][3], const Options &options);
            static void EncodeHMode(uint8_t *outputBuffer, MFloat &bestError, const ParallelMath::Int16CompFlag groupings[16], const MUInt15 pixels[16][3], HModeEval &he, const MFloat preWeightedPixels[16][3], const Options &options);

            static void EncodeVirtualTModePunchthrough(uint8_t *outputBuffer, MFloat &bestError, const ParallelMath::Int16CompFlag isIsolated[16], const MUInt15 pixels[16][3], const MFloat preWeightedPixels[16][3], const ParallelMath::Int16CompFlag isTransparent[16], const ParallelMath::Int16CompFlag& anyTransparent, const ParallelMath::Int16CompFlag& allTransparent, const Options &options);

            static MUInt15 DecodePlanarCoeff(const MUInt15 &coeff, int ch);
            static void EncodePlanar(uint8_t *outputBuffer, MFloat &bestError, const MUInt15 pixels[16][3], const MFloat preWeightedPixels[16][3], const Options &options);

            static void CompressETC1BlockInternal(MFloat &bestTotalError, uint8_t *outputBuffer, const MUInt15 pixels[16][3], const MFloat preWeightedPixels[16][3], DifferentialResolveStorage& compressionData, const Options &options, bool punchthrough);
            static void CompressETC1PunchthroughBlockInternal(MFloat &bestTotalError, uint8_t *outputBuffer, const MUInt15 pixels[16][3], const MFloat preWeightedPixels[16][3], const ParallelMath::Int16CompFlag isTransparent[16], DifferentialResolveStorage& compressionData, const Options &options);
            static void CompressETC2AlphaBlockInternal(uint8_t *outputBuffer, const MUInt15 pixels[16], bool is11Bit, bool isSigned, const Options &options);

            static void ExtractBlocks(MUInt15 pixels[16][3], MFloat preWeightedPixels[16][3], const PixelBlockU8 *inputBlocks, const Options &options);

            static void ResolveHalfBlockFakeBT709RoundingAccurate(MUInt15 quantized[3], const MUInt15 sectorCumulative[3], bool isDifferential);
            static void ResolveHalfBlockFakeBT709RoundingFast(MUInt15 quantized[3], const MUInt15 sectorCumulative[3], bool isDifferential);
            static void ResolveTHFakeBT709Rounding(MUInt15 quantized[3], const MUInt15 target[3], const MUInt15 &granularity);
            static void ConvertToFakeBT709(MFloat yuv[3], const MUInt15 color[3]);
            static void ConvertToFakeBT709(MFloat yuv[3], const MFloat color[3]);
            static void ConvertToFakeBT709(MFloat yuv[3], const MFloat &r, const MFloat &g, const MFloat &b);
            static void ConvertFromFakeBT709(MFloat rgb[3], const MFloat yuv[3]);

            static void QuantizeETC2Alpha(int tableIndex, const MUInt15& value, const MUInt15& baseValue, const MUInt15& multiplier, bool is11Bit, bool isSigned, MUInt15& outIndexes, MUInt15& outQuantizedValues);

            static void EmitTModeBlock(uint8_t *outputBuffer, const ParallelMath::ScalarUInt16 lineColor[3], const ParallelMath::ScalarUInt16 isolatedColor[3], int32_t packedSelectors, ParallelMath::ScalarUInt16 table, bool opaque);
            static void EmitHModeBlock(uint8_t *outputBuffer, const ParallelMath::ScalarUInt16 blockColors[2], ParallelMath::ScalarUInt16 sectorBits, ParallelMath::ScalarUInt16 signBits, ParallelMath::ScalarUInt16 table, bool opaque);
            static void EmitETC1Block(uint8_t *outputBuffer, int blockBestFlip, int blockBestD, const int blockBestColors[2][3], const int blockBestTables[2], const ParallelMath::ScalarUInt16 blockBestSelectors[2], bool transparent);

            static const int g_flipTables[2][2][8];
        };
    }
}

#endif
