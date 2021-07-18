#pragma once
#ifndef __CVTT_S3TC_H__
#define __CVTT_S3TC_H__

#include "ConvectionKernels_ParallelMath.h"

namespace cvtt
{
    namespace Internal
    {
        template<int TVectorSize>
        class EndpointRefiner;
    }

    struct PixelBlockU8;
}

namespace cvtt
{
    namespace Internal
    {
        class S3TCComputer
        {
        public:
            typedef ParallelMath::Float MFloat;
            typedef ParallelMath::SInt16 MSInt16;
            typedef ParallelMath::UInt15 MUInt15;
            typedef ParallelMath::UInt16 MUInt16;
            typedef ParallelMath::SInt32 MSInt32;

            static void Init(MFloat& error);
            static void QuantizeTo6Bits(MUInt15& v);
            static void QuantizeTo5Bits(MUInt15& v);
            static void QuantizeTo565(MUInt15 endPoint[3]);
            static MFloat ParanoidFactorForSpan(const MSInt16& span);
            static MFloat ParanoidDiff(const MUInt15& a, const MUInt15& b, const MFloat& d);
            static void TestSingleColor(uint32_t flags, const MUInt15 pixels[16][4], const MFloat floatPixels[16][4], int range, const float* channelWeights,
                MFloat &bestError, MUInt15 bestEndpoints[2][3], MUInt15 bestIndexes[16], MUInt15 &bestRange, const ParallelMath::RoundTowardNearestForScope *rtn);
            static void TestEndpoints(uint32_t flags, const MUInt15 pixels[16][4], const MFloat floatPixels[16][4], const MFloat preWeightedPixels[16][4], const MUInt15 unquantizedEndPoints[2][3], int range, const float* channelWeights,
                MFloat &bestError, MUInt15 bestEndpoints[2][3], MUInt15 bestIndexes[16], MUInt15 &bestRange, EndpointRefiner<3> *refiner, const ParallelMath::RoundTowardNearestForScope *rtn);
            static void TestCounts(uint32_t flags, const int *counts, int nCounts, const MUInt15 &numElements, const MUInt15 pixels[16][4], const MFloat floatPixels[16][4], const MFloat preWeightedPixels[16][4], bool alphaTest,
                const MFloat floatSortedInputs[16][4], const MFloat preWeightedFloatSortedInputs[16][4], const float *channelWeights, MFloat &bestError, MUInt15 bestEndpoints[2][3], MUInt15 bestIndexes[16], MUInt15 &bestRange,
                const ParallelMath::RoundTowardNearestForScope* rtn);
            static void PackExplicitAlpha(uint32_t flags, const PixelBlockU8* inputs, int inputChannel, uint8_t* packedBlocks, size_t packedBlockStride);
            static void PackInterpolatedAlpha(uint32_t flags, const PixelBlockU8* inputs, int inputChannel, uint8_t* packedBlocks, size_t packedBlockStride, bool isSigned, int maxTweakRounds, int numRefineRounds);
            static void PackRGB(uint32_t flags, const PixelBlockU8* inputs, uint8_t* packedBlocks, size_t packedBlockStride, const float channelWeights[4], bool alphaTest, float alphaThreshold, bool exhaustive, int maxTweakRounds, int numRefineRounds);
        };
    }
}

#endif
