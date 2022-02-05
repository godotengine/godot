#pragma once

#include "ConvectionKernels_ParallelMath.h"


namespace cvtt
{
    namespace Tables
    {
        namespace BC7SC
        {
            struct Table;
        }
    }

    namespace Internal
    {
        namespace BC67
        {
            struct WorkInfo;
        }

        template<int TVectorSize>
        class IndexSelectorHDR;
    }

    struct PixelBlockU8;
}

namespace cvtt
{
    namespace Internal
    {
        class BC7Computer
        {
        public:
            static void Pack(uint32_t flags, const PixelBlockU8* inputs, uint8_t* packedBlocks, const float channelWeights[4], const BC7EncodingPlan &encodingPlan, int numRefineRounds);
            static void UnpackOne(PixelBlockU8 &output, const uint8_t* packedBlock);

        private:
            static const int MaxTweakRounds = 4;

            typedef ParallelMath::SInt16 MSInt16;
            typedef ParallelMath::UInt15 MUInt15;
            typedef ParallelMath::UInt16 MUInt16;
            typedef ParallelMath::SInt32 MSInt32;
            typedef ParallelMath::Float MFloat;

            static void TweakAlpha(const MUInt15 original[2], int tweak, int range, MUInt15 result[2]);
            static void Quantize(MUInt15* color, int bits, int channels);
            static void QuantizeP(MUInt15* color, int bits, uint16_t p, int channels);
            static void Unquantize(MUInt15* color, int bits, int channels);
            static void CompressEndpoints0(MUInt15 ep[2][4], uint16_t p[2]);
            static void CompressEndpoints1(MUInt15 ep[2][4], uint16_t p);
            static void CompressEndpoints2(MUInt15 ep[2][4]);
            static void CompressEndpoints3(MUInt15 ep[2][4], uint16_t p[2]);
            static void CompressEndpoints4(MUInt15 epRGB[2][3], MUInt15 epA[2]);
            static void CompressEndpoints5(MUInt15 epRGB[2][3], MUInt15 epA[2]);
            static void CompressEndpoints6(MUInt15 ep[2][4], uint16_t p[2]);
            static void CompressEndpoints7(MUInt15 ep[2][4], uint16_t p[2]);
            static void TrySingleColorRGBAMultiTable(uint32_t flags, const MUInt15 pixels[16][4], const MFloat average[4], int numRealChannels, const uint8_t *fragmentStart, int shapeLength, const MFloat &staticAlphaError, const ParallelMath::Int16CompFlag punchThroughInvalid[4], MFloat& shapeBestError, MUInt15 shapeBestEP[2][4], MUInt15 *fragmentBestIndexes, const float *channelWeightsSq, const cvtt::Tables::BC7SC::Table*const* tables, int numTables, const ParallelMath::RoundTowardNearestForScope *rtn);
            static void TrySinglePlane(uint32_t flags, const MUInt15 pixels[16][4], const MFloat floatPixels[16][4], const float channelWeights[4], const BC7EncodingPlan &encodingPlan, int numRefineRounds, BC67::WorkInfo& work, const ParallelMath::RoundTowardNearestForScope *rtn);
            static void TryDualPlane(uint32_t flags, const MUInt15 pixels[16][4], const MFloat floatPixels[16][4], const float channelWeights[4], const BC7EncodingPlan &encodingPlan, int numRefineRounds, BC67::WorkInfo& work, const ParallelMath::RoundTowardNearestForScope *rtn);

            template<class T>
            static void Swap(T& a, T& b);
        };


        class BC6HComputer
        {
        public:
            static void Pack(uint32_t flags, const PixelBlockF16* inputs, uint8_t* packedBlocks, const float channelWeights[4], bool isSigned, int numTweakRounds, int numRefineRounds);
            static void UnpackOne(PixelBlockF16 &output, const uint8_t *pBC, bool isSigned);

        private:
            typedef ParallelMath::Float MFloat;
            typedef ParallelMath::SInt16 MSInt16;
            typedef ParallelMath::UInt16 MUInt16;
            typedef ParallelMath::UInt15 MUInt15;
            typedef ParallelMath::AInt16 MAInt16;
            typedef ParallelMath::SInt32 MSInt32;
            typedef ParallelMath::UInt31 MUInt31;

            static const int MaxTweakRounds = 4;
            static const int MaxRefineRounds = 3;

            static MSInt16 QuantizeSingleEndpointElementSigned(const MSInt16 &elem2CL, int precision, const ParallelMath::RoundUpForScope* ru);
            static MUInt15 QuantizeSingleEndpointElementUnsigned(const MUInt15 &elem, int precision, const ParallelMath::RoundUpForScope* ru);
            static void UnquantizeSingleEndpointElementSigned(const MSInt16 &comp, int precision, MSInt16 &outUnquantized, MSInt16 &outUnquantizedFinished2CL);
            static void UnquantizeSingleEndpointElementUnsigned(const MUInt15 &comp, int precision, MUInt16 &outUnquantized, MUInt16 &outUnquantizedFinished);
            static void QuantizeEndpointsSigned(const MSInt16 endPoints[2][3], const MFloat floatPixelsColorSpace[16][3], const MFloat floatPixelsLinearWeighted[16][3], MAInt16 quantizedEndPoints[2][3], MUInt15 indexes[16], IndexSelectorHDR<3> &indexSelector, int fixupIndex, int precision, int indexRange, const float *channelWeights, bool fastIndexing, const ParallelMath::RoundTowardNearestForScope *rtn);
            static void QuantizeEndpointsUnsigned(const MSInt16 endPoints[2][3], const MFloat floatPixelsColorSpace[16][3], const MFloat floatPixelsLinearWeighted[16][3], MAInt16 quantizedEndPoints[2][3], MUInt15 indexes[16], IndexSelectorHDR<3> &indexSelector, int fixupIndex, int precision, int indexRange, const float *channelWeights, bool fastIndexing, const ParallelMath::RoundTowardNearestForScope *rtn);
            static void EvaluatePartitionedLegality(const MAInt16 ep0[2][3], const MAInt16 ep1[2][3], int aPrec, const int bPrec[3], bool isTransformed, MAInt16 outEncodedEPs[2][2][3], ParallelMath::Int16CompFlag& outIsLegal);
            static void EvaluateSingleLegality(const MAInt16 ep[2][3], int aPrec, const int bPrec[3], bool isTransformed, MAInt16 outEncodedEPs[2][3], ParallelMath::Int16CompFlag& outIsLegal);
            static void SignExtendSingle(int &v, int bits);
        };
    }
}
