#pragma once
#ifndef __CVTT_BCCOMMON_H__
#define __CVTT_BCCOMMON_H__

#include "ConvectionKernels_AggregatedError.h"
#include "ConvectionKernels_ParallelMath.h"

namespace cvtt
{
    namespace Internal
    {
        class BCCommon
        {
        public:
            typedef ParallelMath::Float MFloat;
            typedef ParallelMath::UInt16 MUInt16;
            typedef ParallelMath::UInt15 MUInt15;
            typedef ParallelMath::AInt16 MAInt16;
            typedef ParallelMath::SInt16 MSInt16;
            typedef ParallelMath::SInt32 MSInt32;

            static int TweakRoundsForRange(int range);

            template<int TVectorSize>
            static void ComputeErrorLDR(uint32_t flags, const MUInt15 reconstructed[TVectorSize], const MUInt15 original[TVectorSize], int numRealChannels, AggregatedError<TVectorSize> &aggError)
            {
                for (int ch = 0; ch < numRealChannels; ch++)
                    aggError.Add(ParallelMath::SqDiffUInt8(reconstructed[ch], original[ch]), ch);
            }

            template<int TVectorSize>
            static void ComputeErrorLDR(uint32_t flags, const MUInt15 reconstructed[TVectorSize], const MUInt15 original[TVectorSize], AggregatedError<TVectorSize> &aggError)
            {
                ComputeErrorLDR<TVectorSize>(flags, reconstructed, original, TVectorSize, aggError);
            }

            template<int TVectorSize>
            static MFloat ComputeErrorLDRSimple(uint32_t flags, const MUInt15 reconstructed[TVectorSize], const MUInt15 original[TVectorSize], int numRealChannels, const float *channelWeightsSq)
            {
                AggregatedError<TVectorSize> aggError;
                ComputeErrorLDR<TVectorSize>(flags, reconstructed, original, numRealChannels, aggError);
                return aggError.Finalize(flags, channelWeightsSq);
            }

            template<int TVectorSize>
            static MFloat ComputeErrorHDRFast(uint32_t flags, const MSInt16 reconstructed[TVectorSize], const MSInt16 original[TVectorSize], const float channelWeightsSq[TVectorSize])
            {
                MFloat error = ParallelMath::MakeFloatZero();
                if (flags & Flags::Uniform)
                {
                    for (int ch = 0; ch < TVectorSize; ch++)
                        error = error + ParallelMath::SqDiffSInt16(reconstructed[ch], original[ch]);
                }
                else
                {
                    for (int ch = 0; ch < TVectorSize; ch++)
                        error = error + ParallelMath::SqDiffSInt16(reconstructed[ch], original[ch]) * ParallelMath::MakeFloat(channelWeightsSq[ch]);
                }

                return error;
            }

            template<int TVectorSize>
            static MFloat ComputeErrorHDRSlow(uint32_t flags, const MSInt16 reconstructed[TVectorSize], const MSInt16 original[TVectorSize], const float channelWeightsSq[TVectorSize])
            {
                MFloat error = ParallelMath::MakeFloatZero();
                if (flags & Flags::Uniform)
                {
                    for (int ch = 0; ch < TVectorSize; ch++)
                        error = error + ParallelMath::SqDiff2CL(reconstructed[ch], original[ch]);
                }
                else
                {
                    for (int ch = 0; ch < TVectorSize; ch++)
                        error = error + ParallelMath::SqDiff2CL(reconstructed[ch], original[ch]) * ParallelMath::MakeFloat(channelWeightsSq[ch]);
                }

                return error;
            }

            template<int TChannelCount>
            static void PreWeightPixelsLDR(MFloat preWeightedPixels[16][TChannelCount], const MUInt15 pixels[16][TChannelCount], const float channelWeights[TChannelCount])
            {
                for (int px = 0; px < 16; px++)
                {
                    for (int ch = 0; ch < TChannelCount; ch++)
                        preWeightedPixels[px][ch] = ParallelMath::ToFloat(pixels[px][ch]) * channelWeights[ch];
                }
            }

            template<int TChannelCount>
            static void PreWeightPixelsHDR(MFloat preWeightedPixels[16][TChannelCount], const MSInt16 pixels[16][TChannelCount], const float channelWeights[TChannelCount])
            {
                for (int px = 0; px < 16; px++)
                {
                    for (int ch = 0; ch < TChannelCount; ch++)
                        preWeightedPixels[px][ch] = ParallelMath::ToFloat(pixels[px][ch]) * channelWeights[ch];
                }
            }
        };
    }
}

#endif
