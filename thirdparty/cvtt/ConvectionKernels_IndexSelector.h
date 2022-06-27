#pragma once
#ifndef __CVTT_INDEXSELECTOR_H__
#define __CVTT_INDEXSELECTOR_H__

#include "ConvectionKernels_ParallelMath.h"

namespace cvtt
{
    namespace Internal
    {
        extern const ParallelMath::UInt16 g_weightReciprocals[17];

        template<int TVectorSize>
        class IndexSelector
        {
        public:
            typedef ParallelMath::Float MFloat;
            typedef ParallelMath::UInt16 MUInt16;
            typedef ParallelMath::UInt15 MUInt15;
            typedef ParallelMath::SInt16 MSInt16;
            typedef ParallelMath::AInt16 MAInt16;
            typedef ParallelMath::SInt32 MSInt32;
            typedef ParallelMath::UInt31 MUInt31;


            template<class TInterpolationEPType, class TColorEPType>
            void Init(const float *channelWeights, const TInterpolationEPType interpolationEndPoints[2][TVectorSize], const TColorEPType colorSpaceEndpoints[2][TVectorSize], int range)
            {
                // In BC6H, the interpolation endpoints are higher-precision than the endpoints in color space.
                // We need to select indexes using the color-space endpoints.

                m_isUniform = true;
                for (int ch = 1; ch < TVectorSize; ch++)
                {
                    if (channelWeights[ch] != channelWeights[0])
                        m_isUniform = false;
                }

                // To work with channel weights, we need something where:
                // pxDiff = px - ep[0]
                // epDiff = ep[1] - ep[0]
                //
                // weightedEPDiff = epDiff * channelWeights
                // normalizedWeightedAxis = weightedEPDiff / len(weightedEPDiff)
                // normalizedIndex = dot(pxDiff * channelWeights, normalizedWeightedAxis) / len(weightedEPDiff)
                // index = normalizedIndex * maxValue
                //
                // Equivalent to:
                // axis = channelWeights * maxValue * epDiff * channelWeights / lenSquared(epDiff * channelWeights)
                // index = dot(axis, pxDiff)

                for (int ep = 0; ep < 2; ep++)
                    for (int ch = 0; ch < TVectorSize; ch++)
                        m_endPoint[ep][ch] = ParallelMath::LosslessCast<MAInt16>::Cast(interpolationEndPoints[ep][ch]);

                m_range = range;
                m_maxValue = static_cast<float>(range - 1);

                MFloat epDiffWeighted[TVectorSize];
                for (int ch = 0; ch < TVectorSize; ch++)
                {
                    m_origin[ch] = ParallelMath::ToFloat(colorSpaceEndpoints[0][ch]);
                    MFloat opposingOriginCh = ParallelMath::ToFloat(colorSpaceEndpoints[1][ch]);
                    epDiffWeighted[ch] = (opposingOriginCh - m_origin[ch]) * channelWeights[ch];
                }

                MFloat lenSquared = epDiffWeighted[0] * epDiffWeighted[0];
                for (int ch = 1; ch < TVectorSize; ch++)
                    lenSquared = lenSquared + epDiffWeighted[ch] * epDiffWeighted[ch];

                ParallelMath::MakeSafeDenominator(lenSquared);

                MFloat maxValueDividedByLengthSquared = ParallelMath::MakeFloat(m_maxValue) / lenSquared;

                for (int ch = 0; ch < TVectorSize; ch++)
                    m_axis[ch] = epDiffWeighted[ch] * channelWeights[ch] * maxValueDividedByLengthSquared;
            }

            template<bool TSigned>
            void Init(const float channelWeights[TVectorSize], const MUInt15 endPoints[2][TVectorSize], int range)
            {
                MAInt16 converted[2][TVectorSize];
                for (int epi = 0; epi < 2; epi++)
                    for (int ch = 0; ch < TVectorSize; ch++)
                        converted[epi][ch] = ParallelMath::LosslessCast<MAInt16>::Cast(endPoints[epi][ch]);

                Init<MUInt15, MUInt15>(channelWeights, endPoints, endPoints, range);
            }

            void ReconstructLDR_BC7(const MUInt15 &index, MUInt15* pixel, int numRealChannels)
            {
                MUInt15 weight = ParallelMath::LosslessCast<MUInt15>::Cast(ParallelMath::RightShift(ParallelMath::CompactMultiply(g_weightReciprocals[m_range], index) + 256, 9));

                for (int ch = 0; ch < numRealChannels; ch++)
                {
                    MUInt15 ep0f = ParallelMath::LosslessCast<MUInt15>::Cast(ParallelMath::CompactMultiply((ParallelMath::MakeUInt15(64) - weight), ParallelMath::LosslessCast<MUInt15>::Cast(m_endPoint[0][ch])));
                    MUInt15 ep1f = ParallelMath::LosslessCast<MUInt15>::Cast(ParallelMath::CompactMultiply(weight, ParallelMath::LosslessCast<MUInt15>::Cast(m_endPoint[1][ch])));
                    pixel[ch] = ParallelMath::LosslessCast<MUInt15>::Cast(ParallelMath::RightShift(ep0f + ep1f + ParallelMath::MakeUInt15(32), 6));
                }
            }

            void ReconstructLDRPrecise(const MUInt15 &index, MUInt15* pixel, int numRealChannels)
            {
                MUInt15 weight = ParallelMath::LosslessCast<MUInt15>::Cast(ParallelMath::RightShift(ParallelMath::CompactMultiply(g_weightReciprocals[m_range], index) + 64, 7));

                for (int ch = 0; ch < numRealChannels; ch++)
                {
                    MUInt15 ep0f = ParallelMath::LosslessCast<MUInt15>::Cast(ParallelMath::CompactMultiply((ParallelMath::MakeUInt15(256) - weight), ParallelMath::LosslessCast<MUInt15>::Cast(m_endPoint[0][ch])));
                    MUInt15 ep1f = ParallelMath::LosslessCast<MUInt15>::Cast(ParallelMath::CompactMultiply(weight, ParallelMath::LosslessCast<MUInt15>::Cast(m_endPoint[1][ch])));
                    pixel[ch] = ParallelMath::LosslessCast<MUInt15>::Cast(ParallelMath::RightShift(ep0f + ep1f + ParallelMath::MakeUInt15(128), 8));
                }
            }

            void ReconstructLDR_BC7(const MUInt15 &index, MUInt15* pixel)
            {
                ReconstructLDR_BC7(index, pixel, TVectorSize);
            }

            void ReconstructLDRPrecise(const MUInt15 &index, MUInt15* pixel)
            {
                ReconstructLDRPrecise(index, pixel, TVectorSize);
            }

            MUInt15 SelectIndexLDR(const MFloat* pixel, const ParallelMath::RoundTowardNearestForScope* rtn) const
            {
                MFloat dist = (pixel[0] - m_origin[0]) * m_axis[0];
                for (int ch = 1; ch < TVectorSize; ch++)
                    dist = dist + (pixel[ch] - m_origin[ch]) * m_axis[ch];

                return ParallelMath::RoundAndConvertToU15(ParallelMath::Clamp(dist, 0.0f, m_maxValue), rtn);
            }

        protected:
            MAInt16 m_endPoint[2][TVectorSize];

        private:
            MFloat m_origin[TVectorSize];
            MFloat m_axis[TVectorSize];
            int m_range;
            float m_maxValue;
            bool m_isUniform;
        };
    }
}

#endif

