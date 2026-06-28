#pragma once
#ifndef __CVTT_ENDPOINTREFINER_H__
#define __CVTT_ENDPOINTREFINER_H__

#include "ConvectionKernels_ParallelMath.h"

namespace cvtt
{
    namespace Internal
    {
        // Solve for a, b where v = a*t + b
        // This allows endpoints to be mapped to where T=0 and T=1
        // Least squares from totals:
        // a = (tv - t*v/w)/(tt - t*t/w)
        // b = (v - a*t)/w
        template<int TVectorSize>
        class EndpointRefiner
        {
        public:
            typedef ParallelMath::Float MFloat;
            typedef ParallelMath::UInt16 MUInt16;
            typedef ParallelMath::UInt15 MUInt15;
            typedef ParallelMath::AInt16 MAInt16;
            typedef ParallelMath::SInt16 MSInt16;
            typedef ParallelMath::SInt32 MSInt32;

            MFloat m_tv[TVectorSize];
            MFloat m_v[TVectorSize];
            MFloat m_tt;
            MFloat m_t;
            MFloat m_w;
            int m_wu;

            float m_rcpMaxIndex;
            float m_channelWeights[TVectorSize];
            float m_rcpChannelWeights[TVectorSize];

            void Init(int indexRange, const float channelWeights[TVectorSize])
            {
                for (int ch = 0; ch < TVectorSize; ch++)
                {
                    m_tv[ch] = ParallelMath::MakeFloatZero();
                    m_v[ch] = ParallelMath::MakeFloatZero();
                }
                m_tt = ParallelMath::MakeFloatZero();
                m_t = ParallelMath::MakeFloatZero();
                m_w = ParallelMath::MakeFloatZero();

                m_rcpMaxIndex = 1.0f / static_cast<float>(indexRange - 1);

                for (int ch = 0; ch < TVectorSize; ch++)
                {
                    m_channelWeights[ch] = channelWeights[ch];
                    m_rcpChannelWeights[ch] = 1.0f;
                    if (m_channelWeights[ch] != 0.0f)
                        m_rcpChannelWeights[ch] = 1.0f / channelWeights[ch];
                }

                m_wu = 0;
            }

            void ContributePW(const MFloat *pwFloatPixel, const MUInt15 &index, const MFloat &weight)
            {
                MFloat t = ParallelMath::ToFloat(index) * m_rcpMaxIndex;

                for (int ch = 0; ch < TVectorSize; ch++)
                {
                    MFloat v = pwFloatPixel[ch] * weight;

                    m_tv[ch] = m_tv[ch] + t * v;
                    m_v[ch] = m_v[ch] + v;
                }
                m_tt = m_tt + weight * t * t;
                m_t = m_t + weight * t;
                m_w = m_w + weight;
            }

            void ContributeUnweightedPW(const MFloat *pwFloatPixel, const MUInt15 &index, int numRealChannels)
            {
                MFloat t = ParallelMath::ToFloat(index) * m_rcpMaxIndex;

                for (int ch = 0; ch < numRealChannels; ch++)
                {
                    MFloat v = pwFloatPixel[ch];

                    m_tv[ch] = m_tv[ch] + t * v;
                    m_v[ch] = m_v[ch] + v;
                }
                m_tt = m_tt + t * t;
                m_t = m_t + t;
                m_wu++;
            }

            void ContributeUnweightedPW(const MFloat *floatPixel, const MUInt15 &index)
            {
                ContributeUnweightedPW(floatPixel, index, TVectorSize);
            }

            void GetRefinedEndpoints(MFloat endPoint[2][TVectorSize])
            {
                // a = (tv - t*v/w)/(tt - t*t/w)
                // b = (v - a*t)/w
                MFloat w = m_w + ParallelMath::MakeFloat(static_cast<float>(m_wu));

                ParallelMath::MakeSafeDenominator(w);
                MFloat wRcp = ParallelMath::Reciprocal(w);

                MFloat adenom = (m_tt * w - m_t * m_t) * wRcp;

                ParallelMath::FloatCompFlag adenomZero = ParallelMath::Equal(adenom, ParallelMath::MakeFloatZero());
                ParallelMath::ConditionalSet(adenom, adenomZero, ParallelMath::MakeFloat(1.0f));

                for (int ch = 0; ch < TVectorSize; ch++)
                {
                    /*
                    if (adenom == 0.0)
                    p1 = p2 = er.v / er.w;
                    else
                    {
                    float4 a = (er.tv - er.t*er.v / er.w) / adenom;
                    float4 b = (er.v - a * er.t) / er.w;
                    p1 = b;
                    p2 = a + b;
                    }
                    */

                    MFloat a = (m_tv[ch] - m_t * m_v[ch] * wRcp) / adenom;
                    MFloat b = (m_v[ch] - a * m_t) * wRcp;

                    MFloat p1 = b;
                    MFloat p2 = a + b;

                    ParallelMath::ConditionalSet(p1, adenomZero, (m_v[ch] * wRcp));
                    ParallelMath::ConditionalSet(p2, adenomZero, p1);

                    // Unweight
                    float inverseWeight = m_rcpChannelWeights[ch];

                    endPoint[0][ch] = p1 * inverseWeight;
                    endPoint[1][ch] = p2 * inverseWeight;
                }
            }

            void GetRefinedEndpointsLDR(MUInt15 endPoint[2][TVectorSize], int numRealChannels, const ParallelMath::RoundTowardNearestForScope *roundingMode)
            {
                MFloat floatEndPoint[2][TVectorSize];
                GetRefinedEndpoints(floatEndPoint);

                for (int epi = 0; epi < 2; epi++)
                    for (int ch = 0; ch < TVectorSize; ch++)
                        endPoint[epi][ch] = ParallelMath::RoundAndConvertToU15(ParallelMath::Clamp(floatEndPoint[epi][ch], 0.0f, 255.0f), roundingMode);
            }

            void GetRefinedEndpointsLDR(MUInt15 endPoint[2][TVectorSize], const ParallelMath::RoundTowardNearestForScope *roundingMode)
            {
                GetRefinedEndpointsLDR(endPoint, TVectorSize, roundingMode);
            }

            void GetRefinedEndpointsHDR(MSInt16 endPoint[2][TVectorSize], bool isSigned, const ParallelMath::RoundTowardNearestForScope *roundingMode)
            {
                MFloat floatEndPoint[2][TVectorSize];
                GetRefinedEndpoints(floatEndPoint);

                for (int epi = 0; epi < 2; epi++)
                {
                    for (int ch = 0; ch < TVectorSize; ch++)
                    {
                        MFloat f = floatEndPoint[epi][ch];
                        if (isSigned)
                            endPoint[epi][ch] = ParallelMath::LosslessCast<MSInt16>::Cast(ParallelMath::RoundAndConvertToS16(ParallelMath::Clamp(f, -31743.0f, 31743.0f), roundingMode));
                        else
                            endPoint[epi][ch] = ParallelMath::LosslessCast<MSInt16>::Cast(ParallelMath::RoundAndConvertToU15(ParallelMath::Clamp(f, 0.0f, 31743.0f), roundingMode));
                    }
                }
            }
        };
    }
}

#endif

