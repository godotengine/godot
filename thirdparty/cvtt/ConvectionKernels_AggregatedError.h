#pragma once
#ifndef __CVTT_AGGREGATEDERROR_H__
#define __CVTT_AGGREGATEDERROR_H__

#include "ConvectionKernels_ParallelMath.h"

namespace cvtt
{
    namespace Internal
    {
        template<int TVectorSize>
        class AggregatedError
        {
        public:
            typedef ParallelMath::UInt16 MUInt16;
            typedef ParallelMath::UInt31 MUInt31;
            typedef ParallelMath::Float MFloat;

            AggregatedError()
            {
                for (int ch = 0; ch < TVectorSize; ch++)
                    m_errorUnweighted[ch] = ParallelMath::MakeUInt31(0);
            }

            void Add(const MUInt16 &channelErrorUnweighted, int ch)
            {
                m_errorUnweighted[ch] = m_errorUnweighted[ch] + ParallelMath::ToUInt31(channelErrorUnweighted);
            }

            MFloat Finalize(uint32_t flags, const float channelWeightsSq[TVectorSize]) const
            {
                if (flags & cvtt::Flags::Uniform)
                {
                    MUInt31 total = m_errorUnweighted[0];
                    for (int ch = 1; ch < TVectorSize; ch++)
                        total = total + m_errorUnweighted[ch];
                    return ParallelMath::ToFloat(total);
                }
                else
                {
                    MFloat total = ParallelMath::ToFloat(m_errorUnweighted[0]) * channelWeightsSq[0];
                    for (int ch = 1; ch < TVectorSize; ch++)
                        total = total + ParallelMath::ToFloat(m_errorUnweighted[ch]) * channelWeightsSq[ch];
                    return total;
                }
            }

        private:
            MUInt31 m_errorUnweighted[TVectorSize];
        };
    }
}

#endif

