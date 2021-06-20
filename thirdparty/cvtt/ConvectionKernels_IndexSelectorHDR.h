#pragma once
#ifndef __CVTT_INDEXSELECTORHDR_H__
#define __CVTT_INDEXSELECTORHDR_H__

#include "ConvectionKernels_ParallelMath.h"
#include "ConvectionKernels_IndexSelector.h"

namespace cvtt
{
    namespace Internal
    {
        ParallelMath::SInt16 UnscaleHDRValueSigned(const ParallelMath::SInt16 &v);
        ParallelMath::UInt15 UnscaleHDRValueUnsigned(const ParallelMath::UInt16 &v);

        template<int TVectorSize>
        class IndexSelectorHDR : public IndexSelector<TVectorSize>
        {
        public:
            typedef ParallelMath::UInt15 MUInt15;
            typedef ParallelMath::UInt16 MUInt16;
            typedef ParallelMath::UInt31 MUInt31;
            typedef ParallelMath::SInt16 MSInt16;
            typedef ParallelMath::SInt32 MSInt32;
            typedef ParallelMath::Float MFloat;

        private:

            MUInt15 InvertSingle(const MUInt15& anIndex) const
            {
                MUInt15 inverted = m_maxValueMinusOne - anIndex;
                return ParallelMath::Select(m_isInverted, inverted, anIndex);
            }

            void ReconstructHDRSignedUninverted(const MUInt15 &index, MSInt16* pixel) const
            {
                MUInt15 weight = ParallelMath::LosslessCast<MUInt15>::Cast(ParallelMath::RightShift(ParallelMath::CompactMultiply(g_weightReciprocals[m_range], index) + 256, 9));

                for (int ch = 0; ch < TVectorSize; ch++)
                {
                    MSInt16 ep0 = ParallelMath::LosslessCast<MSInt16>::Cast(this->m_endPoint[0][ch]);
                    MSInt16 ep1 = ParallelMath::LosslessCast<MSInt16>::Cast(this->m_endPoint[1][ch]);

                    MSInt32 pixel32 = ParallelMath::XMultiply((ParallelMath::MakeUInt15(64) - weight), ep0) + ParallelMath::XMultiply(weight, ep1);

                    pixel32 = ParallelMath::RightShift(pixel32 + ParallelMath::MakeSInt32(32), 6);

                    pixel[ch] = UnscaleHDRValueSigned(ParallelMath::ToSInt16(pixel32));
                }
            }

            void ReconstructHDRUnsignedUninverted(const MUInt15 &index, MSInt16* pixel) const
            {
                MUInt15 weight = ParallelMath::LosslessCast<MUInt15>::Cast(ParallelMath::RightShift(ParallelMath::CompactMultiply(g_weightReciprocals[m_range], index) + 256, 9));

                for (int ch = 0; ch < TVectorSize; ch++)
                {
                    MUInt16 ep0 = ParallelMath::LosslessCast<MUInt16>::Cast(this->m_endPoint[0][ch]);
                    MUInt16 ep1 = ParallelMath::LosslessCast<MUInt16>::Cast(this->m_endPoint[1][ch]);

                    MUInt31 pixel31 = ParallelMath::XMultiply((ParallelMath::MakeUInt15(64) - weight), ep0) + ParallelMath::XMultiply(weight, ep1);

                    pixel31 = ParallelMath::RightShift(pixel31 + ParallelMath::MakeUInt31(32), 6);

                    pixel[ch] = ParallelMath::LosslessCast<MSInt16>::Cast(UnscaleHDRValueUnsigned(ParallelMath::ToUInt16(pixel31)));
                }
            }

            MFloat ErrorForInterpolatorComponent(int index, int ch, const MFloat *pixel) const
            {
                MFloat diff = pixel[ch] - m_reconstructedInterpolators[index][ch];
                return diff * diff;
            }

            MFloat ErrorForInterpolator(int index, const MFloat *pixel) const
            {
                MFloat error = ErrorForInterpolatorComponent(index, 0, pixel);
                for (int ch = 1; ch < TVectorSize; ch++)
                    error = error + ErrorForInterpolatorComponent(index, ch, pixel);
                return error;
            }

        public:

            void InitHDR(int range, bool isSigned, bool fastIndexing, const float *channelWeights)
            {
                assert(range <= 16);

                m_range = range;

                m_isInverted = ParallelMath::MakeBoolInt16(false);
                m_maxValueMinusOne = ParallelMath::MakeUInt15(static_cast<uint16_t>(range - 1));

                if (!fastIndexing)
                {
                    for (int i = 0; i < range; i++)
                    {
                        MSInt16 recon2CL[TVectorSize];

                        if (isSigned)
                            ReconstructHDRSignedUninverted(ParallelMath::MakeUInt15(static_cast<uint16_t>(i)), recon2CL);
                        else
                            ReconstructHDRUnsignedUninverted(ParallelMath::MakeUInt15(static_cast<uint16_t>(i)), recon2CL);

                        for (int ch = 0; ch < TVectorSize; ch++)
                            m_reconstructedInterpolators[i][ch] = ParallelMath::TwosCLHalfToFloat(recon2CL[ch]) * channelWeights[ch];
                    }
                }
            }

            void ReconstructHDRSigned(const MUInt15 &index, MSInt16* pixel) const
            {
                ReconstructHDRSignedUninverted(InvertSingle(index), pixel);
            }

            void ReconstructHDRUnsigned(const MUInt15 &index, MSInt16* pixel) const
            {
                ReconstructHDRUnsignedUninverted(InvertSingle(index), pixel);
            }

            void ConditionalInvert(const ParallelMath::Int16CompFlag &invert)
            {
                m_isInverted = invert;
            }

            MUInt15 SelectIndexHDRSlow(const MFloat* pixel, const ParallelMath::RoundTowardNearestForScope*) const
            {
                MUInt15 index = ParallelMath::MakeUInt15(0);

                MFloat bestError = ErrorForInterpolator(0, pixel);
                for (int i = 1; i < m_range; i++)
                {
                    MFloat error = ErrorForInterpolator(i, pixel);
                    ParallelMath::FloatCompFlag errorBetter = ParallelMath::Less(error, bestError);
                    ParallelMath::ConditionalSet(index, ParallelMath::FloatFlagToInt16(errorBetter), ParallelMath::MakeUInt15(static_cast<uint16_t>(i)));
                    bestError = ParallelMath::Min(bestError, error);
                }

                return InvertSingle(index);
            }

            MUInt15 SelectIndexHDRFast(const MFloat* pixel, const ParallelMath::RoundTowardNearestForScope* rtn) const
            {
                return InvertSingle(this->SelectIndexLDR(pixel, rtn));
            }

        private:
            MFloat m_reconstructedInterpolators[16][TVectorSize];
            ParallelMath::Int16CompFlag m_isInverted;
            MUInt15 m_maxValueMinusOne;
            int m_range;
        };
    }
}
#endif

