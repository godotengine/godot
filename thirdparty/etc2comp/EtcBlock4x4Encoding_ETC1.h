/*
 * Copyright 2015 The Etc2Comp Authors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *  http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include "EtcBlock4x4Encoding.h"
#include "EtcBlock4x4EncodingBits.h"
#include "EtcDifferentialTrys.h"
#include "EtcIndividualTrys.h"

namespace Etc
{

	// base class for Block4x4Encoding_RGB8
	class Block4x4Encoding_ETC1 : public Block4x4Encoding
	{
	public:

		Block4x4Encoding_ETC1(void);
		virtual ~Block4x4Encoding_ETC1(void);

		virtual void InitFromSource(Block4x4 *a_pblockParent,
									ColorFloatRGBA *a_pafrgbaSource,

									unsigned char *a_paucEncodingBits,
									ErrorMetric a_errormetric);

		virtual void InitFromEncodingBits(Block4x4 *a_pblockParent,
											unsigned char *a_paucEncodingBits,
											ColorFloatRGBA *a_pafrgbaSource, 

											ErrorMetric a_errormetric);

		virtual void PerformIteration(float a_fEffort);

		inline virtual bool GetFlip(void)
		{
			return m_boolFlip;
		}

		inline virtual bool IsDifferential(void)
		{
			return m_boolDiff;
		}

		virtual void SetEncodingBits(void);

		void Decode(void);

		inline ColorFloatRGBA GetColor1(void) const
		{
			return m_frgbaColor1;
		}

		inline ColorFloatRGBA GetColor2(void) const
		{
			return m_frgbaColor2;
		}

		inline const unsigned int * GetSelectors(void) const
		{
			return m_auiSelectors;
		}

		inline unsigned int GetCW1(void) const
		{
			return m_uiCW1;
		}

		inline unsigned int GetCW2(void) const
		{
			return m_uiCW2;
		}

		inline bool HasSeverelyBentDifferentialColors(void) const
		{
			return m_boolSeverelyBentDifferentialColors;
		}

	protected:

		static const unsigned int s_auiPixelOrderFlip0[PIXELS];
		static const unsigned int s_auiPixelOrderFlip1[PIXELS];
		static const unsigned int s_auiPixelOrderHScan[PIXELS];

		static const unsigned int s_auiLeftPixelMapping[8];
		static const unsigned int s_auiRightPixelMapping[8];
		static const unsigned int s_auiTopPixelMapping[8];
		static const unsigned int s_auiBottomPixelMapping[8];

		static const unsigned int SELECTOR_BITS = 2;
		static const unsigned int SELECTORS = 1 << SELECTOR_BITS;

		static const unsigned int CW_BITS = 3;
		static const unsigned int CW_RANGES = 1 << CW_BITS;

		static float s_aafCwTable[CW_RANGES][SELECTORS];
		static unsigned char s_aucDifferentialCwRange[256];

		static const int MAX_DIFFERENTIAL = 3;
		static const int MIN_DIFFERENTIAL = -4;

		void InitFromEncodingBits_Selectors(void);

		void PerformFirstIteration(void);
		void CalculateMostLikelyFlip(void);

		void TryDifferential(bool a_boolFlip, unsigned int a_uiRadius,
								int a_iGrayOffset1, int a_iGrayOffset2);
		void TryDifferentialHalf(DifferentialTrys::Half *a_phalf);

		void TryIndividual(bool a_boolFlip, unsigned int a_uiRadius);
		void TryIndividualHalf(IndividualTrys::Half *a_phalf);

		void TryDegenerates1(void);
		void TryDegenerates2(void);
		void TryDegenerates3(void);
		void TryDegenerates4(void);

		void CalculateSelectors();
		void CalculateHalfOfTheSelectors(unsigned int a_uiHalf,
											const unsigned int *pauiPixelMapping);

		// calculate the distance2 of r_frgbaPixel from r_frgbaTarget's gray line
		inline float CalcGrayDistance2(ColorFloatRGBA &r_frgbaPixel, 
										ColorFloatRGBA &r_frgbaTarget)
		{
			float fDeltaGray = ((r_frgbaPixel.fR - r_frgbaTarget.fR) +
								(r_frgbaPixel.fG - r_frgbaTarget.fG) +
								(r_frgbaPixel.fB - r_frgbaTarget.fB)) / 3.0f;

			ColorFloatRGBA frgbaPointOnGrayLine = (r_frgbaTarget + fDeltaGray).ClampRGB();

			float fDR = r_frgbaPixel.fR - frgbaPointOnGrayLine.fR;
			float fDG = r_frgbaPixel.fG - frgbaPointOnGrayLine.fG;
			float fDB = r_frgbaPixel.fB - frgbaPointOnGrayLine.fB;

			return (fDR*fDR) + (fDG*fDG) + (fDB*fDB);
		}

		void SetEncodingBits_Selectors(void);

		// intermediate encoding
		bool			m_boolDiff;
		bool			m_boolFlip;
		ColorFloatRGBA	m_frgbaColor1;
		ColorFloatRGBA	m_frgbaColor2;
		unsigned int	m_uiCW1;
		unsigned int	m_uiCW2;
		unsigned int	m_auiSelectors[PIXELS];

		// state shared between iterations
		ColorFloatRGBA	m_frgbaSourceAverageLeft;
		ColorFloatRGBA	m_frgbaSourceAverageRight;
		ColorFloatRGBA	m_frgbaSourceAverageTop;
		ColorFloatRGBA	m_frgbaSourceAverageBottom;
		bool			m_boolMostLikelyFlip;

		// stats
		float			m_fError1;	// error for Etc1 half 1
		float			m_fError2;	// error for Etc1 half 2
		bool			m_boolSeverelyBentDifferentialColors;	// only valid if m_boolDiff;

		// final encoding
		Block4x4EncodingBits_RGB8 *m_pencodingbitsRGB8;		// or RGB8 portion of Block4x4EncodingBits_RGB8A8

		private:

		void CalculateSourceAverages(void);

	};

} // namespace Etc
