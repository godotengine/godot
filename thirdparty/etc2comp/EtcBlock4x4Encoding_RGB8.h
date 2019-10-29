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

#include "EtcBlock4x4Encoding_ETC1.h"

namespace Etc
{

	class Block4x4Encoding_RGB8 : public Block4x4Encoding_ETC1
	{
	public:

		Block4x4Encoding_RGB8(void);
		virtual ~Block4x4Encoding_RGB8(void);

		virtual void InitFromEncodingBits(Block4x4 *a_pblockParent,
											unsigned char *a_paucEncodingBits,
											ColorFloatRGBA *a_pafrgbaSource,

											ErrorMetric a_errormetric);

		virtual void PerformIteration(float a_fEffort);
		
		virtual void SetEncodingBits(void);

		inline ColorFloatRGBA GetColor3(void) const
		{
			return m_frgbaColor3;
		}

	protected:

		static const unsigned int PLANAR_CORNER_COLORS = 3;
		static const unsigned int MAX_PLANAR_REGRESSION_SIZE = 4;
		static const unsigned int TH_DISTANCES = 8;

		static float s_afTHDistanceTable[TH_DISTANCES];

		void TryPlanar(unsigned int a_uiRadius);
		void TryTAndH(unsigned int a_uiRadius);

		void InitFromEncodingBits_Planar(void);

		ColorFloatRGBA	m_frgbaColor3;		// used for planar

		void SetEncodingBits_T(void);
		void SetEncodingBits_H(void);
		void SetEncodingBits_Planar(void);

		// state shared between iterations
		ColorFloatRGBA	m_frgbaOriginalColor1_TAndH;
		ColorFloatRGBA	m_frgbaOriginalColor2_TAndH;

		void CalculateBaseColorsForTAndH(void);
		void TryT(unsigned int a_uiRadius);
		void TryT_BestSelectorCombination(void);
		void TryH(unsigned int a_uiRadius);
		void TryH_BestSelectorCombination(void);

	private:

		void InitFromEncodingBits_T(void);
		void InitFromEncodingBits_H(void);

		void CalculatePlanarCornerColors(void);

		void ColorRegression(ColorFloatRGBA *a_pafrgbaPixels, unsigned int a_uiPixels,
			ColorFloatRGBA *a_pfrgbaSlope, ColorFloatRGBA *a_pfrgbaOffset);

		bool TwiddlePlanar(void);
		bool TwiddlePlanarR();
		bool TwiddlePlanarG();
		bool TwiddlePlanarB();

		void DecodePixels_T(void);
		void DecodePixels_H(void);
		void DecodePixels_Planar(void);

	};

} // namespace Etc
