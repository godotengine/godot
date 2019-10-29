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

#include "EtcBlock4x4Encoding_RGB8.h"
#include "EtcErrorMetric.h"
#include "EtcBlock4x4EncodingBits.h"

namespace Etc
{

	// ################################################################################
	// Block4x4Encoding_RGB8A1
	// RGB8A1 if not completely opaque or transparent
	// ################################################################################

	class Block4x4Encoding_RGB8A1 : public Block4x4Encoding_RGB8
	{
	public:

		static const unsigned int TRANSPARENT_SELECTOR = 2;

		Block4x4Encoding_RGB8A1(void);
		virtual ~Block4x4Encoding_RGB8A1(void);

		virtual void InitFromSource(Block4x4 *a_pblockParent,
									ColorFloatRGBA *a_pafrgbaSource,
									unsigned char *a_paucEncodingBits,
									ErrorMetric a_errormetric);

		virtual void InitFromEncodingBits(Block4x4 *a_pblockParent,
											unsigned char *a_paucEncodingBits,
											ColorFloatRGBA *a_pafrgbaSource,
											ErrorMetric a_errormetric);

		virtual void PerformIteration(float a_fEffort);

		virtual void SetEncodingBits(void);

		void InitFromEncodingBits_ETC1(Block4x4 *a_pblockParent,
										unsigned char *a_paucEncodingBits,
										ColorFloatRGBA *a_pafrgbaSource,
										ErrorMetric a_errormetric);

		void InitFromEncodingBits_T(void);
		void InitFromEncodingBits_H(void);

		void PerformFirstIteration(void);

		void Decode_ETC1(void);
		void DecodePixels_T(void);
		void DecodePixels_H(void);
		void SetEncodingBits_ETC1(void);
		void SetEncodingBits_T(void);
		void SetEncodingBits_H(void);

	protected:

		bool m_boolOpaque;				// all source pixels have alpha >= 0.5
		bool m_boolTransparent;			// all source pixels have alpha < 0.5
		bool m_boolPunchThroughPixels;	// some source pixels have alpha < 0.5

		static float s_aafCwOpaqueUnsetTable[CW_RANGES][SELECTORS];

	private:

		void TryDifferential(bool a_boolFlip, unsigned int a_uiRadius,
								int a_iGrayOffset1, int a_iGrayOffset2);
		void TryDifferentialHalf(DifferentialTrys::Half *a_phalf);

		void TryT(unsigned int a_uiRadius);
		void TryT_BestSelectorCombination(void);
		void TryH(unsigned int a_uiRadius);
		void TryH_BestSelectorCombination(void);

		void TryDegenerates1(void);
		void TryDegenerates2(void);
		void TryDegenerates3(void);
		void TryDegenerates4(void);

	};

	// ################################################################################
	// Block4x4Encoding_RGB8A1_Opaque
	// RGB8A1 if all pixels have alpha==1
	// ################################################################################

	class Block4x4Encoding_RGB8A1_Opaque : public Block4x4Encoding_RGB8A1
	{
	public:

		virtual void PerformIteration(float a_fEffort);

		void PerformFirstIteration(void);

	private:

	};

	// ################################################################################
	// Block4x4Encoding_RGB8A1_Transparent
	// RGB8A1 if all pixels have alpha==0
	// ################################################################################

	class Block4x4Encoding_RGB8A1_Transparent : public Block4x4Encoding_RGB8A1
	{
	public:

		virtual void PerformIteration(float a_fEffort);

	private:

	};

} // namespace Etc
