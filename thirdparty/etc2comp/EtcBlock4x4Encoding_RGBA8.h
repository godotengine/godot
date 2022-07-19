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

namespace Etc
{
	class Block4x4EncodingBits_A8;

	// ################################################################################
	// Block4x4Encoding_RGBA8
	// RGBA8 if not completely opaque or transparent
	// ################################################################################

	class Block4x4Encoding_RGBA8 : public Block4x4Encoding_RGB8
	{
	public:

		Block4x4Encoding_RGBA8(void);
		virtual ~Block4x4Encoding_RGBA8(void);

		virtual void InitFromSource(Block4x4 *a_pblockParent,
									ColorFloatRGBA *a_pafrgbaSource,
									unsigned char *a_paucEncodingBits, ErrorMetric a_errormetric);

		virtual void InitFromEncodingBits(Block4x4 *a_pblockParent,
											unsigned char *a_paucEncodingBits,
											ColorFloatRGBA *a_pafrgbaSource,
											ErrorMetric a_errormetric);

		virtual void PerformIteration(float a_fEffort);

		virtual void SetEncodingBits(void);

	protected:

		static const unsigned int MODIFIER_TABLE_ENTRYS = 16;
		static const unsigned int ALPHA_SELECTOR_BITS = 3;
		static const unsigned int ALPHA_SELECTORS = 1 << ALPHA_SELECTOR_BITS;

		static float s_aafModifierTable[MODIFIER_TABLE_ENTRYS][ALPHA_SELECTORS];

		void CalculateA8(float a_fRadius);

		Block4x4EncodingBits_A8 *m_pencodingbitsA8;	// A8 portion of Block4x4EncodingBits_RGBA8

		float m_fBase;
		float m_fMultiplier;
		unsigned int m_uiModifierTableIndex;
		unsigned int m_auiAlphaSelectors[PIXELS];

	private:

		inline float DecodePixelAlpha(float a_fBase, float a_fMultiplier,
										unsigned int a_uiTableIndex, unsigned int a_uiSelector)
		{
			float fPixelAlpha = a_fBase + 
								a_fMultiplier*s_aafModifierTable[a_uiTableIndex][a_uiSelector];
			if (fPixelAlpha < 0.0f)
			{
				fPixelAlpha = 0.0f;
			}
			else if (fPixelAlpha > 1.0f)
			{
				fPixelAlpha = 1.0f;
			}

			return fPixelAlpha;
		}

	};

	// ################################################################################
	// Block4x4Encoding_RGBA8_Opaque
	// RGBA8 if all pixels have alpha==1
	// ################################################################################

	class Block4x4Encoding_RGBA8_Opaque : public Block4x4Encoding_RGBA8
	{
	public:

		virtual void PerformIteration(float a_fEffort);

		virtual void SetEncodingBits(void);

	};

	// ################################################################################
	// Block4x4Encoding_RGBA8_Transparent
	// RGBA8 if all pixels have alpha==0
	// ################################################################################

	class Block4x4Encoding_RGBA8_Transparent : public Block4x4Encoding_RGBA8
	{
	public:

		virtual void PerformIteration(float a_fEffort);

		virtual void SetEncodingBits(void);

	};

	// ----------------------------------------------------------------------------------------------------
	//

} // namespace Etc
