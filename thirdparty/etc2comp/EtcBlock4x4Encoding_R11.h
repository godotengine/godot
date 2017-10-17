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
	class Block4x4EncodingBits_R11;

	// ################################################################################
	// Block4x4Encoding_R11
	// ################################################################################

	class Block4x4Encoding_R11 : public Block4x4Encoding_RGB8
	{
	public:

		Block4x4Encoding_R11(void);
		virtual ~Block4x4Encoding_R11(void);

		virtual void InitFromSource(Block4x4 *a_pblockParent,
			ColorFloatRGBA *a_pafrgbaSource,
			unsigned char *a_paucEncodingBits, ErrorMetric a_errormetric);

		virtual void InitFromEncodingBits(Block4x4 *a_pblockParent,
			unsigned char *a_paucEncodingBits,
			ColorFloatRGBA *a_pafrgbaSource,
			ErrorMetric a_errormetric);

		virtual void PerformIteration(float a_fEffort);

		virtual void SetEncodingBits(void);

		inline float GetRedBase(void) const
		{
			return m_fRedBase;
		}

		inline float GetRedMultiplier(void) const
		{
			return m_fRedMultiplier;
		}

		inline int GetRedTableIndex(void) const
		{
			return m_uiRedModifierTableIndex;
		}

		inline const unsigned int * GetRedSelectors(void) const
		{
			return m_auiRedSelectors;
		}

	protected:

		static const unsigned int MODIFIER_TABLE_ENTRYS = 16;
		static const unsigned int SELECTOR_BITS = 3;
		static const unsigned int SELECTORS = 1 << SELECTOR_BITS;

		static float s_aafModifierTable[MODIFIER_TABLE_ENTRYS][SELECTORS];

		void CalculateR11(unsigned int a_uiSelectorsUsed, 
							float a_fBaseRadius, float a_fMultiplierRadius);

		

	
		inline float DecodePixelRed(float a_fBase, float a_fMultiplier,
			unsigned int a_uiTableIndex, unsigned int a_uiSelector)
		{
			float fMultiplier = a_fMultiplier;
			if (fMultiplier <= 0.0f)
			{
				fMultiplier = 1.0f / 8.0f;
			}

			float fPixelRed = a_fBase * 8 + 4 +
				8 * fMultiplier*s_aafModifierTable[a_uiTableIndex][a_uiSelector]*255;
			fPixelRed /= 2047.0f;

			if (fPixelRed < 0.0f)
			{
				fPixelRed = 0.0f;
			}
			else if (fPixelRed > 1.0f)
			{
				fPixelRed = 1.0f;
			}

			return fPixelRed;
		}

		Block4x4EncodingBits_R11 *m_pencodingbitsR11;

		float m_fRedBase;
		float m_fRedMultiplier;
		float m_fRedBlockError;
		unsigned int m_uiRedModifierTableIndex;
		unsigned int m_auiRedSelectors[PIXELS];

		
	};

	// ----------------------------------------------------------------------------------------------------
	//

} // namespace Etc
