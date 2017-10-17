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
#include "EtcBlock4x4Encoding_R11.h"

namespace Etc
{
	class Block4x4EncodingBits_RG11;

	// ################################################################################
	// Block4x4Encoding_RG11
	// ################################################################################

	class Block4x4Encoding_RG11 : public Block4x4Encoding_R11
	{
		float m_fGrnBase;
		float m_fGrnMultiplier;
		float m_fGrnBlockError;
		unsigned int m_auiGrnSelectors[PIXELS];
		unsigned int m_uiGrnModifierTableIndex;
	public:

		Block4x4Encoding_RG11(void);
		virtual ~Block4x4Encoding_RG11(void);

		virtual void InitFromSource(Block4x4 *a_pblockParent,
			ColorFloatRGBA *a_pafrgbaSource,

			unsigned char *a_paucEncodingBits, ErrorMetric a_errormetric);

		virtual void InitFromEncodingBits(Block4x4 *a_pblockParent,
			unsigned char *a_paucEncodingBits,
			ColorFloatRGBA *a_pafrgbaSource,

			ErrorMetric a_errormetric);

		virtual void PerformIteration(float a_fEffort);

		virtual void SetEncodingBits(void);

		Block4x4EncodingBits_RG11 *m_pencodingbitsRG11;

		void CalculateG11(unsigned int a_uiSelectorsUsed, float a_fBaseRadius, float a_fMultiplierRadius);

		inline float GetGrnBase(void) const
		{
			return m_fGrnBase;
		}

		inline float GetGrnMultiplier(void) const
		{
			return m_fGrnMultiplier;
		}

		inline int GetGrnTableIndex(void) const
		{
			return m_uiGrnModifierTableIndex;
		}

		inline const unsigned int * GetGrnSelectors(void) const
		{
			return m_auiGrnSelectors;
		}

	};

	// ----------------------------------------------------------------------------------------------------
	//

} // namespace Etc
