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

/*
EtcBlock4x4Encoding_RG11.cpp

Block4x4Encoding_RG11 is the encoder to use when targetting file format RG11 and SRG11 (signed RG11).

*/

#include "EtcConfig.h"
#include "EtcBlock4x4Encoding_RG11.h"

#include "EtcBlock4x4EncodingBits.h"
#include "EtcBlock4x4.h"

#include <stdio.h>
#include <string.h>
#include <assert.h>
#include <float.h>
#include <limits>

namespace Etc
{
	// ----------------------------------------------------------------------------------------------------
	//
	Block4x4Encoding_RG11::Block4x4Encoding_RG11(void)
	{
		m_pencodingbitsRG11 = nullptr;
	}

	Block4x4Encoding_RG11::~Block4x4Encoding_RG11(void) {}
	// ----------------------------------------------------------------------------------------------------
	// initialization prior to encoding
	// a_pblockParent points to the block associated with this encoding
	// a_errormetric is used to choose the best encoding
	// a_pafrgbaSource points to a 4x4 block subset of the source image
	// a_paucEncodingBits points to the final encoding bits
	//
	void Block4x4Encoding_RG11::InitFromSource(Block4x4 *a_pblockParent,
		ColorFloatRGBA *a_pafrgbaSource,
		unsigned char *a_paucEncodingBits, ErrorMetric a_errormetric)
	{
		Block4x4Encoding::Init(a_pblockParent, a_pafrgbaSource,a_errormetric);

		m_pencodingbitsRG11 = (Block4x4EncodingBits_RG11 *)a_paucEncodingBits;
	}

	// ----------------------------------------------------------------------------------------------------
	// initialization from the encoding bits of a previous encoding
	// a_pblockParent points to the block associated with this encoding
	// a_errormetric is used to choose the best encoding
	// a_pafrgbaSource points to a 4x4 block subset of the source image
	// a_paucEncodingBits points to the final encoding bits of a previous encoding
	//
	void Block4x4Encoding_RG11::InitFromEncodingBits(Block4x4 *a_pblockParent,
		unsigned char *a_paucEncodingBits,
		ColorFloatRGBA *a_pafrgbaSource,
		ErrorMetric a_errormetric)
	{

		m_pencodingbitsRG11 = (Block4x4EncodingBits_RG11 *)a_paucEncodingBits;

		// init RGB portion
		Block4x4Encoding_RGB8::InitFromEncodingBits(a_pblockParent,
			(unsigned char *)m_pencodingbitsRG11,
			a_pafrgbaSource,
			a_errormetric);
		m_fError = 0.0f;

		{
			m_mode = MODE_RG11;
			if (a_pblockParent->GetImageSource()->GetFormat() == Image::Format::SIGNED_RG11)
			{
				m_fRedBase = (float)(signed char)m_pencodingbitsRG11->data.baseR;
				m_fGrnBase = (float)(signed char)m_pencodingbitsRG11->data.baseG;
			}
			else
			{
				m_fRedBase = (float)(unsigned char)m_pencodingbitsRG11->data.baseR;
				m_fGrnBase = (float)(unsigned char)m_pencodingbitsRG11->data.baseG;
			}
			m_fRedMultiplier = (float)m_pencodingbitsRG11->data.multiplierR;
			m_fGrnMultiplier = (float)m_pencodingbitsRG11->data.multiplierG;
			m_uiRedModifierTableIndex = m_pencodingbitsRG11->data.tableIndexR;
			m_uiGrnModifierTableIndex = m_pencodingbitsRG11->data.tableIndexG;

			unsigned long long int ulliSelectorBitsR = 0;
			ulliSelectorBitsR |= (unsigned long long int)m_pencodingbitsRG11->data.selectorsR0 << 40;
			ulliSelectorBitsR |= (unsigned long long int)m_pencodingbitsRG11->data.selectorsR1 << 32;
			ulliSelectorBitsR |= (unsigned long long int)m_pencodingbitsRG11->data.selectorsR2 << 24;
			ulliSelectorBitsR |= (unsigned long long int)m_pencodingbitsRG11->data.selectorsR3 << 16;
			ulliSelectorBitsR |= (unsigned long long int)m_pencodingbitsRG11->data.selectorsR4 << 8;
			ulliSelectorBitsR |= (unsigned long long int)m_pencodingbitsRG11->data.selectorsR5;

			unsigned long long int ulliSelectorBitsG = 0;
			ulliSelectorBitsG |= (unsigned long long int)m_pencodingbitsRG11->data.selectorsG0 << 40;
			ulliSelectorBitsG |= (unsigned long long int)m_pencodingbitsRG11->data.selectorsG1 << 32;
			ulliSelectorBitsG |= (unsigned long long int)m_pencodingbitsRG11->data.selectorsG2 << 24;
			ulliSelectorBitsG |= (unsigned long long int)m_pencodingbitsRG11->data.selectorsG3 << 16;
			ulliSelectorBitsG |= (unsigned long long int)m_pencodingbitsRG11->data.selectorsG4 << 8;
			ulliSelectorBitsG |= (unsigned long long int)m_pencodingbitsRG11->data.selectorsG5;

			
			for (unsigned int uiPixel = 0; uiPixel < PIXELS; uiPixel++)
			{
				unsigned int uiShift = 45 - (3 * uiPixel);
				m_auiRedSelectors[uiPixel] = (ulliSelectorBitsR >> uiShift) & (SELECTORS - 1);
				m_auiGrnSelectors[uiPixel] = (ulliSelectorBitsG >> uiShift) & (SELECTORS - 1);
			}

			
			for (unsigned int uiPixel = 0; uiPixel < PIXELS; uiPixel++)
			{
				float fRedDecodedData = 0.0f;
				float fGrnDecodedData = 0.0f;
				if (a_pblockParent->GetImageSource()->GetFormat() == Image::Format::RG11)
				{
					fRedDecodedData = DecodePixelRed(m_fRedBase, m_fRedMultiplier, m_uiRedModifierTableIndex, m_auiRedSelectors[uiPixel]);
					fGrnDecodedData = DecodePixelRed(m_fGrnBase, m_fGrnMultiplier, m_uiGrnModifierTableIndex, m_auiGrnSelectors[uiPixel]);
				}
				else if (a_pblockParent->GetImageSource()->GetFormat() == Image::Format::SIGNED_RG11)
				{
					fRedDecodedData = DecodePixelRed(m_fRedBase + 128, m_fRedMultiplier, m_uiRedModifierTableIndex, m_auiRedSelectors[uiPixel]);
					fGrnDecodedData = DecodePixelRed(m_fGrnBase + 128, m_fGrnMultiplier, m_uiGrnModifierTableIndex, m_auiGrnSelectors[uiPixel]);
				}
				else
				{
					assert(0);
				}
				m_afrgbaDecodedColors[uiPixel] = ColorFloatRGBA(fRedDecodedData, fGrnDecodedData, 0.0f, 1.0f);
			}

		}

		CalcBlockError();
 	}

	// ----------------------------------------------------------------------------------------------------
	// perform a single encoding iteration
	// replace the encoding if a better encoding was found
	// subsequent iterations generally take longer for each iteration
	// set m_boolDone if encoding is perfect or encoding is finished based on a_fEffort
	//
	void Block4x4Encoding_RG11::PerformIteration(float a_fEffort)
	{
		assert(!m_boolDone);

		switch (m_uiEncodingIterations)
		{
		case 0:
			m_fError = FLT_MAX;
			m_fGrnBlockError = FLT_MAX;		// artificially high value
			m_fRedBlockError = FLT_MAX;
			CalculateR11(8, 0.0f, 0.0f);
			CalculateG11(8, 0.0f, 0.0f);
			m_fError = (m_fGrnBlockError + m_fRedBlockError);
			break;

		case 1:
			CalculateR11(8, 2.0f, 1.0f);
			CalculateG11(8, 2.0f, 1.0f);
			m_fError = (m_fGrnBlockError + m_fRedBlockError);
			if (a_fEffort <= 24.5f)
			{
				m_boolDone = true;
			}
			break;

		case 2:
			CalculateR11(8, 12.0f, 1.0f);
			CalculateG11(8, 12.0f, 1.0f);
			m_fError = (m_fGrnBlockError + m_fRedBlockError);
			if (a_fEffort <= 49.5f)
			{
				m_boolDone = true;
			}
			break;

		case 3:
			CalculateR11(7, 6.0f, 1.0f);
			CalculateG11(7, 6.0f, 1.0f);
			m_fError = (m_fGrnBlockError + m_fRedBlockError);
			break;

		case 4:
			CalculateR11(6, 3.0f, 1.0f);
			CalculateG11(6, 3.0f, 1.0f);
			m_fError = (m_fGrnBlockError + m_fRedBlockError);
			break;

		case 5:
			CalculateR11(5, 1.0f, 0.0f);
			CalculateG11(5, 1.0f, 0.0f);
			m_fError = (m_fGrnBlockError + m_fRedBlockError);
			m_boolDone = true;
			break;

		default:
			assert(0);
			break;
		}

		m_uiEncodingIterations++;
		SetDoneIfPerfect();
	}

	// ----------------------------------------------------------------------------------------------------
	// find the best combination of base color, multiplier and selectors
	//
	// a_uiSelectorsUsed limits the number of selector combinations to try
	// a_fBaseRadius limits the range of base colors to try
	// a_fMultiplierRadius limits the range of multipliers to try
	//
	void Block4x4Encoding_RG11::CalculateG11(unsigned int a_uiSelectorsUsed,
		float a_fBaseRadius, float a_fMultiplierRadius)
	{
		// maps from virtual (monotonic) selector to etc selector
		static const unsigned int auiVirtualSelectorMap[8] = { 3, 2, 1, 0, 4, 5, 6, 7 };

		// find min/max Grn
		float fMinGrn = 1.0f;
		float fMaxGrn = 0.0f;
		for (unsigned int uiPixel = 0; uiPixel < PIXELS; uiPixel++)
		{
			// ignore border pixels
			float fAlpha = m_pafrgbaSource[uiPixel].fA;
			if (isnan(fAlpha))
			{
				continue;
			}

			float fGrn = m_pafrgbaSource[uiPixel].fG;

			if (fGrn < fMinGrn)
			{
				fMinGrn = fGrn;
			}
			if (fGrn > fMaxGrn)
			{
				fMaxGrn = fGrn;
			}
		}
		assert(fMinGrn <= fMaxGrn);

		float fGrnRange = (fMaxGrn - fMinGrn);

		// try each modifier table entry							  
		for (unsigned int uiTableEntry = 0; uiTableEntry < MODIFIER_TABLE_ENTRYS; uiTableEntry++)
		{
			for (unsigned int uiMinVirtualSelector = 0;
			uiMinVirtualSelector <= (8 - a_uiSelectorsUsed);
				uiMinVirtualSelector++)
			{
				unsigned int uiMaxVirtualSelector = uiMinVirtualSelector + a_uiSelectorsUsed - 1;

				unsigned int uiMinSelector = auiVirtualSelectorMap[uiMinVirtualSelector];
				unsigned int uiMaxSelector = auiVirtualSelectorMap[uiMaxVirtualSelector];

				float fTableEntryCenter = -s_aafModifierTable[uiTableEntry][uiMinSelector];

				float fTableEntryRange = s_aafModifierTable[uiTableEntry][uiMaxSelector] -
					s_aafModifierTable[uiTableEntry][uiMinSelector];

				float fCenterRatio = fTableEntryCenter / fTableEntryRange;

				float fCenter = fMinGrn + fCenterRatio*fGrnRange;
				fCenter = roundf(255.0f * fCenter) / 255.0f;

				float fMinBase = fCenter - (a_fBaseRadius / 255.0f);
				if (fMinBase < 0.0f)
				{
					fMinBase = 0.0f;
				}

				float fMaxBase = fCenter + (a_fBaseRadius / 255.0f);
				if (fMaxBase > 1.0f)
				{
					fMaxBase = 1.0f;
				}

				for (float fBase = fMinBase; fBase <= fMaxBase; fBase += (0.999999f / 255.0f))
				{
					float fRangeMultiplier = roundf(fGrnRange / fTableEntryRange);

					float fMinMultiplier = fRangeMultiplier - a_fMultiplierRadius;
					if (fMinMultiplier < 1.0f)
					{
						fMinMultiplier = 0.0f;
					}
					else if (fMinMultiplier > 15.0f)
					{
						fMinMultiplier = 15.0f;
					}

					float fMaxMultiplier = fRangeMultiplier + a_fMultiplierRadius;
					if (fMaxMultiplier < 1.0f)
					{
						fMaxMultiplier = 1.0f;
					}
					else if (fMaxMultiplier > 15.0f)
					{
						fMaxMultiplier = 15.0f;
					}

					for (float fMultiplier = fMinMultiplier; fMultiplier <= fMaxMultiplier; fMultiplier += 1.0f)
					{
						// find best selector for each pixel
						unsigned int auiBestSelectors[PIXELS];
						float afBestGrnError[PIXELS];
						float afBestPixelGrn[PIXELS];

						for (unsigned int uiPixel = 0; uiPixel < PIXELS; uiPixel++)
						{
							float fBestPixelGrnError = FLT_MAX;

							for (unsigned int uiSelector = 0; uiSelector < SELECTORS; uiSelector++)
							{
								//DecodePixelRed is not red channel specific
								float fPixelGrn = DecodePixelRed(fBase * 255.0f, fMultiplier, uiTableEntry, uiSelector);
								
								ColorFloatRGBA frgba(m_pafrgbaSource[uiPixel].fR, fPixelGrn, 0.0f, 1.0f);
									
								float fPixelGrnError = CalcPixelError(frgba, 1.0f, m_pafrgbaSource[uiPixel]);

								if (fPixelGrnError < fBestPixelGrnError)
								{
									fBestPixelGrnError = fPixelGrnError;
									auiBestSelectors[uiPixel] = uiSelector;
									afBestGrnError[uiPixel] = fBestPixelGrnError;
									afBestPixelGrn[uiPixel] = fPixelGrn;
								}
							}
						}
						float fBlockError = 0.0f;
						for (unsigned int uiPixel = 0; uiPixel < PIXELS; uiPixel++)
						{
							fBlockError += afBestGrnError[uiPixel];
						}

						if (fBlockError < m_fGrnBlockError)
						{
							m_fGrnBlockError = fBlockError;

							if (m_pblockParent->GetImageSource()->GetFormat() == Image::Format::RG11)
							{
								m_fGrnBase = 255.0f * fBase;
							}
							else if (m_pblockParent->GetImageSource()->GetFormat() == Image::Format::SIGNED_RG11)
							{
								m_fGrnBase = (fBase * 255) - 128;
							}
							else
							{
								assert(0);
							}
							m_fGrnMultiplier = fMultiplier;
							m_uiGrnModifierTableIndex = uiTableEntry;
							for (unsigned int uiPixel = 0; uiPixel < PIXELS; uiPixel++)
							{
								m_auiGrnSelectors[uiPixel] = auiBestSelectors[uiPixel];
								m_afrgbaDecodedColors[uiPixel].fG = afBestPixelGrn[uiPixel];
								m_afDecodedAlphas[uiPixel] = 1.0f;
							}
						}
					}
				}

			}
		}
	}
	
	// ----------------------------------------------------------------------------------------------------
	// set the encoding bits based on encoding state
	//
	void Block4x4Encoding_RG11::SetEncodingBits(void)
	{
		unsigned long long int ulliSelectorBitsR = 0;
		unsigned long long int ulliSelectorBitsG = 0;
		for (unsigned int uiPixel = 0; uiPixel < PIXELS; uiPixel++)
		{
			unsigned int uiShift = 45 - (3 * uiPixel);
			ulliSelectorBitsR |= ((unsigned long long int)m_auiRedSelectors[uiPixel]) << uiShift;
			ulliSelectorBitsG |= ((unsigned long long int)m_auiGrnSelectors[uiPixel]) << uiShift;
		}
		if (m_pblockParent->GetImageSource()->GetFormat() == Image::Format::RG11)
		{
			m_pencodingbitsRG11->data.baseR = (unsigned char)roundf(m_fRedBase);
		}
		else if (m_pblockParent->GetImageSource()->GetFormat() == Image::Format::SIGNED_RG11)
		{
			m_pencodingbitsRG11->data.baseR = (signed char)roundf(m_fRedBase);
		}
		else
		{
			assert(0);
		}
		m_pencodingbitsRG11->data.tableIndexR = m_uiRedModifierTableIndex;
		m_pencodingbitsRG11->data.multiplierR = (unsigned char)roundf(m_fRedMultiplier);

		m_pencodingbitsRG11->data.selectorsR0 = ulliSelectorBitsR >> 40;
		m_pencodingbitsRG11->data.selectorsR1 = ulliSelectorBitsR >> 32;
		m_pencodingbitsRG11->data.selectorsR2 = ulliSelectorBitsR >> 24;
		m_pencodingbitsRG11->data.selectorsR3 = ulliSelectorBitsR >> 16;
		m_pencodingbitsRG11->data.selectorsR4 = ulliSelectorBitsR >> 8;
		m_pencodingbitsRG11->data.selectorsR5 = ulliSelectorBitsR;

		if (m_pblockParent->GetImageSource()->GetFormat() == Image::Format::RG11)
		{
			m_pencodingbitsRG11->data.baseG = (unsigned char)roundf(m_fGrnBase);
		}
		else if (m_pblockParent->GetImageSource()->GetFormat() == Image::Format::SIGNED_RG11)
		{
			m_pencodingbitsRG11->data.baseG = (signed char)roundf(m_fGrnBase);
		}
		else
		{
			assert(0);
		}
		m_pencodingbitsRG11->data.tableIndexG = m_uiGrnModifierTableIndex;
		m_pencodingbitsRG11->data.multiplierG = (unsigned char)roundf(m_fGrnMultiplier);

		m_pencodingbitsRG11->data.selectorsG0 = ulliSelectorBitsG >> 40;
		m_pencodingbitsRG11->data.selectorsG1 = ulliSelectorBitsG >> 32;
		m_pencodingbitsRG11->data.selectorsG2 = ulliSelectorBitsG >> 24;
		m_pencodingbitsRG11->data.selectorsG3 = ulliSelectorBitsG >> 16;
		m_pencodingbitsRG11->data.selectorsG4 = ulliSelectorBitsG >> 8;
		m_pencodingbitsRG11->data.selectorsG5 = ulliSelectorBitsG;

	}

	// ----------------------------------------------------------------------------------------------------
	//
}
