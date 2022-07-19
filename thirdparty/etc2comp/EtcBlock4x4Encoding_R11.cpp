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
EtcBlock4x4Encoding_R11.cpp

Block4x4Encoding_R11 is the encoder to use when targetting file format R11 and SR11 (signed R11).  

*/

#include "EtcConfig.h"
#include "EtcBlock4x4Encoding_R11.h"

#include "EtcBlock4x4EncodingBits.h"
#include "EtcBlock4x4.h"

#include <stdio.h>
#include <string.h>
#include <assert.h>
#include <float.h>
#include <limits>

namespace Etc
{

	// modifier values to use for R11, SR11, RG11 and SRG11
	float Block4x4Encoding_R11::s_aafModifierTable[MODIFIER_TABLE_ENTRYS][SELECTORS]
	{
		{ -3.0f / 255.0f, -6.0f / 255.0f,  -9.0f / 255.0f, -15.0f / 255.0f, 2.0f / 255.0f, 5.0f / 255.0f, 8.0f / 255.0f, 14.0f / 255.0f },
		{ -3.0f / 255.0f, -7.0f / 255.0f, -10.0f / 255.0f, -13.0f / 255.0f, 2.0f / 255.0f, 6.0f / 255.0f, 9.0f / 255.0f, 12.0f / 255.0f },
		{ -2.0f / 255.0f, -5.0f / 255.0f,  -8.0f / 255.0f, -13.0f / 255.0f, 1.0f / 255.0f, 4.0f / 255.0f, 7.0f / 255.0f, 12.0f / 255.0f },
		{ -2.0f / 255.0f, -4.0f / 255.0f,  -6.0f / 255.0f, -13.0f / 255.0f, 1.0f / 255.0f, 3.0f / 255.0f, 5.0f / 255.0f, 12.0f / 255.0f },

		{ -3.0f / 255.0f, -6.0f / 255.0f,  -8.0f / 255.0f, -12.0f / 255.0f, 2.0f / 255.0f, 5.0f / 255.0f, 7.0f / 255.0f, 11.0f / 255.0f },
		{ -3.0f / 255.0f, -7.0f / 255.0f,  -9.0f / 255.0f, -11.0f / 255.0f, 2.0f / 255.0f, 6.0f / 255.0f, 8.0f / 255.0f, 10.0f / 255.0f },
		{ -4.0f / 255.0f, -7.0f / 255.0f,  -8.0f / 255.0f, -11.0f / 255.0f, 3.0f / 255.0f, 6.0f / 255.0f, 7.0f / 255.0f, 10.0f / 255.0f },
		{ -3.0f / 255.0f, -5.0f / 255.0f,  -8.0f / 255.0f, -11.0f / 255.0f, 2.0f / 255.0f, 4.0f / 255.0f, 7.0f / 255.0f, 10.0f / 255.0f },

		{ -2.0f / 255.0f, -6.0f / 255.0f,  -8.0f / 255.0f, -10.0f / 255.0f, 1.0f / 255.0f, 5.0f / 255.0f, 7.0f / 255.0f,  9.0f / 255.0f },
		{ -2.0f / 255.0f, -5.0f / 255.0f,  -8.0f / 255.0f, -10.0f / 255.0f, 1.0f / 255.0f, 4.0f / 255.0f, 7.0f / 255.0f,  9.0f / 255.0f },
		{ -2.0f / 255.0f, -4.0f / 255.0f,  -8.0f / 255.0f, -10.0f / 255.0f, 1.0f / 255.0f, 3.0f / 255.0f, 7.0f / 255.0f,  9.0f / 255.0f },
		{ -2.0f / 255.0f, -5.0f / 255.0f,  -7.0f / 255.0f, -10.0f / 255.0f, 1.0f / 255.0f, 4.0f / 255.0f, 6.0f / 255.0f,  9.0f / 255.0f },

		{ -3.0f / 255.0f, -4.0f / 255.0f,  -7.0f / 255.0f, -10.0f / 255.0f, 2.0f / 255.0f, 3.0f / 255.0f, 6.0f / 255.0f,  9.0f / 255.0f },
		{ -1.0f / 255.0f, -2.0f / 255.0f,  -3.0f / 255.0f, -10.0f / 255.0f, 0.0f / 255.0f, 1.0f / 255.0f, 2.0f / 255.0f,  9.0f / 255.0f },
		{ -4.0f / 255.0f, -6.0f / 255.0f,  -8.0f / 255.0f,  -9.0f / 255.0f, 3.0f / 255.0f, 5.0f / 255.0f, 7.0f / 255.0f,  8.0f / 255.0f },
		{ -3.0f / 255.0f, -5.0f / 255.0f,  -7.0f / 255.0f,  -9.0f / 255.0f, 2.0f / 255.0f, 4.0f / 255.0f, 6.0f / 255.0f,  8.0f / 255.0f }
	};

	// ----------------------------------------------------------------------------------------------------
	//
	Block4x4Encoding_R11::Block4x4Encoding_R11(void)
	{

		m_pencodingbitsR11 = nullptr;

	}

	Block4x4Encoding_R11::~Block4x4Encoding_R11(void) {}
	// ----------------------------------------------------------------------------------------------------
	// initialization prior to encoding
	// a_pblockParent points to the block associated with this encoding
	// a_errormetric is used to choose the best encoding
	// a_pafrgbaSource points to a 4x4 block subset of the source image
	// a_paucEncodingBits points to the final encoding bits
	//
	void Block4x4Encoding_R11::InitFromSource(Block4x4 *a_pblockParent,
		ColorFloatRGBA *a_pafrgbaSource,
		unsigned char *a_paucEncodingBits, ErrorMetric a_errormetric)
	{
		Block4x4Encoding::Init(a_pblockParent, a_pafrgbaSource,a_errormetric);

		m_pencodingbitsR11 = (Block4x4EncodingBits_R11 *)a_paucEncodingBits;
	}

	// ----------------------------------------------------------------------------------------------------
	// initialization from the encoding bits of a previous encoding
	// a_pblockParent points to the block associated with this encoding
	// a_errormetric is used to choose the best encoding
	// a_pafrgbaSource points to a 4x4 block subset of the source image
	// a_paucEncodingBits points to the final encoding bits of a previous encoding
	//
	void Block4x4Encoding_R11::InitFromEncodingBits(Block4x4 *a_pblockParent,
		unsigned char *a_paucEncodingBits,
		ColorFloatRGBA *a_pafrgbaSource,
		ErrorMetric a_errormetric)
	{
		m_pencodingbitsR11 = (Block4x4EncodingBits_R11 *)a_paucEncodingBits;

		// init RGB portion
		Block4x4Encoding_RGB8::InitFromEncodingBits(a_pblockParent,
			(unsigned char *)m_pencodingbitsR11,
			a_pafrgbaSource,
			a_errormetric);

		// init R11 portion
		{
			m_mode = MODE_R11;
			if (a_pblockParent->GetImageSource()->GetFormat() == Image::Format::SIGNED_R11 || a_pblockParent->GetImageSource()->GetFormat() == Image::Format::SIGNED_RG11)
			{
				m_fRedBase = (float)(signed char)m_pencodingbitsR11->data.base;
			}
			else
			{
				m_fRedBase = (float)(unsigned char)m_pencodingbitsR11->data.base;
			}
			m_fRedMultiplier = (float)m_pencodingbitsR11->data.multiplier;
			m_uiRedModifierTableIndex = m_pencodingbitsR11->data.table;

			unsigned long long int ulliSelectorBits = 0;
			ulliSelectorBits |= (unsigned long long int)m_pencodingbitsR11->data.selectors0 << 40;
			ulliSelectorBits |= (unsigned long long int)m_pencodingbitsR11->data.selectors1 << 32;
			ulliSelectorBits |= (unsigned long long int)m_pencodingbitsR11->data.selectors2 << 24;
			ulliSelectorBits |= (unsigned long long int)m_pencodingbitsR11->data.selectors3 << 16;
			ulliSelectorBits |= (unsigned long long int)m_pencodingbitsR11->data.selectors4 << 8;
			ulliSelectorBits |= (unsigned long long int)m_pencodingbitsR11->data.selectors5;
			for (unsigned int uiPixel = 0; uiPixel < PIXELS; uiPixel++)
			{
				unsigned int uiShift = 45 - (3 * uiPixel);
				m_auiRedSelectors[uiPixel] = (ulliSelectorBits >> uiShift) & (SELECTORS - 1);
			}

			// decode the red channel
			// calc red error
			for (unsigned int uiPixel = 0; uiPixel < PIXELS; uiPixel++)
			{
				float fDecodedPixelData = 0.0f;
				if (a_pblockParent->GetImageSource()->GetFormat() == Image::Format::R11 || a_pblockParent->GetImageSource()->GetFormat() == Image::Format::RG11)
				{
					fDecodedPixelData = DecodePixelRed(m_fRedBase, m_fRedMultiplier,
						m_uiRedModifierTableIndex,
						m_auiRedSelectors[uiPixel]);
				}
				else if (a_pblockParent->GetImageSource()->GetFormat() == Image::Format::SIGNED_R11 || a_pblockParent->GetImageSource()->GetFormat() == Image::Format::SIGNED_RG11)
				{
					fDecodedPixelData = DecodePixelRed(m_fRedBase + 128, m_fRedMultiplier,
						m_uiRedModifierTableIndex,
						m_auiRedSelectors[uiPixel]);
				}
				else
				{
					assert(0);
				}
				m_afrgbaDecodedColors[uiPixel] = ColorFloatRGBA(fDecodedPixelData, 0.0f, 0.0f, 1.0f);
			}
			CalcBlockError();
		}

	}

	// ----------------------------------------------------------------------------------------------------
	// perform a single encoding iteration
	// replace the encoding if a better encoding was found
	// subsequent iterations generally take longer for each iteration
	// set m_boolDone if encoding is perfect or encoding is finished based on a_fEffort
	//
	void Block4x4Encoding_R11::PerformIteration(float a_fEffort)
	{
		assert(!m_boolDone);
		m_mode = MODE_R11;

		switch (m_uiEncodingIterations)
		{
		case 0:
			m_fError = FLT_MAX;
			m_fRedBlockError = FLT_MAX;		// artificially high value
			CalculateR11(8, 0.0f, 0.0f);
			m_fError = m_fRedBlockError;
			break;

		case 1:
			CalculateR11(8, 2.0f, 1.0f);
			m_fError = m_fRedBlockError;
			if (a_fEffort <= 24.5f)
			{
				m_boolDone = true;
			}
			break;

		case 2:
			CalculateR11(8, 12.0f, 1.0f);
			m_fError = m_fRedBlockError;
			if (a_fEffort <= 49.5f)
			{
				m_boolDone = true;
			}
			break;

		case 3:
			CalculateR11(7, 6.0f, 1.0f);
			m_fError = m_fRedBlockError;
			break;

		case 4:
			CalculateR11(6, 3.0f, 1.0f);
			m_fError = m_fRedBlockError;
			break;

		case 5:
			CalculateR11(5, 1.0f, 0.0f);
			m_fError = m_fRedBlockError;
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
	void Block4x4Encoding_R11::CalculateR11(unsigned int a_uiSelectorsUsed, 
												float a_fBaseRadius, float a_fMultiplierRadius)
	{
		// maps from virtual (monotonic) selector to ETC selector
		static const unsigned int auiVirtualSelectorMap[8] = {3, 2, 1, 0, 4, 5, 6, 7};

		// find min/max red
		float fMinRed = 1.0f;
		float fMaxRed = 0.0f;
		for (unsigned int uiPixel = 0; uiPixel < PIXELS; uiPixel++)
		{
			// ignore border pixels
			float fAlpha = m_pafrgbaSource[uiPixel].fA;
			if (isnan(fAlpha))
			{
				continue;
			}

			float fRed = m_pafrgbaSource[uiPixel].fR;

			if (fRed < fMinRed)
			{
				fMinRed = fRed;
			}
			if (fRed > fMaxRed)
			{
				fMaxRed = fRed;
			}
		}
		assert(fMinRed <= fMaxRed);

		float fRedRange = (fMaxRed - fMinRed);

		// try each modifier table entry							  
		for (unsigned int uiTableEntry = 0; uiTableEntry < MODIFIER_TABLE_ENTRYS; uiTableEntry++)
		{
			for (unsigned int uiMinVirtualSelector = 0; 
					uiMinVirtualSelector <= (8- a_uiSelectorsUsed); 
					uiMinVirtualSelector++)
			{
				unsigned int uiMaxVirtualSelector = uiMinVirtualSelector + a_uiSelectorsUsed - 1;

				unsigned int uiMinSelector = auiVirtualSelectorMap[uiMinVirtualSelector];
				unsigned int uiMaxSelector = auiVirtualSelectorMap[uiMaxVirtualSelector];

				float fTableEntryCenter = -s_aafModifierTable[uiTableEntry][uiMinSelector];

				float fTableEntryRange = s_aafModifierTable[uiTableEntry][uiMaxSelector] -
											s_aafModifierTable[uiTableEntry][uiMinSelector];

				float fCenterRatio = fTableEntryCenter / fTableEntryRange;

				float fCenter = fMinRed + fCenterRatio*fRedRange;
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
					float fRangeMultiplier = roundf(fRedRange / fTableEntryRange);

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
						float afBestRedError[PIXELS];
						float afBestPixelRed[PIXELS];

						for (unsigned int uiPixel = 0; uiPixel < PIXELS; uiPixel++)
						{
							float fBestPixelRedError = FLT_MAX;

							for (unsigned int uiSelector = 0; uiSelector < SELECTORS; uiSelector++)
							{
								float fPixelRed = DecodePixelRed(fBase * 255.0f, fMultiplier, uiTableEntry, uiSelector);

								ColorFloatRGBA frgba(fPixelRed, m_pafrgbaSource[uiPixel].fG,0.0f,1.0f);

								float fPixelRedError = CalcPixelError(frgba, 1.0f, m_pafrgbaSource[uiPixel]);

								if (fPixelRedError < fBestPixelRedError)
								{
									fBestPixelRedError = fPixelRedError;
									auiBestSelectors[uiPixel] = uiSelector;
									afBestRedError[uiPixel] = fBestPixelRedError;
									afBestPixelRed[uiPixel] = fPixelRed;
								}
							}
						}
						float fBlockError = 0.0f;  
						for (unsigned int uiPixel = 0; uiPixel < PIXELS; uiPixel++)
						{
							fBlockError += afBestRedError[uiPixel];
						}
						if (fBlockError < m_fRedBlockError)
						{
							m_fRedBlockError = fBlockError;

							if (m_pblockParent->GetImageSource()->GetFormat() == Image::Format::R11 || m_pblockParent->GetImageSource()->GetFormat() == Image::Format::RG11)
							{
								m_fRedBase = 255.0f * fBase;
							}
							else if (m_pblockParent->GetImageSource()->GetFormat() == Image::Format::SIGNED_R11 || m_pblockParent->GetImageSource()->GetFormat() == Image::Format::SIGNED_RG11)
							{
								m_fRedBase = (fBase * 255) - 128;
							}
							else
							{
								assert(0);
							}
							m_fRedMultiplier = fMultiplier;
							m_uiRedModifierTableIndex = uiTableEntry;

							for (unsigned int uiPixel = 0; uiPixel < PIXELS; uiPixel++)
							{
								m_auiRedSelectors[uiPixel] = auiBestSelectors[uiPixel];
								float fBestPixelRed = afBestPixelRed[uiPixel];
								m_afrgbaDecodedColors[uiPixel] = ColorFloatRGBA(fBestPixelRed, 0.0f, 0.0f, 1.0f);
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
	void Block4x4Encoding_R11::SetEncodingBits(void)
	{
		if (m_pblockParent->GetImageSource()->GetFormat() == Image::Format::R11 || m_pblockParent->GetImageSource()->GetFormat() == Image::Format::RG11)
		{
			m_pencodingbitsR11->data.base = (unsigned char)roundf(m_fRedBase);
		}
		else if (m_pblockParent->GetImageSource()->GetFormat() == Image::Format::SIGNED_R11 || m_pblockParent->GetImageSource()->GetFormat() == Image::Format::SIGNED_RG11)
		{
			m_pencodingbitsR11->data.base = (signed char)roundf(m_fRedBase);
		}
		else
		{
			assert(0);
		}
		m_pencodingbitsR11->data.table = m_uiRedModifierTableIndex;
		m_pencodingbitsR11->data.multiplier = (unsigned char)roundf(m_fRedMultiplier);

		unsigned long long int ulliSelectorBits = 0;
		for (unsigned int uiPixel = 0; uiPixel < PIXELS; uiPixel++)
		{
			unsigned int uiShift = 45 - (3 * uiPixel);
			ulliSelectorBits |= ((unsigned long long int)m_auiRedSelectors[uiPixel]) << uiShift;
		}

		m_pencodingbitsR11->data.selectors0 = ulliSelectorBits >> 40;
		m_pencodingbitsR11->data.selectors1 = ulliSelectorBits >> 32;
		m_pencodingbitsR11->data.selectors2 = ulliSelectorBits >> 24;
		m_pencodingbitsR11->data.selectors3 = ulliSelectorBits >> 16;
		m_pencodingbitsR11->data.selectors4 = ulliSelectorBits >> 8;
		m_pencodingbitsR11->data.selectors5 = ulliSelectorBits;
	}

	// ----------------------------------------------------------------------------------------------------
	//
}
