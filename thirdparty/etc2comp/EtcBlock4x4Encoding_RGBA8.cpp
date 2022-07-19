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
EtcBlock4x4Encoding_RGBA8.cpp contains:
	Block4x4Encoding_RGBA8
	Block4x4Encoding_RGBA8_Opaque
	Block4x4Encoding_RGBA8_Transparent

These encoders are used when targetting file format RGBA8.

Block4x4Encoding_RGBA8_Opaque is used when all pixels in the 4x4 block are opaque
Block4x4Encoding_RGBA8_Transparent is used when all pixels in the 4x4 block are transparent
Block4x4Encoding_RGBA8 is used when there is a mixture of alphas in the 4x4 block

*/

#include "EtcConfig.h"
#include "EtcBlock4x4Encoding_RGBA8.h"

#include "EtcBlock4x4EncodingBits.h"
#include "EtcBlock4x4.h"

#include <stdio.h>
#include <string.h>
#include <assert.h>
#include <float.h>
#include <limits>

namespace Etc
{

	// ####################################################################################################
	// Block4x4Encoding_RGBA8
	// ####################################################################################################

	float Block4x4Encoding_RGBA8::s_aafModifierTable[MODIFIER_TABLE_ENTRYS][ALPHA_SELECTORS]
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
	Block4x4Encoding_RGBA8::Block4x4Encoding_RGBA8(void)
	{

		m_pencodingbitsA8 = nullptr;

	}
	Block4x4Encoding_RGBA8::~Block4x4Encoding_RGBA8(void) {}
	// ----------------------------------------------------------------------------------------------------
	// initialization prior to encoding
	// a_pblockParent points to the block associated with this encoding
	// a_errormetric is used to choose the best encoding
	// a_pafrgbaSource points to a 4x4 block subset of the source image
	// a_paucEncodingBits points to the final encoding bits
	//
	void Block4x4Encoding_RGBA8::InitFromSource(Block4x4 *a_pblockParent,
												ColorFloatRGBA *a_pafrgbaSource,
												unsigned char *a_paucEncodingBits, ErrorMetric a_errormetric)
	{
		Block4x4Encoding::Init(a_pblockParent, a_pafrgbaSource,a_errormetric);

		m_pencodingbitsA8 = (Block4x4EncodingBits_A8 *)a_paucEncodingBits;
		m_pencodingbitsRGB8 = (Block4x4EncodingBits_RGB8 *)(a_paucEncodingBits + sizeof(Block4x4EncodingBits_A8));

	}

	// ----------------------------------------------------------------------------------------------------
	// initialization from the encoding bits of a previous encoding
	// a_pblockParent points to the block associated with this encoding
	// a_errormetric is used to choose the best encoding
	// a_pafrgbaSource points to a 4x4 block subset of the source image
	// a_paucEncodingBits points to the final encoding bits of a previous encoding
	//
	void Block4x4Encoding_RGBA8::InitFromEncodingBits(Block4x4 *a_pblockParent,
														unsigned char *a_paucEncodingBits,
														ColorFloatRGBA *a_pafrgbaSource,
														ErrorMetric a_errormetric)
	{

		m_pencodingbitsA8 = (Block4x4EncodingBits_A8 *)a_paucEncodingBits;
		m_pencodingbitsRGB8 = (Block4x4EncodingBits_RGB8 *)(a_paucEncodingBits + sizeof(Block4x4EncodingBits_A8));

		// init RGB portion
		Block4x4Encoding_RGB8::InitFromEncodingBits(a_pblockParent,
													(unsigned char *) m_pencodingbitsRGB8,
													a_pafrgbaSource,
													a_errormetric);

		// init A8 portion
		// has to be done after InitFromEncodingBits()
		{
			m_fBase = m_pencodingbitsA8->data.base / 255.0f;
			m_fMultiplier = (float)m_pencodingbitsA8->data.multiplier;
			m_uiModifierTableIndex = m_pencodingbitsA8->data.table;

			unsigned long long int ulliSelectorBits = 0;
			ulliSelectorBits |= (unsigned long long int)m_pencodingbitsA8->data.selectors0 << 40;
			ulliSelectorBits |= (unsigned long long int)m_pencodingbitsA8->data.selectors1 << 32;
			ulliSelectorBits |= (unsigned long long int)m_pencodingbitsA8->data.selectors2 << 24;
			ulliSelectorBits |= (unsigned long long int)m_pencodingbitsA8->data.selectors3 << 16;
			ulliSelectorBits |= (unsigned long long int)m_pencodingbitsA8->data.selectors4 << 8;
			ulliSelectorBits |= (unsigned long long int)m_pencodingbitsA8->data.selectors5;
			for (unsigned int uiPixel = 0; uiPixel < PIXELS; uiPixel++)
			{
				unsigned int uiShift = 45 - (3 * uiPixel);
				m_auiAlphaSelectors[uiPixel] = (ulliSelectorBits >> uiShift) & (ALPHA_SELECTORS - 1);
			}

			// decode the alphas
			// calc alpha error
			m_fError = 0.0f;
			for (unsigned int uiPixel = 0; uiPixel < PIXELS; uiPixel++)
			{
				m_afDecodedAlphas[uiPixel] = DecodePixelAlpha(m_fBase, m_fMultiplier,
					m_uiModifierTableIndex,
					m_auiAlphaSelectors[uiPixel]);

				float fDeltaAlpha = m_afDecodedAlphas[uiPixel] - m_pafrgbaSource[uiPixel].fA;
				m_fError += fDeltaAlpha * fDeltaAlpha;
			}
		}

		// redo error calc to include alpha
		CalcBlockError();

	}

	// ----------------------------------------------------------------------------------------------------
	// perform a single encoding iteration
	// replace the encoding if a better encoding was found
	// subsequent iterations generally take longer for each iteration
	// set m_boolDone if encoding is perfect or encoding is finished based on a_fEffort
	//
	// similar to Block4x4Encoding_RGB8_Base::Encode_RGB8(), but with alpha added
	//
	void Block4x4Encoding_RGBA8::PerformIteration(float a_fEffort)
	{
		assert(!m_boolDone);

		if (m_uiEncodingIterations == 0)
		{
			if (a_fEffort < 24.9f)
			{
				CalculateA8(0.0f);
			}
			else if (a_fEffort < 49.9f)
			{
				CalculateA8(1.0f);
			}
			else
			{
				CalculateA8(2.0f);
			}
		}

		Block4x4Encoding_RGB8::PerformIteration(a_fEffort);

	}

	// ----------------------------------------------------------------------------------------------------
	// find the best combination of base alpga, multiplier and selectors
	//
	// a_fRadius limits the range of base alpha to try
	//
	void Block4x4Encoding_RGBA8::CalculateA8(float a_fRadius)
	{

		// find min/max alpha
		float fMinAlpha = 1.0f;
		float fMaxAlpha = 0.0f;
		for (unsigned int uiPixel = 0; uiPixel < PIXELS; uiPixel++)
		{
			float fAlpha = m_pafrgbaSource[uiPixel].fA;

			// ignore border pixels
			if (isnan(fAlpha))
			{
				continue;
			}

			if (fAlpha < fMinAlpha)
			{
				fMinAlpha = fAlpha;
			}
			if (fAlpha > fMaxAlpha)
			{
				fMaxAlpha = fAlpha;
			}
		}
		assert(fMinAlpha <= fMaxAlpha);

		float fAlphaRange = fMaxAlpha - fMinAlpha;

		// try each modifier table entry
		m_fError = FLT_MAX;		// artificially high value
		for (unsigned int uiTableEntry = 0; uiTableEntry < MODIFIER_TABLE_ENTRYS; uiTableEntry++)
		{
			static const unsigned int MIN_VALUE_SELECTOR = 3;
			static const unsigned int MAX_VALUE_SELECTOR = 7;

			float fTableEntryCenter = -s_aafModifierTable[uiTableEntry][MIN_VALUE_SELECTOR];

			float fTableEntryRange = s_aafModifierTable[uiTableEntry][MAX_VALUE_SELECTOR] -
				s_aafModifierTable[uiTableEntry][MIN_VALUE_SELECTOR];

			float fCenterRatio = fTableEntryCenter / fTableEntryRange;

			float fCenter = fMinAlpha + fCenterRatio*fAlphaRange;
			fCenter = roundf(255.0f * fCenter) / 255.0f;

			float fMinBase = fCenter - (a_fRadius / 255.0f);
			if (fMinBase < 0.0f)
			{
				fMinBase = 0.0f;
			}

			float fMaxBase = fCenter + (a_fRadius / 255.0f);
			if (fMaxBase > 1.0f)
			{
				fMaxBase = 1.0f;
			}

			for (float fBase = fMinBase; fBase <= fMaxBase; fBase += (0.999999f / 255.0f))
			{

				float fRangeMultiplier = roundf(fAlphaRange / fTableEntryRange);

				float fMinMultiplier = fRangeMultiplier - a_fRadius;
				if (fMinMultiplier < 1.0f)
				{
					fMinMultiplier = 1.0f;
				}
				else if (fMinMultiplier > 15.0f)
				{
					fMinMultiplier = 15.0f;
				}

				float fMaxMultiplier = fRangeMultiplier + a_fRadius;
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
					float afBestAlphaError[PIXELS];
					float afBestDecodedAlphas[PIXELS];
					for (unsigned int uiPixel = 0; uiPixel < PIXELS; uiPixel++)
					{
						float fBestPixelAlphaError = FLT_MAX;
						for (unsigned int uiSelector = 0; uiSelector < ALPHA_SELECTORS; uiSelector++)
						{
							float fDecodedAlpha = DecodePixelAlpha(fBase, fMultiplier, uiTableEntry, uiSelector);

							// border pixels (NAN) should have zero error
							float fPixelDeltaAlpha = isnan(m_pafrgbaSource[uiPixel].fA) ?
															0.0f :
															fDecodedAlpha - m_pafrgbaSource[uiPixel].fA;

							float fPixelAlphaError = fPixelDeltaAlpha * fPixelDeltaAlpha;

							if (fPixelAlphaError < fBestPixelAlphaError)
							{
								fBestPixelAlphaError = fPixelAlphaError;
								auiBestSelectors[uiPixel] = uiSelector;
								afBestAlphaError[uiPixel] = fBestPixelAlphaError;
								afBestDecodedAlphas[uiPixel] = fDecodedAlpha;
							}
						}
					}

					float fBlockError = 0.0f;
					for (unsigned int uiPixel = 0; uiPixel < PIXELS; uiPixel++)
					{
						fBlockError += afBestAlphaError[uiPixel];
					}

					if (fBlockError < m_fError)
					{
						m_fError = fBlockError;

						m_fBase = fBase;
						m_fMultiplier = fMultiplier;
						m_uiModifierTableIndex = uiTableEntry;
						for (unsigned int uiPixel = 0; uiPixel < PIXELS; uiPixel++)
						{
							m_auiAlphaSelectors[uiPixel] = auiBestSelectors[uiPixel];
							m_afDecodedAlphas[uiPixel] = afBestDecodedAlphas[uiPixel];
						}
					}
				}
			}

		}

	}

	// ----------------------------------------------------------------------------------------------------
	// set the encoding bits based on encoding state
	//
	void Block4x4Encoding_RGBA8::SetEncodingBits(void)
	{

		// set the RGB8 portion
		Block4x4Encoding_RGB8::SetEncodingBits();

		// set the A8 portion
		{
			m_pencodingbitsA8->data.base = (unsigned char)roundf(255.0f * m_fBase);
			m_pencodingbitsA8->data.table = m_uiModifierTableIndex;
			m_pencodingbitsA8->data.multiplier = (unsigned char)roundf(m_fMultiplier);

			unsigned long long int ulliSelectorBits = 0;
			for (unsigned int uiPixel = 0; uiPixel < PIXELS; uiPixel++)
			{
				unsigned int uiShift = 45 - (3 * uiPixel);
				ulliSelectorBits |= ((unsigned long long int)m_auiAlphaSelectors[uiPixel]) << uiShift;
			}

			m_pencodingbitsA8->data.selectors0 = ulliSelectorBits >> 40;
			m_pencodingbitsA8->data.selectors1 = ulliSelectorBits >> 32;
			m_pencodingbitsA8->data.selectors2 = ulliSelectorBits >> 24;
			m_pencodingbitsA8->data.selectors3 = ulliSelectorBits >> 16;
			m_pencodingbitsA8->data.selectors4 = ulliSelectorBits >> 8;
			m_pencodingbitsA8->data.selectors5 = ulliSelectorBits;
		}

	}

	// ####################################################################################################
	// Block4x4Encoding_RGBA8_Opaque
	// ####################################################################################################

	// ----------------------------------------------------------------------------------------------------
	// perform a single encoding iteration
	// replace the encoding if a better encoding was found
	// subsequent iterations generally take longer for each iteration
	// set m_boolDone if encoding is perfect or encoding is finished based on a_fEffort
	//
	void Block4x4Encoding_RGBA8_Opaque::PerformIteration(float a_fEffort)
	{
		assert(!m_boolDone);

		if (m_uiEncodingIterations == 0)
		{
			m_fError = 0.0f;

			for (unsigned int uiPixel = 0; uiPixel < PIXELS; uiPixel++)
			{
				m_afDecodedAlphas[uiPixel] = 1.0f;
			}
		}

		Block4x4Encoding_RGB8::PerformIteration(a_fEffort);

	}

	// ----------------------------------------------------------------------------------------------------
	// set the encoding bits based on encoding state
	//
	void Block4x4Encoding_RGBA8_Opaque::SetEncodingBits(void)
	{

		// set the RGB8 portion
		Block4x4Encoding_RGB8::SetEncodingBits();

		// set the A8 portion
		m_pencodingbitsA8->data.base = 255;
		m_pencodingbitsA8->data.table = 15;
		m_pencodingbitsA8->data.multiplier = 15;
		m_pencodingbitsA8->data.selectors0 = 0xFF;
		m_pencodingbitsA8->data.selectors1 = 0xFF;
		m_pencodingbitsA8->data.selectors2 = 0xFF;
		m_pencodingbitsA8->data.selectors3 = 0xFF;
		m_pencodingbitsA8->data.selectors4 = 0xFF;
		m_pencodingbitsA8->data.selectors5 = 0xFF;

	}

	// ####################################################################################################
	// Block4x4Encoding_RGBA8_Transparent
	// ####################################################################################################

	// ----------------------------------------------------------------------------------------------------
	// perform a single encoding iteration
	// replace the encoding if a better encoding was found
	// subsequent iterations generally take longer for each iteration
	// set m_boolDone if encoding is perfect or encoding is finished based on a_fEffort
	//
	void Block4x4Encoding_RGBA8_Transparent::PerformIteration(float )
	{
		assert(!m_boolDone);
		assert(m_uiEncodingIterations == 0);

		m_mode = MODE_ETC1;
		m_boolDiff = true;
		m_boolFlip = false;

		for (unsigned int uiPixel = 0; uiPixel < PIXELS; uiPixel++)
		{
			m_afrgbaDecodedColors[uiPixel] = ColorFloatRGBA();
			m_afDecodedAlphas[uiPixel] = 0.0f;
		}

		m_fError = 0.0f;

		m_boolDone = true;
		m_uiEncodingIterations++;

	}

	// ----------------------------------------------------------------------------------------------------
	// set the encoding bits based on encoding state
	//
	void Block4x4Encoding_RGBA8_Transparent::SetEncodingBits(void)
	{

		Block4x4Encoding_RGB8::SetEncodingBits();

		// set the A8 portion
		m_pencodingbitsA8->data.base = 0;
		m_pencodingbitsA8->data.table = 0;
		m_pencodingbitsA8->data.multiplier = 1;
		m_pencodingbitsA8->data.selectors0 = 0;
		m_pencodingbitsA8->data.selectors1 = 0;
		m_pencodingbitsA8->data.selectors2 = 0;
		m_pencodingbitsA8->data.selectors3 = 0;
		m_pencodingbitsA8->data.selectors4 = 0;
		m_pencodingbitsA8->data.selectors5 = 0;

	}

	// ----------------------------------------------------------------------------------------------------
	//
}
