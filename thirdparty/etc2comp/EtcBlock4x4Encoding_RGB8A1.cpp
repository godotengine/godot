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
EtcBlock4x4Encoding_RGB8A1.cpp contains:
	Block4x4Encoding_RGB8A1
	Block4x4Encoding_RGB8A1_Opaque
	Block4x4Encoding_RGB8A1_Transparent

These encoders are used when targetting file format RGB8A1.

Block4x4Encoding_RGB8A1_Opaque is used when all pixels in the 4x4 block are opaque
Block4x4Encoding_RGB8A1_Transparent is used when all pixels in the 4x4 block are transparent
Block4x4Encoding_RGB8A1 is used when there is a mixture of alphas in the 4x4 block

*/

#include "EtcConfig.h"
#include "EtcBlock4x4Encoding_RGB8A1.h"

#include "EtcBlock4x4.h"
#include "EtcBlock4x4EncodingBits.h"
#include "EtcBlock4x4Encoding_RGB8.h"

#include <stdio.h>
#include <string.h>
#include <assert.h>

namespace Etc
{
	
	// ####################################################################################################
	// Block4x4Encoding_RGB8A1
	// ####################################################################################################

	float Block4x4Encoding_RGB8A1::s_aafCwOpaqueUnsetTable[CW_RANGES][SELECTORS] =
	{
		{ 0.0f / 255.0f, 8.0f / 255.0f, 0.0f / 255.0f, -8.0f / 255.0f },
		{ 0.0f / 255.0f, 17.0f / 255.0f, 0.0f / 255.0f, -17.0f / 255.0f },
		{ 0.0f / 255.0f, 29.0f / 255.0f, 0.0f / 255.0f, -29.0f / 255.0f },
		{ 0.0f / 255.0f, 42.0f / 255.0f, 0.0f / 255.0f, -42.0f / 255.0f },
		{ 0.0f / 255.0f, 60.0f / 255.0f, 0.0f / 255.0f, -60.0f / 255.0f },
		{ 0.0f / 255.0f, 80.0f / 255.0f, 0.0f / 255.0f, -80.0f / 255.0f },
		{ 0.0f / 255.0f, 106.0f / 255.0f, 0.0f / 255.0f, -106.0f / 255.0f },
		{ 0.0f / 255.0f, 183.0f / 255.0f, 0.0f / 255.0f, -183.0f / 255.0f }
	};

	// ----------------------------------------------------------------------------------------------------
	//
	Block4x4Encoding_RGB8A1::Block4x4Encoding_RGB8A1(void)
	{
		m_pencodingbitsRGB8 = nullptr;
		m_boolOpaque = false;
		m_boolTransparent = false;
		m_boolPunchThroughPixels = true;

	}
	Block4x4Encoding_RGB8A1::~Block4x4Encoding_RGB8A1(void) {}
	// ----------------------------------------------------------------------------------------------------
	// initialization prior to encoding
	// a_pblockParent points to the block associated with this encoding
	// a_errormetric is used to choose the best encoding
	// a_pafrgbaSource points to a 4x4 block subset of the source image
	// a_paucEncodingBits points to the final encoding bits
	//
	void Block4x4Encoding_RGB8A1::InitFromSource(Block4x4 *a_pblockParent,
													ColorFloatRGBA *a_pafrgbaSource,
													unsigned char *a_paucEncodingBits,
													ErrorMetric a_errormetric)
	{

		Block4x4Encoding_RGB8::InitFromSource(a_pblockParent,
			a_pafrgbaSource,
			a_paucEncodingBits,
			a_errormetric);

		m_boolOpaque = a_pblockParent->GetSourceAlphaMix() == Block4x4::SourceAlphaMix::OPAQUE;
		m_boolTransparent = a_pblockParent->GetSourceAlphaMix() == Block4x4::SourceAlphaMix::TRANSPARENT;
		m_boolPunchThroughPixels = a_pblockParent->HasPunchThroughPixels();

		for (unsigned int uiPixel = 0; uiPixel < PIXELS; uiPixel++)
		{
			if (m_pafrgbaSource[uiPixel].fA >= 0.5f)
			{
				m_afDecodedAlphas[uiPixel] = 1.0f;
			}
			else
			{
				m_afDecodedAlphas[uiPixel] = 0.0f;
			}
		}

	}

	// ----------------------------------------------------------------------------------------------------
	// initialization from the encoding bits of a previous encoding
	// a_pblockParent points to the block associated with this encoding
	// a_errormetric is used to choose the best encoding
	// a_pafrgbaSource points to a 4x4 block subset of the source image
	// a_paucEncodingBits points to the final encoding bits of a previous encoding
	//
	void Block4x4Encoding_RGB8A1::InitFromEncodingBits(Block4x4 *a_pblockParent,
														unsigned char *a_paucEncodingBits,
														ColorFloatRGBA *a_pafrgbaSource,
														ErrorMetric a_errormetric)
	{


		InitFromEncodingBits_ETC1(a_pblockParent,
			a_paucEncodingBits,
			a_pafrgbaSource,
			a_errormetric);

		m_pencodingbitsRGB8 = (Block4x4EncodingBits_RGB8 *)a_paucEncodingBits;

		// detect if there is a T, H or Planar mode present
		int iRed1 = m_pencodingbitsRGB8->differential.red1;
		int iDRed2 = m_pencodingbitsRGB8->differential.dred2;
		int iRed2 = iRed1 + iDRed2;

		int iGreen1 = m_pencodingbitsRGB8->differential.green1;
		int iDGreen2 = m_pencodingbitsRGB8->differential.dgreen2;
		int iGreen2 = iGreen1 + iDGreen2;

		int iBlue1 = m_pencodingbitsRGB8->differential.blue1;
		int iDBlue2 = m_pencodingbitsRGB8->differential.dblue2;
		int iBlue2 = iBlue1 + iDBlue2;

		if (iRed2 < 0 || iRed2 > 31)
		{
			InitFromEncodingBits_T();
		}
		else if (iGreen2 < 0 || iGreen2 > 31)
		{
			InitFromEncodingBits_H();
		}
		else if (iBlue2 < 0 || iBlue2 > 31)
		{
			Block4x4Encoding_RGB8::InitFromEncodingBits_Planar();
		}

	}

	// ----------------------------------------------------------------------------------------------------
	// initialization from the encoding bits of a previous encoding assuming the encoding is an ETC1 mode.
	// if it isn't an ETC1 mode, this will be overwritten later
	//
	void Block4x4Encoding_RGB8A1::InitFromEncodingBits_ETC1(Block4x4 *a_pblockParent,
		unsigned char *a_paucEncodingBits,
		ColorFloatRGBA *a_pafrgbaSource,
		ErrorMetric a_errormetric)
	{
		Block4x4Encoding::Init(a_pblockParent, a_pafrgbaSource,
			a_errormetric);

		m_pencodingbitsRGB8 = (Block4x4EncodingBits_RGB8 *)a_paucEncodingBits;

		m_mode = MODE_ETC1;
		m_boolDiff = true;
		m_boolFlip = m_pencodingbitsRGB8->differential.flip;
		m_boolOpaque = m_pencodingbitsRGB8->differential.diff;

		int iR2 = m_pencodingbitsRGB8->differential.red1 + m_pencodingbitsRGB8->differential.dred2;
		if (iR2 < 0)
		{
			iR2 = 0;
		}
		else if (iR2 > 31)
		{
			iR2 = 31;
		}

		int iG2 = m_pencodingbitsRGB8->differential.green1 + m_pencodingbitsRGB8->differential.dgreen2;
		if (iG2 < 0)
		{
			iG2 = 0;
		}
		else if (iG2 > 31)
		{
			iG2 = 31;
		}

		int iB2 = m_pencodingbitsRGB8->differential.blue1 + m_pencodingbitsRGB8->differential.dblue2;
		if (iB2 < 0)
		{
			iB2 = 0;
		}
		else if (iB2 > 31)
		{
			iB2 = 31;
		}

		m_frgbaColor1 = ColorFloatRGBA::ConvertFromRGB5(m_pencodingbitsRGB8->differential.red1, m_pencodingbitsRGB8->differential.green1, m_pencodingbitsRGB8->differential.blue1);
		m_frgbaColor2 = ColorFloatRGBA::ConvertFromRGB5((unsigned char)iR2, (unsigned char)iG2, (unsigned char)iB2);

		m_uiCW1 = m_pencodingbitsRGB8->differential.cw1;
		m_uiCW2 = m_pencodingbitsRGB8->differential.cw2;

		Block4x4Encoding_ETC1::InitFromEncodingBits_Selectors();

		Decode_ETC1();

		CalcBlockError();

	}

	// ----------------------------------------------------------------------------------------------------
	// initialization from the encoding bits of a previous encoding if T mode is detected
	//
	void Block4x4Encoding_RGB8A1::InitFromEncodingBits_T(void)
	{
		m_mode = MODE_T;

		unsigned char ucRed1 = (unsigned char)((m_pencodingbitsRGB8->t.red1a << 2) +
								m_pencodingbitsRGB8->t.red1b);
		unsigned char ucGreen1 = m_pencodingbitsRGB8->t.green1;
		unsigned char ucBlue1 = m_pencodingbitsRGB8->t.blue1;

		unsigned char ucRed2 = m_pencodingbitsRGB8->t.red2;
		unsigned char ucGreen2 = m_pencodingbitsRGB8->t.green2;
		unsigned char ucBlue2 = m_pencodingbitsRGB8->t.blue2;

		m_frgbaColor1 = ColorFloatRGBA::ConvertFromRGB4(ucRed1, ucGreen1, ucBlue1);
		m_frgbaColor2 = ColorFloatRGBA::ConvertFromRGB4(ucRed2, ucGreen2, ucBlue2);

		m_uiCW1 = (m_pencodingbitsRGB8->t.da << 1) + m_pencodingbitsRGB8->t.db;

		Block4x4Encoding_ETC1::InitFromEncodingBits_Selectors();

		DecodePixels_T();

		CalcBlockError();
	}

	// ----------------------------------------------------------------------------------------------------
	// initialization from the encoding bits of a previous encoding if H mode is detected
	//
	void Block4x4Encoding_RGB8A1::InitFromEncodingBits_H(void)
	{
		m_mode = MODE_H;

		unsigned char ucRed1 = m_pencodingbitsRGB8->h.red1;
		unsigned char ucGreen1 = (unsigned char)((m_pencodingbitsRGB8->h.green1a << 1) +
									m_pencodingbitsRGB8->h.green1b);
		unsigned char ucBlue1 = (unsigned char)((m_pencodingbitsRGB8->h.blue1a << 3) +
								(m_pencodingbitsRGB8->h.blue1b << 1) +
								m_pencodingbitsRGB8->h.blue1c);

		unsigned char ucRed2 = m_pencodingbitsRGB8->h.red2;
		unsigned char ucGreen2 = (unsigned char)((m_pencodingbitsRGB8->h.green2a << 1) +
									m_pencodingbitsRGB8->h.green2b);
		unsigned char ucBlue2 = m_pencodingbitsRGB8->h.blue2;

		m_frgbaColor1 = ColorFloatRGBA::ConvertFromRGB4(ucRed1, ucGreen1, ucBlue1);
		m_frgbaColor2 = ColorFloatRGBA::ConvertFromRGB4(ucRed2, ucGreen2, ucBlue2);

		// used to determine the LSB of the CW
		unsigned int uiRGB1 = (unsigned int)(((int)ucRed1 << 16) + ((int)ucGreen1 << 8) + (int)ucBlue1);
		unsigned int uiRGB2 = (unsigned int)(((int)ucRed2 << 16) + ((int)ucGreen2 << 8) + (int)ucBlue2);

		m_uiCW1 = (m_pencodingbitsRGB8->h.da << 2) + (m_pencodingbitsRGB8->h.db << 1);
		if (uiRGB1 >= uiRGB2)
		{
			m_uiCW1++;
		}

		Block4x4Encoding_ETC1::InitFromEncodingBits_Selectors();

		DecodePixels_H();

		CalcBlockError();
	}

	// ----------------------------------------------------------------------------------------------------
	// for ETC1 modes, set the decoded colors and decoded alpha based on the encoding state
	//
	void Block4x4Encoding_RGB8A1::Decode_ETC1(void)
	{

		const unsigned int *pauiPixelOrder = m_boolFlip ? s_auiPixelOrderFlip1 : s_auiPixelOrderFlip0;

		for (unsigned int uiPixelOrder = 0; uiPixelOrder < PIXELS; uiPixelOrder++)
		{
			ColorFloatRGBA *pfrgbaCenter = uiPixelOrder < 8 ? &m_frgbaColor1 : &m_frgbaColor2;
			unsigned int uiCW = uiPixelOrder < 8 ? m_uiCW1 : m_uiCW2;

			unsigned int uiPixel = pauiPixelOrder[uiPixelOrder];

			float fDelta;
			if (m_boolOpaque)
				fDelta = Block4x4Encoding_ETC1::s_aafCwTable[uiCW][m_auiSelectors[uiPixel]];
			else 
				fDelta = s_aafCwOpaqueUnsetTable[uiCW][m_auiSelectors[uiPixel]];

			if (m_boolOpaque == false && m_auiSelectors[uiPixel] == TRANSPARENT_SELECTOR)
			{
				m_afrgbaDecodedColors[uiPixel] = ColorFloatRGBA();
				m_afDecodedAlphas[uiPixel] = 0.0f;
			}
			else
			{
				m_afrgbaDecodedColors[uiPixel] = (*pfrgbaCenter + fDelta).ClampRGB();
				m_afDecodedAlphas[uiPixel] = 1.0f;
			}
		}

	}

	// ----------------------------------------------------------------------------------------------------
	// for T mode, set the decoded colors and decoded alpha based on the encoding state
	//
	void Block4x4Encoding_RGB8A1::DecodePixels_T(void)
	{

		float fDistance = s_afTHDistanceTable[m_uiCW1];
		ColorFloatRGBA frgbaDistance(fDistance, fDistance, fDistance, 0.0f);

		for (unsigned int uiPixel = 0; uiPixel < PIXELS; uiPixel++)
		{
			switch (m_auiSelectors[uiPixel])
			{
			case 0:
				m_afrgbaDecodedColors[uiPixel] = m_frgbaColor1;
				m_afDecodedAlphas[uiPixel] = 1.0f;
				break;

			case 1:
				m_afrgbaDecodedColors[uiPixel] = (m_frgbaColor2 + frgbaDistance).ClampRGB();
				m_afDecodedAlphas[uiPixel] = 1.0f;
				break;

			case 2:
				if (m_boolOpaque == false)
				{
					m_afrgbaDecodedColors[uiPixel] = ColorFloatRGBA();
					m_afDecodedAlphas[uiPixel] = 0.0f;
				}
				else
				{
					m_afrgbaDecodedColors[uiPixel] = m_frgbaColor2;
					m_afDecodedAlphas[uiPixel] = 1.0f;
				}
				break;

			case 3:
				m_afrgbaDecodedColors[uiPixel] = (m_frgbaColor2 - frgbaDistance).ClampRGB();
				m_afDecodedAlphas[uiPixel] = 1.0f;
				break;
			}

		}

	}

	// ----------------------------------------------------------------------------------------------------
	// for H mode, set the decoded colors and decoded alpha based on the encoding state
	//
	void Block4x4Encoding_RGB8A1::DecodePixels_H(void)
	{

		float fDistance = s_afTHDistanceTable[m_uiCW1];
		ColorFloatRGBA frgbaDistance(fDistance, fDistance, fDistance, 0.0f);

		for (unsigned int uiPixel = 0; uiPixel < PIXELS; uiPixel++)
		{
			switch (m_auiSelectors[uiPixel])
			{
			case 0:
				m_afrgbaDecodedColors[uiPixel] = (m_frgbaColor1 + frgbaDistance).ClampRGB();
				m_afDecodedAlphas[uiPixel] = 1.0f;
				break;

			case 1:
				m_afrgbaDecodedColors[uiPixel] = (m_frgbaColor1 - frgbaDistance).ClampRGB();
				m_afDecodedAlphas[uiPixel] = 1.0f;
				break;

			case 2:
				if (m_boolOpaque == false)
				{
					m_afrgbaDecodedColors[uiPixel] = ColorFloatRGBA();
					m_afDecodedAlphas[uiPixel] = 0.0f;
				}
				else
				{
					m_afrgbaDecodedColors[uiPixel] = (m_frgbaColor2 + frgbaDistance).ClampRGB();
					m_afDecodedAlphas[uiPixel] = 1.0f;
				}
				break;

			case 3:
				m_afrgbaDecodedColors[uiPixel] = (m_frgbaColor2 - frgbaDistance).ClampRGB();
				m_afDecodedAlphas[uiPixel] = 1.0f;
				break;
			}

		}

	}


	// ----------------------------------------------------------------------------------------------------
	// perform a single encoding iteration
	// replace the encoding if a better encoding was found
	// subsequent iterations generally take longer for each iteration
	// set m_boolDone if encoding is perfect or encoding is finished based on a_fEffort
	//
	// RGB8A1 can't use individual mode
	// RGB8A1 with transparent pixels can't use planar mode
	//
	void Block4x4Encoding_RGB8A1::PerformIteration(float a_fEffort)
	{
		assert(!m_boolOpaque);
		assert(!m_boolTransparent);
		assert(!m_boolDone);

		switch (m_uiEncodingIterations)
		{
		case 0:
			PerformFirstIteration();
			break;

		case 1:
			TryDifferential(m_boolMostLikelyFlip, 1, 0, 0);
			break;

		case 2:
			TryDifferential(!m_boolMostLikelyFlip, 1, 0, 0);
			if (a_fEffort <= 39.5f)
			{
				m_boolDone = true;
			}
			break;

		case 3:
			Block4x4Encoding_RGB8::CalculateBaseColorsForTAndH();
			TryT(1);
			TryH(1);
			if (a_fEffort <= 49.5f)
			{
				m_boolDone = true;
			}
			break;

		case 4:
			TryDegenerates1();
			if (a_fEffort <= 59.5f)
			{
				m_boolDone = true;
			}
			break;

		case 5:
			TryDegenerates2();
			if (a_fEffort <= 69.5f)
			{
				m_boolDone = true;
			}
			break;

		case 6:
			TryDegenerates3();
			if (a_fEffort <= 79.5f)
			{
				m_boolDone = true;
			}
			break;

		case 7:
			TryDegenerates4();
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
	// find best initial encoding to ensure block has a valid encoding
	//
	void Block4x4Encoding_RGB8A1::PerformFirstIteration(void)
	{
		Block4x4Encoding_ETC1::CalculateMostLikelyFlip();

		m_fError = FLT_MAX;

		TryDifferential(m_boolMostLikelyFlip, 0, 0, 0);
		SetDoneIfPerfect();
		if (m_boolDone)
		{
			return;
		}
		TryDifferential(!m_boolMostLikelyFlip, 0, 0, 0);
		SetDoneIfPerfect();

	}

	// ----------------------------------------------------------------------------------------------------
	// mostly copied from ETC1
	// differences:
	//		Block4x4Encoding_RGB8A1 encodingTry = *this;
	//
	void Block4x4Encoding_RGB8A1::TryDifferential(bool a_boolFlip, unsigned int a_uiRadius, 
													int a_iGrayOffset1, int a_iGrayOffset2)
	{

		ColorFloatRGBA frgbaColor1;
		ColorFloatRGBA frgbaColor2;

		const unsigned int *pauiPixelMapping1;
		const unsigned int *pauiPixelMapping2;

		if (a_boolFlip)
		{
			frgbaColor1 = m_frgbaSourceAverageTop;
			frgbaColor2 = m_frgbaSourceAverageBottom;

			pauiPixelMapping1 = s_auiTopPixelMapping;
			pauiPixelMapping2 = s_auiBottomPixelMapping;
		}
		else
		{
			frgbaColor1 = m_frgbaSourceAverageLeft;
			frgbaColor2 = m_frgbaSourceAverageRight;

			pauiPixelMapping1 = s_auiLeftPixelMapping;
			pauiPixelMapping2 = s_auiRightPixelMapping;
		}

		DifferentialTrys trys(frgbaColor1, frgbaColor2, pauiPixelMapping1, pauiPixelMapping2, 
								a_uiRadius, a_iGrayOffset1, a_iGrayOffset2);

		Block4x4Encoding_RGB8A1 encodingTry = *this;
		encodingTry.m_boolFlip = a_boolFlip;

		encodingTry.TryDifferentialHalf(&trys.m_half1);
		encodingTry.TryDifferentialHalf(&trys.m_half2);

		// find best halves that are within differential range
		DifferentialTrys::Try *ptryBest1 = nullptr;
		DifferentialTrys::Try *ptryBest2 = nullptr;
		encodingTry.m_fError = FLT_MAX;

		// see if the best of each half are in differential range
		int iDRed = trys.m_half2.m_ptryBest->m_iRed - trys.m_half1.m_ptryBest->m_iRed;
		int iDGreen = trys.m_half2.m_ptryBest->m_iGreen - trys.m_half1.m_ptryBest->m_iGreen;
		int iDBlue = trys.m_half2.m_ptryBest->m_iBlue - trys.m_half1.m_ptryBest->m_iBlue;
		if (iDRed >= -4 && iDRed <= 3 && iDGreen >= -4 && iDGreen <= 3 && iDBlue >= -4 && iDBlue <= 3)
		{
			ptryBest1 = trys.m_half1.m_ptryBest;
			ptryBest2 = trys.m_half2.m_ptryBest;
			encodingTry.m_fError = trys.m_half1.m_ptryBest->m_fError + trys.m_half2.m_ptryBest->m_fError;
		}
		else
		{
			// else, find the next best halves that are in differential range
			for (DifferentialTrys::Try *ptry1 = &trys.m_half1.m_atry[0];
			ptry1 < &trys.m_half1.m_atry[trys.m_half1.m_uiTrys];
				ptry1++)
			{
				for (DifferentialTrys::Try *ptry2 = &trys.m_half2.m_atry[0];
				ptry2 < &trys.m_half2.m_atry[trys.m_half2.m_uiTrys];
					ptry2++)
				{
					iDRed = ptry2->m_iRed - ptry1->m_iRed;
					bool boolValidRedDelta = iDRed <= 3 && iDRed >= -4;
					iDGreen = ptry2->m_iGreen - ptry1->m_iGreen;
					bool boolValidGreenDelta = iDGreen <= 3 && iDGreen >= -4;
					iDBlue = ptry2->m_iBlue - ptry1->m_iBlue;
					bool boolValidBlueDelta = iDBlue <= 3 && iDBlue >= -4;

					if (boolValidRedDelta && boolValidGreenDelta && boolValidBlueDelta)
					{
						float fError = ptry1->m_fError + ptry2->m_fError;

						if (fError < encodingTry.m_fError)
						{
							encodingTry.m_fError = fError;

							ptryBest1 = ptry1;
							ptryBest2 = ptry2;
						}
					}

				}
			}
			assert(encodingTry.m_fError < FLT_MAX);
			assert(ptryBest1 != nullptr);
			assert(ptryBest2 != nullptr);
		}

		if (encodingTry.m_fError < m_fError)
		{
			m_mode = MODE_ETC1;
			m_boolDiff = true;
			m_boolFlip = encodingTry.m_boolFlip;
			m_frgbaColor1 = ColorFloatRGBA::ConvertFromRGB5((unsigned char)ptryBest1->m_iRed, (unsigned char)ptryBest1->m_iGreen, (unsigned char)ptryBest1->m_iBlue);
			m_frgbaColor2 = ColorFloatRGBA::ConvertFromRGB5((unsigned char)ptryBest2->m_iRed, (unsigned char)ptryBest2->m_iGreen, (unsigned char)ptryBest2->m_iBlue);
			m_uiCW1 = ptryBest1->m_uiCW;
			m_uiCW2 = ptryBest2->m_uiCW;

			m_fError = 0.0f;
			for (unsigned int uiPixelOrder = 0; uiPixelOrder < PIXELS / 2; uiPixelOrder++)
			{
				unsigned int uiPixel1 = pauiPixelMapping1[uiPixelOrder];
				unsigned int uiPixel2 = pauiPixelMapping2[uiPixelOrder];

				unsigned int uiSelector1 = ptryBest1->m_auiSelectors[uiPixelOrder];
				unsigned int uiSelector2 = ptryBest2->m_auiSelectors[uiPixelOrder];

				m_auiSelectors[uiPixel1] = uiSelector1;
				m_auiSelectors[uiPixel2] = ptryBest2->m_auiSelectors[uiPixelOrder];

				if (uiSelector1 == TRANSPARENT_SELECTOR)
				{
					m_afrgbaDecodedColors[uiPixel1] = ColorFloatRGBA();
					m_afDecodedAlphas[uiPixel1] = 0.0f;
				}
				else
				{
					float fDeltaRGB1 = s_aafCwOpaqueUnsetTable[m_uiCW1][uiSelector1];
					m_afrgbaDecodedColors[uiPixel1] = (m_frgbaColor1 + fDeltaRGB1).ClampRGB();
					m_afDecodedAlphas[uiPixel1] = 1.0f;
				}

				if (uiSelector2 == TRANSPARENT_SELECTOR)
				{
					m_afrgbaDecodedColors[uiPixel2] = ColorFloatRGBA();
					m_afDecodedAlphas[uiPixel2] = 0.0f;
				}
				else
				{
					float fDeltaRGB2 = s_aafCwOpaqueUnsetTable[m_uiCW2][uiSelector2];
					m_afrgbaDecodedColors[uiPixel2] = (m_frgbaColor2 + fDeltaRGB2).ClampRGB();
					m_afDecodedAlphas[uiPixel2] = 1.0f;
				}

				float fDeltaA1 = m_afDecodedAlphas[uiPixel1] - m_pafrgbaSource[uiPixel1].fA;
				m_fError += fDeltaA1 * fDeltaA1;
				float fDeltaA2 = m_afDecodedAlphas[uiPixel2] - m_pafrgbaSource[uiPixel2].fA;
				m_fError += fDeltaA2 * fDeltaA2;
			}

			m_fError1 = ptryBest1->m_fError;
			m_fError2 = ptryBest2->m_fError;
			m_boolSeverelyBentDifferentialColors = trys.m_boolSeverelyBentColors;
			m_fError = m_fError1 + m_fError2;

			// sanity check
			{
				int iRed1 = m_frgbaColor1.IntRed(31.0f);
				int iGreen1 = m_frgbaColor1.IntGreen(31.0f);
				int iBlue1 = m_frgbaColor1.IntBlue(31.0f);

				int iRed2 = m_frgbaColor2.IntRed(31.0f);
				int iGreen2 = m_frgbaColor2.IntGreen(31.0f);
				int iBlue2 = m_frgbaColor2.IntBlue(31.0f);

				iDRed = iRed2 - iRed1;
				iDGreen = iGreen2 - iGreen1;
				iDBlue = iBlue2 - iBlue1;

				assert(iDRed >= -4 && iDRed < 4);
				assert(iDGreen >= -4 && iDGreen < 4);
				assert(iDBlue >= -4 && iDBlue < 4);
			}
		}

	}

	// ----------------------------------------------------------------------------------------------------
	// mostly copied from ETC1
	// differences:
	//		uses s_aafCwOpaqueUnsetTable
	//		color for selector set to 0,0,0,0
	//
	void Block4x4Encoding_RGB8A1::TryDifferentialHalf(DifferentialTrys::Half *a_phalf)
	{

		a_phalf->m_ptryBest = nullptr;
		float fBestTryError = FLT_MAX;

		a_phalf->m_uiTrys = 0;
		for (int iRed = a_phalf->m_iRed - (int)a_phalf->m_uiRadius;
		iRed <= a_phalf->m_iRed + (int)a_phalf->m_uiRadius;
			iRed++)
		{
			assert(iRed >= 0 && iRed <= 31);

			for (int iGreen = a_phalf->m_iGreen - (int)a_phalf->m_uiRadius;
			iGreen <= a_phalf->m_iGreen + (int)a_phalf->m_uiRadius;
				iGreen++)
			{
				assert(iGreen >= 0 && iGreen <= 31);

				for (int iBlue = a_phalf->m_iBlue - (int)a_phalf->m_uiRadius;
				iBlue <= a_phalf->m_iBlue + (int)a_phalf->m_uiRadius;
					iBlue++)
				{
					assert(iBlue >= 0 && iBlue <= 31);

					DifferentialTrys::Try *ptry = &a_phalf->m_atry[a_phalf->m_uiTrys];
					assert(ptry < &a_phalf->m_atry[DifferentialTrys::Half::MAX_TRYS]);

					ptry->m_iRed = iRed;
					ptry->m_iGreen = iGreen;
					ptry->m_iBlue = iBlue;
					ptry->m_fError = FLT_MAX;
					ColorFloatRGBA frgbaColor = ColorFloatRGBA::ConvertFromRGB5((unsigned char)iRed, (unsigned char)iGreen, (unsigned char)iBlue);

					// try each CW
					for (unsigned int uiCW = 0; uiCW < CW_RANGES; uiCW++)
					{
						unsigned int auiPixelSelectors[PIXELS / 2];
						ColorFloatRGBA	afrgbaDecodedColors[PIXELS / 2];
						float afPixelErrors[PIXELS / 2] = { FLT_MAX, FLT_MAX, FLT_MAX, FLT_MAX,
							FLT_MAX, FLT_MAX, FLT_MAX, FLT_MAX };

						// pre-compute decoded pixels for each selector
						ColorFloatRGBA afrgbaSelectors[SELECTORS];
						assert(SELECTORS == 4);
						afrgbaSelectors[0] = (frgbaColor + s_aafCwOpaqueUnsetTable[uiCW][0]).ClampRGB();
						afrgbaSelectors[1] = (frgbaColor + s_aafCwOpaqueUnsetTable[uiCW][1]).ClampRGB();
						afrgbaSelectors[2] = ColorFloatRGBA();
						afrgbaSelectors[3] = (frgbaColor + s_aafCwOpaqueUnsetTable[uiCW][3]).ClampRGB();

						for (unsigned int uiPixel = 0; uiPixel < 8; uiPixel++)
						{
							ColorFloatRGBA *pfrgbaSourcePixel = &m_pafrgbaSource[a_phalf->m_pauiPixelMapping[uiPixel]];
							ColorFloatRGBA frgbaDecodedPixel;

							for (unsigned int uiSelector = 0; uiSelector < SELECTORS; uiSelector++)
							{
								if (pfrgbaSourcePixel->fA < 0.5f)
								{
									uiSelector = TRANSPARENT_SELECTOR;
								}
								else if (uiSelector == TRANSPARENT_SELECTOR)
								{
									continue;
								}

								frgbaDecodedPixel = afrgbaSelectors[uiSelector];

								float fPixelError;
								
								fPixelError = CalcPixelError(frgbaDecodedPixel, m_afDecodedAlphas[a_phalf->m_pauiPixelMapping[uiPixel]],
																	*pfrgbaSourcePixel);

								if (fPixelError < afPixelErrors[uiPixel])
								{
									auiPixelSelectors[uiPixel] = uiSelector;
									afrgbaDecodedColors[uiPixel] = frgbaDecodedPixel;
									afPixelErrors[uiPixel] = fPixelError;
								}

								if (uiSelector == TRANSPARENT_SELECTOR)
								{
									break;
								}
							}
						}

						// add up all pixel errors
						float fCWError = 0.0f;
						for (unsigned int uiPixel = 0; uiPixel < 8; uiPixel++)
						{
							fCWError += afPixelErrors[uiPixel];
						}

						// if best CW so far
						if (fCWError < ptry->m_fError)
						{
							ptry->m_uiCW = uiCW;
							for (unsigned int uiPixel = 0; uiPixel < 8; uiPixel++)
							{
								ptry->m_auiSelectors[uiPixel] = auiPixelSelectors[uiPixel];
							}
							ptry->m_fError = fCWError;
						}

					}

					if (ptry->m_fError < fBestTryError)
					{
						a_phalf->m_ptryBest = ptry;
						fBestTryError = ptry->m_fError;
					}

					assert(ptry->m_fError < FLT_MAX);

					a_phalf->m_uiTrys++;
				}
			}
		}

	}

	// ----------------------------------------------------------------------------------------------------
	// try encoding in T mode
	// save this encoding if it improves the error
	//
	// since pixels that use base color1 don't use the distance table, color1 and color2 can be twiddled independently
	// better encoding can be found if TWIDDLE_RADIUS is set to 2, but it will be much slower
	//
	void Block4x4Encoding_RGB8A1::TryT(unsigned int a_uiRadius)
	{
		Block4x4Encoding_RGB8A1 encodingTry = *this;

		// init "try"
		{
			encodingTry.m_mode = MODE_T;
			encodingTry.m_boolDiff = true;
			encodingTry.m_boolFlip = false;
			encodingTry.m_fError = FLT_MAX;
		}

		int iColor1Red = m_frgbaOriginalColor1_TAndH.IntRed(15.0f);
		int iColor1Green = m_frgbaOriginalColor1_TAndH.IntGreen(15.0f);
		int iColor1Blue = m_frgbaOriginalColor1_TAndH.IntBlue(15.0f);

		int iMinRed1 = iColor1Red - (int)a_uiRadius;
		if (iMinRed1 < 0)
		{
			iMinRed1 = 0;
		}
		int iMaxRed1 = iColor1Red + (int)a_uiRadius;
		if (iMaxRed1 > 15)
		{
			iMaxRed1 = 15;
		}

		int iMinGreen1 = iColor1Green - (int)a_uiRadius;
		if (iMinGreen1 < 0)
		{
			iMinGreen1 = 0;
		}
		int iMaxGreen1 = iColor1Green + (int)a_uiRadius;
		if (iMaxGreen1 > 15)
		{
			iMaxGreen1 = 15;
		}

		int iMinBlue1 = iColor1Blue - (int)a_uiRadius;
		if (iMinBlue1 < 0)
		{
			iMinBlue1 = 0;
		}
		int iMaxBlue1 = iColor1Blue + (int)a_uiRadius;
		if (iMaxBlue1 > 15)
		{
			iMaxBlue1 = 15;
		}

		int iColor2Red = m_frgbaOriginalColor2_TAndH.IntRed(15.0f);
		int iColor2Green = m_frgbaOriginalColor2_TAndH.IntGreen(15.0f);
		int iColor2Blue = m_frgbaOriginalColor2_TAndH.IntBlue(15.0f);

		int iMinRed2 = iColor2Red - (int)a_uiRadius;
		if (iMinRed2 < 0)
		{
			iMinRed2 = 0;
		}
		int iMaxRed2 = iColor2Red + (int)a_uiRadius;
		if (iMaxRed2 > 15)
		{
			iMaxRed2 = 15;
		}

		int iMinGreen2 = iColor2Green - (int)a_uiRadius;
		if (iMinGreen2 < 0)
		{
			iMinGreen2 = 0;
		}
		int iMaxGreen2 = iColor2Green + (int)a_uiRadius;
		if (iMaxGreen2 > 15)
		{
			iMaxGreen2 = 15;
		}

		int iMinBlue2 = iColor2Blue - (int)a_uiRadius;
		if (iMinBlue2 < 0)
		{
			iMinBlue2 = 0;
		}
		int iMaxBlue2 = iColor2Blue + (int)a_uiRadius;
		if (iMaxBlue2 > 15)
		{
			iMaxBlue2 = 15;
		}

		for (unsigned int uiDistance = 0; uiDistance < TH_DISTANCES; uiDistance++)
		{
			encodingTry.m_uiCW1 = uiDistance;

			// twiddle m_frgbaOriginalColor2_TAndH
			// twiddle color2 first, since it affects 3 selectors, while color1 only affects one selector
			//
			for (int iRed2 = iMinRed2; iRed2 <= iMaxRed2; iRed2++)
			{
				for (int iGreen2 = iMinGreen2; iGreen2 <= iMaxGreen2; iGreen2++)
				{
					for (int iBlue2 = iMinBlue2; iBlue2 <= iMaxBlue2; iBlue2++)
					{
						for (unsigned int uiBaseColorSwaps = 0; uiBaseColorSwaps < 2; uiBaseColorSwaps++)
						{
							if (uiBaseColorSwaps == 0)
							{
								encodingTry.m_frgbaColor1 = m_frgbaOriginalColor1_TAndH;
								encodingTry.m_frgbaColor2 = ColorFloatRGBA::ConvertFromRGB4((unsigned char)iRed2, (unsigned char)iGreen2, (unsigned char)iBlue2);
							}
							else
							{
								encodingTry.m_frgbaColor1 = ColorFloatRGBA::ConvertFromRGB4((unsigned char)iRed2, (unsigned char)iGreen2, (unsigned char)iBlue2);
								encodingTry.m_frgbaColor2 = m_frgbaOriginalColor1_TAndH;
							}

							encodingTry.TryT_BestSelectorCombination();

							if (encodingTry.m_fError < m_fError)
							{
								m_mode = encodingTry.m_mode;
								m_boolDiff = encodingTry.m_boolDiff;
								m_boolFlip = encodingTry.m_boolFlip;

								m_frgbaColor1 = encodingTry.m_frgbaColor1;
								m_frgbaColor2 = encodingTry.m_frgbaColor2;
								m_uiCW1 = encodingTry.m_uiCW1;

								for (unsigned int uiPixel = 0; uiPixel < PIXELS; uiPixel++)
								{
									m_auiSelectors[uiPixel] = encodingTry.m_auiSelectors[uiPixel];
									m_afrgbaDecodedColors[uiPixel] = encodingTry.m_afrgbaDecodedColors[uiPixel];
								}

								m_fError = encodingTry.m_fError;
							}
						}
					}
				}
			}

			// twiddle m_frgbaOriginalColor1_TAndH
			for (int iRed1 = iMinRed1; iRed1 <= iMaxRed1; iRed1++)
			{
				for (int iGreen1 = iMinGreen1; iGreen1 <= iMaxGreen1; iGreen1++)
				{
					for (int iBlue1 = iMinBlue1; iBlue1 <= iMaxBlue1; iBlue1++)
					{
						for (unsigned int uiBaseColorSwaps = 0; uiBaseColorSwaps < 2; uiBaseColorSwaps++)
						{
							if (uiBaseColorSwaps == 0)
							{
								encodingTry.m_frgbaColor1 = ColorFloatRGBA::ConvertFromRGB4((unsigned char)iRed1, (unsigned char)iGreen1, (unsigned char)iBlue1);
								encodingTry.m_frgbaColor2 = m_frgbaOriginalColor2_TAndH;
							}
							else
							{
								encodingTry.m_frgbaColor1 = m_frgbaOriginalColor2_TAndH;
								encodingTry.m_frgbaColor2 = ColorFloatRGBA::ConvertFromRGB4((unsigned char)iRed1, (unsigned char)iGreen1, (unsigned char)iBlue1);
							}

							encodingTry.TryT_BestSelectorCombination();

							if (encodingTry.m_fError < m_fError)
							{
								m_mode = encodingTry.m_mode;
								m_boolDiff = encodingTry.m_boolDiff;
								m_boolFlip = encodingTry.m_boolFlip;

								m_frgbaColor1 = encodingTry.m_frgbaColor1;
								m_frgbaColor2 = encodingTry.m_frgbaColor2;
								m_uiCW1 = encodingTry.m_uiCW1;

								for (unsigned int uiPixel = 0; uiPixel < PIXELS; uiPixel++)
								{
									m_auiSelectors[uiPixel] = encodingTry.m_auiSelectors[uiPixel];
									m_afrgbaDecodedColors[uiPixel] = encodingTry.m_afrgbaDecodedColors[uiPixel];
								}

								m_fError = encodingTry.m_fError;
							}
						}
					}
				}
			}

		}

	}

	// ----------------------------------------------------------------------------------------------------
	// find best selector combination for TryT
	// called on an encodingTry
	//
	void Block4x4Encoding_RGB8A1::TryT_BestSelectorCombination(void)
	{

		float fDistance = s_afTHDistanceTable[m_uiCW1];

		unsigned int auiBestPixelSelectors[PIXELS];
		float afBestPixelErrors[PIXELS] = { FLT_MAX, FLT_MAX, FLT_MAX, FLT_MAX, FLT_MAX, FLT_MAX, FLT_MAX, FLT_MAX,
			FLT_MAX, FLT_MAX, FLT_MAX, FLT_MAX, FLT_MAX, FLT_MAX, FLT_MAX, FLT_MAX };
		ColorFloatRGBA	afrgbaBestDecodedPixels[PIXELS];
		ColorFloatRGBA afrgbaDecodedPixel[SELECTORS];

		assert(SELECTORS == 4);
		afrgbaDecodedPixel[0] = m_frgbaColor1;
		afrgbaDecodedPixel[1] = (m_frgbaColor2 + fDistance).ClampRGB();
		afrgbaDecodedPixel[2] = ColorFloatRGBA();
		afrgbaDecodedPixel[3] = (m_frgbaColor2 - fDistance).ClampRGB();

		// try each selector
		for (unsigned int uiPixel = 0; uiPixel < PIXELS; uiPixel++)
		{
			unsigned int uiMinSelector = 0;
			unsigned int uiMaxSelector = SELECTORS - 1;

			if (m_pafrgbaSource[uiPixel].fA < 0.5f)
			{
				uiMinSelector = 2;
				uiMaxSelector = 2;
			}

			for (unsigned int uiSelector = uiMinSelector; uiSelector <= uiMaxSelector; uiSelector++)
			{
				float fPixelError = CalcPixelError(afrgbaDecodedPixel[uiSelector], m_afDecodedAlphas[uiPixel],
													m_pafrgbaSource[uiPixel]);

				if (fPixelError < afBestPixelErrors[uiPixel])
				{
					afBestPixelErrors[uiPixel] = fPixelError;
					auiBestPixelSelectors[uiPixel] = uiSelector;
					afrgbaBestDecodedPixels[uiPixel] = afrgbaDecodedPixel[uiSelector];
				}
			}
		}
		

		// add up all of the pixel errors
		float fBlockError = 0.0f;
		for (unsigned int uiPixel = 0; uiPixel < PIXELS; uiPixel++)
		{
			fBlockError += afBestPixelErrors[uiPixel];
		}

		if (fBlockError < m_fError)
		{
			m_fError = fBlockError;

			for (unsigned int uiPixel = 0; uiPixel < PIXELS; uiPixel++)
			{
				m_auiSelectors[uiPixel] = auiBestPixelSelectors[uiPixel];
				m_afrgbaDecodedColors[uiPixel] = afrgbaBestDecodedPixels[uiPixel];
			}
		}

	}

	// ----------------------------------------------------------------------------------------------------
	// try encoding in H mode
	// save this encoding if it improves the error
	//
	// since all pixels use the distance table, color1 and color2 can NOT be twiddled independently
	// TWIDDLE_RADIUS of 2 is WAY too slow
	//
	void Block4x4Encoding_RGB8A1::TryH(unsigned int a_uiRadius)
	{
		Block4x4Encoding_RGB8A1 encodingTry = *this;

		// init "try"
		{
			encodingTry.m_mode = MODE_H;
			encodingTry.m_boolDiff = true;
			encodingTry.m_boolFlip = false;
			encodingTry.m_fError = FLT_MAX;
		}

		int iColor1Red = m_frgbaOriginalColor1_TAndH.IntRed(15.0f);
		int iColor1Green = m_frgbaOriginalColor1_TAndH.IntGreen(15.0f);
		int iColor1Blue = m_frgbaOriginalColor1_TAndH.IntBlue(15.0f);

		int iMinRed1 = iColor1Red - (int)a_uiRadius;
		if (iMinRed1 < 0)
		{
			iMinRed1 = 0;
		}
		int iMaxRed1 = iColor1Red + (int)a_uiRadius;
		if (iMaxRed1 > 15)
		{
			iMaxRed1 = 15;
		}

		int iMinGreen1 = iColor1Green - (int)a_uiRadius;
		if (iMinGreen1 < 0)
		{
			iMinGreen1 = 0;
		}
		int iMaxGreen1 = iColor1Green + (int)a_uiRadius;
		if (iMaxGreen1 > 15)
		{
			iMaxGreen1 = 15;
		}

		int iMinBlue1 = iColor1Blue - (int)a_uiRadius;
		if (iMinBlue1 < 0)
		{
			iMinBlue1 = 0;
		}
		int iMaxBlue1 = iColor1Blue + (int)a_uiRadius;
		if (iMaxBlue1 > 15)
		{
			iMaxBlue1 = 15;
		}

		int iColor2Red = m_frgbaOriginalColor2_TAndH.IntRed(15.0f);
		int iColor2Green = m_frgbaOriginalColor2_TAndH.IntGreen(15.0f);
		int iColor2Blue = m_frgbaOriginalColor2_TAndH.IntBlue(15.0f);

		int iMinRed2 = iColor2Red - (int)a_uiRadius;
		if (iMinRed2 < 0)
		{
			iMinRed2 = 0;
		}
		int iMaxRed2 = iColor2Red + (int)a_uiRadius;
		if (iMaxRed2 > 15)
		{
			iMaxRed2 = 15;
		}

		int iMinGreen2 = iColor2Green - (int)a_uiRadius;
		if (iMinGreen2 < 0)
		{
			iMinGreen2 = 0;
		}
		int iMaxGreen2 = iColor2Green + (int)a_uiRadius;
		if (iMaxGreen2 > 15)
		{
			iMaxGreen2 = 15;
		}

		int iMinBlue2 = iColor2Blue - (int)a_uiRadius;
		if (iMinBlue2 < 0)
		{
			iMinBlue2 = 0;
		}
		int iMaxBlue2 = iColor2Blue + (int)a_uiRadius;
		if (iMaxBlue2 > 15)
		{
			iMaxBlue2 = 15;
		}

		for (unsigned int uiDistance = 0; uiDistance < TH_DISTANCES; uiDistance++)
		{
			encodingTry.m_uiCW1 = uiDistance;

			// twiddle m_frgbaOriginalColor1_TAndH
			for (int iRed1 = iMinRed1; iRed1 <= iMaxRed1; iRed1++)
			{
				for (int iGreen1 = iMinGreen1; iGreen1 <= iMaxGreen1; iGreen1++)
				{
					for (int iBlue1 = iMinBlue1; iBlue1 <= iMaxBlue1; iBlue1++)
					{
						encodingTry.m_frgbaColor1 = ColorFloatRGBA::ConvertFromRGB4((unsigned char)iRed1, (unsigned char)iGreen1, (unsigned char)iBlue1);
						encodingTry.m_frgbaColor2 = m_frgbaOriginalColor2_TAndH;

						// if color1 == color2, H encoding issues can pop up, so abort
						if (iRed1 == iColor2Red && iGreen1 == iColor2Green && iBlue1 == iColor2Blue)
						{
							continue;
						}

						encodingTry.TryH_BestSelectorCombination();

						if (encodingTry.m_fError < m_fError)
						{
							m_mode = encodingTry.m_mode;
							m_boolDiff = encodingTry.m_boolDiff;
							m_boolFlip = encodingTry.m_boolFlip;

							m_frgbaColor1 = encodingTry.m_frgbaColor1;
							m_frgbaColor2 = encodingTry.m_frgbaColor2;
							m_uiCW1 = encodingTry.m_uiCW1;

							for (unsigned int uiPixel = 0; uiPixel < PIXELS; uiPixel++)
							{
								m_auiSelectors[uiPixel] = encodingTry.m_auiSelectors[uiPixel];
								m_afrgbaDecodedColors[uiPixel] = encodingTry.m_afrgbaDecodedColors[uiPixel];
							}

							m_fError = encodingTry.m_fError;
						}
					}
				}
			}

			// twiddle m_frgbaOriginalColor2_TAndH
			for (int iRed2 = iMinRed2; iRed2 <= iMaxRed2; iRed2++)
			{
				for (int iGreen2 = iMinGreen2; iGreen2 <= iMaxGreen2; iGreen2++)
				{
					for (int iBlue2 = iMinBlue2; iBlue2 <= iMaxBlue2; iBlue2++)
					{
						encodingTry.m_frgbaColor1 = m_frgbaOriginalColor1_TAndH;
						encodingTry.m_frgbaColor2 = ColorFloatRGBA::ConvertFromRGB4((unsigned char)iRed2, (unsigned char)iGreen2, (unsigned char)iBlue2);

						// if color1 == color2, H encoding issues can pop up, so abort
						if (iRed2 == iColor1Red && iGreen2 == iColor1Green && iBlue2 == iColor1Blue)
						{
							continue;
						}

						encodingTry.TryH_BestSelectorCombination();

						if (encodingTry.m_fError < m_fError)
						{
							m_mode = encodingTry.m_mode;
							m_boolDiff = encodingTry.m_boolDiff;
							m_boolFlip = encodingTry.m_boolFlip;

							m_frgbaColor1 = encodingTry.m_frgbaColor1;
							m_frgbaColor2 = encodingTry.m_frgbaColor2;
							m_uiCW1 = encodingTry.m_uiCW1;

							for (unsigned int uiPixel = 0; uiPixel < PIXELS; uiPixel++)
							{
								m_auiSelectors[uiPixel] = encodingTry.m_auiSelectors[uiPixel];
								m_afrgbaDecodedColors[uiPixel] = encodingTry.m_afrgbaDecodedColors[uiPixel];
							}

							m_fError = encodingTry.m_fError;
						}
					}
				}
			}

		}

	}

	// ----------------------------------------------------------------------------------------------------
	// find best selector combination for TryH
	// called on an encodingTry
	//
	void Block4x4Encoding_RGB8A1::TryH_BestSelectorCombination(void)
	{

		// abort if colors and CW will pose an encoding problem
		{
			unsigned int uiRed1 = (unsigned int)m_frgbaColor1.IntRed(255.0f);
			unsigned int uiGreen1 = (unsigned int)m_frgbaColor1.IntGreen(255.0f);
			unsigned int uiBlue1 = (unsigned int)m_frgbaColor1.IntBlue(255.0f);
			unsigned int uiColorValue1 = (uiRed1 << 16) + (uiGreen1 << 8) + uiBlue1;

			unsigned int uiRed2 = (unsigned int)m_frgbaColor2.IntRed(255.0f);
			unsigned int uiGreen2 = (unsigned int)m_frgbaColor2.IntGreen(255.0f);
			unsigned int uiBlue2 = (unsigned int)m_frgbaColor2.IntBlue(255.0f);
			unsigned int uiColorValue2 = (uiRed2 << 16) + (uiGreen2 << 8) + uiBlue2;

			unsigned int uiCWLsb = m_uiCW1 & 1;

			if ((uiColorValue1 >= (uiColorValue2 & uiCWLsb)) == 0 ||
				(uiColorValue1 < (uiColorValue2 & uiCWLsb)) == 1)
			{
				return;
			}
		}

		float fDistance = s_afTHDistanceTable[m_uiCW1];

		unsigned int auiBestPixelSelectors[PIXELS];
		float afBestPixelErrors[PIXELS] = { FLT_MAX, FLT_MAX, FLT_MAX, FLT_MAX, FLT_MAX, FLT_MAX, FLT_MAX, FLT_MAX,
											FLT_MAX, FLT_MAX, FLT_MAX, FLT_MAX, FLT_MAX, FLT_MAX, FLT_MAX, FLT_MAX };
		ColorFloatRGBA	afrgbaBestDecodedPixels[PIXELS];
		ColorFloatRGBA afrgbaDecodedPixel[SELECTORS];

		assert(SELECTORS == 4);
		afrgbaDecodedPixel[0] = (m_frgbaColor1 + fDistance).ClampRGB();
		afrgbaDecodedPixel[1] = (m_frgbaColor1 - fDistance).ClampRGB();
		afrgbaDecodedPixel[2] = ColorFloatRGBA();;
		afrgbaDecodedPixel[3] = (m_frgbaColor2 - fDistance).ClampRGB();


		// try each selector
		for (unsigned int uiPixel = 0; uiPixel < PIXELS; uiPixel++)
		{
			unsigned int uiMinSelector = 0;
			unsigned int uiMaxSelector = SELECTORS - 1;

			if (m_pafrgbaSource[uiPixel].fA < 0.5f)
			{
				uiMinSelector = 2;
				uiMaxSelector = 2;
			}

			for (unsigned int uiSelector = uiMinSelector; uiSelector <= uiMaxSelector; uiSelector++)
			{
				float fPixelError = CalcPixelError(afrgbaDecodedPixel[uiSelector], m_afDecodedAlphas[uiPixel],
													m_pafrgbaSource[uiPixel]);

				if (fPixelError < afBestPixelErrors[uiPixel])
				{
					afBestPixelErrors[uiPixel] = fPixelError;
					auiBestPixelSelectors[uiPixel] = uiSelector;
					afrgbaBestDecodedPixels[uiPixel] = afrgbaDecodedPixel[uiSelector];
				}
			}
		}
		

		// add up all of the pixel errors
		float fBlockError = 0.0f;
		for (unsigned int uiPixel = 0; uiPixel < PIXELS; uiPixel++)
		{
			fBlockError += afBestPixelErrors[uiPixel];
		}

		if (fBlockError < m_fError)
		{
			m_fError = fBlockError;

			for (unsigned int uiPixel = 0; uiPixel < PIXELS; uiPixel++)
			{
				m_auiSelectors[uiPixel] = auiBestPixelSelectors[uiPixel];
				m_afrgbaDecodedColors[uiPixel] = afrgbaBestDecodedPixels[uiPixel];
			}
		}

	}

	// ----------------------------------------------------------------------------------------------------
	// try version 1 of the degenerate search
	// degenerate encodings use basecolor movement and a subset of the selectors to find useful encodings
	// each subsequent version of the degenerate search uses more basecolor movement and is less likely to
	//		be successfull
	//
	void Block4x4Encoding_RGB8A1::TryDegenerates1(void)
	{

		TryDifferential(m_boolMostLikelyFlip, 1, -2, 0);
		TryDifferential(m_boolMostLikelyFlip, 1, 2, 0);
		TryDifferential(m_boolMostLikelyFlip, 1, 0, 2);
		TryDifferential(m_boolMostLikelyFlip, 1, 0, -2);

	}

	// ----------------------------------------------------------------------------------------------------
	// try version 2 of the degenerate search
	// degenerate encodings use basecolor movement and a subset of the selectors to find useful encodings
	// each subsequent version of the degenerate search uses more basecolor movement and is less likely to
	//		be successfull
	//
	void Block4x4Encoding_RGB8A1::TryDegenerates2(void)
	{

		TryDifferential(!m_boolMostLikelyFlip, 1, -2, 0);
		TryDifferential(!m_boolMostLikelyFlip, 1, 2, 0);
		TryDifferential(!m_boolMostLikelyFlip, 1, 0, 2);
		TryDifferential(!m_boolMostLikelyFlip, 1, 0, -2);

	}

	// ----------------------------------------------------------------------------------------------------
	// try version 3 of the degenerate search
	// degenerate encodings use basecolor movement and a subset of the selectors to find useful encodings
	// each subsequent version of the degenerate search uses more basecolor movement and is less likely to
	//		be successfull
	//
	void Block4x4Encoding_RGB8A1::TryDegenerates3(void)
	{

		TryDifferential(m_boolMostLikelyFlip, 1, -2, -2);
		TryDifferential(m_boolMostLikelyFlip, 1, -2, 2);
		TryDifferential(m_boolMostLikelyFlip, 1, 2, -2);
		TryDifferential(m_boolMostLikelyFlip, 1, 2, 2);

	}

	// ----------------------------------------------------------------------------------------------------
	// try version 4 of the degenerate search
	// degenerate encodings use basecolor movement and a subset of the selectors to find useful encodings
	// each subsequent version of the degenerate search uses more basecolor movement and is less likely to
	//		be successfull
	//
	void Block4x4Encoding_RGB8A1::TryDegenerates4(void)
	{

		TryDifferential(m_boolMostLikelyFlip, 1, -4, 0);
		TryDifferential(m_boolMostLikelyFlip, 1, 4, 0);
		TryDifferential(m_boolMostLikelyFlip, 1, 0, 4);
		TryDifferential(m_boolMostLikelyFlip, 1, 0, -4);

	}

	// ----------------------------------------------------------------------------------------------------
	// set the encoding bits based on encoding state
	//
	void Block4x4Encoding_RGB8A1::SetEncodingBits(void)
	{
		switch (m_mode)
		{
		case MODE_ETC1:
			SetEncodingBits_ETC1();
			break;

		case MODE_T:
			SetEncodingBits_T();
			break;

		case MODE_H:
			SetEncodingBits_H();
			break;

		case MODE_PLANAR:
			Block4x4Encoding_RGB8::SetEncodingBits_Planar();
			break;

		default:
			assert(false);
		}
	}

	// ----------------------------------------------------------------------------------------------------
	// set the encoding bits based on encoding state if ETC1 mode
	//
	void Block4x4Encoding_RGB8A1::SetEncodingBits_ETC1(void)
	{

		// there is no individual mode in RGB8A1
		assert(m_boolDiff);

		int iRed1 = m_frgbaColor1.IntRed(31.0f);
		int iGreen1 = m_frgbaColor1.IntGreen(31.0f);
		int iBlue1 = m_frgbaColor1.IntBlue(31.0f);

		int iRed2 = m_frgbaColor2.IntRed(31.0f);
		int iGreen2 = m_frgbaColor2.IntGreen(31.0f);
		int iBlue2 = m_frgbaColor2.IntBlue(31.0f);

		int iDRed2 = iRed2 - iRed1;
		int iDGreen2 = iGreen2 - iGreen1;
		int iDBlue2 = iBlue2 - iBlue1;

		assert(iDRed2 >= -4 && iDRed2 < 4);
		assert(iDGreen2 >= -4 && iDGreen2 < 4);
		assert(iDBlue2 >= -4 && iDBlue2 < 4);

		m_pencodingbitsRGB8->differential.red1 = iRed1;
		m_pencodingbitsRGB8->differential.green1 = iGreen1;
		m_pencodingbitsRGB8->differential.blue1 = iBlue1;

		m_pencodingbitsRGB8->differential.dred2 = iDRed2;
		m_pencodingbitsRGB8->differential.dgreen2 = iDGreen2;
		m_pencodingbitsRGB8->differential.dblue2 = iDBlue2;

		m_pencodingbitsRGB8->individual.cw1 = m_uiCW1;
		m_pencodingbitsRGB8->individual.cw2 = m_uiCW2;

		SetEncodingBits_Selectors();

		// in RGB8A1 encoding bits, opaque replaces differential
		m_pencodingbitsRGB8->differential.diff = !m_boolPunchThroughPixels;

		m_pencodingbitsRGB8->individual.flip = m_boolFlip;

	}

	// ----------------------------------------------------------------------------------------------------
	// set the encoding bits based on encoding state if T mode
	//
	void Block4x4Encoding_RGB8A1::SetEncodingBits_T(void)
	{
		static const bool SANITY_CHECK = true;

		assert(m_mode == MODE_T);
		assert(m_boolDiff == true);

		unsigned int uiRed1 = (unsigned int)m_frgbaColor1.IntRed(15.0f);
		unsigned int uiGreen1 = (unsigned int)m_frgbaColor1.IntGreen(15.0f);
		unsigned int uiBlue1 = (unsigned int)m_frgbaColor1.IntBlue(15.0f);

		unsigned int uiRed2 = (unsigned int)m_frgbaColor2.IntRed(15.0f);
		unsigned int uiGreen2 = (unsigned int)m_frgbaColor2.IntGreen(15.0f);
		unsigned int uiBlue2 = (unsigned int)m_frgbaColor2.IntBlue(15.0f);

		m_pencodingbitsRGB8->t.red1a = uiRed1 >> 2;
		m_pencodingbitsRGB8->t.red1b = uiRed1;
		m_pencodingbitsRGB8->t.green1 = uiGreen1;
		m_pencodingbitsRGB8->t.blue1 = uiBlue1;

		m_pencodingbitsRGB8->t.red2 = uiRed2;
		m_pencodingbitsRGB8->t.green2 = uiGreen2;
		m_pencodingbitsRGB8->t.blue2 = uiBlue2;

		m_pencodingbitsRGB8->t.da = m_uiCW1 >> 1;
		m_pencodingbitsRGB8->t.db = m_uiCW1;

		// in RGB8A1 encoding bits, opaque replaces differential
		m_pencodingbitsRGB8->differential.diff = !m_boolPunchThroughPixels;

		Block4x4Encoding_ETC1::SetEncodingBits_Selectors();

		// create an invalid R differential to trigger T mode
		m_pencodingbitsRGB8->t.detect1 = 0;
		m_pencodingbitsRGB8->t.detect2 = 0;
		int iRed2 = (int)m_pencodingbitsRGB8->differential.red1 + (int)m_pencodingbitsRGB8->differential.dred2;
		if (iRed2 >= 4)
		{
			m_pencodingbitsRGB8->t.detect1 = 7;
			m_pencodingbitsRGB8->t.detect2 = 0;
		}
		else
		{
			m_pencodingbitsRGB8->t.detect1 = 0;
			m_pencodingbitsRGB8->t.detect2 = 1;
		}

		if (SANITY_CHECK)
		{
			iRed2 = (int)m_pencodingbitsRGB8->differential.red1 + (int)m_pencodingbitsRGB8->differential.dred2;

			// make sure red overflows
			assert(iRed2 < 0 || iRed2 > 31);
		}

	}

	// ----------------------------------------------------------------------------------------------------
	// set the encoding bits based on encoding state if H mode
	//
	// colors and selectors may need to swap in order to generate lsb of distance index
	//
	void Block4x4Encoding_RGB8A1::SetEncodingBits_H(void)
	{
		static const bool SANITY_CHECK = true;

		assert(m_mode == MODE_H);
		assert(m_boolDiff == true);

		unsigned int uiRed1 = (unsigned int)m_frgbaColor1.IntRed(15.0f);
		unsigned int uiGreen1 = (unsigned int)m_frgbaColor1.IntGreen(15.0f);
		unsigned int uiBlue1 = (unsigned int)m_frgbaColor1.IntBlue(15.0f);

		unsigned int uiRed2 = (unsigned int)m_frgbaColor2.IntRed(15.0f);
		unsigned int uiGreen2 = (unsigned int)m_frgbaColor2.IntGreen(15.0f);
		unsigned int uiBlue2 = (unsigned int)m_frgbaColor2.IntBlue(15.0f);

		unsigned int uiColor1 = (uiRed1 << 16) + (uiGreen1 << 8) + uiBlue1;
		unsigned int uiColor2 = (uiRed2 << 16) + (uiGreen2 << 8) + uiBlue2;

		bool boolOddDistance = m_uiCW1 & 1;
		bool boolSwapColors = (uiColor1 < uiColor2) ^ !boolOddDistance;

		if (boolSwapColors)
		{
			m_pencodingbitsRGB8->h.red1 = uiRed2;
			m_pencodingbitsRGB8->h.green1a = uiGreen2 >> 1;
			m_pencodingbitsRGB8->h.green1b = uiGreen2;
			m_pencodingbitsRGB8->h.blue1a = uiBlue2 >> 3;
			m_pencodingbitsRGB8->h.blue1b = uiBlue2 >> 1;
			m_pencodingbitsRGB8->h.blue1c = uiBlue2;

			m_pencodingbitsRGB8->h.red2 = uiRed1;
			m_pencodingbitsRGB8->h.green2a = uiGreen1 >> 1;
			m_pencodingbitsRGB8->h.green2b = uiGreen1;
			m_pencodingbitsRGB8->h.blue2 = uiBlue1;

			m_pencodingbitsRGB8->h.da = m_uiCW1 >> 2;
			m_pencodingbitsRGB8->h.db = m_uiCW1 >> 1;
		}
		else
		{
			m_pencodingbitsRGB8->h.red1 = uiRed1;
			m_pencodingbitsRGB8->h.green1a = uiGreen1 >> 1;
			m_pencodingbitsRGB8->h.green1b = uiGreen1;
			m_pencodingbitsRGB8->h.blue1a = uiBlue1 >> 3;
			m_pencodingbitsRGB8->h.blue1b = uiBlue1 >> 1;
			m_pencodingbitsRGB8->h.blue1c = uiBlue1;

			m_pencodingbitsRGB8->h.red2 = uiRed2;
			m_pencodingbitsRGB8->h.green2a = uiGreen2 >> 1;
			m_pencodingbitsRGB8->h.green2b = uiGreen2;
			m_pencodingbitsRGB8->h.blue2 = uiBlue2;

			m_pencodingbitsRGB8->h.da = m_uiCW1 >> 2;
			m_pencodingbitsRGB8->h.db = m_uiCW1 >> 1;
		}

		// in RGB8A1 encoding bits, opaque replaces differential
		m_pencodingbitsRGB8->differential.diff = !m_boolPunchThroughPixels;

		Block4x4Encoding_ETC1::SetEncodingBits_Selectors();

		if (boolSwapColors)
		{
			m_pencodingbitsRGB8->h.selectors ^= 0x0000FFFF;
		}

		// create an invalid R differential to trigger T mode
		m_pencodingbitsRGB8->h.detect1 = 0;
		m_pencodingbitsRGB8->h.detect2 = 0;
		m_pencodingbitsRGB8->h.detect3 = 0;
		int iRed2 = (int)m_pencodingbitsRGB8->differential.red1 + (int)m_pencodingbitsRGB8->differential.dred2;
		int iGreen2 = (int)m_pencodingbitsRGB8->differential.green1 + (int)m_pencodingbitsRGB8->differential.dgreen2;
		if (iRed2 < 0 || iRed2 > 31)
		{
			m_pencodingbitsRGB8->h.detect1 = 1;
		}
		if (iGreen2 >= 4)
		{
			m_pencodingbitsRGB8->h.detect2 = 7;
			m_pencodingbitsRGB8->h.detect3 = 0;
		}
		else
		{
			m_pencodingbitsRGB8->h.detect2 = 0;
			m_pencodingbitsRGB8->h.detect3 = 1;
		}

		if (SANITY_CHECK)
		{
			iRed2 = (int)m_pencodingbitsRGB8->differential.red1 + (int)m_pencodingbitsRGB8->differential.dred2;
			iGreen2 = (int)m_pencodingbitsRGB8->differential.green1 + (int)m_pencodingbitsRGB8->differential.dgreen2;

			// make sure red doesn't overflow and green does
			assert(iRed2 >= 0 && iRed2 <= 31);
			assert(iGreen2 < 0 || iGreen2 > 31);
		}

	}

	// ####################################################################################################
	// Block4x4Encoding_RGB8A1_Opaque
	// ####################################################################################################

	// ----------------------------------------------------------------------------------------------------
	// perform a single encoding iteration
	// replace the encoding if a better encoding was found
	// subsequent iterations generally take longer for each iteration
	// set m_boolDone if encoding is perfect or encoding is finished based on a_fEffort
	//
	void Block4x4Encoding_RGB8A1_Opaque::PerformIteration(float a_fEffort)
	{
		assert(!m_boolPunchThroughPixels);
		assert(!m_boolTransparent);
		assert(!m_boolDone);

		switch (m_uiEncodingIterations)
		{
		case 0:
			PerformFirstIteration();
			break;

		case 1:
			Block4x4Encoding_ETC1::TryDifferential(m_boolMostLikelyFlip, 1, 0, 0);
			break;

		case 2:
			Block4x4Encoding_ETC1::TryDifferential(!m_boolMostLikelyFlip, 1, 0, 0);
			break;

		case 3:
			Block4x4Encoding_RGB8::TryPlanar(1);
			break;

		case 4:
			Block4x4Encoding_RGB8::TryTAndH(1);
			if (a_fEffort <= 49.5f)
			{
				m_boolDone = true;
			}
			break;

		case 5:
			Block4x4Encoding_ETC1::TryDegenerates1();
			if (a_fEffort <= 59.5f)
			{
				m_boolDone = true;
			}
			break;

		case 6:
			Block4x4Encoding_ETC1::TryDegenerates2();
			if (a_fEffort <= 69.5f)
			{
				m_boolDone = true;
			}
			break;

		case 7:
			Block4x4Encoding_ETC1::TryDegenerates3();
			if (a_fEffort <= 79.5f)
			{
				m_boolDone = true;
			}
			break;

		case 8:
			Block4x4Encoding_ETC1::TryDegenerates4();
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
	// find best initial encoding to ensure block has a valid encoding
	//
	void Block4x4Encoding_RGB8A1_Opaque::PerformFirstIteration(void)
	{
		
		// set decoded alphas
		// calculate alpha error
		m_fError = 0.0f;
		for (unsigned int uiPixel = 0; uiPixel < PIXELS; uiPixel++)
		{
			m_afDecodedAlphas[uiPixel] = 1.0f;

			float fDeltaA = 1.0f - m_pafrgbaSource[uiPixel].fA;
			m_fError += fDeltaA * fDeltaA;
		}

		CalculateMostLikelyFlip();

		m_fError = FLT_MAX;

		Block4x4Encoding_ETC1::TryDifferential(m_boolMostLikelyFlip, 0, 0, 0);
		SetDoneIfPerfect();
		if (m_boolDone)
		{
			return;
		}
		Block4x4Encoding_ETC1::TryDifferential(!m_boolMostLikelyFlip, 0, 0, 0);
		SetDoneIfPerfect();
		if (m_boolDone)
		{
			return;
		}
		Block4x4Encoding_RGB8::TryPlanar(0);
		SetDoneIfPerfect();
		if (m_boolDone)
		{
			return;
		}
		Block4x4Encoding_RGB8::TryTAndH(0);
		SetDoneIfPerfect();
	}

	// ####################################################################################################
	// Block4x4Encoding_RGB8A1_Transparent
	// ####################################################################################################

	// ----------------------------------------------------------------------------------------------------
	// perform a single encoding iteration
	// replace the encoding if a better encoding was found
	// subsequent iterations generally take longer for each iteration
	// set m_boolDone if encoding is perfect or encoding is finished based on a_fEffort
	//
	void Block4x4Encoding_RGB8A1_Transparent::PerformIteration(float )
	{
		assert(!m_boolOpaque);
		assert(m_boolTransparent);
		assert(!m_boolDone);
		assert(m_uiEncodingIterations == 0);

		m_mode = MODE_ETC1;
		m_boolDiff = true;
		m_boolFlip = false;

		m_uiCW1 = 0;
		m_uiCW2 = 0;

		m_frgbaColor1 = ColorFloatRGBA();
		m_frgbaColor2 = ColorFloatRGBA();

		for (unsigned int uiPixel = 0; uiPixel < PIXELS; uiPixel++)
		{
			m_auiSelectors[uiPixel] = TRANSPARENT_SELECTOR;

			m_afrgbaDecodedColors[uiPixel] = ColorFloatRGBA();
			m_afDecodedAlphas[uiPixel] = 0.0f;
		}

		CalcBlockError();

		m_boolDone = true;
		m_uiEncodingIterations++;

	}

	// ----------------------------------------------------------------------------------------------------
	//
}
