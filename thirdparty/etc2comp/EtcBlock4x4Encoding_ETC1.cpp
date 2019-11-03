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
EtcBlock4x4Encoding_ETC1.cpp

Block4x4Encoding_ETC1 is the encoder to use when targetting file format ETC1.  This encoder is also
used for the ETC1 subset of file format RGB8, RGBA8 and RGB8A1

*/

#include "EtcConfig.h"
#include "EtcBlock4x4Encoding_ETC1.h"

#include "EtcBlock4x4.h"
#include "EtcBlock4x4EncodingBits.h"
#include "EtcDifferentialTrys.h"

#include <stdio.h>
#include <string.h>
#include <assert.h>
#include <float.h>
#include <limits>

namespace Etc
{

	// pixel processing order if the flip bit = 0 (horizontal split)
	const unsigned int Block4x4Encoding_ETC1::s_auiPixelOrderFlip0[PIXELS] = { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15 };

	// pixel processing order if the flip bit = 1 (vertical split)
	const unsigned int Block4x4Encoding_ETC1::s_auiPixelOrderFlip1[PIXELS] = { 0, 1, 4, 5, 8, 9, 12, 13, 2, 3, 6, 7, 10, 11, 14, 15 };

	// pixel processing order for horizontal scan (ETC normally does a vertical scan)
	const unsigned int Block4x4Encoding_ETC1::s_auiPixelOrderHScan[PIXELS] = { 0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15 };

	// pixel indices for different block halves
	const unsigned int Block4x4Encoding_ETC1::s_auiLeftPixelMapping[8] = { 0, 1, 2, 3, 4, 5, 6, 7 };
	const unsigned int Block4x4Encoding_ETC1::s_auiRightPixelMapping[8] = { 8, 9, 10, 11, 12, 13, 14, 15 };
	const unsigned int Block4x4Encoding_ETC1::s_auiTopPixelMapping[8] = { 0, 1, 4, 5, 8, 9, 12, 13 };
	const unsigned int Block4x4Encoding_ETC1::s_auiBottomPixelMapping[8] = { 2, 3, 6, 7, 10, 11, 14, 15 };

	// CW ranges that the ETC1 decoders use
	// CW is basically a contrast for the different selector bits, since these values are offsets to the base color
	// the first axis in the array is indexed by the CW in the encoding bits
	// the second axis in the array is indexed by the selector bits
	float Block4x4Encoding_ETC1::s_aafCwTable[CW_RANGES][SELECTORS] =
	{
		{ 2.0f / 255.0f, 8.0f / 255.0f, -2.0f / 255.0f, -8.0f / 255.0f },
		{ 5.0f / 255.0f, 17.0f / 255.0f, -5.0f / 255.0f, -17.0f / 255.0f },
		{ 9.0f / 255.0f, 29.0f / 255.0f, -9.0f / 255.0f, -29.0f / 255.0f },
		{ 13.0f / 255.0f, 42.0f / 255.0f, -13.0f / 255.0f, -42.0f / 255.0f },
		{ 18.0f / 255.0f, 60.0f / 255.0f, -18.0f / 255.0f, -60.0f / 255.0f },
		{ 24.0f / 255.0f, 80.0f / 255.0f, -24.0f / 255.0f, -80.0f / 255.0f },
		{ 33.0f / 255.0f, 106.0f / 255.0f, -33.0f / 255.0f, -106.0f / 255.0f },
		{ 47.0f / 255.0f, 183.0f / 255.0f, -47.0f / 255.0f, -183.0f / 255.0f }
	};

	// ----------------------------------------------------------------------------------------------------
	//
	Block4x4Encoding_ETC1::Block4x4Encoding_ETC1(void)
	{
		m_mode = MODE_ETC1;
		m_boolDiff = false;
		m_boolFlip = false;
		m_frgbaColor1 = ColorFloatRGBA();
		m_frgbaColor2 = ColorFloatRGBA();
		m_uiCW1 = 0;
		m_uiCW2 = 0;
		for (unsigned int uiPixel = 0; uiPixel < PIXELS; uiPixel++)
		{
			m_auiSelectors[uiPixel] = 0;
			m_afDecodedAlphas[uiPixel] = 1.0f;
		}

		m_boolMostLikelyFlip = false;

		m_fError = -1.0f;

		m_fError1 = -1.0f;
		m_fError2 = -1.0f;
		m_boolSeverelyBentDifferentialColors = false;

		for (unsigned int uiPixel = 0; uiPixel < PIXELS; uiPixel++)
		{
			m_afDecodedAlphas[uiPixel] = 1.0f;
		}

	}

	 Block4x4Encoding_ETC1::~Block4x4Encoding_ETC1(void) {}

	// ----------------------------------------------------------------------------------------------------
	// initialization prior to encoding
	// a_pblockParent points to the block associated with this encoding
	// a_errormetric is used to choose the best encoding
	// a_pafrgbaSource points to a 4x4 block subset of the source image
	// a_paucEncodingBits points to the final encoding bits
	//
	void Block4x4Encoding_ETC1::InitFromSource(Block4x4 *a_pblockParent,
												ColorFloatRGBA *a_pafrgbaSource,
												unsigned char *a_paucEncodingBits, ErrorMetric a_errormetric)
	{

		Block4x4Encoding::Init(a_pblockParent, a_pafrgbaSource,a_errormetric);

		for (unsigned int uiPixel = 0; uiPixel < PIXELS; uiPixel++)
		{
			m_afDecodedAlphas[uiPixel] = 1.0f;
		}

		m_fError = -1.0f;

		m_pencodingbitsRGB8 = (Block4x4EncodingBits_RGB8 *)(a_paucEncodingBits);

	}

	// ----------------------------------------------------------------------------------------------------
	// initialization from the encoding bits of a previous encoding
	// a_pblockParent points to the block associated with this encoding
	// a_errormetric is used to choose the best encoding
	// a_pafrgbaSource points to a 4x4 block subset of the source image
	// a_paucEncodingBits points to the final encoding bits of a previous encoding
	//
	void Block4x4Encoding_ETC1::InitFromEncodingBits(Block4x4 *a_pblockParent,
														unsigned char *a_paucEncodingBits,
														ColorFloatRGBA *a_pafrgbaSource, 
														ErrorMetric a_errormetric)
	{

		Block4x4Encoding::Init(a_pblockParent, a_pafrgbaSource,a_errormetric);
		m_fError = -1.0f;

		m_pencodingbitsRGB8 = (Block4x4EncodingBits_RGB8 *)a_paucEncodingBits;

		m_mode = MODE_ETC1;
		m_boolDiff = m_pencodingbitsRGB8->individual.diff;
		m_boolFlip = m_pencodingbitsRGB8->individual.flip;
		if (m_boolDiff)
		{
			int iR2 = (int)(m_pencodingbitsRGB8->differential.red1 + m_pencodingbitsRGB8->differential.dred2);
			if (iR2 < 0)
			{
				iR2 = 0;
			}
			else if (iR2 > 31)
			{
				iR2 = 31;
			}

			int iG2 = (int)(m_pencodingbitsRGB8->differential.green1 + m_pencodingbitsRGB8->differential.dgreen2);
			if (iG2 < 0)
			{
				iG2 = 0;
			}
			else if (iG2 > 31)
			{
				iG2 = 31;
			}

			int iB2 = (int)(m_pencodingbitsRGB8->differential.blue1 + m_pencodingbitsRGB8->differential.dblue2);
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

		}
		else
		{
			m_frgbaColor1 = ColorFloatRGBA::ConvertFromRGB4(m_pencodingbitsRGB8->individual.red1, m_pencodingbitsRGB8->individual.green1, m_pencodingbitsRGB8->individual.blue1);
			m_frgbaColor2 = ColorFloatRGBA::ConvertFromRGB4(m_pencodingbitsRGB8->individual.red2, m_pencodingbitsRGB8->individual.green2, m_pencodingbitsRGB8->individual.blue2);
		}

		m_uiCW1 = m_pencodingbitsRGB8->individual.cw1;
		m_uiCW2 = m_pencodingbitsRGB8->individual.cw2;

		InitFromEncodingBits_Selectors();

		Decode();

		CalcBlockError();
	}

	// ----------------------------------------------------------------------------------------------------
	// init the selectors from a prior encoding
	//
	void Block4x4Encoding_ETC1::InitFromEncodingBits_Selectors(void)
	{

		unsigned char *paucSelectors = (unsigned char *)&m_pencodingbitsRGB8->individual.selectors;

		for (unsigned int iPixel = 0; iPixel < PIXELS; iPixel++)
		{
			unsigned int uiByteMSB = (unsigned int)(1 - (iPixel / 8));
			unsigned int uiByteLSB = (unsigned int)(3 - (iPixel / 8));
			unsigned int uiShift = (unsigned int)(iPixel & 7);

			unsigned int uiSelectorMSB = (unsigned int)((paucSelectors[uiByteMSB] >> uiShift) & 1);
			unsigned int uiSelectorLSB = (unsigned int)((paucSelectors[uiByteLSB] >> uiShift) & 1);

			m_auiSelectors[iPixel] = (uiSelectorMSB << 1) + uiSelectorLSB;
		}

	}

	// ----------------------------------------------------------------------------------------------------
	// perform a single encoding iteration
	// replace the encoding if a better encoding was found
	// subsequent iterations generally take longer for each iteration
	// set m_boolDone if encoding is perfect or encoding is finished based on a_fEffort
	//
	void Block4x4Encoding_ETC1::PerformIteration(float a_fEffort)
	{
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
			TryIndividual(m_boolMostLikelyFlip, 1);
			if (a_fEffort <= 49.5f)
			{
				m_boolDone = true;
			}
			break;

		case 3:
			TryDifferential(!m_boolMostLikelyFlip, 1, 0, 0);
			if (a_fEffort <= 59.5f)
			{
				m_boolDone = true;
			}
			break;

		case 4:
			TryIndividual(!m_boolMostLikelyFlip, 1);
			if (a_fEffort <= 69.5f)
			{
				m_boolDone = true;
			}
			break;

		case 5:
			TryDegenerates1();
			if (a_fEffort <= 79.5f)
			{
				m_boolDone = true;
			}
			break;

		case 6:
			TryDegenerates2();
			if (a_fEffort <= 89.5f)
			{
				m_boolDone = true;
			}
			break;

		case 7:
			TryDegenerates3();
			if (a_fEffort <= 99.5f)
			{
				m_boolDone = true;
			}
			break;

		case 8:
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
	void Block4x4Encoding_ETC1::PerformFirstIteration(void)
	{
		CalculateMostLikelyFlip();

		m_fError = FLT_MAX;

		TryDifferential(m_boolMostLikelyFlip, 0, 0, 0);
		SetDoneIfPerfect();
		if (m_boolDone)
		{
			return;
		}

		TryIndividual(m_boolMostLikelyFlip, 0);
		SetDoneIfPerfect();
		if (m_boolDone)
		{
			return;
		}
		TryDifferential(!m_boolMostLikelyFlip, 0, 0, 0);
		SetDoneIfPerfect();
		if (m_boolDone)
		{
			return;
		}
		TryIndividual(!m_boolMostLikelyFlip, 0);

	}

	// ----------------------------------------------------------------------------------------------------
	// algorithm:
	// create a source average color for the Left, Right, Top and Bottom halves using the 8 pixels in each half
	// note: the "gray line" is the line of equal delta RGB that goes thru the average color
	// for each half:
	//		see how close each of the 8 pixels are to the "gray line" that goes thru the source average color
	//		create an error value that is the sum of the distances from the gray line
	// h_error is the sum of Left and Right errors
	// v_error is the sum of Top and Bottom errors
	//
	void Block4x4Encoding_ETC1::CalculateMostLikelyFlip(void)
	{
		static const bool DEBUG_PRINT = false;

		CalculateSourceAverages();

		float fLeftGrayErrorSum = 0.0f;
		float fRightGrayErrorSum = 0.0f;
		float fTopGrayErrorSum = 0.0f;
		float fBottomGrayErrorSum = 0.0f;

		for (unsigned int uiPixel = 0; uiPixel < 8; uiPixel++)
		{
			ColorFloatRGBA *pfrgbaLeft = &m_pafrgbaSource[uiPixel];
			ColorFloatRGBA *pfrgbaRight = &m_pafrgbaSource[uiPixel + 8];
			ColorFloatRGBA *pfrgbaTop = &m_pafrgbaSource[s_auiTopPixelMapping[uiPixel]];
			ColorFloatRGBA *pfrgbaBottom = &m_pafrgbaSource[s_auiBottomPixelMapping[uiPixel]];

			float fLeftGrayError = CalcGrayDistance2(*pfrgbaLeft, m_frgbaSourceAverageLeft);
			float fRightGrayError = CalcGrayDistance2(*pfrgbaRight, m_frgbaSourceAverageRight);
			float fTopGrayError = CalcGrayDistance2(*pfrgbaTop, m_frgbaSourceAverageTop);
			float fBottomGrayError = CalcGrayDistance2(*pfrgbaBottom, m_frgbaSourceAverageBottom);

			fLeftGrayErrorSum += fLeftGrayError;
			fRightGrayErrorSum += fRightGrayError;
			fTopGrayErrorSum += fTopGrayError;
			fBottomGrayErrorSum += fBottomGrayError;
		}

		if (DEBUG_PRINT)
		{
			printf("\n%.2f %.2f\n", fLeftGrayErrorSum + fRightGrayErrorSum, fTopGrayErrorSum + fBottomGrayErrorSum);
		}

		m_boolMostLikelyFlip = (fTopGrayErrorSum + fBottomGrayErrorSum) < (fLeftGrayErrorSum + fRightGrayErrorSum);

	}

	// ----------------------------------------------------------------------------------------------------
	// calculate source pixel averages for each 2x2 quadrant in a 4x4 block
	// these are used to determine the averages for each of the 4 different halves (left, right, top, bottom)
	// ignore pixels that have alpha == NAN (these are border pixels outside of the source image)
	// weight the averages based on a pixel's alpha
	//
	void Block4x4Encoding_ETC1::CalculateSourceAverages(void)
	{
		static const bool DEBUG_PRINT = false;

		bool boolRGBX = m_pblockParent->GetImageSource()->GetErrorMetric() == ErrorMetric::RGBX;

		if (m_pblockParent->GetSourceAlphaMix() == Block4x4::SourceAlphaMix::OPAQUE || boolRGBX)
		{
			ColorFloatRGBA frgbaSumUL = m_pafrgbaSource[0] + m_pafrgbaSource[1] + m_pafrgbaSource[4] + m_pafrgbaSource[5];
			ColorFloatRGBA frgbaSumLL = m_pafrgbaSource[2] + m_pafrgbaSource[3] + m_pafrgbaSource[6] + m_pafrgbaSource[7];
			ColorFloatRGBA frgbaSumUR = m_pafrgbaSource[8] + m_pafrgbaSource[9] + m_pafrgbaSource[12] + m_pafrgbaSource[13];
			ColorFloatRGBA frgbaSumLR = m_pafrgbaSource[10] + m_pafrgbaSource[11] + m_pafrgbaSource[14] + m_pafrgbaSource[15];

			m_frgbaSourceAverageLeft = (frgbaSumUL + frgbaSumLL) * 0.125f;
			m_frgbaSourceAverageRight = (frgbaSumUR + frgbaSumLR) * 0.125f;
			m_frgbaSourceAverageTop = (frgbaSumUL + frgbaSumUR) * 0.125f;
			m_frgbaSourceAverageBottom = (frgbaSumLL + frgbaSumLR) * 0.125f;
		}
		else
		{
			float afSourceAlpha[PIXELS];

			// treat alpha NAN as 0.0f
			for (unsigned int uiPixel = 0; uiPixel < PIXELS; uiPixel++)
			{
				afSourceAlpha[uiPixel] = isnan(m_pafrgbaSource[uiPixel].fA) ? 
																		0.0f : 
																		m_pafrgbaSource[uiPixel].fA;
			}

			ColorFloatRGBA afrgbaAlphaWeightedSource[PIXELS];
			for (unsigned int uiPixel = 0; uiPixel < PIXELS; uiPixel++)
			{
				afrgbaAlphaWeightedSource[uiPixel] = m_pafrgbaSource[uiPixel] * afSourceAlpha[uiPixel];
			}

			ColorFloatRGBA frgbaSumUL = afrgbaAlphaWeightedSource[0] +
										afrgbaAlphaWeightedSource[1] +
										afrgbaAlphaWeightedSource[4] +
										afrgbaAlphaWeightedSource[5];

			ColorFloatRGBA frgbaSumLL = afrgbaAlphaWeightedSource[2] +
										afrgbaAlphaWeightedSource[3] +
										afrgbaAlphaWeightedSource[6] +
										afrgbaAlphaWeightedSource[7];

			ColorFloatRGBA frgbaSumUR = afrgbaAlphaWeightedSource[8] +
										afrgbaAlphaWeightedSource[9] +
										afrgbaAlphaWeightedSource[12] +
										afrgbaAlphaWeightedSource[13];

			ColorFloatRGBA frgbaSumLR = afrgbaAlphaWeightedSource[10] +
										afrgbaAlphaWeightedSource[11] +
										afrgbaAlphaWeightedSource[14] +
										afrgbaAlphaWeightedSource[15];

			float fWeightSumUL = afSourceAlpha[0] +
									afSourceAlpha[1] +
									afSourceAlpha[4] +
									afSourceAlpha[5];

			float fWeightSumLL = afSourceAlpha[2] +
									afSourceAlpha[3] +
									afSourceAlpha[6] +
									afSourceAlpha[7];

			float fWeightSumUR = afSourceAlpha[8] +
									afSourceAlpha[9] +
									afSourceAlpha[12] +
									afSourceAlpha[13];

			float fWeightSumLR = afSourceAlpha[10] +
									afSourceAlpha[11] +
									afSourceAlpha[14] +
									afSourceAlpha[15];

			ColorFloatRGBA frgbaSumLeft = frgbaSumUL + frgbaSumLL;
			ColorFloatRGBA frgbaSumRight = frgbaSumUR + frgbaSumLR;
			ColorFloatRGBA frgbaSumTop = frgbaSumUL + frgbaSumUR;
			ColorFloatRGBA frgbaSumBottom = frgbaSumLL + frgbaSumLR;

			float fWeightSumLeft = fWeightSumUL + fWeightSumLL;
			float fWeightSumRight = fWeightSumUR + fWeightSumLR;
			float fWeightSumTop = fWeightSumUL + fWeightSumUR;
			float fWeightSumBottom = fWeightSumLL + fWeightSumLR;

			// check to see if there is at least 1 pixel with  non-zero alpha
			// completely transparent block should not make it to this code
			assert((fWeightSumLeft + fWeightSumRight) > 0.0f);
			assert((fWeightSumTop + fWeightSumBottom) > 0.0f);

			if (fWeightSumLeft > 0.0f)
			{
				m_frgbaSourceAverageLeft = frgbaSumLeft * (1.0f/fWeightSumLeft);
			}
			if (fWeightSumRight > 0.0f)
			{
				m_frgbaSourceAverageRight = frgbaSumRight * (1.0f/fWeightSumRight);
			}
			if (fWeightSumTop > 0.0f)
			{
				m_frgbaSourceAverageTop = frgbaSumTop * (1.0f/fWeightSumTop);
			}
			if (fWeightSumBottom > 0.0f)
			{
				m_frgbaSourceAverageBottom = frgbaSumBottom * (1.0f/fWeightSumBottom);
			}

			if (fWeightSumLeft == 0.0f)
			{
				assert(fWeightSumRight > 0.0f);
				m_frgbaSourceAverageLeft = m_frgbaSourceAverageRight;
			}
			if (fWeightSumRight == 0.0f)
			{
				assert(fWeightSumLeft > 0.0f);
				m_frgbaSourceAverageRight = m_frgbaSourceAverageLeft;
			}
			if (fWeightSumTop == 0.0f)
			{
				assert(fWeightSumBottom > 0.0f);
				m_frgbaSourceAverageTop = m_frgbaSourceAverageBottom;
			}
			if (fWeightSumBottom == 0.0f)
			{
				assert(fWeightSumTop > 0.0f);
				m_frgbaSourceAverageBottom = m_frgbaSourceAverageTop;
			}
		}

		

		if (DEBUG_PRINT)
		{
			printf("\ntarget: [%.2f,%.2f,%.2f] [%.2f,%.2f,%.2f] [%.2f,%.2f,%.2f] [%.2f,%.2f,%.2f]\n",
				m_frgbaSourceAverageLeft.fR, m_frgbaSourceAverageLeft.fG, m_frgbaSourceAverageLeft.fB,
				m_frgbaSourceAverageRight.fR, m_frgbaSourceAverageRight.fG, m_frgbaSourceAverageRight.fB,
				m_frgbaSourceAverageTop.fR, m_frgbaSourceAverageTop.fG, m_frgbaSourceAverageTop.fB,
				m_frgbaSourceAverageBottom.fR, m_frgbaSourceAverageBottom.fG, m_frgbaSourceAverageBottom.fB);
		}

	}

	// ----------------------------------------------------------------------------------------------------
	// try an ETC1 differential mode encoding
	// use a_boolFlip to set the encoding F bit
	// use a_uiRadius to alter basecolor components in the range[-a_uiRadius:a_uiRadius]
	// use a_iGrayOffset1 and a_iGrayOffset2 to offset the basecolor to search for degenerate encodings
	// replace the encoding if the encoding error is less than previous encoding
	//
	void Block4x4Encoding_ETC1::TryDifferential(bool a_boolFlip, unsigned int a_uiRadius,
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

		Block4x4Encoding_ETC1 encodingTry = *this;
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

			for (unsigned int uiPixelOrder = 0; uiPixelOrder < PIXELS / 2; uiPixelOrder++)
			{
				unsigned int uiPixel1 = pauiPixelMapping1[uiPixelOrder];
				unsigned int uiPixel2 = pauiPixelMapping2[uiPixelOrder];

				unsigned int uiSelector1 = ptryBest1->m_auiSelectors[uiPixelOrder];
				unsigned int uiSelector2 = ptryBest2->m_auiSelectors[uiPixelOrder];

				m_auiSelectors[uiPixel1] = uiSelector1;
				m_auiSelectors[uiPixel2] = ptryBest2->m_auiSelectors[uiPixelOrder];

				float fDeltaRGB1 = s_aafCwTable[m_uiCW1][uiSelector1];
				float fDeltaRGB2 = s_aafCwTable[m_uiCW2][uiSelector2];

				m_afrgbaDecodedColors[uiPixel1] = (m_frgbaColor1 + fDeltaRGB1).ClampRGB();
				m_afrgbaDecodedColors[uiPixel2] = (m_frgbaColor2 + fDeltaRGB2).ClampRGB();
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
	// try an ETC1 differential mode encoding for a half of a 4x4 block
	// vary the basecolor components using a radius
	//
	void Block4x4Encoding_ETC1::TryDifferentialHalf(DifferentialTrys::Half *a_phalf)
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
						ColorFloatRGBA	afrgbaDecodedPixels[PIXELS / 2];
						float afPixelErrors[PIXELS / 2] = { FLT_MAX, FLT_MAX, FLT_MAX, FLT_MAX, 
															FLT_MAX, FLT_MAX, FLT_MAX, FLT_MAX };

						// pre-compute decoded pixels for each selector
						ColorFloatRGBA afrgbaSelectors[SELECTORS];
						assert(SELECTORS == 4);
						afrgbaSelectors[0] = (frgbaColor + s_aafCwTable[uiCW][0]).ClampRGB();
						afrgbaSelectors[1] = (frgbaColor + s_aafCwTable[uiCW][1]).ClampRGB();
						afrgbaSelectors[2] = (frgbaColor + s_aafCwTable[uiCW][2]).ClampRGB();
						afrgbaSelectors[3] = (frgbaColor + s_aafCwTable[uiCW][3]).ClampRGB();

						for (unsigned int uiPixel = 0; uiPixel < 8; uiPixel++)
						{
							ColorFloatRGBA *pfrgbaSourcePixel = &m_pafrgbaSource[a_phalf->m_pauiPixelMapping[uiPixel]];
							ColorFloatRGBA frgbaDecodedPixel;

							for (unsigned int uiSelector = 0; uiSelector < SELECTORS; uiSelector++)
							{
								frgbaDecodedPixel = afrgbaSelectors[uiSelector];

								float fPixelError;

								fPixelError = CalcPixelError(frgbaDecodedPixel, m_afDecodedAlphas[a_phalf->m_pauiPixelMapping[uiPixel]],
																	*pfrgbaSourcePixel);

								if (fPixelError < afPixelErrors[uiPixel])
								{
									auiPixelSelectors[uiPixel] = uiSelector;
									afrgbaDecodedPixels[uiPixel] = frgbaDecodedPixel;
									afPixelErrors[uiPixel] = fPixelError;
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
	// try an ETC1 individual mode encoding
	// use a_boolFlip to set the encoding F bit
	// use a_uiRadius to alter basecolor components in the range[-a_uiRadius:a_uiRadius]
	// replace the encoding if the encoding error is less than previous encoding
	//
	void Block4x4Encoding_ETC1::TryIndividual(bool a_boolFlip, unsigned int a_uiRadius)
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

		IndividualTrys trys(frgbaColor1, frgbaColor2, pauiPixelMapping1, pauiPixelMapping2, a_uiRadius);

		Block4x4Encoding_ETC1 encodingTry = *this;
		encodingTry.m_boolFlip = a_boolFlip;

		encodingTry.TryIndividualHalf(&trys.m_half1);
		encodingTry.TryIndividualHalf(&trys.m_half2);

		// use the best of each half
		IndividualTrys::Try *ptryBest1 = trys.m_half1.m_ptryBest;
		IndividualTrys::Try *ptryBest2 = trys.m_half2.m_ptryBest;
		encodingTry.m_fError = trys.m_half1.m_ptryBest->m_fError + trys.m_half2.m_ptryBest->m_fError;

		if (encodingTry.m_fError < m_fError)
		{
			m_mode = MODE_ETC1;
			m_boolDiff = false;
			m_boolFlip = encodingTry.m_boolFlip;
			m_frgbaColor1 = ColorFloatRGBA::ConvertFromRGB4((unsigned char)ptryBest1->m_iRed, (unsigned char)ptryBest1->m_iGreen, (unsigned char)ptryBest1->m_iBlue);
			m_frgbaColor2 = ColorFloatRGBA::ConvertFromRGB4((unsigned char)ptryBest2->m_iRed, (unsigned char)ptryBest2->m_iGreen, (unsigned char)ptryBest2->m_iBlue);
			m_uiCW1 = ptryBest1->m_uiCW;
			m_uiCW2 = ptryBest2->m_uiCW;

			for (unsigned int uiPixelOrder = 0; uiPixelOrder < PIXELS / 2; uiPixelOrder++)
			{
				unsigned int uiPixel1 = pauiPixelMapping1[uiPixelOrder];
				unsigned int uiPixel2 = pauiPixelMapping2[uiPixelOrder];

				unsigned int uiSelector1 = ptryBest1->m_auiSelectors[uiPixelOrder];
				unsigned int uiSelector2 = ptryBest2->m_auiSelectors[uiPixelOrder];

				m_auiSelectors[uiPixel1] = uiSelector1;
				m_auiSelectors[uiPixel2] = ptryBest2->m_auiSelectors[uiPixelOrder];

				float fDeltaRGB1 = s_aafCwTable[m_uiCW1][uiSelector1];
				float fDeltaRGB2 = s_aafCwTable[m_uiCW2][uiSelector2];

				m_afrgbaDecodedColors[uiPixel1] = (m_frgbaColor1 + fDeltaRGB1).ClampRGB();
				m_afrgbaDecodedColors[uiPixel2] = (m_frgbaColor2 + fDeltaRGB2).ClampRGB();
			}

			m_fError1 = ptryBest1->m_fError;
			m_fError2 = ptryBest2->m_fError;
			m_fError = m_fError1 + m_fError2;
		}

	}

	// ----------------------------------------------------------------------------------------------------
	// try an ETC1 differential mode encoding for a half of a 4x4 block
	// vary the basecolor components using a radius
	//
	void Block4x4Encoding_ETC1::TryIndividualHalf(IndividualTrys::Half *a_phalf)
	{

		a_phalf->m_ptryBest = nullptr;
		float fBestTryError = FLT_MAX;

		a_phalf->m_uiTrys = 0;
		for (int iRed = a_phalf->m_iRed - (int)a_phalf->m_uiRadius;
			iRed <= a_phalf->m_iRed + (int)a_phalf->m_uiRadius;
			iRed++)
		{
			assert(iRed >= 0 && iRed <= 15);

			for (int iGreen = a_phalf->m_iGreen - (int)a_phalf->m_uiRadius;
				iGreen <= a_phalf->m_iGreen + (int)a_phalf->m_uiRadius;
				iGreen++)
			{
				assert(iGreen >= 0 && iGreen <= 15);

				for (int iBlue = a_phalf->m_iBlue - (int)a_phalf->m_uiRadius;
					iBlue <= a_phalf->m_iBlue + (int)a_phalf->m_uiRadius;
					iBlue++)
				{
					assert(iBlue >= 0 && iBlue <= 15);

					IndividualTrys::Try *ptry = &a_phalf->m_atry[a_phalf->m_uiTrys];
					assert(ptry < &a_phalf->m_atry[IndividualTrys::Half::MAX_TRYS]);

					ptry->m_iRed = iRed;
					ptry->m_iGreen = iGreen;
					ptry->m_iBlue = iBlue;
					ptry->m_fError = FLT_MAX;
					ColorFloatRGBA frgbaColor = ColorFloatRGBA::ConvertFromRGB4((unsigned char)iRed, (unsigned char)iGreen, (unsigned char)iBlue);

					// try each CW
					for (unsigned int uiCW = 0; uiCW < CW_RANGES; uiCW++)
					{
						unsigned int auiPixelSelectors[PIXELS / 2];
						ColorFloatRGBA	afrgbaDecodedPixels[PIXELS / 2];
						float afPixelErrors[PIXELS / 2] = { FLT_MAX, FLT_MAX, FLT_MAX, FLT_MAX,
															FLT_MAX, FLT_MAX, FLT_MAX, FLT_MAX };

						// pre-compute decoded pixels for each selector
						ColorFloatRGBA afrgbaSelectors[SELECTORS];
						assert(SELECTORS == 4);
						afrgbaSelectors[0] = (frgbaColor + s_aafCwTable[uiCW][0]).ClampRGB();
						afrgbaSelectors[1] = (frgbaColor + s_aafCwTable[uiCW][1]).ClampRGB();
						afrgbaSelectors[2] = (frgbaColor + s_aafCwTable[uiCW][2]).ClampRGB();
						afrgbaSelectors[3] = (frgbaColor + s_aafCwTable[uiCW][3]).ClampRGB();

						for (unsigned int uiPixel = 0; uiPixel < 8; uiPixel++)
						{
							ColorFloatRGBA *pfrgbaSourcePixel = &m_pafrgbaSource[a_phalf->m_pauiPixelMapping[uiPixel]];
							ColorFloatRGBA frgbaDecodedPixel;

							for (unsigned int uiSelector = 0; uiSelector < SELECTORS; uiSelector++)
							{
								frgbaDecodedPixel = afrgbaSelectors[uiSelector];

								float fPixelError;

								fPixelError = CalcPixelError(frgbaDecodedPixel, m_afDecodedAlphas[a_phalf->m_pauiPixelMapping[uiPixel]],
										*pfrgbaSourcePixel);

								if (fPixelError < afPixelErrors[uiPixel])
								{
									auiPixelSelectors[uiPixel] = uiSelector;
									afrgbaDecodedPixels[uiPixel] = frgbaDecodedPixel;
									afPixelErrors[uiPixel] = fPixelError;
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
	// try version 1 of the degenerate search
	// degenerate encodings use basecolor movement and a subset of the selectors to find useful encodings
	// each subsequent version of the degenerate search uses more basecolor movement and is less likely to
	//		be successfull
	//
	void Block4x4Encoding_ETC1::TryDegenerates1(void)
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
	void Block4x4Encoding_ETC1::TryDegenerates2(void)
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
	void Block4x4Encoding_ETC1::TryDegenerates3(void)
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
	void Block4x4Encoding_ETC1::TryDegenerates4(void)
	{

		TryDifferential(m_boolMostLikelyFlip, 1, -4, 0);
		TryDifferential(m_boolMostLikelyFlip, 1, 4, 0);
		TryDifferential(m_boolMostLikelyFlip, 1, 0, 4);
		TryDifferential(m_boolMostLikelyFlip, 1, 0, -4);

	}

	// ----------------------------------------------------------------------------------------------------
	// find the best selector for each pixel based on a particular basecolor and CW that have been previously set
	// calculate the selectors for each half of the block separately
	// set the block error as the sum of each half's error
	//
	void Block4x4Encoding_ETC1::CalculateSelectors()
	{
		if (m_boolFlip)
		{
			CalculateHalfOfTheSelectors(0, s_auiTopPixelMapping);
			CalculateHalfOfTheSelectors(1, s_auiBottomPixelMapping);
		}
		else
		{
			CalculateHalfOfTheSelectors(0, s_auiLeftPixelMapping);
			CalculateHalfOfTheSelectors(1, s_auiRightPixelMapping);
		}

		m_fError = m_fError1 + m_fError2;
	}

	// ----------------------------------------------------------------------------------------------------
	// choose best selectors for half of the block
	// calculate the error for half of the block
	//
	void Block4x4Encoding_ETC1::CalculateHalfOfTheSelectors(unsigned int a_uiHalf,
		const unsigned int *pauiPixelMapping)
	{
		static const bool DEBUG_PRINT = false;

		ColorFloatRGBA *pfrgbaColor = a_uiHalf ? &m_frgbaColor2 : &m_frgbaColor1;
		unsigned int *puiCW = a_uiHalf ? &m_uiCW2 : &m_uiCW1;

		float *pfHalfError = a_uiHalf ? &m_fError2 : &m_fError1;
		*pfHalfError = FLT_MAX;

		// try each CW
		for (unsigned int uiCW = 0; uiCW < CW_RANGES; uiCW++)
		{
			if (DEBUG_PRINT)
			{
				printf("\ncw=%u\n", uiCW);
			}

			unsigned int auiPixelSelectors[PIXELS / 2];
			ColorFloatRGBA	afrgbaDecodedPixels[PIXELS / 2];
			float afPixelErrors[PIXELS / 2] = { FLT_MAX, FLT_MAX, FLT_MAX, FLT_MAX, FLT_MAX, FLT_MAX, FLT_MAX, FLT_MAX };

			for (unsigned int uiPixel = 0; uiPixel < 8; uiPixel++)
			{
				if (DEBUG_PRINT)
				{
					printf("\tsource [%.2f,%.2f,%.2f]\n", m_pafrgbaSource[pauiPixelMapping[uiPixel]].fR,
						m_pafrgbaSource[pauiPixelMapping[uiPixel]].fG, m_pafrgbaSource[pauiPixelMapping[uiPixel]].fB);
				}

				ColorFloatRGBA *pfrgbaSourcePixel = &m_pafrgbaSource[pauiPixelMapping[uiPixel]];
				ColorFloatRGBA frgbaDecodedPixel;

				for (unsigned int uiSelector = 0; uiSelector < SELECTORS; uiSelector++)
				{
					float fDeltaRGB = s_aafCwTable[uiCW][uiSelector];

					frgbaDecodedPixel = (*pfrgbaColor + fDeltaRGB).ClampRGB();

					float fPixelError;
					
					fPixelError = CalcPixelError(frgbaDecodedPixel, m_afDecodedAlphas[pauiPixelMapping[uiPixel]],
														*pfrgbaSourcePixel);
					
					if (DEBUG_PRINT)
					{
						printf("\tpixel %u, index %u [%.2f,%.2f,%.2f], error %.2f", uiPixel, uiSelector,
							frgbaDecodedPixel.fR,
							frgbaDecodedPixel.fG,
							frgbaDecodedPixel.fB,
							fPixelError);
					}

					if (fPixelError < afPixelErrors[uiPixel])
					{
						if (DEBUG_PRINT)
						{
							printf(" *");
						}

						auiPixelSelectors[uiPixel] = uiSelector;
						afrgbaDecodedPixels[uiPixel] = frgbaDecodedPixel;
						afPixelErrors[uiPixel] = fPixelError;
					}

					if (DEBUG_PRINT)
					{
						printf("\n");
					}
				}
			}

			// add up all pixel errors
			float fCWError = 0.0f;
			for (unsigned int uiPixel = 0; uiPixel < 8; uiPixel++)
			{
				fCWError += afPixelErrors[uiPixel];
			}
			if (DEBUG_PRINT)
			{
				printf("\terror %.2f\n", fCWError);
			}

			// if best CW so far
			if (fCWError < *pfHalfError)
			{
				*pfHalfError = fCWError;
				*puiCW = uiCW;
				for (unsigned int uiPixel = 0; uiPixel < 8; uiPixel++)
				{
					m_auiSelectors[pauiPixelMapping[uiPixel]] = auiPixelSelectors[uiPixel];
					m_afrgbaDecodedColors[pauiPixelMapping[uiPixel]] = afrgbaDecodedPixels[uiPixel];
				}
			}
		}

	}

	// ----------------------------------------------------------------------------------------------------
	// set the encoding bits based on encoding state
	//
	void Block4x4Encoding_ETC1::SetEncodingBits(void)
	{
		assert(m_mode == MODE_ETC1);

		if (m_boolDiff)
		{
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

			m_pencodingbitsRGB8->differential.red1 = (unsigned int)iRed1;
			m_pencodingbitsRGB8->differential.green1 = (unsigned int)iGreen1;
			m_pencodingbitsRGB8->differential.blue1 = (unsigned int)iBlue1;

			m_pencodingbitsRGB8->differential.dred2 = iDRed2;
			m_pencodingbitsRGB8->differential.dgreen2 = iDGreen2;
			m_pencodingbitsRGB8->differential.dblue2 = iDBlue2;
		}
		else
		{
			m_pencodingbitsRGB8->individual.red1 = (unsigned int)m_frgbaColor1.IntRed(15.0f);
			m_pencodingbitsRGB8->individual.green1 = (unsigned int)m_frgbaColor1.IntGreen(15.0f);
			m_pencodingbitsRGB8->individual.blue1 = (unsigned int)m_frgbaColor1.IntBlue(15.0f);

			m_pencodingbitsRGB8->individual.red2 = (unsigned int)m_frgbaColor2.IntRed(15.0f);
			m_pencodingbitsRGB8->individual.green2 = (unsigned int)m_frgbaColor2.IntGreen(15.0f);
			m_pencodingbitsRGB8->individual.blue2 = (unsigned int)m_frgbaColor2.IntBlue(15.0f);
		}

		m_pencodingbitsRGB8->individual.cw1 = m_uiCW1;
		m_pencodingbitsRGB8->individual.cw2 = m_uiCW2;

		SetEncodingBits_Selectors();

		m_pencodingbitsRGB8->individual.diff = (unsigned int)m_boolDiff;
		m_pencodingbitsRGB8->individual.flip = (unsigned int)m_boolFlip;

	}

	// ----------------------------------------------------------------------------------------------------
	// set the selectors in the encoding bits
	//
	void Block4x4Encoding_ETC1::SetEncodingBits_Selectors(void)
	{

		m_pencodingbitsRGB8->individual.selectors = 0;
		for (unsigned int uiPixel = 0; uiPixel < PIXELS; uiPixel++)
		{
			unsigned int uiSelector = m_auiSelectors[uiPixel];

			// set index msb
			m_pencodingbitsRGB8->individual.selectors |= (uiSelector >> 1) << (uiPixel ^ 8);

			// set index lsb
			m_pencodingbitsRGB8->individual.selectors |= (uiSelector & 1) << ((16 + uiPixel) ^ 8);
		}

	}

	// ----------------------------------------------------------------------------------------------------
	// set the decoded colors and decoded alpha based on the encoding state
	//
	void Block4x4Encoding_ETC1::Decode(void)
	{

		const unsigned int *pauiPixelOrder = m_boolFlip ? s_auiPixelOrderFlip1 : s_auiPixelOrderFlip0;

		for (unsigned int uiPixelOrder = 0; uiPixelOrder < PIXELS; uiPixelOrder++)
		{
			ColorFloatRGBA *pfrgbaCenter = uiPixelOrder < 8 ? &m_frgbaColor1 : &m_frgbaColor2;
			unsigned int uiCW = uiPixelOrder < 8 ? m_uiCW1 : m_uiCW2;

			unsigned int uiPixel = pauiPixelOrder[uiPixelOrder];

			float fDelta = s_aafCwTable[uiCW][m_auiSelectors[uiPixel]];
			m_afrgbaDecodedColors[uiPixel] = (*pfrgbaCenter + fDelta).ClampRGB();
			m_afDecodedAlphas[uiPixel] = 1.0f;
		}

	}

	// ----------------------------------------------------------------------------------------------------
	//

} // namespace Etc
