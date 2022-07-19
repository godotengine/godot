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
EtcBlock4x4Encoding_RGB8.cpp

Block4x4Encoding_RGB8 is the encoder to use for the ETC2 extensions when targetting file format RGB8.  
This encoder is also used for the ETC2 subset of file format RGBA8.

Block4x4Encoding_ETC1 encodes the ETC1 subset of RGB8.

*/

#include "EtcConfig.h"
#include "EtcBlock4x4Encoding_RGB8.h"

#include "EtcBlock4x4EncodingBits.h"
#include "EtcBlock4x4.h"
#include "EtcMath.h"

#include <stdio.h>
#include <string.h>
#include <assert.h>
#include <float.h>
#include <limits>

namespace Etc
{
	float Block4x4Encoding_RGB8::s_afTHDistanceTable[TH_DISTANCES] =
	{
		3.0f / 255.0f,
		6.0f / 255.0f,
		11.0f / 255.0f,
		16.0f / 255.0f,
		23.0f / 255.0f,
		32.0f / 255.0f,
		41.0f / 255.0f,
		64.0f / 255.0f
	};

	// ----------------------------------------------------------------------------------------------------
	//
	Block4x4Encoding_RGB8::Block4x4Encoding_RGB8(void)
	{

		m_pencodingbitsRGB8 = nullptr;

	}

	Block4x4Encoding_RGB8::~Block4x4Encoding_RGB8(void) {}
	// ----------------------------------------------------------------------------------------------------
	// initialization from the encoding bits of a previous encoding
	// a_pblockParent points to the block associated with this encoding
	// a_errormetric is used to choose the best encoding
	// a_pafrgbaSource points to a 4x4 block subset of the source image
	// a_paucEncodingBits points to the final encoding bits of a previous encoding
	//
	void Block4x4Encoding_RGB8::InitFromEncodingBits(Block4x4 *a_pblockParent,
														unsigned char *a_paucEncodingBits,
														ColorFloatRGBA *a_pafrgbaSource,
														ErrorMetric a_errormetric)
	{
		
		// handle ETC1 modes
		Block4x4Encoding_ETC1::InitFromEncodingBits(a_pblockParent,
													a_paucEncodingBits, a_pafrgbaSource,a_errormetric);

		m_pencodingbitsRGB8 = (Block4x4EncodingBits_RGB8 *)a_paucEncodingBits;

		// detect if there is a T, H or Planar mode present
		if (m_pencodingbitsRGB8->differential.diff)
		{
			int iRed1 = (int)m_pencodingbitsRGB8->differential.red1;
			int iDRed2 = m_pencodingbitsRGB8->differential.dred2;
			int iRed2 = iRed1 + iDRed2;

			int iGreen1 = (int)m_pencodingbitsRGB8->differential.green1;
			int iDGreen2 = m_pencodingbitsRGB8->differential.dgreen2;
			int iGreen2 = iGreen1 + iDGreen2;

			int iBlue1 = (int)m_pencodingbitsRGB8->differential.blue1;
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
				InitFromEncodingBits_Planar();
			}
		}

	}

	// ----------------------------------------------------------------------------------------------------
	// initialization from the encoding bits of a previous encoding if T mode is detected
	//
	void Block4x4Encoding_RGB8::InitFromEncodingBits_T(void)
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
	void Block4x4Encoding_RGB8::InitFromEncodingBits_H(void)
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
	// initialization from the encoding bits of a previous encoding if Planar mode is detected
	//
	void Block4x4Encoding_RGB8::InitFromEncodingBits_Planar(void)
	{

		m_mode = MODE_PLANAR;

		unsigned char ucOriginRed = m_pencodingbitsRGB8->planar.originRed;
		unsigned char ucOriginGreen = (unsigned char)((m_pencodingbitsRGB8->planar.originGreen1 << 6) +
										m_pencodingbitsRGB8->planar.originGreen2);
		unsigned char ucOriginBlue = (unsigned char)((m_pencodingbitsRGB8->planar.originBlue1 << 5) +
										(m_pencodingbitsRGB8->planar.originBlue2 << 3) +
										(m_pencodingbitsRGB8->planar.originBlue3 << 1) +
										m_pencodingbitsRGB8->planar.originBlue4);

		unsigned char ucHorizRed = (unsigned char)((m_pencodingbitsRGB8->planar.horizRed1 << 1) +
									m_pencodingbitsRGB8->planar.horizRed2);
		unsigned char ucHorizGreen = m_pencodingbitsRGB8->planar.horizGreen;
		unsigned char ucHorizBlue = (unsigned char)((m_pencodingbitsRGB8->planar.horizBlue1 << 5) +
									m_pencodingbitsRGB8->planar.horizBlue2);

		unsigned char ucVertRed = (unsigned char)((m_pencodingbitsRGB8->planar.vertRed1 << 3) +
									m_pencodingbitsRGB8->planar.vertRed2);
		unsigned char ucVertGreen = (unsigned char)((m_pencodingbitsRGB8->planar.vertGreen1 << 2) +
									m_pencodingbitsRGB8->planar.vertGreen2);
		unsigned char ucVertBlue = m_pencodingbitsRGB8->planar.vertBlue;

		m_frgbaColor1 = ColorFloatRGBA::ConvertFromR6G7B6(ucOriginRed, ucOriginGreen, ucOriginBlue);
		m_frgbaColor2 = ColorFloatRGBA::ConvertFromR6G7B6(ucHorizRed, ucHorizGreen, ucHorizBlue);
		m_frgbaColor3 = ColorFloatRGBA::ConvertFromR6G7B6(ucVertRed, ucVertGreen, ucVertBlue);

		DecodePixels_Planar();

		CalcBlockError();

	}

	// ----------------------------------------------------------------------------------------------------
	// perform a single encoding iteration
	// replace the encoding if a better encoding was found
	// subsequent iterations generally take longer for each iteration
	// set m_boolDone if encoding is perfect or encoding is finished based on a_fEffort
	//
	void Block4x4Encoding_RGB8::PerformIteration(float a_fEffort)
	{
		assert(!m_boolDone);

		switch (m_uiEncodingIterations)
		{
		case 0:
			Block4x4Encoding_ETC1::PerformFirstIteration();
			if (m_boolDone)
			{
				break;
			}
			TryPlanar(0);
			SetDoneIfPerfect();
			if (m_boolDone)
			{
				break;
			}
			TryTAndH(0);
			break;

		case 1:
			Block4x4Encoding_ETC1::TryDifferential(m_boolMostLikelyFlip, 1, 0, 0);
			break;

		case 2:
			Block4x4Encoding_ETC1::TryIndividual(m_boolMostLikelyFlip, 1);
			break;

		case 3:
			Block4x4Encoding_ETC1::TryDifferential(!m_boolMostLikelyFlip, 1, 0, 0);
			break;

		case 4:
			Block4x4Encoding_ETC1::TryIndividual(!m_boolMostLikelyFlip, 1);
			break;

		case 5:
			TryPlanar(1);
			if (a_fEffort <= 49.5f)
			{
				m_boolDone = true;
			}
			break;

		case 6:
			TryTAndH(1);
			if (a_fEffort <= 59.5f)
			{
				m_boolDone = true;
			}
			break;

		case 7:
			Block4x4Encoding_ETC1::TryDegenerates1();
			if (a_fEffort <= 69.5f)
			{
				m_boolDone = true;
			}
			break;

		case 8:
			Block4x4Encoding_ETC1::TryDegenerates2();
			if (a_fEffort <= 79.5f)
			{
				m_boolDone = true;
			}
			break;

		case 9:
			Block4x4Encoding_ETC1::TryDegenerates3();
			if (a_fEffort <= 89.5f)
			{
				m_boolDone = true;
			}
			break;

		case 10:
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
	// try encoding in Planar mode
	// save this encoding if it improves the error
	//
	void Block4x4Encoding_RGB8::TryPlanar(unsigned int a_uiRadius)
	{
		Block4x4Encoding_RGB8 encodingTry = *this;

		// init "try"
		{
			encodingTry.m_mode = MODE_PLANAR;
			encodingTry.m_boolDiff = true;
			encodingTry.m_boolFlip = false;
		}

		encodingTry.CalculatePlanarCornerColors();

		encodingTry.DecodePixels_Planar();

		encodingTry.CalcBlockError();

		if (a_uiRadius > 0)
		{
			encodingTry.TwiddlePlanar();
		}

		if (encodingTry.m_fError < m_fError)
		{
			m_mode = MODE_PLANAR;
			m_boolDiff = true;
			m_boolFlip = false;
			m_frgbaColor1 = encodingTry.m_frgbaColor1;
			m_frgbaColor2 = encodingTry.m_frgbaColor2;
			m_frgbaColor3 = encodingTry.m_frgbaColor3;

			for (unsigned int uiPixel = 0; uiPixel < PIXELS; uiPixel++)
			{
				m_afrgbaDecodedColors[uiPixel] = encodingTry.m_afrgbaDecodedColors[uiPixel];
			}

			m_fError = encodingTry.m_fError;
		}

	}

	// ----------------------------------------------------------------------------------------------------
	// try encoding in T mode or H mode
	// save this encoding if it improves the error
	//
	void Block4x4Encoding_RGB8::TryTAndH(unsigned int a_uiRadius)
	{

		CalculateBaseColorsForTAndH();

		TryT(a_uiRadius);

		TryH(a_uiRadius);

	}

	// ----------------------------------------------------------------------------------------------------
	// calculate original values for base colors
	// store them in m_frgbaOriginalColor1 and m_frgbaOriginalColor2
	//
	void Block4x4Encoding_RGB8::CalculateBaseColorsForTAndH(void)
	{

		bool boolRGBX = m_pblockParent->GetImageSource()->GetErrorMetric() == ErrorMetric::RGBX;

		ColorFloatRGBA frgbaBlockAverage = (m_frgbaSourceAverageLeft + m_frgbaSourceAverageRight) * 0.5f;

		// find pixel farthest from average gray line
		unsigned int uiFarthestPixel = 0;
		float fFarthestGrayDistance2 = 0.0f;
		unsigned int uiTransparentPixels = 0;
		for (unsigned int uiPixel = 0; uiPixel < PIXELS; uiPixel++)
		{
			// don't count transparent
			if (m_pafrgbaSource[uiPixel].fA == 0.0f && !boolRGBX)
			{
				uiTransparentPixels++;
			}
			else
			{
				float fGrayDistance2 = CalcGrayDistance2(m_pafrgbaSource[uiPixel], frgbaBlockAverage);

				if (fGrayDistance2 > fFarthestGrayDistance2)
				{
					uiFarthestPixel = uiPixel;
					fFarthestGrayDistance2 = fGrayDistance2;
				}
			}
		}
		// a transparent block should not reach this method
		assert(uiTransparentPixels < PIXELS);

		// set the original base colors to:
		//		half way to the farthest pixel and
		//		the mirror color on the other side of the average
		ColorFloatRGBA frgbaOffset = (m_pafrgbaSource[uiFarthestPixel] - frgbaBlockAverage) * 0.5f;
		m_frgbaOriginalColor1_TAndH = (frgbaBlockAverage + frgbaOffset).QuantizeR4G4B4();
		m_frgbaOriginalColor2_TAndH = (frgbaBlockAverage - frgbaOffset).ClampRGB().QuantizeR4G4B4();	// the "other side" might be out of range

		// move base colors to find best fit
		for (unsigned int uiIteration = 0; uiIteration < 10; uiIteration++)
		{
			// find the center of pixels closest to each color
			float fPixelsCloserToColor1 = 0.0f;
			ColorFloatRGBA frgbSumPixelsCloserToColor1;
			float fPixelsCloserToColor2 = 0.0f;
			ColorFloatRGBA frgbSumPixelsCloserToColor2;
			for (unsigned int uiPixel = 0; uiPixel < PIXELS; uiPixel++)
			{
				// don't count transparent pixels
				if (m_pafrgbaSource[uiPixel].fA == 0.0f)
				{
					continue;
				}

				float fGrayDistance2ToColor1 = CalcGrayDistance2(m_pafrgbaSource[uiPixel], m_frgbaOriginalColor1_TAndH);
				float fGrayDistance2ToColor2 = CalcGrayDistance2(m_pafrgbaSource[uiPixel], m_frgbaOriginalColor2_TAndH);

				ColorFloatRGBA frgbaAlphaWeightedSource = m_pafrgbaSource[uiPixel] * m_pafrgbaSource[uiPixel].fA;
					
				if (fGrayDistance2ToColor1 <= fGrayDistance2ToColor2)
				{
					fPixelsCloserToColor1 += m_pafrgbaSource[uiPixel].fA;
					frgbSumPixelsCloserToColor1 = frgbSumPixelsCloserToColor1 + frgbaAlphaWeightedSource;
				}
				else
				{
					fPixelsCloserToColor2 += m_pafrgbaSource[uiPixel].fA;
					frgbSumPixelsCloserToColor2 = frgbSumPixelsCloserToColor2 + frgbaAlphaWeightedSource;
				}
			}
			if (fPixelsCloserToColor1 == 0.0f || fPixelsCloserToColor2 == 0.0f)
			{
				break;
			}

			ColorFloatRGBA frgbAvgColor1Pixels = (frgbSumPixelsCloserToColor1 * (1.0f / fPixelsCloserToColor1)).QuantizeR4G4B4();
			ColorFloatRGBA frgbAvgColor2Pixels = (frgbSumPixelsCloserToColor2 * (1.0f / fPixelsCloserToColor2)).QuantizeR4G4B4();

			if (frgbAvgColor1Pixels.fR == m_frgbaOriginalColor1_TAndH.fR &&
				frgbAvgColor1Pixels.fG == m_frgbaOriginalColor1_TAndH.fG &&
				frgbAvgColor1Pixels.fB == m_frgbaOriginalColor1_TAndH.fB &&
				frgbAvgColor2Pixels.fR == m_frgbaOriginalColor2_TAndH.fR &&
				frgbAvgColor2Pixels.fG == m_frgbaOriginalColor2_TAndH.fG &&
				frgbAvgColor2Pixels.fB == m_frgbaOriginalColor2_TAndH.fB)
			{
				break;
			}

			m_frgbaOriginalColor1_TAndH = frgbAvgColor1Pixels;
			m_frgbaOriginalColor2_TAndH = frgbAvgColor2Pixels;
		}

	}

	// ----------------------------------------------------------------------------------------------------
	// try encoding in T mode
	// save this encoding if it improves the error
	//
	// since pixels that use base color1 don't use the distance table, color1 and color2 can be twiddled independently
	// better encoding can be found if TWIDDLE_RADIUS is set to 2, but it will be much slower
	//
	void Block4x4Encoding_RGB8::TryT(unsigned int a_uiRadius)
	{
		Block4x4Encoding_RGB8 encodingTry = *this;

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
	void Block4x4Encoding_RGB8::TryT_BestSelectorCombination(void)
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
		afrgbaDecodedPixel[2] = m_frgbaColor2;
		afrgbaDecodedPixel[3] = (m_frgbaColor2 - fDistance).ClampRGB();
		
		// try each selector
		for (unsigned int uiSelector = 0; uiSelector < SELECTORS; uiSelector++)
		{
			for (unsigned int uiPixel = 0; uiPixel < PIXELS; uiPixel++)
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
	// try encoding in T mode
	// save this encoding if it improves the error
	//
	// since all pixels use the distance table, color1 and color2 can NOT be twiddled independently
	// TWIDDLE_RADIUS of 2 is WAY too slow
	//
	void Block4x4Encoding_RGB8::TryH(unsigned int a_uiRadius)
	{
		Block4x4Encoding_RGB8 encodingTry = *this;

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
	void Block4x4Encoding_RGB8::TryH_BestSelectorCombination(void)
	{

		float fDistance = s_afTHDistanceTable[m_uiCW1];

		unsigned int auiBestPixelSelectors[PIXELS];
		float afBestPixelErrors[PIXELS] = { FLT_MAX, FLT_MAX, FLT_MAX, FLT_MAX, FLT_MAX, FLT_MAX, FLT_MAX, FLT_MAX,
			FLT_MAX, FLT_MAX, FLT_MAX, FLT_MAX, FLT_MAX, FLT_MAX, FLT_MAX, FLT_MAX };
		ColorFloatRGBA	afrgbaBestDecodedPixels[PIXELS];
		ColorFloatRGBA afrgbaDecodedPixel[SELECTORS];
		
		assert(SELECTORS == 4);
		afrgbaDecodedPixel[0] = (m_frgbaColor1 + fDistance).ClampRGB();
		afrgbaDecodedPixel[1] = (m_frgbaColor1 - fDistance).ClampRGB();
		afrgbaDecodedPixel[2] = (m_frgbaColor2 + fDistance).ClampRGB();
		afrgbaDecodedPixel[3] = (m_frgbaColor2 - fDistance).ClampRGB();
		
		// try each selector
		for (unsigned int uiSelector = 0; uiSelector < SELECTORS; uiSelector++)
		{
			for (unsigned int uiPixel = 0; uiPixel < PIXELS; uiPixel++)
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
	// use linear regression to find the best fit for colors along the edges of the 4x4 block
	//
	void Block4x4Encoding_RGB8::CalculatePlanarCornerColors(void)
	{
		ColorFloatRGBA afrgbaRegression[MAX_PLANAR_REGRESSION_SIZE];
		ColorFloatRGBA frgbaSlope;
		ColorFloatRGBA frgbaOffset;

		// top edge
		afrgbaRegression[0] = m_pafrgbaSource[0];
		afrgbaRegression[1] = m_pafrgbaSource[4];
		afrgbaRegression[2] = m_pafrgbaSource[8];
		afrgbaRegression[3] = m_pafrgbaSource[12];
		ColorRegression(afrgbaRegression, 4, &frgbaSlope, &frgbaOffset);
		m_frgbaColor1 = frgbaOffset;
		m_frgbaColor2 = (frgbaSlope * 4.0f) + frgbaOffset;

		// left edge
		afrgbaRegression[0] = m_pafrgbaSource[0];
		afrgbaRegression[1] = m_pafrgbaSource[1];
		afrgbaRegression[2] = m_pafrgbaSource[2];
		afrgbaRegression[3] = m_pafrgbaSource[3];
		ColorRegression(afrgbaRegression, 4, &frgbaSlope, &frgbaOffset);
		m_frgbaColor1 = (m_frgbaColor1 + frgbaOffset) * 0.5f;		// average with top edge
		m_frgbaColor3 = (frgbaSlope * 4.0f) + frgbaOffset;

		// right edge
		afrgbaRegression[0] = m_pafrgbaSource[12];
		afrgbaRegression[1] = m_pafrgbaSource[13];
		afrgbaRegression[2] = m_pafrgbaSource[14];
		afrgbaRegression[3] = m_pafrgbaSource[15];
		ColorRegression(afrgbaRegression, 4, &frgbaSlope, &frgbaOffset);
		m_frgbaColor2 = (m_frgbaColor2 + frgbaOffset) * 0.5f;		// average with top edge

		// bottom edge
		afrgbaRegression[0] = m_pafrgbaSource[3];
		afrgbaRegression[1] = m_pafrgbaSource[7];
		afrgbaRegression[2] = m_pafrgbaSource[11];
		afrgbaRegression[3] = m_pafrgbaSource[15];
		ColorRegression(afrgbaRegression, 4, &frgbaSlope, &frgbaOffset);
		m_frgbaColor3 = (m_frgbaColor3 + frgbaOffset) * 0.5f;		// average with left edge

		// quantize corner colors to 6/7/6
		m_frgbaColor1 = m_frgbaColor1.QuantizeR6G7B6();
		m_frgbaColor2 = m_frgbaColor2.QuantizeR6G7B6();
		m_frgbaColor3 = m_frgbaColor3.QuantizeR6G7B6();

	}

	// ----------------------------------------------------------------------------------------------------
	// try different corner colors by slightly changing R, G and B independently
	//
	// R, G and B decoding and errors are independent, so R, G and B twiddles can be independent
	//
	// return true if improvement
	//
	bool Block4x4Encoding_RGB8::TwiddlePlanar(void)
	{
		bool boolImprovement = false;

		while (TwiddlePlanarR())
		{
			boolImprovement = true;
		}

		while (TwiddlePlanarG())
		{
			boolImprovement = true;
		}

		while (TwiddlePlanarB())
		{
			boolImprovement = true;
		}

		return boolImprovement;
	}

	// ----------------------------------------------------------------------------------------------------
	// try different corner colors by slightly changing R
	//
	bool Block4x4Encoding_RGB8::TwiddlePlanarR()
	{
		bool boolImprovement = false;

		Block4x4Encoding_RGB8 encodingTry = *this;

		// init "try"
		{
			encodingTry.m_mode = MODE_PLANAR;
			encodingTry.m_boolDiff = true;
			encodingTry.m_boolFlip = false;
		}

		int iOriginRed = encodingTry.m_frgbaColor1.IntRed(63.0f);
		int iHorizRed = encodingTry.m_frgbaColor2.IntRed(63.0f);
		int iVertRed = encodingTry.m_frgbaColor3.IntRed(63.0f);

		for (int iTryOriginRed = iOriginRed - 1; iTryOriginRed <= iOriginRed + 1; iTryOriginRed++)
		{
			// check for out of range
			if (iTryOriginRed < 0 || iTryOriginRed > 63)
			{
				continue;
			}

			encodingTry.m_frgbaColor1.fR = ((iTryOriginRed << 2) + (iTryOriginRed >> 4)) / 255.0f;

			for (int iTryHorizRed = iHorizRed - 1; iTryHorizRed <= iHorizRed + 1; iTryHorizRed++)
			{
				// check for out of range
				if (iTryHorizRed < 0 || iTryHorizRed > 63)
				{
					continue;
				}

				encodingTry.m_frgbaColor2.fR = ((iTryHorizRed << 2) + (iTryHorizRed >> 4)) / 255.0f;

				for (int iTryVertRed = iVertRed - 1; iTryVertRed <= iVertRed + 1; iTryVertRed++)
				{
					// check for out of range
					if (iTryVertRed < 0 || iTryVertRed > 63)
					{
						continue;
					}

					// don't bother with null twiddle
					if (iTryOriginRed == iOriginRed && iTryHorizRed == iHorizRed && iTryVertRed == iVertRed)
					{
						continue;
					}

					encodingTry.m_frgbaColor3.fR = ((iTryVertRed << 2) + (iTryVertRed >> 4)) / 255.0f;

					encodingTry.DecodePixels_Planar();

					encodingTry.CalcBlockError();

					if (encodingTry.m_fError < m_fError)
					{
						m_mode = MODE_PLANAR;
						m_boolDiff = true;
						m_boolFlip = false;
						m_frgbaColor1 = encodingTry.m_frgbaColor1;
						m_frgbaColor2 = encodingTry.m_frgbaColor2;
						m_frgbaColor3 = encodingTry.m_frgbaColor3;

						for (unsigned int uiPixel = 0; uiPixel < PIXELS; uiPixel++)
						{
							m_afrgbaDecodedColors[uiPixel] = encodingTry.m_afrgbaDecodedColors[uiPixel];
						}

						m_fError = encodingTry.m_fError;

						boolImprovement = true;
					}
				}
			}
		}

		return boolImprovement;
	}

	// ----------------------------------------------------------------------------------------------------
	// try different corner colors by slightly changing G
	//
	bool Block4x4Encoding_RGB8::TwiddlePlanarG()
	{
		bool boolImprovement = false;

		Block4x4Encoding_RGB8 encodingTry = *this;

		// init "try"
		{
			encodingTry.m_mode = MODE_PLANAR;
			encodingTry.m_boolDiff = true;
			encodingTry.m_boolFlip = false;
		}

		int iOriginGreen = encodingTry.m_frgbaColor1.IntGreen(127.0f);
		int iHorizGreen = encodingTry.m_frgbaColor2.IntGreen(127.0f);
		int iVertGreen = encodingTry.m_frgbaColor3.IntGreen(127.0f);

		for (int iTryOriginGreen = iOriginGreen - 1; iTryOriginGreen <= iOriginGreen + 1; iTryOriginGreen++)
		{
			// check for out of range
			if (iTryOriginGreen < 0 || iTryOriginGreen > 127)
			{
				continue;
			}

			encodingTry.m_frgbaColor1.fG = ((iTryOriginGreen << 1) + (iTryOriginGreen >> 6)) / 255.0f;

			for (int iTryHorizGreen = iHorizGreen - 1; iTryHorizGreen <= iHorizGreen + 1; iTryHorizGreen++)
			{
				// check for out of range
				if (iTryHorizGreen < 0 || iTryHorizGreen > 127)
				{
					continue;
				}

				encodingTry.m_frgbaColor2.fG = ((iTryHorizGreen << 1) + (iTryHorizGreen >> 6)) / 255.0f;

				for (int iTryVertGreen = iVertGreen - 1; iTryVertGreen <= iVertGreen + 1; iTryVertGreen++)
				{
					// check for out of range
					if (iTryVertGreen < 0 || iTryVertGreen > 127)
					{
						continue;
					}

					// don't bother with null twiddle
					if (iTryOriginGreen == iOriginGreen && 
						iTryHorizGreen == iHorizGreen && 
						iTryVertGreen == iVertGreen)
					{
						continue;
					}

					encodingTry.m_frgbaColor3.fG = ((iTryVertGreen << 1) + (iTryVertGreen >> 6)) / 255.0f;

					encodingTry.DecodePixels_Planar();

					encodingTry.CalcBlockError();

					if (encodingTry.m_fError < m_fError)
					{
						m_mode = MODE_PLANAR;
						m_boolDiff = true;
						m_boolFlip = false;
						m_frgbaColor1 = encodingTry.m_frgbaColor1;
						m_frgbaColor2 = encodingTry.m_frgbaColor2;
						m_frgbaColor3 = encodingTry.m_frgbaColor3;

						for (unsigned int uiPixel = 0; uiPixel < PIXELS; uiPixel++)
						{
							m_afrgbaDecodedColors[uiPixel] = encodingTry.m_afrgbaDecodedColors[uiPixel];
						}

						m_fError = encodingTry.m_fError;

						boolImprovement = true;
					}
				}
			}
		}

		return boolImprovement;
	}

	// ----------------------------------------------------------------------------------------------------
	// try different corner colors by slightly changing B
	//
	bool Block4x4Encoding_RGB8::TwiddlePlanarB()
	{
		bool boolImprovement = false;

		Block4x4Encoding_RGB8 encodingTry = *this;

		// init "try"
		{
			encodingTry.m_mode = MODE_PLANAR;
			encodingTry.m_boolDiff = true;
			encodingTry.m_boolFlip = false;
		}

		int iOriginBlue = encodingTry.m_frgbaColor1.IntBlue(63.0f);
		int iHorizBlue = encodingTry.m_frgbaColor2.IntBlue(63.0f);
		int iVertBlue = encodingTry.m_frgbaColor3.IntBlue(63.0f);

		for (int iTryOriginBlue = iOriginBlue - 1; iTryOriginBlue <= iOriginBlue + 1; iTryOriginBlue++)
		{
			// check for out of range
			if (iTryOriginBlue < 0 || iTryOriginBlue > 63)
			{
				continue;
			}

			encodingTry.m_frgbaColor1.fB = ((iTryOriginBlue << 2) + (iTryOriginBlue >> 4)) / 255.0f;

			for (int iTryHorizBlue = iHorizBlue - 1; iTryHorizBlue <= iHorizBlue + 1; iTryHorizBlue++)
			{
				// check for out of range
				if (iTryHorizBlue < 0 || iTryHorizBlue > 63)
				{
					continue;
				}

				encodingTry.m_frgbaColor2.fB = ((iTryHorizBlue << 2) + (iTryHorizBlue >> 4)) / 255.0f;

				for (int iTryVertBlue = iVertBlue - 1; iTryVertBlue <= iVertBlue + 1; iTryVertBlue++)
				{
					// check for out of range
					if (iTryVertBlue < 0 || iTryVertBlue > 63)
					{
						continue;
					}

					// don't bother with null twiddle
					if (iTryOriginBlue == iOriginBlue && iTryHorizBlue == iHorizBlue && iTryVertBlue == iVertBlue)
					{
						continue;
					}

					encodingTry.m_frgbaColor3.fB = ((iTryVertBlue << 2) + (iTryVertBlue >> 4)) / 255.0f;

					encodingTry.DecodePixels_Planar();

					encodingTry.CalcBlockError();

					if (encodingTry.m_fError < m_fError)
					{
						m_mode = MODE_PLANAR;
						m_boolDiff = true;
						m_boolFlip = false;
						m_frgbaColor1 = encodingTry.m_frgbaColor1;
						m_frgbaColor2 = encodingTry.m_frgbaColor2;
						m_frgbaColor3 = encodingTry.m_frgbaColor3;

						for (unsigned int uiPixel = 0; uiPixel < PIXELS; uiPixel++)
						{
							m_afrgbaDecodedColors[uiPixel] = encodingTry.m_afrgbaDecodedColors[uiPixel];
						}

						m_fError = encodingTry.m_fError;

						boolImprovement = true;
					}
				}
			}
		}

		return boolImprovement;
	}

	// ----------------------------------------------------------------------------------------------------
	// set the encoding bits based on encoding state
	//
	void Block4x4Encoding_RGB8::SetEncodingBits(void)
	{

		switch (m_mode)
		{
		case MODE_ETC1:
			Block4x4Encoding_ETC1::SetEncodingBits();
			break;

		case MODE_T:
			SetEncodingBits_T();
			break;

		case MODE_H:
			SetEncodingBits_H();
			break;

		case MODE_PLANAR:
			SetEncodingBits_Planar();
			break;

		default:
			assert(false);
		}

	}

	// ----------------------------------------------------------------------------------------------------
	// set the encoding bits based on encoding state for T mode
	//
	void Block4x4Encoding_RGB8::SetEncodingBits_T(void)
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

		m_pencodingbitsRGB8->t.diff = 1;

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
	// set the encoding bits based on encoding state for H mode
	//
	// colors and selectors may need to swap in order to generate lsb of distance index
	//
	void Block4x4Encoding_RGB8::SetEncodingBits_H(void)
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

		m_pencodingbitsRGB8->h.diff = 1;

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

	// ----------------------------------------------------------------------------------------------------
	// set the encoding bits based on encoding state for Planar mode
	//
	void Block4x4Encoding_RGB8::SetEncodingBits_Planar(void)
	{
		static const bool SANITY_CHECK = true;

		assert(m_mode == MODE_PLANAR);
		assert(m_boolDiff == true);

		unsigned int uiOriginRed = (unsigned int)m_frgbaColor1.IntRed(63.0f);
		unsigned int uiOriginGreen = (unsigned int)m_frgbaColor1.IntGreen(127.0f);
		unsigned int uiOriginBlue = (unsigned int)m_frgbaColor1.IntBlue(63.0f);

		unsigned int uiHorizRed = (unsigned int)m_frgbaColor2.IntRed(63.0f);
		unsigned int uiHorizGreen = (unsigned int)m_frgbaColor2.IntGreen(127.0f);
		unsigned int uiHorizBlue = (unsigned int)m_frgbaColor2.IntBlue(63.0f);

		unsigned int uiVertRed = (unsigned int)m_frgbaColor3.IntRed(63.0f);
		unsigned int uiVertGreen = (unsigned int)m_frgbaColor3.IntGreen(127.0f);
		unsigned int uiVertBlue = (unsigned int)m_frgbaColor3.IntBlue(63.0f);

		m_pencodingbitsRGB8->planar.originRed = uiOriginRed;
		m_pencodingbitsRGB8->planar.originGreen1 = uiOriginGreen >> 6;
		m_pencodingbitsRGB8->planar.originGreen2 = uiOriginGreen;
		m_pencodingbitsRGB8->planar.originBlue1 = uiOriginBlue >> 5;
		m_pencodingbitsRGB8->planar.originBlue2 = uiOriginBlue >> 3;
		m_pencodingbitsRGB8->planar.originBlue3 = uiOriginBlue >> 1;
		m_pencodingbitsRGB8->planar.originBlue4 = uiOriginBlue;

		m_pencodingbitsRGB8->planar.horizRed1 = uiHorizRed >> 1;
		m_pencodingbitsRGB8->planar.horizRed2 = uiHorizRed;
		m_pencodingbitsRGB8->planar.horizGreen = uiHorizGreen;
		m_pencodingbitsRGB8->planar.horizBlue1 = uiHorizBlue >> 5;
		m_pencodingbitsRGB8->planar.horizBlue2 = uiHorizBlue;

		m_pencodingbitsRGB8->planar.vertRed1 = uiVertRed >> 3;
		m_pencodingbitsRGB8->planar.vertRed2 = uiVertRed;
		m_pencodingbitsRGB8->planar.vertGreen1 = uiVertGreen >> 2;
		m_pencodingbitsRGB8->planar.vertGreen2 = uiVertGreen;
		m_pencodingbitsRGB8->planar.vertBlue = uiVertBlue;

		m_pencodingbitsRGB8->planar.diff = 1;

		// create valid RG differentials and an invalid B differential to trigger planar mode
		m_pencodingbitsRGB8->planar.detect1 = 0;
		m_pencodingbitsRGB8->planar.detect2 = 0;
		m_pencodingbitsRGB8->planar.detect3 = 0;
		m_pencodingbitsRGB8->planar.detect4 = 0;
		int iRed2 = (int)m_pencodingbitsRGB8->differential.red1 + (int)m_pencodingbitsRGB8->differential.dred2;
		int iGreen2 = (int)m_pencodingbitsRGB8->differential.green1 + (int)m_pencodingbitsRGB8->differential.dgreen2;
		int iBlue2 = (int)m_pencodingbitsRGB8->differential.blue1 + (int)m_pencodingbitsRGB8->differential.dblue2;
		if (iRed2 < 0 || iRed2 > 31)
		{
			m_pencodingbitsRGB8->planar.detect1 = 1;
		}
		if (iGreen2 < 0 || iGreen2 > 31)
		{
			m_pencodingbitsRGB8->planar.detect2 = 1;
		}
		if (iBlue2 >= 4)
		{
			m_pencodingbitsRGB8->planar.detect3 = 7;
			m_pencodingbitsRGB8->planar.detect4 = 0;
		}
		else
		{
			m_pencodingbitsRGB8->planar.detect3 = 0;
			m_pencodingbitsRGB8->planar.detect4 = 1;
		}

		if (SANITY_CHECK)
		{
			iRed2 = (int)m_pencodingbitsRGB8->differential.red1 + (int)m_pencodingbitsRGB8->differential.dred2;
			iGreen2 = (int)m_pencodingbitsRGB8->differential.green1 + (int)m_pencodingbitsRGB8->differential.dgreen2;
			iBlue2 = (int)m_pencodingbitsRGB8->differential.blue1 + (int)m_pencodingbitsRGB8->differential.dblue2;

			// make sure red and green don't overflow and blue does
			assert(iRed2 >= 0 && iRed2 <= 31);
			assert(iGreen2 >= 0 && iGreen2 <= 31);
			assert(iBlue2 < 0 || iBlue2 > 31);
		}

	}

	// ----------------------------------------------------------------------------------------------------
	// set the decoded colors and decoded alpha based on the encoding state for T mode
	//
	void Block4x4Encoding_RGB8::DecodePixels_T(void)
	{

		float fDistance = s_afTHDistanceTable[m_uiCW1];
		ColorFloatRGBA frgbaDistance(fDistance, fDistance, fDistance, 0.0f);

		for (unsigned int uiPixel = 0; uiPixel < PIXELS; uiPixel++)
		{
			switch (m_auiSelectors[uiPixel])
			{
			case 0:
				m_afrgbaDecodedColors[uiPixel] = m_frgbaColor1;
				break;

			case 1:
				m_afrgbaDecodedColors[uiPixel] = (m_frgbaColor2 + frgbaDistance).ClampRGB();
				break;

			case 2:
				m_afrgbaDecodedColors[uiPixel] = m_frgbaColor2;
				break;

			case 3:
				m_afrgbaDecodedColors[uiPixel] = (m_frgbaColor2 - frgbaDistance).ClampRGB();
				break;
			}

		}

	}

	// ----------------------------------------------------------------------------------------------------
	// set the decoded colors and decoded alpha based on the encoding state for H mode
	//
	void Block4x4Encoding_RGB8::DecodePixels_H(void)
	{

		float fDistance = s_afTHDistanceTable[m_uiCW1];
		ColorFloatRGBA frgbaDistance(fDistance, fDistance, fDistance, 0.0f);

		for (unsigned int uiPixel = 0; uiPixel < PIXELS; uiPixel++)
		{
			switch (m_auiSelectors[uiPixel])
			{
			case 0:
				m_afrgbaDecodedColors[uiPixel] = (m_frgbaColor1 + frgbaDistance).ClampRGB();
				break;

			case 1:
				m_afrgbaDecodedColors[uiPixel] = (m_frgbaColor1 - frgbaDistance).ClampRGB();
				break;

			case 2:
				m_afrgbaDecodedColors[uiPixel] = (m_frgbaColor2 + frgbaDistance).ClampRGB();
				break;

			case 3:
				m_afrgbaDecodedColors[uiPixel] = (m_frgbaColor2 - frgbaDistance).ClampRGB();
				break;
			}

		}

	}

	// ----------------------------------------------------------------------------------------------------
	// set the decoded colors and decoded alpha based on the encoding state for Planar mode
	//
	void Block4x4Encoding_RGB8::DecodePixels_Planar(void)
	{

		int iRO = (int)roundf(m_frgbaColor1.fR * 255.0f);
		int iGO = (int)roundf(m_frgbaColor1.fG * 255.0f);
		int iBO = (int)roundf(m_frgbaColor1.fB * 255.0f);

		int iRH = (int)roundf(m_frgbaColor2.fR * 255.0f);
		int iGH = (int)roundf(m_frgbaColor2.fG * 255.0f);
		int iBH = (int)roundf(m_frgbaColor2.fB * 255.0f);

		int iRV = (int)roundf(m_frgbaColor3.fR * 255.0f);
		int iGV = (int)roundf(m_frgbaColor3.fG * 255.0f);
		int iBV = (int)roundf(m_frgbaColor3.fB * 255.0f);

		for (unsigned int uiPixel = 0; uiPixel < PIXELS; uiPixel++)
		{
			int iX = (int)(uiPixel >> 2);
			int iY = (int)(uiPixel & 3);

			int iR = (iX*(iRH - iRO) + iY*(iRV - iRO) + 4*iRO + 2) >> 2;
			int iG = (iX*(iGH - iGO) + iY*(iGV - iGO) + 4*iGO + 2) >> 2;
			int iB = (iX*(iBH - iBO) + iY*(iBV - iBO) + 4*iBO + 2) >> 2;

			ColorFloatRGBA frgba;
			frgba.fR = (float)iR / 255.0f;
			frgba.fG = (float)iG / 255.0f;
			frgba.fB = (float)iB / 255.0f;
			frgba.fA = 1.0f;

			m_afrgbaDecodedColors[uiPixel] = frgba.ClampRGB();
		}

	}

	// ----------------------------------------------------------------------------------------------------
	// perform a linear regression for the a_uiPixels in a_pafrgbaPixels[]
	//
	// output the closest color line using a_pfrgbaSlope and a_pfrgbaOffset
	//
	void Block4x4Encoding_RGB8::ColorRegression(ColorFloatRGBA *a_pafrgbaPixels, unsigned int a_uiPixels,
												ColorFloatRGBA *a_pfrgbaSlope, ColorFloatRGBA *a_pfrgbaOffset)
	{
		typedef struct
		{
			float f[4];
		} Float4;

		Float4 *paf4Pixels = (Float4 *)(a_pafrgbaPixels);
		Float4 *pf4Slope = (Float4 *)(a_pfrgbaSlope);
		Float4 *pf4Offset = (Float4 *)(a_pfrgbaOffset);

		float afX[MAX_PLANAR_REGRESSION_SIZE];
		float afY[MAX_PLANAR_REGRESSION_SIZE];

		// handle r, g and b separately.  don't bother with a
		for (unsigned int uiComponent = 0; uiComponent < 3; uiComponent++)
		{
			for (unsigned int uiPixel = 0; uiPixel < a_uiPixels; uiPixel++)
			{
				afX[uiPixel] = (float)uiPixel;
				afY[uiPixel] = paf4Pixels[uiPixel].f[uiComponent];
				
			}
			Etc::Regression(afX, afY, a_uiPixels,
				&(pf4Slope->f[uiComponent]), &(pf4Offset->f[uiComponent]));
		}

	}

	// ----------------------------------------------------------------------------------------------------
	//
}
