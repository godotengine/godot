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
EtcBlock4x4Encoding.cpp

Block4x4Encoding is the abstract base class for the different encoders.  Each encoder targets a 
particular file format (e.g. ETC1, RGB8, RGBA8, R11)

*/

#include "EtcConfig.h"
#include "EtcBlock4x4Encoding.h"

#include "EtcBlock4x4EncodingBits.h"
#include "EtcBlock4x4.h"

#include <stdio.h>
#include <string.h>
#include <assert.h>

namespace Etc
{
	// ----------------------------------------------------------------------------------------------------
	//
	const float Block4x4Encoding::LUMA_WEIGHT = 3.0f;
	const float Block4x4Encoding::CHROMA_BLUE_WEIGHT = 0.5f;

	// ----------------------------------------------------------------------------------------------------
	//
	Block4x4Encoding::Block4x4Encoding(void)
	{

		m_pblockParent = nullptr;

		m_pafrgbaSource = nullptr;

		m_boolBorderPixels = false;

		m_fError = -1.0f;

		m_mode = MODE_UNKNOWN;

		m_uiEncodingIterations = 0;
		m_boolDone = false;

		for (unsigned int uiPixel = 0; uiPixel < PIXELS; uiPixel++)
		{
			m_afrgbaDecodedColors[uiPixel] = ColorFloatRGBA(-1.0f, -1.0f, -1.0f, -1.0f);
			m_afDecodedAlphas[uiPixel] = -1.0f;
		}

	}

	// ----------------------------------------------------------------------------------------------------
	// initialize the generic encoding for a 4x4 block
	// a_pblockParent points to the block associated with this encoding
	// a_errormetric is used to choose the best encoding
	// init the decoded pixels to -1 to mark them as undefined
	// init the error to -1 to mark it as undefined
	//
	void Block4x4Encoding::Init(Block4x4 *a_pblockParent,
								ColorFloatRGBA *a_pafrgbaSource,
								ErrorMetric a_errormetric)
	{

		m_pblockParent = a_pblockParent;

		m_pafrgbaSource = a_pafrgbaSource;

		m_boolBorderPixels = m_pblockParent->HasBorderPixels();

		m_fError = -1.0f;

		m_uiEncodingIterations = 0;

		m_errormetric = a_errormetric;

		for (unsigned int uiPixel = 0; uiPixel < PIXELS; uiPixel++)
		{
			m_afrgbaDecodedColors[uiPixel] = ColorFloatRGBA(-1.0f, -1.0f, -1.0f, -1.0f);
			m_afDecodedAlphas[uiPixel] = -1.0f;
		}

	}

	// ----------------------------------------------------------------------------------------------------
	// calculate the error for the block by summing the pixel errors
	//
	void Block4x4Encoding::CalcBlockError(void)
	{
		m_fError = 0.0f;

		for (unsigned int uiPixel = 0; uiPixel < PIXELS; uiPixel++)
		{
			m_fError += CalcPixelError(m_afrgbaDecodedColors[uiPixel], m_afDecodedAlphas[uiPixel],
										m_pafrgbaSource[uiPixel]);
		}
		
	}

	// ----------------------------------------------------------------------------------------------------
	// calculate the error between the source pixel and the decoded pixel
	// the error amount is base on the error metric
	//
	float Block4x4Encoding::CalcPixelError(ColorFloatRGBA a_frgbaDecodedColor, float a_fDecodedAlpha,
											ColorFloatRGBA a_frgbaSourcePixel)
	{

		// if a border pixel
		if (isnan(a_frgbaSourcePixel.fA))
		{
			return 0.0f;
		}

		if (m_errormetric == ErrorMetric::RGBA)
		{
			assert(a_fDecodedAlpha >= 0.0f);

			float fDRed = (a_fDecodedAlpha * a_frgbaDecodedColor.fR) -
							(a_frgbaSourcePixel.fA * a_frgbaSourcePixel.fR);
			float fDGreen = (a_fDecodedAlpha * a_frgbaDecodedColor.fG) -
							(a_frgbaSourcePixel.fA * a_frgbaSourcePixel.fG);
			float fDBlue = (a_fDecodedAlpha * a_frgbaDecodedColor.fB) -
							(a_frgbaSourcePixel.fA * a_frgbaSourcePixel.fB);

			float fDAlpha = a_fDecodedAlpha - a_frgbaSourcePixel.fA;

			return fDRed*fDRed + fDGreen*fDGreen + fDBlue*fDBlue + fDAlpha*fDAlpha;
		}
		else if (m_errormetric == ErrorMetric::RGBX)
		{
			assert(a_fDecodedAlpha >= 0.0f);

			float fDRed = a_frgbaDecodedColor.fR - a_frgbaSourcePixel.fR;
			float fDGreen = a_frgbaDecodedColor.fG - a_frgbaSourcePixel.fG;
			float fDBlue = a_frgbaDecodedColor.fB - a_frgbaSourcePixel.fB;
			float fDAlpha = a_fDecodedAlpha - a_frgbaSourcePixel.fA;

			return fDRed*fDRed + fDGreen*fDGreen + fDBlue*fDBlue + fDAlpha*fDAlpha;
		}
		else if (m_errormetric == ErrorMetric::REC709)
		{
			assert(a_fDecodedAlpha >= 0.0f);

			float fLuma1 = a_frgbaSourcePixel.fR*0.2126f + a_frgbaSourcePixel.fG*0.7152f + a_frgbaSourcePixel.fB*0.0722f;
			float fChromaR1 = 0.5f * ((a_frgbaSourcePixel.fR - fLuma1) * (1.0f / (1.0f - 0.2126f)));
			float fChromaB1 = 0.5f * ((a_frgbaSourcePixel.fB - fLuma1) * (1.0f / (1.0f - 0.0722f)));

			float fLuma2 = a_frgbaDecodedColor.fR*0.2126f +
							a_frgbaDecodedColor.fG*0.7152f +
							a_frgbaDecodedColor.fB*0.0722f;
			float fChromaR2 = 0.5f * ((a_frgbaDecodedColor.fR - fLuma2) * (1.0f / (1.0f - 0.2126f)));
			float fChromaB2 = 0.5f * ((a_frgbaDecodedColor.fB - fLuma2) * (1.0f / (1.0f - 0.0722f)));

			float fDeltaL = a_frgbaSourcePixel.fA * fLuma1 - a_fDecodedAlpha * fLuma2;
			float fDeltaCr = a_frgbaSourcePixel.fA * fChromaR1 - a_fDecodedAlpha * fChromaR2;
			float fDeltaCb = a_frgbaSourcePixel.fA * fChromaB1 - a_fDecodedAlpha * fChromaB2;

			float fDAlpha = a_fDecodedAlpha - a_frgbaSourcePixel.fA;

			// Favor Luma accuracy over Chroma, and Red over Blue 
			return LUMA_WEIGHT*fDeltaL*fDeltaL +
					fDeltaCr*fDeltaCr +
					CHROMA_BLUE_WEIGHT*fDeltaCb*fDeltaCb +
					fDAlpha*fDAlpha;
	#if 0
			float fDRed = a_frgbaDecodedPixel.fR - a_frgbaSourcePixel.fR;
			float fDGreen = a_frgbaDecodedPixel.fG - a_frgbaSourcePixel.fG;
			float fDBlue = a_frgbaDecodedPixel.fB - a_frgbaSourcePixel.fB;
			return 2.0f * 3.0f * fDeltaL * fDeltaL + fDRed*fDRed + fDGreen*fDGreen + fDBlue*fDBlue;
#endif
		}
		else if (m_errormetric == ErrorMetric::NORMALXYZ)
		{
			float fDecodedX = 2.0f * a_frgbaDecodedColor.fR - 1.0f;
			float fDecodedY = 2.0f * a_frgbaDecodedColor.fG - 1.0f;
			float fDecodedZ = 2.0f * a_frgbaDecodedColor.fB - 1.0f;

			float fDecodedLength = sqrtf(fDecodedX*fDecodedX + fDecodedY*fDecodedY + fDecodedZ*fDecodedZ);

			if (fDecodedLength < 0.5f)
			{
				return 1.0f;
			}
			else if (fDecodedLength == 0.0f)
			{
				fDecodedX = 1.0f;
				fDecodedY = 0.0f;
				fDecodedZ = 0.0f;
			}
			else
			{
				fDecodedX /= fDecodedLength;
				fDecodedY /= fDecodedLength;
				fDecodedZ /= fDecodedLength;
			}

			float fSourceX = 2.0f * a_frgbaSourcePixel.fR - 1.0f;
			float fSourceY = 2.0f * a_frgbaSourcePixel.fG - 1.0f;
			float fSourceZ = 2.0f * a_frgbaSourcePixel.fB - 1.0f;

			float fSourceLength = sqrtf(fSourceX*fSourceX + fSourceY*fSourceY + fSourceZ*fSourceZ);

			if (fSourceLength == 0.0f)
			{
				fSourceX = 1.0f;
				fSourceY = 0.0f;
				fSourceZ = 0.0f;
			}
			else
			{
				fSourceX /= fSourceLength;
				fSourceY /= fSourceLength;
				fSourceZ /= fSourceLength;
			}

			float fDotProduct = fSourceX*fDecodedX + fSourceY*fDecodedY + fSourceZ*fDecodedZ;
			float fNormalizedDotProduct = 1.0f - 0.5f * (fDotProduct + 1.0f);
			float fDotProductError = fNormalizedDotProduct * fNormalizedDotProduct;
			
			float fLength2 = fDecodedX*fDecodedX + fDecodedY*fDecodedY + fDecodedZ*fDecodedZ;
			float fLength2Error = fabsf(1.0f - fLength2);

			float fDeltaW = a_frgbaDecodedColor.fA - a_frgbaSourcePixel.fA;
			float fErrorW = fDeltaW * fDeltaW;

			return fDotProductError + fLength2Error + fErrorW;
		}
		else // ErrorMetric::NUMERIC
		{
			assert(a_fDecodedAlpha >= 0.0f);

			float fDX = a_frgbaDecodedColor.fR - a_frgbaSourcePixel.fR;
			float fDY = a_frgbaDecodedColor.fG - a_frgbaSourcePixel.fG;
			float fDZ = a_frgbaDecodedColor.fB - a_frgbaSourcePixel.fB;
			float fDW = a_frgbaDecodedColor.fA - a_frgbaSourcePixel.fA;

			return fDX*fDX + fDY*fDY + fDZ*fDZ + fDW*fDW;
		}

	}

	// ----------------------------------------------------------------------------------------------------
	//

} // namespace Etc

