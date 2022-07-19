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
EtcDifferentialTrys.cpp

Gathers the results of the various encoding trys for both halves of a 4x4 block for Differential mode

*/

#include "EtcConfig.h"
#include "EtcDifferentialTrys.h"

#include <assert.h>

namespace Etc
{

	// ----------------------------------------------------------------------------------------------------
	// construct a list of trys (encoding attempts)
	//
	// a_frgbaColor1 is the basecolor for the first half
	// a_frgbaColor2 is the basecolor for the second half
	// a_pauiPixelMapping1 is the pixel order for the first half
	// a_pauiPixelMapping2 is the pixel order for the second half
	// a_uiRadius is the amount to vary the base colors
	//
	DifferentialTrys::DifferentialTrys(ColorFloatRGBA a_frgbaColor1, ColorFloatRGBA a_frgbaColor2,
										const unsigned int *a_pauiPixelMapping1,
										const unsigned int *a_pauiPixelMapping2,
										unsigned int a_uiRadius,
										int a_iGrayOffset1, int a_iGrayOffset2)
	{
		assert(a_uiRadius <= MAX_RADIUS);

		m_boolSeverelyBentColors = false;

		ColorFloatRGBA frgbaQuantizedColor1 = a_frgbaColor1.QuantizeR5G5B5();
		ColorFloatRGBA frgbaQuantizedColor2 = a_frgbaColor2.QuantizeR5G5B5();

		// quantize base colors
		// ensure that trys with a_uiRadius don't overflow
		int iRed1 = MoveAwayFromEdge(frgbaQuantizedColor1.IntRed(31.0f)+a_iGrayOffset1, a_uiRadius);
		int iGreen1 = MoveAwayFromEdge(frgbaQuantizedColor1.IntGreen(31.0f) + a_iGrayOffset1, a_uiRadius);
		int iBlue1 = MoveAwayFromEdge(frgbaQuantizedColor1.IntBlue(31.0f) + a_iGrayOffset1, a_uiRadius);
		int iRed2 = MoveAwayFromEdge(frgbaQuantizedColor2.IntRed(31.0f) + a_iGrayOffset2, a_uiRadius);
		int iGreen2 = MoveAwayFromEdge(frgbaQuantizedColor2.IntGreen(31.0f) + a_iGrayOffset2, a_uiRadius);
		int iBlue2 = MoveAwayFromEdge(frgbaQuantizedColor2.IntBlue(31.0f) + a_iGrayOffset2, a_uiRadius);

		int iDeltaRed = iRed2 - iRed1;
		int iDeltaGreen = iGreen2 - iGreen1;
		int iDeltaBlue = iBlue2 - iBlue1;

		// make sure components are within range
		{
			if (iDeltaRed > 3)
			{
				if (iDeltaRed > 7)
				{
					m_boolSeverelyBentColors = true;
				}

				iRed1 += (iDeltaRed - 3) / 2;
				iRed2 = iRed1 + 3;
				iDeltaRed = 3;
			}
			else if (iDeltaRed < -4)
			{
				if (iDeltaRed < -8)
				{
					m_boolSeverelyBentColors = true;
				}

				iRed1 += (iDeltaRed + 4) / 2;
				iRed2 = iRed1 - 4;
				iDeltaRed = -4;
			}
			assert(iRed1 >= (signed)(0 + a_uiRadius) && iRed1 <= (signed)(31 - a_uiRadius));
			assert(iRed2 >= (signed)(0 + a_uiRadius) && iRed2 <= (signed)(31 - a_uiRadius));
			assert(iDeltaRed >= -4 && iDeltaRed <= 3);

			if (iDeltaGreen > 3)
			{
				if (iDeltaGreen > 7)
				{
					m_boolSeverelyBentColors = true;
				}

				iGreen1 += (iDeltaGreen - 3) / 2;
				iGreen2 = iGreen1 + 3;
				iDeltaGreen = 3;
			}
			else if (iDeltaGreen < -4)
			{
				if (iDeltaGreen < -8)
				{
					m_boolSeverelyBentColors = true;
				}

				iGreen1 += (iDeltaGreen + 4) / 2;
				iGreen2 = iGreen1 - 4;
				iDeltaGreen = -4;
			}
			assert(iGreen1 >= (signed)(0 + a_uiRadius) && iGreen1 <= (signed)(31 - a_uiRadius));
			assert(iGreen2 >= (signed)(0 + a_uiRadius) && iGreen2 <= (signed)(31 - a_uiRadius));
			assert(iDeltaGreen >= -4 && iDeltaGreen <= 3);

			if (iDeltaBlue > 3)
			{
				if (iDeltaBlue > 7)
				{
					m_boolSeverelyBentColors = true;
				}

				iBlue1 += (iDeltaBlue - 3) / 2;
				iBlue2 = iBlue1 + 3;
				iDeltaBlue = 3;
			}
			else if (iDeltaBlue < -4)
			{
				if (iDeltaBlue < -8)
				{
					m_boolSeverelyBentColors = true;
				}

				iBlue1 += (iDeltaBlue + 4) / 2;
				iBlue2 = iBlue1 - 4;
				iDeltaBlue = -4;
			}
			assert(iBlue1 >= (signed)(0+a_uiRadius) && iBlue1 <= (signed)(31 - a_uiRadius));
			assert(iBlue2 >= (signed)(0 + a_uiRadius) && iBlue2 <= (signed)(31 - a_uiRadius));
			assert(iDeltaBlue >= -4 && iDeltaBlue <= 3);
		}

		m_half1.Init(iRed1, iGreen1, iBlue1, a_pauiPixelMapping1, a_uiRadius);
		m_half2.Init(iRed2, iGreen2, iBlue2, a_pauiPixelMapping2, a_uiRadius);

	}

	// ----------------------------------------------------------------------------------------------------
	//
	void DifferentialTrys::Half::Init(int a_iRed, int a_iGreen, int a_iBlue, 
										const unsigned int *a_pauiPixelMapping, unsigned int a_uiRadius)
	{

		m_iRed = a_iRed;
		m_iGreen = a_iGreen;
		m_iBlue = a_iBlue;

		m_pauiPixelMapping = a_pauiPixelMapping;
		m_uiRadius = a_uiRadius;

		m_uiTrys = 0;

	}

	// ----------------------------------------------------------------------------------------------------
	//

} // namespace Etc
