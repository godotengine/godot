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
EtcIndividualTrys.cpp

Gathers the results of the various encoding trys for both halves of a 4x4 block for Individual mode

*/

#include "EtcConfig.h"
#include "EtcIndividualTrys.h"

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
	IndividualTrys::IndividualTrys(ColorFloatRGBA a_frgbaColor1, ColorFloatRGBA a_frgbaColor2,
									const unsigned int *a_pauiPixelMapping1,
									const unsigned int *a_pauiPixelMapping2,
									unsigned int a_uiRadius)
	{
		assert(a_uiRadius <= MAX_RADIUS);

		ColorFloatRGBA frgbaQuantizedColor1 = a_frgbaColor1.QuantizeR4G4B4();
		ColorFloatRGBA frgbaQuantizedColor2 = a_frgbaColor2.QuantizeR4G4B4();

		// quantize base colors
		// ensure that trys with a_uiRadius don't overflow
		int iRed1 = MoveAwayFromEdge(frgbaQuantizedColor1.IntRed(15.0f), a_uiRadius);
		int iGreen1 = MoveAwayFromEdge(frgbaQuantizedColor1.IntGreen(15.0f), a_uiRadius);
		int iBlue1 = MoveAwayFromEdge(frgbaQuantizedColor1.IntBlue(15.0f), a_uiRadius);
		int iRed2 = MoveAwayFromEdge(frgbaQuantizedColor2.IntRed(15.0f), a_uiRadius);
		int iGreen2 = MoveAwayFromEdge(frgbaQuantizedColor2.IntGreen(15.0f), a_uiRadius);
		int iBlue2 = MoveAwayFromEdge(frgbaQuantizedColor2.IntBlue(15.0f), a_uiRadius);

		m_half1.Init(iRed1, iGreen1, iBlue1, a_pauiPixelMapping1, a_uiRadius);
		m_half2.Init(iRed2, iGreen2, iBlue2, a_pauiPixelMapping2, a_uiRadius);

	}

	// ----------------------------------------------------------------------------------------------------
	//
	void IndividualTrys::Half::Init(int a_iRed, int a_iGreen, int a_iBlue,
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
