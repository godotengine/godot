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

#include "EtcConfig.h"
#include "EtcMath.h"

namespace Etc
{

	// ----------------------------------------------------------------------------------------------------
	// calculate the line that best fits the set of XY points contained in a_afX[] and a_afY[]
	// use a_fSlope and a_fOffset to define that line
	//
	bool Regression(float a_afX[], float a_afY[], unsigned int a_Points,
					float *a_fSlope, float *a_fOffset)
	{
		float fPoints = (float)a_Points;

		float fSumX = 0.0f;
		float fSumY = 0.0f;
		float fSumXY = 0.0f;
		float fSumX2 = 0.0f;

		for (unsigned int uiPoint = 0; uiPoint < a_Points; uiPoint++)
		{
			fSumX += a_afX[uiPoint];
			fSumY += a_afY[uiPoint];
			fSumXY += a_afX[uiPoint] * a_afY[uiPoint];
			fSumX2 += a_afX[uiPoint] * a_afX[uiPoint];
		}

		float fDivisor = fPoints*fSumX2 - fSumX*fSumX;

		// if vertical line
		if (fDivisor == 0.0f)
		{
			*a_fSlope = 0.0f;
			*a_fOffset = 0.0f;
			return true;
		}

		*a_fSlope = (fPoints*fSumXY - fSumX*fSumY) / fDivisor;
		*a_fOffset = (fSumY - (*a_fSlope)*fSumX) / fPoints;

		return false;
	}

	// ----------------------------------------------------------------------------------------------------
	//

} // namespace Etc
