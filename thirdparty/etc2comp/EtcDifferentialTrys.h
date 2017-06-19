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

#pragma once

#include "EtcColorFloatRGBA.h"

namespace Etc
{

	class DifferentialTrys
	{
	public:

		static const unsigned int MAX_RADIUS = 2;

		DifferentialTrys(ColorFloatRGBA a_frgbaColor1,
							ColorFloatRGBA a_frgbaColor2,
							const unsigned int *a_pauiPixelMapping1,
							const unsigned int *a_pauiPixelMapping2,
							unsigned int a_uiRadius,
							int a_iGrayOffset1, int a_iGrayOffset2);

		inline static int MoveAwayFromEdge(int a_i, int a_iDistance)
		{
			if (a_i < (0+ a_iDistance))
			{
				return (0 + a_iDistance);
			}
			else if (a_i > (31- a_iDistance))
			{
				return (31 - a_iDistance);
			}

			return a_i;
		}

		class Try
		{
        public :
			static const unsigned int SELECTORS = 8;	// per half

			int m_iRed;
			int m_iGreen;
			int m_iBlue;
			unsigned int m_uiCW;
			unsigned int m_auiSelectors[SELECTORS];
			float m_fError;
        };

		class Half
		{
		public:

			static const unsigned int MAX_TRYS = 125;

			void Init(int a_iRed, int a_iGreen, int a_iBlue, 
						const unsigned int *a_pauiPixelMapping,
						unsigned int a_uiRadius);

			// center of trys
			int m_iRed;
			int m_iGreen;
			int m_iBlue;

			const unsigned int *m_pauiPixelMapping;
			unsigned int m_uiRadius;

			unsigned int m_uiTrys;
			Try m_atry[MAX_TRYS];

			Try *m_ptryBest;
		};

		Half m_half1;
		Half m_half2;

		bool m_boolSeverelyBentColors;
	};

	// ----------------------------------------------------------------------------------------------------
	//

} // namespace Etc
