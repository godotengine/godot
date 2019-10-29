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

#include <math.h>

namespace Etc
{

	inline float LogToLinear(float a_fLog)
	{
		static const float ALPHA = 0.055f;
		static const float ONE_PLUS_ALPHA = 1.0f + ALPHA;

		if (a_fLog <= 0.04045f)
		{
			return a_fLog / 12.92f;
		}
		else
		{
			return powf((a_fLog + ALPHA) / ONE_PLUS_ALPHA, 2.4f);
		}
	}

	inline float LinearToLog(float &a_fLinear)
	{
		static const float ALPHA = 0.055f;
		static const float ONE_PLUS_ALPHA = 1.0f + ALPHA;

		if (a_fLinear <= 0.0031308f)
		{
			return 12.92f * a_fLinear;
		}
		else
		{
			return ONE_PLUS_ALPHA * powf(a_fLinear, (1.0f/2.4f)) - ALPHA;
		}
	}

	class ColorR8G8B8A8
	{
	public:

		unsigned char ucR;
		unsigned char ucG;
		unsigned char ucB;
		unsigned char ucA;

	};
}
