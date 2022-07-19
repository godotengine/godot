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

#include "EtcConfig.h"
#include "EtcColor.h"

#include <math.h>

namespace Etc
{

	class ColorFloatRGBA
    {
    public:

		ColorFloatRGBA(void)
        {
            fR = fG = fB = fA = 0.0f;
        }

		ColorFloatRGBA(float a_fR, float a_fG, float a_fB, float a_fA)
        {
            fR = a_fR;
            fG = a_fG;
            fB = a_fB;
            fA = a_fA;
        }

		inline ColorFloatRGBA operator+(ColorFloatRGBA& a_rfrgba)
		{
			ColorFloatRGBA frgba;
			frgba.fR = fR + a_rfrgba.fR;
			frgba.fG = fG + a_rfrgba.fG;
			frgba.fB = fB + a_rfrgba.fB;
			frgba.fA = fA + a_rfrgba.fA;
			return frgba;
		}

		inline ColorFloatRGBA operator+(float a_f)
		{
			ColorFloatRGBA frgba;
			frgba.fR = fR + a_f;
			frgba.fG = fG + a_f;
			frgba.fB = fB + a_f;
			frgba.fA = fA;
			return frgba;
		}

		inline ColorFloatRGBA operator-(float a_f)
		{
			ColorFloatRGBA frgba;
			frgba.fR = fR - a_f;
			frgba.fG = fG - a_f;
			frgba.fB = fB - a_f;
			frgba.fA = fA;
			return frgba;
		}

		inline ColorFloatRGBA operator-(ColorFloatRGBA& a_rfrgba)
		{
			ColorFloatRGBA frgba;
			frgba.fR = fR - a_rfrgba.fR;
			frgba.fG = fG - a_rfrgba.fG;
			frgba.fB = fB - a_rfrgba.fB;
			frgba.fA = fA - a_rfrgba.fA;
			return frgba;
		}

		inline ColorFloatRGBA operator*(float a_f)
		{
			ColorFloatRGBA frgba;
			frgba.fR = fR * a_f;
			frgba.fG = fG * a_f;
			frgba.fB = fB * a_f;
			frgba.fA = fA;

			return frgba;
		}

		inline ColorFloatRGBA ScaleRGB(float a_f)
		{
			ColorFloatRGBA frgba;
			frgba.fR = a_f * fR;
			frgba.fG = a_f * fG;
			frgba.fB = a_f * fB;
			frgba.fA = fA;

			return frgba;
		}

		inline ColorFloatRGBA RoundRGB(void)
		{
			ColorFloatRGBA frgba;
			frgba.fR = roundf(fR);
			frgba.fG = roundf(fG);
			frgba.fB = roundf(fB);

			return frgba;
		}

		inline ColorFloatRGBA ToLinear()
		{
			ColorFloatRGBA frgbaLinear;
			frgbaLinear.fR = LogToLinear(fR);
			frgbaLinear.fG = LogToLinear(fG);
			frgbaLinear.fB = LogToLinear(fB);
			frgbaLinear.fA = fA;

			return frgbaLinear;
		}

		inline ColorFloatRGBA ToLog(void)
		{
			ColorFloatRGBA frgbaLog;
			frgbaLog.fR = LinearToLog(fR);
			frgbaLog.fG = LinearToLog(fG);
			frgbaLog.fB = LinearToLog(fB);
			frgbaLog.fA = fA;

			return frgbaLog;
		}

		inline static ColorFloatRGBA ConvertFromRGBA8(unsigned char a_ucR, 
			unsigned char a_ucG, unsigned char a_ucB, unsigned char a_ucA)
		{
			ColorFloatRGBA frgba;

			frgba.fR = (float)a_ucR / 255.0f;
			frgba.fG = (float)a_ucG / 255.0f;
			frgba.fB = (float)a_ucB / 255.0f;
			frgba.fA = (float)a_ucA / 255.0f;

			return frgba;
		}

		inline static ColorFloatRGBA ConvertFromRGB4(unsigned char a_ucR4,
														unsigned char a_ucG4,
														unsigned char a_ucB4)
		{
			ColorFloatRGBA frgba;

			unsigned char ucR8 = (unsigned char)((a_ucR4 << 4) + a_ucR4);
			unsigned char ucG8 = (unsigned char)((a_ucG4 << 4) + a_ucG4);
			unsigned char ucB8 = (unsigned char)((a_ucB4 << 4) + a_ucB4);

			frgba.fR = (float)ucR8 / 255.0f;
			frgba.fG = (float)ucG8 / 255.0f;
			frgba.fB = (float)ucB8 / 255.0f;
			frgba.fA = 1.0f;

			return frgba;
		}

		inline static ColorFloatRGBA ConvertFromRGB5(unsigned char a_ucR5,
			unsigned char a_ucG5,
			unsigned char a_ucB5)
		{
			ColorFloatRGBA frgba;

			unsigned char ucR8 = (unsigned char)((a_ucR5 << 3) + (a_ucR5 >> 2));
			unsigned char ucG8 = (unsigned char)((a_ucG5 << 3) + (a_ucG5 >> 2));
			unsigned char ucB8 = (unsigned char)((a_ucB5 << 3) + (a_ucB5 >> 2));

			frgba.fR = (float)ucR8 / 255.0f;
			frgba.fG = (float)ucG8 / 255.0f;
			frgba.fB = (float)ucB8 / 255.0f;
			frgba.fA = 1.0f;

			return frgba;
		}

		inline static ColorFloatRGBA ConvertFromR6G7B6(unsigned char a_ucR6,
			unsigned char a_ucG7,
			unsigned char a_ucB6)
		{
			ColorFloatRGBA frgba;

			unsigned char ucR8 = (unsigned char)((a_ucR6 << 2) + (a_ucR6 >> 4));
			unsigned char ucG8 = (unsigned char)((a_ucG7 << 1) + (a_ucG7 >> 6));
			unsigned char ucB8 = (unsigned char)((a_ucB6 << 2) + (a_ucB6 >> 4));

			frgba.fR = (float)ucR8 / 255.0f;
			frgba.fG = (float)ucG8 / 255.0f;
			frgba.fB = (float)ucB8 / 255.0f;
			frgba.fA = 1.0f;

			return frgba;
		}

		// quantize to 4 bits, expand to 8 bits
		inline ColorFloatRGBA QuantizeR4G4B4(void) const
		{
			ColorFloatRGBA frgba = *this;

			// quantize to 4 bits
			frgba = frgba.ClampRGB().ScaleRGB(15.0f).RoundRGB();
			unsigned int uiR4 = (unsigned int)frgba.fR;
			unsigned int uiG4 = (unsigned int)frgba.fG;
			unsigned int uiB4 = (unsigned int)frgba.fB;

			// expand to 8 bits
			frgba.fR = (float) ((uiR4 << 4) + uiR4);
			frgba.fG = (float) ((uiG4 << 4) + uiG4);
			frgba.fB = (float) ((uiB4 << 4) + uiB4);

			frgba = frgba.ScaleRGB(1.0f/255.0f);

			return frgba;
		}

		// quantize to 5 bits, expand to 8 bits
		inline ColorFloatRGBA QuantizeR5G5B5(void) const
		{
			ColorFloatRGBA frgba = *this;

			// quantize to 5 bits
			frgba = frgba.ClampRGB().ScaleRGB(31.0f).RoundRGB();
			unsigned int uiR5 = (unsigned int)frgba.fR;
			unsigned int uiG5 = (unsigned int)frgba.fG;
			unsigned int uiB5 = (unsigned int)frgba.fB;

			// expand to 8 bits
			frgba.fR = (float)((uiR5 << 3) + (uiR5 >> 2));
			frgba.fG = (float)((uiG5 << 3) + (uiG5 >> 2));
			frgba.fB = (float)((uiB5 << 3) + (uiB5 >> 2));

			frgba = frgba.ScaleRGB(1.0f / 255.0f);

			return frgba;
		}

		// quantize to 6/7/6 bits, expand to 8 bits
		inline ColorFloatRGBA QuantizeR6G7B6(void) const
		{
			ColorFloatRGBA frgba = *this;

			// quantize to 6/7/6 bits
			ColorFloatRGBA frgba6 = frgba.ClampRGB().ScaleRGB(63.0f).RoundRGB();
			ColorFloatRGBA frgba7 = frgba.ClampRGB().ScaleRGB(127.0f).RoundRGB();
			unsigned int uiR6 = (unsigned int)frgba6.fR;
			unsigned int uiG7 = (unsigned int)frgba7.fG;
			unsigned int uiB6 = (unsigned int)frgba6.fB;

			// expand to 8 bits
			frgba.fR = (float)((uiR6 << 2) + (uiR6 >> 4));
			frgba.fG = (float)((uiG7 << 1) + (uiG7 >> 6));
			frgba.fB = (float)((uiB6 << 2) + (uiB6 >> 4));

			frgba = frgba.ScaleRGB(1.0f / 255.0f);

			return frgba;
		}

		inline ColorFloatRGBA ClampRGB(void)
		{
			ColorFloatRGBA frgba = *this;
			if (frgba.fR < 0.0f) { frgba.fR = 0.0f; }
			if (frgba.fR > 1.0f) { frgba.fR = 1.0f; }
			if (frgba.fG < 0.0f) { frgba.fG = 0.0f; }
			if (frgba.fG > 1.0f) { frgba.fG = 1.0f; }
			if (frgba.fB < 0.0f) { frgba.fB = 0.0f; }
			if (frgba.fB > 1.0f) { frgba.fB = 1.0f; }

			return frgba;
		}

		inline ColorFloatRGBA ClampRGBA(void)
		{
			ColorFloatRGBA frgba = *this;
			if (frgba.fR < 0.0f) { frgba.fR = 0.0f; }
			if (frgba.fR > 1.0f) { frgba.fR = 1.0f; }
			if (frgba.fG < 0.0f) { frgba.fG = 0.0f; }
			if (frgba.fG > 1.0f) { frgba.fG = 1.0f; }
			if (frgba.fB < 0.0f) { frgba.fB = 0.0f; }
			if (frgba.fB > 1.0f) { frgba.fB = 1.0f; }
			if (frgba.fA < 0.0f) { frgba.fA = 0.0f; }
			if (frgba.fA > 1.0f) { frgba.fA = 1.0f; }

			return frgba;
		}

		inline int IntRed(float a_fScale)
		{
			return (int)roundf(fR * a_fScale);
		}

		inline int IntGreen(float a_fScale)
		{
			return (int)roundf(fG * a_fScale);
		}

		inline int IntBlue(float a_fScale)
		{
			return (int)roundf(fB * a_fScale);
		}

		inline int IntAlpha(float a_fScale)
		{
			return (int)roundf(fA * a_fScale);
		}

		float	fR, fG, fB, fA;
    };

}

