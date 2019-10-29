//============================================================================
//
// Modulation data specifies weightings of colorA to colorB for each pixel
//
// For mode = 0
//	00: 0/8
//  01: 3/8
//  10: 5/8
//  11: 8/8
//
// For mode = 1
//  00: 0/8
//  01: 4/8
//  10: 4/8 with alpha punchthrough
//  11: 8/8
//
// For colorIsOpaque=0
//  3 bits A
//  4 bits R
//  4 bits G
//  3/4 bits B
//
// For colorIsOpaque=1
//  5 bits R
//  5 bits G
//  4/5 bits B
//
//============================================================================

#pragma once
#include "ColorRgba.h"

//============================================================================

namespace Javelin
{
//============================================================================

	struct PvrTcPacket
	{
		unsigned int    modulationData;
		unsigned        usePunchthroughAlpha : 1;
		unsigned        colorA          	 : 14;
		unsigned        colorAIsOpaque  	 : 1;
		unsigned        colorB        		 : 15;
		unsigned        colorBIsOpaque  	 : 1;
		
		ColorRgb<int> GetColorRgbA() const;
		ColorRgb<int> GetColorRgbB() const;
		ColorRgba<int> GetColorRgbaA() const;
		ColorRgba<int> GetColorRgbaB() const;
		
		void SetColorA(const ColorRgb<unsigned char>& c);
		void SetColorB(const ColorRgb<unsigned char>& c);

		void SetColorA(const ColorRgba<unsigned char>& c);
		void SetColorB(const ColorRgba<unsigned char>& c);
		
		static const unsigned char BILINEAR_FACTORS[16][4];
		static const unsigned char WEIGHTS[8][4];
	};

//============================================================================
} // namespace Javelin
//============================================================================
