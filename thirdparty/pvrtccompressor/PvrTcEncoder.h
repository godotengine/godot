//============================================================================

#pragma once
#include "ColorRgba.h"

//============================================================================

namespace Javelin
{
//============================================================================

	class AlphaBitmap;
	class RgbBitmap;
	class RgbaBitmap;
	
	class PvrTcEncoder
	{
	public:
		// Result must be large enough for bitmap.GetArea()/4 bytes
		static void EncodeAlpha2Bpp(void* result, const AlphaBitmap& bitmap);
		
		// Result must be large enough for bitmap.GetArea()/2 bytes
		static void EncodeAlpha4Bpp(void* result, const AlphaBitmap& bitmap);
		
		// Result must be large enough for bitmap.GetArea()/2 bytes
		static void EncodeRgb4Bpp(void* result, const RgbBitmap& bitmap);

		// Result must be large enough for bitmap.GetArea()/2 bytes
		static void EncodeRgb4Bpp(void* result, const RgbaBitmap& bitmap);

		// Result must be large enough for bitmap.GetArea()/2 bytes
		static void EncodeRgba4Bpp(void* result, const RgbaBitmap& bitmap);

	private:
		static unsigned GetMortonNumber(int x, int y);
	};
	
//============================================================================
}
//============================================================================
