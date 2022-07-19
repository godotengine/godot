//============================================================================

#pragma once
#include "Point2.h"
#include "ColorRgba.h"

//============================================================================

namespace Javelin
{
//============================================================================

    class PvrTcDecoder
    {
    public:
        static void DecodeRgb4Bpp(ColorRgb<unsigned char>* result, const Point2<int>& size, const void* data);
        static void DecodeRgba4Bpp(ColorRgba<unsigned char>* result, const Point2<int>& size, const void* data);
        
    private:
		static unsigned GetMortonNumber(int x, int y);
    };
    
//============================================================================
}
//============================================================================
