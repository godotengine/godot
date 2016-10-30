#pragma once

#include "Bitmap.h"
#include "ColorRgba.h"

namespace Javelin {

class RgbBitmap : public Bitmap {
public:
    RgbBitmap(int w, int h)
        : Bitmap(w, h, 3) {
    }

    const ColorRgb<unsigned char> *GetData() const { 
        return reinterpret_cast<ColorRgb<unsigned char> *>(data); 
    }

    ColorRgb<unsigned char> *GetData() { 
        return reinterpret_cast<ColorRgb<unsigned char> *>(data); 
    }
};

}
