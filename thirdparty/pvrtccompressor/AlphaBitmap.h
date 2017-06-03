#pragma once

#include "Bitmap.h"

namespace Javelin {

class AlphaBitmap : public Bitmap {
public:
    AlphaBitmap(int w, int h)
        : Bitmap(w, h, 1) {
    }

    const unsigned char *GetData() const { return data; }

    unsigned char *GetData() { return data; }
};

}
