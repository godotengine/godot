#pragma once

#include "Point2.h"

namespace Javelin {

class Bitmap {
public:
    int width;
    int height;
    unsigned char *data;

    Bitmap(int w, int h, int bytesPerPixel)
        : width(w)
        , height(h)
        , data(new unsigned char[width * height * bytesPerPixel]) {
    }

    virtual ~Bitmap() {
        delete [] data;
    }

    Point2<int> GetSize() const { return Point2<int>(width, height); }

    int GetArea() const { return width * height; }

    int GetBitmapWidth() const { return width; }

    int GetBitmapHeight() const { return height; }

    const unsigned char *GetRawData() const { return data; }
};

}
