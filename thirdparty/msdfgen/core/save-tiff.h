
#pragma once

#include "BitmapRef.hpp"

namespace msdfgen {

/// Saves the bitmap as an uncompressed floating-point TIFF file.
bool saveTiff(const BitmapConstRef<float, 1> &bitmap, const char *filename);
bool saveTiff(const BitmapConstRef<float, 3> &bitmap, const char *filename);
bool saveTiff(const BitmapConstRef<float, 4> &bitmap, const char *filename);

}
