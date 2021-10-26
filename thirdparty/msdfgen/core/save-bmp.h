
#pragma once

#include "BitmapRef.hpp"

namespace msdfgen {

/// Saves the bitmap as a BMP file.
bool saveBmp(const BitmapConstRef<byte, 1> &bitmap, const char *filename);
bool saveBmp(const BitmapConstRef<byte, 3> &bitmap, const char *filename);
bool saveBmp(const BitmapConstRef<byte, 4> &bitmap, const char *filename);
bool saveBmp(const BitmapConstRef<float, 1> &bitmap, const char *filename);
bool saveBmp(const BitmapConstRef<float, 3> &bitmap, const char *filename);
bool saveBmp(const BitmapConstRef<float, 4> &bitmap, const char *filename);

}
