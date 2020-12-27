
#pragma once

#include "Vector2.h"
#include "BitmapRef.hpp"

#define MSDFGEN_DEFAULT_ERROR_CORRECTION_THRESHOLD 1.001

namespace msdfgen {

/// Attempts to patch multi-channel signed distance field values that may cause interpolation artifacts. (Already called by generateMSDF)
void msdfErrorCorrection(const BitmapRef<float, 3> &output, const Vector2 &threshold);
void msdfErrorCorrection(const BitmapRef<float, 4> &output, const Vector2 &threshold);

}
