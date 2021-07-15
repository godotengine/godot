
#pragma once

#include "Vector2.h"
#include "Scanline.h"
#include "Shape.h"
#include "BitmapRef.hpp"

namespace msdfgen {

/// Analytically constructs a scanline at y evaluating fill by linear interpolation of the SDF.
void scanlineSDF(Scanline &line, const BitmapConstRef<float, 1> &sdf, const Vector2 &scale, const Vector2 &translate, bool inverseYAxis, double y);
void scanlineSDF(Scanline &line, const BitmapConstRef<float, 3> &sdf, const Vector2 &scale, const Vector2 &translate, bool inverseYAxis, double y);
void scanlineSDF(Scanline &line, const BitmapConstRef<float, 4> &sdf, const Vector2 &scale, const Vector2 &translate, bool inverseYAxis, double y);

/// Estimates the portion of the area that will be filled incorrectly when rendering using the SDF.
double estimateSDFError(const BitmapConstRef<float, 1> &sdf, const Shape &shape, const Vector2 &scale, const Vector2 &translate, int scanlinesPerRow, FillRule fillRule = FILL_NONZERO);
double estimateSDFError(const BitmapConstRef<float, 3> &sdf, const Shape &shape, const Vector2 &scale, const Vector2 &translate, int scanlinesPerRow, FillRule fillRule = FILL_NONZERO);
double estimateSDFError(const BitmapConstRef<float, 4> &sdf, const Shape &shape, const Vector2 &scale, const Vector2 &translate, int scanlinesPerRow, FillRule fillRule = FILL_NONZERO);

}
