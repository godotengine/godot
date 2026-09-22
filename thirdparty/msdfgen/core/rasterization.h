
#pragma once

#include "Vector2.hpp"
#include "Shape.h"
#include "Projection.h"
#include "Scanline.h"
#include "BitmapRef.hpp"

namespace msdfgen {

/// Rasterizes the shape into a monochrome bitmap.
void rasterize(BitmapSection<float, 1> output, const Shape &shape, const Projection &projection, FillRule fillRule = FILL_NONZERO);
/// Fixes the sign of the input signed distance field, so that it matches the shape's rasterized fill.
void distanceSignCorrection(BitmapSection<float, 1> sdf, const Shape &shape, const Projection &projection, float sdfZeroValue = .5f, FillRule fillRule = FILL_NONZERO);
void distanceSignCorrection(BitmapSection<float, 3> sdf, const Shape &shape, const Projection &projection, float sdfZeroValue = .5f, FillRule fillRule = FILL_NONZERO);
void distanceSignCorrection(BitmapSection<float, 4> sdf, const Shape &shape, const Projection &projection, float sdfZeroValue = .5f, FillRule fillRule = FILL_NONZERO);

// Old versions of the function API's kept for backwards compatibility
void rasterize(const BitmapSection<float, 1> &output, const Shape &shape, const Vector2 &scale, const Vector2 &translate, FillRule fillRule = FILL_NONZERO);
void distanceSignCorrection(BitmapSection<float, 1> sdf, const Shape &shape, const Projection &projection, FillRule fillRule);
void distanceSignCorrection(BitmapSection<float, 3> sdf, const Shape &shape, const Projection &projection, FillRule fillRule);
void distanceSignCorrection(BitmapSection<float, 4> sdf, const Shape &shape, const Projection &projection, FillRule fillRule);
void distanceSignCorrection(const BitmapSection<float, 1> &sdf, const Shape &shape, const Vector2 &scale, const Vector2 &translate, FillRule fillRule = FILL_NONZERO);
void distanceSignCorrection(const BitmapSection<float, 3> &sdf, const Shape &shape, const Vector2 &scale, const Vector2 &translate, FillRule fillRule = FILL_NONZERO);
void distanceSignCorrection(const BitmapSection<float, 4> &sdf, const Shape &shape, const Vector2 &scale, const Vector2 &translate, FillRule fillRule = FILL_NONZERO);

}
