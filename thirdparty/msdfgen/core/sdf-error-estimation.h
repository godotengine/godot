
#pragma once

#include "Vector2.hpp"
#include "Shape.h"
#include "Projection.h"
#include "Scanline.h"
#include "BitmapRef.hpp"

namespace msdfgen {

/// Analytically constructs a scanline at y evaluating fill by linear interpolation of the SDF.
void scanlineSDF(Scanline &line, const BitmapConstSection<float, 1> &sdf, const Projection &projection, double y, YAxisOrientation yAxisOrientation = MSDFGEN_Y_AXIS_DEFAULT_ORIENTATION);
void scanlineSDF(Scanline &line, const BitmapConstSection<float, 3> &sdf, const Projection &projection, double y, YAxisOrientation yAxisOrientation = MSDFGEN_Y_AXIS_DEFAULT_ORIENTATION);
void scanlineSDF(Scanline &line, const BitmapConstSection<float, 4> &sdf, const Projection &projection, double y, YAxisOrientation yAxisOrientation = MSDFGEN_Y_AXIS_DEFAULT_ORIENTATION);

/// Estimates the portion of the area that will be filled incorrectly when rendering using the SDF.
double estimateSDFError(const BitmapConstSection<float, 1> &sdf, const Shape &shape, const Projection &projection, int scanlinesPerRow, FillRule fillRule = FILL_NONZERO);
double estimateSDFError(const BitmapConstSection<float, 3> &sdf, const Shape &shape, const Projection &projection, int scanlinesPerRow, FillRule fillRule = FILL_NONZERO);
double estimateSDFError(const BitmapConstSection<float, 4> &sdf, const Shape &shape, const Projection &projection, int scanlinesPerRow, FillRule fillRule = FILL_NONZERO);

// Old version of the function API's kept for backwards compatibility
void scanlineSDF(Scanline &line, const BitmapConstSection<float, 1> &sdf, const Projection &projection, double y, bool inverseYAxis);
void scanlineSDF(Scanline &line, const BitmapConstSection<float, 3> &sdf, const Projection &projection, double y, bool inverseYAxis);
void scanlineSDF(Scanline &line, const BitmapConstSection<float, 4> &sdf, const Projection &projection, double y, bool inverseYAxis);
void scanlineSDF(Scanline &line, const BitmapConstSection<float, 1> &sdf, const Vector2 &scale, const Vector2 &translate, bool inverseYAxis, double y);
void scanlineSDF(Scanline &line, const BitmapConstSection<float, 3> &sdf, const Vector2 &scale, const Vector2 &translate, bool inverseYAxis, double y);
void scanlineSDF(Scanline &line, const BitmapConstSection<float, 4> &sdf, const Vector2 &scale, const Vector2 &translate, bool inverseYAxis, double y);
double estimateSDFError(const BitmapConstSection<float, 1> &sdf, const Shape &shape, const Vector2 &scale, const Vector2 &translate, int scanlinesPerRow, FillRule fillRule = FILL_NONZERO);
double estimateSDFError(const BitmapConstSection<float, 3> &sdf, const Shape &shape, const Vector2 &scale, const Vector2 &translate, int scanlinesPerRow, FillRule fillRule = FILL_NONZERO);
double estimateSDFError(const BitmapConstSection<float, 4> &sdf, const Shape &shape, const Vector2 &scale, const Vector2 &translate, int scanlinesPerRow, FillRule fillRule = FILL_NONZERO);

}
