
#pragma once

/*
 * MULTI-CHANNEL SIGNED DISTANCE FIELD GENERATOR v1.8 (2020-10-17)
 * ---------------------------------------------------------------
 * A utility by Viktor Chlumsky, (c) 2014 - 2020
 *
 * The technique used to generate multi-channel distance fields in this code
 * has been developed by Viktor Chlumsky in 2014 for his master's thesis,
 * "Shape Decomposition for Multi-Channel Distance Fields". It provides improved
 * quality of sharp corners in glyphs and other 2D shapes compared to monochrome
 * distance fields. To reconstruct an image of the shape, apply the median of three
 * operation on the triplet of sampled signed distance values.
 *
 */

#include "core/arithmetics.hpp"
#include "core/Vector2.h"
#include "core/Scanline.h"
#include "core/Shape.h"
#include "core/BitmapRef.hpp"
#include "core/Bitmap.h"
#include "core/bitmap-interpolation.hpp"
#include "core/pixel-conversion.hpp"
#include "core/edge-coloring.h"
#include "core/msdf-error-correction.h"
#include "core/render-sdf.h"
#include "core/rasterization.h"
#include "core/sdf-error-estimation.h"
#include "core/save-bmp.h"
#include "core/save-tiff.h"
#include "core/shape-description.h"

#define MSDFGEN_VERSION "1.8"

namespace msdfgen {

/// Generates a conventional single-channel signed distance field.
void generateSDF(const BitmapRef<float, 1> &output, const Shape &shape, double range, const Vector2 &scale, const Vector2 &translate, bool overlapSupport = true);

/// Generates a single-channel signed pseudo-distance field.
void generatePseudoSDF(const BitmapRef<float, 1> &output, const Shape &shape, double range, const Vector2 &scale, const Vector2 &translate, bool overlapSupport = true);

/// Generates a multi-channel signed distance field. Edge colors must be assigned first! (See edgeColoringSimple)
void generateMSDF(const BitmapRef<float, 3> &output, const Shape &shape, double range, const Vector2 &scale, const Vector2 &translate, double edgeThreshold = MSDFGEN_DEFAULT_ERROR_CORRECTION_THRESHOLD, bool overlapSupport = true);

/// Generates a multi-channel signed distance field with true distance in the alpha channel. Edge colors must be assigned first.
void generateMTSDF(const BitmapRef<float, 4> &output, const Shape &shape, double range, const Vector2 &scale, const Vector2 &translate, double edgeThreshold = MSDFGEN_DEFAULT_ERROR_CORRECTION_THRESHOLD, bool overlapSupport = true);

// Original simpler versions of the previous functions, which work well under normal circumstances, but cannot deal with overlapping contours.
void generateSDF_legacy(const BitmapRef<float, 1> &output, const Shape &shape, double range, const Vector2 &scale, const Vector2 &translate);
void generatePseudoSDF_legacy(const BitmapRef<float, 1> &output, const Shape &shape, double range, const Vector2 &scale, const Vector2 &translate);
void generateMSDF_legacy(const BitmapRef<float, 3> &output, const Shape &shape, double range, const Vector2 &scale, const Vector2 &translate, double edgeThreshold = MSDFGEN_DEFAULT_ERROR_CORRECTION_THRESHOLD);
void generateMTSDF_legacy(const BitmapRef<float, 4> &output, const Shape &shape, double range, const Vector2 &scale, const Vector2 &translate, double edgeThreshold = MSDFGEN_DEFAULT_ERROR_CORRECTION_THRESHOLD);

}
