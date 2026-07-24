// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#ifndef LIB_JXL_SPLINES_H_
#define LIB_JXL_SPLINES_H_

#include <jxl/memory_manager.h>

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

#include "lib/jxl/base/compiler_specific.h"
#include "lib/jxl/base/rect.h"
#include "lib/jxl/base/status.h"
#include "lib/jxl/chroma_from_luma.h"
#include "lib/jxl/image.h"

namespace jxl {

class ANSSymbolReader;
class BitReader;

static constexpr float kDesiredRenderingDistance = 1.f;

typedef std::array<float, 32> Dct32;

enum SplineEntropyContexts : size_t {
  kQuantizationAdjustmentContext = 0,
  kStartingPositionContext,
  kNumSplinesContext,
  kNumControlPointsContext,
  kControlPointsContext,
  kDCTContext,
  kNumSplineContexts
};

struct Spline {
  struct Point {
    Point() : x(0.0f), y(0.0f) {}
    Point(float x, float y) : x(x), y(y) {}
    float x, y;
    bool operator==(const Point& other) const {
      return std::fabs(x - other.x) < 1e-3f && std::fabs(y - other.y) < 1e-3f;
    }
  };
  std::vector<Point> control_points;
  // X, Y, B.
  std::array<Dct32, 3> color_dct;
  // Splines are draws by normalized Gaussian splatting. This controls the
  // Gaussian's parameter along the spline.
  Dct32 sigma_dct;
};

class QuantizedSplineEncoder;

class QuantizedSpline {
 public:
  QuantizedSpline() = default;

  // TODO(eustas): move this out of library code
  static StatusOr<QuantizedSpline> Create(const Spline& original,
                                          int32_t quantization_adjustment,
                                          float y_to_x, float y_to_b);

  Status Dequantize(const Spline::Point& starting_point,
                    int32_t quantization_adjustment, float y_to_x, float y_to_b,
                    uint64_t image_size, uint64_t* total_estimated_area_reached,
                    Spline& result) const;

  Status Decode(const std::vector<uint8_t>& context_map,
                ANSSymbolReader* decoder, BitReader* br,
                size_t max_control_points, size_t* total_num_control_points);

 private:
  friend class QuantizedSplineEncoder;

  std::vector<std::pair<int64_t, int64_t>>
      control_points_;  // Double delta-encoded.
  int color_dct_[3][32] = {};
  int sigma_dct_[32] = {};
};

// A single "drawable unit" of a spline, i.e. a line of the region in which we
// render each Gaussian. The structure doesn't actually depend on the exact
// row, which allows reuse for different y values (which are tracked
// separately).
struct SplineSegment {
  float center_x, center_y;
  float maximum_distance;
  float inv_sigma;
  float sigma_over_4_times_intensity;
  float color[3];
};

class Splines {
 public:
  Splines() = default;
  explicit Splines(const int32_t quantization_adjustment,
                   std::vector<QuantizedSpline> splines,
                   std::vector<Spline::Point> starting_points)
      : quantization_adjustment_(quantization_adjustment),
        splines_(std::move(splines)),
        starting_points_(std::move(starting_points)) {}

  bool HasAny() const { return !splines_.empty(); }

  void Clear();

  Status Decode(JxlMemoryManager* memory_manager, BitReader* br,
                size_t num_pixels);

  void AddTo(Image3F* opsin, const Rect& opsin_rect) const;
  void AddToRow(float* JXL_RESTRICT row_x, float* JXL_RESTRICT row_y,
                float* JXL_RESTRICT row_b, size_t y, size_t x0,
                size_t x1) const;
  void SubtractFrom(Image3F* opsin) const;

  const std::vector<QuantizedSpline>& QuantizedSplines() const {
    return splines_;
  }
  const std::vector<Spline::Point>& StartingPoints() const {
    return starting_points_;
  }

  int32_t GetQuantizationAdjustment() const { return quantization_adjustment_; }

  Status InitializeDrawCache(size_t image_xsize, size_t image_ysize,
                             const ColorCorrelation& color_correlation);

 private:
  template <bool>
  void ApplyToRow(float* JXL_RESTRICT row_x, float* JXL_RESTRICT row_y,
                  float* JXL_RESTRICT row_b, size_t y, size_t x0,
                  size_t x1) const;
  template <bool>
  void Apply(Image3F* opsin, const Rect& opsin_rect) const;

  // If positive, quantization weights are multiplied by 1 + this/8, which
  // increases precision. If negative, they are divided by 1 - this/8. If 0,
  // they are unchanged.
  int32_t quantization_adjustment_ = 0;
  std::vector<QuantizedSpline> splines_;
  std::vector<Spline::Point> starting_points_;
  std::vector<SplineSegment> segments_;
  std::vector<size_t> segment_indices_;
  std::vector<size_t> segment_y_start_;
};

}  // namespace jxl

#endif  // LIB_JXL_SPLINES_H_
