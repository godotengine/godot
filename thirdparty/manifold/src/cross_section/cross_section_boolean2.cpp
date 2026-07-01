// Copyright 2026 The Manifold Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "manifold/cross_section.h"

// Temporary backend-selection placeholder. The boolean2 implementation lands
// in the follow-up backend PR; this file intentionally returns empty
// CrossSection results so the staged backend cannot appear partially correct.

using namespace manifold;

namespace manifold {
struct PathImpl {
  PathImpl(Polygons paths) : paths_(std::move(paths)) {}
  operator const Polygons&() const { return paths_; }
  const Polygons paths_;
};
}  // namespace manifold

namespace {
std::shared_ptr<const PathImpl> empty_paths() {
  return std::make_shared<const PathImpl>(Polygons{});
}
}  // namespace

namespace manifold {

CrossSection::CrossSection() : paths_(empty_paths()) {}

CrossSection::~CrossSection() = default;
CrossSection::CrossSection(CrossSection&&) noexcept = default;
CrossSection& CrossSection::operator=(CrossSection&&) noexcept = default;

CrossSection::CrossSection(const CrossSection& other) {
  (void)other;
  paths_ = empty_paths();
}

CrossSection& CrossSection::operator=(const CrossSection& other) {
  (void)other;
  paths_ = empty_paths();
  transform_ = la::identity;
  return *this;
}

CrossSection::CrossSection(std::shared_ptr<const PathImpl> paths)
    : paths_(empty_paths()) {
  (void)paths;
}

CrossSection::CrossSection(const SimplePolygon& contour, FillRule fillrule) {
  (void)contour;
  (void)fillrule;
  paths_ = empty_paths();
}

CrossSection::CrossSection(const Polygons& contours, FillRule fillrule) {
  (void)contours;
  (void)fillrule;
  paths_ = empty_paths();
}

CrossSection::CrossSection(const Rect& rect) {
  (void)rect;
  paths_ = empty_paths();
}

std::shared_ptr<const PathImpl> CrossSection::GetPaths() const {
  std::lock_guard<std::mutex> lock(*pathsMutex_);
  return paths_;
}

CrossSection CrossSection::Square(const vec2 size, bool center) {
  (void)size;
  (void)center;
  return CrossSection();
}

CrossSection CrossSection::Circle(double radius, int circularSegments) {
  (void)radius;
  (void)circularSegments;
  return CrossSection();
}

CrossSection CrossSection::Boolean(const CrossSection& second,
                                   OpType op) const {
  (void)second;
  (void)op;
  return CrossSection();
}

CrossSection CrossSection::BatchBoolean(
    const std::vector<CrossSection>& crossSections, OpType op) {
  (void)crossSections;
  (void)op;
  return CrossSection();
}

CrossSection CrossSection::operator+(const CrossSection& other) const {
  return Boolean(other, OpType::Add);
}

CrossSection& CrossSection::operator+=(const CrossSection& other) {
  *this = *this + other;
  return *this;
}

CrossSection CrossSection::operator-(const CrossSection& other) const {
  return Boolean(other, OpType::Subtract);
}

CrossSection& CrossSection::operator-=(const CrossSection& other) {
  *this = *this - other;
  return *this;
}

CrossSection CrossSection::operator^(const CrossSection& other) const {
  return Boolean(other, OpType::Intersect);
}

CrossSection& CrossSection::operator^=(const CrossSection& other) {
  *this = *this ^ other;
  return *this;
}

CrossSection CrossSection::Compose(
    const std::vector<CrossSection>& crossSections) {
  (void)crossSections;
  return CrossSection();
}

std::vector<CrossSection> CrossSection::Decompose() const { return {}; }

CrossSection CrossSection::Translate(const vec2 v) const {
  (void)v;
  return CrossSection();
}

CrossSection CrossSection::Rotate(double degrees) const {
  (void)degrees;
  return CrossSection();
}

CrossSection CrossSection::Scale(const vec2 scale) const {
  (void)scale;
  return CrossSection();
}

CrossSection CrossSection::Mirror(const vec2 ax) const {
  (void)ax;
  return CrossSection();
}

CrossSection CrossSection::Transform(const mat2x3& m) const {
  (void)m;
  return CrossSection();
}

CrossSection CrossSection::Warp(std::function<void(vec2&)> warpFunc) const {
  (void)warpFunc;
  return CrossSection();
}

CrossSection CrossSection::WarpBatch(
    std::function<void(VecView<vec2>)> warpFunc) const {
  (void)warpFunc;
  return CrossSection();
}

CrossSection CrossSection::Simplify(double epsilon) const {
  (void)epsilon;
  return CrossSection();
}

CrossSection CrossSection::Offset(double delta, JoinType jt, double miterLimit,
                                  int circularSegments) const {
  (void)delta;
  (void)jt;
  (void)miterLimit;
  (void)circularSegments;
  return CrossSection();
}

CrossSection CrossSection::Hull(
    const std::vector<CrossSection>& crossSections) {
  (void)crossSections;
  return CrossSection();
}

CrossSection CrossSection::Hull() const { return CrossSection(); }

CrossSection CrossSection::Hull(SimplePolygon pts) {
  (void)pts;
  return CrossSection();
}

CrossSection CrossSection::Hull(const Polygons polys) {
  (void)polys;
  return CrossSection();
}

double CrossSection::Area() const { return 0.0; }

size_t CrossSection::NumVert() const { return 0; }

size_t CrossSection::NumContour() const { return 0; }

bool CrossSection::IsEmpty() const { return true; }

Rect CrossSection::Bounds() const { return {}; }

Polygons CrossSection::ToPolygons() const { return {}; }

}  // namespace manifold
