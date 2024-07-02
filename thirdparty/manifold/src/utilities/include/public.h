// Copyright 2021 The Manifold Authors.
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

#pragma once
#define GLM_ENABLE_EXPERIMENTAL  // needed for glm/gtx/compatibility.hpp
#define GLM_FORCE_EXPLICIT_CTOR
#include <glm/ext/matrix_transform.hpp>
#include <glm/glm.hpp>
#include <glm/gtc/constants.hpp>
#include <glm/gtx/compatibility.hpp>
#include <glm/gtx/rotate_vector.hpp>
#include <limits>
#include <memory>
#include <unordered_map>
#include <vector>

#ifdef MANIFOLD_DEBUG
#include <iomanip>
#include <iostream>
#include <sstream>
#endif

constexpr std::size_t operator""_z(unsigned long long n) { return n; }

namespace manifold {

constexpr float kTolerance = 1e-5;

/** @defgroup Connections
 *  @brief Move data in and out of the Manifold class.
 *  @{
 */

/**
 * Sine function where multiples of 90 degrees come out exact.
 *
 * @param x Angle in degrees.
 */
inline float sind(float x) {
  if (!std::isfinite(x)) return sin(x);
  if (x < 0.0f) return -sind(-x);
  int quo;
  x = remquo(fabs(x), 90.0f, &quo);
  switch (quo % 4) {
    case 0:
      return sin(glm::radians(x));
    case 1:
      return cos(glm::radians(x));
    case 2:
      return -sin(glm::radians(x));
    case 3:
      return -cos(glm::radians(x));
  }
  return 0.0f;
}

/**
 * Cosine function where multiples of 90 degrees come out exact.
 *
 * @param x Angle in degrees.
 */
inline float cosd(float x) { return sind(x + 90.0f); }

/**
 * This 4x3 matrix can be used as an input to Manifold.Transform() to turn an
 * object. Turns along the shortest path from given up-vector to (0, 0, 1).
 *
 * @param up The vector to be turned to point upwards. Length does not matter.
 */
inline glm::mat4x3 RotateUp(glm::vec3 up) {
  up = glm::normalize(up);
  glm::vec3 axis = glm::cross(up, {0, 0, 1});
  float angle = glm::asin(glm::length(axis));
  if (glm::dot(up, {0, 0, 1}) < 0) angle = glm::pi<float>() - angle;
  return glm::mat4x3(glm::rotate(glm::mat4(1), angle, axis));
}

/**
 * Determines if the three points are wound counter-clockwise, clockwise, or
 * colinear within the specified tolerance.
 *
 * @param p0 First point
 * @param p1 Second point
 * @param p2 Third point
 * @param tol Tolerance value for colinearity
 * @return int, like Signum, this returns 1 for CCW, -1 for CW, and 0 if within
 * tol of colinear.
 */
inline int CCW(glm::vec2 p0, glm::vec2 p1, glm::vec2 p2, float tol) {
  glm::vec2 v1 = p1 - p0;
  glm::vec2 v2 = p2 - p0;
  float area = v1.x * v2.y - v1.y * v2.x;
  float base2 = glm::max(glm::dot(v1, v1), glm::dot(v2, v2));
  if (area * area * 4 <= base2 * tol * tol)
    return 0;
  else
    return area > 0 ? 1 : -1;
}

/**
 * Single polygon contour, wound CCW. First and last point are implicitly
 * connected. Should ensure all input is
 * [&epsilon;-valid](https://github.com/elalish/manifold/wiki/Manifold-Library#definition-of-%CE%B5-valid).
 */
using SimplePolygon = std::vector<glm::vec2>;

/**
 * Set of polygons with holes. Order of contours is arbitrary. Can contain any
 * depth of nested holes and any number of separate polygons. Should ensure all
 * input is
 * [&epsilon;-valid](https://github.com/elalish/manifold/wiki/Manifold-Library#definition-of-%CE%B5-valid).
 */
using Polygons = std::vector<SimplePolygon>;

/**
 * The triangle-mesh input and output of this library.
 */
struct Mesh {
  /// Required: The X-Y-Z positions of all vertices.
  std::vector<glm::vec3> vertPos;
  /// Required: The vertex indices of the three triangle corners in CCW (from
  /// the outside) order, for each triangle.
  std::vector<glm::ivec3> triVerts;
  /// Optional: The X-Y-Z normal vectors of each vertex. If non-empty, must have
  /// the same length as vertPos. If empty, these will be calculated
  /// automatically.
  std::vector<glm::vec3> vertNormal;
  /// Optional: The X-Y-Z-W weighted tangent vectors for smooth Refine(). If
  /// non-empty, must be exactly three times as long as Mesh.triVerts. Indexed
  /// as 3 * tri + i, representing the tangent from Mesh.triVerts[tri][i] along
  /// the CCW edge. If empty, mesh is faceted.
  std::vector<glm::vec4> halfedgeTangent;
  /// The absolute precision of the vertex positions, based on accrued rounding
  /// errors. When creating a Manifold, the precision used will be the maximum
  /// of this and a baseline precision from the size of the bounding box. Any
  /// edge shorter than precision may be collapsed.
  float precision = 0;
};

/**
 * Defines which edges to sharpen and how much for the Manifold.Smooth()
 * constructor.
 */
struct Smoothness {
  /// The halfedge index = 3 * tri + i, referring to Mesh.triVerts[tri][i].
  int halfedge;
  /// A value between 0 and 1, where 0 is sharp and 1 is the default and the
  /// curvature is interpolated between these values. The two paired halfedges
  /// can have different values while maintaining C-1 continuity (except for 0).
  float smoothness;
};

/**
 * Geometric properties of the manifold, created with Manifold.GetProperties().
 */
struct Properties {
  float surfaceArea, volume;
};

struct Box {
  glm::vec3 min = glm::vec3(std::numeric_limits<float>::infinity());
  glm::vec3 max = glm::vec3(-std::numeric_limits<float>::infinity());

  /**
   * Default constructor is an infinite box that contains all space.
   */
  Box() {}

  /**
   * Creates a box that contains the two given points.
   */
  Box(const glm::vec3 p1, const glm::vec3 p2) {
    min = glm::min(p1, p2);
    max = glm::max(p1, p2);
  }

  /**
   * Returns the dimensions of the Box.
   */
  glm::vec3 Size() const { return max - min; }

  /**
   * Returns the center point of the Box.
   */
  glm::vec3 Center() const { return 0.5f * (max + min); }

  /**
   * Returns the absolute-largest coordinate value of any contained
   * point.
   */
  float Scale() const {
    glm::vec3 absMax = glm::max(glm::abs(min), glm::abs(max));
    return glm::max(absMax.x, glm::max(absMax.y, absMax.z));
  }

  /**
   * Does this box contain (includes equal) the given point?
   */
  bool Contains(const glm::vec3& p) const {
    return glm::all(glm::greaterThanEqual(p, min)) &&
           glm::all(glm::greaterThanEqual(max, p));
  }

  /**
   * Does this box contain (includes equal) the given box?
   */
  bool Contains(const Box& box) const {
    return glm::all(glm::greaterThanEqual(box.min, min)) &&
           glm::all(glm::greaterThanEqual(max, box.max));
  }

  /**
   * Expand this box to include the given point.
   */
  void Union(const glm::vec3 p) {
    min = glm::min(min, p);
    max = glm::max(max, p);
  }

  /**
   * Expand this box to include the given box.
   */
  Box Union(const Box& box) const {
    Box out;
    out.min = glm::min(min, box.min);
    out.max = glm::max(max, box.max);
    return out;
  }

  /**
   * Transform the given box by the given axis-aligned affine transform.
   *
   * Ensure the transform passed in is axis-aligned (rotations are all
   * multiples of 90 degrees), or else the resulting bounding box will no longer
   * bound properly.
   */
  Box Transform(const glm::mat4x3& transform) const {
    Box out;
    glm::vec3 minT = transform * glm::vec4(min, 1.0f);
    glm::vec3 maxT = transform * glm::vec4(max, 1.0f);
    out.min = glm::min(minT, maxT);
    out.max = glm::max(minT, maxT);
    return out;
  }

  /**
   * Shift this box by the given vector.
   */
  Box operator+(glm::vec3 shift) const {
    Box out;
    out.min = min + shift;
    out.max = max + shift;
    return out;
  }

  /**
   * Shift this box in-place by the given vector.
   */
  Box& operator+=(glm::vec3 shift) {
    min += shift;
    max += shift;
    return *this;
  }

  /**
   * Scale this box by the given vector.
   */
  Box operator*(glm::vec3 scale) const {
    Box out;
    out.min = min * scale;
    out.max = max * scale;
    return out;
  }

  /**
   * Scale this box in-place by the given vector.
   */
  Box& operator*=(glm::vec3 scale) {
    min *= scale;
    max *= scale;
    return *this;
  }

  /**
   * Does this box overlap the one given (including equality)?
   */
  inline bool DoesOverlap(const Box& box) const {
    return min.x <= box.max.x && min.y <= box.max.y && min.z <= box.max.z &&
           max.x >= box.min.x && max.y >= box.min.y && max.z >= box.min.z;
  }

  /**
   * Does the given point project within the XY extent of this box
   * (including equality)?
   */
  inline bool DoesOverlap(glm::vec3 p) const {  // projected in z
    return p.x <= max.x && p.x >= min.x && p.y <= max.y && p.y >= min.y;
  }

  /**
   * Does this box have finite bounds?
   */
  bool IsFinite() const {
    return glm::all(glm::isfinite(min)) && glm::all(glm::isfinite(max));
  }
};

/**
 * Axis-aligned rectangular bounds.
 */
struct Rect {
  glm::vec2 min = glm::vec2(std::numeric_limits<float>::infinity());
  glm::vec2 max = glm::vec2(-std::numeric_limits<float>::infinity());

  /**
   * Default constructor is an empty rectangle..
   */
  Rect() {}

  /**
   * Create a rectangle that contains the two given points.
   */
  Rect(const glm::vec2 a, const glm::vec2 b) {
    min = glm::min(a, b);
    max = glm::max(a, b);
  }

  /** @name Information
   *  Details of the rectangle
   */
  ///@{

  /**
   * Return the dimensions of the rectangle.
   */
  glm::vec2 Size() const { return max - min; }

  /**
   * Return the area of the rectangle.
   */
  float Area() const {
    auto sz = Size();
    return sz.x * sz.y;
  }

  /**
   * Returns the absolute-largest coordinate value of any contained
   * point.
   */
  float Scale() const {
    glm::vec2 absMax = glm::max(glm::abs(min), glm::abs(max));
    return glm::max(absMax.x, absMax.y);
  }

  /**
   * Returns the center point of the rectangle.
   */
  glm::vec2 Center() const { return 0.5f * (max + min); }

  /**
   * Does this rectangle contain (includes on border) the given point?
   */
  bool Contains(const glm::vec2& p) const {
    return glm::all(glm::greaterThanEqual(p, min)) &&
           glm::all(glm::greaterThanEqual(max, p));
  }

  /**
   * Does this rectangle contain (includes equal) the given rectangle?
   */
  bool Contains(const Rect& rect) const {
    return glm::all(glm::greaterThanEqual(rect.min, min)) &&
           glm::all(glm::greaterThanEqual(max, rect.max));
  }

  /**
   * Does this rectangle overlap the one given (including equality)?
   */
  bool DoesOverlap(const Rect& rect) const {
    return min.x <= rect.max.x && min.y <= rect.max.y && max.x >= rect.min.x &&
           max.y >= rect.min.y;
  }

  /**
   * Is the rectangle empty (containing no space)?
   */
  bool IsEmpty() const { return max.y <= min.y || max.x <= min.x; };

  /**
   * Does this recangle have finite bounds?
   */
  bool IsFinite() const {
    return glm::all(glm::isfinite(min)) && glm::all(glm::isfinite(max));
  }

  ///@}

  /** @name Modification
   */
  ///@{

  /**
   * Expand this rectangle (in place) to include the given point.
   */
  void Union(const glm::vec2 p) {
    min = glm::min(min, p);
    max = glm::max(max, p);
  }

  /**
   * Expand this rectangle to include the given Rect.
   */
  Rect Union(const Rect& rect) const {
    Rect out;
    out.min = glm::min(min, rect.min);
    out.max = glm::max(max, rect.max);
    return out;
  }

  /**
   * Shift this rectangle by the given vector.
   */
  Rect operator+(const glm::vec2 shift) const {
    Rect out;
    out.min = min + shift;
    out.max = max + shift;
    return out;
  }

  /**
   * Shift this rectangle in-place by the given vector.
   */
  Rect& operator+=(const glm::vec2 shift) {
    min += shift;
    max += shift;
    return *this;
  }

  /**
   * Scale this rectangle by the given vector.
   */
  Rect operator*(const glm::vec2 scale) const {
    Rect out;
    out.min = min * scale;
    out.max = max * scale;
    return out;
  }

  /**
   * Scale this rectangle in-place by the given vector.
   */
  Rect& operator*=(const glm::vec2 scale) {
    min *= scale;
    max *= scale;
    return *this;
  }

  /**
   * Transform the rectangle by the given axis-aligned affine transform.
   *
   * Ensure the transform passed in is axis-aligned (rotations are all
   * multiples of 90 degrees), or else the resulting rectangle will no longer
   * bound properly.
   */
  Rect Transform(const glm::mat3x2& m) const {
    Rect rect;
    rect.min = m * glm::vec3(min, 1);
    rect.max = m * glm::vec3(max, 1);
    return rect;
  }
  ///@}
};
/** @} */

/** @addtogroup Core
 *  @{
 */

/**
 * Boolean operation type: Add (Union), Subtract (Difference), and Intersect.
 */
enum class OpType { Add, Subtract, Intersect };

/**
 * These static properties control how circular shapes are quantized by
 * default on construction. If circularSegments is specified, it takes
 * precedence. If it is zero, then instead the minimum is used of the segments
 * calculated based on edge length and angle, rounded up to the nearest
 * multiple of four. To get numbers not divisible by four, circularSegments
 * must be specified.
 */
class Quality {
 private:
  inline static int circularSegments_ = 0;
  inline static float circularAngle_ = 10.0f;
  inline static float circularEdgeLength_ = 1.0f;

 public:
  /**
   * Sets an angle constraint the default number of circular segments for the
   * CrossSection::Circle(), Manifold::Cylinder(), Manifold::Sphere(), and
   * Manifold::Revolve() constructors. The number of segments will be rounded up
   * to the nearest factor of four.
   *
   * @param angle The minimum angle in degrees between consecutive segments. The
   * angle will increase if the the segments hit the minimum edge length.
   * Default is 10 degrees.
   */
  static void SetMinCircularAngle(float angle) {
    if (angle <= 0) return;
    circularAngle_ = angle;
  }

  /**
   * Sets a length constraint the default number of circular segments for the
   * CrossSection::Circle(), Manifold::Cylinder(), Manifold::Sphere(), and
   * Manifold::Revolve() constructors. The number of segments will be rounded up
   * to the nearest factor of four.
   *
   * @param length The minimum length of segments. The length will
   * increase if the the segments hit the minimum angle. Default is 1.0.
   */
  static void SetMinCircularEdgeLength(float length) {
    if (length <= 0) return;
    circularEdgeLength_ = length;
  }

  /**
   * Sets the default number of circular segments for the
   * CrossSection::Circle(), Manifold::Cylinder(), Manifold::Sphere(), and
   * Manifold::Revolve() constructors. Overrides the edge length and angle
   * constraints and sets the number of segments to exactly this value.
   *
   * @param number Number of circular segments. Default is 0, meaning no
   * constraint is applied.
   */
  static void SetCircularSegments(int number) {
    if (number < 3 && number != 0) return;
    circularSegments_ = number;
  }

  /**
   * Determine the result of the SetMinCircularAngle(),
   * SetMinCircularEdgeLength(), and SetCircularSegments() defaults.
   *
   * @param radius For a given radius of circle, determine how many default
   * segments there will be.
   */
  static int GetCircularSegments(float radius) {
    if (circularSegments_ > 0) return circularSegments_;
    int nSegA = 360.0f / circularAngle_;
    int nSegL = 2.0f * radius * glm::pi<float>() / circularEdgeLength_;
    int nSeg = fmin(nSegA, nSegL) + 3;
    nSeg -= nSeg % 4;
    return std::max(nSeg, 3);
  }
};
/** @} */

/** @defgroup Debug
 *  @brief Debugging features
 *
 * The features require compiler flags to be enabled. Assertions are enabled
 * with the MANIFOLD_DEBUG flag and then controlled with ExecutionParams.
 * Exceptions are only thrown if the MANIFOLD_EXCEPTIONS flag is set. Import and
 * Export of 3D models is only supported with the MANIFOLD_EXPORT flag, which
 * also requires linking in the Assimp dependency.
 *  @{
 */

/** @defgroup Exceptions
 *  @brief Custom Exceptions
 * @{
 */
#ifdef MANIFOLD_DEBUG
struct userErr : public virtual std::runtime_error {
  using std::runtime_error::runtime_error;
};
struct topologyErr : public virtual std::runtime_error {
  using std::runtime_error::runtime_error;
};
struct geometryErr : public virtual std::runtime_error {
  using std::runtime_error::runtime_error;
};
using logicErr = std::logic_error;
#endif
/** @} */

/**
 * Global parameters that control debugging output. Only has an
 * effect when compiled with the MANIFOLD_DEBUG flag.
 */
struct ExecutionParams {
  /// Perform extra sanity checks and assertions on the intermediate data
  /// structures.
  bool intermediateChecks = false;
  /// Verbose output primarily of the Boolean, including timing info and vector
  /// sizes.
  bool verbose = false;
  /// If processOverlaps is false, a geometric check will be performed to assert
  /// all triangles are CCW.
  bool processOverlaps = true;
  /// Suppresses printed errors regarding CW triangles. Has no effect if
  /// processOverlaps is true.
  bool suppressErrors = false;
  /// Deterministic outputs. Will disable some parallel optimizations.
  bool deterministic = false;
  /// Perform optional but recommended triangle cleanups in SimplifyTopology()
  bool cleanupTriangles = true;
};

#ifdef MANIFOLD_DEBUG

template <typename T>
inline std::ostream& operator<<(std::ostream& stream, const glm::tvec2<T>& v) {
  return stream << "x = " << v.x << ", y = " << v.y;
}

template <typename T>
inline std::ostream& operator<<(std::ostream& stream, const glm::tvec3<T>& v) {
  return stream << "x = " << v.x << ", y = " << v.y << ", z = " << v.z;
}

template <typename T>
inline std::ostream& operator<<(std::ostream& stream, const glm::tvec4<T>& v) {
  return stream << "x = " << v.x << ", y = " << v.y << ", z = " << v.z
                << ", w = " << v.w;
}

inline std::ostream& operator<<(std::ostream& stream, const glm::mat3& mat) {
  glm::mat3 tam = glm::transpose(mat);
  return stream << tam[0] << std::endl
                << tam[1] << std::endl
                << tam[2] << std::endl;
}

inline std::ostream& operator<<(std::ostream& stream, const glm::mat4x3& mat) {
  glm::mat3x4 tam = glm::transpose(mat);
  return stream << tam[0] << std::endl
                << tam[1] << std::endl
                << tam[2] << std::endl;
}

inline std::ostream& operator<<(std::ostream& stream, const Box& box) {
  return stream << "min: " << box.min << ", "
                << "max: " << box.max;
}

inline std::ostream& operator<<(std::ostream& stream, const Rect& box) {
  return stream << "min: " << box.min << ", "
                << "max: " << box.max;
}

/**
 * Print the contents of this vector to standard output. Only exists if compiled
 * with MANIFOLD_DEBUG flag.
 */
template <typename T>
void Dump(const std::vector<T>& vec) {
  std::cout << "Vec = " << std::endl;
  for (int i = 0; i < vec.size(); ++i) {
    std::cout << i << ", " << vec[i] << ", " << std::endl;
  }
  std::cout << std::endl;
}

template <typename T>
void Diff(const std::vector<T>& a, const std::vector<T>& b) {
  std::cout << "Diff = " << std::endl;
  if (a.size() != b.size()) {
    std::cout << "a and b must have the same length, aborting Diff"
              << std::endl;
    return;
  }
  for (int i = 0; i < a.size(); ++i) {
    if (a[i] != b[i])
      std::cout << i << ": " << a[i] << ", " << b[i] << std::endl;
  }
  std::cout << std::endl;
}
/** @} */
#endif
}  // namespace manifold

#undef HOST_DEVICE
