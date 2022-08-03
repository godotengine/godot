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
#define GLM_FORCE_EXPLICIT_CTOR
#include <glm/ext/matrix_transform.hpp>
#include <glm/glm.hpp>
#include <glm/gtc/constants.hpp>
#include <glm/gtx/compatibility.hpp>
#include <glm/gtx/rotate_vector.hpp>
#include <limits>
#include <unordered_map>
#include <vector>

#ifdef MANIFOLD_DEBUG
#include <iomanip>
#include <iostream>
#include <sstream>
#endif

namespace manifold {

constexpr float kTolerance = 1e-5;

#ifdef __CUDACC__
#define HOST_DEVICE __host__ __device__
#else
#define HOST_DEVICE
#endif

#ifdef MANIFOLD_DEBUG
/** @defgroup Exceptions
 *  @brief Custom exceptions
 *  @{
 */
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
/** @} */
#endif

/** @addtogroup Private
 *  @{
 */
inline HOST_DEVICE int Signum(float val) { return (val > 0) - (val < 0); }

inline HOST_DEVICE int CCW(glm::vec2 p0, glm::vec2 p1, glm::vec2 p2,
                           float tol) {
  glm::vec2 v1 = p1 - p0;
  glm::vec2 v2 = p2 - p0;
  float area = v1.x * v2.y - v1.y * v2.x;
  float base2 = glm::max(glm::dot(v1, v1), glm::dot(v2, v2));
  if (area * area <= base2 * tol * tol)
    return 0;
  else
    return area > 0 ? 1 : -1;
}

struct ExecutionParams {
  bool intermediateChecks = false;
  bool verbose = false;
  bool suppressErrors = false;
  bool processOverlaps = false;
};

struct Halfedge {
  int startVert, endVert;
  int pairedHalfedge;
  int face;
  HOST_DEVICE bool IsForward() const { return startVert < endVert; }
  HOST_DEVICE bool operator<(const Halfedge& other) const {
    return startVert == other.startVert ? endVert < other.endVert
                                        : startVert < other.startVert;
  }
};
/** @} */

/** @defgroup Connections
 *  @brief Move data in and out of the Manifold class.
 *  @{
 */

/**
 * Sine function where multiples of 90 degrees come out exact.
 *
 * @param x Angle in degrees.
 */
inline HOST_DEVICE float sind(float x) {
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
inline HOST_DEVICE float cosd(float x) { return sind(x + 90.0f); }

/**
 * This 4x3 matrix can be used as an input to Manifold.Transform() to turn an
 * object. Turns along the shortest path from given up-vector to (0, 0, 1).
 *
 * @param up The vector to be turned to point upwards. Length does not matter.
 */
inline HOST_DEVICE glm::mat4x3 RotateUp(glm::vec3 up) {
  up = glm::normalize(up);
  glm::vec3 axis = glm::cross(up, {0, 0, 1});
  float angle = glm::asin(glm::length(axis));
  if (glm::dot(up, {0, 0, 1}) < 0) angle = glm::pi<float>() - angle;
  return glm::mat4x3(glm::rotate(glm::mat4(1), angle, axis));
}

/**
 * Polygon vertex.
 */
struct PolyVert {
  /// X-Y position
  glm::vec2 pos;
  /// ID or index into another vertex vector
  int idx;
};

/**
 * Single polygon contour, wound CCW. First and last point are implicitly
 * connected. Should ensure all input is
 * [&epsilon;-valid](https://github.com/elalish/manifold/wiki/Manifold-Library#definition-of-%CE%B5-valid).
 */
using SimplePolygon = std::vector<PolyVert>;

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

/**
 * Discrete curvature of a manifold calculated at every vertex. See
 * Manifold.GetCurvature() for details.
 */
struct Curvature {
  float maxMeanCurvature, minMeanCurvature;
  float maxGaussianCurvature, minGaussianCurvature;
  std::vector<float> vertMeanCurvature, vertGaussianCurvature;
};

/**
 * Part of MeshRelation - represents a single triangle relation to an original
 * Mesh.
 */
struct BaryRef {
  /// The unique ID of the mesh instance of this triangle. If .meshID and .tri
  /// match for two triangles, then they are coplanar and came from the same
  /// face.
  int meshID;
  /// The OriginalID of the mesh this triangle came from. This ID is ideal for
  /// reapplying properties like UV coordinates to the output mesh.
  int originalID;
  /// The triangle index of the original triangle this was part of:
  /// Mesh.triVerts[tri].
  int tri;
  /// For the three corners of the output triangle, new (intersection) vertices
  /// store an index to the MeshRelation.barycentric vector, while original
  /// vertices have negative values referring to Mesh.triVerts[tri][i + 3].
  glm::ivec3 vertBary;
};

/**
 * Represents the relationship of this output Mesh to all input Meshes that
 * eventually led to it, see Manifold.GetMeshRelation().
 */
struct MeshRelation {
  /// A vector of shared barycentric coordinates representing the position of a
  /// vertex relative to its original triangle.
  std::vector<glm::vec3> barycentric;
  /// A vector matching Mesh.triVerts that contains the relation of each output
  /// triangle to a single input triangle.
  std::vector<BaryRef> triBary;

  /**
   * A convenience function to get the barycentric coordinates of a given
   * corner.
   *
   * @param tri A valid triangle index of Mesh.triVerts.
   * @param vert The corner of the triangle: 0, 1, or 2.
   */
  inline glm::vec3 UVW(int tri, int vert) {
    glm::vec3 uvw(0.0f);
    const int idx = triBary[tri].vertBary[vert];
    if (idx < 0) {
      uvw[idx + 3] = 1;
    } else {
      uvw = barycentric[idx];
    }
    return uvw;
  }
};

/**
 * Axis-aligned bounding box
 */
struct Box {
  glm::vec3 min = glm::vec3(std::numeric_limits<float>::infinity());
  glm::vec3 max = glm::vec3(-std::numeric_limits<float>::infinity());

  /**
   * Default constructor is an infinite box that contains all space.
   */
  HOST_DEVICE Box() {}

  /**
   * Creates a box that contains the two given points.
   */
  HOST_DEVICE Box(const glm::vec3 p1, const glm::vec3 p2) {
    min = glm::min(p1, p2);
    max = glm::max(p1, p2);
  }

  /**
   * Returns the dimensions of the Box.
   */
  HOST_DEVICE glm::vec3 Size() const { return max - min; }

  /**
   * Returns the center point of the Box.
   */
  HOST_DEVICE glm::vec3 Center() const { return 0.5f * (max + min); }

  /**
   * Returns the absolute-largest coordinate value of any contained
   * point.
   */
  HOST_DEVICE float Scale() const {
    glm::vec3 absMax = glm::max(glm::abs(min), glm::abs(max));
    return glm::max(absMax.x, glm::max(absMax.y, absMax.z));
  }

  /**
   * Does this box contain (includes equal) the given box?
   */
  HOST_DEVICE bool Contains(const Box& box) const {
    return glm::all(glm::greaterThanEqual(box.min, min)) &&
           glm::all(glm::greaterThanEqual(max, box.max));
  }

  /**
   * Expand this box to include the given point.
   */
  HOST_DEVICE void Union(const glm::vec3 p) {
    min = glm::min(min, p);
    max = glm::max(max, p);
  }

  /**
   * Expand this box to include the given box.
   */
  HOST_DEVICE Box Union(const Box& box) const {
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
  HOST_DEVICE Box Transform(const glm::mat4x3& transform) const {
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
  HOST_DEVICE Box operator+(glm::vec3 shift) const {
    Box out;
    out.min = min + shift;
    out.max = max + shift;
    return out;
  }

  /**
   * Shift this box in-place by the given vector.
   */
  HOST_DEVICE Box& operator+=(glm::vec3 shift) {
    min += shift;
    max += shift;
    return *this;
  }

  /**
   * Scale this box by the given vector.
   */
  HOST_DEVICE Box operator*(glm::vec3 scale) const {
    Box out;
    out.min = min * scale;
    out.max = max * scale;
    return out;
  }

  /**
   * Scale this box in-place by the given vector.
   */
  HOST_DEVICE Box& operator*=(glm::vec3 scale) {
    min *= scale;
    max *= scale;
    return *this;
  }

  /**
   * Does this box overlap the one given (including equality)?
   */
  HOST_DEVICE bool DoesOverlap(const Box& box) const {
    return min.x <= box.max.x && min.y <= box.max.y && min.z <= box.max.z &&
           max.x >= box.min.x && max.y >= box.min.y && max.z >= box.min.z;
  }

  /**
   * Does the given point project within the XY extent of this box
   * (including equality)?
   */
  HOST_DEVICE bool DoesOverlap(glm::vec3 p) const {  // projected in z
    return p.x <= max.x && p.x >= min.x && p.y <= max.y && p.y >= min.y;
  }

  /**
   * Does this box have finite bounds?
   */
  HOST_DEVICE bool IsFinite() const {
    return glm::all(glm::isfinite(min)) && glm::all(glm::isfinite(max));
  }
};

#ifdef MANIFOLD_DEBUG
inline std::ostream& operator<<(std::ostream& stream, const Box& box) {
  return stream << "min: " << box.min.x << ", " << box.min.y << ", "
                << box.min.z << ", "
                << "max: " << box.max.x << ", " << box.max.y << ", "
                << box.max.z;
}

inline std::ostream& operator<<(std::ostream& stream, const Halfedge& edge) {
  return stream << "startVert = " << edge.startVert
                << ", endVert = " << edge.endVert
                << ", pairedHalfedge = " << edge.pairedHalfedge
                << ", face = " << edge.face;
}

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

inline std::ostream& operator<<(std::ostream& stream, const glm::mat4x3& mat) {
  glm::mat3x4 tam = glm::transpose(mat);
  return stream << tam[0] << std::endl
                << tam[1] << std::endl
                << tam[2] << std::endl;
}

inline std::ostream& operator<<(std::ostream& stream, const BaryRef& ref) {
  return stream << "meshID: " << ref.meshID
                << ", originalID: " << ref.originalID << ", tri: " << ref.tri
                << ", uvw idx: " << ref.vertBary;
}

/**
 * Print the contents of this vector to standard output.
 */
template <typename T>
void Dump(const std::vector<T>& vec) {
  std::cout << "Vec = " << std::endl;
  for (int i = 0; i < vec.size(); ++i) {
    std::cout << i << ", " << vec[i] << ", " << std::endl;
  }
  std::cout << std::endl;
}
/** @} */
#endif
}  // namespace manifold

#undef HOST_DEVICE
