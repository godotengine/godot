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
#include <functional>
#include <memory>

#include "manifold/common.h"
#include "manifold/vec_view.h"

namespace manifold {

/**
 * @ingroup Debug
 *
 * Allows modification of the assertions checked in MANIFOLD_DEBUG mode.
 *
 * @return ExecutionParams&
 */
ExecutionParams& ManifoldParams();

class CsgNode;
class CsgLeafNode;

/** @ingroup Connections
 *  @{
 */

template <typename Precision, typename I = uint32_t>
struct MeshGLP {
  /// Number of property vertices
  I NumVert() const { return vertProperties.size() / numProp; };
  /// Number of triangles
  I NumTri() const { return triVerts.size() / 3; };
  /// Number of properties per vertex, always >= 3.
  I numProp = 3;
  /// Flat, GL-style interleaved list of all vertex properties: propVal =
  /// vertProperties[vert * numProp + propIdx]. The first three properties are
  /// always the position x, y, z.
  std::vector<Precision> vertProperties;
  /// The vertex indices of the three triangle corners in CCW (from the outside)
  /// order, for each triangle.
  std::vector<I> triVerts;
  /// Optional: A list of only the vertex indicies that need to be merged to
  /// reconstruct the manifold.
  std::vector<I> mergeFromVert;
  /// Optional: The same length as mergeFromVert, and the corresponding value
  /// contains the vertex to merge with. It will have an identical position, but
  /// the other properties may differ.
  std::vector<I> mergeToVert;
  /// Optional: Indicates runs of triangles that correspond to a particular
  /// input mesh instance. The runs encompass all of triVerts and are sorted
  /// by runOriginalID. Run i begins at triVerts[runIndex[i]] and ends at
  /// triVerts[runIndex[i+1]]. All runIndex values are divisible by 3. Returned
  /// runIndex will always be 1 longer than runOriginalID, but same length is
  /// also allowed as input: triVerts.size() will be automatically appended in
  /// this case.
  std::vector<I> runIndex;
  /// Optional: The OriginalID of the mesh this triangle run came from. This ID
  /// is ideal for reapplying materials to the output mesh. Multiple runs may
  /// have the same ID, e.g. representing different copies of the same input
  /// mesh. If you create an input MeshGL that you want to be able to reference
  /// as one or more originals, be sure to set unique values from ReserveIDs().
  std::vector<uint32_t> runOriginalID;
  /// Optional: For each run, a 3x4 transform is stored representing how the
  /// corresponding original mesh was transformed to create this triangle run.
  /// This matrix is stored in column-major order and the length of the overall
  /// vector is 12 * runOriginalID.size().
  std::vector<Precision> runTransform;
  /// Optional: Length NumTri, contains an ID of the source face this triangle
  /// comes from. When auto-generated, this ID will be a triangle index into the
  /// original mesh. All neighboring coplanar triangles from that input mesh
  /// will refer to a single triangle of that group as the faceID. When
  /// supplying faceIDs, ensure that triangles with the same ID are in fact
  /// coplanar and have consistent properties (within some tolerance) or the
  /// output will be surprising.
  std::vector<I> faceID;
  /// Optional: The X-Y-Z-W weighted tangent vectors for smooth Refine(). If
  /// non-empty, must be exactly four times as long as Mesh.triVerts. Indexed
  /// as 4 * (3 * tri + i) + j, i < 3, j < 4, representing the tangent value
  /// Mesh.triVerts[tri][i] along the CCW edge. If empty, mesh is faceted.
  std::vector<Precision> halfedgeTangent;
  /// The absolute precision of the vertex positions, based on accrued rounding
  /// errors. When creating a Manifold, the precision used will be the maximum
  /// of this and a baseline precision from the size of the bounding box. Any
  /// edge shorter than precision may be collapsed.
  Precision precision = 0;

  MeshGLP() = default;

  bool Merge();

  la::vec<Precision, 3> GetVertPos(size_t i) const {
    size_t offset = i * numProp;
    return la::vec<Precision, 3>(vertProperties[offset],
                                 vertProperties[offset + 1],
                                 vertProperties[offset + 2]);
  }

  la::vec<I, 3> GetTriVerts(size_t i) const {
    size_t offset = 3 * i;
    return la::vec<I, 3>(triVerts[offset], triVerts[offset + 1],
                         triVerts[offset + 2]);
  }

  la::vec<Precision, 4> GetTangent(size_t i) const {
    size_t offset = 4 * i;
    return la::vec<Precision, 4>(
        halfedgeTangent[offset], halfedgeTangent[offset + 1],
        halfedgeTangent[offset + 2], halfedgeTangent[offset + 3]);
  }
};

/**
 * An alternative to Mesh for output suitable for pushing into graphics
 * libraries directly. This may not be manifold since the verts are duplicated
 * along property boundaries that do not match. The additional merge vectors
 * store this missing information, allowing the manifold to be reconstructed.
 */
using MeshGL = MeshGLP<float>;
using MeshGL64 = MeshGLP<double, size_t>;
/** @} */

/** @defgroup Core
 *  @brief The central classes of the library
 *  @{
 */

/**
 * This library's internal representation of an oriented, 2-manifold, triangle
 * mesh - a simple boundary-representation of a solid object. Use this class to
 * store and operate on solids, and use MeshGL for input and output.
 *
 * In addition to storing geometric data, a Manifold can also store an arbitrary
 * number of vertex properties. These could be anything, e.g. normals, UV
 * coordinates, colors, etc, but this library is completely agnostic. All
 * properties are merely float values indexed by channel number. It is up to the
 * user to associate channel numbers with meaning.
 *
 * Manifold allows vertex properties to be shared for efficient storage, or to
 * have multiple property verts associated with a single geometric vertex,
 * allowing sudden property changes, e.g. at Boolean intersections, without
 * sacrificing manifoldness.
 *
 * Manifolds also keep track of their relationships to their inputs, via
 * OriginalIDs and the faceIDs and transforms accessible through MeshGL. This
 * allows object-level properties to be re-associated with the output after many
 * operations, particularly useful for materials. Since separate object's
 * properties are not mixed, there is no requirement that channels have
 * consistent meaning between different inputs.
 */
class Manifold {
 public:
  /** @name Creation
   *  Constructors
   */
  ///@{
  Manifold();
  ~Manifold();
  Manifold(const Manifold& other);
  Manifold& operator=(const Manifold& other);
  Manifold(Manifold&&) noexcept;
  Manifold& operator=(Manifold&&) noexcept;

  Manifold(const MeshGL&);
  Manifold(const MeshGL64&);

  static Manifold Smooth(const MeshGL&,
                         const std::vector<Smoothness>& sharpenedEdges = {});
  static Manifold Smooth(const MeshGL64&,
                         const std::vector<Smoothness>& sharpenedEdges = {});
  static Manifold Tetrahedron();
  static Manifold Cube(vec3 size = vec3(1.0), bool center = false);
  static Manifold Cylinder(double height, double radiusLow,
                           double radiusHigh = -1.0, int circularSegments = 0,
                           bool center = false);
  static Manifold Sphere(double radius, int circularSegments = 0);
  static Manifold Extrude(const Polygons& crossSection, double height,
                          int nDivisions = 0, double twistDegrees = 0.0,
                          vec2 scaleTop = vec2(1.0));
  static Manifold Revolve(const Polygons& crossSection,
                          int circularSegments = 0,
                          double revolveDegrees = 360.0f);
  static Manifold LevelSet(std::function<double(vec3)> sdf, Box bounds,
                           double edgeLength, double level = 0,
                           double precision = -1, bool canParallel = true);
  ///@}

  /** @name Topological
   *  No geometric calculations.
   */
  ///@{
  static Manifold Compose(const std::vector<Manifold>&);
  std::vector<Manifold> Decompose() const;
  ///@}

  /** @name Information
   *  Details of the manifold
   */
  ///@{
  MeshGL GetMeshGL(ivec3 normalIdx = ivec3(0)) const;
  MeshGL64 GetMeshGL64(ivec3 normalIdx = ivec3(0)) const;
  bool IsEmpty() const;
  enum class Error {
    NoError,
    NonFiniteVertex,
    NotManifold,
    VertexOutOfBounds,
    PropertiesWrongLength,
    MissingPositionProperties,
    MergeVectorsDifferentLengths,
    MergeIndexOutOfBounds,
    TransformWrongLength,
    RunIndexWrongLength,
    FaceIDWrongLength,
    InvalidConstruction,
  };
  Error Status() const;
  size_t NumVert() const;
  size_t NumEdge() const;
  size_t NumTri() const;
  size_t NumProp() const;
  size_t NumPropVert() const;
  Box BoundingBox() const;
  double Precision() const;
  int Genus() const;
  Properties GetProperties() const;
  double MinGap(const Manifold& other, double searchLength) const;
  ///@}

  /** @name Mesh ID
   *  Details of the manifold's relation to its input meshes, for the purposes
   * of reapplying mesh properties.
   */
  ///@{
  int OriginalID() const;
  Manifold AsOriginal(const std::vector<double>& propertyTolerance = {}) const;
  static uint32_t ReserveIDs(uint32_t);
  ///@}

  /** @name Modification
   */
  ///@{
  Manifold Translate(vec3) const;
  Manifold Scale(vec3) const;
  Manifold Rotate(double xDegrees, double yDegrees = 0.0,
                  double zDegrees = 0.0) const;
  Manifold Transform(const mat3x4&) const;
  Manifold Mirror(vec3) const;
  Manifold Warp(std::function<void(vec3&)>) const;
  Manifold WarpBatch(std::function<void(VecView<vec3>)>) const;
  Manifold SetProperties(
      int, std::function<void(double*, vec3, const double*)>) const;
  Manifold CalculateCurvature(int gaussianIdx, int meanIdx) const;
  Manifold CalculateNormals(int normalIdx, double minSharpAngle = 60) const;
  Manifold SmoothByNormals(int normalIdx) const;
  Manifold SmoothOut(double minSharpAngle = 60, double minSmoothness = 0) const;
  Manifold Refine(int) const;
  Manifold RefineToLength(double) const;
  Manifold RefineToPrecision(double) const;
  ///@}

  /** @name Boolean
   *  Combine two manifolds
   */
  ///@{
  Manifold Boolean(const Manifold& second, OpType op) const;
  static Manifold BatchBoolean(const std::vector<Manifold>& manifolds,
                               OpType op);
  // Boolean operation shorthand
  Manifold operator+(const Manifold&) const;  // Add (Union)
  Manifold& operator+=(const Manifold&);
  Manifold operator-(const Manifold&) const;  // Subtract (Difference)
  Manifold& operator-=(const Manifold&);
  Manifold operator^(const Manifold&) const;  // Intersect
  Manifold& operator^=(const Manifold&);
  std::pair<Manifold, Manifold> Split(const Manifold&) const;
  std::pair<Manifold, Manifold> SplitByPlane(vec3 normal,
                                             double originOffset) const;
  Manifold TrimByPlane(vec3 normal, double originOffset) const;
  ///@}

  /** @name 2D from 3D
   */
  ///@{
  Polygons Slice(double height = 0) const;
  Polygons Project() const;
  ///@}

  /** @name Convex hull
   */
  ///@{
  Manifold Hull() const;
  static Manifold Hull(const std::vector<Manifold>& manifolds);
  static Manifold Hull(const std::vector<vec3>& pts);
  ///@}

  /** @name Testing hooks
   *  These are just for internal testing.
   */
  ///@{
  bool MatchesTriNormals() const;
  size_t NumDegenerateTris() const;
  size_t NumOverlaps(const Manifold& second) const;
  ///@}

  struct Impl;

 private:
  Manifold(std::shared_ptr<CsgNode> pNode_);
  Manifold(std::shared_ptr<Impl> pImpl_);
  static Manifold Invalid();
  mutable std::shared_ptr<CsgNode> pNode_;

  CsgLeafNode& GetCsgLeafNode() const;
};
/** @} */
}  // namespace manifold
