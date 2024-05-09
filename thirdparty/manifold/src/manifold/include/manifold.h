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

#include "cross_section.h"
#include "public.h"
#include "vec_view.h"

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

/**
 * An alternative to Mesh for output suitable for pushing into graphics
 * libraries directly. This may not be manifold since the verts are duplicated
 * along property boundaries that do not match. The additional merge vectors
 * store this missing information, allowing the manifold to be reconstructed.
 */
struct MeshGL {
  /// Number of property vertices
  uint32_t NumVert() const {
    if (vertProperties.size() / numProp >= static_cast<std::vector<float>::size_type>(std::numeric_limits<int>::max()))
      throw std::out_of_range("mesh too large");
    return vertProperties.size() / numProp;
  };
  /// Number of triangles
  uint32_t NumTri() const {
    if (vertProperties.size() / numProp >= static_cast<std::vector<float>::size_type>(std::numeric_limits<int>::max()))
      throw std::out_of_range("mesh too large");
    return triVerts.size() / 3;
  };

  /// Number of properties per vertex, always >= 3.
  uint32_t numProp = 3;
  /// Flat, GL-style interleaved list of all vertex properties: propVal =
  /// vertProperties[vert * numProp + propIdx]. The first three properties are
  /// always the position x, y, z.
  std::vector<float> vertProperties;
  /// The vertex indices of the three triangle corners in CCW (from the outside)
  /// order, for each triangle.
  std::vector<uint32_t> triVerts;
  /// Optional: A list of only the vertex indicies that need to be merged to
  /// reconstruct the manifold.
  std::vector<uint32_t> mergeFromVert;
  /// Optional: The same length as mergeFromVert, and the corresponding value
  /// contains the vertex to merge with. It will have an identical position, but
  /// the other properties may differ.
  std::vector<uint32_t> mergeToVert;
  /// Optional: Indicates runs of triangles that correspond to a particular
  /// input mesh instance. The runs encompass all of triVerts and are sorted
  /// by runOriginalID. Run i begins at triVerts[runIndex[i]] and ends at
  /// triVerts[runIndex[i+1]]. All runIndex values are divisible by 3.
  std::vector<uint32_t> runIndex;
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
  std::vector<float> runTransform;
  /// Optional: Length NumTri, contains an ID of the source face this triangle
  /// comes from. When auto-generated, this ID will be a triangle index into the
  /// original mesh. All neighboring coplanar triangles from that input mesh
  /// will refer to a single triangle of that group as the faceID. When
  /// supplying faceIDs, ensure that triangles with the same ID are in fact
  /// coplanar and have consistent properties (within some tolerance) or the
  /// output will be surprising.
  std::vector<uint32_t> faceID;
  /// Optional: The X-Y-Z-W weighted tangent vectors for smooth Refine(). If
  /// non-empty, must be exactly four times as long as Mesh.triVerts. Indexed
  /// as 4 * (3 * tri + i) + j, i < 3, j < 4, representing the tangent value
  /// Mesh.triVerts[tri][i] along the CCW edge. If empty, mesh is faceted.
  std::vector<float> halfedgeTangent;
  /// The absolute precision of the vertex positions, based on accrued rounding
  /// errors. When creating a Manifold, the precision used will be the maximum
  /// of this and a baseline precision from the size of the bounding box. Any
  /// edge shorter than precision may be collapsed.
  float precision = 0;

  MeshGL() = default;
  MeshGL(const Mesh& mesh);

  bool Merge();
};
/** @} */

/** @defgroup Core
 *  @brief The central classes of the library
 *  @{
 */

/**
 * This library's internal representation of an oriented, 2-manifold, triangle
 * mesh - a simple boundary-representation of a solid object. Use this class to
 * store and operate on solids, and use MeshGL for input and output, or
 * potentially Mesh if only basic geometry is required.
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

  Manifold(const MeshGL&, const std::vector<float>& propertyTolerance = {});
  Manifold(const Mesh&);

  static Manifold Smooth(const MeshGL&,
                         const std::vector<Smoothness>& sharpenedEdges = {});
  static Manifold Smooth(const Mesh&,
                         const std::vector<Smoothness>& sharpenedEdges = {});
  static Manifold Tetrahedron();
  static Manifold Cube(glm::vec3 size = glm::vec3(1.0f), bool center = false);
  static Manifold Cylinder(float height, float radiusLow,
                           float radiusHigh = -1.0f, int circularSegments = 0,
                           bool center = false);
  static Manifold Sphere(float radius, int circularSegments = 0);
  static Manifold Extrude(const CrossSection& crossSection, float height,
                          int nDivisions = 0, float twistDegrees = 0.0f,
                          glm::vec2 scaleTop = glm::vec2(1.0f));
  static Manifold Revolve(const CrossSection& crossSection,
                          int circularSegments = 0,
                          float revolveDegrees = 360.0f);
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
  Mesh GetMesh() const;
  MeshGL GetMeshGL(glm::ivec3 normalIdx = glm::ivec3(0)) const;
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
  int NumVert() const;
  int NumEdge() const;
  int NumTri() const;
  int NumProp() const;
  int NumPropVert() const;
  Box BoundingBox() const;
  float Precision() const;
  int Genus() const;
  Properties GetProperties() const;
  float MinGap(const Manifold& other, float searchLength) const;
  ///@}

  /** @name Mesh ID
   *  Details of the manifold's relation to its input meshes, for the purposes
   * of reapplying mesh properties.
   */
  ///@{
  int OriginalID() const;
  Manifold AsOriginal() const;
  static uint32_t ReserveIDs(uint32_t);
  ///@}

  /** @name Modification
   */
  ///@{
  Manifold Translate(glm::vec3) const;
  Manifold Scale(glm::vec3) const;
  Manifold Rotate(float xDegrees, float yDegrees = 0.0f,
                  float zDegrees = 0.0f) const;
  Manifold Transform(const glm::mat4x3&) const;
  Manifold Mirror(glm::vec3) const;
  Manifold Warp(std::function<void(glm::vec3&)>) const;
  Manifold WarpBatch(std::function<void(VecView<glm::vec3>)>) const;
  Manifold SetProperties(
      int, std::function<void(float*, glm::vec3, const float*)>) const;
  Manifold CalculateCurvature(int gaussianIdx, int meanIdx) const;
  Manifold CalculateNormals(int normalIdx, float minSharpAngle = 60) const;
  Manifold SmoothByNormals(int normalIdx) const;
  Manifold SmoothOut(float minSharpAngle = 60, float minSmoothness = 0) const;
  Manifold Refine(int) const;
  Manifold RefineToLength(float) const;
  // Manifold RefineToPrecision(float);
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
  std::pair<Manifold, Manifold> SplitByPlane(glm::vec3 normal,
                                             float originOffset) const;
  Manifold TrimByPlane(glm::vec3 normal, float originOffset) const;
  ///@}

  /** @name 2D from 3D
   */
  ///@{
  CrossSection Slice(float height = 0) const;
  CrossSection Project() const;
  ///@}

  /** @name Convex hull
   */
  ///@{
  Manifold Hull() const;
  static Manifold Hull(const std::vector<Manifold>& manifolds);
  static Manifold Hull(const std::vector<glm::vec3>& pts);
  ///@}

  /** @name Testing hooks
   *  These are just for internal testing.
   */
  ///@{
  bool MatchesTriNormals() const;
  int NumDegenerateTris() const;
  int NumOverlaps(const Manifold& second) const;
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
