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
#include <cstdint>  // uint32_t, uint64_t
#include <functional>
#include <memory>  // needed for shared_ptr
#include <mutex>

#include "manifold/common.h"
#include "manifold/mesh.h"
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

/** @addtogroup Core
 *  @brief The central classes of the library
 *  @{
 */

/**
 * @brief This library's internal representation of an oriented, 2-manifold,
 * triangle mesh - a simple boundary-representation of a solid object. Use this
 * class to store and operate on solids, and use MeshGL for input and output.
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
  /** @name Basics
   *  Copy / move / assignment
   */
  ///@{
  Manifold();
  ~Manifold();
  Manifold(const Manifold& other);
  Manifold& operator=(const Manifold& other);
  Manifold(Manifold&&) noexcept;
  Manifold& operator=(Manifold&&) noexcept;
  ///@}

  /** @name Input & Output
   *  Create and retrieve arbitrary manifolds
   */
  ///@{
  Manifold(const MeshGL&);
  Manifold(const MeshGL64&);
  MeshGL GetMeshGL(int normalIdx = -1) const;
  MeshGL64 GetMeshGL64(int normalIdx = -1) const;
  ///@}

  /** @name Constructors
   *  Topological ops, primitives, and SDF
   */
  ///@{
  std::vector<Manifold> Decompose() const;
  [[deprecated(
      "Compose is deprecated, use BatchBoolean with OpType::Add instead.")]]
  static Manifold Compose(const std::vector<Manifold>&);
  static Manifold Tetrahedron();
  static Manifold Cube(vec3 size = vec3(1.0), bool center = false);
  static Manifold Cylinder(double height, double radiusLow,
                           double radiusHigh = -1.0, int circularSegments = 0,
                           bool center = false);
  static Manifold Sphere(double radius, int circularSegments = 0);
  static Manifold LevelSet(std::function<double(vec3)> sdf, Box bounds,
                           double edgeLength, double level = 0,
                           double tolerance = -1, bool canParallel = true);
  ///@}

  /** @name Polygons
   * 3D to 2D and 2D to 3D
   */
  ///@{
  Polygons Slice(double height = 0) const;
  Polygons Project() const;
  static Manifold Extrude(const Polygons& crossSection, double height,
                          int nDivisions = 0, double twistDegrees = 0.0,
                          vec2 scaleTop = vec2(1.0));
  static Manifold Revolve(const Polygons& crossSection,
                          int circularSegments = 0,
                          double revolveDegrees = 360.0f);
  ///@}

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
    ResultTooLarge,
    InvalidTangents,
    Cancelled,
  };

  /** @name Information
   *  Details of the manifold
   */
  ///@{
  Error Status() const;

  /// Returns a copy of this Manifold with the given ExecutionContext attached.
  /// The attachment is consumed only by `Status()` (for deferred CSG trees)
  /// and the eager ops (`Refine` / `RefineToLength` / `RefineToTolerance`,
  /// `Hull`, `MinkowskiSum` / `MinkowskiDifference`); those snapshot the ctx
  /// and report progress / observe cancellation through it. Other queries
  /// that force evaluation (`Volume`, `GetMeshGL`, `BoundingBox`, etc.) do
  /// not currently observe attached ctx.
  ///
  /// Deferred ops (Boolean operators, Translate / Rotate / Scale / Transform
  /// / Mirror / Warp / SetTolerance / Simplify, BatchBoolean, the
  /// vector-of-Manifold Hull) ignore any attached ctx and produce a result
  /// with no attached ctx. Inputs are not mutated. The idiom for observing a
  /// deferred tree is therefore:
  ///
  ///   (a + b - c).WithContext(ctx).Status();
  ///
  /// while the idiom for observing an eager op is:
  ///
  ///   m.WithContext(ctx).Refine(n);
  ///   m.WithContext(ctx).MinkowskiSum(other);
  ///
  /// Raw copy / assignment preserves the attachment (it's the same logical
  /// Manifold). Only ops that derive a *new* Manifold drop the attachment.
  Manifold WithContext(const ExecutionContext& ctx) const;

  bool IsEmpty() const;
  size_t NumVert() const;
  size_t NumEdge() const;
  size_t NumTri() const;
  size_t NumProp() const;
  size_t NumPropVert() const;
  Box BoundingBox() const;
  int Genus() const;
  double GetTolerance() const;
  ///@}

  /** @name Measurement
   */
  ///@{
  double SurfaceArea() const;
  double Volume() const;
  double MinGap(const Manifold& other, double searchLength) const;
  std::vector<RayHit> RayCast(vec3 origin, vec3 endpoint) const;
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

  /** @name Transformations
   */
  ///@{
  Manifold Translate(vec3) const;
  Manifold Scale(vec3) const;
  Manifold Rotate(double xDegrees, double yDegrees = 0.0,
                  double zDegrees = 0.0) const;
  Manifold Mirror(vec3) const;
  Manifold Transform(const mat3x4&) const;
  Manifold Warp(std::function<void(vec3&)>) const;
  Manifold WarpBatch(std::function<void(VecView<vec3>)>) const;
  Manifold SetTolerance(double) const;
  Manifold Simplify(double tolerance = 0) const;
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
  Manifold MinkowskiSum(const Manifold&) const;
  Manifold MinkowskiDifference(const Manifold&) const;
  ///@}

  /** @name Properties
   * Create and modify vertex properties.
   */
  ///@{
  Manifold SetProperties(
      int numProp,
      std::function<void(double*, vec3, const double*)> propFunc) const;
  Manifold CalculateCurvature(int gaussianIdx, int meanIdx) const;
  Manifold CalculateNormals(int normalIdx = 0,
                            double minSharpAngle = 52.5) const;
  ///@}

  /** @name Smoothing
   * Smooth meshes by calculating tangent vectors and refining to a higher
   * triangle count.
   */
  ///@{
  Manifold Refine(int) const;
  Manifold RefineToLength(double) const;
  Manifold RefineToTolerance(double) const;
  Manifold SmoothByNormals(int normalIdx = 0) const;
  Manifold SmoothOut(double minSharpAngle = 52.5,
                     double minSmoothness = 0) const;
  static Manifold Smooth(const MeshGL&,
                         const std::vector<Smoothness>& sharpenedEdges = {});
  static Manifold Smooth(const MeshGL64&,
                         const std::vector<Smoothness>& sharpenedEdges = {});
  ///@}

  /** @name Convex Hull
   */
  ///@{
  Manifold Hull() const;
  static Manifold Hull(const std::vector<Manifold>& manifolds);
  static Manifold Hull(const std::vector<vec3>& pts);
  ///@}

  /** @name I/O
   * Self-contained mechanism for reading and writing high precision Manifold
   * data.  Write function creates special-purpose OBJ files, and Read function
   * reads them in.
   *
   * To work with a file, the caller should prepare the ifstream/ostream
   * themselves, as follows:
   *
   * Reading:
   * @code
   * std::ifstream ifile;
   * ifile.open(filename);
   * if (ifile.is_open()) {
   *   Manifold obj_m = Manifold::ReadOBJ(ifile);
   *   ifile.close();
   *   if (obj_m.Status() != Manifold::Error::NoError) {
   *      std::cerr << "Failed reading " << filename << ":\n";
   *      std::cerr << Manifold::ToString(obj_m.Status()) << "\n";
   *   }
   *   ifile.close();
   * }
   * @endcode
   *
   * Writing:
   * @code
   * std::ofstream ofile;
   * ofile.open(filename);
   * if (ofile.is_open()) {
   *    if (!m.WriteOBJ(ofile)) {
   *       std::cerr << "Failed writing to " << filename << "\n";
   *    }
   * }
   * ofile.close();
   * @endcode
   */
#ifndef MANIFOLD_NO_IOSTREAM
  static Manifold ReadOBJ(std::istream& stream);
  bool WriteOBJ(std::ostream& stream) const;
#endif

  /** @name Testing Hooks
   *  These are just for internal testing.
   */
  ///@{
  bool MatchesTriNormals() const;
  size_t NumDegenerateTris() const;
  double GetEpsilon() const;
  ///@}

  struct Impl;

  /// @internal Wrap a fully-built Impl into a leaf-node Manifold.
  /// Caller is responsible for the invariants the public ctors enforce
  /// (in particular, calling `MakeEmpty(status)` on error). Used by
  /// ctx-aware static factories on `ExecutionContext`.
  static Manifold FromImpl(std::shared_ptr<Impl> pImpl);

 private:
  Manifold(std::shared_ptr<CsgNode> pNode_);
  Manifold(std::shared_ptr<Impl> pImpl_);
  static Manifold Invalid();
  static Manifold PropagateStatus(Error status);
  mutable std::shared_ptr<std::mutex> pNodeMutex_ =
      std::make_shared<std::mutex>();
  mutable std::shared_ptr<CsgNode> pNode_;
  // Optional attached ExecutionContext. shared_ptr so the Impl outlives
  // the user's ExecutionContext if a ctx-attached Manifold survives it.
  // Propagates through copy ctor / op= (raw copy preserves the attachment).
  // Manifold-returning ops do *not* propagate it: derived Manifolds get a
  // null ctx_. Eager ops (Status, Refine family) snapshot ctx_ to observe
  // their in-call work; the snapshot uses std::atomic_load, which pins the
  // Impl across long-running evaluations even if a concurrent op= reseats
  // ctx_ mid-eval.
  //
  // Accessed only via std::atomic_load / std::atomic_store: no const method
  // mutates ctx_, but op= and the copy ctor write it on a Manifold that
  // may be concurrently observed by const methods on other threads. The
  // atomic-shared-ptr free functions give a torn-read-free snapshot
  // without taking a lock. (pNode_ uses a mutex instead because lazy CSG
  // eval mutates it through const methods, which atomic_load can't model.)
  std::shared_ptr<ExecutionContext::Impl> ctx_;

  std::shared_ptr<CsgNode> LoadPNode() const;
  CsgLeafNode& GetCsgLeafNode(ExecutionContext::Impl* ctx = nullptr) const;
};
/** @} */

/** @addtogroup Debug
 *  @ingroup Optional
 *  @brief Debugging features
 *
 * The features require compiler flags to be enabled. Assertions are enabled
 * with the MANIFOLD_DEBUG flag and then controlled with ExecutionParams.
 *  @{
 */
#ifdef MANIFOLD_DEBUG
inline std::string ToString(const Manifold::Error& error) {
  switch (error) {
    case Manifold::Error::NoError:
      return "No Error";
    case Manifold::Error::NonFiniteVertex:
      return "Non Finite Vertex";
    case Manifold::Error::NotManifold:
      return "Not Manifold";
    case Manifold::Error::VertexOutOfBounds:
      return "Vertex Out Of Bounds";
    case Manifold::Error::PropertiesWrongLength:
      return "Properties Wrong Length";
    case Manifold::Error::MissingPositionProperties:
      return "Missing Position Properties";
    case Manifold::Error::MergeVectorsDifferentLengths:
      return "Merge Vectors Different Lengths";
    case Manifold::Error::MergeIndexOutOfBounds:
      return "Merge Index Out Of Bounds";
    case Manifold::Error::TransformWrongLength:
      return "Transform Wrong Length";
    case Manifold::Error::RunIndexWrongLength:
      return "Run Index Wrong Length";
    case Manifold::Error::FaceIDWrongLength:
      return "Face ID Wrong Length";
    case Manifold::Error::InvalidConstruction:
      return "Invalid Construction";
    case Manifold::Error::ResultTooLarge:
      return "Result Too Large";
    case Manifold::Error::InvalidTangents:
      return "Invalid Tangents";
    case Manifold::Error::Cancelled:
      return "Cancelled";
    default:
      return "Unknown Error";
  };
}

inline std::ostream& operator<<(std::ostream& stream,
                                const Manifold::Error& error) {
  return stream << ToString(error);
}
#endif
/** @} */
}  // namespace manifold
