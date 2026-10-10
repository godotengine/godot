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

#include <algorithm>

#include "boolean3.h"
#include "csg_tree.h"
#include "execution_impl.h"
#include "impl.h"
#include "parallel.h"
#include "shared.h"

namespace {
using namespace manifold;

ExecutionParams manifoldParams;

Manifold Halfspace(Box bBox, vec3 normal, double originOffset) {
  normal = la::normalize(normal);
  Manifold cutter = Manifold::Cube(vec3(2.0), true).Translate({1.0, 0.0, 0.0});
  double size = la::length(bBox.Center() - normal * originOffset) +
                0.5 * la::length(bBox.Size());
  cutter = cutter.Scale(vec3(size)).Translate({originOffset, 0.0, 0.0});
  double yDeg = degrees(-math::asin(normal.z));
  double zDeg = degrees(math::atan2(normal.y, normal.x));
  return cutter.Rotate(0.0, yDeg, zDeg);
}
}  // namespace

namespace manifold {

static int circularSegments_ = DEFAULT_SEGMENTS;
static double circularAngle_ = DEFAULT_ANGLE;
static double circularEdgeLength_ = DEFAULT_LENGTH;

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
void Quality::SetMinCircularAngle(double angle) {
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
void Quality::SetMinCircularEdgeLength(double length) {
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
void Quality::SetCircularSegments(int number) {
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
int Quality::GetCircularSegments(double radius) {
  if (circularSegments_ > 0) return circularSegments_;
  int nSegA = 360.0 / circularAngle_;
  int nSegL = 2.0 * std::abs(radius) * kPi / circularEdgeLength_;
  int nSeg = fmin(nSegA, nSegL) + 3;
  nSeg -= nSeg % 4;
  return std::max(nSeg, 4);
}

/**
 * Resets the circular construction parameters to their defaults if
 * SetMinCircularAngle, SetMinCircularEdgeLength, or SetCircularSegments have
 * been called.
 */
void Quality::ResetToDefaults() {
  circularSegments_ = DEFAULT_SEGMENTS;
  circularAngle_ = DEFAULT_ANGLE;
  circularEdgeLength_ = DEFAULT_LENGTH;
}

/**
 * Construct an empty Manifold.
 *
 */
Manifold::Manifold() : pNode_{std::make_shared<CsgLeafNode>()} {}
Manifold::~Manifold() = default;
Manifold::Manifold(Manifold&&) noexcept = default;
Manifold& Manifold::operator=(Manifold&&) noexcept = default;

Manifold::Manifold(const Manifold& other) {
  std::lock_guard<std::mutex> lock(*other.pNodeMutex_);
  pNode_ = other.pNode_;
  std::atomic_store(&ctx_, std::atomic_load(&other.ctx_));
}

Manifold::Manifold(std::shared_ptr<CsgNode> pNode) : pNode_(pNode) {}

Manifold::Manifold(std::shared_ptr<Impl> pImpl_)
    : pNode_(std::make_shared<CsgLeafNode>(pImpl_)) {}

Manifold Manifold::Invalid() {
  auto pImpl_ = std::make_shared<Impl>();
  pImpl_->status_ = Error::InvalidConstruction;
  return Manifold(pImpl_);
}

Manifold Manifold::PropagateStatus(Error status) {
  auto pImpl = std::make_shared<Impl>();
  pImpl->status_ = status;
  return Manifold(pImpl);
}

Manifold Manifold::FromImpl(std::shared_ptr<Impl> pImpl) {
  return Manifold(std::move(pImpl));
}

Manifold& Manifold::operator=(const Manifold& other) {
  if (this != &other) {
    std::scoped_lock lock(*pNodeMutex_, *other.pNodeMutex_);
    pNode_ = other.pNode_;
    std::atomic_store(&ctx_, std::atomic_load(&other.ctx_));
  }
  return *this;
}

/**
 * Returns a copy of this Manifold with the given ExecutionContext attached.
 * The attachment is consumed by the next eager op on the result (Status,
 * Refine family). Deferred ops produce a result with no attached ctx; raw
 * copy preserves the attachment. See ExecutionContext in common.h for the
 * full model.
 */
Manifold Manifold::WithContext(const ExecutionContext& ctx) const {
  Manifold result = *this;
  std::atomic_store(&result.ctx_, ctx.impl_);
  return result;
}

std::shared_ptr<CsgNode> Manifold::LoadPNode() const {
  std::lock_guard<std::mutex> lock(*pNodeMutex_);
  return pNode_;
}

CsgLeafNode& Manifold::GetCsgLeafNode(ExecutionContext::Impl* ctx) const {
  std::lock_guard<std::mutex> lock(*pNodeMutex_);
  if (ctx != nullptr) {
    // Reset counters to reflect this evaluation. For pre-evaluated leaf
    // Manifolds, NumLeaves() returns 1 so totalBooleans is 0 — no work.
    // This also prevents stale counters from a previous use of the same
    // ctx. The cancel flag is intentionally NOT reset — sticky cancel is
    // part of the documented API contract (see ExecutionContext in
    // common.h).
    const size_t leaves = pNode_->NumLeaves();
    const int booleans = leaves > 0 ? static_cast<int>(leaves - 1) : 0;
    // Reset numerators before denominators so a concurrent `Progress()`
    // observer cannot see (old donePhases / new totalPhases), which could
    // yield Progress > 1.0 transiently when a ctx is reused across evals.
    ctx->doneBooleans.store(0, std::memory_order_relaxed);
    ctx->donePhases.store(0, std::memory_order_relaxed);
    ctx->totalBooleans.store(booleans, std::memory_order_relaxed);
    ctx->totalPhases.store(booleans * kPhasesPerBoolean,
                           std::memory_order_relaxed);
  }
  if (pNode_->GetNodeType() != CsgNodeType::Leaf) {
    pNode_ = pNode_->ToLeafNode(ctx);
  }
  return *std::static_pointer_cast<CsgLeafNode>(pNode_);
}

/**
 * Convert a MeshGL into a Manifold, retaining its properties and merging only
 * the positions according to the merge vectors. Will return an empty Manifold
 * and set an Error Status if the result is not an oriented 2-manifold. Will
 * collapse degenerate triangles and unnecessary vertices.
 *
 * All fields are read, making this structure suitable for a lossless round-trip
 * of data from GetMeshGL. For multi-material input, use ReserveIDs to set a
 * unique originalID for each material, and sort the materials into triangle
 * runs.
 *
 * @param meshGL The input MeshGL.
 */
Manifold::Manifold(const MeshGL& meshGL)
    : pNode_(std::make_shared<CsgLeafNode>(std::make_shared<Impl>(meshGL))) {}

/**
 * Convert a MeshGL into a Manifold, retaining its properties and merging only
 * the positions according to the merge vectors. Will return an empty Manifold
 * and set an Error Status if the result is not an oriented 2-manifold. Will
 * collapse degenerate triangles and unnecessary vertices.
 *
 * All fields are read, making this structure suitable for a lossless round-trip
 * of data from GetMeshGL. For multi-material input, use ReserveIDs to set a
 * unique originalID for each material, and sort the materials into triangle
 * runs.
 *
 * @param meshGL64 The input MeshGL64.
 */
Manifold::Manifold(const MeshGL64& meshGL64)
    : pNode_(std::make_shared<CsgLeafNode>(std::make_shared<Impl>(meshGL64))) {}

/**
 * Returns a MeshGL that is designed
 * to easily push into a renderer, including all interleaved vertex properties
 * that may have been input. It also includes relations to all the input meshes
 * that form a part of this result and the transforms applied to each.
 *
 * @param normalIdx If this manifold has properties corresponding to normal
 * vectors, you can specify the first of the three consecutive property channels
 * forming the (x, y, z) normals, which will cause this output MeshGL to
 * automatically update these normals according to the applied transforms and
 * front/back side. normalIdx + 3 must be <= numProp, and all original meshes
 * must use the same channels for their normals. Default is -1: if
 * `CalculateNormals()` recorded normals at the standard slot, that slot is
 * used automatically; otherwise no normals are applied. If normals are
 * selected, the runTransform matrices will be removed from the output, to
 * avoid them being double-applied when round-tripped.
 * Passing a non-negative `normalIdx` is the legacy interface from before
 * `CalculateNormals` recorded the slot on the Manifold itself; prefer the
 * no-arg form after `CalculateNormals(0)`. The explicit-idx path will be
 * removed in a future release.
 */
MeshGL Manifold::GetMeshGL(int normalIdx) const {
  const Impl& impl = *GetCsgLeafNode().GetImpl();
  if (normalIdx < 0 && impl.AllHaveNormals()) normalIdx = 0;
  return GetMeshGLImpl<float, uint32_t>(impl, normalIdx);
}

/**
 * Returns a MeshGL64 that is designed
 * to easily push into a renderer, including all interleaved vertex properties
 * that may have been input. It also includes relations to all the input meshes
 * that form a part of this result and the transforms applied to each.
 *
 * @param normalIdx If this manifold has properties corresponding to normal
 * vectors, you can specify the first of the three consecutive property channels
 * forming the (x, y, z) normals, which will cause this output MeshGL to
 * automatically update these normals according to the applied transforms and
 * front/back side. normalIdx + 3 must be <= numProp, and all original meshes
 * must use the same channels for their normals. Default is -1: if
 * `CalculateNormals()` recorded normals at the standard slot, that slot is
 * used automatically; otherwise no normals are applied. If normals are
 * selected, the runTransform matrices will be removed from the output, to
 * avoid them being double-applied when round-tripped.
 * Same deprecation note as `GetMeshGL`: prefer the no-arg form after
 * `CalculateNormals(0)`.
 */
MeshGL64 Manifold::GetMeshGL64(int normalIdx) const {
  const Impl& impl = *GetCsgLeafNode().GetImpl();
  if (normalIdx < 0 && impl.AllHaveNormals()) normalIdx = 0;
  return GetMeshGLImpl<double, uint64_t>(impl, normalIdx);
}

/**
 * Does the Manifold have any triangles?
 */
bool Manifold::IsEmpty() const { return GetCsgLeafNode().GetImpl()->IsEmpty(); }
/**
 * Returns the reason for an input Mesh producing an empty Manifold. This Status
 * will carry on through operations like NaN propogation, ensuring an errored
 * mesh doesn't get mysteriously lost. Empty meshes may still show
 * NoError, for instance the intersection of non-overlapping meshes.
 */
Manifold::Error Manifold::Status() const {
  // Routes through any attached ExecutionContext (see WithContext). The
  // atomic_load temporary pins the Impl's lifetime for the duration of the full
  // expression -- through the lazy eval inside GetCsgLeafNode -- so a
  // concurrent op= reseating ctx_ on this Manifold can't free the Impl out
  // from under us.
  return GetCsgLeafNode(std::atomic_load(&ctx_).get()).GetImpl()->status_;
}
/**
 * The number of vertices in the Manifold.
 */
size_t Manifold::NumVert() const {
  return GetCsgLeafNode().GetImpl()->NumVert();
}
/**
 * The number of edges in the Manifold.
 */
size_t Manifold::NumEdge() const {
  return GetCsgLeafNode().GetImpl()->NumEdge();
}
/**
 * The number of triangles in the Manifold.
 */
size_t Manifold::NumTri() const { return GetCsgLeafNode().GetImpl()->NumTri(); }
/**
 * The number of properties per vertex in the Manifold.
 */
size_t Manifold::NumProp() const {
  return GetCsgLeafNode().GetImpl()->NumProp();
}
/**
 * The number of property vertices in the Manifold. This will always be >=
 * NumVert, as some physical vertices may be duplicated to account for different
 * properties on different neighboring triangles.
 */
size_t Manifold::NumPropVert() const {
  return GetCsgLeafNode().GetImpl()->NumPropVert();
}

/**
 * Returns the axis-aligned bounding box of all the Manifold's vertices.
 */
Box Manifold::BoundingBox() const { return GetCsgLeafNode().GetImpl()->bBox_; }

/**
 * Returns the epsilon value of this Manifold's vertices, which tracks the
 * approximate rounding error over all the transforms and operations that have
 * led to this state. This is the value of &epsilon; defining
 * [&epsilon;-valid](https://github.com/elalish/manifold/wiki/Manifold-Library#definition-of-%CE%B5-valid).
 */
double Manifold::GetEpsilon() const {
  return GetCsgLeafNode().GetImpl()->epsilon_;
}

/**
 * Returns the tolerance value of this Manifold. Triangles that are coplanar
 * within tolerance tend to be merged and edges shorter than tolerance tend to
 * be collapsed.
 */
double Manifold::GetTolerance() const {
  return GetCsgLeafNode().GetImpl()->tolerance_;
}

/**
 * Return a copy of the manifold with the set tolerance value.
 * This performs mesh simplification when the tolerance value is increased.
 */
Manifold Manifold::SetTolerance(double tolerance) const {
  auto leafImpl = GetCsgLeafNode().GetImpl();
  if (leafImpl->status_ != Error::NoError)
    return PropagateStatus(leafImpl->status_);
  auto impl = std::make_shared<Impl>(*leafImpl);
  if (tolerance > impl->tolerance_) {
    impl->tolerance_ = tolerance;
    impl->SetNormalsAndCoplanar();
    impl->SimplifyTopology();
    impl->SortGeometry();
  } else {
    // for reducing tolerance, we need to make sure it is still at least
    // equal to epsilon.
    impl->tolerance_ = std::max(impl->epsilon_, tolerance);
  }
  return Manifold(impl);
}

/**
 * Return a copy of the manifold simplified to the given tolerance, but with its
 * actual tolerance value unchanged. If the tolerance is not given or is less
 * than the current tolerance, the current tolerance is used for simplification.
 * The result will contain a subset of the original verts and all surfaces will
 * have moved by less than tolerance.
 */
Manifold Manifold::Simplify(double tolerance) const {
  auto leafImpl = GetCsgLeafNode().GetImpl();
  if (leafImpl->status_ != Error::NoError)
    return PropagateStatus(leafImpl->status_);
  auto impl = std::make_shared<Impl>(*leafImpl);
  const double oldTolerance = impl->tolerance_;
  if (tolerance == 0) tolerance = oldTolerance;
  if (tolerance > oldTolerance) {
    impl->tolerance_ = tolerance;
    impl->SetNormalsAndCoplanar();
  }
  impl->SimplifyTopology();
  impl->SortGeometry();
  impl->tolerance_ = oldTolerance;
  return Manifold(impl);
}

/**
 * The genus is a topological property of the manifold, representing the number
 * of "handles". A sphere is 0, torus 1, etc. It is only meaningful for a single
 * mesh, so it is best to call Decompose() first.
 */
int Manifold::Genus() const {
  int chi = NumVert() - NumEdge() + NumTri();
  return 1 - chi / 2;
}

/**
 * Returns the surface area of the manifold.
 */
double Manifold::SurfaceArea() const {
  return GetCsgLeafNode().GetImpl()->GetProperty(Impl::Property::SurfaceArea);
}

/**
 * Returns the volume of the manifold.
 */
double Manifold::Volume() const {
  return GetCsgLeafNode().GetImpl()->GetProperty(Impl::Property::Volume);
}

/**
 * If this mesh is an original, this returns its meshID that can be referenced
 * by product manifolds' MeshRelation. If this manifold is a product, this
 * returns -1.
 */
int Manifold::OriginalID() const {
  return GetCsgLeafNode().GetImpl()->meshRelation_.originalID;
}

/**
 * This removes all relations (originalID, faceID, transform) to ancestor meshes
 * and this new Manifold is marked an original. It also recreates faces
 * - these don't get joined at boundaries where originalID changes, so the
 * reset may allow triangles of flat faces to be further collapsed with
 * Simplify().
 */
Manifold Manifold::AsOriginal() const {
  auto oldImpl = GetCsgLeafNode().GetImpl();
  if (oldImpl->status_ != Error::NoError)
    return PropagateStatus(oldImpl->status_);
  auto newImpl = std::make_shared<Impl>(*oldImpl);
  newImpl->InitializeOriginal();
  newImpl->SetNormalsAndCoplanar();
  return Manifold(std::make_shared<CsgLeafNode>(newImpl));
}

/**
 * Returns the first of n sequential new unique mesh IDs for marking sets of
 * triangles that can be looked up after further operations. Assign to
 * MeshGL.runOriginalID vector.
 */
uint32_t Manifold::ReserveIDs(uint32_t n) {
  return Manifold::Impl::ReserveIDs(n);
}

/**
 * The triangle normal vectors are saved over the course of operations rather
 * than recalculated to avoid rounding error. This checks that triangles still
 * match their normal vectors within Precision().
 */
bool Manifold::MatchesTriNormals() const {
  return GetCsgLeafNode().GetImpl()->MatchesTriNormals();
}

/**
 * The number of triangles that are colinear within Precision(). This library
 * attempts to remove all of these, but it cannot always remove all of them
 * without changing the mesh by too much.
 */
size_t Manifold::NumDegenerateTris() const {
  return GetCsgLeafNode().GetImpl()->NumDegenerateTris();
}

/**
 * Move this Manifold in space. This operation can be chained. Transforms are
 * combined and applied lazily.
 *
 * @param v The vector to add to every vertex.
 */
Manifold Manifold::Translate(vec3 v) const {
  return Manifold(LoadPNode()->Translate(v));
}

/**
 * Scale this Manifold in space. This operation can be chained. Transforms are
 * combined and applied lazily.
 *
 * @param v The vector to multiply every vertex by per component.
 */
Manifold Manifold::Scale(vec3 v) const {
  return Manifold(LoadPNode()->Scale(v));
}

/**
 * Applies an Euler angle rotation to the manifold, This operation can be
 * chained. Transforms are combined and applied lazily.
 *
 * We use degrees so that we can minimize rounding error, and eliminate it
 * completely for any multiples of 90 degrees. Additionally, more efficient code
 * paths are used to update the manifold when the transforms only rotate by
 * multiples of 90 degrees.
 *
 * From the reference frame of the model being rotated, rotations are applied in
 * *z-y'-x"* order. That is yaw first, then pitch and finally roll.
 *
 * From the global reference frame, a model will be rotated in *x-y-z* order.
 * That is about the global X axis, then global Y axis, and finally global Z.
 *
 * @param xDegrees First rotation, degrees about the global X-axis.
 * @param yDegrees Second rotation, degrees about the global Y-axis.
 * @param zDegrees Third rotation, degrees about the global Z-axis.
 */
Manifold Manifold::Rotate(double xDegrees, double yDegrees,
                          double zDegrees) const {
  return Manifold(LoadPNode()->Rotate(xDegrees, yDegrees, zDegrees));
}

/**
 * Transform this Manifold in space. The first three columns form a 3x3 matrix
 * transform and the last is a translation vector. This operation can be
 * chained. Transforms are combined and applied lazily.
 *
 * @param m The affine transform matrix to apply to all the vertices.
 */
Manifold Manifold::Transform(const mat3x4& m) const {
  return Manifold(LoadPNode()->Transform(m));
}

/**
 * Mirror this Manifold over the plane described by the unit form of the given
 * normal vector. If the length of the normal is zero, an empty Manifold is
 * returned. This operation can be chained. Transforms are combined and applied
 * lazily.
 *
 * @param normal The normal vector of the plane to be mirrored over
 */
Manifold Manifold::Mirror(vec3 normal) const {
  auto leafImpl = GetCsgLeafNode().GetImpl();
  if (leafImpl->status_ != Error::NoError)
    return PropagateStatus(leafImpl->status_);
  if (la::length(normal) == 0.) {
    return Manifold();
  }
  auto n = la::normalize(normal);
  auto m = mat3x4(mat3(la::identity) - 2.0 * la::outerprod(n, n), vec3());
  return Manifold(LoadPNode()->Transform(m));
}

/**
 * This function does not change the topology, but allows the vertices to be
 * moved according to any arbitrary input function. It is easy to create a
 * function that warps a geometrically valid object into one which overlaps, but
 * that is not checked here, so it is up to the user to choose their function
 * with discretion.
 *
 * Any normals recording set by `CalculateNormals()` is preserved across the
 * Warp, but the stored values reflect the pre-warp surface and may no longer
 * match the new geometry. Re-call `CalculateNormals()` if accurate normals
 * matter after a non-rigid warp.
 *
 * @param warpFunc A function that modifies a given vertex position.
 */
Manifold Manifold::Warp(std::function<void(vec3&)> warpFunc) const {
  auto oldImpl = GetCsgLeafNode().GetImpl();
  if (oldImpl->status_ != Error::NoError)
    return PropagateStatus(oldImpl->status_);
  auto pImpl = std::make_shared<Impl>(*oldImpl);
  pImpl->Warp(warpFunc);
  return Manifold(std::make_shared<CsgLeafNode>(pImpl));
}

/**
 * Same as Manifold::Warp but calls warpFunc with
 * a VecView which is roughly equivalent to std::span
 * pointing to all vec3 elements to be modified in-place. Like Warp, this
 * preserves any normals recording without updating the stored values;
 * re-call `CalculateNormals()` if accurate normals matter after a non-rigid
 * warp.
 *
 * @param warpFunc A function that modifies multiple vertex positions.
 */
Manifold Manifold::WarpBatch(
    std::function<void(VecView<vec3>)> warpFunc) const {
  auto oldImpl = GetCsgLeafNode().GetImpl();
  if (oldImpl->status_ != Error::NoError)
    return PropagateStatus(oldImpl->status_);
  auto pImpl = std::make_shared<Impl>(*oldImpl);
  pImpl->WarpBatch(warpFunc);
  return Manifold(std::make_shared<CsgLeafNode>(pImpl));
}

/**
 * Create a new copy of this manifold with updated vertex properties by
 * supplying a function that takes the existing position and properties as
 * input. You may specify any number of output properties, allowing creation and
 * removal of channels. Note: undefined behavior will result if you read past
 * the number of input properties or write past the number of output properties.
 *
 * If propFunc is a nullptr, this function will just set the channel to zeroes.
 *
 * Any normals recording set by `CalculateNormals()` is preserved. If the new
 * properties overwrite slot 0..2 with non-normal data, the recording becomes
 * stale; re-call `CalculateNormals()` (or use a numProp < 3 call followed by
 * CalculateNormals) to reset.
 *
 * @param numProp The new number of properties per vertex.
 * @param propFunc A function that modifies the properties of a given vertex.
 */
Manifold Manifold::SetProperties(
    int numProp,
    std::function<void(double* newProp, vec3 position, const double* oldProp)>
        propFunc) const {
  auto leafImpl = GetCsgLeafNode().GetImpl();
  if (leafImpl->status_ != Error::NoError)
    return PropagateStatus(leafImpl->status_);
  auto pImpl = std::make_shared<Impl>(*leafImpl);
  const int oldNumProp = NumProp();
  const Vec<double> oldProperties = pImpl->properties_;

  if (numProp == 0) {
    pImpl->properties_.clear();
  } else {
    pImpl->properties_ = Vec<double>(numProp * NumPropVert(), 0);
    for_each_n(
        propFunc == nullptr ? ExecutionPolicy::Par : ExecutionPolicy::Seq,
        countAt(0), NumTri(), [&](int tri) {
          for (int i : {0, 1, 2}) {
            const int edge = 3 * tri + i;
            const int vert = pImpl->halfedge_.Start(edge);
            const int propVert = pImpl->halfedge_.Prop(edge);
            if (propFunc == nullptr) {
              for (int p = 0; p < numProp; ++p) {
                pImpl->properties_[numProp * propVert + p] = 0;
              }
            } else {
              propFunc(&pImpl->properties_[numProp * propVert],
                       pImpl->vertPos_[vert],
                       oldProperties.data() + oldNumProp * propVert);
            }
          }
        });
  }

  pImpl->numProp_ = numProp;
  return Manifold(std::make_shared<CsgLeafNode>(pImpl));
}

/**
 * Curvature is the inverse of the radius of curvature, and signed such that
 * positive is convex and negative is concave. There are two orthogonal
 * principal curvatures at any point on a manifold, with one maximum and the
 * other minimum. Gaussian curvature is their product, while mean
 * curvature is their sum. This approximates them for every vertex and assigns
 * them as vertex properties on the given channels.
 *
 * @param gaussianIdx The property channel index in which to store the Gaussian
 * curvature. An index < 0 will be ignored (stores nothing). The property set
 * will be automatically expanded to include the channel index specified.
 *
 * @param meanIdx The property channel index in which to store the mean
 * curvature. An index < 0 will be ignored (stores nothing). The property set
 * will be automatically expanded to include the channel index specified.
 */
Manifold Manifold::CalculateCurvature(int gaussianIdx, int meanIdx) const {
  auto leafImpl = GetCsgLeafNode().GetImpl();
  if (leafImpl->status_ != Error::NoError)
    return PropagateStatus(leafImpl->status_);
  auto pImpl = std::make_shared<Impl>(*leafImpl);
  pImpl->CalculateCurvature(gaussianIdx, meanIdx);
  return Manifold(std::make_shared<CsgLeafNode>(pImpl));
}

/**
 * Fills in vertex properties for normal vectors, calculated from the mesh
 * geometry.
 *
 * @param normalIdx The property channel in which to store the X values of the
 * normals. The X, Y, and Z channels will be sequential. The property set will
 * be automatically expanded such that NumProp will be at least normalIdx + 3.
 * Default is 0, the standard slot (MeshGL channels 3, 4, 5); the Manifold
 * records the recording per-meshID in that case so subsequent `GetMeshGL()` /
 * `GetMeshGL64()` without an explicit normalIdx will return world-frame
 * normals and mark each output run via runFlags bit 1. Non-zero values are
 * retained for compatibility and will not be supported in a future release.
 *
 * @param minSharpAngle Any edges with angles greater than this value will
 * remain sharp, getting different normal vector properties on each side of the
 * edge. By default, no edges are sharp and all normals are shared. With a value
 * of zero, the model is faceted and all normals match their triangle normals,
 * but in this case it would be better not to calculate normals at all.
 */
Manifold Manifold::CalculateNormals(int normalIdx, double minSharpAngle) const {
  auto leafImpl = GetCsgLeafNode().GetImpl();
  if (leafImpl->status_ != Error::NoError)
    return PropagateStatus(leafImpl->status_);
  auto pImpl = std::make_shared<Impl>(*leafImpl);
  pImpl->SetNormals(normalIdx, minSharpAngle);
  // Mark per-meshID hasNormals so GetMeshGL(-1) can auto-substitute slot 0 on
  // export. Restricted to the standard slot since a non-zero slot would be
  // ambiguous when round-tripping through MeshGL.
  if (normalIdx == 0) {
    for (auto& m : pImpl->meshRelation_.meshIDtransform) {
      m.second.hasNormals = true;
    }
  }
  return Manifold(std::make_shared<CsgLeafNode>(pImpl));
}

/**
 * Smooths out the Manifold by filling in the halfedgeTangent vectors. The
 * geometry will remain unchanged until Refine, RefineToLength, or
 * RefineToTolerance is called to interpolate the surface. This version uses the
 * supplied vertex normal properties to define the tangent vectors. Zero-length
 * normals are considered missing and will defer to their neighboring normals
 * instead. If all normals are missing, the vertex pseudonormal will be used.
 *
 * @param normalIdx The first property channel of the normals. NumProp must be
 * at least normalIdx + 3. Any vertex where multiple normals exist and don't
 * agree will result in a sharp edge. Default is 0, the standard slot.
 * Non-zero values are retained for compatibility and will not be supported in
 * a future release.
 */
Manifold Manifold::SmoothByNormals(int normalIdx) const {
  auto leafImpl = GetCsgLeafNode().GetImpl();
  if (leafImpl->status_ != Error::NoError)
    return PropagateStatus(leafImpl->status_);
  auto pImpl = std::make_shared<Impl>(*leafImpl);
  if (!IsEmpty()) {
    pImpl->CreateTangents(normalIdx);
  }
  return Manifold(std::make_shared<CsgLeafNode>(pImpl));
}

/**
 * Smooths out the Manifold by filling in the halfedgeTangent vectors. The
 * geometry will remain unchanged until Refine, RefineToLength, or
 * RefineToTolerance is called to interpolate the surface. This version uses the
 * geometry of the triangles and pseudo-normals to define the tangent vectors.
 * Faces of two coplanar triangles will be marked as quads, while faces with
 * three or more will be flat.
 *
 * @param minSharpAngle degrees, default 52.5. Any edges with angles greater
 * than this value will remain sharp. The rest will be smoothed to G1
 * continuity. With a value of zero, the model is faceted, but in this case
 * there is no point in smoothing.
 *
 * @param minSmoothness range: 0 - 1, default 0. The smoothness applied to sharp
 * angles. The default gives a hard edge, while values > 0 will give a small
 * fillet on these sharp edges. A value of 1 is equivalent to a minSharpAngle of
 * 180 - all edges will be smooth.
 */
Manifold Manifold::SmoothOut(double minSharpAngle, double minSmoothness) const {
  auto leafImpl = GetCsgLeafNode().GetImpl();
  if (leafImpl->status_ != Error::NoError)
    return PropagateStatus(leafImpl->status_);
  auto pImpl = std::make_shared<Impl>(*leafImpl);
  if (!IsEmpty()) {
    pImpl->CreateTangents(pImpl->SharpenEdges(minSharpAngle, minSmoothness));
  }
  return Manifold(std::make_shared<CsgLeafNode>(pImpl));
}

/**
 * Increase the density of the mesh by splitting every edge into n pieces. For
 * instance, with n = 2, each triangle will be split into 4 triangles. Quads
 * will ignore their interior triangle bisector. These will all be coplanar (and
 * will not be immediately collapsed) unless the Mesh/Manifold has
 * halfedgeTangents specified (e.g. from the Smooth() constructor), in which
 * case the new vertices will be moved to the interpolated surface according to
 * their barycentric coordinates. Any normals recording set by
 * `CalculateNormals()` is preserved; the new verts get linearly-interpolated
 * normals, which are less precise than recomputed ones but still meaningful.
 *
 * @param n The number of pieces to split every edge into. Must be > 1.
 */
Manifold Manifold::Refine(int n) const {
  auto ctx = std::atomic_load(&ctx_);
  auto leafImpl = GetCsgLeafNode(ctx.get()).GetImpl();
  if (leafImpl->status_ != Error::NoError)
    return PropagateStatus(leafImpl->status_);
  auto pImpl = std::make_shared<Impl>(*leafImpl);
  if (n > 1) {
    pImpl->Refine([n](vec3, vec4, vec4) { return n - 1; }, false, ctx.get());
  }
  return Manifold(std::make_shared<CsgLeafNode>(pImpl));
}

/**
 * Increase the density of the mesh by splitting each edge into pieces of
 * roughly the input length. Interior verts are added to keep the rest of the
 * triangulation edges also of roughly the same length. If halfedgeTangents are
 * present (e.g. from the Smooth() constructor), the new vertices will be moved
 * to the interpolated surface according to their barycentric coordinates. Quads
 * will ignore their interior triangle bisector. Any normals recording set by
 * `CalculateNormals()` is preserved; the new verts get linearly-interpolated
 * normals, which are less precise than recomputed ones but still meaningful.
 *
 * @param length The length that edges will be broken down to.
 */
Manifold Manifold::RefineToLength(double length) const {
  length = std::abs(length);
  auto ctx = std::atomic_load(&ctx_);
  auto leafImpl = GetCsgLeafNode(ctx.get()).GetImpl();
  if (leafImpl->status_ != Error::NoError)
    return PropagateStatus(leafImpl->status_);
  auto pImpl = std::make_shared<Impl>(*leafImpl);
  pImpl->Refine(
      [length](vec3 edge, vec4, vec4) {
        return static_cast<int>(la::length(edge) / length);
      },
      false, ctx.get());
  return Manifold(std::make_shared<CsgLeafNode>(pImpl));
}

/**
 * Increase the density of the mesh by splitting each edge into pieces such that
 * any point on the resulting triangles is roughly within tolerance of the
 * smoothly curved surface defined by the tangent vectors. This means tightly
 * curving regions will be divided more finely than smoother regions. If
 * halfedgeTangents are not present, the result will simply be a copy of the
 * original. Quads will ignore their interior triangle bisector. Any normals
 * recording set by `CalculateNormals()` is preserved; the new verts get
 * linearly-interpolated normals.
 *
 * @param tolerance The desired maximum distance between the faceted mesh
 * produced and the exact smoothly curving surface. All vertices are exactly on
 * the surface, within rounding error.
 */
Manifold Manifold::RefineToTolerance(double tolerance) const {
  tolerance = std::abs(tolerance);
  auto ctx = std::atomic_load(&ctx_);
  auto leafImpl = GetCsgLeafNode(ctx.get()).GetImpl();
  if (leafImpl->status_ != Error::NoError)
    return PropagateStatus(leafImpl->status_);
  auto pImpl = std::make_shared<Impl>(*leafImpl);
  if (!pImpl->halfedgeTangent_.empty()) {
    pImpl->Refine(
        [tolerance](vec3 edge, vec4 tangentStart, vec4 tangentEnd) {
          const vec3 edgeNorm = la::normalize(edge);
          // Weight heuristic
          const vec3 tStart = vec3(tangentStart);
          const vec3 tEnd = vec3(tangentEnd);
          // Perpendicular to edge
          const vec3 start = tStart - edgeNorm * la::dot(edgeNorm, tStart);
          const vec3 end = tEnd - edgeNorm * la::dot(edgeNorm, tEnd);
          // Circular arc result plus heuristic term for non-circular curves
          const double d = 0.5 * (la::length(start) + la::length(end)) +
                           la::length(start - end);
          return static_cast<int>(std::sqrt(3 * d / (4 * tolerance)));
        },
        true, ctx.get());
  }
  return Manifold(std::make_shared<CsgLeafNode>(pImpl));
}

/**
 * The central operation of this library: the Boolean combines two manifolds
 * into another by calculating their intersections and removing the unused
 * portions.
 * [&epsilon;-valid](https://github.com/elalish/manifold/wiki/Manifold-Library#definition-of-%CE%B5-valid)
 * inputs will produce &epsilon;-valid output. &epsilon;-invalid input may fail
 * triangulation.
 *
 * These operations are optimized to produce nearly-instant results if either
 * input is empty or their bounding boxes do not overlap.
 *
 * @param second The other Manifold.
 * @param op The type of operation to perform.
 */
Manifold Manifold::Boolean(const Manifold& second, OpType op) const {
  return Manifold(LoadPNode()->Boolean(second.LoadPNode(), op));
}

/**
 * Perform the given boolean operation on a list of Manifolds. In case of
 * Subtract, all Manifolds in the tail are differenced from the head. The
 * empty-input case returns a default-constructed Manifold; the
 * single-input case returns the input unchanged (a no-op identity, including
 * any attached ExecutionContext on that single input).
 */
Manifold Manifold::BatchBoolean(const std::vector<Manifold>& manifolds,
                                OpType op) {
  if (manifolds.size() == 0)
    return Manifold();
  else if (manifolds.size() == 1)
    return manifolds[0];
  std::vector<std::shared_ptr<CsgNode>> children;
  children.reserve(manifolds.size());
  for (const auto& m : manifolds) children.push_back(m.LoadPNode());
  return Manifold(std::make_shared<CsgOpNode>(children, op));
}

/**
 * Shorthand for Boolean Union.
 */
Manifold Manifold::operator+(const Manifold& Q) const {
  return Boolean(Q, OpType::Add);
}

/**
 * Shorthand for Boolean Union assignment.
 */
Manifold& Manifold::operator+=(const Manifold& Q) {
  *this = *this + Q;
  return *this;
}

/**
 * Shorthand for Boolean Difference.
 */
Manifold Manifold::operator-(const Manifold& Q) const {
  return Boolean(Q, OpType::Subtract);
}

/**
 * Shorthand for Boolean Difference assignment.
 */
Manifold& Manifold::operator-=(const Manifold& Q) {
  *this = *this - Q;
  return *this;
}

/**
 * Shorthand for Boolean Intersection.
 */
Manifold Manifold::operator^(const Manifold& Q) const {
  return Boolean(Q, OpType::Intersect);
}

/**
 * Shorthand for Boolean Intersection assignment.
 */
Manifold& Manifold::operator^=(const Manifold& Q) {
  *this = *this ^ Q;
  return *this;
}

/**
 * Split cuts this manifold in two using the cutter manifold. The first result
 * is the intersection, second is the difference. This is more efficient than
 * doing them separately.
 *
 * @param cutter
 */
std::pair<Manifold, Manifold> Manifold::Split(const Manifold& cutter) const {
  auto impl1 = GetCsgLeafNode().GetImpl();
  auto impl2 = cutter.GetCsgLeafNode().GetImpl();

  Boolean3 boolean(*impl1, *impl2, OpType::Subtract);
  auto result1 = std::make_shared<CsgLeafNode>(
      std::make_unique<Impl>(boolean.Result(OpType::Intersect)));
  auto result2 = std::make_shared<CsgLeafNode>(
      std::make_unique<Impl>(boolean.Result(OpType::Subtract)));
  return std::make_pair(Manifold(result1), Manifold(result2));
}

/**
 * Convenient version of Split() for a half-space.
 *
 * @param normal This vector is normal to the cutting plane and its length does
 * not matter. The first result is in the direction of this vector, the second
 * result is on the opposite side.
 * @param originOffset The distance of the plane from the origin in the
 * direction of the normal vector.
 */
std::pair<Manifold, Manifold> Manifold::SplitByPlane(
    vec3 normal, double originOffset) const {
  auto leafImpl = GetCsgLeafNode().GetImpl();
  if (leafImpl->status_ != Error::NoError) {
    Manifold err = PropagateStatus(leafImpl->status_);
    return {err, err};
  }
  if (IsEmpty()) return {Manifold(), Manifold()};
  return Split(Halfspace(BoundingBox(), normal, originOffset));
}

/**
 * Identical to SplitByPlane(), but calculating and returning only the first
 * result.
 *
 * @param normal This vector is normal to the cutting plane and its length does
 * not matter. The result is in the direction of this vector from the plane.
 * @param originOffset The distance of the plane from the origin in the
 * direction of the normal vector.
 */
Manifold Manifold::TrimByPlane(vec3 normal, double originOffset) const {
  return *this ^ Halfspace(BoundingBox(), normal, originOffset);
}

/**
 * Compute the minkowski sum of this manifold with another.
 * This corresponds to the morphological dilation of the manifold.
 *
 * @note Performance is best when using convex objects. For non-convex inputs,
 * performance scales with the product of face counts, so keep face counts low.
 *
 * @param other The other manifold to minkowski sum to this one.
 */
Manifold Manifold::MinkowskiSum(const Manifold& other) const {
  auto ctx = std::atomic_load(&ctx_);
  auto aImpl = GetCsgLeafNode(ctx.get()).GetImpl();
  if (aImpl->status_ != Error::NoError) return PropagateStatus(aImpl->status_);
  auto bImpl = other.GetCsgLeafNode(ctx.get()).GetImpl();
  if (bImpl->status_ != Error::NoError) return PropagateStatus(bImpl->status_);
  return aImpl->Minkowski(*bImpl, false, ctx.get());
}

/**
 * Subtract the sweep of the other manifold across this manifold's surface.
 * This corresponds to the morphological erosion of the manifold.
 *
 * @note Performance is best when using convex objects. For non-convex inputs,
 * performance scales with the product of face counts, so keep face counts low.
 *
 * @param other The other manifold to minkowski subtract from this one.
 */
Manifold Manifold::MinkowskiDifference(const Manifold& other) const {
  auto ctx = std::atomic_load(&ctx_);
  auto aImpl = GetCsgLeafNode(ctx.get()).GetImpl();
  if (aImpl->status_ != Error::NoError) return PropagateStatus(aImpl->status_);
  auto bImpl = other.GetCsgLeafNode(ctx.get()).GetImpl();
  if (bImpl->status_ != Error::NoError) return PropagateStatus(bImpl->status_);
  return aImpl->Minkowski(*bImpl, true, ctx.get());
}

/**
 * Returns the cross section of this object parallel to the X-Y plane at the
 * specified Z height, defaulting to zero. Using a height equal to the bottom of
 * the bounding box will return the bottom faces, while using a height equal to
 * the top of the bounding box will return empty.
 */
Polygons Manifold::Slice(double height) const {
  return GetCsgLeafNode().GetImpl()->Slice(height);
}

/**
 * Returns polygons representing the projected outline of this object
 * onto the X-Y plane. These polygons will often self-intersect, so it is
 * recommended to run them through the positive fill rule of CrossSection to get
 * a sensible result before using them.
 */
Polygons Manifold::Project() const {
  return GetCsgLeafNode().GetImpl()->Project();
}

ExecutionParams& ManifoldParams() { return manifoldParams; }

/**
 * Compute the convex hull of a set of points. If the given points are fewer
 * than 4, or they are all coplanar, an empty Manifold will be returned.
 *
 * @param pts A vector of 3-dimensional points over which to compute a convex
 * hull.
 */
Manifold Manifold::Hull(const std::vector<vec3>& pts) {
  std::shared_ptr<Impl> impl = std::make_shared<Impl>();
  impl->Hull(Vec<vec3>(pts));
  return Manifold(std::make_shared<CsgLeafNode>(impl));
}

/**
 * Compute the convex hull of this manifold.
 */
Manifold Manifold::Hull() const {
  auto ctx = std::atomic_load(&ctx_);
  auto srcImpl = GetCsgLeafNode(ctx.get()).GetImpl();
  if (srcImpl->status_ != Error::NoError)
    return PropagateStatus(srcImpl->status_);
  std::shared_ptr<Impl> impl = std::make_shared<Impl>();
  impl->Hull(srcImpl->vertPos_, ctx.get());
  return Manifold(std::make_shared<CsgLeafNode>(impl));
}

/**
 * Compute the convex hull enveloping a set of manifolds. If the input list
 * is empty (or all input manifolds are empty), returns a default-constructed
 * Manifold with no attached ExecutionContext (no source operand to inherit
 * from).
 *
 * @param manifolds A vector of manifolds over which to compute a convex hull.
 */
Manifold Manifold::Hull(const std::vector<Manifold>& manifolds) {
  for (const auto& man : manifolds) {
    auto status = man.Status();
    if (status != Error::NoError) return PropagateStatus(status);
  }
  std::vector<vec3> vertPos;
  size_t size = 0;
  for (const auto& man : manifolds) size += man.NumVert();
  if (size == 0) return Manifold();
  vertPos.reserve(size);
  for (const auto& man : manifolds) {
    const auto& impl = man.GetCsgLeafNode().GetImpl();
    vertPos.insert(vertPos.end(), impl->vertPos_.begin(), impl->vertPos_.end());
  }
  std::shared_ptr<Impl> impl = std::make_shared<Impl>();
  impl->Hull(VecView<const vec3>(vertPos));
  return Manifold(std::make_shared<CsgLeafNode>(impl));
}

/**
 * Returns the minimum gap between two manifolds. Returns a double between
 * 0 and searchLength.
 *
 * @param other The other manifold to compute the minimum gap to.
 * @param searchLength The maximum distance to search for a minimum gap.
 */
double Manifold::MinGap(const Manifold& other, double searchLength) const {
  auto intersect = *this ^ other;
  if (!intersect.IsEmpty()) return 0.0;

  return GetCsgLeafNode().GetImpl()->MinGap(*other.GetCsgLeafNode().GetImpl(),
                                            searchLength);
}

/**
 * Cast a ray segment against this manifold, returning all hits sorted by
 * distance from origin.
 *
 * @param origin The start point of the ray segment.
 * @param endpoint The end point of the ray segment.
 * @return A vector of RayHit sorted by distance, empty on miss.
 */
std::vector<RayHit> Manifold::RayCast(vec3 origin, vec3 endpoint) const {
  return GetCsgLeafNode().GetImpl()->RayCast(origin, endpoint);
}
}  // namespace manifold
