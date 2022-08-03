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

#include "boolean3.h"
#include "csg_tree.h"
#include "impl.h"
#include "par.h"

namespace {
using namespace manifold;
using namespace thrust::placeholders;

ExecutionParams params;

struct MakeTri {
  const Halfedge* halfedges;

  __host__ __device__ void operator()(thrust::tuple<glm::ivec3&, int> inOut) {
    glm::ivec3& tri = thrust::get<0>(inOut);
    const int face = 3 * thrust::get<1>(inOut);

    for (int i : {0, 1, 2}) {
      tri[i] = halfedges[face + i].startVert;
    }
  }
};

Manifold Halfspace(Box bBox, glm::vec3 normal, float originOffset) {
  normal = glm::normalize(normal);
  Manifold cutter =
      Manifold::Cube(glm::vec3(2.0f), true).Translate({1.0f, 0.0f, 0.0f});
  float size = glm::length(bBox.Center() - normal * originOffset) +
               0.5f * glm::length(bBox.Size());
  cutter = cutter.Scale(glm::vec3(size)).Translate({originOffset, 0.0f, 0.0f});
  float yDeg = glm::degrees(-glm::asin(normal.z));
  float zDeg = glm::degrees(glm::atan(normal.y, normal.x));
  return cutter.Rotate(0.0f, yDeg, zDeg);
}
}  // namespace

namespace manifold {

/**
 * Construct an empty Manifold.
 *
 */
Manifold::Manifold() : pNode_{std::make_shared<CsgLeafNode>()} {}
Manifold::~Manifold() = default;
Manifold::Manifold(Manifold&&) noexcept = default;
Manifold& Manifold::operator=(Manifold&&) noexcept = default;

Manifold::Manifold(const Manifold& other) : pNode_(other.pNode_) {}

Manifold::Manifold(std::shared_ptr<CsgNode> pNode) : pNode_(pNode) {}

Manifold::Manifold(std::shared_ptr<Impl> pImpl_)
    : pNode_(std::make_shared<CsgLeafNode>(pImpl_)) {}

Manifold& Manifold::operator=(const Manifold& other) {
  if (this != &other) {
    pNode_ = other.pNode_;
  }
  return *this;
}

CsgLeafNode& Manifold::GetCsgLeafNode() const {
  if (pNode_->GetNodeType() != CsgNodeType::LEAF) {
    pNode_ = pNode_->ToLeafNode();
  }
  return *std::static_pointer_cast<CsgLeafNode>(pNode_);
}

/**
 * Convert a Mesh into a Manifold. Will throw a topologyErr exception if the
 * input is not an oriented 2-manifold. Will collapse degenerate triangles and
 * unnecessary vertices.
 *
 * The three optional inputs should all be specified if any are. These define
 * any properties you may have on this mesh. These properties are not saved in
 * the Manifold, but rather used to determine which coplanar triangles can be
 * safely merged due to all properties being colinear. Any edges that define
 * property boundaries will be retained in the output of arbitrary Boolean
 * operations so that these properties can be properly reapplied to the result
 * using the MeshRelation.
 *
 * @param mesh The input Mesh.
 * @param triProperties A vector of the same length as triVerts, filled with
 * references to the properties index. Note the same vertex can have different
 * properties in different triangles.
 * @param properties A vector whose length is the largest index in
 * triProperties times the length of propertyTolerance (the number of
 * properties). Think of it as a propery matrix indexed as [index *
 * numProperties + propertyNum].
 * @param propertyTolerance A vector of precision values for each property.
 * This is the amount of interpolation error allowed before two neighboring
 * triangles are considered not coplanar. A good place to start is 1e-5 times
 * the largest value you expect this property to take.
 */
Manifold::Manifold(const Mesh& mesh,
                   const std::vector<glm::ivec3>& triProperties,
                   const std::vector<float>& properties,
                   const std::vector<float>& propertyTolerance)
    : pNode_(std::make_shared<CsgLeafNode>(std::make_shared<Impl>(
          mesh, triProperties, properties, propertyTolerance))) {}

/**
 * This returns a Mesh of simple vectors of vertices and triangles suitable for
 * saving or other operations outside of the context of this library.
 */
Mesh Manifold::GetMesh() const {
  const Impl& impl = *GetCsgLeafNode().GetImpl();

  Mesh result;
  result.vertPos.insert(result.vertPos.end(), impl.vertPos_.begin(),
                        impl.vertPos_.end());
  result.vertNormal.insert(result.vertNormal.end(), impl.vertNormal_.begin(),
                           impl.vertNormal_.end());
  result.halfedgeTangent.insert(result.halfedgeTangent.end(),
                                impl.halfedgeTangent_.begin(),
                                impl.halfedgeTangent_.end());

  result.triVerts.resize(NumTri());
  // note that `triVerts` is `std::vector`, so we cannot use thrust::device
  thrust::for_each_n(thrust::host, zip(result.triVerts.begin(), countAt(0)),
                     NumTri(), MakeTri({impl.halfedge_.cptrH()}));

  return result;
}

int Manifold::circularSegments_ = 0;
float Manifold::circularAngle_ = 10.0f;
float Manifold::circularEdgeLength_ = 1.0f;

/**
 * Sets an angle constraint the default number of circular segments for the
 * Cylinder(), Sphere(), and Revolve() constructors. The number of segments will
 * be rounded up to the nearest factor of four.
 *
 * @param angle The minimum angle in degrees between consecutive segments. The
 * angle will increase if the the segments hit the minimum edge length. Default
 * is 10 degrees.
 */
void Manifold::SetMinCircularAngle(float angle) {
  if (angle <= 0) return;
  Manifold::circularAngle_ = angle;
}

/**
 * Sets a length constraint the default number of circular segments for the
 * Cylinder(), Sphere(), and Revolve() constructors. The number of segments will
 * be rounded up to the nearest factor of four.
 *
 * @param length The minimum length of segments. The length will
 * increase if the the segments hit the minimum angle. Default is 1.0.
 */
void Manifold::SetMinCircularEdgeLength(float length) {
  if (length <= 0) return;
  Manifold::circularEdgeLength_ = length;
}

/**
 * Sets the default number of circular segments for the
 * Cylinder(), Sphere(), and Revolve() constructors. Overrides the edge length
 * and angle constraints and sets the number of segements to exactly this value.
 *
 * @param number Number of circular segments. Default is 0, meaning no
 * constraint is applied.
 */
void Manifold::SetCircularSegments(int number) {
  if (number < 3 && number != 0) return;
  Manifold::circularSegments_ = number;
}

/**
 * Determine the result of the SetMinCircularAngle(),
 * SetMinCircularEdgeLength(), and SetCircularSegments() defaults.
 *
 * @param radius For a given radius of circle, determine how many default
 * segments there will be.
 */
int Manifold::GetCircularSegments(float radius) {
  if (Manifold::circularSegments_ > 0) return Manifold::circularSegments_;
  int nSegA = 360.0f / Manifold::circularAngle_;
  int nSegL = 2.0f * radius * glm::pi<float>() / Manifold::circularEdgeLength_;
  int nSeg = fmin(nSegA, nSegL) + 3;
  nSeg -= nSeg % 4;
  return nSeg;
}

/**
 * Does the Manifold have any triangles?
 */
bool Manifold::IsEmpty() const { return GetCsgLeafNode().GetImpl()->IsEmpty(); }
/**
 * Returns the reason for an input Mesh producing an empty Manifold. This Status
 * only applies to Manifolds newly-created from an input Mesh - once they are
 * combined into a new Manifold via operations, the status reverts to NO_ERROR,
 * simply processing the problem mesh as empty. Likewise, empty meshes may still
 * show NO_ERROR, for instance if they are small enough relative to their
 * precision to be collapsed to nothing.
 */
Manifold::Error Manifold::Status() const {
  return GetCsgLeafNode().GetImpl()->status_;
}
/**
 * The number of vertices in the Manifold.
 */
int Manifold::NumVert() const { return GetCsgLeafNode().GetImpl()->NumVert(); }
/**
 * The number of edges in the Manifold.
 */
int Manifold::NumEdge() const { return GetCsgLeafNode().GetImpl()->NumEdge(); }
/**
 * The number of triangles in the Manifold.
 */
int Manifold::NumTri() const { return GetCsgLeafNode().GetImpl()->NumTri(); }

/**
 * Returns the axis-aligned bounding box of all the Manifold's vertices.
 */
Box Manifold::BoundingBox() const { return GetCsgLeafNode().GetBoundingBox(); }

/**
 * Returns the precision of this Manifold's vertices, which tracks the
 * approximate rounding error over all the transforms and operations that have
 * led to this state. Any triangles that are colinear within this precision are
 * considered degenerate and removed. This is the value of &epsilon; defining
 * [&epsilon;-valid](https://github.com/elalish/manifold/wiki/Manifold-Library#definition-of-%CE%B5-valid).
 */
float Manifold::Precision() const {
  return GetCsgLeafNode().GetImpl()->precision_;
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
 * Returns the surface area and volume of the manifold. These properties are
 * clamped to zero for a given face if they are within the Precision(). This
 * means degenerate manifolds can by identified by testing these properties as
 * == 0.
 */
Properties Manifold::GetProperties() const {
  return GetCsgLeafNode().GetImpl()->GetProperties();
}

/**
 * Curvature is the inverse of the radius of curvature, and signed such that
 * positive is convex and negative is concave. There are two orthogonal
 * principal curvatures at any point on a manifold, with one maximum and the
 * other minimum. Gaussian curvature is their product, while mean
 * curvature is their sum. This approximates them for every vertex (returned as
 * vectors in the structure) and also returns their minimum and maximum values.
 */
Curvature Manifold::GetCurvature() const {
  return GetCsgLeafNode().GetImpl()->GetCurvature();
}

/**
 * Gets the relationship to the previous meshes, for the purpose of assinging
 * properties like texture coordinates. The triBary vector is the same length as
 * Mesh.triVerts: BaryRef.originalID indicates the source mesh and BaryRef.tri
 * is that mesh's triangle index to which these barycentric coordinates refer.
 * BaryRef.vertBary gives an index for each vertex into the barycentric vector
 * if that index is >= 0, indicating it is a new vertex. If the index is < 0,
 * this indicates it is an original vertex, the index + 3 vert of the referenced
 * triangle.
 *
 * BaryRef.meshID is a unique ID to the particular instance of a given mesh. For
 * instance, if you want to convert the triangle mesh to a polygon mesh, all the
 * triangles from a given face will have the same .meshID and .tri values.
 */
MeshRelation Manifold::GetMeshRelation() const {
  MeshRelation out;
  const auto& relation = GetCsgLeafNode().GetImpl()->meshRelation_;
  out.triBary.insert(out.triBary.end(), relation.triBary.begin(),
                     relation.triBary.end());
  out.barycentric.insert(out.barycentric.end(), relation.barycentric.begin(),
                         relation.barycentric.end());
  return out;
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
 * If you copy a manifold, but you want this new copy to have new properties
 * (e.g. a different UV mapping), you can reset its meshIDs to a new original,
 * meaning it will now be referenced by its descendents instead of the meshes it
 * was built from, allowing you to differentiate the copies when applying your
 * properties to the final result.
 *
 * This function also condenses all coplanar faces in the relation, and
 * collapses those edges. If you want to have inconsistent properties across
 * these faces, meaning you want to preserve some of these edges, you should
 * instead call GetMesh(), calculate your properties and use these to construct
 * a new manifold.
 */
Manifold Manifold::AsOriginal() const {
  auto newImpl = std::make_shared<Impl>(*GetCsgLeafNode().GetImpl());
  newImpl->InitializeNewReference();
  newImpl->SimplifyTopology();
  newImpl->Finish();
  return Manifold(std::make_shared<CsgLeafNode>(newImpl));
}

/**
 * Should always be true. Also checks saneness of the internal data structures.
 */
bool Manifold::IsManifold() const {
  return GetCsgLeafNode().GetImpl()->Is2Manifold();
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
int Manifold::NumDegenerateTris() const {
  return GetCsgLeafNode().GetImpl()->NumDegenerateTris();
}

/**
 * This is a checksum-style verification of the collider, simply returning the
 * total number of edge-face bounding box overlaps between this and other.
 *
 * @param other A Manifold to overlap with.
 */
int Manifold::NumOverlaps(const Manifold& other) const {
  SparseIndices overlaps = GetCsgLeafNode().GetImpl()->EdgeCollisions(
      *other.GetCsgLeafNode().GetImpl());
  int num_overlaps = overlaps.size();

  overlaps = other.GetCsgLeafNode().GetImpl()->EdgeCollisions(
      *GetCsgLeafNode().GetImpl());
  return num_overlaps += overlaps.size();
}

/**
 * Move this Manifold in space. This operation can be chained. Transforms are
 * combined and applied lazily.
 *
 * @param v The vector to add to every vertex.
 */
Manifold Manifold::Translate(glm::vec3 v) const {
  return Manifold(pNode_->Translate(v));
}

/**
 * Scale this Manifold in space. This operation can be chained. Transforms are
 * combined and applied lazily.
 *
 * @param v The vector to multiply every vertex by per component.
 */
Manifold Manifold::Scale(glm::vec3 v) const {
  return Manifold(pNode_->Scale(v));
}

/**
 * Applys an Euler angle rotation to the manifold, first about the X axis, then
 * Y, then Z, in degrees. We use degrees so that we can minimize rounding error,
 * and elimiate it completely for any multiples of 90 degrees. Addtionally, more
 * efficient code paths are used to update the manifold when the transforms only
 * rotate by multiples of 90 degrees. This operation can be chained. Transforms
 * are combined and applied lazily.
 *
 * @param xDegrees First rotation, degrees about the X-axis.
 * @param yDegrees Second rotation, degrees about the Y-axis.
 * @param zDegrees Third rotation, degrees about the Z-axis.
 */
Manifold Manifold::Rotate(float xDegrees, float yDegrees,
                          float zDegrees) const {
  return Manifold(pNode_->Rotate(xDegrees, yDegrees, zDegrees));
}

/**
 * Transform this Manifold in space. The first three columns form a 3x3 matrix
 * transform and the last is a translation vector. This operation can be
 * chained. Transforms are combined and applied lazily.
 *
 * @param m The affine transform matrix to apply to all the vertices.
 */
Manifold Manifold::Transform(const glm::mat4x3& m) const {
  return Manifold(pNode_->Transform(m));
}

/**
 * This function does not change the topology, but allows the vertices to be
 * moved according to any arbitrary input function. It is easy to create a
 * function that warps a geometrically valid object into one which overlaps, but
 * that is not checked here, so it is up to the user to choose their function
 * with discretion.
 *
 * @param warpFunc A function that modifies a given vertex position.
 */
Manifold Manifold::Warp(std::function<void(glm::vec3&)> warpFunc) const {
  auto pImpl = std::make_shared<Impl>(*GetCsgLeafNode().GetImpl());
  thrust::for_each_n(thrust::host, pImpl->vertPos_.begin(), NumVert(),
                     warpFunc);
  pImpl->Update();
  pImpl->faceNormal_.resize(0);  // force recalculation of triNormal
  pImpl->CalculateNormals();
  pImpl->SetPrecision();
  return Manifold(std::make_shared<CsgLeafNode>(pImpl));
}

/**
 * Increase the density of the mesh by splitting every edge into n pieces. For
 * instance, with n = 2, each triangle will be split into 4 triangles. These
 * will all be coplanar (and will not be immediately collapsed) unless the
 * Mesh/Manifold has halfedgeTangents specified (e.g. from the Smooth()
 * constructor), in which case the new vertices will be moved to the
 * interpolated surface according to their barycentric coordinates.
 *
 * @param n The number of pieces to split every edge into. Must be > 1.
 */
Manifold Manifold::Refine(int n) const {
  auto pImpl = std::make_shared<Impl>(*GetCsgLeafNode().GetImpl());
  pImpl->Refine(n);
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
  std::vector<std::shared_ptr<CsgNode>> children({pNode_, second.pNode_});
  return Manifold(std::make_shared<CsgOpNode>(children, op));
}

/**
 * Shorthand for Boolean Union.
 */
Manifold Manifold::operator+(const Manifold& Q) const {
  return Boolean(Q, OpType::ADD);
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
  return Boolean(Q, OpType::SUBTRACT);
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
  return Boolean(Q, OpType::INTERSECT);
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

  Boolean3 boolean(*impl1, *impl2, OpType::SUBTRACT);
  auto result1 = std::make_shared<CsgLeafNode>(
      std::make_unique<Impl>(boolean.Result(OpType::INTERSECT)));
  auto result2 = std::make_shared<CsgLeafNode>(
      std::make_unique<Impl>(boolean.Result(OpType::SUBTRACT)));
  return std::make_pair(Manifold(result1), Manifold(result2));
}

/**
 * Convient version of Split() for a half-space.
 *
 * @param normal This vector is normal to the cutting plane and its length does
 * not matter. The first result is in the direction of this vector, the second
 * result is on the opposite side.
 * @param originOffset The distance of the plane from the origin in the
 * direction of the normal vector.
 */
std::pair<Manifold, Manifold> Manifold::SplitByPlane(glm::vec3 normal,
                                                     float originOffset) const {
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
Manifold Manifold::TrimByPlane(glm::vec3 normal, float originOffset) const {
  return *this ^ Halfspace(BoundingBox(), normal, originOffset);
}

ExecutionParams& ManifoldParams() { return params; }
}  // namespace manifold
