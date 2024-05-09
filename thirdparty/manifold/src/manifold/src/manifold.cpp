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
#include <map>
#include <numeric>

#include "QuickHull.hpp"
#include "boolean3.h"
#include "csg_tree.h"
#include "impl.h"
#include "par.h"
#include "tri_dist.h"

namespace {
using namespace manifold;
using namespace thrust::placeholders;

ExecutionParams manifoldParams;

struct MakeTri {
  VecView<const Halfedge> halfedges;

  void operator()(thrust::tuple<glm::ivec3&, int> inOut) {
    glm::ivec3& tri = thrust::get<0>(inOut);
    const int face = 3 * thrust::get<1>(inOut);

    for (int i : {0, 1, 2}) {
      tri[i] = halfedges[face + i].startVert;
    }
  }
};

struct UpdateProperties {
  float* properties;
  const int numProp;
  const float* oldProperties;
  const int numOldProp;
  const glm::vec3* vertPos;
  const glm::ivec3* triProperties;
  const Halfedge* halfedges;
  std::function<void(float*, glm::vec3, const float*)> propFunc;

  void operator()(int tri) {
    for (int i : {0, 1, 2}) {
      const int vert = halfedges[3 * tri + i].startVert;
      const int propVert = triProperties[tri][i];
      propFunc(properties + numProp * propVert, vertPos[vert],
               oldProperties + numOldProp * propVert);
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

Manifold Manifold::Invalid() {
  auto pImpl_ = std::make_shared<Impl>();
  pImpl_->status_ = Error::InvalidConstruction;
  return Manifold(pImpl_);
}

Manifold& Manifold::operator=(const Manifold& other) {
  if (this != &other) {
    pNode_ = other.pNode_;
  }
  return *this;
}

CsgLeafNode& Manifold::GetCsgLeafNode() const {
  if (pNode_->GetNodeType() != CsgNodeType::Leaf) {
    pNode_ = pNode_->ToLeafNode();
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
 * @param propertyTolerance A vector of precision values for each property
 * beyond position. If specified, the propertyTolerance vector must have size =
 * numProp - 3. This is the amount of interpolation error allowed before two
 * neighboring triangles are considered to be on a property boundary edge.
 * Property boundary edges will be retained across operations even if the
 * triangles are coplanar. Defaults to 1e-5, which works well for most
 * properties in the [-1, 1] range.
 */
Manifold::Manifold(const MeshGL& meshGL,
                   const std::vector<float>& propertyTolerance)
    : pNode_(std::make_shared<CsgLeafNode>(
          std::make_shared<Impl>(meshGL, propertyTolerance))) {}

/**
 * Convert a Mesh into a Manifold. Will return an empty Manifold
 * and set an Error Status if the Mesh is not an oriented 2-manifold. Will
 * collapse degenerate triangles and unnecessary vertices.
 *
 * @param mesh The input Mesh.
 */
Manifold::Manifold(const Mesh& mesh) {
  Impl::MeshRelationD relation = {(int)ReserveIDs(1)};
  pNode_ =
      std::make_shared<CsgLeafNode>(std::make_shared<Impl>(mesh, relation));
}

/**
 * This returns a Mesh of simple vectors of vertices and triangles suitable for
 * saving or other operations outside of the context of this library.
 */
Mesh Manifold::GetMesh() const {
  ZoneScoped;
  const Impl& impl = *GetCsgLeafNode().GetImpl();

  Mesh result;
  result.precision = Precision();
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
                     NumTri(), MakeTri({impl.halfedge_}));

  return result;
}

/**
 * The most complete output of this library, returning a MeshGL that is designed
 * to easily push into a renderer, including all interleaved vertex properties
 * that may have been input. It also includes relations to all the input meshes
 * that form a part of this result and the transforms applied to each.
 *
 * @param normalIdx If the original MeshGL inputs that formed this manifold had
 * properties corresponding to normal vectors, you can specify which property
 * channels these are (x, y, z), which will cause this output MeshGL to
 * automatically update these normals according to the applied transforms and
 * front/back side. Each channel must be >= 3 and < numProp, and all original
 * MeshGLs must use the same channels for their normals.
 */
MeshGL Manifold::GetMeshGL(glm::ivec3 normalIdx) const {
  ZoneScoped;
  const Impl& impl = *GetCsgLeafNode().GetImpl();

  const int numProp = NumProp();
  const int numVert = NumPropVert();
  const int numTri = NumTri();

  const bool isOriginal = impl.meshRelation_.originalID >= 0;
  const bool updateNormals =
      !isOriginal && glm::all(glm::greaterThan(normalIdx, glm::ivec3(2)));

  MeshGL out;
  out.precision = Precision();
  out.numProp = 3 + numProp;
  out.triVerts.resize(3 * numTri);

  const int numHalfedge = impl.halfedgeTangent_.size();
  out.halfedgeTangent.resize(4 * numHalfedge);
  for (int i = 0; i < numHalfedge; ++i) {
    const glm::vec4 t = impl.halfedgeTangent_[i];
    out.halfedgeTangent[4 * i] = t.x;
    out.halfedgeTangent[4 * i + 1] = t.y;
    out.halfedgeTangent[4 * i + 2] = t.z;
    out.halfedgeTangent[4 * i + 3] = t.w;
  }

  // Sort the triangles into runs
  out.faceID.resize(numTri);
  std::vector<int> triNew2Old(numTri);
  std::iota(triNew2Old.begin(), triNew2Old.end(), 0);
  VecView<const TriRef> triRef = impl.meshRelation_.triRef;
  // Don't sort originals - keep them in order
  if (!isOriginal) {
    std::sort(triNew2Old.begin(), triNew2Old.end(), [triRef](int a, int b) {
      return triRef[a].originalID == triRef[b].originalID
                 ? triRef[a].meshID < triRef[b].meshID
                 : triRef[a].originalID < triRef[b].originalID;
    });
  }

  std::vector<glm::mat3> runNormalTransform;

  auto addRun = [updateNormals, isOriginal](
                    MeshGL& out, std::vector<glm::mat3>& runNormalTransform,
                    int tri, const Impl::Relation& rel) {
    out.runIndex.push_back(3 * tri);
    out.runOriginalID.push_back(rel.originalID);
    if (updateNormals) {
      runNormalTransform.push_back(NormalTransform(rel.transform) *
                                   (rel.backSide ? -1.0f : 1.0f));
    }
    if (!isOriginal) {
      for (const int col : {0, 1, 2, 3}) {
        for (const int row : {0, 1, 2}) {
          out.runTransform.push_back(rel.transform[col][row]);
        }
      }
    }
  };

  auto meshIDtransform = impl.meshRelation_.meshIDtransform;
  int lastID = -1;
  for (int tri = 0; tri < numTri; ++tri) {
    const int oldTri = triNew2Old[tri];
    const auto ref = triRef[oldTri];
    const int meshID = ref.meshID;

    out.faceID[tri] = ref.tri;
    for (const int i : {0, 1, 2})
      out.triVerts[3 * tri + i] = impl.halfedge_[3 * oldTri + i].startVert;

    if (meshID != lastID) {
      Impl::Relation rel;
      auto it = meshIDtransform.find(meshID);
      if (it != meshIDtransform.end()) rel = it->second;
      addRun(out, runNormalTransform, tri, rel);
      meshIDtransform.erase(meshID);
      lastID = meshID;
    }
  }
  // Add runs for originals that did not contribute any faces to the output
  for (const auto& pair : meshIDtransform) {
    addRun(out, runNormalTransform, numTri, pair.second);
  }
  out.runIndex.push_back(3 * numTri);

  // Early return for no props
  if (numProp == 0) {
    out.vertProperties.resize(3 * numVert);
    for (int i = 0; i < numVert; ++i) {
      const glm::vec3 v = impl.vertPos_[i];
      out.vertProperties[3 * i] = v.x;
      out.vertProperties[3 * i + 1] = v.y;
      out.vertProperties[3 * i + 2] = v.z;
    }
    return out;
  }

  // Duplicate verts with different props
  std::vector<int> vert2idx(impl.NumVert(), -1);
  std::vector<std::vector<glm::ivec2>> vertPropPair(impl.NumVert());
  out.vertProperties.reserve(numVert * static_cast<size_t>(out.numProp));

  for (int run = 0; run < out.runOriginalID.size(); ++run) {
    for (int tri = out.runIndex[run] / 3; tri < out.runIndex[run + 1] / 3;
         ++tri) {
      const glm::ivec3 triProp =
          impl.meshRelation_.triProperties[triNew2Old[tri]];
      for (const int i : {0, 1, 2}) {
        const int prop = triProp[i];
        const int vert = out.triVerts[3 * tri + i];

        auto& bin = vertPropPair[vert];
        bool bFound = false;
        for (int k = 0; k < bin.size(); ++k) {
          if (bin[k].x == prop) {
            bFound = true;
            out.triVerts[3 * tri + i] = bin[k].y;
            break;
          }
        }
        if (bFound) continue;
        const int idx = out.vertProperties.size() / out.numProp;
        out.triVerts[3 * tri + i] = idx;
        bin.push_back({prop, idx});

        for (int p : {0, 1, 2}) {
          out.vertProperties.push_back(impl.vertPos_[vert][p]);
        }
        for (int p = 0; p < numProp; ++p) {
          out.vertProperties.push_back(
              impl.meshRelation_.properties[prop * numProp + p]);
        }

        if (updateNormals) {
          glm::vec3 normal;
          const int start = out.vertProperties.size() - out.numProp;
          for (int i : {0, 1, 2}) {
            normal[i] = out.vertProperties[start + normalIdx[i]];
          }
          normal = glm::normalize(runNormalTransform[run] * normal);
          for (int i : {0, 1, 2}) {
            out.vertProperties[start + normalIdx[i]] = normal[i];
          }
        }

        if (vert2idx[vert] == -1) {
          vert2idx[vert] = idx;
        } else {
          out.mergeFromVert.push_back(idx);
          out.mergeToVert.push_back(vert2idx[vert]);
        }
      }
    }
  }
  return out;
}

/**
 * Does the Manifold have any triangles?
 */
bool Manifold::IsEmpty() const { return GetCsgLeafNode().GetImpl()->IsEmpty(); }
/**
 * Returns the reason for an input Mesh producing an empty Manifold. This Status
 * only applies to Manifolds newly-created from an input Mesh - once they are
 * combined into a new Manifold via operations, the status reverts to NoError,
 * simply processing the problem mesh as empty. Likewise, empty meshes may still
 * show NoError, for instance if they are small enough relative to their
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
 * The number of properties per vertex in the Manifold.
 */
int Manifold::NumProp() const { return GetCsgLeafNode().GetImpl()->NumProp(); }
/**
 * The number of property vertices in the Manifold. This will always be >=
 * NumVert, as some physical vertices may be duplicated to account for different
 * properties on different neighboring triangles.
 */
int Manifold::NumPropVert() const {
  return GetCsgLeafNode().GetImpl()->NumPropVert();
}

/**
 * Returns the axis-aligned bounding box of all the Manifold's vertices.
 */
Box Manifold::BoundingBox() const { return GetCsgLeafNode().GetImpl()->bBox_; }

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
 * Returns the surface area and volume of the manifold.
 */
Properties Manifold::GetProperties() const {
  return GetCsgLeafNode().GetImpl()->GetProperties();
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
 * This function condenses all coplanar faces in the relation, and
 * collapses those edges. In the process the relation to ancestor meshes is lost
 * and this new Manifold is marked an original. Properties are preserved, so if
 * they do not match across an edge, that edge will be kept.
 */
Manifold Manifold::AsOriginal() const {
  auto newImpl = std::make_shared<Impl>(*GetCsgLeafNode().GetImpl());
  newImpl->meshRelation_.originalID = ReserveIDs(1);
  newImpl->InitializeOriginal();
  newImpl->CreateFaces();
  newImpl->SimplifyTopology();
  newImpl->Finish();
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
  return num_overlaps + overlaps.size();
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
 * Applies an Euler angle rotation to the manifold, first about the X axis, then
 * Y, then Z, in degrees. We use degrees so that we can minimize rounding error,
 * and eliminate it completely for any multiples of 90 degrees. Additionally,
 * more efficient code paths are used to update the manifold when the transforms
 * only rotate by multiples of 90 degrees. This operation can be chained.
 * Transforms are combined and applied lazily.
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
 * Mirror this Manifold over the plane described by the unit form of the given
 * normal vector. If the length of the normal is zero, an empty Manifold is
 * returned. This operation can be chained. Transforms are combined and applied
 * lazily.
 *
 * @param normal The normal vector of the plane to be mirrored over
 */
Manifold Manifold::Mirror(glm::vec3 normal) const {
  if (glm::length(normal) == 0.) {
    return Manifold();
  }
  auto n = glm::normalize(normal);
  auto m = glm::mat4x3(glm::mat3(1.0f) - 2.0f * glm::outerProduct(n, n));
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
  pImpl->Warp(warpFunc);
  return Manifold(std::make_shared<CsgLeafNode>(pImpl));
}

/**
 * Same as Manifold::Warp but calls warpFunc with with
 * a VecView which is roughly equivalent to std::span
 * pointing to all vec3 elements to be modified in-place
 *
 * @param warpFunc A function that modifies multiple vertex positions.
 */
Manifold Manifold::WarpBatch(
    std::function<void(VecView<glm::vec3>)> warpFunc) const {
  auto pImpl = std::make_shared<Impl>(*GetCsgLeafNode().GetImpl());
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
 * @param numProp The new number of properties per vertex.
 * @param propFunc A function that modifies the properties of a given vertex.
 */
Manifold Manifold::SetProperties(
    int numProp, std::function<void(float* newProp, glm::vec3 position,
                                    const float* oldProp)>
                     propFunc) const {
  auto pImpl = std::make_shared<Impl>(*GetCsgLeafNode().GetImpl());
  const int oldNumProp = NumProp();
  const Vec<float> oldProperties = pImpl->meshRelation_.properties;

  auto& triProperties = pImpl->meshRelation_.triProperties;
  if (numProp == 0) {
    triProperties.resize(0);
    pImpl->meshRelation_.properties.resize(0);
  } else {
    if (triProperties.size() == 0) {
      const int numTri = NumTri();
      triProperties.resize(numTri);
      int idx = 0;
      for (int i = 0; i < numTri; ++i) {
        for (const int j : {0, 1, 2}) {
          triProperties[i][j] = idx++;
        }
      }
      pImpl->meshRelation_.properties = Vec<float>(numProp * idx, 0);
    } else {
      pImpl->meshRelation_.properties = Vec<float>(numProp * NumPropVert(), 0);
    }
    thrust::for_each_n(
        thrust::host, countAt(0), NumTri(),
        UpdateProperties({pImpl->meshRelation_.properties.data(), numProp,
                          oldProperties.data(), oldNumProp,
                          pImpl->vertPos_.data(), triProperties.data(),
                          pImpl->halfedge_.data(), propFunc}));
  }

  pImpl->meshRelation_.numProp = numProp;
  pImpl->CreateFaces();
  pImpl->Finish();
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
  auto pImpl = std::make_shared<Impl>(*GetCsgLeafNode().GetImpl());
  pImpl->CalculateCurvature(gaussianIdx, meanIdx);
  return Manifold(std::make_shared<CsgLeafNode>(pImpl));
}

/**
 * Fills in vertex properties for normal vectors, calculated from the mesh
 * geometry. Flat faces composed of three or more triangles will remain flat.
 *
 * @param normalIdx The property channel in which to store the X
 * values of the normals. The X, Y, and Z channels will be sequential. The
 * property set will be automatically expanded such that NumProp will be at
 * least normalIdx + 3.
 *
 * @param minSharpAngle Any edges with angles greater than this value will
 * remain sharp, getting different normal vector properties on each side of the
 * edge. By default, no edges are sharp and all normals are shared. With a value
 * of zero, the model is faceted and all normals match their triangle normals,
 * but in this case it would be better not to calculate normals at all.
 */
Manifold Manifold::CalculateNormals(int normalIdx, float minSharpAngle) const {
  auto pImpl = std::make_shared<Impl>(*GetCsgLeafNode().GetImpl());
  pImpl->SetNormals(normalIdx, minSharpAngle);
  return Manifold(std::make_shared<CsgLeafNode>(pImpl));
}

/**
 * Smooths out the Manifold by filling in the halfedgeTangent vectors. The
 * geometry will remain unchanged until Refine or RefineToLength is called to
 * interpolate the surface. This version uses the supplied vertex normal
 * properties to define the tangent vectors. Faces of two coplanar triangles
 * will be marked as quads, while faces with three or more will be flat.
 *
 * @param normalIdx The first property channel of the normals. NumProp must be
 * at least normalIdx + 3. Any vertex where multiple normals exist and don't
 * agree will result in a sharp edge.
 */
Manifold Manifold::SmoothByNormals(int normalIdx) const {
  auto pImpl = std::make_shared<Impl>(*GetCsgLeafNode().GetImpl());
  if (!IsEmpty()) {
    pImpl->CreateTangents(normalIdx);
  }
  return Manifold(std::make_shared<CsgLeafNode>(pImpl));
}

/**
 * Smooths out the Manifold by filling in the halfedgeTangent vectors. The
 * geometry will remain unchanged until Refine or RefineToLength is called to
 * interpolate the surface. This version uses the geometry of the triangles and
 * pseudo-normals to define the tangent vectors. Faces of two coplanar triangles
 * will be marked as quads.
 *
 * @param minSharpAngle degrees, default 60. Any edges with angles greater than
 * this value will remain sharp. The rest will be smoothed to G1 continuity,
 * with the caveat that flat faces of three or more triangles will always remain
 * flat. With a value of zero, the model is faceted, but in this case there is
 * no point in smoothing.
 *
 * @param minSmoothness range: 0 - 1, default 0. The smoothness applied to sharp
 * angles. The default gives a hard edge, while values > 0 will give a small
 * fillet on these sharp edges. A value of 1 is equivalent to a minSharpAngle of
 * 180 - all edges will be smooth.
 */
Manifold Manifold::SmoothOut(float minSharpAngle, float minSmoothness) const {
  auto pImpl = std::make_shared<Impl>(*GetCsgLeafNode().GetImpl());
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
 * their barycentric coordinates.
 *
 * @param n The number of pieces to split every edge into. Must be > 1.
 */
Manifold Manifold::Refine(int n) const {
  auto pImpl = std::make_shared<Impl>(*GetCsgLeafNode().GetImpl());
  if (n > 1) {
    pImpl->Refine([n](glm::vec3 edge) { return n - 1; });
  }
  return Manifold(std::make_shared<CsgLeafNode>(pImpl));
}

/**
 * Increase the density of the mesh by splitting each edge into pieces of
 * roughly the input length. Interior verts are added to keep the rest of the
 * triangulation edges also of roughly the same length. If halfedgeTangents are
 * present (e.g. from the Smooth() constructor), the new vertices will be moved
 * to the interpolated surface according to their barycentric coordinates. Quads
 * will ignore their interior triangle bisector.
 *
 * @param length The length that edges will be broken down to.
 */
Manifold Manifold::RefineToLength(float length) const {
  length = glm::abs(length);
  auto pImpl = std::make_shared<Impl>(*GetCsgLeafNode().GetImpl());
  pImpl->Refine(
      [length](glm::vec3 edge) { return glm::length(edge) / length; });
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
  return Manifold(pNode_->Boolean(second.pNode_, op));
}

/**
 * Perform the given boolean operation on a list of Manifolds. In case of
 * Subtract, all Manifolds in the tail are differenced from the head.
 */
Manifold Manifold::BatchBoolean(const std::vector<Manifold>& manifolds,
                                OpType op) {
  if (manifolds.size() == 0)
    return Manifold();
  else if (manifolds.size() == 1)
    return manifolds[0];
  std::vector<std::shared_ptr<CsgNode>> children;
  children.reserve(manifolds.size());
  for (const auto& m : manifolds) children.push_back(m.pNode_);
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

/**
 * Returns the cross section of this object parallel to the X-Y plane at the
 * specified Z height, defaulting to zero. Using a height equal to the bottom of
 * the bounding box will return the bottom faces, while using a height equal to
 * the top of the bounding box will return empty.
 */
CrossSection Manifold::Slice(float height) const {
  return GetCsgLeafNode().GetImpl()->Slice(height);
}

/**
 * Returns a cross section representing the projected outline of this object
 * onto the X-Y plane.
 */
CrossSection Manifold::Project() const {
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
Manifold Manifold::Hull(const std::vector<glm::vec3>& pts) {
  ZoneScoped;
  const int numVert = pts.size();
  if (numVert < 4) return Manifold();

  std::vector<quickhull::Vector3<double>> vertices(numVert);
  for (int i = 0; i < numVert; i++) {
    vertices[i] = {pts[i].x, pts[i].y, pts[i].z};
  }

  quickhull::QuickHull<double> qh;
  // bools: correct triangle winding, and use original indices
  auto hull = qh.getConvexHull(vertices, false, true);
  const auto& triangles = hull.getIndexBuffer();
  const int numTris = triangles.size() / 3;

  Mesh mesh;
  mesh.vertPos = pts;
  mesh.triVerts.reserve(numTris);
  for (int i = 0; i < numTris; i++) {
    const int j = i * 3;
    mesh.triVerts.push_back({triangles[j], triangles[j + 1], triangles[j + 2]});
  }
  return Manifold(mesh);
}

/**
 * Compute the convex hull of this manifold.
 */
Manifold Manifold::Hull() const { return Hull(GetMesh().vertPos); }

/**
 * Compute the convex hull enveloping a set of manifolds.
 *
 * @param manifolds A vector of manifolds over which to compute a convex hull.
 */
Manifold Manifold::Hull(const std::vector<Manifold>& manifolds) {
  return Compose(manifolds).Hull();
}

/**
 * Returns the minimum gap between two manifolds. Returns a float between
 * 0 and searchLength.
 *
 * @param other The other manifold to compute the minimum gap to.
 * @param searchLength The maximum distance to search for a minimum gap.
 */
float Manifold::MinGap(const Manifold& other, float searchLength) const {
  auto intersect = *this ^ other;
  auto prop = intersect.GetProperties();

  if (prop.volume != 0) return 0.0f;

  return GetCsgLeafNode().GetImpl()->MinGap(*other.GetCsgLeafNode().GetImpl(),
                                            searchLength);
}
}  // namespace manifold
