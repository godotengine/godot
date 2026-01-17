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
#include <map>

#include "collider.h"
#include "manifold/common.h"
#include "manifold/manifold.h"
#include "shared.h"
#include "vec.h"

namespace manifold {

/** @ingroup Private */
struct Manifold::Impl {
  struct Relation {
    int originalID = -1;
    mat3x4 transform = la::identity;
    bool backSide = false;
  };
  struct MeshRelationD {
    /// The originalID of this Manifold if it is an original; -1 otherwise.
    int originalID = -1;
    std::map<int, Relation> meshIDtransform;
    Vec<TriRef> triRef;
  };
  struct BaryIndices {
    int tri, start4, end4;
  };

  Box bBox_;
  double epsilon_ = -1;
  double tolerance_ = -1;
  int numProp_ = 0;
  Error status_ = Error::NoError;
  Vec<vec3> vertPos_;
  Vec<Halfedge> halfedge_;
  Vec<double> properties_;
  // Note that vertNormal_ is not precise due to the use of an approximated acos
  // function
  Vec<vec3> vertNormal_;
  Vec<vec3> faceNormal_;
  Vec<vec4> halfedgeTangent_;
  MeshRelationD meshRelation_;
  Collider collider_;

  static std::atomic<uint32_t> meshIDCounter_;
  static uint32_t ReserveIDs(uint32_t);

  Impl() {}
  enum class Shape { Tetrahedron, Cube, Octahedron };
  Impl(Shape, const mat3x4 = la::identity);

  template <typename Precision, typename I>
  Impl(const MeshGLP<Precision, I>& meshGL) {
    const uint32_t numVert = meshGL.NumVert();
    const uint32_t numTri = meshGL.NumTri();

    if (numVert == 0 && numTri == 0) {
      MakeEmpty(Error::NoError);
      return;
    }

    if (numVert < 4 || numTri < 4) {
      MakeEmpty(Error::NotManifold);
      return;
    }

    if (meshGL.numProp < 3) {
      MakeEmpty(Error::MissingPositionProperties);
      return;
    }

    if (meshGL.mergeFromVert.size() != meshGL.mergeToVert.size()) {
      MakeEmpty(Error::MergeVectorsDifferentLengths);
      return;
    }

    if (!meshGL.runTransform.empty() &&
        12 * meshGL.runOriginalID.size() != meshGL.runTransform.size()) {
      MakeEmpty(Error::TransformWrongLength);
      return;
    }

    if (!meshGL.runOriginalID.empty() && !meshGL.runIndex.empty() &&
        meshGL.runOriginalID.size() + 1 != meshGL.runIndex.size() &&
        meshGL.runOriginalID.size() != meshGL.runIndex.size()) {
      MakeEmpty(Error::RunIndexWrongLength);
      return;
    }

    if (!meshGL.faceID.empty() && meshGL.faceID.size() != meshGL.NumTri()) {
      MakeEmpty(Error::FaceIDWrongLength);
      return;
    }

    if (!manifold::all_of(meshGL.vertProperties.begin(),
                          meshGL.vertProperties.end(),
                          [](Precision x) { return std::isfinite(x); })) {
      MakeEmpty(Error::NonFiniteVertex);
      return;
    }

    if (!manifold::all_of(meshGL.runTransform.begin(),
                          meshGL.runTransform.end(),
                          [](Precision x) { return std::isfinite(x); })) {
      MakeEmpty(Error::InvalidConstruction);
      return;
    }

    if (!manifold::all_of(meshGL.halfedgeTangent.begin(),
                          meshGL.halfedgeTangent.end(),
                          [](Precision x) { return std::isfinite(x); })) {
      MakeEmpty(Error::InvalidConstruction);
      return;
    }

    std::vector<int> prop2vert;
    if (!meshGL.mergeFromVert.empty()) {
      prop2vert.resize(numVert);
      std::iota(prop2vert.begin(), prop2vert.end(), 0);
      for (size_t i = 0; i < meshGL.mergeFromVert.size(); ++i) {
        const uint32_t from = meshGL.mergeFromVert[i];
        const uint32_t to = meshGL.mergeToVert[i];
        if (from >= numVert || to >= numVert) {
          MakeEmpty(Error::MergeIndexOutOfBounds);
          return;
        }
        prop2vert[from] = to;
      }
    }

    const auto numProp = meshGL.numProp - 3;
    numProp_ = numProp;
    properties_.resize_nofill(meshGL.NumVert() * numProp);
    tolerance_ = meshGL.tolerance;
    // This will have unreferenced duplicate positions that will be removed by
    // Impl::RemoveUnreferencedVerts().
    vertPos_.resize_nofill(meshGL.NumVert());

    for (size_t i = 0; i < meshGL.NumVert(); ++i) {
      for (const int j : {0, 1, 2})
        vertPos_[i][j] = meshGL.vertProperties[meshGL.numProp * i + j];
      for (size_t j = 0; j < numProp; ++j)
        properties_[i * numProp + j] =
            meshGL.vertProperties[meshGL.numProp * i + 3 + j];
    }

    halfedgeTangent_.resize_nofill(meshGL.halfedgeTangent.size() / 4);
    for (size_t i = 0; i < halfedgeTangent_.size(); ++i) {
      for (const int j : {0, 1, 2, 3})
        halfedgeTangent_[i][j] = meshGL.halfedgeTangent[4 * i + j];
    }

    Vec<TriRef> triRef;
    triRef.resize_nofill(meshGL.NumTri());

    auto runIndex = meshGL.runIndex;
    const auto runEnd = meshGL.triVerts.size();
    if (runIndex.empty()) {
      runIndex = {0, static_cast<I>(runEnd)};
    } else if (runIndex.size() == meshGL.runOriginalID.size()) {
      runIndex.push_back(runEnd);
    } else if (runIndex.size() == 1) {
      runIndex.push_back(runEnd);
    }

    const auto startID =
        Impl::ReserveIDs(std::max(1_uz, meshGL.runOriginalID.size()));
    auto runOriginalID = meshGL.runOriginalID;
    if (runOriginalID.empty()) {
      runOriginalID.push_back(startID);
    }
    for (size_t i = 0; i < runOriginalID.size(); ++i) {
      const int meshID = startID + i;
      const int originalID = runOriginalID[i];
      for (size_t tri = runIndex[i] / 3; tri < runIndex[i + 1] / 3; ++tri) {
        TriRef& ref = triRef[tri];
        ref.meshID = meshID;
        ref.originalID = originalID;
        ref.faceID = meshGL.faceID.empty() ? -1 : meshGL.faceID[tri];
        ref.coplanarID = tri;
      }

      if (meshGL.runTransform.empty()) {
        meshRelation_.meshIDtransform[meshID] = {originalID};
      } else {
        const Precision* m = meshGL.runTransform.data() + 12 * i;
        meshRelation_.meshIDtransform[meshID] = {originalID,
                                                 {{m[0], m[1], m[2]},
                                                  {m[3], m[4], m[5]},
                                                  {m[6], m[7], m[8]},
                                                  {m[9], m[10], m[11]}}};
      }
    }

    Vec<ivec3> triProp;
    triProp.reserve(numTri);
    Vec<ivec3> triVert;
    const bool needsPropMap = numProp > 0 && !prop2vert.empty();
    if (needsPropMap) triVert.reserve(numTri);
    if (triRef.size() > 0) meshRelation_.triRef.reserve(numTri);
    for (size_t i = 0; i < numTri; ++i) {
      ivec3 triP, triV;
      for (const size_t j : {0, 1, 2}) {
        uint32_t vert = (uint32_t)meshGL.triVerts[3 * i + j];
        if (vert >= numVert) {
          MakeEmpty(Error::VertexOutOfBounds);
          return;
        }
        triP[j] = vert;
        triV[j] = prop2vert.empty() ? vert : prop2vert[vert];
      }
      if (triV[0] != triV[1] && triV[1] != triV[2] && triV[2] != triV[0]) {
        if (needsPropMap) {
          triProp.push_back(triP);
          triVert.push_back(triV);
        } else {
          triProp.push_back(triV);
        }
        if (triRef.size() > 0) {
          meshRelation_.triRef.push_back(triRef[i]);
        }
      }
    }

    CreateHalfedges(triProp, triVert);
    if (!IsManifold()) {
      MakeEmpty(Error::NotManifold);
      return;
    }

    CalculateBBox();
    SetEpsilon(-1, std::is_same<Precision, float>::value);

    // we need to split pinched verts before calculating vertex normals, because
    // the algorithm doesn't work with pinched verts
    CleanupTopology();
    CalculateNormals();

    DedupePropVerts();
    MarkCoplanar();

    RemoveDegenerates();
    RemoveUnreferencedVerts();
    Finish();

    if (!IsFinite()) {
      MakeEmpty(Error::NonFiniteVertex);
      return;
    }

    // A Manifold created from an input mesh is never an original - the input is
    // the original.
    meshRelation_.originalID = -1;
  }

  template <typename F>
  inline void ForVert(int halfedge, F func) {
    int current = halfedge;
    do {
      current = NextHalfedge(halfedge_[current].pairedHalfedge);
      func(current);
    } while (current != halfedge);
  }

  template <typename T>
  void ForVert(
      int halfedge, std::function<T(int halfedge)> transform,
      std::function<void(int halfedge, const T& here, T& next)> binaryOp) {
    T here = transform(halfedge);
    int current = halfedge;
    do {
      const int nextHalfedge = NextHalfedge(halfedge_[current].pairedHalfedge);
      T next = transform(nextHalfedge);
      binaryOp(current, here, next);
      here = next;
      current = nextHalfedge;
    } while (current != halfedge);
  }

  void MarkCoplanar();
  void DedupePropVerts();
  void RemoveUnreferencedVerts();
  void InitializeOriginal(bool keepFaceID = false);
  void CreateHalfedges(const Vec<ivec3>& triProp,
                       const Vec<ivec3>& triVert = {});
  void CalculateNormals();
  void IncrementMeshIDs();

  void Update();
  void MakeEmpty(Error status);
  void Warp(std::function<void(vec3&)> warpFunc);
  void WarpBatch(std::function<void(VecView<vec3>)> warpFunc);
  Impl Transform(const mat3x4& transform) const;

  bool IsEmpty() const { return NumTri() == 0; }
  size_t NumVert() const { return vertPos_.size(); }
  size_t NumEdge() const { return halfedge_.size() / 2; }
  size_t NumTri() const { return halfedge_.size() / 3; }
  size_t NumProp() const { return numProp_; }
  size_t NumPropVert() const {
    return NumProp() == 0 ? NumVert() : properties_.size() / NumProp();
  }

  // properties.cpp
  enum class Property { Volume, SurfaceArea };
  double GetProperty(Property prop) const;
  void CalculateCurvature(int gaussianIdx, int meanIdx);
  void CalculateBBox();
  bool IsFinite() const;
  bool IsIndexInBounds(VecView<const ivec3> triVerts) const;
  void SetEpsilon(double minEpsilon = -1, bool useSingle = false);
  bool IsManifold() const;
  bool Is2Manifold() const;
  bool IsSelfIntersecting() const;
  bool MatchesTriNormals() const;
  int NumDegenerateTris() const;
  double MinGap(const Impl& other, double searchLength) const;

  // sort.cpp
  void Finish();
  void SortVerts();
  void ReindexVerts(const Vec<int>& vertNew2Old, size_t numOldVert);
  void CompactProps();
  void GetFaceBoxMorton(Vec<Box>& faceBox, Vec<uint32_t>& faceMorton) const;
  void SortFaces(Vec<Box>& faceBox, Vec<uint32_t>& faceMorton);
  void GatherFaces(const Vec<int>& faceNew2Old);
  void GatherFaces(const Impl& old, const Vec<int>& faceNew2Old);

  // face_op.cpp
  void Face2Tri(const Vec<int>& faceEdge, const Vec<TriRef>& halfedgeRef,
                bool allowConvex = false);
  Polygons Slice(double height) const;
  Polygons Project() const;

  // edge_op.cpp
  void CleanupTopology();
  void SimplifyTopology(int firstNewVert = 0);
  void RemoveDegenerates(int firstNewVert = 0);
  void CollapseShortEdges(int firstNewVert = 0);
  void CollapseColinearEdges(int firstNewVert = 0);
  void SwapDegenerates(int firstNewVert = 0);
  void DedupeEdge(int edge);
  bool CollapseEdge(int edge, std::vector<int>& edges);
  void RecursiveEdgeSwap(int edge, int& tag, std::vector<int>& visited,
                         std::vector<int>& edgeSwapStack,
                         std::vector<int>& edges);
  void RemoveIfFolded(int edge);
  void PairUp(int edge0, int edge1);
  void UpdateVert(int vert, int startEdge, int endEdge);
  void FormLoop(int current, int end);
  void CollapseTri(const ivec3& triEdge);
  void SplitPinchedVerts();
  void DedupeEdges();

  // subdivision.cpp
  int GetNeighbor(int tri) const;
  ivec4 GetHalfedges(int tri) const;
  BaryIndices GetIndices(int halfedge) const;
  void FillRetainedVerts(Vec<Barycentric>& vertBary) const;
  Vec<Barycentric> Subdivide(std::function<int(vec3, vec4, vec4)>,
                             bool = false);

  // smoothing.cpp
  bool IsInsideQuad(int halfedge) const;
  bool IsMarkedInsideQuad(int halfedge) const;
  vec3 GetNormal(int halfedge, int normalIdx) const;
  vec4 TangentFromNormal(const vec3& normal, int halfedge) const;
  std::vector<Smoothness> UpdateSharpenedEdges(
      const std::vector<Smoothness>&) const;
  Vec<bool> FlatFaces() const;
  Vec<int> VertFlatFace(const Vec<bool>&) const;
  Vec<int> VertHalfedge() const;
  std::vector<Smoothness> SharpenEdges(double minSharpAngle,
                                       double minSmoothness) const;
  void SharpenTangent(int halfedge, double smoothness);
  void SetNormals(int normalIdx, double minSharpAngle);
  void LinearizeFlatTangents();
  void DistributeTangents(const Vec<bool>& fixedHalfedges);
  void CreateTangents(int normalIdx);
  void CreateTangents(std::vector<Smoothness>);
  void Refine(std::function<int(vec3, vec4, vec4)>, bool = false);

  // quickhull.cpp
  void Hull(VecView<vec3> vertPos);
};

extern std::mutex dump_lock;
std::ostream& operator<<(std::ostream& stream, const Manifold::Impl& impl);
}  // namespace manifold
