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
#include "execution_impl.h"
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
    // True when this meshID's contribution to properties_ slots 0..2 holds
    // world-frame vertex normals (set by CalculateNormals at slot 0). Carries
    // through Transforms and Booleans. Exported as runFlags bit 1.
    bool hasNormals = false;

    mat3 GetNormalTransform() const {
      return NormalTransform(transform) * (backSide ? -1.0 : 1.0);
    }

    mat3 GetInverseNormalTransform() const {
      return InverseNormalTransform(transform) * (backSide ? -1.0 : 1.0);
    }
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

  // True only when every meshID carries normals at slot 0..2 - the
  // condition under which GetMeshGL(-1) can safely auto-substitute that
  // slot. A mixed Boolean output (some meshIDs with normals, some
  // without) returns false; the output MeshGL's per-run bit 1 still
  // marks the with-normals runs individually.
  bool AllHaveNormals() const {
    if (meshRelation_.meshIDtransform.empty()) return false;
    for (const auto& m : meshRelation_.meshIDtransform) {
      if (!m.second.hasNormals) return false;
    }
    return true;
  }

  // True iff the meshID owning `tri` has hasNormals set. Returns false when
  // the meshID isn't in meshRelation_.meshIDtransform (treat as no-normals).
  static bool TriHasNormals(const MeshRelationD& meshRelation, int tri) {
    const int meshID = meshRelation.triRef[tri].meshID;
    auto it = meshRelation.meshIDtransform.find(meshID);
    return it != meshRelation.meshIDtransform.end() && it->second.hasNormals;
  }
  Vec<vec3> vertPos_;
  Halfedges halfedge_;
  Vec<double> properties_;
  Vec<vec3> vertNormal_;
  Vec<vec3> faceNormal_;
  Vec<vec4> halfedgeTangent_;
  MeshRelationD meshRelation_;
  Collider collider_;

  static std::atomic<uint32_t> meshIDCounter_;
  static uint32_t ReserveIDs(uint32_t);

  Impl() {};
  enum class Shape { Tetrahedron, Cube, Octahedron };
  Impl(Shape, const mat3x4 = la::identity);

  template <typename Precision, typename I>
  Impl(const MeshGLP<Precision, I>& meshGL,
       ExecutionContext::Impl* ctx = nullptr);

  // sdf.cpp. Populates an empty Impl with the level set of `sdf`. `ctx`
  // (when non-null) checks cancel and credits Progress() between the five
  // phases counted by kPhasesPerLevelSet.
  void CreateLevelSet(std::function<double(vec3)> sdf, Box bounds,
                      double edgeLength, double level, double tolerance,
                      bool canParallel, ExecutionContext::Impl* ctx = nullptr);

  template <typename F>
  inline void ForVert(int halfedge, F func);

  template <typename T>
  void ForVert(
      int halfedge, std::function<T(int halfedge)> transform,
      std::function<void(int halfedge, const T& here, T& next)> binaryOp);

  void SetNormalsAndCoplanar();
  void DedupePropVerts();
  void RemoveUnreferencedVerts();
  void InitializeOriginal();
  void CreateHalfedges(const Vec<ivec3>& triProp,
                       const Vec<ivec3>& triVert = {});
  void CalculateVertNormals();
  void IncrementMeshIDs();

  // Eager-transform slot 0..2 of properties_ for propVerts whose meshID
  // carries hasNormals. Used by both Impl::Transform and Compose. The buffer
  // is laid out as `properties[(offset + propVert) * stride + i]` so the
  // helper can target either an in-place properties_ vector (offset=0) or
  // a per-node slice of a combined properties array (offset=propVertIndices,
  // stride=numPropOut).
  static void EagerTransformPropNormals(const Halfedges& halfedge,
                                        const MeshRelationD& meshRelation,
                                        const mat3& normalTransform,
                                        Vec<double>& properties,
                                        int numPropVert, int stride,
                                        int offset = 0);

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
  bool IsConvex() const;
  double MinGap(const Impl& other, double searchLength) const;

  // boolean3.cpp
  std::vector<RayHit> RayCast(vec3 origin, vec3 endpoint) const;

  // sort.cpp
  void SortGeometry(ExecutionContext::Impl* ctx = nullptr);
  void SortVerts(ExecutionContext::Impl* ctx = nullptr);
  void ReindexVerts(const Vec<int>& vertNew2Old, size_t numOldVert,
                    ExecutionContext::Impl* ctx = nullptr);
  void CompactProps(ExecutionContext::Impl* ctx = nullptr);
  void GetFaceBoxMorton(Vec<Box>& faceBox, Vec<uint32_t>& faceMorton,
                        ExecutionContext::Impl* ctx = nullptr) const;
  void SortFaces(Vec<Box>& faceBox, Vec<uint32_t>& faceMorton,
                 ExecutionContext::Impl* ctx = nullptr);
  void GatherFaces(const Vec<int>& faceNew2Old,
                   ExecutionContext::Impl* ctx = nullptr);
  void GatherFaces(const Impl& old, const Vec<int>& faceNew2Old,
                   ExecutionContext::Impl* ctx = nullptr);
  void ReorderHalfedges(ExecutionContext::Impl* ctx = nullptr);

  // face_op.cpp
  void Face2Tri(const Vec<int>& faceEdge, const VecView<const Halfedge>&,
                const Vec<TriRef>& halfedgeRef, bool allowConvex = false,
                ExecutionContext::Impl* ctx = nullptr);
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
  bool CollapseEdge(int edge, Vec<int>& edges, double tol = -1,
                    int firstNewVert = 0);
  void RecursiveEdgeSwap(int edge, int& tag, Vec<int>& visited,
                         Vec<int>& edgeSwapStack, Vec<int>& edges);
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
  bool ValidTangents() const;
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
  void CreateTangents(std::vector<Smoothness>,
                      ExecutionContext::Impl* ctx = nullptr);
  void Refine(std::function<int(vec3, vec4, vec4)>, bool = false,
              ExecutionContext::Impl* ctx = nullptr);

  // quickhull.cpp
  void Hull(VecView<const vec3> vertPos, ExecutionContext::Impl* ctx = nullptr);

  // minkowski.cpp
  Manifold Minkowski(const Impl& other, bool inset,
                     ExecutionContext::Impl* ctx = nullptr) const;
};

extern std::mutex dump_lock;
std::ostream& operator<<(std::ostream& stream, const Manifold::Impl& impl);

// ------------------------------------------------------------------------------
// Template implementations follow:

template <typename F>
inline void Manifold::Impl::ForVert(int halfedge, F func) {
  int current = halfedge;
  do {
    current = NextHalfedge(halfedge_.Pair(current));
    func(current);
  } while (current != halfedge);
}

template <typename T>
void Manifold::Impl::ForVert(
    int halfedge, std::function<T(int halfedge)> transform,
    std::function<void(int halfedge, const T& here, T& next)> binaryOp) {
  T here = transform(halfedge);
  int current = halfedge;
  do {
    const int nextHalfedge = NextHalfedge(halfedge_.Pair(current));
    T next = transform(nextHalfedge);
    binaryOp(current, here, next);
    here = next;
    current = nextHalfedge;
  } while (current != halfedge);
}

template <typename Precision, typename I>
Manifold::Impl::Impl(const MeshGLP<Precision, I>& meshGL,
                     ExecutionContext::Impl* ctx) {
  // Entry-time cancel wins over empty/malformed input; past this gate,
  // validation errors win over races.
  if (IsCancelled(ctx)) {
    MakeEmpty(Error::Cancelled);
    return;
  }

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

  if (!manifold::all_of(meshGL.runTransform.begin(), meshGL.runTransform.end(),
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
    const bool backside = meshGL.Backside(i);
    // Per-run hasNormals (runFlags bit 1). Defensively require numProp >= 3
    // so a caller setting the bit on a too-small MeshGL doesn't make us read
    // past the property bounds.
    const bool runHasN = meshGL.HasNormals(i) && numProp >= 3;
    for (size_t tri = runIndex[i] / 3; tri < runIndex[i + 1] / 3; ++tri) {
      TriRef& ref = triRef[tri];
      ref.meshID = meshID;
      ref.originalID = originalID;
      ref.faceID = meshGL.faceID.empty() ? -1 : meshGL.faceID[tri];
      ref.coplanarID = tri;
    }

    if (meshGL.runTransform.empty()) {
      meshRelation_.meshIDtransform[meshID] = {originalID, la::identity, false,
                                               runHasN};
    } else {
      const Precision* m = meshGL.runTransform.data() + 12 * i;
      meshRelation_.meshIDtransform[meshID] = {originalID,
                                               {{m[0], m[1], m[2]},
                                                {m[3], m[4], m[5]},
                                                {m[6], m[7], m[8]},
                                                {m[9], m[10], m[11]}},
                                               backside,
                                               runHasN};
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

  // Phase boundaries; count of ADVANCE_PHASE_OR_RETURN calls below must
  // equal kPhasesPerFromMesh.
  CreateHalfedges(triProp, triVert);
  if (!IsManifold()) {
    MakeEmpty(Error::NotManifold);
    return;
  }
  ADVANCE_PHASE_OR_RETURN(ctx);

  CalculateBBox();
  SetEpsilon(-1, std::is_same<Precision, float>::value);

  // we need to split pinched verts before calculating vertex normals, because
  // the algorithm doesn't work with pinched verts
  CleanupTopology();
  ADVANCE_PHASE_OR_RETURN(ctx);

  DedupePropVerts();
  ADVANCE_PHASE_OR_RETURN(ctx);

  SetNormalsAndCoplanar();
  ADVANCE_PHASE_OR_RETURN(ctx);

  RemoveDegenerates();
  ADVANCE_PHASE_OR_RETURN(ctx);

  RemoveUnreferencedVerts();
  ADVANCE_PHASE_OR_RETURN(ctx);

  SortGeometry(ctx);
  ADVANCE_PHASE_OR_RETURN(ctx);

  if (!IsFinite()) {
    MakeEmpty(Error::NonFiniteVertex);
    return;
  }

  // A Manifold created from an input mesh is never an original - the input is
  // the original.
  meshRelation_.originalID = -1;
}

template <typename Precision, typename I>
inline MeshGLP<Precision, I> GetMeshGLImpl(const manifold::Manifold::Impl& impl,
                                           int normalIdx) {
  ZoneScoped;
  const int numProp = impl.NumProp();
  const int numVert = impl.NumPropVert();
  const int numTri = impl.NumTri();

  const bool isOriginal = impl.meshRelation_.originalID >= 0;
  const bool updateNormals = !isOriginal && normalIdx >= 0;

  MeshGLP<Precision, I> out;
  out.numProp = 3 + numProp;
  out.tolerance = impl.tolerance_;
  if (std::is_same<Precision, float>::value)
    out.tolerance =
        std::max(out.tolerance,
                 static_cast<Precision>(std::numeric_limits<float>::epsilon() *
                                        impl.bBox_.Scale()));
  out.triVerts.resize(3 * numTri);

  const int numHalfedge = impl.halfedgeTangent_.size();
  out.halfedgeTangent.resize(4 * numHalfedge);
  for (int i = 0; i < numHalfedge; ++i) {
    const vec4 t = impl.halfedgeTangent_[i];
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
    std::stable_sort(triNew2Old.begin(), triNew2Old.end(),
                     [triRef](int a, int b) {
                       return triRef[a].originalID == triRef[b].originalID
                                  ? triRef[a].meshID < triRef[b].meshID
                                  : triRef[a].originalID < triRef[b].originalID;
                     });
  }

  // runFlags layout: bit 0 = backSide, bit 1 = hasNormals (slot 0..2 of the
  // extra properties is world-frame vertex normals; consumers should skip
  // re-applying runTransform to those channels).
  auto addRun = [isOriginal](MeshGLP<Precision, I>& out, int tri,
                             const manifold::Manifold::Impl::Relation& rel) {
    out.runIndex.push_back(3 * tri);
    out.runOriginalID.push_back(rel.originalID);
    // runFlags carries hasNormals (bit 1) which we want on originals too;
    // runTransform is just metadata so skip it for originals where it would
    // always be identity.
    const uint8_t flags = (rel.backSide ? 1u : 0u) | (rel.hasNormals ? 2u : 0u);
    out.runFlags.push_back(flags);
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

    out.faceID[tri] = ref.faceID >= 0 ? ref.faceID : ref.coplanarID;
    for (const int i : {0, 1, 2})
      out.triVerts[3 * tri + i] = impl.halfedge_.Start(3 * oldTri + i);

    if (meshID != lastID) {
      manifold::Manifold::Impl::Relation rel;
      auto it = meshIDtransform.find(meshID);
      if (it != meshIDtransform.end()) rel = it->second;
      addRun(out, tri, rel);
      meshIDtransform.erase(meshID);
      lastID = meshID;
    }
  }
  // Add runs for originals that did not contribute any faces to the output
  for (const auto& pair : meshIDtransform) {
    addRun(out, numTri, pair.second);
  }
  out.runIndex.push_back(3 * numTri);

  // Early return for no props
  if (numProp == 0) {
    out.vertProperties.resize(3 * numVert);
    for (int i = 0; i < numVert; ++i) {
      const vec3 v = impl.vertPos_[i];
      out.vertProperties[3 * i] = v.x;
      out.vertProperties[3 * i + 1] = v.y;
      out.vertProperties[3 * i + 2] = v.z;
    }
    return out;
  }
  // Duplicate verts with different props
  std::vector<int> vert2idx(impl.NumVert(), -1);
  std::vector<std::vector<ivec2>> vertPropPair(impl.NumVert());
  out.vertProperties.reserve(numVert * static_cast<size_t>(out.numProp));

  for (size_t run = 0; run < out.runOriginalID.size(); ++run) {
    for (size_t tri = out.runIndex[run] / 3; tri < out.runIndex[run + 1] / 3;
         ++tri) {
      for (const int i : {0, 1, 2}) {
        const int prop = impl.halfedge_.Prop(3 * triNew2Old[tri] + i);
        const int vert = out.triVerts[3 * tri + i];

        auto& bin = vertPropPair[vert];
        bool bFound = false;
        for (const auto& b : bin) {
          if (b.x == prop) {
            bFound = true;
            out.triVerts[3 * tri + i] = b.y;
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
          out.vertProperties.push_back(impl.properties_[prop * numProp + p]);
        }

        // Normalize the requested normal slot. For runs that already carry
        // world-frame normals (hasNormals bit), just normalize; for legacy
        // callers asking to interpret a slot as normals on a run without
        // hasNormals, apply the per-run inverse-frame transform first.
        // TODO: collapse the !runHasN branch into a no-op once the explicit-
        // normalIdx parameter on GetMeshGL is removed and `updateNormals`
        // becomes implied by the hasNormals bit.
        if (updateNormals) {
          vec3 normal;
          const int start = out.vertProperties.size() - out.numProp;
          for (int i : {0, 1, 2}) {
            normal[i] = out.vertProperties[start + 3 + normalIdx + i];
          }
          const bool runHasN = !isOriginal && (out.runFlags[run] & 2) != 0;
          if (!isOriginal && !runHasN) {
            const Precision* m = out.runTransform.data() + 12 * run;
            const mat3x4 t({m[0], m[1], m[2]}, {m[3], m[4], m[5]},
                           {m[6], m[7], m[8]}, {m[9], m[10], m[11]});
            normal = NormalTransform(t) *
                     ((out.runFlags[run] & 1) ? -1.0 : 1.0) * normal;
          }
          normal = SafeNormalize(normal);
          for (int i : {0, 1, 2}) {
            out.vertProperties[start + 3 + normalIdx + i] = normal[i];
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

// Entry-time cancel wins over empty/malformed input; past this gate,
// validation errors win over races.
template <typename P, typename I>
std::shared_ptr<Manifold::Impl> MakeSmoothImpl(
    const MeshGLP<P, I>& meshGL, const std::vector<Smoothness>& sharpenedEdges,
    ExecutionContext::Impl* ctx = nullptr) {
  if (IsCancelled(ctx)) {
    auto impl = std::make_shared<Manifold::Impl>();
    impl->MakeEmpty(Manifold::Error::Cancelled);
    return impl;
  }

  DEBUG_ASSERT(meshGL.halfedgeTangent.empty(), std::runtime_error,
               "when supplying tangents, the normal constructor should be used "
               "rather than Smooth().");

  MeshGLP<P, I> meshTmp = meshGL;
  meshTmp.faceID.resize(meshGL.NumTri());
  std::iota(meshTmp.faceID.begin(), meshTmp.faceID.end(), 0);

  std::shared_ptr<Manifold::Impl> impl =
      std::make_shared<Manifold::Impl>(meshTmp, ctx);
  // Skip tangent creation if ingest failed; phase counters must not
  // credit smoothing phases that never ran.
  if (impl->status_ != Manifold::Error::NoError) return impl;
  impl->CreateTangents(impl->UpdateSharpenedEdges(sharpenedEdges), ctx);
  // NumTri() is 0 after MakeEmpty, so this loop is a no-op on cancel.
  const size_t numTri = impl->NumTri();
  for (size_t i = 0; i < numTri; ++i) {
    if (meshGL.faceID.size() == numTri) {
      impl->meshRelation_.triRef[i].faceID =
          meshGL.faceID[impl->meshRelation_.triRef[i].faceID];
    } else {
      impl->meshRelation_.triRef[i].faceID = -1;
    }
  }
  return impl;
}
}  // namespace manifold
