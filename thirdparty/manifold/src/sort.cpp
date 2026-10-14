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

#include <atomic>
#include <set>

#include "disjoint_sets.h"
#include "impl.h"
#include "parallel.h"
#include "shared.h"

namespace {
using namespace manifold;

constexpr uint32_t kNoCode = 0xFFFFFFFFu;

uint32_t MortonCode(vec3 position, Box bBox) {
  // Unreferenced vertices are marked NaN, and this will sort them to the end
  // (the Morton code only uses the first 30 of 32 bits).
  if (std::isnan(position.x)) return kNoCode;

  return Collider::MortonCode(position, bBox);
}

struct ReindexFace {
  Halfedges& halfedge;
  VecView<vec4> halfedgeTangent;
  const Halfedges& oldHalfedge;
  VecView<const vec4> oldHalfedgeTangent;
  VecView<const int> faceNew2Old;
  VecView<const int> faceOld2New;

  void operator()(int newFace) {
    const int oldFace = faceNew2Old[newFace];
    for (const int i : {0, 1, 2}) {
      const int oldEdge = 3 * oldFace + i;
      Halfedge edge = oldHalfedge.Get(oldEdge);
      const int pairedFace = edge.pairedHalfedge / 3;
      const int offset = edge.pairedHalfedge - 3 * pairedFace;
      edge.pairedHalfedge = 3 * faceOld2New[pairedFace] + offset;
      const int newEdge = 3 * newFace + i;
      halfedge.Set(newEdge, edge.startVert, edge.pairedHalfedge, edge.propVert);
      if (!oldHalfedgeTangent.empty()) {
        halfedgeTangent[newEdge] = oldHalfedgeTangent[oldEdge];
      }
    }
  }
};

template <typename Precision, typename I>
bool MergeMeshGLP(MeshGLP<Precision, I>& mesh) {
  ZoneScoped;
  std::multiset<std::pair<int, int>> openEdges;

  std::vector<int> merge(mesh.NumVert());
  std::iota(merge.begin(), merge.end(), 0);
  for (size_t i = 0; i < mesh.mergeFromVert.size(); ++i) {
    merge[mesh.mergeFromVert[i]] = mesh.mergeToVert[i];
  }

  const auto numVert = mesh.NumVert();
  const auto numTri = mesh.NumTri();
  const int next[3] = {1, 2, 0};
  for (size_t tri = 0; tri < numTri; ++tri) {
    for (int i : {0, 1, 2}) {
      auto edge = std::make_pair(merge[mesh.triVerts[3 * tri + next[i]]],
                                 merge[mesh.triVerts[3 * tri + i]]);
      auto it = openEdges.find(edge);
      if (it == openEdges.end()) {
        std::swap(edge.first, edge.second);
        openEdges.insert(edge);
      } else {
        openEdges.erase(it);
      }
    }
  }
  if (openEdges.empty()) {
    return false;
  }

  const auto numOpenVert = openEdges.size();
  Vec<int> openVerts(numOpenVert);
  int i = 0;
  for (const auto& edge : openEdges) {
    const int vert = edge.first;
    openVerts[i++] = vert;
  }

  Vec<Precision> vertPropD(mesh.vertProperties);
  Box bBox;
  for (const int i : {0, 1, 2}) {
    auto iPos =
        StridedRange(vertPropD.begin() + i, vertPropD.end(), mesh.numProp);
    auto minMax = manifold::transform_reduce(
        iPos.begin(), iPos.end(),
        std::make_pair(std::numeric_limits<double>::infinity(),
                       -std::numeric_limits<double>::infinity()),
        [](auto a, auto b) {
          return std::make_pair(std::min(a.first, b.first),
                                std::max(a.second, b.second));
        },
        [](double f) { return std::make_pair(f, f); });
    bBox.min[i] = minMax.first;
    bBox.max[i] = minMax.second;
  }

  const double tolerance = std::max(static_cast<double>(mesh.tolerance),
                                    (std::is_same<Precision, float>::value
                                         ? std::numeric_limits<float>::epsilon()
                                         : kPrecision) *
                                        bBox.Scale());

  auto policy = autoPolicy(numOpenVert, 1e5);
  Vec<Box> vertBox(numOpenVert);
  Vec<uint32_t> vertMorton(numOpenVert);

  for_each_n(policy, countAt(0), numOpenVert,
             [&vertMorton, &vertBox, &openVerts, &bBox, &mesh,
              tolerance](const int i) {
               int vert = openVerts[i];

               const vec3 center(mesh.vertProperties[mesh.numProp * vert],
                                 mesh.vertProperties[mesh.numProp * vert + 1],
                                 mesh.vertProperties[mesh.numProp * vert + 2]);

               vertBox[i].min = center - tolerance / 2.0;
               vertBox[i].max = center + tolerance / 2.0;

               vertMorton[i] = MortonCode(center, bBox);
             });

  Vec<int> vertNew2Old(numOpenVert);
  sequence(vertNew2Old.begin(), vertNew2Old.end());

  stable_sort(vertNew2Old.begin(), vertNew2Old.end(),
              [&vertMorton](const int& a, const int& b) {
                return vertMorton[a] < vertMorton[b];
              });

  Permute(vertMorton, vertNew2Old);
  Permute(vertBox, vertNew2Old);
  Permute(openVerts, vertNew2Old);

  Collider collider(vertBox, vertMorton);
  DisjointSets uf(numVert);

  auto f = [&uf, &openVerts](int a, int b) {
    return uf.unite(openVerts[a], openVerts[b]);
  };
  auto recorder = MakeSimpleRecorder(f);
  collider.Collisions<true>(recorder, vertBox.cview(), false);

  for (size_t i = 0; i < mesh.mergeFromVert.size(); ++i) {
    uf.unite(static_cast<int>(mesh.mergeFromVert[i]),
             static_cast<int>(mesh.mergeToVert[i]));
  }

  mesh.mergeToVert.clear();
  mesh.mergeFromVert.clear();
  for (size_t v = 0; v < numVert; ++v) {
    const size_t mergeTo = uf.find(v);
    if (mergeTo != v) {
      mesh.mergeFromVert.push_back(v);
      mesh.mergeToVert.push_back(mergeTo);
    }
  }

  return true;
}

}  // namespace

namespace manifold {

/**
 * Once halfedge_ has been filled in, this function can be called to create the
 * rest of the internal data structures. This function also removes the verts
 * and halfedges flagged for removal (NaN verts and -1 halfedges).
 */
void Manifold::Impl::SortGeometry(ExecutionContext::Impl* ctx) {
  if (halfedge_.size() == 0) {
    collider_ = {};
    return;
  }

  // Invariant: every ctx-passing parallel op is followed by IsCancelled to
  // keep partial output from feeding unconditional downstream consumers.

  halfedge_.MakeUnique();
  SortVerts(ctx);
  if (IsCancelled(ctx)) return;
  Vec<Box> faceBox;
  Vec<uint32_t> faceMorton;
  GetFaceBoxMorton(faceBox, faceMorton, ctx);
  if (IsCancelled(ctx)) return;
  SortFaces(faceBox, faceMorton, ctx);
  if (IsCancelled(ctx)) return;
  if (halfedge_.size() == 0) {
    collider_ = {};
    return;
  }
  collider_ = Collider(faceBox, faceMorton);
  bBox_ = collider_.GetBoundingBox();
  CompactProps(ctx);
  if (IsCancelled(ctx)) return;

  DEBUG_ASSERT(halfedge_.size() % 6 == 0, topologyErr,
               "Not an even number of faces after sorting faces!");

#ifdef MANIFOLD_DEBUG
  if (ManifoldParams().intermediateChecks) {
    auto MaxOrMinus = [](int a, int b) {
      return std::min(a, b) < 0 ? -1 : std::max(a, b);
    };
    int face = 0;
    Halfedge extrema = {0, 0, 0};
    for (size_t i = 0; i < halfedge_.size(); i++) {
      const int start =
          halfedge_.IsForward(i) ? halfedge_.Start(i) : halfedge_.End(i);
      const int end =
          halfedge_.IsForward(i) ? halfedge_.End(i) : halfedge_.Start(i);
      extrema.startVert = std::min(extrema.startVert, start);
      extrema.endVert = std::min(extrema.endVert, end);
      extrema.pairedHalfedge =
          MaxOrMinus(extrema.pairedHalfedge, halfedge_.Pair(i));
      face = MaxOrMinus(face, i / 3);
    }
    DEBUG_ASSERT(extrema.startVert >= 0, topologyErr,
                 "Vertex index is negative!");
    DEBUG_ASSERT(extrema.endVert < static_cast<int>(NumVert()), topologyErr,
                 "Vertex index exceeds number of verts!");
    DEBUG_ASSERT(extrema.pairedHalfedge >= 0, topologyErr,
                 "Halfedge index is negative!");
    DEBUG_ASSERT(extrema.pairedHalfedge < 2 * static_cast<int>(NumEdge()),
                 topologyErr, "Halfedge index exceeds number of halfedges!");
    DEBUG_ASSERT(face >= 0, topologyErr, "Face index is negative!");
    DEBUG_ASSERT(face < static_cast<int>(NumTri()), topologyErr,
                 "Face index exceeds number of faces!");
  }
#endif

  DEBUG_ASSERT(meshRelation_.triRef.size() == NumTri() ||
                   meshRelation_.triRef.size() == 0,
               logicErr, "Mesh Relation doesn't fit!");

  DEBUG_ASSERT(Is2Manifold(), logicErr, "mesh is not 2-manifold!");
}

/**
 * Sorts the vertices according to their Morton code.
 */
void Manifold::Impl::SortVerts(ExecutionContext::Impl* ctx) {
  ZoneScoped;
  // Invariant: every ctx-passing parallel op is followed by IsCancelled to
  // keep partial output from feeding unconditional downstream consumers.
  const auto numVert = NumVert();
  Vec<uint32_t> vertMorton(numVert);
  auto policy = autoPolicy(numVert, 1e5);
  for_each_n(policy, countAt(0), numVert, ctx,
             [this, &vertMorton](const int vert) {
               vertMorton[vert] = MortonCode(vertPos_[vert], bBox_);
             });
  if (IsCancelled(ctx)) return;

  Vec<int> vertNew2Old(numVert);
  sequence(vertNew2Old.begin(), vertNew2Old.end());

  stable_sort(vertNew2Old.begin(), vertNew2Old.end(),
              [&vertMorton](const int& a, const int& b) {
                return vertMorton[a] < vertMorton[b];
              });

  ReindexVerts(vertNew2Old, numVert, ctx);
  if (IsCancelled(ctx)) return;

  // Verts were flagged for removal with NaNs and assigned kNoCode to sort
  // them to the end, which allows them to be removed.
  const auto newNumVert =
      std::lower_bound(vertNew2Old.begin(), vertNew2Old.end(), kNoCode,
                       [&vertMorton](const int vert, const uint32_t val) {
                         return vertMorton[vert] < val;
                       }) -
      vertNew2Old.begin();

  vertNew2Old.resize(newNumVert);
  Permute(vertPos_, vertNew2Old);

  if (vertNormal_.size() == numVert) {
    Permute(vertNormal_, vertNew2Old);
  }
}

/**
 * Updates the halfedges to point to new vert indices based on a mapping,
 * vertNew2Old. This may be a subset, so the total number of original verts is
 * also given.
 */
void Manifold::Impl::ReindexVerts(const Vec<int>& vertNew2Old,
                                  size_t oldNumVert,
                                  ExecutionContext::Impl* ctx) {
  ZoneScoped;
  // Invariant: every ctx-passing parallel op is followed by IsCancelled to
  // keep partial output from feeding unconditional downstream consumers.
  Vec<int> vertOld2New(oldNumVert);
  scatter(countAt(0), countAt(static_cast<int>(NumVert())), vertNew2Old.begin(),
          vertOld2New.begin());
  const bool hasProp = NumProp() > 0;
  for_each_n(autoPolicy(oldNumVert, 1e5), countAt(0), halfedge_.size(), ctx,
             [this, &vertOld2New, hasProp](int idx) {
               const int startVert = halfedge_.Start(idx);
               if (startVert < 0) return;
               const int newStart = vertOld2New[startVert];
               halfedge_.SetStart(idx, newStart);
               if (!hasProp) {
                 halfedge_.SetProp(idx, newStart);
               }
             });
  if (IsCancelled(ctx)) return;
}

/**
 * Removes unreferenced property verts and reindexes propVerts.
 */
void Manifold::Impl::CompactProps(ExecutionContext::Impl* ctx) {
  ZoneScoped;
  if (numProp_ == 0) return;
  // Invariant: every ctx-passing parallel op is followed by IsCancelled to
  // keep partial output from feeding unconditional downstream consumers.

  const int numProp = NumProp();
  const auto numVerts = properties_.size() / numProp;
  Vec<int> keep(numVerts, 0);
  auto policy = autoPolicy(numVerts, 1e5);

  for_each_n(policy, countAt(0), halfedge_.size(), ctx, [this, &keep](int idx) {
    reinterpret_cast<std::atomic<int>*>(&keep[halfedge_.Prop(idx)])
        ->store(1, std::memory_order_relaxed);
  });
  if (IsCancelled(ctx)) return;
  Vec<int> propOld2New(numVerts + 1, 0);
  inclusive_scan(keep.begin(), keep.end(), propOld2New.begin() + 1);

  Vec<double> oldProp = properties_;
  const int numVertsNew = propOld2New[numVerts];
  auto& properties = properties_;
  properties.resize_nofill(numProp * numVertsNew);
  for_each_n(
      policy, countAt(0), numVerts, ctx,
      [&properties, &oldProp, &propOld2New, &keep, &numProp](const int oldIdx) {
        if (keep[oldIdx] == 0) return;
        for (int p = 0; p < numProp; ++p) {
          properties[propOld2New[oldIdx] * numProp + p] =
              oldProp[oldIdx * numProp + p];
        }
      });
  if (IsCancelled(ctx)) return;
  for_each_n(policy, countAt(0), halfedge_.size(), ctx,
             [this, &propOld2New](int idx) {
               halfedge_.SetProp(idx, propOld2New[halfedge_.Prop(idx)]);
             });
  if (IsCancelled(ctx)) return;
}

/**
 * Fills the faceBox and faceMorton input with the bounding boxes and Morton
 * codes of the faces, respectively. The Morton code is based on the center of
 * the bounding box.
 */
void Manifold::Impl::GetFaceBoxMorton(Vec<Box>& faceBox,
                                      Vec<uint32_t>& faceMorton,
                                      ExecutionContext::Impl* ctx) const {
  ZoneScoped;
  // Invariant: every ctx-passing parallel op is followed by IsCancelled to
  // keep partial output from feeding unconditional downstream consumers.
  // faceBox should be initialized
  faceBox.resize(NumTri(), Box());
  faceMorton.resize_nofill(NumTri());
  for_each_n(autoPolicy(NumTri(), 1e5), countAt(0), NumTri(), ctx,
             [this, &faceBox, &faceMorton](const int face) {
               // Removed tris are marked by all halfedges having pairedHalfedge
               // = -1, and this will sort them to the end (the Morton code only
               // uses the first 30 of 32 bits).
               if (halfedge_.Pair(3 * face) < 0) {
                 faceMorton[face] = kNoCode;
                 return;
               }

               vec3 center(0.0);

               for (const int i : {0, 1, 2}) {
                 const vec3 pos = vertPos_[halfedge_.Start(3 * face + i)];
                 center += pos;
                 faceBox[face].Union(pos);
               }
               center /= 3;

               faceMorton[face] = MortonCode(center, bBox_);
             });
  if (IsCancelled(ctx)) return;
}

/**
 * Sorts the faces of this manifold according to their input Morton code. The
 * bounding box and Morton code arrays are also sorted accordingly.
 */
void Manifold::Impl::SortFaces(Vec<Box>& faceBox, Vec<uint32_t>& faceMorton,
                               ExecutionContext::Impl* ctx) {
  ZoneScoped;
  // Invariant: every ctx-passing parallel op is followed by IsCancelled to
  // keep partial output from feeding unconditional downstream consumers.
  Vec<int> faceNew2Old(NumTri());
  sequence(faceNew2Old.begin(), faceNew2Old.end());

  stable_sort(faceNew2Old.begin(), faceNew2Old.end(),
              [&faceMorton](const int& a, const int& b) {
                return faceMorton[a] < faceMorton[b];
              });

  // Tris were flagged for removal with pairedHalfedge = -1 and assigned kNoCode
  // to sort them to the end, which allows them to be removed.
  const int newNumTri =
      std::lower_bound(faceNew2Old.begin(), faceNew2Old.end(), kNoCode,
                       [&faceMorton](const int face, const uint32_t val) {
                         return faceMorton[face] < val;
                       }) -
      faceNew2Old.begin();
  faceNew2Old.resize(newNumTri);

  Permute(faceMorton, faceNew2Old);
  Permute(faceBox, faceNew2Old);
  GatherFaces(faceNew2Old, ctx);
  if (IsCancelled(ctx)) return;
}

/**
 * Creates the halfedge_ vector for this manifold by copying a set of faces from
 * another manifold, given by oldHalfedge. Input faceNew2Old defines the old
 * faces to gather into this.
 */
void Manifold::Impl::GatherFaces(const Vec<int>& faceNew2Old,
                                 ExecutionContext::Impl* ctx) {
  ZoneScoped;
  // Invariant: every ctx-passing parallel op is followed by IsCancelled to
  // keep partial output from feeding unconditional downstream consumers.
  const auto numTri = faceNew2Old.size();
  if (meshRelation_.triRef.size() == NumTri())
    Permute(meshRelation_.triRef, faceNew2Old);
  if (faceNormal_.size() == NumTri()) Permute(faceNormal_, faceNew2Old);

  Halfedges oldHalfedge(std::move(halfedge_));
  halfedge_ = Halfedges();
  Vec<vec4> oldHalfedgeTangent(std::move(halfedgeTangent_));
  Vec<int> faceOld2New(oldHalfedge.size() / 3);
  auto policy = autoPolicy(numTri, 1e5);
  scatter(countAt(0_uz), countAt(numTri), faceNew2Old.begin(),
          faceOld2New.begin());

  halfedge_.resize_nofill(3 * numTri);
  if (oldHalfedgeTangent.size() != 0)
    halfedgeTangent_.resize_nofill(3 * numTri);
  for_each_n(policy, countAt(0), numTri, ctx,
             ReindexFace{halfedge_, halfedgeTangent_, oldHalfedge,
                         oldHalfedgeTangent, faceNew2Old, faceOld2New});
  if (IsCancelled(ctx)) return;
}

void Manifold::Impl::GatherFaces(const Impl& old, const Vec<int>& faceNew2Old,
                                 ExecutionContext::Impl* ctx) {
  ZoneScoped;
  // Invariant: every ctx-passing parallel op is followed by IsCancelled to
  // keep partial output from feeding unconditional downstream consumers.
  const auto numTri = faceNew2Old.size();

  meshRelation_.triRef.resize_nofill(numTri);
  gather(faceNew2Old.begin(), faceNew2Old.end(),
         old.meshRelation_.triRef.begin(), meshRelation_.triRef.begin());

  for (const auto& pair : old.meshRelation_.meshIDtransform) {
    meshRelation_.meshIDtransform[pair.first] = pair.second;
  }

  if (old.NumProp() > 0) {
    numProp_ = old.numProp_;
    properties_ = old.properties_;
  }

  if (old.faceNormal_.size() == old.NumTri()) {
    faceNormal_.resize_nofill(numTri);
    gather(faceNew2Old.begin(), faceNew2Old.end(), old.faceNormal_.begin(),
           faceNormal_.begin());
  }

  Vec<int> faceOld2New(old.NumTri());
  scatter(countAt(0_uz), countAt(numTri), faceNew2Old.begin(),
          faceOld2New.begin());

  halfedge_.resize_nofill(3 * numTri);
  if (old.halfedgeTangent_.size() != 0)
    halfedgeTangent_.resize_nofill(3 * numTri);
  for_each_n(autoPolicy(numTri, 1e5), countAt(0), numTri, ctx,
             ReindexFace{halfedge_, halfedgeTangent_, old.halfedge_,
                         old.halfedgeTangent_, faceNew2Old, faceOld2New});
  if (IsCancelled(ctx)) return;
}

void Manifold::Impl::ReorderHalfedges(ExecutionContext::Impl* ctx) {
  ZoneScoped;
  // Invariant: every ctx-passing parallel op is followed by IsCancelled to
  // keep partial output from feeding unconditional downstream consumers.
  // halfedges in the same face are added in non-deterministic order, so we have
  // to reorder them for determinism

  // step 1: reorder within the same face, such that the halfedge with the
  // smallest starting vertex is placed first
  for_each(autoPolicy(halfedge_.size() / 3), countAt(0_uz),
           countAt(halfedge_.size() / 3), ctx, [this](size_t tri) {
             std::array<Halfedge, 3> face = {halfedge_.Get(tri * 3),
                                             halfedge_.Get(tri * 3 + 1),
                                             halfedge_.Get(tri * 3 + 2)};
             if (face[0].startVert < 0) return;
             int index = 0;
             for (int i : {1, 2})
               if (face[i].startVert < face[index].startVert) index = i;
             for (int i : {0, 1, 2}) {
               const auto& f = face[(index + i) % 3];
               halfedge_.Set(tri * 3 + i, f.startVert, f.pairedHalfedge,
                             f.propVert);
             }
           });
  if (IsCancelled(ctx)) return;
  // step 2: fix paired halfedge
  for_each(autoPolicy(halfedge_.size() / 3), countAt(0_uz),
           countAt(halfedge_.size() / 3), ctx, [this](size_t tri) {
             for (int i : {0, 1, 2}) {
               const int currIdx = tri * 3 + i;
               const int startVert = halfedge_.Start(currIdx);
               if (startVert < 0) return;
               int oppositeFace = halfedge_.Pair(currIdx) / 3;
               int index = -1;
               for (int j : {0, 1, 2})
                 if (startVert == halfedge_.End(oppositeFace * 3 + j))
                   index = j;
               halfedge_.SetPair(currIdx, oppositeFace * 3 + index);
             }
           });
  if (IsCancelled(ctx)) return;
}

/**
 * Updates the mergeFromVert and mergeToVert vectors in order to create a
 * manifold solid. If the MeshGL is already manifold, no change will occur and
 * the function will return false. Otherwise, this will merge verts along open
 * edges within tolerance (the maximum of the MeshGL tolerance and the
 * baseline bounding-box tolerance), keeping any from the existing merge
 * vectors, and return true.
 *
 * There is no guarantee the result will be manifold - this is a best-effort
 * helper function designed primarily to aid in the case where a manifold
 * multi-material MeshGL was produced, but its merge vectors were lost due to
 * a round-trip through a file format. Constructing a Manifold from the result
 * will report an error status if it is not manifold.
 */
template <>
bool MeshGL::Merge() {
  return MergeMeshGLP(*this);
}

template <>
bool MeshGL64::Merge() {
  return MergeMeshGLP(*this);
}

}  // namespace manifold
