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
#include <numeric>
#include <set>

#include "impl.h"
#include "par.h"

namespace {
using namespace manifold;

constexpr uint32_t kNoCode = 0xFFFFFFFFu;

struct Extrema : public thrust::binary_function<Halfedge, Halfedge, Halfedge> {
  void MakeForward(Halfedge& a) {
    if (!a.IsForward()) {
      int tmp = a.startVert;
      a.startVert = a.endVert;
      a.endVert = tmp;
    }
  }

  int MaxOrMinus(int a, int b) {
    return glm::min(a, b) < 0 ? -1 : glm::max(a, b);
  }

  Halfedge operator()(Halfedge a, Halfedge b) {
    MakeForward(a);
    MakeForward(b);
    a.startVert = glm::min(a.startVert, b.startVert);
    a.endVert = glm::max(a.endVert, b.endVert);
    a.face = MaxOrMinus(a.face, b.face);
    a.pairedHalfedge = MaxOrMinus(a.pairedHalfedge, b.pairedHalfedge);
    return a;
  }
};

uint32_t SpreadBits3(uint32_t v) {
  v = 0xFF0000FFu & (v * 0x00010001u);
  v = 0x0F00F00Fu & (v * 0x00000101u);
  v = 0xC30C30C3u & (v * 0x00000011u);
  v = 0x49249249u & (v * 0x00000005u);
  return v;
}

uint32_t MortonCode(glm::vec3 position, Box bBox) {
  // Unreferenced vertices are marked NaN, and this will sort them to the end
  // (the Morton code only uses the first 30 of 32 bits).
  if (isnan(position.x)) return kNoCode;

  glm::vec3 xyz = (position - bBox.min) / (bBox.max - bBox.min);
  xyz = glm::min(glm::vec3(1023.0f), glm::max(glm::vec3(0.0f), 1024.0f * xyz));
  uint32_t x = SpreadBits3(static_cast<uint32_t>(xyz.x));
  uint32_t y = SpreadBits3(static_cast<uint32_t>(xyz.y));
  uint32_t z = SpreadBits3(static_cast<uint32_t>(xyz.z));
  return x * 4 + y * 2 + z;
}

struct Morton {
  const Box bBox;

  void operator()(thrust::tuple<uint32_t&, const glm::vec3&> inout) {
    glm::vec3 position = thrust::get<1>(inout);
    thrust::get<0>(inout) = MortonCode(position, bBox);
  }
};

struct FaceMortonBox {
  VecView<const Halfedge> halfedge;
  VecView<const glm::vec3> vertPos;
  const Box bBox;

  void operator()(thrust::tuple<uint32_t&, Box&, int> inout) {
    uint32_t& mortonCode = thrust::get<0>(inout);
    Box& faceBox = thrust::get<1>(inout);
    int face = thrust::get<2>(inout);

    // Removed tris are marked by all halfedges having pairedHalfedge = -1, and
    // this will sort them to the end (the Morton code only uses the first 30 of
    // 32 bits).
    if (halfedge[3 * face].pairedHalfedge < 0) {
      mortonCode = kNoCode;
      return;
    }

    glm::vec3 center(0.0f);

    for (const int i : {0, 1, 2}) {
      const glm::vec3 pos = vertPos[halfedge[3 * face + i].startVert];
      center += pos;
      faceBox.Union(pos);
    }
    center /= 3;

    mortonCode = MortonCode(center, bBox);
  }
};

struct Reindex {
  VecView<const int> indexInv;

  void operator()(Halfedge& edge) {
    if (edge.startVert < 0) return;
    edge.startVert = indexInv[edge.startVert];
    edge.endVert = indexInv[edge.endVert];
  }
};

struct MarkProp {
  VecView<int> keep;

  void operator()(glm::ivec3 triProp) {
    for (const int i : {0, 1, 2}) {
      reinterpret_cast<std::atomic<int>*>(&keep[triProp[i]])
          ->store(1, std::memory_order_relaxed);
    }
  }
};

struct GatherProps {
  VecView<float> properties;
  VecView<const float> oldProperties;
  const int numProp;

  void operator()(thrust::tuple<int, int, int> in) {
    const int oldIdx = thrust::get<0>(in);
    const int newIdx = thrust::get<1>(in);
    const int keep = thrust::get<2>(in);
    if (keep == 0) return;
    for (int p = 0; p < numProp; ++p) {
      properties[newIdx * numProp + p] = oldProperties[oldIdx * numProp + p];
    }
  }
};

struct ReindexProps {
  VecView<const int> old2new;

  void operator()(glm::ivec3& triProp) {
    for (const int i : {0, 1, 2}) {
      triProp[i] = old2new[triProp[i]];
    }
  }
};

template <typename T>
void Permute(Vec<T>& inOut, const Vec<int>& new2Old) {
  Vec<T> tmp(std::move(inOut));
  inOut.resize(new2Old.size());
  gather(autoPolicy(new2Old.size()), new2Old.begin(), new2Old.end(),
         tmp.begin(), inOut.begin());
}

template void Permute<TriRef>(Vec<TriRef>&, const Vec<int>&);
template void Permute<glm::vec3>(Vec<glm::vec3>&, const Vec<int>&);

struct ReindexFace {
  VecView<Halfedge> halfedge;
  VecView<glm::vec4> halfedgeTangent;
  VecView<const Halfedge> oldHalfedge;
  VecView<const glm::vec4> oldHalfedgeTangent;
  VecView<const int> faceNew2Old;
  VecView<const int> faceOld2New;

  void operator()(int newFace) {
    const int oldFace = faceNew2Old[newFace];
    for (const int i : {0, 1, 2}) {
      const int oldEdge = 3 * oldFace + i;
      Halfedge edge = oldHalfedge[oldEdge];
      edge.face = newFace;
      const int pairedFace = edge.pairedHalfedge / 3;
      const int offset = edge.pairedHalfedge - 3 * pairedFace;
      edge.pairedHalfedge = 3 * faceOld2New[pairedFace] + offset;
      const int newEdge = 3 * newFace + i;
      halfedge[newEdge] = edge;
      if (!oldHalfedgeTangent.empty()) {
        halfedgeTangent[newEdge] = oldHalfedgeTangent[oldEdge];
      }
    }
  }
};

struct VertMortonBox {
  VecView<const float> vertProperties;
  const uint32_t numProp;
  const float tol;
  const Box bBox;

  void operator()(thrust::tuple<uint32_t&, Box&, int> inout) {
    uint32_t& mortonCode = thrust::get<0>(inout);
    Box& vertBox = thrust::get<1>(inout);
    int vert = thrust::get<2>(inout);

    const glm::vec3 center(vertProperties[numProp * vert],
                           vertProperties[numProp * vert + 1],
                           vertProperties[numProp * vert + 2]);

    vertBox.min = center - tol / 2;
    vertBox.max = center + tol / 2;

    mortonCode = MortonCode(center, bBox);
  }
};

struct Duplicate {
  thrust::pair<float, float> operator()(float x) {
    return thrust::make_pair(x, x);
  }
};

struct MinMax : public thrust::binary_function<thrust::pair<float, float>,
                                               thrust::pair<float, float>,
                                               thrust::pair<float, float>> {
  thrust::pair<float, float> operator()(thrust::pair<float, float> a,
                                        thrust::pair<float, float> b) {
    return thrust::make_pair(glm::min(a.first, b.first),
                             glm::max(a.second, b.second));
  }
};
}  // namespace

namespace manifold {

/**
 * Once halfedge_ has been filled in, this function can be called to create the
 * rest of the internal data structures. This function also removes the verts
 * and halfedges flagged for removal (NaN verts and -1 halfedges).
 */
void Manifold::Impl::Finish() {
  if (halfedge_.size() == 0) return;

  CalculateBBox();
  SetPrecision(precision_);
  if (!bBox_.IsFinite()) {
    // Decimated out of existence - early out.
    MarkFailure(Error::NoError);
    return;
  }

  SortVerts();
  Vec<Box> faceBox;
  Vec<uint32_t> faceMorton;
  GetFaceBoxMorton(faceBox, faceMorton);
  SortFaces(faceBox, faceMorton);
  if (halfedge_.size() == 0) return;
  CompactProps();

  ASSERT(halfedge_.size() % 6 == 0, topologyErr,
         "Not an even number of faces after sorting faces!");

#ifdef MANIFOLD_DEBUG
  Halfedge extrema = {0, 0, 0, 0};
  extrema = reduce<Halfedge>(autoPolicy(halfedge_.size()), halfedge_.begin(),
                             halfedge_.end(), extrema, Extrema());
#endif

  ASSERT(extrema.startVert >= 0, topologyErr, "Vertex index is negative!");
  ASSERT(extrema.endVert < NumVert(), topologyErr,
         "Vertex index exceeds number of verts!");
  ASSERT(extrema.face >= 0, topologyErr, "Face index is negative!");
  ASSERT(extrema.face < NumTri(), topologyErr,
         "Face index exceeds number of faces!");
  ASSERT(extrema.pairedHalfedge >= 0, topologyErr,
         "Halfedge index is negative!");
  ASSERT(extrema.pairedHalfedge < 2 * NumEdge(), topologyErr,
         "Halfedge index exceeds number of halfedges!");
  ASSERT(meshRelation_.triRef.size() == NumTri() ||
             meshRelation_.triRef.size() == 0,
         logicErr, "Mesh Relation doesn't fit!");
  ASSERT(faceNormal_.size() == NumTri() || faceNormal_.size() == 0, logicErr,
         "faceNormal size = " + std::to_string(faceNormal_.size()) +
             ", NumTri = " + std::to_string(NumTri()));
  // TODO: figure out why this has a flaky failure and then enable reading
  // vertNormals from a Mesh.
  // ASSERT(vertNormal_.size() == NumVert() || vertNormal_.size() == 0,
  // logicErr,
  //        "vertNormal size = " + std::to_string(vertNormal_.size()) +
  //            ", NumVert = " + std::to_string(NumVert()));

  CalculateNormals();
  collider_ = Collider(faceBox, faceMorton);

  ASSERT(Is2Manifold(), logicErr, "mesh is not 2-manifold!");
}

/**
 * Sorts the vertices according to their Morton code.
 */
void Manifold::Impl::SortVerts() {
  ZoneScoped;
  const int numVert = NumVert();
  Vec<uint32_t> vertMorton(numVert);
  auto policy = autoPolicy(numVert);
  for_each_n(policy, zip(vertMorton.begin(), vertPos_.cbegin()), numVert,
             Morton({bBox_}));

  Vec<int> vertNew2Old(numVert);
  sequence(policy, vertNew2Old.begin(), vertNew2Old.end());

  stable_sort(policy, zip(vertMorton.begin(), vertNew2Old.begin()),
              zip(vertMorton.end(), vertNew2Old.end()),
              [](const thrust::tuple<uint32_t, int>& a,
                 const thrust::tuple<uint32_t, int>& b) {
                return thrust::get<0>(a) < thrust::get<0>(b);
              });

  ReindexVerts(vertNew2Old, numVert);

  // Verts were flagged for removal with NaNs and assigned kNoCode to sort
  // them to the end, which allows them to be removed.
  const int newNumVert =
      std::lower_bound(vertMorton.begin(), vertMorton.end(), kNoCode) -
      vertMorton.begin();

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
void Manifold::Impl::ReindexVerts(const Vec<int>& vertNew2Old, int oldNumVert) {
  ZoneScoped;
  Vec<int> vertOld2New(oldNumVert);
  scatter(autoPolicy(oldNumVert), countAt(0), countAt(NumVert()),
          vertNew2Old.begin(), vertOld2New.begin());
  for_each(autoPolicy(oldNumVert), halfedge_.begin(), halfedge_.end(),
           Reindex({vertOld2New}));
}

/**
 * Removes unreferenced property verts and reindexes triProperties.
 */
void Manifold::Impl::CompactProps() {
  ZoneScoped;
  if (meshRelation_.numProp == 0) return;

  const int numVerts = meshRelation_.properties.size() / meshRelation_.numProp;
  Vec<int> keep(numVerts, 0);
  auto policy = autoPolicy(numVerts);

  for_each(policy, meshRelation_.triProperties.cbegin(),
           meshRelation_.triProperties.cend(), MarkProp({keep}));
  Vec<int> propOld2New(numVerts + 1, 0);
  inclusive_scan(policy, keep.begin(), keep.end(), propOld2New.begin() + 1);

  Vec<float> oldProp = meshRelation_.properties;
  const int numVertsNew = propOld2New[numVerts];
  meshRelation_.properties.resize(meshRelation_.numProp * numVertsNew);
  for_each_n(
      policy, zip(countAt(0), propOld2New.cbegin(), keep.cbegin()), numVerts,
      GatherProps({meshRelation_.properties, oldProp, meshRelation_.numProp}));
  for_each_n(policy, meshRelation_.triProperties.begin(), NumTri(),
             ReindexProps({propOld2New}));
}

/**
 * Fills the faceBox and faceMorton input with the bounding boxes and Morton
 * codes of the faces, respectively. The Morton code is based on the center of
 * the bounding box.
 */
void Manifold::Impl::GetFaceBoxMorton(Vec<Box>& faceBox,
                                      Vec<uint32_t>& faceMorton) const {
  ZoneScoped;
  faceBox.resize(NumTri());
  faceMorton.resize(NumTri());
  for_each_n(autoPolicy(NumTri()),
             zip(faceMorton.begin(), faceBox.begin(), countAt(0)), NumTri(),
             FaceMortonBox({halfedge_, vertPos_, bBox_}));
}

/**
 * Sorts the faces of this manifold according to their input Morton code. The
 * bounding box and Morton code arrays are also sorted accordingly.
 */
void Manifold::Impl::SortFaces(Vec<Box>& faceBox, Vec<uint32_t>& faceMorton) {
  ZoneScoped;
  Vec<int> faceNew2Old(NumTri());
  auto policy = autoPolicy(faceNew2Old.size());
  sequence(policy, faceNew2Old.begin(), faceNew2Old.end());

  stable_sort(policy, zip(faceMorton.begin(), faceNew2Old.begin()),
              zip(faceMorton.end(), faceNew2Old.end()),
              [](const thrust::tuple<uint32_t, int>& a,
                 const thrust::tuple<uint32_t, int>& b) {
                return thrust::get<0>(a) < thrust::get<0>(b);
              });

  // Tris were flagged for removal with pairedHalfedge = -1 and assigned kNoCode
  // to sort them to the end, which allows them to be removed.
  const int newNumTri =
      find<decltype(faceMorton.begin())>(policy, faceMorton.begin(),
                                         faceMorton.end(), kNoCode) -
      faceMorton.begin();
  faceMorton.resize(newNumTri);
  faceNew2Old.resize(newNumTri);

  Permute(faceBox, faceNew2Old);
  GatherFaces(faceNew2Old);
}

/**
 * Creates the halfedge_ vector for this manifold by copying a set of faces from
 * another manifold, given by oldHalfedge. Input faceNew2Old defines the old
 * faces to gather into this.
 */
void Manifold::Impl::GatherFaces(const Vec<int>& faceNew2Old) {
  ZoneScoped;
  const int numTri = faceNew2Old.size();
  if (meshRelation_.triRef.size() == NumTri())
    Permute(meshRelation_.triRef, faceNew2Old);
  if (meshRelation_.triProperties.size() == NumTri())
    Permute(meshRelation_.triProperties, faceNew2Old);
  if (faceNormal_.size() == NumTri()) Permute(faceNormal_, faceNew2Old);

  Vec<Halfedge> oldHalfedge(std::move(halfedge_));
  Vec<glm::vec4> oldHalfedgeTangent(std::move(halfedgeTangent_));
  Vec<int> faceOld2New(oldHalfedge.size() / 3);
  auto policy = autoPolicy(numTri);
  scatter(policy, countAt(0), countAt(numTri), faceNew2Old.begin(),
          faceOld2New.begin());

  halfedge_.resize(3 * numTri);
  if (oldHalfedgeTangent.size() != 0) halfedgeTangent_.resize(3 * numTri);
  for_each_n(policy, countAt(0), numTri,
             ReindexFace({halfedge_, halfedgeTangent_, oldHalfedge,
                          oldHalfedgeTangent, faceNew2Old, faceOld2New}));
}

void Manifold::Impl::GatherFaces(const Impl& old, const Vec<int>& faceNew2Old) {
  ZoneScoped;
  const int numTri = faceNew2Old.size();
  auto policy = autoPolicy(numTri);

  meshRelation_.triRef.resize(numTri);
  gather(policy, faceNew2Old.begin(), faceNew2Old.end(),
         old.meshRelation_.triRef.begin(), meshRelation_.triRef.begin());

  for (const auto& pair : old.meshRelation_.meshIDtransform) {
    meshRelation_.meshIDtransform[pair.first] = pair.second;
  }

  if (old.meshRelation_.triProperties.size() > 0) {
    meshRelation_.triProperties.resize(numTri);
    gather(policy, faceNew2Old.begin(), faceNew2Old.end(),
           old.meshRelation_.triProperties.begin(),
           meshRelation_.triProperties.begin());
    meshRelation_.numProp = old.meshRelation_.numProp;
    meshRelation_.properties = old.meshRelation_.properties;
  }

  if (old.faceNormal_.size() == old.NumTri()) {
    faceNormal_.resize(numTri);
    gather(policy, faceNew2Old.begin(), faceNew2Old.end(),
           old.faceNormal_.begin(), faceNormal_.begin());
  }

  Vec<int> faceOld2New(old.NumTri());
  scatter(policy, countAt(0), countAt(numTri), faceNew2Old.begin(),
          faceOld2New.begin());

  halfedge_.resize(3 * numTri);
  if (old.halfedgeTangent_.size() != 0) halfedgeTangent_.resize(3 * numTri);
  for_each_n(policy, countAt(0), numTri,
             ReindexFace({halfedge_, halfedgeTangent_, old.halfedge_,
                          old.halfedgeTangent_, faceNew2Old, faceOld2New}));
}

/// Constructs a position-only MeshGL from the input Mesh.
MeshGL::MeshGL(const Mesh& mesh) {
  numProp = 3;
  precision = mesh.precision;
  vertProperties.resize(numProp * mesh.vertPos.size());
  for (int i = 0; i < mesh.vertPos.size(); ++i) {
    for (int j : {0, 1, 2}) vertProperties[3 * i + j] = mesh.vertPos[i][j];
  }
  triVerts.resize(3 * mesh.triVerts.size());
  for (int i = 0; i < mesh.triVerts.size(); ++i) {
    for (int j : {0, 1, 2}) triVerts[3 * i + j] = mesh.triVerts[i][j];
  }
  halfedgeTangent.resize(4 * mesh.halfedgeTangent.size());
  for (int i = 0; i < mesh.halfedgeTangent.size(); ++i) {
    for (int j : {0, 1, 2, 3})
      halfedgeTangent[4 * i + j] = mesh.halfedgeTangent[i][j];
  }
}

/**
 * Updates the mergeFromVert and mergeToVert vectors in order to create a
 * manifold solid. If the MeshGL is already manifold, no change will occur and
 * the function will return false. Otherwise, this will merge verts along open
 * edges within precision (the maximum of the MeshGL precision and the baseline
 * bounding-box precision), keeping any from the existing merge vectors.
 *
 * There is no guarantee the result will be manifold - this is a best-effort
 * helper function designed primarily to aid in the case where a manifold
 * multi-material MeshGL was produced, but its merge vectors were lost due to a
 * round-trip through a file format. Constructing a Manifold from the result
 * will report a Status if it is not manifold.
 */
bool MeshGL::Merge() {
  ZoneScoped;
  std::multiset<std::pair<int, int>> openEdges;

  std::vector<int> merge(NumVert());
  std::iota(merge.begin(), merge.end(), 0);
  for (int i = 0; i < mergeFromVert.size(); ++i) {
    merge[mergeFromVert[i]] = mergeToVert[i];
  }

  const int numVert = NumVert();
  const int numTri = NumTri();
  const int next[3] = {1, 2, 0};
  for (int tri = 0; tri < numTri; ++tri) {
    for (int i : {0, 1, 2}) {
      auto edge = std::make_pair(merge[triVerts[3 * tri + next[i]]],
                                 merge[triVerts[3 * tri + i]]);
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

  const int numOpenVert = openEdges.size();
  Vec<int> openVerts(numOpenVert);
  int i = 0;
  for (const auto& edge : openEdges) {
    const int vert = edge.first;
    openVerts[i++] = vert;
  }

  Vec<float> vertPropD(vertProperties);
  Box bBox;
  for (const int i : {0, 1, 2}) {
    strided_range<Vec<float>::Iter> iPos(vertPropD.begin() + i, vertPropD.end(),
                                         numProp);
    auto minMax = transform_reduce<thrust::pair<float, float>>(
        autoPolicy(numVert), iPos.begin(), iPos.end(), Duplicate(),
        thrust::make_pair(std::numeric_limits<float>::infinity(),
                          -std::numeric_limits<float>::infinity()),
        MinMax());
    bBox.min[i] = minMax.first;
    bBox.max[i] = minMax.second;
  }
  precision = MaxPrecision(precision, bBox);
  if (precision < 0) return false;

  auto policy = autoPolicy(numOpenVert);
  Vec<Box> vertBox(numOpenVert);
  Vec<uint32_t> vertMorton(numOpenVert);

  for_each_n(policy,
             zip(vertMorton.begin(), vertBox.begin(), openVerts.cbegin()),
             numOpenVert, VertMortonBox({vertPropD, numProp, precision, bBox}));

  stable_sort(policy,
              zip(vertMorton.begin(), vertBox.begin(), openVerts.begin()),
              zip(vertMorton.end(), vertBox.end(), openVerts.end()),
              [](const thrust::tuple<uint32_t, Box, int>& a,
                 const thrust::tuple<uint32_t, Box, int>& b) {
                return thrust::get<0>(a) < thrust::get<0>(b);
              });

  Collider collider(vertBox, vertMorton);
  SparseIndices toMerge = collider.Collisions<true>(vertBox.cview());

  UnionFind<> uf(numVert);
  for (int i = 0; i < mergeFromVert.size(); ++i) {
    uf.unionXY(static_cast<int>(mergeFromVert[i]),
               static_cast<int>(mergeToVert[i]));
  }
  for (int i = 0; i < toMerge.size(); ++i) {
    uf.unionXY(openVerts[toMerge.Get(i, false)],
               openVerts[toMerge.Get(i, true)]);
  }

  mergeToVert.clear();
  mergeFromVert.clear();
  for (int v = 0; v < numVert; ++v) {
    const int mergeTo = uf.find(v);
    if (mergeTo != v) {
      mergeFromVert.push_back(v);
      mergeToVert.push_back(mergeTo);
    }
  }

  return true;
}
}  // namespace manifold
