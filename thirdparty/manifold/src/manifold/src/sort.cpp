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

#include <../third_party/thrust/thrust/sequence.h>

#include "impl.h"
#include "par.h"

namespace {
using namespace manifold;

constexpr uint32_t kNoCode = 0xFFFFFFFFu;

struct Extrema : public thrust::binary_function<Halfedge, Halfedge, Halfedge> {
  __host__ __device__ void MakeForward(Halfedge& a) {
    if (!a.IsForward()) {
      int tmp = a.startVert;
      a.startVert = a.endVert;
      a.endVert = tmp;
    }
  }

  __host__ __device__ int MaxOrMinus(int a, int b) {
    return glm::min(a, b) < 0 ? -1 : glm::max(a, b);
  }

  __host__ __device__ Halfedge operator()(Halfedge a, Halfedge b) {
    MakeForward(a);
    MakeForward(b);
    a.startVert = glm::min(a.startVert, b.startVert);
    a.endVert = glm::max(a.endVert, b.endVert);
    a.face = MaxOrMinus(a.face, b.face);
    a.pairedHalfedge = MaxOrMinus(a.pairedHalfedge, b.pairedHalfedge);
    return a;
  }
};

__host__ __device__ uint32_t SpreadBits3(uint32_t v) {
  v = 0xFF0000FFu & (v * 0x00010001u);
  v = 0x0F00F00Fu & (v * 0x00000101u);
  v = 0xC30C30C3u & (v * 0x00000011u);
  v = 0x49249249u & (v * 0x00000005u);
  return v;
}

__host__ __device__ uint32_t MortonCode(glm::vec3 position, Box bBox) {
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

  __host__ __device__ void operator()(
      thrust::tuple<uint32_t&, const glm::vec3&> inout) {
    glm::vec3 position = thrust::get<1>(inout);
    thrust::get<0>(inout) = MortonCode(position, bBox);
  }
};

struct FaceMortonBox {
  const Halfedge* halfedge;
  const glm::vec3* vertPos;
  const Box bBox;

  __host__ __device__ void operator()(
      thrust::tuple<uint32_t&, Box&, int> inout) {
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
  const int* indexInv;

  __host__ __device__ void operator()(Halfedge& edge) {
    if (edge.startVert < 0) return;
    edge.startVert = indexInv[edge.startVert];
    edge.endVert = indexInv[edge.endVert];
  }
};

template <typename T>
void Permute(VecDH<T>& inOut, const VecDH<int>& new2Old) {
  VecDH<T> tmp(std::move(inOut));
  inOut.resize(new2Old.size());
  gather(autoPolicy(new2Old.size()), new2Old.begin(), new2Old.end(),
         tmp.begin(), inOut.begin());
}

template void Permute<BaryRef>(VecDH<BaryRef>&, const VecDH<int>&);
template void Permute<glm::vec3>(VecDH<glm::vec3>&, const VecDH<int>&);

struct ReindexFace {
  Halfedge* halfedge;
  glm::vec4* halfedgeTangent;
  const Halfedge* oldHalfedge;
  const glm::vec4* oldHalfedgeTangent;
  const int* faceNew2Old;
  const int* faceOld2New;

  __host__ __device__ void operator()(int newFace) {
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
      if (oldHalfedgeTangent != nullptr) {
        halfedgeTangent[newEdge] = oldHalfedgeTangent[oldEdge];
      }
    }
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
    // Decimated out of existance - early out.
    MarkFailure(Error::NO_ERROR);
    return;
  }

  SortVerts();
  VecDH<Box> faceBox;
  VecDH<uint32_t> faceMorton;
  GetFaceBoxMorton(faceBox, faceMorton);
  SortFaces(faceBox, faceMorton);
  if (halfedge_.size() == 0) return;

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
  ASSERT(meshRelation_.triBary.size() == NumTri() ||
             meshRelation_.triBary.size() == 0,
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
}

/**
 * Sorts the vertices according to their Morton code.
 */
void Manifold::Impl::SortVerts() {
  const int numVert = NumVert();
  VecDH<uint32_t> vertMorton(numVert);
  auto policy = autoPolicy(numVert);
  for_each_n(policy, zip(vertMorton.begin(), vertPos_.cbegin()), numVert,
             Morton({bBox_}));

  VecDH<int> vertNew2Old(numVert);
  sequence(policy, vertNew2Old.begin(), vertNew2Old.end());
  sort_by_key(policy, vertMorton.begin(), vertMorton.end(),
              zip(vertPos_.begin(), vertNew2Old.begin()));

  ReindexVerts(vertNew2Old, numVert);

  // Verts were flagged for removal with NaNs and assigned kNoCode to sort
  // them to the end, which allows them to be removed.
  const int newNumVert =
      find<decltype(vertMorton.begin())>(policy, vertMorton.begin(),
                                         vertMorton.end(), kNoCode) -
      vertMorton.begin();
  vertPos_.resize(newNumVert);
  if (vertNormal_.size() == numVert) {
    Permute(vertNormal_, vertNew2Old);
    vertNormal_.resize(newNumVert);
  }
}

/**
 * Updates the halfedges to point to new vert indices based on a mapping,
 * vertNew2Old. This may be a subset, so the total number of original verts is
 * also given.
 */
void Manifold::Impl::ReindexVerts(const VecDH<int>& vertNew2Old,
                                  int oldNumVert) {
  VecDH<int> vertOld2New(oldNumVert);
  scatter(autoPolicy(oldNumVert), countAt(0), countAt(NumVert()),
          vertNew2Old.begin(), vertOld2New.begin());
  for_each(autoPolicy(oldNumVert), halfedge_.begin(), halfedge_.end(),
           Reindex({vertOld2New.cptrD()}));
}

/**
 * Fills the faceBox and faceMorton input with the bounding boxes and Morton
 * codes of the faces, respectively. The Morton code is based on the center of
 * the bounding box.
 */
void Manifold::Impl::GetFaceBoxMorton(VecDH<Box>& faceBox,
                                      VecDH<uint32_t>& faceMorton) const {
  faceBox.resize(NumTri());
  faceMorton.resize(NumTri());
  for_each_n(autoPolicy(NumTri()),
             zip(faceMorton.begin(), faceBox.begin(), countAt(0)), NumTri(),
             FaceMortonBox({halfedge_.cptrD(), vertPos_.cptrD(), bBox_}));
}

/**
 * Sorts the faces of this manifold according to their input Morton code. The
 * bounding box and Morton code arrays are also sorted accordingly.
 */
void Manifold::Impl::SortFaces(VecDH<Box>& faceBox,
                               VecDH<uint32_t>& faceMorton) {
  VecDH<int> faceNew2Old(NumTri());
  auto policy = autoPolicy(faceNew2Old.size());
  sequence(policy, faceNew2Old.begin(), faceNew2Old.end());

  sort_by_key(policy, faceMorton.begin(), faceMorton.end(),
              zip(faceBox.begin(), faceNew2Old.begin()));

  // Tris were flagged for removal with pairedHalfedge = -1 and assigned kNoCode
  // to sort them to the end, which allows them to be removed.
  const int newNumTri =
      find<decltype(faceMorton.begin())>(policy, faceMorton.begin(),
                                         faceMorton.end(), kNoCode) -
      faceMorton.begin();
  faceBox.resize(newNumTri);
  faceMorton.resize(newNumTri);
  faceNew2Old.resize(newNumTri);

  GatherFaces(faceNew2Old);
}

/**
 * Creates the halfedge_ vector for this manifold by copying a set of faces from
 * another manifold, given by oldHalfedge. Input faceNew2Old defines the old
 * faces to gather into this.
 */
void Manifold::Impl::GatherFaces(const VecDH<int>& faceNew2Old) {
  const int numTri = faceNew2Old.size();
  if (meshRelation_.triBary.size() == NumTri())
    Permute(meshRelation_.triBary, faceNew2Old);

  if (faceNormal_.size() == NumTri()) Permute(faceNormal_, faceNew2Old);

  VecDH<Halfedge> oldHalfedge(std::move(halfedge_));
  VecDH<glm::vec4> oldHalfedgeTangent(std::move(halfedgeTangent_));
  VecDH<int> faceOld2New(oldHalfedge.size() / 3);
  auto policy = autoPolicy(numTri);
  scatter(policy, countAt(0), countAt(numTri), faceNew2Old.begin(),
          faceOld2New.begin());

  halfedge_.resize(3 * numTri);
  if (oldHalfedgeTangent.size() != 0) halfedgeTangent_.resize(3 * numTri);
  for_each_n(policy, countAt(0), numTri,
             ReindexFace({halfedge_.ptrD(), halfedgeTangent_.ptrD(),
                          oldHalfedge.cptrD(), oldHalfedgeTangent.cptrD(),
                          faceNew2Old.cptrD(), faceOld2New.cptrD()}));
}

void Manifold::Impl::GatherFaces(const Impl& old,
                                 const VecDH<int>& faceNew2Old) {
  const int numTri = faceNew2Old.size();
  meshRelation_.triBary.resize(numTri);
  auto policy = autoPolicy(numTri);
  gather(policy, faceNew2Old.begin(), faceNew2Old.end(),
         old.meshRelation_.triBary.begin(), meshRelation_.triBary.begin());
  meshRelation_.barycentric = old.meshRelation_.barycentric;

  if (old.faceNormal_.size() == old.NumTri()) {
    faceNormal_.resize(numTri);
    gather(policy, faceNew2Old.begin(), faceNew2Old.end(),
           old.faceNormal_.begin(), faceNormal_.begin());
  }

  VecDH<int> faceOld2New(old.NumTri());
  scatter(policy, countAt(0), countAt(numTri), faceNew2Old.begin(),
          faceOld2New.begin());

  halfedge_.resize(3 * numTri);
  if (old.halfedgeTangent_.size() != 0) halfedgeTangent_.resize(3 * numTri);
  for_each_n(policy, countAt(0), numTri,
             ReindexFace({halfedge_.ptrD(), halfedgeTangent_.ptrD(),
                          old.halfedge_.cptrD(), old.halfedgeTangent_.cptrD(),
                          faceNew2Old.cptrD(), faceOld2New.cptrD()}));
}
}  // namespace manifold
