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

#include "boolean3.h"
#include "par.h"
#include "polygon.h"

using namespace manifold;
using namespace thrust::placeholders;

namespace {

struct AbsSum : public thrust::binary_function<int, int, int> {
  __host__ __device__ int operator()(int a, int b) { return abs(a) + abs(b); }
};

struct DuplicateVerts {
  glm::vec3 *vertPosR;

  __host__ __device__ void operator()(thrust::tuple<int, int, glm::vec3> in) {
    int inclusion = abs(thrust::get<0>(in));
    int vertR = thrust::get<1>(in);
    glm::vec3 vertPosP = thrust::get<2>(in);

    for (int i = 0; i < inclusion; ++i) {
      vertPosR[vertR + i] = vertPosP;
    }
  }
};

struct CountVerts {
  int *count;
  const int *inclusion;

  __host__ __device__ void operator()(const Halfedge &edge) {
    AtomicAdd(count[edge.face], glm::abs(inclusion[edge.startVert]));
  }
};

struct CountNewVerts {
  int *countP;
  int *countQ;
  const Halfedge *halfedges;

  __host__ __device__ void operator()(thrust::tuple<int, int, int> in) {
    int edgeP = thrust::get<0>(in);
    int faceQ = thrust::get<1>(in);
    int inclusion = glm::abs(thrust::get<2>(in));

    AtomicAdd(countQ[faceQ], inclusion);
    const Halfedge half = halfedges[edgeP];
    AtomicAdd(countP[half.face], inclusion);
    AtomicAdd(countP[halfedges[half.pairedHalfedge].face], inclusion);
  }
};

struct NotZero : public thrust::unary_function<int, int> {
  __host__ __device__ int operator()(int x) const { return x > 0 ? 1 : 0; }
};

std::tuple<VecDH<int>, VecDH<int>> SizeOutput(
    Manifold::Impl &outR, const Manifold::Impl &inP, const Manifold::Impl &inQ,
    const VecDH<int> &i03, const VecDH<int> &i30, const VecDH<int> &i12,
    const VecDH<int> &i21, const SparseIndices &p1q2, const SparseIndices &p2q1,
    bool invertQ) {
  VecDH<int> sidesPerFacePQ(inP.NumTri() + inQ.NumTri(), 0);
  auto sidesPerFaceP = sidesPerFacePQ.ptrD();
  auto sidesPerFaceQ = sidesPerFacePQ.ptrD() + inP.NumTri();

  auto policy =
      autoPolicy(std::max(inP.halfedge_.size(), inQ.halfedge_.size()));

  for_each(policy, inP.halfedge_.begin(), inP.halfedge_.end(),
           CountVerts({sidesPerFaceP, i03.cptrD()}));
  for_each(policy, inQ.halfedge_.begin(), inQ.halfedge_.end(),
           CountVerts({sidesPerFaceQ, i30.cptrD()}));
  for_each_n(
      policy, zip(p1q2.begin(0), p1q2.begin(1), i12.begin()), i12.size(),
      CountNewVerts({sidesPerFaceP, sidesPerFaceQ, inP.halfedge_.cptrD()}));
  for_each_n(
      policy, zip(p2q1.begin(1), p2q1.begin(0), i21.begin()), i21.size(),
      CountNewVerts({sidesPerFaceQ, sidesPerFaceP, inQ.halfedge_.cptrD()}));

  VecDH<int> facePQ2R(inP.NumTri() + inQ.NumTri() + 1, 0);
  auto keepFace =
      thrust::make_transform_iterator(sidesPerFacePQ.begin(), NotZero());
  inclusive_scan(policy, keepFace, keepFace + sidesPerFacePQ.size(),
                 facePQ2R.begin() + 1);
  int numFaceR = facePQ2R.back();
  facePQ2R.resize(inP.NumTri() + inQ.NumTri());

  outR.faceNormal_.resize(numFaceR);
  auto next = copy_if<decltype(outR.faceNormal_.begin())>(
      policy, inP.faceNormal_.begin(), inP.faceNormal_.end(), keepFace,
      outR.faceNormal_.begin(), thrust::identity<bool>());
  if (invertQ) {
    auto start = thrust::make_transform_iterator(inQ.faceNormal_.begin(),
                                                 thrust::negate<glm::vec3>());
    auto end = thrust::make_transform_iterator(inQ.faceNormal_.end(),
                                               thrust::negate<glm::vec3>());
    copy_if<decltype(inQ.faceNormal_.begin())>(policy, start, end,
                                               keepFace + inP.NumTri(), next,
                                               thrust::identity<bool>());
  } else {
    copy_if<decltype(inQ.faceNormal_.begin())>(
        policy, inQ.faceNormal_.begin(), inQ.faceNormal_.end(),
        keepFace + inP.NumTri(), next, thrust::identity<bool>());
  }

  auto newEnd = remove<decltype(sidesPerFacePQ.begin())>(
      policy, sidesPerFacePQ.begin(), sidesPerFacePQ.end(), 0);
  VecDH<int> faceEdge(newEnd - sidesPerFacePQ.begin() + 1, 0);
  inclusive_scan(policy, sidesPerFacePQ.begin(), newEnd, faceEdge.begin() + 1);
  outR.halfedge_.resize(faceEdge.back());

  return std::make_tuple(faceEdge, facePQ2R);
}

struct EdgePos {
  int vert;
  float edgePos;
  bool isStart;
};

void AddNewEdgeVerts(
    std::map<int, std::vector<EdgePos>> &edgesP,
    std::map<std::pair<int, int>, std::vector<EdgePos>> &edgesNew,
    const SparseIndices &p1q2, const VecDH<int> &i12, const VecDH<int> &v12R,
    const VecDH<Halfedge> &halfedgeP, bool forward) {
  // For each edge of P that intersects a face of Q (p1q2), add this vertex to
  // P's corresponding edge vector and to the two new edges, which are
  // intersections between the face of Q and the two faces of P attached to the
  // edge. The direction and duplicity are given by i12, while v12R remaps to
  // the output vert index. When forward is false, all is reversed.
  const VecDH<int> &p1 = p1q2.Get(!forward);
  const VecDH<int> &q2 = p1q2.Get(forward);
  for (int i = 0; i < p1q2.size(); ++i) {
    const int edgeP = p1[i];
    const int faceQ = q2[i];
    const int vert = v12R[i];
    const int inclusion = i12[i];

    auto &edgePosP = edgesP[edgeP];

    Halfedge halfedge = halfedgeP[edgeP];
    std::pair<int, int> key = {halfedgeP[halfedge.pairedHalfedge].face, faceQ};
    if (!forward) std::swap(key.first, key.second);
    auto &edgePosRight = edgesNew[key];

    key = {halfedge.face, faceQ};
    if (!forward) std::swap(key.first, key.second);
    auto &edgePosLeft = edgesNew[key];

    EdgePos edgePos = {vert, 0.0f, inclusion < 0};
    EdgePos edgePosRev = edgePos;
    edgePosRev.isStart = !edgePos.isStart;

    for (int j = 0; j < glm::abs(inclusion); ++j) {
      edgePosP.push_back(edgePos);
      edgePosRight.push_back(forward ? edgePos : edgePosRev);
      edgePosLeft.push_back(forward ? edgePosRev : edgePos);
      ++edgePos.vert;
      ++edgePosRev.vert;
    }
  }
}

std::vector<Halfedge> PairUp(std::vector<EdgePos> &edgePos) {
  // Pair start vertices with end vertices to form edges. The choice of pairing
  // is arbitrary for the manifoldness guarantee, but must be ordered to be
  // geometrically valid. If the order does not go start-end-start-end... then
  // the input and output are not geometrically valid and this algorithm becomes
  // a heuristic.
  ASSERT(edgePos.size() % 2 == 0, topologyErr,
         "Non-manifold edge! Not an even number of points.");
  int nEdges = edgePos.size() / 2;
  auto middle = std::partition(edgePos.begin(), edgePos.end(),
                               [](EdgePos x) { return x.isStart; });
  ASSERT(middle - edgePos.begin() == nEdges, topologyErr, "Non-manifold edge!");
  auto cmp = [](EdgePos a, EdgePos b) { return a.edgePos < b.edgePos; };
  std::sort(edgePos.begin(), middle, cmp);
  std::sort(middle, edgePos.end(), cmp);
  std::vector<Halfedge> edges;
  for (int i = 0; i < nEdges; ++i)
    edges.push_back({edgePos[i].vert, edgePos[i + nEdges].vert, -1, -1});
  return edges;
}

// A Ref carries the reference of a halfedge's startVert back to the input
// manifolds. PQ is 0 if the halfedge comes from triangle tri of P, and 1 for Q.
// vert is 0, 1, or 2 to denote which vertex of tri it is, and -1 if it is new.
struct Ref {
  int PQ, tri, vert;
};

void AppendPartialEdges(Manifold::Impl &outR, VecDH<char> &wholeHalfedgeP,
                        VecDH<int> &facePtrR,
                        std::map<int, std::vector<EdgePos>> &edgesP,
                        VecDH<Ref> &halfedgeRef, const Manifold::Impl &inP,
                        const VecDH<int> &i03, const VecDH<int> &vP2R,
                        const VecDH<int>::IterC faceP2R, bool forward) {
  // Each edge in the map is partially retained; for each of these, look up
  // their original verts and include them based on their winding number (i03),
  // while remaping them to the output using vP2R. Use the verts position
  // projected along the edge vector to pair them up, then distribute these
  // edges to their faces.
  VecDH<Halfedge> &halfedgeR = outR.halfedge_;
  const VecDH<glm::vec3> &vertPosP = inP.vertPos_;
  const VecDH<Halfedge> &halfedgeP = inP.halfedge_;

  for (auto &value : edgesP) {
    const int edgeP = value.first;
    std::vector<EdgePos> &edgePosP = value.second;

    const Halfedge &halfedge = halfedgeP[edgeP];
    wholeHalfedgeP[edgeP] = false;
    wholeHalfedgeP[halfedge.pairedHalfedge] = false;

    const int vStart = halfedge.startVert;
    const int vEnd = halfedge.endVert;
    const glm::vec3 edgeVec = vertPosP[vEnd] - vertPosP[vStart];
    // Fill in the edge positions of the old points.
    for (EdgePos &edge : edgePosP) {
      edge.edgePos = glm::dot(outR.vertPos_[edge.vert], edgeVec);
    }

    int inclusion = i03[vStart];
    bool reversed = inclusion < 0;
    EdgePos edgePos = {vP2R[vStart],
                       glm::dot(outR.vertPos_[vP2R[vStart]], edgeVec),
                       inclusion > 0};
    for (int j = 0; j < glm::abs(inclusion); ++j) {
      edgePosP.push_back(edgePos);
      ++edgePos.vert;
    }

    inclusion = i03[vEnd];
    reversed |= inclusion < 0;
    edgePos = {vP2R[vEnd], glm::dot(outR.vertPos_[vP2R[vEnd]], edgeVec),
               inclusion < 0};
    for (int j = 0; j < glm::abs(inclusion); ++j) {
      edgePosP.push_back(edgePos);
      ++edgePos.vert;
    }

    // sort edges into start/end pairs along length
    std::vector<Halfedge> edges = PairUp(edgePosP);

    // add halfedges to result
    const int faceLeftP = halfedge.face;
    const int faceLeft = faceP2R[faceLeftP];
    const int faceRightP = halfedgeP[halfedge.pairedHalfedge].face;
    const int faceRight = faceP2R[faceRightP];
    // Negative inclusion means the halfedges are reversed, which means our
    // reference is now to the endVert instead of the startVert, which is one
    // position advanced CCW. This is only valid if this is a retained vert; it
    // will be ignored later if the vert is new.
    const Ref forwardRef = {forward ? 0 : 1, faceLeftP,
                            (edgeP + (reversed ? 1 : 0)) % 3};
    const Ref backwardRef = {
        forward ? 0 : 1, faceRightP,
        (halfedge.pairedHalfedge + (reversed ? 1 : 0)) % 3};

    for (Halfedge e : edges) {
      const int forwardEdge = facePtrR[faceLeft]++;
      const int backwardEdge = facePtrR[faceRight]++;

      e.face = faceLeft;
      e.pairedHalfedge = backwardEdge;
      halfedgeR[forwardEdge] = e;
      halfedgeRef[forwardEdge] = forwardRef;

      std::swap(e.startVert, e.endVert);
      e.face = faceRight;
      e.pairedHalfedge = forwardEdge;
      halfedgeR[backwardEdge] = e;
      halfedgeRef[backwardEdge] = backwardRef;
    }
  }
}

void AppendNewEdges(
    Manifold::Impl &outR, VecDH<int> &facePtrR,
    std::map<std::pair<int, int>, std::vector<EdgePos>> &edgesNew,
    VecDH<Ref> &halfedgeRef, const VecDH<int> &facePQ2R, const int numFaceP) {
  // Pair up each edge's verts and distribute to faces based on indices in key.
  VecDH<Halfedge> &halfedgeR = outR.halfedge_;
  VecDH<glm::vec3> &vertPosR = outR.vertPos_;

  for (auto &value : edgesNew) {
    const int faceP = value.first.first;
    const int faceQ = value.first.second;
    std::vector<EdgePos> &edgePos = value.second;

    Box bbox;
    for (auto edge : edgePos) {
      bbox.Union(vertPosR[edge.vert]);
    }
    const glm::vec3 size = bbox.Size();
    // Order the points along their longest dimension.
    const int i = (size.x > size.y && size.x > size.z) ? 0
                  : size.y > size.z                    ? 1
                                                       : 2;
    for (auto &edge : edgePos) {
      edge.edgePos = vertPosR[edge.vert][i];
    }

    // sort edges into start/end pairs along length.
    std::vector<Halfedge> edges = PairUp(edgePos);

    // add halfedges to result
    const int faceLeft = facePQ2R[faceP];
    const int faceRight = facePQ2R[numFaceP + faceQ];
    const Ref forwardRef = {0, faceP, -4};
    const Ref backwardRef = {1, faceQ, -4};
    for (Halfedge e : edges) {
      const int forwardEdge = facePtrR[faceLeft]++;
      const int backwardEdge = facePtrR[faceRight]++;

      e.face = faceLeft;
      e.pairedHalfedge = backwardEdge;
      halfedgeR[forwardEdge] = e;
      halfedgeRef[forwardEdge] = forwardRef;

      std::swap(e.startVert, e.endVert);
      e.face = faceRight;
      e.pairedHalfedge = forwardEdge;
      halfedgeR[backwardEdge] = e;
      halfedgeRef[backwardEdge] = backwardRef;
    }
  }
}

struct DuplicateHalfedges {
  Halfedge *halfedgesR;
  Ref *halfedgeRef;
  int *facePtr;
  const Halfedge *halfedgesP;
  const int *i03;
  const int *vP2R;
  const int *faceP2R;
  const bool forward;

  __host__ __device__ void operator()(thrust::tuple<bool, Halfedge, int> in) {
    if (!thrust::get<0>(in)) return;
    Halfedge halfedge = thrust::get<1>(in);
    if (!halfedge.IsForward()) return;
    const int edgeP = thrust::get<2>(in);

    const int inclusion = i03[halfedge.startVert];
    if (inclusion == 0) return;
    if (inclusion < 0) {  // reverse
      int tmp = halfedge.startVert;
      halfedge.startVert = halfedge.endVert;
      halfedge.endVert = tmp;
    }
    halfedge.startVert = vP2R[halfedge.startVert];
    halfedge.endVert = vP2R[halfedge.endVert];
    const int faceLeftP = halfedge.face;
    halfedge.face = faceP2R[faceLeftP];
    const int faceRightP = halfedgesP[halfedge.pairedHalfedge].face;
    const int faceRight = faceP2R[faceRightP];
    // Negative inclusion means the halfedges are reversed, which means our
    // reference is now to the endVert instead of the startVert, which is one
    // position advanced CCW.
    const Ref forwardRef = {forward ? 0 : 1, faceLeftP,
                            (edgeP + (inclusion < 0 ? 1 : 0)) % 3};
    const Ref backwardRef = {
        forward ? 0 : 1, faceRightP,
        (halfedge.pairedHalfedge + (inclusion < 0 ? 1 : 0)) % 3};

    for (int i = 0; i < glm::abs(inclusion); ++i) {
      int forwardEdge = AtomicAdd(facePtr[halfedge.face], 1);
      int backwardEdge = AtomicAdd(facePtr[faceRight], 1);
      halfedge.pairedHalfedge = backwardEdge;

      halfedgesR[forwardEdge] = halfedge;
      halfedgesR[backwardEdge] = {halfedge.endVert, halfedge.startVert,
                                  forwardEdge, faceRight};
      halfedgeRef[forwardEdge] = forwardRef;
      halfedgeRef[backwardEdge] = backwardRef;

      ++halfedge.startVert;
      ++halfedge.endVert;
    }
  }
};

void AppendWholeEdges(Manifold::Impl &outR, VecDH<int> &facePtrR,
                      VecDH<Ref> &halfedgeRef, const Manifold::Impl &inP,
                      const VecDH<char> wholeHalfedgeP, const VecDH<int> &i03,
                      const VecDH<int> &vP2R, const int *faceP2R,
                      bool forward) {
  for_each_n(autoPolicy(inP.halfedge_.size()),
             zip(wholeHalfedgeP.begin(), inP.halfedge_.begin(), countAt(0)),
             inP.halfedge_.size(),
             DuplicateHalfedges({outR.halfedge_.ptrD(), halfedgeRef.ptrD(),
                                 facePtrR.ptrD(), inP.halfedge_.cptrD(),
                                 i03.cptrD(), vP2R.cptrD(), faceP2R, forward}));
}

struct CreateBarycentric {
  glm::vec3 *barycentricR;
  BaryRef *faceRef;
  int *idx;

  const int offsetQ;
  const int firstNewVert;
  const glm::vec3 *vertPosR;
  const glm::vec3 *vertPosP;
  const glm::vec3 *vertPosQ;
  const Halfedge *halfedgeP;
  const Halfedge *halfedgeQ;
  const BaryRef *triBaryP;
  const BaryRef *triBaryQ;
  const glm::vec3 *barycentricP;
  const glm::vec3 *barycentricQ;
  const bool invertQ;
  const float precision;

  __host__ __device__ void operator()(
      thrust::tuple<int &, Ref, Halfedge> inOut) {
    int &halfedgeBary = thrust::get<0>(inOut);
    const Ref halfedgeRef = thrust::get<1>(inOut);
    const Halfedge halfedgeR = thrust::get<2>(inOut);

    const glm::vec3 *barycentric =
        halfedgeRef.PQ == 0 ? barycentricP : barycentricQ;
    const int tri = halfedgeRef.tri;
    const BaryRef oldRef = halfedgeRef.PQ == 0 ? triBaryP[tri] : triBaryQ[tri];

    faceRef[halfedgeR.face] = oldRef;

    if (halfedgeRef.PQ == 1) {
      // Mark the meshID as coming from Q
      faceRef[halfedgeR.face].meshID += offsetQ;
    }

    if (halfedgeR.startVert < firstNewVert) {  // retained vert
      int i = halfedgeRef.vert;
      const int bary = oldRef.vertBary[i];
      if (bary < 0) {
        halfedgeBary = bary;
      } else {
        halfedgeBary = AtomicAdd(*idx, 1);
        barycentricR[halfedgeBary] = barycentric[bary];
      }
    } else {  // new vert
      halfedgeBary = AtomicAdd(*idx, 1);

      const glm::vec3 *vertPos = halfedgeRef.PQ == 0 ? vertPosP : vertPosQ;
      const Halfedge *halfedge = halfedgeRef.PQ == 0 ? halfedgeP : halfedgeQ;

      glm::mat3 triPos;
      for (int i : {0, 1, 2})
        triPos[i] = vertPos[halfedge[3 * tri + i].startVert];

      glm::mat3 uvwOldTri;
      for (int i : {0, 1, 2})
        uvwOldTri[i] = UVW(oldRef.vertBary[i], barycentric);

      const glm::vec3 uvw =
          GetBarycentric(vertPosR[halfedgeR.startVert], triPos, precision);
      barycentricR[halfedgeBary] = uvwOldTri * uvw;
    }
  }
};

std::pair<VecDH<BaryRef>, VecDH<int>> CalculateMeshRelation(
    Manifold::Impl &outR, const VecDH<Ref> &halfedgeRef,
    const Manifold::Impl &inP, const Manifold::Impl &inQ, int firstNewVert,
    int numFaceR, bool invertQ) {
  outR.meshRelation_.barycentric.resize(outR.halfedge_.size());
  VecDH<BaryRef> faceRef(numFaceR);
  VecDH<int> halfedgeBary(halfedgeRef.size());

  const int offsetQ = Manifold::Impl::meshIDCounter_;
  VecDH<int> idx(1, 0);
  for_each_n(
      autoPolicy(halfedgeRef.size()),
      zip(halfedgeBary.begin(), halfedgeRef.begin(), outR.halfedge_.cbegin()),
      halfedgeRef.size(),
      CreateBarycentric(
          {outR.meshRelation_.barycentric.ptrD(), faceRef.ptrD(), idx.ptrD(),
           offsetQ, firstNewVert, outR.vertPos_.cptrD(), inP.vertPos_.cptrD(),
           inQ.vertPos_.cptrD(), inP.halfedge_.cptrD(), inQ.halfedge_.cptrD(),
           inP.meshRelation_.triBary.cptrD(), inQ.meshRelation_.triBary.cptrD(),
           inP.meshRelation_.barycentric.cptrD(),
           inQ.meshRelation_.barycentric.cptrD(), invertQ, outR.precision_}));
  outR.meshRelation_.barycentric.resize(idx[0]);

  return std::make_pair(faceRef, halfedgeBary);
}
}  // namespace

namespace manifold {

Manifold::Impl Boolean3::Result(Manifold::OpType op) const {
#ifdef MANIFOLD_DEBUG
  Timer assemble;
  assemble.Start();
#endif

  ASSERT((expandP_ > 0) == (op == Manifold::OpType::ADD), logicErr,
         "Result op type not compatible with constructor op type.");
  const int c1 = op == Manifold::OpType::INTERSECT ? 0 : 1;
  const int c2 = op == Manifold::OpType::ADD ? 1 : 0;
  const int c3 = op == Manifold::OpType::INTERSECT ? 1 : -1;

  if (w03_.size() == 0) {
    if (w30_.size() != 0 && op == Manifold::OpType::ADD) {
      return inQ_;
    }
    return Manifold::Impl();
  } else if (w30_.size() == 0) {
    if (op == Manifold::OpType::INTERSECT) {
      return Manifold::Impl();
    }
    return inP_;
  }

  const bool invertQ = op == Manifold::OpType::SUBTRACT;

  // Convert winding numbers to inclusion values based on operation type.
  VecDH<int> i12(x12_.size());
  VecDH<int> i21(x21_.size());
  VecDH<int> i03(w03_.size());
  VecDH<int> i30(w30_.size());
  auto policy = autoPolicy(std::max(std::max(x12_.size(), x21_.size()),
                                    std::max(w03_.size(), w30_.size())));

  transform(policy, x12_.begin(), x12_.end(), i12.begin(), c3 * _1);
  transform(policy, x21_.begin(), x21_.end(), i21.begin(), c3 * _1);
  transform(policy, w03_.begin(), w03_.end(), i03.begin(), c1 + c3 * _1);
  transform(policy, w30_.begin(), w30_.end(), i30.begin(), c2 + c3 * _1);

  VecDH<int> vP2R(inP_.NumVert());
  exclusive_scan(policy, i03.begin(), i03.end(), vP2R.begin(), 0, AbsSum());
  int numVertR = AbsSum()(vP2R.back(), i03.back());
  const int nPv = numVertR;

  VecDH<int> vQ2R(inQ_.NumVert());
  exclusive_scan(policy, i30.begin(), i30.end(), vQ2R.begin(), numVertR,
                 AbsSum());
  numVertR = AbsSum()(vQ2R.back(), i30.back());
  const int nQv = numVertR - nPv;

  VecDH<int> v12R(v12_.size());
  if (v12_.size() > 0) {
    exclusive_scan(policy, i12.begin(), i12.end(), v12R.begin(), numVertR,
                   AbsSum());
    numVertR = AbsSum()(v12R.back(), i12.back());
  }
  const int n12 = numVertR - nPv - nQv;

  VecDH<int> v21R(v21_.size());
  if (v21_.size() > 0) {
    exclusive_scan(policy, i21.begin(), i21.end(), v21R.begin(), numVertR,
                   AbsSum());
    numVertR = AbsSum()(v21R.back(), i21.back());
  }
  const int n21 = numVertR - nPv - nQv - n12;

  // Create the output Manifold
  Manifold::Impl outR;

  if (numVertR == 0) return outR;

  outR.precision_ = glm::max(inP_.precision_, inQ_.precision_);

  outR.vertPos_.resize(numVertR);
  // Add vertices, duplicating for inclusion numbers not in [-1, 1].
  // Retained vertices from P and Q:
  for_each_n(policy, zip(i03.begin(), vP2R.begin(), inP_.vertPos_.begin()),
             inP_.NumVert(), DuplicateVerts({outR.vertPos_.ptrD()}));
  for_each_n(policy, zip(i30.begin(), vQ2R.begin(), inQ_.vertPos_.begin()),
             inQ_.NumVert(), DuplicateVerts({outR.vertPos_.ptrD()}));
  // New vertices created from intersections:
  for_each_n(policy, zip(i12.begin(), v12R.begin(), v12_.begin()), i12.size(),
             DuplicateVerts({outR.vertPos_.ptrD()}));
  for_each_n(policy, zip(i21.begin(), v21R.begin(), v21_.begin()), i21.size(),
             DuplicateVerts({outR.vertPos_.ptrD()}));

  PRINT(nPv << " verts from inP");
  PRINT(nQv << " verts from inQ");
  PRINT(n12 << " new verts from edgesP -> facesQ");
  PRINT(n21 << " new verts from facesP -> edgesQ");

  // Build up new polygonal faces from triangle intersections. At this point the
  // calculation switches from parallel to serial.

  // Level 3

  // This key is the forward halfedge index of P or Q. Only includes intersected
  // edges.
  std::map<int, std::vector<EdgePos>> edgesP, edgesQ;
  // This key is the face index of <P, Q>
  std::map<std::pair<int, int>, std::vector<EdgePos>> edgesNew;

  AddNewEdgeVerts(edgesP, edgesNew, p1q2_, i12, v12R, inP_.halfedge_, true);
  AddNewEdgeVerts(edgesQ, edgesNew, p2q1_, i21, v21R, inQ_.halfedge_, false);

  // Level 4
  VecDH<int> faceEdge;
  VecDH<int> facePQ2R;
  std::tie(faceEdge, facePQ2R) =
      SizeOutput(outR, inP_, inQ_, i03, i30, i12, i21, p1q2_, p2q1_, invertQ);

  const int numFaceR = faceEdge.size() - 1;
  // This gets incremented for each halfedge that's added to a face so that the
  // next one knows where to slot in.
  VecDH<int> facePtrR = faceEdge;
  // Intersected halfedges are marked false.
  VecDH<char> wholeHalfedgeP(inP_.halfedge_.size(), true);
  VecDH<char> wholeHalfedgeQ(inQ_.halfedge_.size(), true);
  // The halfedgeRef contains the data that will become triBary once the faces
  // are triangulated.
  VecDH<Ref> halfedgeRef(2 * outR.NumEdge());

  AppendPartialEdges(outR, wholeHalfedgeP, facePtrR, edgesP, halfedgeRef, inP_,
                     i03, vP2R, facePQ2R.begin(), true);
  AppendPartialEdges(outR, wholeHalfedgeQ, facePtrR, edgesQ, halfedgeRef, inQ_,
                     i30, vQ2R, facePQ2R.begin() + inP_.NumTri(), false);

  AppendNewEdges(outR, facePtrR, edgesNew, halfedgeRef, facePQ2R,
                 inP_.NumTri());

  AppendWholeEdges(outR, facePtrR, halfedgeRef, inP_, wholeHalfedgeP, i03, vP2R,
                   facePQ2R.cptrD(), true);
  AppendWholeEdges(outR, facePtrR, halfedgeRef, inQ_, wholeHalfedgeQ, i30, vQ2R,
                   facePQ2R.cptrD() + inP_.NumTri(), false);

  VecDH<BaryRef> faceRef;
  VecDH<int> halfedgeBary;
  std::tie(faceRef, halfedgeBary) = CalculateMeshRelation(
      outR, halfedgeRef, inP_, inQ_, nPv + nQv, numFaceR, invertQ);

#ifdef MANIFOLD_DEBUG
  assemble.Stop();
  Timer triangulate;
  triangulate.Start();
#endif

  // Level 6

  if (ManifoldParams().intermediateChecks)
    ASSERT(outR.IsManifold(), logicErr, "polygon mesh is not manifold!");

  outR.Face2Tri(faceEdge, faceRef, halfedgeBary);

#ifdef MANIFOLD_DEBUG
  triangulate.Stop();
  Timer simplify;
  simplify.Start();
#endif

  if (ManifoldParams().intermediateChecks)
    ASSERT(outR.IsManifold(), logicErr, "triangulated mesh is not manifold!");

  outR.SimplifyTopology();

  if (ManifoldParams().intermediateChecks)
    ASSERT(outR.Is2Manifold(), logicErr, "simplified mesh is not 2-manifold!");

  outR.IncrementMeshIDs(0, outR.NumTri());

#ifdef MANIFOLD_DEBUG
  simplify.Stop();
  Timer sort;
  sort.Start();
#endif

  outR.Finish();

#ifdef MANIFOLD_DEBUG
  sort.Stop();
  if (ManifoldParams().verbose) {
    assemble.Print("Assembly");
    triangulate.Print("Triangulation");
    simplify.Print("Simplification");
    sort.Print("Sorting");
    std::cout << outR.NumVert() << " verts and " << outR.NumTri() << " tris"
              << std::endl;
  }
#endif

  return outR;
}

}  // namespace manifold
