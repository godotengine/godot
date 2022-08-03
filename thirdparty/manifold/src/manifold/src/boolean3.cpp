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

#include <limits>

#include "par.h"

using namespace manifold;

namespace {

// These two functions (Interpolate and Intersect) are the only places where
// floating-point operations take place in the whole Boolean function. These are
// carefully designed to minimize rounding error and to eliminate it at edge
// cases to ensure consistency.

__host__ __device__ glm::vec2 Interpolate(glm::vec3 pL, glm::vec3 pR, float x) {
  float dxL = x - pL.x;
  float dxR = x - pR.x;
  if (dxL * dxR > 0) printf("Not in domain!\n");
  bool useL = fabs(dxL) < fabs(dxR);
  float lambda = (useL ? dxL : dxR) / (pR.x - pL.x);
  if (!isfinite(lambda)) return glm::vec2(pL.y, pL.z);
  glm::vec2 yz;
  yz[0] = (useL ? pL.y : pR.y) + lambda * (pR.y - pL.y);
  yz[1] = (useL ? pL.z : pR.z) + lambda * (pR.z - pL.z);
  return yz;
}

__host__ __device__ glm::vec4 Intersect(const glm::vec3 &pL,
                                        const glm::vec3 &pR,
                                        const glm::vec3 &qL,
                                        const glm::vec3 &qR) {
  float dyL = qL.y - pL.y;
  float dyR = qR.y - pR.y;
  if (dyL * dyR > 0) printf("No intersection!\n");
  bool useL = fabs(dyL) < fabs(dyR);
  float dx = pR.x - pL.x;
  float lambda = (useL ? dyL : dyR) / (dyL - dyR);
  if (!isfinite(lambda)) lambda = 0.0f;
  glm::vec4 xyzz;
  xyzz.x = (useL ? pL.x : pR.x) + lambda * dx;
  float pDy = pR.y - pL.y;
  float qDy = qR.y - qL.y;
  bool useP = fabs(pDy) < fabs(qDy);
  xyzz.y = (useL ? (useP ? pL.y : qL.y) : (useP ? pR.y : qR.y)) +
           lambda * (useP ? pDy : qDy);
  xyzz.z = (useL ? pL.z : pR.z) + lambda * (pR.z - pL.z);
  xyzz.w = (useL ? qL.z : qR.z) + lambda * (qR.z - qL.z);
  return xyzz;
}

struct CopyFaceEdges {
  // x can be either vert or edge (0 or 1).
  thrust::pair<int *, int *> pXq1;
  const Halfedge *halfedgesQ;

  __host__ __device__ void operator()(thrust::tuple<int, int, int> in) {
    int idx = 3 * thrust::get<0>(in);
    const int pX = thrust::get<1>(in);
    const int q2 = thrust::get<2>(in);

    for (const int i : {0, 1, 2}) {
      pXq1.first[idx + i] = pX;
      const int q1 = 3 * q2 + i;
      const Halfedge edge = halfedgesQ[q1];
      pXq1.second[idx + i] = edge.IsForward() ? q1 : edge.pairedHalfedge;
    }
  }
};

SparseIndices Filter11(const Manifold::Impl &inP, const Manifold::Impl &inQ,
                       const SparseIndices &p1q2, const SparseIndices &p2q1) {
  SparseIndices p1q1(3 * p1q2.size() + 3 * p2q1.size());
  auto policy = autoPolicy(p1q2.size());
  for_each_n(policy, zip(countAt(0), p1q2.begin(0), p1q2.begin(1)), p1q2.size(),
             CopyFaceEdges({p1q1.ptrDpq(), inQ.halfedge_.cptrD()}));

  p1q1.SwapPQ();
  for_each_n(policy, zip(countAt(p1q2.size()), p2q1.begin(1), p2q1.begin(0)),
             p2q1.size(),
             CopyFaceEdges({p1q1.ptrDpq(), inP.halfedge_.cptrD()}));
  p1q1.SwapPQ();
  p1q1.Unique();
  return p1q1;
}

__host__ __device__ bool Shadows(float p, float q, float dir) {
  return p == q ? dir < 0 : p < q;
}

__host__ __device__ thrust::pair<int, glm::vec2> Shadow01(
    const int p0, const int q1, const glm::vec3 *vertPosP,
    const glm::vec3 *vertPosQ, const Halfedge *halfedgeQ, const float expandP,
    const glm::vec3 *normalP, const bool reverse) {
  const int q1s = halfedgeQ[q1].startVert;
  const int q1e = halfedgeQ[q1].endVert;
  const float p0x = vertPosP[p0].x;
  const float q1sx = vertPosQ[q1s].x;
  const float q1ex = vertPosQ[q1e].x;
  int s01 = reverse ? Shadows(q1sx, p0x, expandP * normalP[q1s].x) -
                          Shadows(q1ex, p0x, expandP * normalP[q1e].x)
                    : Shadows(p0x, q1ex, expandP * normalP[p0].x) -
                          Shadows(p0x, q1sx, expandP * normalP[p0].x);
  glm::vec2 yz01(NAN);

  if (s01 != 0) {
    yz01 = Interpolate(vertPosQ[q1s], vertPosQ[q1e], vertPosP[p0].x);
    if (reverse) {
      glm::vec3 diff = vertPosQ[q1s] - vertPosP[p0];
      const float start2 = glm::dot(diff, diff);
      diff = vertPosQ[q1e] - vertPosP[p0];
      const float end2 = glm::dot(diff, diff);
      const float dir = start2 < end2 ? normalP[q1s].y : normalP[q1e].y;
      if (!Shadows(yz01[0], vertPosP[p0].y, expandP * dir)) s01 = 0;
    } else {
      if (!Shadows(vertPosP[p0].y, yz01[0], expandP * normalP[p0].y)) s01 = 0;
    }
  }
  return thrust::make_pair(s01, yz01);
}

__host__ __device__ int BinarySearch(
    const thrust::pair<const int *, const int *> keys, const int size,
    const thrust::pair<int, int> key) {
  if (size <= 0) return -1;
  int left = 0;
  int right = size - 1;
  int m;
  thrust::pair<int, int> keyM;
  while (1) {
    m = right - (right - left) / 2;
    keyM = thrust::make_pair(keys.first[m], keys.second[m]);
    if (left == right) break;
    if (keyM > key)
      right = m - 1;
    else
      left = m;
  }
  if (keyM == key)
    return m;
  else
    return -1;
}

struct Kernel11 {
  const glm::vec3 *vertPosP;
  const glm::vec3 *vertPosQ;
  const Halfedge *halfedgeP;
  const Halfedge *halfedgeQ;
  float expandP;
  const glm::vec3 *normalP;

  __host__ __device__ void operator()(
      thrust::tuple<glm::vec4 &, int &, int, int> inout) {
    glm::vec4 &xyzz11 = thrust::get<0>(inout);
    int &s11 = thrust::get<1>(inout);
    const int p1 = thrust::get<2>(inout);
    const int q1 = thrust::get<3>(inout);

    // For pRL[k], qRL[k], k==0 is the left and k==1 is the right.
    int k = 0;
    glm::vec3 pRL[2], qRL[2];
    // Either the left or right must shadow, but not both. This ensures the
    // intersection is between the left and right.
    bool shadows = false;
    s11 = 0;

    const int p0[2] = {halfedgeP[p1].startVert, halfedgeP[p1].endVert};
    for (int i : {0, 1}) {
      const auto syz01 = Shadow01(p0[i], q1, vertPosP, vertPosQ, halfedgeQ,
                                  expandP, normalP, false);
      const int s01 = syz01.first;
      const glm::vec2 yz01 = syz01.second;
      // If the value is NaN, then these do not overlap.
      if (isfinite(yz01[0])) {
        s11 += s01 * (i == 0 ? -1 : 1);
        if (k < 2 && (k == 0 || (s01 != 0) != shadows)) {
          shadows = s01 != 0;
          pRL[k] = vertPosP[p0[i]];
          qRL[k] = glm::vec3(pRL[k].x, yz01);
          ++k;
        }
      }
    }

    const int q0[2] = {halfedgeQ[q1].startVert, halfedgeQ[q1].endVert};
    for (int i : {0, 1}) {
      const auto syz10 = Shadow01(q0[i], p1, vertPosQ, vertPosP, halfedgeP,
                                  expandP, normalP, true);
      const int s10 = syz10.first;
      const glm::vec2 yz10 = syz10.second;
      // If the value is NaN, then these do not overlap.
      if (isfinite(yz10[0])) {
        s11 += s10 * (i == 0 ? -1 : 1);
        if (k < 2 && (k == 0 || (s10 != 0) != shadows)) {
          shadows = s10 != 0;
          qRL[k] = vertPosQ[q0[i]];
          pRL[k] = glm::vec3(qRL[k].x, yz10);
          ++k;
        }
      }
    }

    if (s11 == 0) {  // No intersection
      xyzz11 = glm::vec4(NAN);
    } else {
      // Assert left and right were both found
      if (k != 2) {
        printf("k = %d\n", k);
      }

      xyzz11 = Intersect(pRL[0], pRL[1], qRL[0], qRL[1]);

      const int p1s = halfedgeP[p1].startVert;
      const int p1e = halfedgeP[p1].endVert;
      glm::vec3 diff = vertPosP[p1s] - glm::vec3(xyzz11);
      const float start2 = glm::dot(diff, diff);
      diff = vertPosP[p1e] - glm::vec3(xyzz11);
      const float end2 = glm::dot(diff, diff);
      const float dir = start2 < end2 ? normalP[p1s].z : normalP[p1e].z;

      if (!Shadows(xyzz11.z, xyzz11.w, expandP * dir)) s11 = 0;
    }
  }
};

std::tuple<VecDH<int>, VecDH<glm::vec4>> Shadow11(SparseIndices &p1q1,
                                                  const Manifold::Impl &inP,
                                                  const Manifold::Impl &inQ,
                                                  float expandP) {
  VecDH<int> s11(p1q1.size());
  VecDH<glm::vec4> xyzz11(p1q1.size());

  for_each_n(autoPolicy(p1q1.size()),
             zip(xyzz11.begin(), s11.begin(), p1q1.begin(0), p1q1.begin(1)),
             p1q1.size(),
             Kernel11({inP.vertPos_.cptrD(), inQ.vertPos_.cptrD(),
                       inP.halfedge_.cptrD(), inQ.halfedge_.cptrD(), expandP,
                       inP.vertNormal_.cptrD()}));

  p1q1.KeepFinite(xyzz11, s11);

  return std::make_tuple(s11, xyzz11);
};

struct Kernel02 {
  const glm::vec3 *vertPosP;
  const Halfedge *halfedgeQ;
  const glm::vec3 *vertPosQ;
  const bool forward;
  const float expandP;
  const glm::vec3 *vertNormalP;

  __host__ __device__ void operator()(
      thrust::tuple<int &, float &, int, int> inout) {
    int &s02 = thrust::get<0>(inout);
    float &z02 = thrust::get<1>(inout);
    const int p0 = thrust::get<2>(inout);
    const int q2 = thrust::get<3>(inout);

    // For yzzLR[k], k==0 is the left and k==1 is the right.
    int k = 0;
    glm::vec3 yzzRL[2];
    // Either the left or right must shadow, but not both. This ensures the
    // intersection is between the left and right.
    bool shadows = false;
    int closestVert = -1;
    float minMetric = std::numeric_limits<float>::infinity();
    s02 = 0;

    const glm::vec3 posP = vertPosP[p0];
    for (const int i : {0, 1, 2}) {
      const int q1 = 3 * q2 + i;
      const Halfedge edge = halfedgeQ[q1];
      const int q1F = edge.IsForward() ? q1 : edge.pairedHalfedge;

      if (!forward) {
        const int qVert = halfedgeQ[q1F].startVert;
        const glm::vec3 diff = posP - vertPosQ[qVert];
        const float metric = glm::dot(diff, diff);
        if (metric < minMetric) {
          minMetric = metric;
          closestVert = qVert;
        }
      }

      const auto syz01 = Shadow01(p0, q1F, vertPosP, vertPosQ, halfedgeQ,
                                  expandP, vertNormalP, !forward);
      const int s01 = syz01.first;
      const glm::vec2 yz01 = syz01.second;
      // If the value is NaN, then these do not overlap.
      if (isfinite(yz01[0])) {
        s02 += s01 * (forward == edge.IsForward() ? -1 : 1);
        if (k < 2 && (k == 0 || (s01 != 0) != shadows)) {
          shadows = s01 != 0;
          yzzRL[k++] = glm::vec3(yz01[0], yz01[1], yz01[1]);
        }
      }
    }

    if (s02 == 0) {  // No intersection
      z02 = NAN;
    } else {
      // Assert left and right were both found
      if (k != 2) {
        printf("k = %d\n", k);
      }

      glm::vec3 vertPos = vertPosP[p0];
      z02 = Interpolate(yzzRL[0], yzzRL[1], vertPos.y)[1];
      if (forward) {
        if (!Shadows(vertPos.z, z02, expandP * vertNormalP[p0].z)) s02 = 0;
      } else {
        // ASSERT(closestVert != -1, topologyErr, "No closest vert");
        if (!Shadows(z02, vertPos.z, expandP * vertNormalP[closestVert].z))
          s02 = 0;
      }
    }
  }
};

std::tuple<VecDH<int>, VecDH<float>> Shadow02(const Manifold::Impl &inP,
                                              const Manifold::Impl &inQ,
                                              SparseIndices &p0q2, bool forward,
                                              float expandP) {
  VecDH<int> s02(p0q2.size());
  VecDH<float> z02(p0q2.size());

  auto vertNormalP =
      forward ? inP.vertNormal_.cptrD() : inQ.vertNormal_.cptrD();
  for_each_n(
      autoPolicy(p0q2.size()),
      zip(s02.begin(), z02.begin(), p0q2.begin(!forward), p0q2.begin(forward)),
      p0q2.size(),
      Kernel02({inP.vertPos_.cptrD(), inQ.halfedge_.cptrD(),
                inQ.vertPos_.cptrD(), forward, expandP, vertNormalP}));

  p0q2.KeepFinite(z02, s02);

  return std::make_tuple(s02, z02);
};

struct Kernel12 {
  const thrust::pair<const int *, const int *> p0q2;
  const int *s02;
  const float *z02;
  const int size02;
  const thrust::pair<const int *, const int *> p1q1;
  const int *s11;
  const glm::vec4 *xyzz11;
  const int size11;
  const Halfedge *halfedgesP;
  const Halfedge *halfedgesQ;
  const glm::vec3 *vertPosP;
  const bool forward;

  __host__ __device__ void operator()(
      thrust::tuple<int &, glm::vec3 &, int, int> inout) {
    int &x12 = thrust::get<0>(inout);
    glm::vec3 &v12 = thrust::get<1>(inout);
    const int p1 = thrust::get<2>(inout);
    const int q2 = thrust::get<3>(inout);

    // For xzyLR-[k], k==0 is the left and k==1 is the right.
    int k = 0;
    glm::vec3 xzyLR0[2];
    glm::vec3 xzyLR1[2];
    // Either the left or right must shadow, but not both. This ensures the
    // intersection is between the left and right.
    bool shadows = false;
    x12 = 0;

    const Halfedge edge = halfedgesP[p1];

    for (int vert : {edge.startVert, edge.endVert}) {
      const auto key =
          forward ? thrust::make_pair(vert, q2) : thrust::make_pair(q2, vert);
      const int idx = BinarySearch(p0q2, size02, key);
      if (idx != -1) {
        const int s = s02[idx];
        x12 += s * ((vert == edge.startVert) == forward ? 1 : -1);
        if (k < 2 && (k == 0 || (s != 0) != shadows)) {
          shadows = s != 0;
          xzyLR0[k] = vertPosP[vert];
          thrust::swap(xzyLR0[k].y, xzyLR0[k].z);
          xzyLR1[k] = xzyLR0[k];
          xzyLR1[k][1] = z02[idx];
          k++;
        }
      }
    }

    for (const int i : {0, 1, 2}) {
      const int q1 = 3 * q2 + i;
      const Halfedge edge = halfedgesQ[q1];
      const int q1F = edge.IsForward() ? q1 : edge.pairedHalfedge;
      const auto key =
          forward ? thrust::make_pair(p1, q1F) : thrust::make_pair(q1F, p1);
      const int idx = BinarySearch(p1q1, size11, key);
      if (idx != -1) {  // s is implicitly zero for anything not found
        const int s = s11[idx];
        x12 -= s * (edge.IsForward() ? 1 : -1);
        if (k < 2 && (k == 0 || (s != 0) != shadows)) {
          shadows = s != 0;
          const glm::vec4 xyzz = xyzz11[idx];
          xzyLR0[k][0] = xyzz.x;
          xzyLR0[k][1] = xyzz.z;
          xzyLR0[k][2] = xyzz.y;
          xzyLR1[k] = xzyLR0[k];
          xzyLR1[k][1] = xyzz.w;
          if (!forward) thrust::swap(xzyLR0[k][1], xzyLR1[k][1]);
          k++;
        }
      }
    }

    if (x12 == 0) {  // No intersection
      v12 = glm::vec3(NAN);
    } else {
      // Assert left and right were both found
      if (k != 2) {
        printf("k = %d\n", k);
      }
      const glm::vec4 xzyy =
          Intersect(xzyLR0[0], xzyLR0[1], xzyLR1[0], xzyLR1[1]);
      v12.x = xzyy[0];
      v12.y = xzyy[2];
      v12.z = xzyy[1];
    }
  }
};

std::tuple<VecDH<int>, VecDH<glm::vec3>> Intersect12(
    const Manifold::Impl &inP, const Manifold::Impl &inQ, const VecDH<int> &s02,
    const SparseIndices &p0q2, const VecDH<int> &s11, const SparseIndices &p1q1,
    const VecDH<float> &z02, const VecDH<glm::vec4> &xyzz11,
    SparseIndices &p1q2, bool forward) {
  VecDH<int> x12(p1q2.size());
  VecDH<glm::vec3> v12(p1q2.size());

  for_each_n(
      autoPolicy(p1q2.size()),
      zip(x12.begin(), v12.begin(), p1q2.begin(!forward), p1q2.begin(forward)),
      p1q2.size(),
      Kernel12({p0q2.ptrDpq(), s02.ptrD(), z02.cptrD(), p0q2.size(),
                p1q1.ptrDpq(), s11.ptrD(), xyzz11.cptrD(), p1q1.size(),
                inP.halfedge_.cptrD(), inQ.halfedge_.cptrD(),
                inP.vertPos_.cptrD(), forward}));

  p1q2.KeepFinite(v12, x12);

  return std::make_tuple(x12, v12);
};

VecDH<int> Winding03(const Manifold::Impl &inP, SparseIndices &p0q2,
                     VecDH<int> &s02, bool reverse) {
  // verts that are not shadowed (not in p0q2) have winding number zero.
  VecDH<int> w03(inP.NumVert(), 0);

  auto policy = autoPolicy(p0q2.size());
  if (!is_sorted(policy, p0q2.begin(reverse), p0q2.end(reverse)))
    sort_by_key(policy, p0q2.begin(reverse), p0q2.end(reverse), s02.begin());
  VecDH<int> w03val(w03.size());
  VecDH<int> w03vert(w03.size());
  // sum known s02 values into w03 (winding number)
  auto endPair = reduce_by_key<
      thrust::pair<decltype(w03val.begin()), decltype(w03val.begin())>>(
      policy, p0q2.begin(reverse), p0q2.end(reverse), s02.begin(),
      w03vert.begin(), w03val.begin());
  scatter(autoPolicy(endPair.second - w03val.begin()), w03val.begin(),
          endPair.second, w03vert.begin(), w03.begin());

  if (reverse)
    transform(autoPolicy(w03.size()), w03.begin(), w03.end(), w03.begin(),
              thrust::negate<int>());
  return w03;
};
}  // namespace

namespace manifold {
Boolean3::Boolean3(const Manifold::Impl &inP, const Manifold::Impl &inQ,
                   Manifold::OpType op)
    : inP_(inP), inQ_(inQ), expandP_(op == Manifold::OpType::ADD ? 1.0 : -1.0) {
  // Symbolic perturbation:
  // Union -> expand inP
  // Difference, Intersection -> contract inP

#ifdef MANIFOLD_DEBUG
  Timer broad;
  broad.Start();
#endif

  if (inP.IsEmpty() || inQ.IsEmpty() || !inP.bBox_.DoesOverlap(inQ.bBox_)) {
    PRINT("No overlap, early out");
    w03_.resize(inP.NumVert(), 0);
    w30_.resize(inQ.NumVert(), 0);
    return;
  }

  // Level 3
  // Find edge-triangle overlaps (broad phase)
  p1q2_ = inQ_.EdgeCollisions(inP_);
  p1q2_.Sort();
  PRINT("p1q2 size = " << p1q2_.size());

  p2q1_ = inP_.EdgeCollisions(inQ_);
  p2q1_.SwapPQ();
  p2q1_.Sort();
  PRINT("p2q1 size = " << p2q1_.size());

  // Level 2
  // Find vertices that overlap faces in XY-projection
  SparseIndices p0q2 = inQ.VertexCollisionsZ(inP.vertPos_);
  p0q2.Sort();
  PRINT("p0q2 size = " << p0q2.size());

  SparseIndices p2q0 = inP.VertexCollisionsZ(inQ.vertPos_);
  p2q0.SwapPQ();
  p2q0.Sort();
  PRINT("p2q0 size = " << p2q0.size());

  // Find involved edge pairs from Level 3
  SparseIndices p1q1 = Filter11(inP_, inQ_, p1q2_, p2q1_);
  PRINT("p1q1 size = " << p1q1.size());

#ifdef MANIFOLD_DEBUG
  broad.Stop();
  Timer intersections;
  intersections.Start();
#endif

  // Level 2
  // Build up XY-projection intersection of two edges, including the z-value for
  // each edge, keeping only those whose intersection exists.
  VecDH<int> s11;
  VecDH<glm::vec4> xyzz11;
  std::tie(s11, xyzz11) = Shadow11(p1q1, inP, inQ, expandP_);
  PRINT("s11 size = " << s11.size());

  // Build up Z-projection of vertices onto triangles, keeping only those that
  // fall inside the triangle.
  VecDH<int> s02;
  VecDH<float> z02;
  std::tie(s02, z02) = Shadow02(inP, inQ, p0q2, true, expandP_);
  PRINT("s02 size = " << s02.size());

  VecDH<int> s20;
  VecDH<float> z20;
  std::tie(s20, z20) = Shadow02(inQ, inP, p2q0, false, expandP_);
  PRINT("s20 size = " << s20.size());

  // Level 3
  // Build up the intersection of the edges and triangles, keeping only those
  // that intersect, and record the direction the edge is passing through the
  // triangle.
  std::tie(x12_, v12_) =
      Intersect12(inP, inQ, s02, p0q2, s11, p1q1, z02, xyzz11, p1q2_, true);
  PRINT("x12 size = " << x12_.size());

  std::tie(x21_, v21_) =
      Intersect12(inQ, inP, s20, p2q0, s11, p1q1, z20, xyzz11, p2q1_, false);
  PRINT("x21 size = " << x21_.size());

  // Sum up the winding numbers of all vertices.
  w03_ = Winding03(inP, p0q2, s02, false);

  w30_ = Winding03(inQ, p2q0, s20, true);

#ifdef MANIFOLD_DEBUG
  intersections.Stop();

  if (ManifoldParams().verbose) {
    broad.Print("Broad phase");
    intersections.Print("Intersections");
    MemUsage();
  }
#endif
}
}  // namespace manifold
