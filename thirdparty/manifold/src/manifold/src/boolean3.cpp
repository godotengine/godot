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

glm::vec2 Interpolate(glm::vec3 pL, glm::vec3 pR, float x) {
  const float dxL = x - pL.x;
  const float dxR = x - pR.x;
  ASSERT(dxL * dxR <= 0, logicErr, "Boolean manifold error: not in domain");
  const bool useL = fabs(dxL) < fabs(dxR);
  const glm::vec3 dLR = pR - pL;
  const float lambda = (useL ? dxL : dxR) / dLR.x;
  if (!isfinite(lambda) || !isfinite(dLR.y) || !isfinite(dLR.z))
    return glm::vec2(pL.y, pL.z);
  glm::vec2 yz;
  yz[0] = (useL ? pL.y : pR.y) + lambda * dLR.y;
  yz[1] = (useL ? pL.z : pR.z) + lambda * dLR.z;
  return yz;
}

glm::vec4 Intersect(const glm::vec3 &pL, const glm::vec3 &pR,
                    const glm::vec3 &qL, const glm::vec3 &qR) {
  const float dyL = qL.y - pL.y;
  const float dyR = qR.y - pR.y;
  ASSERT(dyL * dyR <= 0, logicErr, "Boolean manifold error: no intersection");
  const bool useL = fabs(dyL) < fabs(dyR);
  const float dx = pR.x - pL.x;
  float lambda = (useL ? dyL : dyR) / (dyL - dyR);
  if (!isfinite(lambda)) lambda = 0.0f;
  glm::vec4 xyzz;
  xyzz.x = (useL ? pL.x : pR.x) + lambda * dx;
  const float pDy = pR.y - pL.y;
  const float qDy = qR.y - qL.y;
  const bool useP = fabs(pDy) < fabs(qDy);
  xyzz.y = (useL ? (useP ? pL.y : qL.y) : (useP ? pR.y : qR.y)) +
           lambda * (useP ? pDy : qDy);
  xyzz.z = (useL ? pL.z : pR.z) + lambda * (pR.z - pL.z);
  xyzz.w = (useL ? qL.z : qR.z) + lambda * (qR.z - qL.z);
  return xyzz;
}

template <const bool inverted>
struct CopyFaceEdges {
  const SparseIndices &p1q1;
  // const int *p1q1;
  // x can be either vert or edge (0 or 1).
  SparseIndices &pXq1;
  VecView<const Halfedge> halfedgesQ;

  void operator()(thrust::tuple<size_t, size_t> in) {
    int idx = 3 * thrust::get<0>(in);
    size_t i = thrust::get<1>(in);
    int pX = p1q1.Get(i, inverted);
    int q2 = p1q1.Get(i, !inverted);

    for (const int j : {0, 1, 2}) {
      const int q1 = 3 * q2 + j;
      const Halfedge edge = halfedgesQ[q1];
      int a = pX;
      int b = edge.IsForward() ? q1 : edge.pairedHalfedge;
      if (inverted) std::swap(a, b);
      pXq1.Set(idx + static_cast<size_t>(j), a, b);
    }
  }
};

SparseIndices Filter11(const Manifold::Impl &inP, const Manifold::Impl &inQ,
                       const SparseIndices &p1q2, const SparseIndices &p2q1) {
  ZoneScoped;
  SparseIndices p1q1(3 * p1q2.size() + 3 * p2q1.size());
  for_each_n(autoPolicy(p1q2.size()), zip(countAt(0_z), countAt(0_z)),
             p1q2.size(), CopyFaceEdges<false>({p1q2, p1q1, inQ.halfedge_}));
  for_each_n(autoPolicy(p2q1.size()), zip(countAt(p1q2.size()), countAt(0_z)),
             p2q1.size(), CopyFaceEdges<true>({p2q1, p1q1, inP.halfedge_}));
  p1q1.Unique();
  return p1q1;
}

inline bool Shadows(float p, float q, float dir) {
  return p == q ? dir < 0 : p < q;
}

inline thrust::pair<int, glm::vec2> Shadow01(
    const int p0, const int q1, VecView<const glm::vec3> vertPosP,
    VecView<const glm::vec3> vertPosQ, VecView<const Halfedge> halfedgeQ,
    const float expandP, VecView<const glm::vec3> normalP, const bool reverse) {
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

// https://github.com/scandum/binary_search/blob/master/README.md
// much faster than standard binary search on large arrays
size_t monobound_quaternary_search(VecView<const int64_t> array, int64_t key) {
  if (array.size() == 0) {
    return -1;
  }
  size_t bot = 0;
  size_t top = array.size();
  while (top >= 65536) {
    size_t mid = top / 4;
    top -= mid * 3;
    if (key < array[bot + mid * 2]) {
      if (key >= array[bot + mid]) {
        bot += mid;
      }
    } else {
      bot += mid * 2;
      if (key >= array[bot + mid]) {
        bot += mid;
      }
    }
  }

  while (top > 3) {
    size_t mid = top / 2;
    if (key >= array[bot + mid]) {
      bot += mid;
    }
    top -= mid;
  }

  while (top--) {
    if (key == array[bot + top]) {
      return bot + top;
    }
  }
  return -1;
}

struct Kernel11 {
  VecView<const glm::vec3> vertPosP;
  VecView<const glm::vec3> vertPosQ;
  VecView<const Halfedge> halfedgeP;
  VecView<const Halfedge> halfedgeQ;
  float expandP;
  VecView<const glm::vec3> normalP;
  const SparseIndices &p1q1;

  void operator()(thrust::tuple<size_t, glm::vec4 &, int &> inout) {
    const int p1 = p1q1.Get(thrust::get<0>(inout), false);
    const int q1 = p1q1.Get(thrust::get<0>(inout), true);
    glm::vec4 &xyzz11 = thrust::get<1>(inout);
    int &s11 = thrust::get<2>(inout);

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
      ASSERT(k == 2, logicErr, "Boolean manifold error: s11");
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

std::tuple<Vec<int>, Vec<glm::vec4>> Shadow11(SparseIndices &p1q1,
                                              const Manifold::Impl &inP,
                                              const Manifold::Impl &inQ,
                                              float expandP) {
  ZoneScoped;
  Vec<int> s11(p1q1.size());
  Vec<glm::vec4> xyzz11(p1q1.size());

  for_each_n(autoPolicy(p1q1.size()),
             zip(countAt(0_z), xyzz11.begin(), s11.begin()), p1q1.size(),
             Kernel11({inP.vertPos_, inQ.vertPos_, inP.halfedge_, inQ.halfedge_,
                       expandP, inP.vertNormal_, p1q1}));

  p1q1.KeepFinite(xyzz11, s11);

  return std::make_tuple(s11, xyzz11);
};

struct Kernel02 {
  VecView<const glm::vec3> vertPosP;
  VecView<const Halfedge> halfedgeQ;
  VecView<const glm::vec3> vertPosQ;
  const float expandP;
  VecView<const glm::vec3> vertNormalP;
  const SparseIndices &p0q2;
  const bool forward;

  void operator()(thrust::tuple<size_t, int &, float &> inout) {
    const int p0 = p0q2.Get(thrust::get<0>(inout), !forward);
    const int q2 = p0q2.Get(thrust::get<0>(inout), forward);
    int &s02 = thrust::get<1>(inout);
    float &z02 = thrust::get<2>(inout);

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
      ASSERT(k == 2, logicErr, "Boolean manifold error: s02");
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

std::tuple<Vec<int>, Vec<float>> Shadow02(const Manifold::Impl &inP,
                                          const Manifold::Impl &inQ,
                                          SparseIndices &p0q2, bool forward,
                                          float expandP) {
  ZoneScoped;
  Vec<int> s02(p0q2.size());
  Vec<float> z02(p0q2.size());

  auto vertNormalP = forward ? inP.vertNormal_ : inQ.vertNormal_;
  for_each_n(autoPolicy(p0q2.size()),
             zip(countAt(0_z), s02.begin(), z02.begin()), p0q2.size(),
             Kernel02({inP.vertPos_, inQ.halfedge_, inQ.vertPos_, expandP,
                       vertNormalP, p0q2, forward}));

  p0q2.KeepFinite(z02, s02);

  return std::make_tuple(s02, z02);
};

struct Kernel12 {
  VecView<const int64_t> p0q2;
  VecView<const int> s02;
  VecView<const float> z02;
  VecView<const int64_t> p1q1;
  VecView<const int> s11;
  VecView<const glm::vec4> xyzz11;
  VecView<const Halfedge> halfedgesP;
  VecView<const Halfedge> halfedgesQ;
  VecView<const glm::vec3> vertPosP;
  const bool forward;
  const SparseIndices &p1q2;

  void operator()(thrust::tuple<size_t, int &, glm::vec3 &> inout) {
    int p1 = p1q2.Get(thrust::get<0>(inout), !forward);
    int q2 = p1q2.Get(thrust::get<0>(inout), forward);
    int &x12 = thrust::get<1>(inout);
    glm::vec3 &v12 = thrust::get<2>(inout);

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
      const int64_t key = forward ? SparseIndices::EncodePQ(vert, q2)
                                  : SparseIndices::EncodePQ(q2, vert);
      const size_t idx = monobound_quaternary_search(p0q2, key);
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
      const int64_t key = forward ? SparseIndices::EncodePQ(p1, q1F)
                                  : SparseIndices::EncodePQ(q1F, p1);
      const size_t idx = monobound_quaternary_search(p1q1, key);
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
      ASSERT(k == 2, logicErr, "Boolean manifold error: v12");
      const glm::vec4 xzyy =
          Intersect(xzyLR0[0], xzyLR0[1], xzyLR1[0], xzyLR1[1]);
      v12.x = xzyy[0];
      v12.y = xzyy[2];
      v12.z = xzyy[1];
    }
  }
};

std::tuple<Vec<int>, Vec<glm::vec3>> Intersect12(
    const Manifold::Impl &inP, const Manifold::Impl &inQ, const Vec<int> &s02,
    const SparseIndices &p0q2, const Vec<int> &s11, const SparseIndices &p1q1,
    const Vec<float> &z02, const Vec<glm::vec4> &xyzz11, SparseIndices &p1q2,
    bool forward) {
  ZoneScoped;
  Vec<int> x12(p1q2.size());
  Vec<glm::vec3> v12(p1q2.size());

  for_each_n(
      autoPolicy(p1q2.size()), zip(countAt(0_z), x12.begin(), v12.begin()),
      p1q2.size(),
      Kernel12({p0q2.AsVec64(), s02, z02, p1q1.AsVec64(), s11, xyzz11,
                inP.halfedge_, inQ.halfedge_, inP.vertPos_, forward, p1q2}));

  p1q2.KeepFinite(v12, x12);

  return std::make_tuple(x12, v12);
};

Vec<int> Winding03(const Manifold::Impl &inP, Vec<int> &vertices, Vec<int> &s02,
                   bool reverse) {
  ZoneScoped;
  // verts that are not shadowed (not in p0q2) have winding number zero.
  Vec<int> w03(inP.NumVert(), 0);
  // checking is slow, so just sort and reduce
  auto policy = autoPolicy(vertices.size());
  stable_sort(
      policy, zip(vertices.begin(), s02.begin()),
      zip(vertices.end(), s02.end()),
      [](const thrust::tuple<int, int> &a, const thrust::tuple<int, int> &b) {
        return thrust::get<0>(a) < thrust::get<0>(b);
      });
  Vec<int> w03val(w03.size());
  Vec<int> w03vert(w03.size());
  // sum known s02 values into w03 (winding number)
  auto endPair = reduce_by_key<
      thrust::pair<decltype(w03val.begin()), decltype(w03val.begin())>>(
      policy, vertices.begin(), vertices.end(), s02.begin(), w03vert.begin(),
      w03val.begin());
  scatter(policy, w03val.begin(), endPair.second, w03vert.begin(), w03.begin());

  if (reverse)
    transform(policy, w03.begin(), w03.end(), w03.begin(),
              thrust::negate<int>());
  return w03;
};
}  // namespace

namespace manifold {
Boolean3::Boolean3(const Manifold::Impl &inP, const Manifold::Impl &inQ,
                   OpType op)
    : inP_(inP), inQ_(inQ), expandP_(op == OpType::Add ? 1.0 : -1.0) {
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
  p2q1_ = inP_.EdgeCollisions(inQ_, true);  // inverted

  p1q2_.Sort();
  PRINT("p1q2 size = " << p1q2_.size());

  p2q1_.Sort();
  PRINT("p2q1 size = " << p2q1_.size());

  // Level 2
  // Find vertices that overlap faces in XY-projection
  SparseIndices p0q2 = inQ.VertexCollisionsZ(inP.vertPos_);
  p0q2.Sort();
  PRINT("p0q2 size = " << p0q2.size());

  SparseIndices p2q0 = inP.VertexCollisionsZ(inQ.vertPos_, true);  // inverted
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
  Vec<int> s11;
  Vec<glm::vec4> xyzz11;
  std::tie(s11, xyzz11) = Shadow11(p1q1, inP, inQ, expandP_);
  PRINT("s11 size = " << s11.size());

  // Build up Z-projection of vertices onto triangles, keeping only those that
  // fall inside the triangle.
  Vec<int> s02;
  Vec<float> z02;
  std::tie(s02, z02) = Shadow02(inP, inQ, p0q2, true, expandP_);
  PRINT("s02 size = " << s02.size());

  Vec<int> s20;
  Vec<float> z20;
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

  if (x12_.size() + x21_.size() >= std::numeric_limits<int>::max())
    throw std::out_of_range("mesh too large");

  Vec<int> p0 = p0q2.Copy(false);
  p0q2.Resize(0);
  Vec<int> q0 = p2q0.Copy(true);
  p2q0.Resize(0);
  // Sum up the winding numbers of all vertices.
  w03_ = Winding03(inP, p0, s02, false);

  w30_ = Winding03(inQ, q0, s20, true);

#ifdef MANIFOLD_DEBUG
  intersections.Stop();

  if (ManifoldParams().verbose) {
    broad.Print("Broad phase");
    intersections.Print("Intersections");
  }
#endif
}
}  // namespace manifold
