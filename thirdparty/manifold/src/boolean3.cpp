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
#include <unordered_set>

#include "disjoint_sets.h"
#include "parallel.h"

#if (MANIFOLD_PAR == 1)
#include <tbb/combinable.h>
#endif

using namespace manifold;

namespace {

// These two functions (Interpolate and Intersect) are the only places where
// floating-point operations take place in the whole Boolean function. These
// are carefully designed to minimize rounding error and to eliminate it at edge
// cases to ensure consistency.

vec2 Interpolate(vec3 pL, vec3 pR, double x) {
  const double dxL = x - pL.x;
  const double dxR = x - pR.x;
  DEBUG_ASSERT(dxL * dxR <= 0, logicErr,
               "Boolean manifold error: not in domain");
  const bool useL = fabs(dxL) < fabs(dxR);
  const vec3 dLR = pR - pL;
  const double lambda = (useL ? dxL : dxR) / dLR.x;
  if (!std::isfinite(lambda) || !std::isfinite(dLR.y) || !std::isfinite(dLR.z))
    return vec2(pL.y, pL.z);
  vec2 yz;
  yz[0] = lambda * dLR.y + (useL ? pL.y : pR.y);
  yz[1] = lambda * dLR.z + (useL ? pL.z : pR.z);
  return yz;
}

vec4 Intersect(const vec3& pL, const vec3& pR, const vec3& qL, const vec3& qR) {
  const double dyL = qL.y - pL.y;
  const double dyR = qR.y - pR.y;
  DEBUG_ASSERT(dyL * dyR <= 0, logicErr,
               "Boolean manifold error: no intersection");
  const bool useL = fabs(dyL) < fabs(dyR);
  const double dx = pR.x - pL.x;
  double lambda = (useL ? dyL : dyR) / (dyL - dyR);
  if (!std::isfinite(lambda)) lambda = 0.0;
  vec4 xyzz;
  xyzz.x = lambda * dx + (useL ? pL.x : pR.x);
  const double pDy = pR.y - pL.y;
  const double qDy = qR.y - qL.y;
  const bool useP = fabs(pDy) < fabs(qDy);
  xyzz.y = lambda * (useP ? pDy : qDy) +
           (useL ? (useP ? pL.y : qL.y) : (useP ? pR.y : qR.y));
  xyzz.z = lambda * (pR.z - pL.z) + (useL ? pL.z : pR.z);
  xyzz.w = lambda * (qR.z - qL.z) + (useL ? qL.z : qR.z);
  return xyzz;
}

inline bool Shadows(double p, double q, double dir) {
  return p == q ? dir < 0 : p < q;
}

inline std::pair<int, vec2> Shadow01(
    const int p0, const int q1, VecView<const vec3> vertPosP,
    VecView<const vec3> vertPosQ, VecView<const Halfedge> halfedgeQ,
    const double expandP, VecView<const vec3> normal, const bool reverse) {
  const int q1s = halfedgeQ[q1].startVert;
  const int q1e = halfedgeQ[q1].endVert;
  const double p0x = vertPosP[p0].x;
  const double q1sx = vertPosQ[q1s].x;
  const double q1ex = vertPosQ[q1e].x;
  int s01 = reverse ? Shadows(q1sx, p0x, expandP * normal[q1s].x) -
                          Shadows(q1ex, p0x, expandP * normal[q1e].x)
                    : Shadows(p0x, q1ex, expandP * normal[p0].x) -
                          Shadows(p0x, q1sx, expandP * normal[p0].x);
  vec2 yz01(NAN);

  if (s01 != 0) {
    yz01 = Interpolate(vertPosQ[q1s], vertPosQ[q1e], vertPosP[p0].x);
    if (reverse) {
      vec3 diff = vertPosQ[q1s] - vertPosP[p0];
      const double start2 = la::dot(diff, diff);
      diff = vertPosQ[q1e] - vertPosP[p0];
      const double end2 = la::dot(diff, diff);
      const double dir = start2 < end2 ? normal[q1s].y : normal[q1e].y;
      if (!Shadows(yz01[0], vertPosP[p0].y, expandP * dir)) s01 = 0;
    } else {
      if (!Shadows(vertPosP[p0].y, yz01[0], expandP * normal[p0].y)) s01 = 0;
    }
  }
  return std::make_pair(s01, yz01);
}

struct Kernel11 {
  VecView<const vec3> vertPosP;
  VecView<const vec3> vertPosQ;
  VecView<const Halfedge> halfedgeP;
  VecView<const Halfedge> halfedgeQ;
  const double expandP;
  VecView<const vec3> normalP;

  std::pair<int, vec4> operator()(int p1, int q1) {
    vec4 xyzz11 = vec4(NAN);
    int s11 = 0;

    // For pRL[k], qRL[k], k==0 is the left and k==1 is the right.
    int k = 0;
    vec3 pRL[2], qRL[2];
    // Either the left or right must shadow, but not both. This ensures the
    // intersection is between the left and right.
    bool shadows = false;
    s11 = 0;

    const int p0[2] = {halfedgeP[p1].startVert, halfedgeP[p1].endVert};
    for (int i : {0, 1}) {
      const auto [s01, yz01] = Shadow01(p0[i], q1, vertPosP, vertPosQ,
                                        halfedgeQ, expandP, normalP, false);
      // If the value is NaN, then these do not overlap.
      if (std::isfinite(yz01[0])) {
        s11 += s01 * (i == 0 ? -1 : 1);
        if (k < 2 && (k == 0 || (s01 != 0) != shadows)) {
          shadows = s01 != 0;
          pRL[k] = vertPosP[p0[i]];
          qRL[k] = vec3(pRL[k].x, yz01.x, yz01.y);
          ++k;
        }
      }
    }

    const int q0[2] = {halfedgeQ[q1].startVert, halfedgeQ[q1].endVert};
    for (int i : {0, 1}) {
      const auto [s10, yz10] = Shadow01(q0[i], p1, vertPosQ, vertPosP,
                                        halfedgeP, expandP, normalP, true);
      // If the value is NaN, then these do not overlap.
      if (std::isfinite(yz10[0])) {
        s11 += s10 * (i == 0 ? -1 : 1);
        if (k < 2 && (k == 0 || (s10 != 0) != shadows)) {
          shadows = s10 != 0;
          qRL[k] = vertPosQ[q0[i]];
          pRL[k] = vec3(qRL[k].x, yz10.x, yz10.y);
          ++k;
        }
      }
    }

    if (s11 == 0) {  // No intersection
      xyzz11 = vec4(NAN);
    } else {
      DEBUG_ASSERT(k == 2, logicErr, "Boolean manifold error: s11");
      xyzz11 = Intersect(pRL[0], pRL[1], qRL[0], qRL[1]);

      const int p1s = halfedgeP[p1].startVert;
      const int p1e = halfedgeP[p1].endVert;
      vec3 diff = vertPosP[p1s] - vec3(xyzz11);
      const double start2 = la::dot(diff, diff);
      diff = vertPosP[p1e] - vec3(xyzz11);
      const double end2 = la::dot(diff, diff);
      const double dir = start2 < end2 ? normalP[p1s].z : normalP[p1e].z;

      if (!Shadows(xyzz11.z, xyzz11.w, expandP * dir)) s11 = 0;
    }

    return std::make_pair(s11, xyzz11);
  }
};

struct Kernel02 {
  VecView<const vec3> vertPosP;
  VecView<const Halfedge> halfedgeQ;
  VecView<const vec3> vertPosQ;
  const double expandP;
  VecView<const vec3> vertNormalP;
  const bool forward;

  std::pair<int, double> operator()(int p0, int q2) {
    int s02 = 0;
    double z02 = 0.0;

    // For yzzLR[k], k==0 is the left and k==1 is the right.
    int k = 0;
    vec3 yzzRL[2];
    // Either the left or right must shadow, but not both. This ensures the
    // intersection is between the left and right.
    bool shadows = false;
    int closestVert = -1;
    double minMetric = std::numeric_limits<double>::infinity();
    s02 = 0;

    const vec3 posP = vertPosP[p0];
    for (const int i : {0, 1, 2}) {
      const int q1 = 3 * q2 + i;
      const Halfedge edge = halfedgeQ[q1];
      const int q1F = edge.IsForward() ? q1 : edge.pairedHalfedge;

      if (!forward) {
        const int qVert = halfedgeQ[q1F].startVert;
        const vec3 diff = posP - vertPosQ[qVert];
        const double metric = la::dot(diff, diff);
        if (metric < minMetric) {
          minMetric = metric;
          closestVert = qVert;
        }
      }

      const auto syz01 = Shadow01(p0, q1F, vertPosP, vertPosQ, halfedgeQ,
                                  expandP, vertNormalP, !forward);
      const int s01 = syz01.first;
      const vec2 yz01 = syz01.second;
      // If the value is NaN, then these do not overlap.
      if (std::isfinite(yz01[0])) {
        s02 += s01 * (forward == edge.IsForward() ? -1 : 1);
        if (k < 2 && (k == 0 || (s01 != 0) != shadows)) {
          shadows = s01 != 0;
          yzzRL[k++] = vec3(yz01[0], yz01[1], yz01[1]);
        }
      }
    }

    if (s02 == 0) {  // No intersection
      z02 = NAN;
    } else {
      DEBUG_ASSERT(k == 2, logicErr, "Boolean manifold error: s02");
      vec3 vertPos = vertPosP[p0];
      z02 = Interpolate(yzzRL[0], yzzRL[1], vertPos.y)[1];
      if (forward) {
        if (!Shadows(vertPos.z, z02, expandP * vertNormalP[p0].z)) s02 = 0;
      } else {
        // DEBUG_ASSERT(closestVert != -1, topologyErr, "No closest vert");
        if (!Shadows(z02, vertPos.z, expandP * vertNormalP[closestVert].z))
          s02 = 0;
      }
    }
    return std::make_pair(s02, z02);
  }
};

struct Kernel12 {
  VecView<const Halfedge> halfedgesP;
  VecView<const Halfedge> halfedgesQ;
  VecView<const vec3> vertPosP;
  const bool forward;
  Kernel02 k02;
  Kernel11 k11;

  std::pair<int, vec3> operator()(int p1, int q2) {
    int x12 = 0;
    vec3 v12 = vec3(NAN);

    // For xzyLR-[k], k==0 is the left and k==1 is the right.
    int k = 0;
    vec3 xzyLR0[2];
    vec3 xzyLR1[2];
    // Either the left or right must shadow, but not both. This ensures the
    // intersection is between the left and right.
    bool shadows = false;
    x12 = 0;

    const Halfedge edge = halfedgesP[p1];

    for (int vert : {edge.startVert, edge.endVert}) {
      const auto [s, z] = k02(vert, q2);
      if (std::isfinite(z)) {
        x12 += s * ((vert == edge.startVert) == forward ? 1 : -1);
        if (k < 2 && (k == 0 || (s != 0) != shadows)) {
          shadows = s != 0;
          xzyLR0[k] = vertPosP[vert];
          std::swap(xzyLR0[k].y, xzyLR0[k].z);
          xzyLR1[k] = xzyLR0[k];
          xzyLR1[k][1] = z;
          k++;
        }
      }
    }

    for (const int i : {0, 1, 2}) {
      const int q1 = 3 * q2 + i;
      const Halfedge edge = halfedgesQ[q1];
      const int q1F = edge.IsForward() ? q1 : edge.pairedHalfedge;
      const auto [s, xyzz] = forward ? k11(p1, q1F) : k11(q1F, p1);
      if (std::isfinite(xyzz[0])) {
        x12 -= s * (edge.IsForward() ? 1 : -1);
        if (k < 2 && (k == 0 || (s != 0) != shadows)) {
          shadows = s != 0;
          xzyLR0[k][0] = xyzz.x;
          xzyLR0[k][1] = xyzz.z;
          xzyLR0[k][2] = xyzz.y;
          xzyLR1[k] = xzyLR0[k];
          xzyLR1[k][1] = xyzz.w;
          if (!forward) std::swap(xzyLR0[k][1], xzyLR1[k][1]);
          k++;
        }
      }
    }

    if (x12 == 0) {  // No intersection
      v12 = vec3(NAN);
    } else {
      DEBUG_ASSERT(k == 2, logicErr, "Boolean manifold error: v12");
      const vec4 xzyy = Intersect(xzyLR0[0], xzyLR0[1], xzyLR1[0], xzyLR1[1]);
      v12.x = xzyy[0];
      v12.y = xzyy[2];
      v12.z = xzyy[1];
    }
    return std::make_pair(x12, v12);
  }
};

struct Kernel12Tmp {
  Vec<std::array<int, 2>> p1q2_;
  Vec<int> x12_;
  Vec<vec3> v12_;
};

struct Kernel12Recorder {
  using Local = Kernel12Tmp;
  Kernel12& k12;
  bool forward;

#if MANIFOLD_PAR == 1
  tbb::combinable<Kernel12Tmp> store;
  Local& local() { return store.local(); }
#else
  Kernel12Tmp localStore;
  Local& local() { return localStore; }
#endif

  void record(int queryIdx, int leafIdx, Local& tmp) {
    const auto [x12, v12] = k12(queryIdx, leafIdx);
    if (std::isfinite(v12[0])) {
      if (forward)
        tmp.p1q2_.push_back({queryIdx, leafIdx});
      else
        tmp.p1q2_.push_back({leafIdx, queryIdx});
      tmp.x12_.push_back(x12);
      tmp.v12_.push_back(v12);
    }
  }

  Kernel12Tmp get() {
#if MANIFOLD_PAR == 1
    Kernel12Tmp result;
    std::vector<Kernel12Tmp> tmps;
    store.combine_each(
        [&](Kernel12Tmp& data) { tmps.emplace_back(std::move(data)); });
    std::vector<size_t> sizes;
    size_t total_size = 0;
    for (const auto& tmp : tmps) {
      sizes.push_back(total_size);
      total_size += tmp.x12_.size();
    }
    result.p1q2_.resize(total_size);
    result.x12_.resize(total_size);
    result.v12_.resize(total_size);
    for_each_n(ExecutionPolicy::Seq, countAt(0), tmps.size(), [&](size_t i) {
      std::copy(tmps[i].p1q2_.begin(), tmps[i].p1q2_.end(),
                result.p1q2_.begin() + sizes[i]);
      std::copy(tmps[i].x12_.begin(), tmps[i].x12_.end(),
                result.x12_.begin() + sizes[i]);
      std::copy(tmps[i].v12_.begin(), tmps[i].v12_.end(),
                result.v12_.begin() + sizes[i]);
    });
    return result;
#else
    return localStore;
#endif
  }
};

std::tuple<Vec<int>, Vec<vec3>> Intersect12(const Manifold::Impl& inP,
                                            const Manifold::Impl& inQ,
                                            Vec<std::array<int, 2>>& p1q2,
                                            double expandP, bool forward) {
  ZoneScoped;
  // a: 1 (edge), b: 2 (face)
  const Manifold::Impl& a = forward ? inP : inQ;
  const Manifold::Impl& b = forward ? inQ : inP;

  Kernel02 k02{a.vertPos_, b.halfedge_,     b.vertPos_,
               expandP,    inP.vertNormal_, forward};
  Kernel11 k11{inP.vertPos_,  inQ.vertPos_, inP.halfedge_,
               inQ.halfedge_, expandP,      inP.vertNormal_};

  Kernel12 k12{a.halfedge_, b.halfedge_, a.vertPos_, forward, k02, k11};
  Kernel12Recorder recorder{k12, forward, {}};
  auto f = [&a](int i) {
    return a.halfedge_[i].IsForward()
               ? Box(a.vertPos_[a.halfedge_[i].startVert],
                     a.vertPos_[a.halfedge_[i].endVert])
               : Box();
  };
  b.collider_.Collisions<false, decltype(f), Kernel12Recorder>(
      f, a.halfedge_.size(), recorder);

  Kernel12Tmp result = recorder.get();
  p1q2 = std::move(result.p1q2_);
  auto x12 = std::move(result.x12_);
  auto v12 = std::move(result.v12_);
  // sort p1q2 according to edges
  Vec<size_t> i12(p1q2.size());
  sequence(i12.begin(), i12.end());

  int index = forward ? 0 : 1;
  stable_sort(i12.begin(), i12.end(), [&](int a, int b) {
    return p1q2[a][index] < p1q2[b][index] ||
           (p1q2[a][index] == p1q2[b][index] &&
            p1q2[a][1 - index] < p1q2[b][1 - index]);
  });
  Permute(p1q2, i12);
  Permute(x12, i12);
  Permute(v12, i12);
  return std::make_tuple(x12, v12);
};

Vec<int> Winding03(const Manifold::Impl& inP, const Manifold::Impl& inQ,
                   const VecView<std::array<int, 2>> p1q2, double expandP,
                   bool forward) {
  ZoneScoped;
  const Manifold::Impl& a = forward ? inP : inQ;
  const Manifold::Impl& b = forward ? inQ : inP;
  Vec<int> brokenHalfedges;
  int index = forward ? 0 : 1;

  DisjointSets uA(a.vertPos_.size());
  for_each(autoPolicy(a.halfedge_.size()), countAt(0),
           countAt(a.halfedge_.size()), [&](int edge) {
             const Halfedge& he = a.halfedge_[edge];
             if (!he.IsForward()) return;
             // check if the edge is broken
             auto it = std::lower_bound(
                 p1q2.begin(), p1q2.end(), edge,
                 [index](const std::array<int, 2>& collisionPair, int e) {
                   return collisionPair[index] < e;
                 });
             if (it == p1q2.end() || (*it)[index] != edge)
               uA.unite(he.startVert, he.endVert);
           });

  // find components, the hope is the number of components should be small
  std::unordered_set<int> components;
#if (MANIFOLD_PAR == 1)
  if (a.vertPos_.size() > 1e5) {
    tbb::combinable<std::unordered_set<int>> componentsShared;
    for_each(autoPolicy(a.vertPos_.size()), countAt(0),
             countAt(a.vertPos_.size()),
             [&](int v) { componentsShared.local().insert(uA.find(v)); });
    componentsShared.combine_each([&](const std::unordered_set<int>& data) {
      components.insert(data.begin(), data.end());
    });
  } else
#endif
  {
    for (size_t v = 0; v < a.vertPos_.size(); v++)
      components.insert(uA.find(v));
  }
  Vec<int> verts;
  verts.reserve(components.size());
  for (int c : components) verts.push_back(c);

  Vec<int> w03(a.NumVert(), 0);
  Kernel02 k02{a.vertPos_, b.halfedge_,     b.vertPos_,
               expandP,    inP.vertNormal_, forward};
  auto recorderf = [&](int i, int b) {
    const auto [s02, z02] = k02(verts[i], b);
    if (std::isfinite(z02)) w03[verts[i]] += s02 * (!forward ? -1 : 1);
  };
  auto recorder = MakeSimpleRecorder(recorderf);
  auto f = [&](int i) { return a.vertPos_[verts[i]]; };
  b.collider_.Collisions<false, decltype(f), decltype(recorder)>(
      f, verts.size(), recorder);
  // flood fill
  for_each(autoPolicy(w03.size()), countAt(0), countAt(w03.size()),
           [&](size_t i) {
             size_t root = uA.find(i);
             if (root == i) return;
             w03[i] = w03[root];
           });
  return w03;
}
}  // namespace

namespace manifold {
Boolean3::Boolean3(const Manifold::Impl& inP, const Manifold::Impl& inQ,
                   OpType op)
    : inP_(inP), inQ_(inQ), expandP_(op == OpType::Add ? 1.0 : -1.0) {
  // Symbolic perturbation:
  // Union -> expand inP
  // Difference, Intersection -> contract inP
  constexpr size_t INT_MAX_SZ =
      static_cast<size_t>(std::numeric_limits<int>::max());

  if (inP.IsEmpty() || inQ.IsEmpty() || !inP.bBox_.DoesOverlap(inQ.bBox_)) {
    PRINT("No overlap, early out");
    w03_.resize(inP.NumVert(), 0);
    w30_.resize(inQ.NumVert(), 0);
    return;
  }

#ifdef MANIFOLD_DEBUG
  Timer intersections;
  intersections.Start();
#endif

  // Level 3
  // Build up the intersection of the edges and triangles, keeping only those
  // that intersect, and record the direction the edge is passing through the
  // triangle.
  std::tie(x12_, v12_) = Intersect12(inP, inQ, p1q2_, expandP_, true);
  PRINT("x12 size = " << x12_.size());

  std::tie(x21_, v21_) = Intersect12(inP, inQ, p2q1_, expandP_, false);
  PRINT("x21 size = " << x21_.size());

  if (x12_.size() > INT_MAX_SZ || x21_.size() > INT_MAX_SZ) {
    valid = false;
    return;
  }

  // Compute winding numbers of all vertices using flood fill
  // Vertices on the same connected component have the same winding number
  w03_ = Winding03(inP, inQ, p1q2_, expandP_, true);
  w30_ = Winding03(inP, inQ, p2q1_, expandP_, false);

#ifdef MANIFOLD_DEBUG
  intersections.Stop();

  if (ManifoldParams().verbose) {
    intersections.Print("Intersections");
  }
#endif
}
}  // namespace manifold
