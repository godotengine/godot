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

#include <limits>

#if MANIFOLD_PAR == 1
#include <tbb/combinable.h>
#endif

#include "impl.h"
#include "parallel.h"
#include "tri_dist.h"

namespace {
using namespace manifold;

struct CurvatureAngles {
  VecView<double> meanCurvature;
  VecView<double> gaussianCurvature;
  VecView<double> area;
  VecView<double> degree;
  const Halfedges& halfedge;
  VecView<const vec3> vertPos;
  VecView<const vec3> triNormal;

  void operator()(size_t tri) {
    vec3 edge[3];
    vec3 edgeLength(0.0);
    for (int i : {0, 1, 2}) {
      const int edgeIdx = 3 * tri + i;
      const int startVert = halfedge.Start(edgeIdx);
      const int endVert = halfedge.End(edgeIdx);
      edge[i] = vertPos[endVert] - vertPos[startVert];
      edgeLength[i] = la::length(edge[i]);
      edge[i] /= edgeLength[i];
      const int neighborTri = halfedge.Pair(edgeIdx) / 3;
      const double dihedral =
          0.25 * edgeLength[i] *
          math::asin(la::dot(la::cross(triNormal[tri], triNormal[neighborTri]),
                             edge[i]));
      AtomicAdd(meanCurvature[startVert], dihedral);
      AtomicAdd(meanCurvature[endVert], dihedral);
      AtomicAdd(degree[startVert], 1.0);
    }

    vec3 phi;
    phi[0] = math::acos(-la::dot(edge[2], edge[0]));
    phi[1] = math::acos(-la::dot(edge[0], edge[1]));
    phi[2] = kPi - phi[0] - phi[1];
    const double area3 = edgeLength[0] * edgeLength[1] *
                         la::length(la::cross(edge[0], edge[1])) / 6;

    for (int i : {0, 1, 2}) {
      const int vert = halfedge.Start(3 * tri + i);
      AtomicAdd(gaussianCurvature[vert], -phi[i]);
      AtomicAdd(area[vert], area3);
    }
  }
};

struct CheckHalfedges {
  const Halfedges& halfedges;

  bool operator()(size_t edge) const {
    const int start = halfedges.Start(edge);
    const int end = halfedges.End(edge);
    const int pair = halfedges.Pair(edge);
    if (start == -1 && end == -1 && pair == -1) return true;
    if (halfedges.Start(NextHalfedge(edge)) == -1 ||
        halfedges.Start(NextHalfedge(NextHalfedge(edge))) == -1) {
      return false;
    }
    if (pair == -1) return false;

    bool good = true;
    good &= halfedges.Pair(pair) == static_cast<int>(edge);
    good &= start != end;
    good &= start == halfedges.End(pair);
    good &= end == halfedges.Start(pair);
    return good;
  }
};
}  // namespace

namespace manifold {

/**
 * Returns true if this manifold is in fact an oriented even manifold and all of
 * the data structures are consistent.
 */
bool Manifold::Impl::IsManifold() const {
  if (halfedge_.size() == 0) return true;
  if (halfedge_.size() % 3 != 0) return false;
  return all_of(countAt(0_uz), countAt(halfedge_.size()),
                CheckHalfedges{halfedge_});
}

/**
 * Returns true if this manifold is in fact an oriented 2-manifold and all of
 * the data structures are consistent.
 */
bool Manifold::Impl::Is2Manifold() const {
  if (halfedge_.size() == 0) return true;
  if (!IsManifold()) return false;

  Vec<Halfedge> halfedge = halfedge_.ToData();
  stable_sort(halfedge.begin(), halfedge.end());

  return all_of(
      countAt(0_uz), countAt(2 * NumEdge() - 1), [&halfedge](size_t edge) {
        const Halfedge h = halfedge[edge];
        if (h.startVert == -1 && h.endVert == -1 && h.pairedHalfedge == -1)
          return true;
        return h.startVert != halfedge[edge + 1].startVert ||
               h.endVert != halfedge[edge + 1].endVert;
      });
}

#ifdef MANIFOLD_DEBUG
std::mutex dump_lock;
#endif

/**
 * Returns true if this manifold is self-intersecting.
 * Note that this is not checking for epsilon-validity.
 */
bool Manifold::Impl::IsSelfIntersecting() const {
  const double ep = 2 * epsilon_;
  const double epsilonSq = ep * ep;
  Vec<Box> faceBox;
  Vec<uint32_t> faceMorton;
  GetFaceBoxMorton(faceBox, faceMorton);

  std::atomic<bool> intersecting(false);

  auto f = [&](int tri0, int tri1) {
    std::array<vec3, 3> triVerts0, triVerts1;
    for (int i : {0, 1, 2}) {
      triVerts0[i] = vertPos_[halfedge_.Start(3 * tri0 + i)];
      triVerts1[i] = vertPos_[halfedge_.Start(3 * tri1 + i)];
    }
    // if triangles tri0 and tri1 share a vertex, return true to skip the
    // check. we relax the sharing criteria a bit to allow for at most
    // distance epsilon squared
    for (int i : {0, 1, 2})
      for (int j : {0, 1, 2})
        if (distance2(triVerts0[i], triVerts1[j]) <= epsilonSq) return;

    if (DistanceTriangleTriangleSquared(triVerts0, triVerts1) == 0.0) {
      // try to move the triangles around the normal of the other face
      std::array<vec3, 3> tmp0, tmp1;
      for (int i : {0, 1, 2}) tmp0[i] = triVerts0[i] + ep * faceNormal_[tri1];
      if (DistanceTriangleTriangleSquared(tmp0, triVerts1) > 0.0) return;
      for (int i : {0, 1, 2}) tmp0[i] = triVerts0[i] - ep * faceNormal_[tri1];
      if (DistanceTriangleTriangleSquared(tmp0, triVerts1) > 0.0) return;
      for (int i : {0, 1, 2}) tmp1[i] = triVerts1[i] + ep * faceNormal_[tri0];
      if (DistanceTriangleTriangleSquared(triVerts0, tmp1) > 0.0) return;
      for (int i : {0, 1, 2}) tmp1[i] = triVerts1[i] - ep * faceNormal_[tri0];
      if (DistanceTriangleTriangleSquared(triVerts0, tmp1) > 0.0) return;

#ifdef MANIFOLD_DEBUG
      if (ManifoldParams().verbose > 0) {
        dump_lock.lock();
        std::cout << "intersecting:" << std::endl;
        for (int i : {0, 1, 2}) std::cout << triVerts0[i] << " ";
        std::cout << std::endl;
        for (int i : {0, 1, 2}) std::cout << triVerts1[i] << " ";
        std::cout << std::endl;
        dump_lock.unlock();
      }
#endif
      intersecting.store(true);
    }
  };

  auto recorder = MakeSimpleRecorder(f);
  collider_.Collisions<true>(recorder, faceBox.cview());

  return intersecting.load();
}

/**
 * Returns true if all triangles are CCW relative to their triNormals_.
 */
bool Manifold::Impl::MatchesTriNormals() const {
  if (halfedge_.size() == 0 || faceNormal_.size() != NumTri()) return true;
  return all_of(countAt(0_uz), countAt(NumTri()), [this](size_t face) {
    if (halfedge_.Pair(3 * face) < 0) return true;

    const mat2x3 projection = GetAxisAlignedProjection(faceNormal_[face]);
    vec2 v[3];
    double max = -std::numeric_limits<double>::infinity();
    double min = std::numeric_limits<double>::infinity();
    for (int i : {0, 1, 2}) {
      const vec3 p = vertPos_[halfedge_.Start(3 * face + i)];
      v[i] = projection * p;
      const double d = la::dot(p, faceNormal_[face]);
      if (!std::isfinite(d)) return true;
      max = std::max(max, d);
      min = std::min(min, d);
    }
    if (max - min > 2 * tolerance_) return false;

    const int ccw = CCW(v[0], v[1], v[2], epsilon_ * 2);
    return ccw >= 0;
  });
}

/**
 * Returns the number of triangles that are colinear within tolerance_.
 */
int Manifold::Impl::NumDegenerateTris() const {
  if (halfedge_.size() == 0 || faceNormal_.size() != NumTri()) return true;
  return count_if(countAt(0_uz), countAt(NumTri()), [this](size_t face) {
    if (halfedge_.Pair(3 * face) < 0) return true;

    const mat2x3 projection = GetAxisAlignedProjection(faceNormal_[face]);
    vec2 v[3];
    for (int i : {0, 1, 2})
      v[i] = projection * vertPos_[halfedge_.Start(3 * face + i)];

    const int ccw = CCW(v[0], v[1], v[2], tolerance_ / 2);
    return ccw == 0;
  });
}

/**
 * Returns true if the Manifold is genus 0 and contains no concave edges.
 */
bool Manifold::Impl::IsConvex() const {
  // Convex shape must have genus of 0
  int chi = NumVert() - NumEdge() + NumTri();
  int genus = 1 - chi / 2;
  if (genus != 0) return false;

  // Iterate across all edges; return false if any edges are concave
  const size_t nbEdges = halfedge_.size();
  return all_of(countAt(0_uz), countAt(nbEdges), [this](size_t idx) {
    if (!halfedge_.IsForward(idx)) return true;

    const vec3 normal0 = faceNormal_[idx / 3];
    const vec3 normal1 = faceNormal_[halfedge_.Pair(idx) / 3];

    if (linalg::all(linalg::equal(normal0, normal1))) return true;

    const vec3 edgeVec =
        vertPos_[halfedge_.End(idx)] - vertPos_[halfedge_.Start(idx)];
    const bool convex =
        linalg::dot(edgeVec, linalg::cross(normal0, normal1)) > 0;
    return convex;
  });
}

double Manifold::Impl::GetProperty(Property prop) const {
  ZoneScoped;
  if (IsEmpty()) return 0;

  auto Volume = [this](size_t tri) {
    const vec3 v = vertPos_[halfedge_.Start(3 * tri)];
    vec3 crossP = la::cross(vertPos_[halfedge_.Start(3 * tri + 1)] - v,
                            vertPos_[halfedge_.Start(3 * tri + 2)] - v);
    return la::dot(crossP, v) / 6.0;
  };

  auto Area = [this](size_t tri) {
    const vec3 v = vertPos_[halfedge_.Start(3 * tri)];
    return la::length(la::cross(vertPos_[halfedge_.Start(3 * tri + 1)] - v,
                                vertPos_[halfedge_.Start(3 * tri + 2)] - v)) /
           2.0;
  };

  // Kahan summation
  double value = 0;
  double valueCompensation = 0;
  for (size_t i = 0; i < NumTri(); ++i) {
    const double value1 = prop == Property::SurfaceArea ? Area(i) : Volume(i);
    const double t = value + value1;
    valueCompensation += (value - t) + value1;
    value = t;
  }
  value += valueCompensation;
  return value;
}

void Manifold::Impl::CalculateCurvature(int gaussianIdx, int meanIdx) {
  ZoneScoped;
  if (IsEmpty()) return;
  if (gaussianIdx < 0 && meanIdx < 0) return;
  Vec<double> vertMeanCurvature(NumVert(), 0);
  Vec<double> vertGaussianCurvature(NumVert(), kTwoPi);
  Vec<double> vertArea(NumVert(), 0);
  Vec<double> degree(NumVert(), 0);
  auto policy = autoPolicy(NumTri(), 1e4);
  for_each(policy, countAt(0_uz), countAt(NumTri()),
           CurvatureAngles{vertMeanCurvature, vertGaussianCurvature, vertArea,
                           degree, halfedge_, vertPos_, faceNormal_});
  for_each_n(policy, countAt(0), NumVert(),
             [&vertMeanCurvature, &vertGaussianCurvature, &vertArea,
              &degree](const int vert) {
               const double factor = degree[vert] / (6 * vertArea[vert]);
               vertMeanCurvature[vert] *= factor;
               vertGaussianCurvature[vert] *= factor;
             });

  const int oldNumProp = NumProp();
  const int numProp = std::max(oldNumProp, std::max(gaussianIdx, meanIdx) + 1);
  const Vec<double> oldProperties = properties_;
  properties_ = Vec<double>(numProp * NumPropVert(), 0);
  numProp_ = numProp;

  Vec<uint8_t> counters(NumPropVert(), 0);
  for_each_n(policy, countAt(0_uz), NumTri(), [&](const size_t tri) {
    for (const int i : {0, 1, 2}) {
      const int edge = 3 * tri + i;
      const int vert = halfedge_.Start(edge);
      const int propVert = halfedge_.Prop(edge);

      auto old = std::atomic_exchange(
          reinterpret_cast<std::atomic<uint8_t>*>(&counters[propVert]),
          static_cast<uint8_t>(1));
      if (old == 1) continue;

      for (int p = 0; p < oldNumProp; ++p) {
        properties_[numProp * propVert + p] =
            oldProperties[oldNumProp * propVert + p];
      }

      if (gaussianIdx >= 0) {
        properties_[numProp * propVert + gaussianIdx] =
            vertGaussianCurvature[vert];
      }
      if (meanIdx >= 0) {
        properties_[numProp * propVert + meanIdx] = vertMeanCurvature[vert];
      }
    }
  });
}

/**
 * Calculates the bounding box of the entire manifold, which is stored
 * internally to short-cut Boolean operations. Ignores NaNs.
 */
void Manifold::Impl::CalculateBBox() {
  bBox_.min =
      reduce(vertPos_.begin(), vertPos_.end(),
             vec3(std::numeric_limits<double>::infinity()), [](auto a, auto b) {
               if (std::isnan(a.x)) return b;
               if (std::isnan(b.x)) return a;
               return la::min(a, b);
             });
  bBox_.max = reduce(vertPos_.begin(), vertPos_.end(),
                     vec3(-std::numeric_limits<double>::infinity()),
                     [](auto a, auto b) {
                       if (std::isnan(a.x)) return b;
                       if (std::isnan(b.x)) return a;
                       return la::max(a, b);
                     });

  if (!bBox_.IsFinite()) {
    // Decimated out of existence - early out.
    MakeEmpty(Error::NoError);
  }
}

/**
 * Determines if all verts are finite. Checking just the bounding box dimensions
 * is insufficient as it ignores NaNs.
 */
bool Manifold::Impl::IsFinite() const {
  return transform_reduce(
      vertPos_.begin(), vertPos_.end(), true,
      [](bool a, bool b) { return a && b; },
      [](auto v) { return la::all(la::isfinite(v)); });
}

/**
 * Checks that the input triVerts array has all indices inside bounds of the
 * vertPos_ array.
 */
bool Manifold::Impl::IsIndexInBounds(VecView<const ivec3> triVerts) const {
  ivec2 minmax = transform_reduce(
      triVerts.begin(), triVerts.end(),
      ivec2(std::numeric_limits<int>::max(), std::numeric_limits<int>::min()),
      [](auto a, auto b) {
        a[0] = std::min(a[0], b[0]);
        a[1] = std::max(a[1], b[1]);
        return a;
      },
      [](auto tri) {
        return ivec2(std::min(tri[0], std::min(tri[1], tri[2])),
                     std::max(tri[0], std::max(tri[1], tri[2])));
      });

  return minmax[0] >= 0 && minmax[1] < static_cast<int>(NumVert());
}

struct MinDistanceRecorder {
  using Local = double;
  const Manifold::Impl &self, &other;
#if MANIFOLD_PAR == 1
  tbb::combinable<double> store = tbb::combinable<double>(
      []() { return std::numeric_limits<double>::infinity(); });
  Local& local() { return store.local(); }
  double get() {
    double result = std::numeric_limits<double>::infinity();
    store.combine_each([&](double& val) { result = std::min(result, val); });
    return result;
  }
#else
  double result = std::numeric_limits<double>::infinity();
  Local& local() { return result; }
  double get() { return result; }
#endif

  void record(int triOther, int tri, double& minDistance) {
    std::array<vec3, 3> p;
    std::array<vec3, 3> q;

    for (const int j : {0, 1, 2}) {
      p[j] = self.vertPos_[self.halfedge_.Start(3 * tri + j)];
      q[j] = other.vertPos_[other.halfedge_.Start(3 * triOther + j)];
    }
    minDistance = std::min(minDistance, DistanceTriangleTriangleSquared(p, q));
  }
};

/*
 * Returns the minimum gap between two manifolds. Returns a double between
 * 0 and searchLength.
 */
double Manifold::Impl::MinGap(const Manifold::Impl& other,
                              double searchLength) const {
  ZoneScoped;
  Vec<Box> faceBoxOther;
  Vec<uint32_t> faceMortonOther;

  other.GetFaceBoxMorton(faceBoxOther, faceMortonOther);

  transform(faceBoxOther.begin(), faceBoxOther.end(), faceBoxOther.begin(),
            [searchLength](const Box& box) {
              return Box(box.min - vec3(searchLength),
                         box.max + vec3(searchLength));
            });

  MinDistanceRecorder recorder{*this, other};
  collider_.Collisions<false>(recorder, faceBoxOther.cview(), false);
  double minDistanceSquared =
      std::min(recorder.get(), searchLength * searchLength);
  return sqrt(minDistanceSquared);
};

}  // namespace manifold
