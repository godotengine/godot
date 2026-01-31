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

#include <unordered_set>

#include "impl.h"
#include "manifold/common.h"
#include "manifold/polygon.h"
#include "parallel.h"
#include "shared.h"

#if (MANIFOLD_PAR == 1) && __has_include(<tbb/concurrent_map.h>)
#include <tbb/tbb.h>
#define TBB_PREVIEW_CONCURRENT_ORDERED_CONTAINERS 1
#include <tbb/concurrent_map.h>
#endif

namespace {
using namespace manifold;

/**
 * Returns an assembled set of vertex index loops of the input list of
 * Halfedges, where each vert must be referenced the same number of times as a
 * startVert and endVert. If startHalfedgeIdx is given, instead of putting
 * vertex indices into the returned polygons structure, it will use the halfedge
 * indices instead.
 */
std::vector<std::vector<int>> AssembleHalfedges(VecView<Halfedge>::IterC start,
                                                VecView<Halfedge>::IterC end,
                                                const int startHalfedgeIdx) {
  std::multimap<int, int> vert_edge;
  for (auto edge = start; edge != end; ++edge) {
    vert_edge.emplace(
        std::make_pair(edge->startVert, static_cast<int>(edge - start)));
  }

  std::vector<std::vector<int>> polys;
  int startEdge = 0;
  int thisEdge = startEdge;
  while (1) {
    if (thisEdge == startEdge) {
      if (vert_edge.empty()) break;
      startEdge = vert_edge.begin()->second;
      thisEdge = startEdge;
      polys.push_back({});
    }
    polys.back().push_back(startHalfedgeIdx + thisEdge);
    const auto result = vert_edge.find((start + thisEdge)->endVert);
    DEBUG_ASSERT(result != vert_edge.end(), topologyErr, "non-manifold edge");
    thisEdge = result->second;
    vert_edge.erase(result);
  }
  return polys;
}

/**
 * Add the vertex position projection to the indexed polygons.
 */
PolygonsIdx ProjectPolygons(const std::vector<std::vector<int>>& polys,
                            const Vec<Halfedge>& halfedge,
                            const Vec<vec3>& vertPos, mat2x3 projection) {
  PolygonsIdx polygons;
  for (const auto& poly : polys) {
    polygons.push_back({});
    for (const auto& edge : poly) {
      polygons.back().push_back(
          {projection * vertPos[halfedge[edge].startVert], edge});
    }  // for vert
  }  // for poly
  return polygons;
}
}  // namespace

namespace manifold {

using GeneralTriangulation = std::function<std::vector<ivec3>(int)>;
using AddTriangle = std::function<void(int, ivec3, vec3, TriRef)>;

/**
 * Triangulates the faces. In this case, the halfedge_ vector is not yet a set
 * of triangles as required by this data structure, but is instead a set of
 * general faces with the input faceEdge vector having length of the number of
 * faces + 1. The values are indicies into the halfedge_ vector for the first
 * edge of each face, with the final value being the length of the halfedge_
 * vector itself. Upon return, halfedge_ has been lengthened and properly
 * represents the mesh as a set of triangles as usual. In this process the
 * faceNormal_ values are retained, repeated as necessary.
 */
void Manifold::Impl::Face2Tri(const Vec<int>& faceEdge,
                              const Vec<TriRef>& halfedgeRef,
                              bool allowConvex) {
  ZoneScoped;
  Vec<ivec3> triVerts;
  Vec<vec3> triNormal;
  Vec<ivec3> triProp;
  Vec<TriRef>& triRef = meshRelation_.triRef;
  triRef.clear();
  auto processFace = [&](GeneralTriangulation general, AddTriangle addTri,
                         int face) {
    const int firstEdge = faceEdge[face];
    const int lastEdge = faceEdge[face + 1];
    const int numEdge = lastEdge - firstEdge;
    if (numEdge == 0) return;
    DEBUG_ASSERT(numEdge >= 3, topologyErr, "face has less than three edges.");
    const vec3 normal = faceNormal_[face];

    if (numEdge == 3) {  // Single triangle
      ivec3 triEdge(firstEdge, firstEdge + 1, firstEdge + 2);
      ivec3 tri(halfedge_[firstEdge].startVert,
                halfedge_[firstEdge + 1].startVert,
                halfedge_[firstEdge + 2].startVert);
      ivec3 ends(halfedge_[firstEdge].endVert, halfedge_[firstEdge + 1].endVert,
                 halfedge_[firstEdge + 2].endVert);
      if (ends[0] == tri[2]) {
        std::swap(triEdge[1], triEdge[2]);
        std::swap(tri[1], tri[2]);
        std::swap(ends[1], ends[2]);
      }
      DEBUG_ASSERT(ends[0] == tri[1] && ends[1] == tri[2] && ends[2] == tri[0],
                   topologyErr, "These 3 edges do not form a triangle!");

      addTri(face, triEdge, normal, halfedgeRef[firstEdge]);
    } else if (numEdge == 4) {  // Pair of triangles
      const mat2x3 projection = GetAxisAlignedProjection(normal);
      auto triCCW = [&projection, this](const ivec3 tri) {
        return CCW(projection * this->vertPos_[halfedge_[tri[0]].startVert],
                   projection * this->vertPos_[halfedge_[tri[1]].startVert],
                   projection * this->vertPos_[halfedge_[tri[2]].startVert],
                   epsilon_) >= 0;
      };

      std::vector<int> quad = AssembleHalfedges(
          halfedge_.cbegin() + faceEdge[face],
          halfedge_.cbegin() + faceEdge[face + 1], faceEdge[face])[0];

      const la::mat<int, 3, 2> tris[2] = {
          {{quad[0], quad[1], quad[2]}, {quad[0], quad[2], quad[3]}},
          {{quad[1], quad[2], quad[3]}, {quad[0], quad[1], quad[3]}}};

      int choice = 0;

      if (!(triCCW(tris[0][0]) && triCCW(tris[0][1]))) {
        choice = 1;
      } else if (triCCW(tris[1][0]) && triCCW(tris[1][1])) {
        vec3 diag0 = vertPos_[halfedge_[quad[0]].startVert] -
                     vertPos_[halfedge_[quad[2]].startVert];
        vec3 diag1 = vertPos_[halfedge_[quad[1]].startVert] -
                     vertPos_[halfedge_[quad[3]].startVert];
        if (la::length2(diag0) > la::length2(diag1)) {
          choice = 1;
        }
      }

      for (const auto& tri : tris[choice]) {
        addTri(face, tri, normal, halfedgeRef[firstEdge]);
      }
    } else {  // General triangulation
      for (const auto& tri : general(face)) {
        addTri(face, tri, normal, halfedgeRef[firstEdge]);
      }
    }
  };
  auto generalTriangulation = [&](int face) {
    const vec3 normal = faceNormal_[face];
    const mat2x3 projection = GetAxisAlignedProjection(normal);
    const PolygonsIdx polys = ProjectPolygons(
        AssembleHalfedges(halfedge_.cbegin() + faceEdge[face],
                          halfedge_.cbegin() + faceEdge[face + 1],
                          faceEdge[face]),
        halfedge_, vertPos_, projection);
    return TriangulateIdx(polys, epsilon_, allowConvex);
  };
#if (MANIFOLD_PAR == 1) && __has_include(<tbb/tbb.h>)
  tbb::task_group group;
  // map from face to triangle
  tbb::concurrent_unordered_map<int, std::vector<ivec3>> results;
  Vec<size_t> triCount(faceEdge.size());
  triCount.back() = 0;
  // precompute number of triangles per face, and launch async tasks to
  // triangulate complex faces
  for_each(autoPolicy(faceEdge.size(), 1e5), countAt(0_uz),
           countAt(faceEdge.size() - 1), [&](size_t face) {
             triCount[face] = faceEdge[face + 1] - faceEdge[face] - 2;
             DEBUG_ASSERT(triCount[face] >= 1, topologyErr,
                          "face has less than three edges.");
             if (triCount[face] > 2)
               group.run([&, face] {
                 std::vector<ivec3> newTris = generalTriangulation(face);
                 triCount[face] = newTris.size();
                 results[face] = std::move(newTris);
               });
           });
  group.wait();
  // prefix sum computation (assign unique index to each face) and preallocation
  exclusive_scan(triCount.begin(), triCount.end(), triCount.begin(), 0_uz);
  triVerts.resize(triCount.back());
  triProp.resize(triCount.back());
  triNormal.resize(triCount.back());
  triRef.resize(triCount.back());

  auto processFace2 = std::bind(
      processFace, [&](size_t face) { return std::move(results[face]); },
      [&](size_t face, ivec3 tri, vec3 normal, TriRef r) {
        for (const int i : {0, 1, 2}) {
          triVerts[triCount[face]][i] = halfedge_[tri[i]].startVert;
          triProp[triCount[face]][i] = halfedge_[tri[i]].propVert;
        }
        triNormal[triCount[face]] = normal;
        triRef[triCount[face]] = r;
        triCount[face]++;
      },
      std::placeholders::_1);
  // set triangles in parallel
  for_each(autoPolicy(faceEdge.size(), 1e4), countAt(0_uz),
           countAt(faceEdge.size() - 1), processFace2);
#else
  triVerts.reserve(faceEdge.size());
  triNormal.reserve(faceEdge.size());
  triRef.reserve(faceEdge.size());
  auto processFace2 = std::bind(
      processFace, generalTriangulation,
      [&](size_t, ivec3 tri, vec3 normal, TriRef r) {
        ivec3 verts;
        ivec3 props;
        for (const int i : {0, 1, 2}) {
          verts[i] = halfedge_[tri[i]].startVert;
          props[i] = halfedge_[tri[i]].propVert;
        }
        triVerts.push_back(verts);
        triProp.push_back(props);
        triNormal.push_back(normal);
        triRef.push_back(r);
      },
      std::placeholders::_1);
  for (size_t face = 0; face < faceEdge.size() - 1; ++face) {
    processFace2(face);
  }
#endif

  faceNormal_ = std::move(triNormal);
  CreateHalfedges(triProp, triVerts);
}

Polygons Manifold::Impl::Slice(double height) const {
  Box plane = bBox_;
  plane.min.z = plane.max.z = height;
  Vec<Box> query;
  query.push_back(plane);

  std::unordered_set<int> tris;
  auto recordCollision = [&](int, int tri) {
    double min = std::numeric_limits<double>::infinity();
    double max = -std::numeric_limits<double>::infinity();
    for (const int j : {0, 1, 2}) {
      const double z = vertPos_[halfedge_[3 * tri + j].startVert].z;
      min = std::min(min, z);
      max = std::max(max, z);
    }

    if (min <= height && max > height) {
      tris.insert(tri);
    }
  };

  auto recorder = MakeSimpleRecorder(recordCollision);
  collider_.Collisions<false>(query.cview(), recorder, false);

  Polygons polys;
  while (!tris.empty()) {
    const int startTri = *tris.begin();
    SimplePolygon poly;

    int k = 0;
    for (const int j : {0, 1, 2}) {
      if (vertPos_[halfedge_[3 * startTri + j].startVert].z > height &&
          vertPos_[halfedge_[3 * startTri + Next3(j)].startVert].z <= height) {
        k = Next3(j);
        break;
      }
    }

    int tri = startTri;
    do {
      tris.erase(tris.find(tri));
      if (vertPos_[halfedge_[3 * tri + k].endVert].z <= height) {
        k = Next3(k);
      }

      Halfedge up = halfedge_[3 * tri + k];
      const vec3 below = vertPos_[up.startVert];
      const vec3 above = vertPos_[up.endVert];
      const double a = (height - below.z) / (above.z - below.z);
      poly.push_back(vec2(la::lerp(below, above, a)));

      const int pair = up.pairedHalfedge;
      tri = pair / 3;
      k = Next3(pair % 3);
    } while (tri != startTri);

    polys.push_back(poly);
  }

  return polys;
}

Polygons Manifold::Impl::Project() const {
  const mat2x3 projection = GetAxisAlignedProjection({0, 0, 1});
  Vec<Halfedge> cusps(NumEdge());
  cusps.resize(
      copy_if(
          halfedge_.cbegin(), halfedge_.cend(), cusps.begin(),
          [&](Halfedge edge) {
            return faceNormal_[halfedge_[edge.pairedHalfedge].pairedHalfedge /
                               3]
                           .z >= 0 &&
                   faceNormal_[edge.pairedHalfedge / 3].z < 0;
          }) -
      cusps.begin());

  PolygonsIdx polysIndexed =
      ProjectPolygons(AssembleHalfedges(cusps.cbegin(), cusps.cend(), 0), cusps,
                      vertPos_, projection);

  Polygons polys;
  for (const auto& poly : polysIndexed) {
    SimplePolygon simple;
    for (const PolyVert& polyVert : poly) {
      simple.push_back(polyVert.pos);
    }
    polys.push_back(simple);
  }

  return polys;
}
}  // namespace manifold
