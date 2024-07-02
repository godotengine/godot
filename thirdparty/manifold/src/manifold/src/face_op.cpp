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

#if MANIFOLD_PAR == 'T' && __has_include(<tbb/concurrent_map.h>)
#include <tbb/tbb.h>
#define TBB_PREVIEW_CONCURRENT_ORDERED_CONTAINERS 1
#include <tbb/concurrent_map.h>
#endif
#include <unordered_set>

#include "impl.h"
#include "polygon.h"

namespace manifold {

using GeneralTriangulation = std::function<std::vector<glm::ivec3>(int)>;
using AddTriangle = std::function<void(int, glm::ivec3, glm::vec3, TriRef)>;

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
                              const Vec<TriRef>& halfedgeRef) {
  ZoneScoped;
  Vec<glm::ivec3> triVerts;
  Vec<glm::vec3> triNormal;
  Vec<TriRef>& triRef = meshRelation_.triRef;
  triRef.resize(0);
  auto processFace = [&](GeneralTriangulation general, AddTriangle addTri,
                         int face) {
    const int firstEdge = faceEdge[face];
    const int lastEdge = faceEdge[face + 1];
    const int numEdge = lastEdge - firstEdge;
    ASSERT(numEdge >= 3, topologyErr, "face has less than three edges.");
    const glm::vec3 normal = faceNormal_[face];

    if (numEdge == 3) {  // Single triangle
      int mapping[3] = {halfedge_[firstEdge].startVert,
                        halfedge_[firstEdge + 1].startVert,
                        halfedge_[firstEdge + 2].startVert};
      glm::ivec3 tri(halfedge_[firstEdge].startVert,
                     halfedge_[firstEdge + 1].startVert,
                     halfedge_[firstEdge + 2].startVert);
      glm::ivec3 ends(halfedge_[firstEdge].endVert,
                      halfedge_[firstEdge + 1].endVert,
                      halfedge_[firstEdge + 2].endVert);
      if (ends[0] == tri[2]) {
        std::swap(tri[1], tri[2]);
        std::swap(ends[1], ends[2]);
      }
      ASSERT(ends[0] == tri[1] && ends[1] == tri[2] && ends[2] == tri[0],
             topologyErr, "These 3 edges do not form a triangle!");

      addTri(face, tri, normal, halfedgeRef[firstEdge]);
    } else if (numEdge == 4) {  // Pair of triangles
      int mapping[4] = {halfedge_[firstEdge].startVert,
                        halfedge_[firstEdge + 1].startVert,
                        halfedge_[firstEdge + 2].startVert,
                        halfedge_[firstEdge + 3].startVert};
      const glm::mat3x2 projection = GetAxisAlignedProjection(normal);
      auto triCCW = [&projection, this](const glm::ivec3 tri) {
        return CCW(projection * this->vertPos_[tri[0]],
                   projection * this->vertPos_[tri[1]],
                   projection * this->vertPos_[tri[2]], precision_) >= 0;
      };

      glm::ivec3 tri0(halfedge_[firstEdge].startVert,
                      halfedge_[firstEdge].endVert, -1);
      glm::ivec3 tri1(-1, -1, tri0[0]);
      for (const int i : {1, 2, 3}) {
        if (halfedge_[firstEdge + i].startVert == tri0[1]) {
          tri0[2] = halfedge_[firstEdge + i].endVert;
          tri1[0] = tri0[2];
        }
        if (halfedge_[firstEdge + i].endVert == tri0[0]) {
          tri1[1] = halfedge_[firstEdge + i].startVert;
        }
      }
      ASSERT(glm::all(glm::greaterThanEqual(tri0, glm::ivec3(0))) &&
                 glm::all(glm::greaterThanEqual(tri1, glm::ivec3(0))),
             topologyErr, "non-manifold quad!");
      bool firstValid = triCCW(tri0) && triCCW(tri1);
      tri0[2] = tri1[1];
      tri1[2] = tri0[1];
      bool secondValid = triCCW(tri0) && triCCW(tri1);

      if (!secondValid) {
        tri0[2] = tri1[0];
        tri1[2] = tri0[0];
      } else if (firstValid) {
        glm::vec3 firstCross = vertPos_[tri0[0]] - vertPos_[tri1[0]];
        glm::vec3 secondCross = vertPos_[tri0[1]] - vertPos_[tri1[1]];
        if (glm::dot(firstCross, firstCross) <
            glm::dot(secondCross, secondCross)) {
          tri0[2] = tri1[0];
          tri1[2] = tri0[0];
        }
      }

      for (const auto& tri : {tri0, tri1}) {
        addTri(face, tri, normal, halfedgeRef[firstEdge]);
      }
    } else {  // General triangulation
      for (const auto& tri : general(face)) {
        addTri(face, tri, normal, halfedgeRef[firstEdge]);
      }
    }
  };
  auto generalTriangulation = [&](int face) {
    const glm::vec3 normal = faceNormal_[face];
    const glm::mat3x2 projection = GetAxisAlignedProjection(normal);
    const PolygonsIdx polys =
        Face2Polygons(halfedge_.cbegin() + faceEdge[face],
                      halfedge_.cbegin() + faceEdge[face + 1], projection);
    return TriangulateIdx(polys, precision_);
  };
#if MANIFOLD_PAR == 'T' && __has_include(<tbb/tbb.h>)
  tbb::task_group group;
  // map from face to triangle
  tbb::concurrent_unordered_map<int, std::vector<glm::ivec3>> results;
  Vec<size_t> triCount(faceEdge.size());
  triCount.back() = 0;
  // precompute number of triangles per face, and launch async tasks to
  // triangulate complex faces
  for_each(autoPolicy(faceEdge.size()), countAt(0_z),
           countAt(faceEdge.size() - 1), [&](size_t face) {
             triCount[face] = faceEdge[face + 1] - faceEdge[face] - 2;
             ASSERT(triCount[face] >= 1, topologyErr,
                    "face has less than three edges.");
             if (triCount[face] > 2)
               group.run([&, face] {
                 std::vector<glm::ivec3> newTris = generalTriangulation(face);
                 triCount[face] = newTris.size();
                 results[face] = std::move(newTris);
               });
           });
  group.wait();
  // prefix sum computation (assign unique index to each face) and preallocation
  exclusive_scan(autoPolicy(triCount.size()), triCount.begin(), triCount.end(),
                 triCount.begin(), 0_z);
  if (triCount.back() >= std::numeric_limits<int>::max())
    throw std::out_of_range("too many triangles");
  triVerts.resize(triCount.back());
  triNormal.resize(triCount.back());
  triRef.resize(triCount.back());

  auto processFace2 = std::bind(
      processFace, [&](size_t face) { return std::move(results[face]); },
      [&](int face, glm::ivec3 tri, glm::vec3 normal, TriRef r) {
        triVerts[triCount[face]] = tri;
        triNormal[triCount[face]] = normal;
        triRef[triCount[face]] = r;
        triCount[face]++;
      },
      std::placeholders::_1);
  // set triangles in parallel
  for_each(autoPolicy(faceEdge.size()), countAt(0_z),
           countAt(faceEdge.size() - 1), processFace2);
#else
  triVerts.reserve(faceEdge.size());
  triNormal.reserve(faceEdge.size());
  triRef.reserve(faceEdge.size());
  auto processFace2 = std::bind(
      processFace, generalTriangulation,
      [&](int _face, glm::ivec3 tri, glm::vec3 normal, TriRef r) {
        triVerts.push_back(tri);
        triNormal.push_back(normal);
        triRef.push_back(r);
      },
      std::placeholders::_1);
  for (int face = 0; face < faceEdge.size() - 1; ++face) {
    processFace2(face);
  }
#endif

  faceNormal_ = std::move(triNormal);
  CreateHalfedges(triVerts);
}

/**
 * Returns a set of 2D polygons formed by the input projection of the vertices
 * of the list of Halfedges, which must be an even-manifold, meaning each vert
 * must be referenced the same number of times as a startVert and endVert.
 */
PolygonsIdx Manifold::Impl::Face2Polygons(VecView<Halfedge>::IterC start,
                                          VecView<Halfedge>::IterC end,
                                          glm::mat3x2 projection) const {
  std::multimap<int, int> vert_edge;
  for (auto edge = start; edge != end; ++edge) {
    vert_edge.emplace(std::make_pair(edge->startVert, edge - start));
  }

  PolygonsIdx polys;
  int startEdge = 0;
  int thisEdge = startEdge;
  while (1) {
    if (thisEdge == startEdge) {
      if (vert_edge.empty()) break;
      startEdge = vert_edge.begin()->second;
      thisEdge = startEdge;
      polys.push_back({});
    }
    int vert = (start + thisEdge)->startVert;
    polys.back().push_back({projection * vertPos_[vert], vert});
    const auto result = vert_edge.find((start + thisEdge)->endVert);
    ASSERT(result != vert_edge.end(), topologyErr, "non-manifold edge");
    thisEdge = result->second;
    vert_edge.erase(result);
  }
  return polys;
}

CrossSection Manifold::Impl::Slice(float height) const {
  Box plane = bBox_;
  plane.min.z = plane.max.z = height;
  Vec<Box> query;
  query.push_back(plane);
  const SparseIndices collisions =
      collider_.Collisions<false, false>(query.cview());

  std::unordered_set<int> tris;
  for (int i = 0; i < collisions.size(); ++i) {
    const int tri = collisions.Get(i, 1);
    float min = std::numeric_limits<float>::infinity();
    float max = -std::numeric_limits<float>::infinity();
    for (const int j : {0, 1, 2}) {
      const float z = vertPos_[halfedge_[3 * tri + j].startVert].z;
      min = glm::min(min, z);
      max = glm::max(max, z);
    }

    if (min <= height && max > height) {
      tris.insert(tri);
    }
  }

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
      const glm::vec3 below = vertPos_[up.startVert];
      const glm::vec3 above = vertPos_[up.endVert];
      const float a = (height - below.z) / (above.z - below.z);
      poly.push_back(glm::vec2(glm::mix(below, above, a)));

      const int pair = up.pairedHalfedge;
      tri = pair / 3;
      k = Next3(pair % 3);
    } while (tri != startTri);

    polys.push_back(poly);
  }

  return CrossSection(polys);
}

CrossSection Manifold::Impl::Project() const {
  const glm::mat3x2 projection = GetAxisAlignedProjection({0, 0, 1});
  auto policy = autoPolicy(halfedge_.size());

  Vec<Halfedge> cusps(NumEdge());
  cusps.resize(copy_if<decltype(cusps.begin())>(
                   policy, halfedge_.cbegin(), halfedge_.cend(), cusps.begin(),
                   [&](Halfedge edge) {
                     return faceNormal_[edge.face].z >= 0 &&
                            faceNormal_[halfedge_[edge.pairedHalfedge].face].z <
                                0;
                   }) -
               cusps.begin());

  PolygonsIdx polysIndexed =
      Face2Polygons(cusps.cbegin(), cusps.cend(), projection);

  Polygons polys;
  for (const auto& poly : polysIndexed) {
    SimplePolygon simple;
    for (const PolyVert& polyVert : poly) {
      simple.push_back(polyVert.pos);
    }
    polys.push_back(simple);
  }

  return CrossSection(polys).Simplify(precision_);
}
}  // namespace manifold
