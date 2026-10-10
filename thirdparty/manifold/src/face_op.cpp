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

#include <array>
#include <unordered_set>

#include "impl.h"
#include "manifold/common.h"
#include "manifold/polygon.h"
#include "parallel.h"
#include "polygon_internal.h"
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
                            const VecView<const Halfedge>& halfedge,
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

void WriteLocalTriangles(Halfedges& output, VecView<int> contour2Tri,
                         const VecView<const Halfedge>& faceHalfedge,
                         size_t firstTri, const ivec3* triangles, int numTri) {
  DEBUG_ASSERT(numTri <= 2, logicErr,
               "local face path only handles tris/quads");
  std::array<ivec3, 6> localEdges;
  const int firstOut = 3 * firstTri;
  int numEdge = 0;
  for (int tri = 0; tri < numTri; ++tri) {
    for (const int i : {0, 1, 2}) {
      const int out = firstOut + numEdge;
      const int start = triangles[tri][i];
      const int end = triangles[tri][Next3(i)];
      localEdges[numEdge] = {start, end, out};
      output.SetStart(out, faceHalfedge[start].startVert);
      output.SetProp(out, faceHalfedge[start].propVert);
      output.SetPair(out, -1);
      ++numEdge;
    }
  }

  for (int i = 0; i < numEdge; ++i) {
    const ivec3 edge = localEdges[i];
    int pair = -1;
    for (int j = 0; j < numEdge; ++j) {
      if (localEdges[j][0] == edge[1] && localEdges[j][1] == edge[0]) {
        pair = localEdges[j][2];
        break;
      }
    }
    if (pair >= 0) {
      output.SetPair(edge[2], pair);
    } else {
      contour2Tri[edge[0]] = edge[2];
    }
  }
}

void WriteGeneralTriangulation(Halfedges& output, VecView<int> contour2Tri,
                               const VecView<const Halfedge>& faceHalfedge,
                               size_t firstTri,
                               const HalfedgeTriangulation& triangulation,
                               ExecutionContext::Impl* ctx) {
  const int firstOut = 3 * firstTri;
  const size_t numTriHalfedge = 3 * triangulation.NumTri();
  for_each_n(
      autoPolicy(numTriHalfedge, 1e5), countAt(0_uz), numTriHalfedge, ctx,
      [&](size_t local) {
        const int out = firstOut + local;
        const Halfedge& edge =
            triangulation.halfedges[triangulation.contourEnd + local];
        output.SetStart(out, faceHalfedge[edge.startVert].startVert);
        output.SetProp(out, faceHalfedge[edge.startVert].propVert);
        if (edge.pairedHalfedge >= static_cast<int>(triangulation.contourEnd)) {
          output.SetPair(out, firstOut + edge.pairedHalfedge -
                                  static_cast<int>(triangulation.contourEnd));
        } else {
          output.SetPair(out, -1);
        }
      });

  for_each_n(autoPolicy(triangulation.contourEnd, 1e5), countAt(0_uz),
             triangulation.contourEnd, ctx, [&](size_t contour) {
               const Halfedge& edge = triangulation.halfedges[contour];
               if (edge.pairedHalfedge < 0) return;
               DEBUG_ASSERT(edge.pairedHalfedge >=
                                static_cast<int>(triangulation.contourEnd),
                            topologyErr, "contour paired to another contour");
               const int boundary = edge.endVert;
               DEBUG_ASSERT(boundary >= 0 &&
                                boundary < static_cast<int>(contour2Tri.size()),
                            topologyErr, "contour edge index out of bounds");
               contour2Tri[boundary] =
                   firstOut + edge.pairedHalfedge -
                   static_cast<int>(triangulation.contourEnd);
             });
}

void WriteTriRefs(VecView<vec3> triNormal, VecView<TriRef> triRef,
                  size_t firstTri, size_t numTri, vec3 normal, TriRef ref,
                  ExecutionContext::Impl* ctx) {
  for_each_n(autoPolicy(numTri, 1e5), countAt(0_uz), numTri, ctx,
             [&](size_t tri) {
               triNormal[firstTri + tri] = normal;
               triRef[firstTri + tri] = ref;
             });
}
}  // namespace

namespace manifold {

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
                              const VecView<const Halfedge>& faceHalfedge,
                              const Vec<TriRef>& halfedgeRef, bool allowConvex,
                              ExecutionContext::Impl* ctx) {
  ZoneScoped;
  if (IsCancelled(ctx)) return;
  Vec<vec3> triNormal;
  Vec<TriRef>& triRef = meshRelation_.triRef;
  triRef.clear();
  Vec<int> contour2Tri(faceHalfedge.size(), -1);

  auto generalTriangulation = [&](int face) {
    const vec3 normal = faceNormal_[face];
    const mat2x3 projection = GetAxisAlignedProjection(normal);
    const PolygonsIdx polys = ProjectPolygons(
        AssembleHalfedges(faceHalfedge.cbegin() + faceEdge[face],
                          faceHalfedge.cbegin() + faceEdge[face + 1],
                          faceEdge[face]),
        faceHalfedge, vertPos_, projection);
    return TriangulateIdxHalfedges(polys, epsilon_, allowConvex);
  };

  auto outputFace = [&](int face, size_t firstTri,
                        const HalfedgeTriangulation* general) {
    const int firstEdge = faceEdge[face];
    const int lastEdge = faceEdge[face + 1];
    const int numEdge = lastEdge - firstEdge;
    if (numEdge == 0) return;
    DEBUG_ASSERT(numEdge >= 3, topologyErr, "face has less than three edges.");
    const vec3 normal = faceNormal_[face];
    size_t numTri = numEdge - 2;

    if (numEdge == 3) {  // Single triangle
      ivec3 triEdge(firstEdge, firstEdge + 1, firstEdge + 2);
      ivec3 tri(faceHalfedge[firstEdge].startVert,
                faceHalfedge[firstEdge + 1].startVert,
                faceHalfedge[firstEdge + 2].startVert);
      ivec3 ends(faceHalfedge[firstEdge].endVert,
                 faceHalfedge[firstEdge + 1].endVert,
                 faceHalfedge[firstEdge + 2].endVert);
      if (ends[0] == tri[2]) {
        std::swap(triEdge[1], triEdge[2]);
        std::swap(tri[1], tri[2]);
        std::swap(ends[1], ends[2]);
      }
      DEBUG_ASSERT(ends[0] == tri[1] && ends[1] == tri[2] && ends[2] == tri[0],
                   topologyErr, "These 3 edges do not form a triangle!");

      WriteLocalTriangles(halfedge_, contour2Tri, faceHalfedge, firstTri,
                          &triEdge, 1);
    } else if (numEdge == 4) {  // Pair of triangles
      const mat2x3 projection = GetAxisAlignedProjection(normal);
      auto triCCW = [&projection, &faceHalfedge, this](const ivec3 tri) {
        return CCW(projection * this->vertPos_[faceHalfedge[tri[0]].startVert],
                   projection * this->vertPos_[faceHalfedge[tri[1]].startVert],
                   projection * this->vertPos_[faceHalfedge[tri[2]].startVert],
                   epsilon_) >= 0;
      };

      std::vector<int> quad = AssembleHalfedges(
          faceHalfedge.cbegin() + faceEdge[face],
          faceHalfedge.cbegin() + faceEdge[face + 1], faceEdge[face])[0];

      const la::mat<int, 3, 2> tris[2] = {
          {{quad[0], quad[1], quad[2]}, {quad[0], quad[2], quad[3]}},
          {{quad[1], quad[2], quad[3]}, {quad[0], quad[1], quad[3]}}};

      int choice = 0;

      if (!(triCCW(tris[0][0]) && triCCW(tris[0][1]))) {
        choice = 1;
      } else if (triCCW(tris[1][0]) && triCCW(tris[1][1])) {
        vec3 diag0 = vertPos_[faceHalfedge[quad[0]].startVert] -
                     vertPos_[faceHalfedge[quad[2]].startVert];
        vec3 diag1 = vertPos_[faceHalfedge[quad[1]].startVert] -
                     vertPos_[faceHalfedge[quad[3]].startVert];
        if (la::length2(diag0) > la::length2(diag1)) {
          choice = 1;
        }
      }

      WriteLocalTriangles(halfedge_, contour2Tri, faceHalfedge, firstTri,
                          &tris[choice][0], 2);
    } else {  // General triangulation
      DEBUG_ASSERT(general != nullptr, logicErr,
                   "general face missing triangulation result");
      numTri = general->NumTri();
      WriteGeneralTriangulation(halfedge_, contour2Tri, faceHalfedge, firstTri,
                                *general, ctx);
    }

    WriteTriRefs(triNormal, triRef, firstTri, numTri, normal,
                 halfedgeRef[firstEdge], ctx);
  };

  Vec<size_t> triOffset(faceEdge.size());
  triOffset.back() = 0;
#if (MANIFOLD_PAR == 1) && __has_include(<tbb/tbb.h>)
  tbb::task_group group;
  tbb::concurrent_unordered_map<int, HalfedgeTriangulation> results;
  // precompute number of triangles per face, and launch async tasks to
  // triangulate complex faces
  for_each(autoPolicy(faceEdge.size(), 1e5), countAt(0_uz),
           countAt(faceEdge.size() - 1), ctx, [&](size_t face) {
             const int numEdge = faceEdge[face + 1] - faceEdge[face];
             if (numEdge == 0) {
               triOffset[face] = 0;
               return;
             }
             DEBUG_ASSERT(numEdge >= 3, topologyErr,
                          "face has less than three edges.");
             triOffset[face] = numEdge - 2;
             if (numEdge > 4)
               group.run([&, face] {
                 if (IsCancelled(ctx)) return;
                 HalfedgeTriangulation triangulation =
                     generalTriangulation(face);
                 triOffset[face] = triangulation.NumTri();
                 results[face] = std::move(triangulation);
               });
           });
  group.wait();
  if (IsCancelled(ctx)) return;
#else
  std::unordered_map<int, HalfedgeTriangulation> results;
  for (int face = 0; face < static_cast<int>(faceEdge.size()) - 1; ++face) {
    if (IsCancelled(ctx)) return;
    const int numEdge = faceEdge[face + 1] - faceEdge[face];
    if (numEdge == 0) {
      triOffset[face] = 0;
      continue;
    }
    DEBUG_ASSERT(numEdge >= 3, topologyErr, "face has less than three edges.");
    triOffset[face] = numEdge - 2;
    if (numEdge > 4) {
      HalfedgeTriangulation triangulation = generalTriangulation(face);
      triOffset[face] = triangulation.NumTri();
      results.emplace(face, std::move(triangulation));
    }
  }
#endif

  exclusive_scan(triOffset.begin(), triOffset.end(), triOffset.begin(), 0_uz);
  halfedge_.resize_nofill(3 * triOffset.back());
  triNormal.resize(triOffset.back());
  triRef.resize(triOffset.back());

#if (MANIFOLD_PAR == 1) && __has_include(<tbb/tbb.h>)
  auto processFace2 = [&](size_t face) {
    const HalfedgeTriangulation* resultPtr = nullptr;
    auto result = results.find(face);
    if (result != results.end()) resultPtr = &result->second;
    outputFace(face, triOffset[face], resultPtr);
  };
  // set triangles in parallel
  for_each(autoPolicy(faceEdge.size(), 1e4), countAt(0_uz),
           countAt(faceEdge.size() - 1), ctx, processFace2);
#else
  for (size_t face = 0; face < faceEdge.size() - 1; ++face) {
    if (IsCancelled(ctx)) return;
    const HalfedgeTriangulation* resultPtr = nullptr;
    auto result = results.find(face);
    if (result != results.end()) resultPtr = &result->second;
    outputFace(face, triOffset[face], resultPtr);
  }
#endif

  if (IsCancelled(ctx)) return;
  for_each(autoPolicy(faceHalfedge.size(), 1e5), countAt(0_uz),
           countAt(faceHalfedge.size()), ctx, [&](size_t edge) {
             const int triEdge = contour2Tri[edge];
             if (triEdge < 0) return;
             const int pair = faceHalfedge[edge].pairedHalfedge;
             if (pair < 0) return;
             const int pairTri = contour2Tri[pair];
             DEBUG_ASSERT(pairTri >= 0, topologyErr,
                          "boundary edge did not triangulate with its pair");
             halfedge_.SetPair(triEdge, pairTri);
           });
  if (IsCancelled(ctx)) return;
  faceNormal_ = std::move(triNormal);
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
      const double z = vertPos_[halfedge_.Start(3 * tri + j)].z;
      min = std::min(min, z);
      max = std::max(max, z);
    }

    if (min <= height && max > height) {
      tris.insert(tri);
    }
  };

  auto recorder = MakeSimpleRecorder(recordCollision);
  collider_.Collisions<false>(recorder, query.cview(), false);

  Polygons polys;
  while (!tris.empty()) {
    const int startTri = *tris.begin();
    SimplePolygon poly;

    int k = 0;
    for (const int j : {0, 1, 2}) {
      if (vertPos_[halfedge_.Start(3 * startTri + j)].z > height &&
          vertPos_[halfedge_.Start(3 * startTri + Next3(j))].z <= height) {
        k = Next3(j);
        break;
      }
    }

    int tri = startTri;
    do {
      tris.erase(tris.find(tri));
      const int edge = 3 * tri + k;
      if (vertPos_[halfedge_.End(edge)].z <= height) {
        k = Next3(k);
      }

      const int up = 3 * tri + k;
      const vec3 below = vertPos_[halfedge_.Start(up)];
      const vec3 above = vertPos_[halfedge_.End(up)];
      const double a = (height - below.z) / (above.z - below.z);
      poly.push_back(vec2(la::lerp(below, above, a)));

      const int pair = halfedge_.Pair(up);
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
  size_t numCusps = 0;
  for (size_t i = 0; i < halfedge_.size(); ++i) {
    const int pair = halfedge_.Pair(i);
    if (faceNormal_[halfedge_.Pair(pair) / 3].z >= 0 &&
        faceNormal_[pair / 3].z < 0) {
      cusps[numCusps++] = halfedge_.Get(i);
    }
  }
  cusps.resize(numCusps);

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
