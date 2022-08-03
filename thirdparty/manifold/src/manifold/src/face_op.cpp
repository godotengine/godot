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

#include <map>

#include "impl.h"
#include "polygon.h"

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
void Manifold::Impl::Face2Tri(const VecDH<int>& faceEdge,
                              const VecDH<BaryRef>& faceRef,
                              const VecDH<int>& halfedgeBary) {
  VecDH<glm::ivec3> triVerts;
  VecDH<glm::vec3> triNormal;
  VecDH<BaryRef>& triBary = meshRelation_.triBary;
  triBary.resize(0);
  triVerts.reserve(halfedge_.size());
  triNormal.reserve(halfedge_.size());
  triBary.reserve(halfedge_.size());

  for (int face = 0; face < faceEdge.size() - 1; ++face) {
    const int firstEdge = faceEdge[face];
    const int lastEdge = faceEdge[face + 1];
    const int numEdge = lastEdge - firstEdge;
    ASSERT(numEdge >= 3, topologyErr, "face has less than three edges.");
    const glm::vec3 normal = faceNormal_[face];

    auto linearSearch = [](const int* mapping, int value) {
      int i = 0;
      while (mapping[i] != value) ++i;
      return i;
    };

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

      triVerts.push_back(tri);
      triNormal.push_back(normal);
      triBary.push_back(faceRef[face]);
      for (int k : {0, 1, 2}) {
        int index = linearSearch(mapping, tri[k]);
        triBary.back().vertBary[k] = halfedgeBary[firstEdge + index];
      }
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

      for (auto tri : {tri0, tri1}) {
        triVerts.push_back(tri);
        triNormal.push_back(normal);
        triBary.push_back(faceRef[face]);
        for (int k : {0, 1, 2}) {
          int index = linearSearch(mapping, tri[k]);
          triBary.back().vertBary[k] = halfedgeBary[firstEdge + index];
        }
      }
    } else {  // General triangulation
      const glm::mat3x2 projection = GetAxisAlignedProjection(normal);

      std::map<int, int> vertBary;
      for (int j = firstEdge; j < lastEdge; ++j)
        vertBary[halfedge_[j].startVert] = halfedgeBary[j];

      const Polygons polys = Face2Polygons(face, projection, faceEdge);

      std::vector<glm::ivec3> newTris = Triangulate(polys, precision_);

      for (auto tri : newTris) {
        triVerts.push_back(tri);
        triNormal.push_back(normal);
        triBary.push_back(faceRef[face]);
        for (int k : {0, 1, 2}) {
          triBary.back().vertBary[k] = vertBary[tri[k]];
        }
      }
    }
  }
  faceNormal_ = std::move(triNormal);
  CreateHalfedges(triVerts);
}

/**
 * For the input face index, return a set of 2D polygons formed by the input
 * projection of the vertices.
 */
Polygons Manifold::Impl::Face2Polygons(int face, glm::mat3x2 projection,
                                       const VecDH<int>& faceEdge) const {
  const int firstEdge = faceEdge[face];
  const int lastEdge = faceEdge[face + 1];

  std::map<int, int> vert_edge;
  for (int edge = firstEdge; edge < lastEdge; ++edge) {
    const bool inserted =
        vert_edge.emplace(std::make_pair(halfedge_[edge].startVert, edge))
            .second;
    ASSERT(inserted, topologyErr, "face has duplicate vertices.");
  }

  Polygons polys;
  int startEdge = 0;
  int thisEdge = startEdge;
  while (1) {
    if (thisEdge == startEdge) {
      if (vert_edge.empty()) break;
      startEdge = vert_edge.begin()->second;
      thisEdge = startEdge;
      polys.push_back({});
    }
    int vert = halfedge_[thisEdge].startVert;
    polys.back().push_back({projection * vertPos_[vert], vert});
    const auto result = vert_edge.find(halfedge_[thisEdge].endVert);
    ASSERT(result != vert_edge.end(), topologyErr, "nonmanifold edge");
    thisEdge = result->second;
    vert_edge.erase(result);
  }
  return polys;
}
}  // namespace manifold
