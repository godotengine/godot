// Copyright 2023 The Manifold Authors.
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

#include "sdf.h"

#include "hashtable.h"
#include "par.h"
#include "utils.h"
#include "vec.h"

namespace {
using namespace manifold;
Uint64 identity(Uint64 x) { return x; }

glm::ivec3 TetTri0(int i) {
  constexpr glm::ivec3 tetTri0[16] = {{-1, -1, -1},  //
                                      {0, 3, 4},     //
                                      {0, 1, 5},     //
                                      {1, 5, 3},     //
                                      {1, 4, 2},     //
                                      {1, 0, 3},     //
                                      {2, 5, 0},     //
                                      {5, 3, 2},     //
                                      {2, 3, 5},     //
                                      {0, 5, 2},     //
                                      {3, 0, 1},     //
                                      {2, 4, 1},     //
                                      {3, 5, 1},     //
                                      {5, 1, 0},     //
                                      {4, 3, 0},     //
                                      {-1, -1, -1}};
  return tetTri0[i];
}

glm::ivec3 TetTri1(int i) {
  constexpr glm::ivec3 tetTri1[16] = {{-1, -1, -1},  //
                                      {-1, -1, -1},  //
                                      {-1, -1, -1},  //
                                      {3, 4, 1},     //
                                      {-1, -1, -1},  //
                                      {3, 2, 1},     //
                                      {0, 4, 2},     //
                                      {-1, -1, -1},  //
                                      {-1, -1, -1},  //
                                      {2, 4, 0},     //
                                      {1, 2, 3},     //
                                      {-1, -1, -1},  //
                                      {1, 4, 3},     //
                                      {-1, -1, -1},  //
                                      {-1, -1, -1},  //
                                      {-1, -1, -1}};
  return tetTri1[i];
}

glm::ivec4 Neighbors(int i) {
  constexpr glm::ivec4 neighbors[7] = {{0, 0, 0, 1},   //
                                       {1, 0, 0, 0},   //
                                       {0, 1, 0, 0},   //
                                       {0, 0, 1, 0},   //
                                       {-1, 0, 0, 1},  //
                                       {0, -1, 0, 1},  //
                                       {0, 0, -1, 1}};
  return neighbors[i];
}

Uint64 SpreadBits3(Uint64 v) {
  v = v & 0x1fffff;
  v = (v | v << 32) & 0x1f00000000ffff;
  v = (v | v << 16) & 0x1f0000ff0000ff;
  v = (v | v << 8) & 0x100f00f00f00f00f;
  v = (v | v << 4) & 0x10c30c30c30c30c3;
  v = (v | v << 2) & 0x1249249249249249;
  return v;
}

Uint64 SqueezeBits3(Uint64 v) {
  v = v & 0x1249249249249249;
  v = (v ^ v >> 2) & 0x10c30c30c30c30c3;
  v = (v ^ v >> 4) & 0x100f00f00f00f00f;
  v = (v ^ v >> 8) & 0x1f0000ff0000ff;
  v = (v ^ v >> 16) & 0x1f00000000ffff;
  v = (v ^ v >> 32) & 0x1fffff;
  return v;
}

// This is a modified 3D MortonCode, where the xyz code is shifted by one bit
// and the w bit is added as the least significant. This allows 21 bits per x,
// y, and z channel and 1 for w, filling the 64 bit total.
Uint64 MortonCode(const glm::ivec4& index) {
  return static_cast<Uint64>(index.w) | (SpreadBits3(index.x) << 1) |
         (SpreadBits3(index.y) << 2) | (SpreadBits3(index.z) << 3);
}

glm::ivec4 DecodeMorton(Uint64 code) {
  glm::ivec4 index;
  index.x = SqueezeBits3(code >> 1);
  index.y = SqueezeBits3(code >> 2);
  index.z = SqueezeBits3(code >> 3);
  index.w = code & 0x1u;
  return index;
}

struct GridVert {
  float distance = NAN;
  int edgeVerts[7] = {-1, -1, -1, -1, -1, -1, -1};

  int Inside() const { return distance > 0 ? 1 : -1; }

  int NeighborInside(int i) const {
    return Inside() * (edgeVerts[i] < 0 ? 1 : -1);
  }
};

struct ComputeVerts {
  VecView<glm::vec3> vertPos;
  VecView<int> vertIndex;
  HashTableD<GridVert, identity> gridVerts;
  const std::function<float(glm::vec3)> sdf;
  const glm::vec3 origin;
  const glm::ivec3 gridSize;
  const glm::vec3 spacing;
  const float level;

  inline glm::vec3 Position(glm::ivec4 gridIndex) const {
    return origin +
           spacing * (glm::vec3(gridIndex) + (gridIndex.w == 1 ? 0.0f : -0.5f));
  }

  inline float BoundedSDF(glm::ivec4 gridIndex) const {
    const float d = sdf(Position(gridIndex)) - level;

    const glm::ivec3 xyz(gridIndex);
    const bool onLowerBound = glm::any(glm::lessThanEqual(xyz, glm::ivec3(0)));
    const bool onUpperBound = glm::any(glm::greaterThanEqual(xyz, gridSize));
    const bool onHalfBound =
        gridIndex.w == 1 && glm::any(glm::greaterThanEqual(xyz, gridSize - 1));
    if (onLowerBound || onUpperBound || onHalfBound) return glm::min(d, 0.0f);

    return d;
  }

  inline void operator()(Uint64 mortonCode) {
    ZoneScoped;
    if (gridVerts.Full()) return;

    const glm::ivec4 gridIndex = DecodeMorton(mortonCode);

    if (glm::any(glm::greaterThan(glm::ivec3(gridIndex), gridSize))) return;

    const glm::vec3 position = Position(gridIndex);

    GridVert gridVert;
    gridVert.distance = BoundedSDF(gridIndex);

    bool keep = false;
    // These seven edges are uniquely owned by this gridVert; any of them
    // which intersect the surface create a vert.
    for (int i = 0; i < 7; ++i) {
      glm::ivec4 neighborIndex = gridIndex + Neighbors(i);
      if (neighborIndex.w == 2) {
        neighborIndex += 1;
        neighborIndex.w = 0;
      }
      const float val = BoundedSDF(neighborIndex);
      if ((val > 0) == (gridVert.distance > 0)) continue;
      keep = true;

      const int idx = AtomicAdd(vertIndex[0], 1);
      vertPos[idx] =
          (val * position - gridVert.distance * Position(neighborIndex)) /
          (val - gridVert.distance);
      gridVert.edgeVerts[i] = idx;
    }

    if (keep) gridVerts.Insert(mortonCode, gridVert);
  }
};

struct BuildTris {
  VecView<glm::ivec3> triVerts;
  VecView<int> triIndex;
  const HashTableD<GridVert, identity> gridVerts;

  void CreateTri(const glm::ivec3& tri, const int edges[6]) {
    if (tri[0] < 0) return;
    int idx = AtomicAdd(triIndex[0], 1);
    triVerts[idx] = {edges[tri[0]], edges[tri[1]], edges[tri[2]]};
  }

  void CreateTris(const glm::ivec4& tet, const int edges[6]) {
    const int i = (tet[0] > 0 ? 1 : 0) + (tet[1] > 0 ? 2 : 0) +
                  (tet[2] > 0 ? 4 : 0) + (tet[3] > 0 ? 8 : 0);
    CreateTri(TetTri0(i), edges);
    CreateTri(TetTri1(i), edges);
  }

  void operator()(int idx) {
    ZoneScoped;
    Uint64 basekey = gridVerts.KeyAt(idx);
    if (basekey == kOpen) return;

    const GridVert& base = gridVerts.At(idx);
    const glm::ivec4 baseIndex = DecodeMorton(basekey);

    glm::ivec4 leadIndex = baseIndex;
    if (leadIndex.w == 0)
      leadIndex.w = 1;
    else {
      leadIndex += 1;
      leadIndex.w = 0;
    }

    // This GridVert is in charge of the 6 tetrahedra surrounding its edge in
    // the (1,1,1) direction (edge 0).
    glm::ivec4 tet(base.NeighborInside(0), base.Inside(), -2, -2);
    glm::ivec4 thisIndex = baseIndex;
    thisIndex.x += 1;

    GridVert thisVert = gridVerts[MortonCode(thisIndex)];

    tet[2] = base.NeighborInside(1);
    for (const int i : {0, 1, 2}) {
      thisIndex = leadIndex;
      --thisIndex[Prev3(i)];
      // MortonCodes take unsigned input, so check for negatives, given the
      // decrement.
      GridVert nextVert = thisIndex[Prev3(i)] < 0
                              ? GridVert()
                              : gridVerts[MortonCode(thisIndex)];
      tet[3] = base.NeighborInside(Prev3(i) + 4);

      const int edges1[6] = {base.edgeVerts[0],
                             base.edgeVerts[i + 1],
                             nextVert.edgeVerts[Next3(i) + 4],
                             nextVert.edgeVerts[Prev3(i) + 1],
                             thisVert.edgeVerts[i + 4],
                             base.edgeVerts[Prev3(i) + 4]};
      thisVert = nextVert;
      CreateTris(tet, edges1);

      thisIndex = baseIndex;
      ++thisIndex[Next3(i)];
      nextVert = gridVerts[MortonCode(thisIndex)];
      tet[2] = tet[3];
      tet[3] = base.NeighborInside(Next3(i) + 1);

      const int edges2[6] = {base.edgeVerts[0],
                             edges1[5],
                             thisVert.edgeVerts[i + 4],
                             nextVert.edgeVerts[Next3(i) + 4],
                             edges1[3],
                             base.edgeVerts[Next3(i) + 1]};
      thisVert = nextVert;
      CreateTris(tet, edges2);

      tet[2] = tet[3];
    }
  }
};
}  // namespace

namespace manifold {

/** @addtogroup Core
 *  @{
 */

/**
 * Constructs a level-set Mesh from the input Signed-Distance Function (SDF).
 * This uses a form of Marching Tetrahedra (akin to Marching Cubes, but better
 * for manifoldness). Instead of using a cubic grid, it uses a body-centered
 * cubic grid (two shifted cubic grids). This means if your function's interior
 * exceeds the given bounds, you will see a kind of egg-crate shape closing off
 * the manifold, which is due to the underlying grid.
 *
 * @param sdf The signed-distance functor, containing this function signature:
 * `float operator()(glm::vec3 point)`, which returns the
 * signed distance of a given point in R^3. Positive values are inside,
 * negative outside.
 * @param bounds An axis-aligned box that defines the extent of the grid.
 * @param edgeLength Approximate maximum edge length of the triangles in the
 * final result. This affects grid spacing, and hence has a strong effect on
 * performance.
 * @param level You can inset your Mesh by using a positive value, or outset
 * it with a negative value.
 * @param canParallel Parallel policies violate will crash language runtimes
 * with runtime locks that expect to not be called back by unregistered threads.
 * This allows bindings use LevelSet despite being compiled with MANIFOLD_PAR
 * active.
 * @return Mesh This class does not depend on Manifold, so it just returns a
 * Mesh, but it is guaranteed to be manifold and so can always be used as
 * input to the Manifold constructor for further operations.
 */
Mesh LevelSet(std::function<float(glm::vec3)> sdf, Box bounds, float edgeLength,
              float level, bool canParallel) {
  Mesh out;

  const glm::vec3 dim = bounds.Size();
  const glm::ivec3 gridSize(dim / edgeLength);
  const glm::vec3 spacing = dim / (glm::vec3(gridSize));

  const Uint64 maxMorton = MortonCode(glm::ivec4(gridSize + 1, 1));

  // Parallel policies violate will crash language runtimes with runtime locks
  // that expect to not be called back by unregistered threads. This allows
  // bindings use LevelSet despite being compiled with MANIFOLD_PAR
  // active.
  const auto pol = canParallel ? autoPolicy(maxMorton) : ExecutionPolicy::Seq;

  size_t tableSize = glm::min(
      2 * maxMorton, static_cast<Uint64>(10 * glm::pow(maxMorton, 0.667)));
  HashTable<GridVert, identity> gridVerts(tableSize);
  Vec<glm::vec3> vertPos(gridVerts.Size() * 7);

  while (1) {
    Vec<int> index(1, 0);
    for_each_n(pol, countAt(0_z), maxMorton + 1,
               ComputeVerts({vertPos, index, gridVerts.D(), sdf, bounds.min,
                             gridSize + 1, spacing, level}));

    if (gridVerts.Full()) {  // Resize HashTable
      const glm::vec3 lastVert = vertPos[index[0] - 1];
      const Uint64 lastMorton =
          MortonCode(glm::ivec4((lastVert - bounds.min) / spacing, 1));
      const float ratio = static_cast<float>(maxMorton) / lastMorton;

      if (ratio > 1000)  // do not trust the ratio if it is too large
        tableSize *= 2;
      else
        tableSize *= ratio;
      gridVerts = HashTable<GridVert, identity>(tableSize);
      vertPos = Vec<glm::vec3>(gridVerts.Size() * 7);
    } else {  // Success
      vertPos.resize(index[0]);
      break;
    }
  }

  Vec<glm::ivec3> triVerts(gridVerts.Entries() * 12);  // worst case

  Vec<int> index(1, 0);
  for_each_n(pol, countAt(0_z), gridVerts.Size(),
             BuildTris({triVerts, index, gridVerts.D()}));
  triVerts.resize(index[0]);

  out.vertPos.insert(out.vertPos.end(), vertPos.begin(), vertPos.end());
  out.triVerts.insert(out.triVerts.end(), triVerts.begin(), triVerts.end());
  return out;
}
/** @} */
}  // namespace manifold
