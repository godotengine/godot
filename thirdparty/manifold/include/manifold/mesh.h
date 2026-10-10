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

#pragma once

#include <cstdint>
#include <istream>
#include <type_traits>
#include <vector>

#include "manifold/common.h"

namespace manifold {

/** @addtogroup Core
 *  @{
 */

/**
 * @brief Mesh input/output suitable for pushing directly into graphics
 * libraries.
 *
 * The core (non-optional) parts of MeshGL are the triVerts indices buffer and
 * the vertProperties interleaved vertex buffer, which follow the conventions of
 * OpenGL (and other graphic libraries') buffers and are therefore generally
 * easy to map directly to other applications' data structures.
 *
 * The triVerts vector has a stride of 3 and specifies triangles as
 * vertex indices. For triVerts = [2, 4, 5, 3, 1, 6, ...], the triangles are [2,
 * 4, 5], [3, 1, 6], etc. and likewise the halfedges are [2, 4], [4, 5], [5, 2],
 * [3, 1], [1, 6], [6, 3], etc.
 *
 * The triVerts indices should form a manifold mesh: each of the 3 halfedges of
 * each triangle should have exactly one paired halfedge in the list, defined as
 * having the first index of one equal to the second index of the other and
 * vice-versa. However, this is not always possible - consider e.g. a cube with
 * normal-vector properties. Shared vertices would turn the cube into a ball by
 * interpolating normals - the common solution is to duplicate each corner
 * vertex into 3, each with the same position, but different normals
 * corresponding to each face. This is exactly what should be done in MeshGL,
 * however we request two additional vectors in this case: mergeFromVert and
 * mergeToVert. Each vertex mergeFromVert[i] is merged into vertex
 * mergeToVert[i], avoiding unreliable floating-point comparisons to recover the
 * manifold topology. These merges are simply a union, so which is from and to
 * doesn't matter.
 *
 * If you don't have merge vectors, you can create them with the Merge() method,
 * however this will fail if the mesh is not already manifold within the set
 * tolerance. For maximum reliability, always store the merge vectors with the
 * mesh, e.g. using the EXT_mesh_manifold extension in glTF.
 *
 * You can have any number of arbitrary floating-point properties per vertex,
 * and they will all be interpolated as necessary during operations. It is up to
 * you to keep track of which channel represents what type of data. A few of
 * Manifold's methods allow you to specify the channel where normals data
 * starts, in order to update it automatically for transforms and such. This
 * will be easier if your meshes all use the same channels for properties, but
 * this is not a requirement. Operations between meshes with different numbers
 * of properties will simply use the larger numProp and pad the smaller one
 * with zeroes.
 *
 * On output, the triangles are sorted into runs (runIndex, runOriginalID,
 * runTransform) that correspond to different mesh inputs. Other 3D libraries
 * may refer to these runs as primitives of a mesh (as in glTF) or draw calls,
 * as they often represent different materials on different parts of the mesh.
 * It is generally a good idea to maintain a map of OriginalIDs to materials to
 * make it easy to reapply them after a set of Boolean operations. These runs
 * can also be used as input, and thus also ensure a lossless roundtrip of data
 * through MeshGL.
 *
 * As an example, with runIndex = [0, 6, 18, 21] and runOriginalID = [1, 3, 3],
 * there are 7 triangles, where the first two are from the input mesh with ID 1,
 * the next 4 are from an input mesh with ID 3, and the last triangle is from a
 * different copy (instance) of the input mesh with ID 3. These two instances
 * can be distinguished by their different runTransform matrices.
 *
 * You can reconstruct polygonal faces by assembling all the triangles that are
 * from the same run and share the same faceID. These faces will be planar
 * within the output tolerance.
 *
 * The halfedgeTangent vector is used to specify the weighted tangent vectors of
 * each halfedge for the purpose of using the Refine methods to create a
 * smoothly-interpolated surface. They can also be output when calculated
 * automatically by the Smooth functions.
 *
 * MeshGL is an alias for the standard single-precision version. Use MeshGL64 to
 * output the full double precision that Manifold uses internally.
 */
// MeshGLP / MeshGL / MeshGL64 are forward-declared in common.h; the
// default `I = uint32_t` lives on the forward decl.
template <typename Precision, typename I>
struct MeshGLP {
  /// Number of property vertices
  I NumVert() const { return vertProperties.size() / numProp; };
  /// Number of triangles
  I NumTri() const { return triVerts.size() / 3; };
  /// Number of triangle runs
  I NumRun() const { return runOriginalID.size(); };
  /// Number of properties per vertex, always >= 3.
  I numProp = 3;
  /// Flat, GL-style interleaved list of all vertex properties: propVal =
  /// vertProperties[vert * numProp + propIdx]. The first three properties are
  /// always the position x, y, z. The stride of the array is numProp.
  std::vector<Precision> vertProperties;
  /// The vertex indices of the three triangle corners in CCW (from the outside)
  /// order, for each triangle.
  std::vector<I> triVerts;
  /// Optional: A list of only the vertex indicies that need to be merged to
  /// reconstruct the manifold.
  std::vector<I> mergeFromVert;
  /// Optional: The same length as mergeFromVert, and the corresponding value
  /// contains the vertex to merge with. It will have an identical position, but
  /// the other properties may differ.
  std::vector<I> mergeToVert;
  /// Optional: Indicates runs of triangles that correspond to a particular
  /// input mesh instance. The runs encompass all of triVerts and are sorted
  /// by runOriginalID. Run i begins at triVerts[runIndex[i]] and ends at
  /// triVerts[runIndex[i+1]]. All runIndex values are divisible by 3. Returned
  /// runIndex will always be 1 longer than runOriginalID, but same length is
  /// also allowed as input: triVerts.size() will be automatically appended in
  /// this case.
  std::vector<I> runIndex;
  /// Optional: The OriginalID of the mesh this triangle run came from. This ID
  /// is ideal for reapplying materials to the output mesh. Multiple runs may
  /// have the same ID, e.g. representing different copies of the same input
  /// mesh. If you create an input MeshGL that you want to be able to reference
  /// as one or more originals, be sure to set unique values from ReserveIDs().
  std::vector<uint32_t> runOriginalID;
  /// Optional: For each run, a 3x4 transform is stored representing how the
  /// corresponding original mesh was transformed to create this triangle run.
  /// This matrix is stored in column-major order and the length of the overall
  /// vector is 12 * runOriginalID.size().
  std::vector<Precision> runTransform;
  /// Optional: For each run, defines a set of flags giving extra information
  /// about the run. See the corresponding getter functions for details on the
  /// specific flags. These are primarily used on output.
  std::vector<uint8_t> runFlags;
  /// Optional: Length NumTri, contains the source face ID this triangle comes
  /// from. Simplification will maintain all edges between triangles with
  /// different faceIDs. Input faceIDs will be maintained to the outputs, but if
  /// none are given, they will be filled in with Manifold's coplanar face
  /// calculation based on mesh tolerance.
  std::vector<I> faceID;
  /// Optional: The X-Y-Z-W weighted tangent vectors for smooth Refine(). If
  /// non-empty, must be exactly four times as long as Mesh.triVerts. Indexed
  /// as 4 * (3 * tri + i) + j, i < 3, j < 4, representing the tangent value
  /// Mesh.triVerts[tri][i] along the CCW edge. If empty, mesh is faceted.
  std::vector<Precision> halfedgeTangent;
  /// Tolerance for mesh simplification. When creating a Manifold, the tolerance
  /// used will be the maximum of this and a baseline tolerance from the size of
  /// the bounding box. Any edge shorter than tolerance may be collapsed.
  /// Tolerance may be enlarged when floating point error accumulates.
  Precision tolerance = 0;

  MeshGLP() = default;

  /**
   * Updates the mergeFromVert and mergeToVert vectors in order to create a
   * manifold solid. If the MeshGL is already manifold, no change will occur
   * and the function will return false. Otherwise, this will merge verts
   * along open edges within tolerance (the maximum of the MeshGL tolerance
   * and the baseline bounding-box tolerance), keeping any from the existing
   * merge vectors, and return true.
   *
   * There is no guarantee the result will be manifold - this is a
   * best-effort helper function designed primarily to aid in the case where
   * a manifold multi-material MeshGL was produced, but its merge vectors
   * were lost due to a round-trip through a file format. Constructing a
   * Manifold from the result will report an error status if it is not
   * manifold.
   */
  bool Merge();

  /**
   * Returns the x, y, z position of the ith vertex.
   *
   * @param v vertex index.
   */
  la::vec<Precision, 3> GetVertPos(size_t v) const {
    size_t offset = v * numProp;
    return la::vec<Precision, 3>(vertProperties[offset],
                                 vertProperties[offset + 1],
                                 vertProperties[offset + 2]);
  }

  /**
   * Returns the three vertex indices of the ith triangle.
   *
   * @param t triangle index.
   */
  la::vec<I, 3> GetTriVerts(size_t t) const {
    size_t offset = 3 * t;
    return la::vec<I, 3>(triVerts[offset], triVerts[offset + 1],
                         triVerts[offset + 2]);
  }

  /**
   * Returns the x, y, z, w tangent of the ith halfedge.
   *
   * @param h halfedge index (3 * triangle_index + [0|1|2]).
   */
  la::vec<Precision, 4> GetTangent(size_t h) const {
    size_t offset = 4 * h;
    return la::vec<Precision, 4>(
        halfedgeTangent[offset], halfedgeTangent[offset + 1],
        halfedgeTangent[offset + 2], halfedgeTangent[offset + 3]);
  }

  /**
   * Returns the transformation matrix for the specified run.
   *
   * @param run The index of the triangle run (0 <= run < runOriginalID.size()).
   */
  mat3x4 GetRunTransform(size_t run) const {
    size_t offset = 12 * run;
    if (offset + 12 > runTransform.size()) {
      return la::identity;
    }
    return mat3x4(la::mat<Precision, 3, 4>(&runTransform[offset]));
  }

  /**
   * Returns true if this triangle run is on the backside compared to the
   * original mesh, e.g. from a subtraction. Informational only - the framework
   * already orients stored normals so the standard `getMesh()` flow returns
   * world-frame values regardless of this bit.
   *
   * @param run The index of the triangle run (0 <= run < runFlags.size()).
   */
  bool Backside(size_t run) const {
    return run < runFlags.size() && (runFlags[run] & 1) != 0;
  }

  /**
   * Returns true if the first three extra-property channels (slots 3, 4, 5)
   * of this run carry world-frame vertex normals (set by
   * `Manifold::CalculateNormals(0)` and round-tripped via `runFlags` bit 1).
   * Consumers should treat the slot as normals and skip re-applying
   * `runTransform` to it.
   *
   * hasNormals is per-run, so different runs may set it differently.
   * Behavior is undefined when a single propVert is shared by triangles
   * from runs that disagree - the slot has one interpretation, and a
   * Transform rotates it for hasNormals=true and clobbers any
   * hasNormals=false sharer. Standard `CalculateNormals` / Boolean /
   * Compose outputs never produce that shape.
   *
   * @param run The index of the triangle run (0 <= run < runFlags.size()).
   */
  bool HasNormals(size_t run) const {
    return run < runFlags.size() && (runFlags[run] & 2) != 0;
  }
};

// MeshGL / MeshGL64 aliases live in common.h. Pin the default template
// arg so dropping `I = uint32_t` breaks the build instead of silently
// changing MeshGL's index type.
static_assert(std::is_same<MeshGL, MeshGLP<float, uint32_t>>::value,
              "MeshGL default index type must stay uint32_t");

#ifndef MANIFOLD_NO_IOSTREAM
MeshGL64 ReadOBJ(std::istream& stream);
bool WriteOBJ(std::ostream& stream, const MeshGL64& mesh);
#endif
/** @} */
}  // namespace manifold