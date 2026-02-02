#pragma once

#include <cstdint>
#include <vector>

namespace tinyusdz {

///
/// Subdivided Mesh(for rendering. trianglulated)
///
typedef struct {
  // num_triangle_faces = indices.size() / 3
  std::vector<float> vertices;       /// [xyz] * num_vertices
  std::vector<float> vertex_colors;  /// [rgb] * num_vertices

  std::vector<float>
      facevarying_normals;  /// [xyz] * 3(triangle) * num_triangle_faces
  std::vector<float>
      facevarying_tangents;  /// [xyz] * 3(triangle) * num_triangle_faces
  std::vector<float>
      facevarying_binormals;  /// [xyz] * 3(triangle) * num_triangle_faces
  std::vector<float>
      facevarying_uvs;  /// [xy]  * 3(triangle) * num_triangle_faces

  std::vector<int32_t> material_ids;  /// per face materials. -1 = no material. index x num_triangle_faces

  // List of triangle vertex indices. For NanoRT BVH
  std::vector<uint32_t>
      triangulated_indices;  /// 3(triangle) x num_triangle_faces

  // List of original vertex indices. For UV interpolation
  std::vector<uint32_t>
      face_indices;  /// length = sum(for each face_num_verts[i])

  // Offset to `face_indices` for a given face_id.
  std::vector<uint32_t> face_index_offsets;  /// length = face_num_verts.size()

  std::vector<uint8_t> face_num_verts;  /// # of vertices per face

  // face ID for each triangle. For ptex textureing.
  std::vector<uint32_t> face_ids;  /// index x num_triangle_faces

  // Triangule ID of a face(e.g. 0 for triangle primitive. 0 or 1 for quad
  // primitive(tessellated into two-triangles)
  std::vector<uint8_t> face_triangle_ids;  /// index x num_triangle_faces

} SubdividedMesh;

///
/// Initial control mesh(input to Subdivision Surface)
/// All faces should be quad face for this example program.
///
struct ControlQuadMesh {
  std::vector<float> vertices;       // [xyz] * num_vertices
  std::vector<int> indices;          // length = sum_{i}(verts_per_face[i])
  std::vector<int> verts_per_faces;  // should be 4(quad)

  std::vector<float> faevarying_uvs;  // 2 * num_faces(vert_per_faces.size)
  std::vector<int> facevarying_uv_indices;  // length = indices.size()
};

///
/// template class for OSD.
///
struct Vertex {
  // Minimal required interface ----------------------
  Vertex() = default;

  Vertex(const Vertex &src) {
    _position[0] = src._position[0];
    _position[1] = src._position[1];
    _position[2] = src._position[2];
  }

  Vertex &operator=(const Vertex &rhs) = default;

  void Clear(void * = nullptr) { _position[0] = _position[1] = _position[2] = 0.0f; }

  void AddWithWeight(Vertex const &src, float weight) {
    _position[0] += weight * src._position[0];
    _position[1] += weight * src._position[1];
    _position[2] += weight * src._position[2];
  }

  // Public interface ------------------------------------
  void SetPosition(float x, float y, float z) {
    _position[0] = x;
    _position[1] = y;
    _position[2] = z;
  }

  const float *GetPosition() const { return _position; }

 private:
  float _position[3];
};

///
/// Uniformly subdivide the mesh.
///
/// @param[in] level Subdivision level.
/// @param[in] in_mesh Input quad mesh.
/// @param[out] out_mesh Subdivided mesh.
/// @param[in] dump Dump .obj for debugging.
///
bool subdivide(int level, const ControlQuadMesh &in_mesh,
               SubdividedMesh *out_mesh, std::string *err, bool dump = false);

}  // namespace tinyusdz
