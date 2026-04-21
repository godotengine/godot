#include "voxel_chunk.h"
#include "scene/resources/surface_tool.h"

void VoxelChunk::_bind_methods() {
    ClassDB::bind_method(D_METHOD("set_voxel", "x", "y", "z", "type"), &VoxelChunk::set_voxel);
    ClassDB::bind_method(D_METHOD("get_voxel", "x", "y", "z"), &VoxelChunk::get_voxel);
    ClassDB::bind_method(D_METHOD("rebuild_mesh"), &VoxelChunk::rebuild_mesh);
}

bool VoxelChunk::_in_bounds(int x, int y, int z) const {
    return x >= 0 && x < SIZE && y >= 0 && y < HEIGHT && z >= 0 && z < SIZE;
}

bool VoxelChunk::_is_air(int x, int y, int z) const {
    if (!_in_bounds(x, y, z)) return true; // treat out-of-bounds as air (expose face)
    return voxels[x][y][z] == AIR;
}

void VoxelChunk::set_voxel(int x, int y, int z, VoxelType type) {
    if (!_in_bounds(x, y, z)) return;
    voxels[x][y][z] = type;
}

VoxelChunk::VoxelType VoxelChunk::get_voxel(int x, int y, int z) const {
    if (!_in_bounds(x, y, z)) return AIR;
    return voxels[x][y][z];
}

void VoxelChunk::rebuild_mesh() {
    Ref<SurfaceTool> st;
    st.instantiate();
    st->begin(Mesh::PRIMITIVE_TRIANGLES);

    // Face directions: +X, -X, +Y, -Y, +Z, -Z
    const Vector3 normals[6] = {
        Vector3(1,0,0), Vector3(-1,0,0),
        Vector3(0,1,0), Vector3(0,-1,0),
        Vector3(0,0,1), Vector3(0,0,-1)
    };

    // Vertices for each face (two triangles = one quad)
    const Vector3 faces[6][4] = {
        // +X
        { Vector3(1,0,0), Vector3(1,1,0), Vector3(1,1,1), Vector3(1,0,1) },
        // -X
        { Vector3(0,0,1), Vector3(0,1,1), Vector3(0,1,0), Vector3(0,0,0) },
        // +Y
        { Vector3(0,1,0), Vector3(0,1,1), Vector3(1,1,1), Vector3(1,1,0) },
        // -Y
        { Vector3(0,0,1), Vector3(0,0,0), Vector3(1,0,0), Vector3(1,0,1) },
        // +Z
        { Vector3(1,0,1), Vector3(1,1,1), Vector3(0,1,1), Vector3(0,0,1) },
        // -Z
        { Vector3(0,0,0), Vector3(0,1,0), Vector3(1,1,0), Vector3(1,0,0) },
    };

    const int neighbor[6][3] = {
        {1,0,0},{-1,0,0},{0,1,0},{0,-1,0},{0,0,1},{0,0,-1}
    };

    for (int x = 0; x < SIZE; x++) {
        for (int y = 0; y < HEIGHT; y++) {
            for (int z = 0; z < SIZE; z++) {
                if (voxels[x][y][z] == AIR) continue;

                for (int f = 0; f < 6; f++) {
                    int nx = x + neighbor[f][0];
                    int ny = y + neighbor[f][1];
                    int nz = z + neighbor[f][2];

                    if (!_is_air(nx, ny, nz)) continue; // face culling

                    Vector3 base(x, y, z);
                    st->set_normal(normals[f]);

                    // Quad as two triangles
                    st->set_uv(Vector2(0,0)); st->add_vertex(base + faces[f][0]);
                    st->set_uv(Vector2(0,1)); st->add_vertex(base + faces[f][1]);
                    st->set_uv(Vector2(1,1)); st->add_vertex(base + faces[f][2]);

                    st->set_uv(Vector2(0,0)); st->add_vertex(base + faces[f][0]);
                    st->set_uv(Vector2(1,1)); st->add_vertex(base + faces[f][2]);
                    st->set_uv(Vector2(1,0)); st->add_vertex(base + faces[f][3]);
                }
            }
        }
    }

    st->index();
    set_mesh(st->commit());
}