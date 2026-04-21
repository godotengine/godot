#pragma once
#include "scene/3d/mesh_instance_3d.h"

class VoxelChunk : public MeshInstance3D {
    GDCLASS(VoxelChunk, MeshInstance3D);

public:
    static const int SIZE = 16;
    static const int HEIGHT = 128;

    enum VoxelType { AIR = 0, STONE, DIRT, GRASS };

    void set_voxel(int x, int y, int z, int type);
    int get_voxel(int x, int y, int z) const;
    void rebuild_mesh();

protected:
    static void _bind_methods();

private:
    VoxelType voxels[SIZE][HEIGHT][SIZE] = {};
    bool _in_bounds(int x, int y, int z) const;
    bool _is_air(int x, int y, int z) const;
};