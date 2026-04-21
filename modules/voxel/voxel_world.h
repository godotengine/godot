#pragma once
#include "scene/3d/node_3d.h"
#include "voxel_chunk.h"
#include "core/templates/hash_map.h"

class VoxelWorld : public Node3D {
    GDCLASS(VoxelWorld, Node3D);

public:
    void set_view_distance(int d) { view_distance = d; }
    int get_view_distance() const { return view_distance; }
    void update_chunks(Vector3 player_pos);
    void generate_chunk(Vector3i chunk_pos);

protected:
    static void _bind_methods();

private:
    int view_distance = 8;
    HashMap<Vector3i, VoxelChunk*> chunks;
    Vector3i world_to_chunk(Vector3 pos) const;
    void unload_chunk(Vector3i pos);
};