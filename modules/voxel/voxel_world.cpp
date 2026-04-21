#include "voxel_world.h"
#include "core/object/class_db.h"
#include "core/math/math_funcs.h"

void VoxelWorld::_bind_methods() {
    ClassDB::bind_method(D_METHOD("update_chunks", "player_pos"), &VoxelWorld::update_chunks);
    ClassDB::bind_method(D_METHOD("generate_chunk", "chunk_pos"), &VoxelWorld::generate_chunk);
    ClassDB::bind_method(D_METHOD("set_view_distance", "d"), &VoxelWorld::set_view_distance);
    ClassDB::bind_method(D_METHOD("get_view_distance"), &VoxelWorld::get_view_distance);
    ADD_PROPERTY(PropertyInfo(Variant::INT, "view_distance"), "set_view_distance", "get_view_distance");
}

Vector3i VoxelWorld::world_to_chunk(Vector3 pos) const {
    return Vector3i(
        Math::floor(pos.x / VoxelChunk::SIZE),
        0,
        Math::floor(pos.z / VoxelChunk::SIZE)
    );
}

void VoxelWorld::generate_chunk(Vector3i chunk_pos) {
    VoxelChunk* chunk = memnew(VoxelChunk);
    add_child(chunk);
    chunk->set_position(Vector3(
        chunk_pos.x * VoxelChunk::SIZE,
        0,
        chunk_pos.z * VoxelChunk::SIZE
    ));

    // Basic flat terrain — we'll replace with noise later
    for (int x = 0; x < VoxelChunk::SIZE; x++) {
        for (int z = 0; z < VoxelChunk::SIZE; z++) {
            for (int y = 0; y < 8; y++) {
                chunk->set_voxel(x, y, z, y < 7 ? VoxelChunk::STONE : VoxelChunk::GRASS);
            }
        }
    }

    chunk->rebuild_mesh();
    chunks[chunk_pos] = chunk;
}

void VoxelWorld::update_chunks(Vector3 player_pos) {
    Vector3i center = world_to_chunk(player_pos);

    // Load missing chunks
    for (int x = -view_distance; x <= view_distance; x++) {
        for (int z = -view_distance; z <= view_distance; z++) {
            Vector3i key = center + Vector3i(x, 0, z);
            if (!chunks.has(key)) {
                generate_chunk(key);
            }
        }
    }

    // Unload far chunks
    Vector<Vector3i> to_remove;
    for (auto& kv : chunks) {
        if (abs(kv.key.x - center.x) > view_distance + 2 ||
            abs(kv.key.z - center.z) > view_distance + 2) {
            to_remove.push_back(kv.key);
        }
    }
    for (Vector3i key : to_remove) {
        unload_chunk(key);
    }
}

void VoxelWorld::unload_chunk(Vector3i pos) {
    if (chunks.has(pos)) {
        chunks[pos]->queue_free();
        chunks.erase(pos);
    }
}