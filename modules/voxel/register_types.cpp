#include "register_types.h"
#include "voxel_chunk.h"
#include "voxel_world.h"
#include "core/object/class_db.h"

void initialize_voxel_module(ModuleInitializationLevel p_level) {
    if (p_level != MODULE_INITIALIZATION_LEVEL_SCENE) return;
    ClassDB::register_class<VoxelChunk>();
    ClassDB::register_class<VoxelWorld>();
}

void uninitialize_voxel_module(ModuleInitializationLevel p_level) {}