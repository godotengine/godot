#pragma once
#include "scene/3d/node_3d.h"
#include "scene/3d/multimesh_instance_3d.h"
class SceneChunk;
// 非模型的擴展數據，粒子，聲音，自定义逻辑脚本
class SceneAddionsData : public Resource{
    GDCLASS(SceneAddionsData, Resource);

public:
    SceneAddionsData() {}

    String resource_path;
};
class SceneChunkInstanceBase : public Node3D {
    GDCLASS(SceneChunkInstanceBase, Node3D);

public:
    SceneChunkInstanceBase() {}
    void on_chunk_load(SceneChunk* p_chunk) {}
    void update_lod(Vector3 camera_pos) {}
    
    int curr_lod = 0;
};

class SceneChunk : public Node3D {
    GDCLASS(SceneChunk, Node3D);

public:
    SceneChunk() {}

    MultiMeshInstance3D* get_multimesh_instance(const String& name) {
        return Object::cast_to<MultiMeshInstance3D>(get_node(name));
    }

    void add_multimesh_instance(String res_path, const Transform3D& t) {
    }
    struct MeshInstance : public RefCounted{
        ObjectID mult_mesh_instances_id;
        HashMap<int32_t,Transform3D> mesh_transforms;
    };
    HashMap<String,Ref<MeshInstance>> mult_mesh_instances;  
    int curr_id = 0;
};