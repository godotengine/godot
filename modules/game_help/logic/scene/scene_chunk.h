#pragma once
#include "scene/3d/node_3d.h"
#include "scene/3d/multimesh_instance_3d.h"
class SceneChunk;
// 非模型的擴展數據，粒子，聲音，自定义逻辑脚本
class SceneDataCompoent : public RefCounted{
    GDCLASS(SceneDataCompoent, RefCounted);
    static void _bind_methods() {}

public:
    enum ResourceType{
        RT_Mesh,
        RT_Particle,
        RT_Sound,
        RT_Collision,
        RT_Script
    };
    SceneDataCompoent() {}

    String resource_path;
    bool is_visible = true;
    Transform3D transform;

};
class SceneDataCompoentBlock : public RefCounted {
    GDCLASS(SceneDataCompoentBlock, RefCounted);
    static void _bind_methods() {}
public:
    String block_name;
    LocalVector<Ref<SceneDataCompoent>> compoents;
};
class SceneChunkGroupLod : public Resource {
    GDCLASS(SceneChunkGroupLod, Resource);    

public:
    SceneChunkGroupLod() {}

    LocalVector<Ref<SceneDataCompoentBlock>> objects_blocks;
};

class SceneChunkGroup : public Resource {
    GDCLASS(SceneChunkGroup, Resource);
    static void _bind_methods() {}

public:
    SceneChunkGroup() {}

    void on_chunk_load(SceneChunk* p_chunk) {}
    void update_lod(Vector3 camera_pos) {}
    
    LocalVector<Ref<SceneChunkGroupLod>> scene_lods;
};

class SceneChunkGroupInstance : public Node3D {
    GDCLASS(SceneChunkGroupInstance, Node3D);
    static void _bind_methods() {}
public:
    void _notification(int p_what) {
        if (p_what == NOTIFICATION_ENTER_TREE) {
        }
        else if (p_what == NOTIFICATION_EXIT_TREE) {
        }
    }
    struct LodInfo{
        HashMap<ObjectID,Ref<SceneDataCompoent>> data_compoents;
    }
    int curr_lod = 0;
    LocalVector<LodInfo> lods;
};

class SceneChunk : public Node3D {
    GDCLASS(SceneChunk, Node3D);
    static void _bind_methods() {}

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
        ObjectID node_id;
        
    };
    struct Collision{
        Transform3D transform;
        int instance_id;
    };
    struct CollisionInstance : public RefCounted{
        ObjectID mult_mesh_instances_id;
        HashMap<int32_t,Collision> mesh_transforms;
        ObjectID node_id;
        
    };
    HashMap<String,Ref<MeshInstance>> mult_mesh_instances;  
    HashMap<String,Ref<CollisionInstance>> collision_instances;  
    int curr_id = 0;
    List<int> unuse_id_list;
};