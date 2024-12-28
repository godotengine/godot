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
    enum CollisionShapeType{    
        SHAPE_NONE,
		SHAPE_SPHERE, ///< float:"radius"
		SHAPE_BOX, ///< vec3:"extents"
		SHAPE_CAPSULE, ///< dict( float:"radius", float:"height"):capsule
		SHAPE_CYLINDER, ///< dict( float:"radius", float:"height"):cylinder
		SHAPE_MESH, ///< dict( float:"radius", float:"height"):cylinder
        
    };
    SceneDataCompoent() {}

    void set_position(const Vector3& p_position) {
        transform.origin = p_position;
    }
    void set_rotation(const Quaternion& p_rotation) {
        Vector3 scale = transform.basis.get_scale();
        transform.basis.set_quaternion_scale(p_rotation,scale);
    }
    void set_scale(const Vector3& p_scale) {
        transform.basis.set_quaternion_scale(transform.basis.get_quaternion(),p_scale);
    }

    Vector3 get_position() {
        return transform.origin;
    }
    Quaternion get_rotation() {
        return transform.basis.get_quaternion();
    }
    Vector3 get_scale() {
        return transform.basis.get_scale();
    }

    
    void _validate_property(PropertyInfo &r_property) const {
        String prop = r_property.name;
        if(prop.begins_with("collision_")){
            if(collision_shape_type == SHAPE_NONE){
                r_property.usage = PROPERTY_USAGE_NO_EDITOR;
                return;
            }
            if(prop == "collision_flags"){
                return;
            }
            else if(collision_shape_type == SHAPE_SPHERE && !prop.begins_with("collision_sphere_")){
                r_property.usage = PROPERTY_USAGE_NO_EDITOR;
            }
            else if(collision_shape_type == SHAPE_BOX && !prop.begins_with("collision_box_")){
                r_property.usage = PROPERTY_USAGE_NO_EDITOR;
            }
            else if(collision_shape_type == SHAPE_CAPSULE && !prop.begins_with("collision_capsule_")){
                r_property.usage = PROPERTY_USAGE_NO_EDITOR;
            }
            else if(collision_shape_type == SHAPE_CYLINDER && !prop.begins_with("collision_cylinder_")){
                r_property.usage = PROPERTY_HINT_RANGE;
            }
            else if(collision_shape_type == SHAPE_MESH && !prop.begins_with("collision_mesh_")){
                r_property.usage = PROPERTY_HINT_RANGE;
            }
            
        }

    }
    void set_resource_path(const String& p_path) {
        resource_path = p_path;
    }
    String get_resource_path() {
        return resource_path;
    }

    void set_collision_shape_type(CollisionShapeType type){
        collision_shape_type = type;
	    notify_property_list_changed();
    }
    CollisionShapeType get_collision_shape_type(){
        return collision_shape_type;
    }

    String resource_path;
    bool is_visible = true;
    Transform3D transform;

    CollisionShapeType collision_shape_type = SHAPE_SPHERE;
    uint32_t collision_flags = 0;

    Vector3 collosion_box_size = Vector3(1,1,1);

    float collision_cylinder_height = 1;
    float collision_cylinder_radius = 1;

    float collision_sphere_radius = 1;

    float collision_capsule_height = 1;
    float collision_capsule_radius = 1;
    String collision_mesh_path;

};
class SceneDataCompoentBlock : public SceneDataCompoent {
    GDCLASS(SceneDataCompoentBlock, SceneDataCompoent);
    static void _bind_methods() {}
public:
    void set_editor_source_scene_path(const String& p_path) {
        editor_source_scene_path = p_path;
    }
protected:
    String resource_group;
    String resource_tag;
    String editor_source_scene_path;
    String block_name;

    LocalVector<Ref<SceneDataCompoent>> compoents;
};
class SceneChunkGroupLod : public SceneDataCompoentBlock {
    GDCLASS(SceneChunkGroupLod, SceneDataCompoentBlock);
	static void _bind_methods() {}

public:
    SceneChunkGroupLod() {}

    LocalVector<Ref<SceneDataCompoent>> objects_blocks;
};

class SceneChunkGroup : public Resource {
    GDCLASS(SceneChunkGroup, Resource);
    static void _bind_methods() {}

public:
    SceneChunkGroup() {}

    void on_chunk_load(SceneChunk* p_chunk) {}
    void update_lod(Vector3 camera_pos) {}
    String resource_group;
    String resource_tag;
    
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
	struct LodInfo {
		HashMap<ObjectID, Ref<SceneDataCompoent>> data_compoents;
	};
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

    int add_multimesh_instance(const String& res_path, const Transform3D& t) ;
    void remove_multimesh_instance(const String& res_path, int id) ;

    int get_free_id() {
        int id = 0;
        if (unuse_id_list.size() > 0) {
            id = unuse_id_list.front()->get();
            unuse_id_list.pop_front();
        }
        else {
            id = curr_id;
        }
        return curr_id++;
    }

    struct MeshInstanceInfo{
        Transform3D transform;
        Color color;
        Color custom_data;
    };
    struct MeshInstance : public RefCounted{

		void update_mesh_instance();
		void set_mesh_transform(int mesh_id, const Transform3D& t);
		void set_mesh_color(int mesh_id, const Color& color);
		void set_mesh_custom_data(int mesh_id, const Color& color);
        void remove_instance(int mesh_id) {
            mesh_transforms.erase(mesh_id);
            dirty = true;
        }
        ObjectID mult_mesh_instances_id;
        HashMap<int32_t,MeshInstanceInfo> mesh_transforms;
        HashMap<int32_t,int32_t> mesh_id_maps;
        ObjectID node_id;
        Ref<MultiMesh> multimesh;
        Ref<ResourceLoader::LoadToken> load_token;
        bool dirty = false;
        
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
