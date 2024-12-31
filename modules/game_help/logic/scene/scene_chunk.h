#pragma once
#include "scene/3d/node_3d.h"
#include "scene/3d/multimesh_instance_3d.h"
#include "scene/resources/3d/shape_3d.h"
class SceneChunk;
class SceneChunkGroupInstance;
class MeshCollisionResource : public Resource{
    GDCLASS(MeshCollisionResource, Resource);
    static void _bind_methods() {
        ClassDB::bind_method(D_METHOD("set_points", "points"), &MeshCollisionResource::set_points);
        ClassDB::bind_method(D_METHOD("get_points"), &MeshCollisionResource::get_points);

        ADD_PROPERTY(PropertyInfo(Variant::ARRAY, "points"), "set_points", "get_points");
    }
public:
    void set_points(const Vector<Vector3> &p_points) { points = p_points; }
    Vector<Vector3> get_points() const { return points; }
protected:
    Vector<Vector3> points;
};
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

    virtual void show(int p_show_level,const Transform3D& p_parent, SceneChunkGroupInstance * instance) ;

    String resource_path;
    ResourceType resource_type = ResourceType::RT_Mesh;
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
    bool is_visible = true;

};
class SceneDataCompoentBlock : public SceneDataCompoent {
    GDCLASS(SceneDataCompoentBlock, SceneDataCompoent);
    static void _bind_methods() {}
public:
    void set_editor_source_scene_path(const String& p_path) {
        editor_source_scene_path = p_path;
    }
    virtual void show(int p_show_level,const Transform3D& p_parent, SceneChunkGroupInstance * instance) override;
protected:
    String editor_source_scene_path;
    String block_name;

    LocalVector<Pair<Ref<SceneDataCompoent>,int>> compoents;
};
class SceneBlock : public Resource {
    GDCLASS(SceneBlock, Resource);
    static void _bind_methods() {}
public:
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
    virtual void show(int p_show_level,const Transform3D& p_parent, SceneChunkGroupInstance * instance);
public:
    Transform3D transform;
    LocalVector<Pair<Ref<SceneDataCompoent>,int>> blocks;
};

class SceneResource : public Resource {
    GDCLASS(SceneResource, Resource);
    static void _bind_methods() {}

public:
    SceneResource() {}

    void on_chunk_load(SceneChunk* p_chunk) {}
    void update_lod(Vector3 camera_pos) {}
    virtual void show(int lod,int p_show_level,const Transform3D& p_parent, SceneChunkGroupInstance * instance);
    String resource_group;
    String resource_tag;

    int lod_count = 0;
    float lod_distance[4] = {32,64,128,256};
    
    LocalVector<Pair<Ref<SceneBlock>,int>>  scene_lods;
};

class SceneChunkGroupInstance : public Node3D {
    GDCLASS(SceneChunkGroupInstance, Node3D);
    static void _bind_methods() {}
public:
    void _notification(int p_what) {
        if (p_what == NOTIFICATION_ENTER_TREE) {
        }
        else if (p_what == NOTIFICATION_EXIT_TREE) {
            clear_show_instance_ids();
        }
    }
	struct LodInfo {
		HashMap<ObjectID, Ref<SceneDataCompoent>> data_compoents;
	};
    int _add_mesh_instance(const String& p_path,const Transform3D& t);
    int _add_collision_instance(const Transform3D& t,SceneDataCompoent::CollisionShapeType type,const Vector3& box_size,float height,float radius);
    int _add_mesh_collision_instance(const String& p_path,const Transform3D& t);

    void set_lod(int p_lod) ;
    void clear_show_instance_ids();
    SceneChunk* get_chunk();
protected:
    ObjectID chunk_id;
    int curr_lod = 0;
    Ref<SceneResource> resource;
    HashMap<int32_t,String> curr_show_meshinstance_ids;
    HashMap<int32_t,String> curr_show_mesh_collision_ids;
    HashSet<int32_t> curr_show_collision_ids;
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

    int add_collision_instance(const Transform3D& t,SceneDataCompoent::CollisionShapeType type,const Vector3& box_size,float height,float radius) ;
    void remove_collision_instance( int id) ;

    
    int add_mesh_collision_instance(const Transform3D& t,const String& p_path) ;
    void remove_mesh_collision_instance( int id,const String& p_path) ;

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
		RID node_id;
        RID shape;
        int collision_layer = 0;
        int collision_mask = 0;
    };
    struct MeshCollisionInstance : public RefCounted{
        HashMap<int32_t,Collision> mesh_transforms;
        RID shape;
		void clear() {

		}
        
    };
    HashMap<String,Ref<MeshInstance>> mult_mesh_instances;  
    HashMap<int,Collision> collision_instances;
    HashMap<String,Ref<MeshCollisionInstance>> mesh_collision_instances;  
    int curr_id = 0;
    List<int> unuse_id_list;
};
