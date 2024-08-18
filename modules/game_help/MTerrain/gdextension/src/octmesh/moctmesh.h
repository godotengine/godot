#ifndef _MOCTMESH
#define _MOCTMESH



#define OCT_POINT_ID_START 0
#define CURRENT_LOD -2
#define INVALID_LOD -3

#include "scene/3d/node_3d.h"

#include <atomic>
#include <mutex>
#include "mmesh_lod.h"
#include "../moctree.h"

using namespace godot;


class MOctMesh : public Node3D {
    GDCLASS(MOctMesh,Node3D);

    friend class MOctree;
    
    protected:
    static void _bind_methods();    
    

    private:
    /// Static Part
    static WorkerThreadPool::TaskID thread_task_id;
    static std::mutex update_mutex;
    static bool is_updating;
    static bool is_octree_inserted;
    static uint16_t oct_id;
    static int32_t last_oct_point_id;
    static HashMap<int32_t,MOctMesh*> octpoint_to_octmesh;
    static MOctree* octree;
    

    public:
    static bool is_my_octree(MOctree* input);
    static uint16_t get_oct_id();
    static bool set_octree(MOctree* input);
    static void remove_octree(MOctree* input);
    static void insert_points();
    static int32_t add_octmesh(MOctMesh* input); // use update_mutex
    static void remove_octmesh(int32_t id); // use update_mutex
    static void move_octmesh(MOctMesh* input);
    static void octree_update(const Vector<MOctree::PointUpdate>* update_info);
    static void octree_thread_update(void* input); // use update_mutex
    static void update_tick();

    public:
    
    // Non static part
    private:
    std::atomic<int8_t> lod{-1};
    bool enable_global_illumination = true;
    bool ignore_occlusion_culling = false;
    RenderingServer::ShadowCastingSetting shadow_setting = RenderingServer::ShadowCastingSetting::SHADOW_CASTING_SETTING_ON;
    float transparency = 0.0;
    float lod_bias = 1.0;
    float extra_cull_margin = 0.0;
    AABB custom_aabb;
    
    //int8_t lod = -1;
    int32_t oct_point_id = INVALID_OCT_POINT_ID;
    RID instance; // use with update_mutex protection
    RID current_mesh; // use with update_mutex protection
    Ref<MMeshLod> mesh_lod;
    Ref<Material> material_override;
    mutable Dictionary instance_shader_parameters;

    void _update_visibilty();

    public:
    Vector3 oct_position;
    MOctMesh();
    ~MOctMesh();

    // CURRENT_LOD means update current mesh without changing LOD
    // INVALID_LOD is invalide object, or it will removed
    void update_lod_mesh(int8_t new_lod=CURRENT_LOD); // must be called with update_mutex protection
    Ref<Mesh> get_active_mesh();

    void set_mesh_lod(Ref<MMeshLod> input); // use update_mutex
    Ref<MMeshLod> get_mesh_lod();

    void set_material_override(Ref<Material> input);
    Ref<Material> get_material_override();

    void set_shadow_setting(RenderingServer::ShadowCastingSetting input);
    RenderingServer::ShadowCastingSetting get_shadow_setting();

    void set_ignore_occlusion_culling(bool input);
    bool get_ignore_occlusion_culling();

    void set_enable_global_illumination(bool input);
    bool get_enable_global_illumination();

    void set_transparency(float input);
    float get_transparency();

    void set_lod_bias(float input);
    float get_lod_bias();

    void set_extra_cull_margin(float input);
    float get_extra_cull_margin();

    void set_custom_aabb(AABB input);
    AABB get_custom_aabb();

    void set_instance_shader_parameters(Dictionary input);
    Dictionary get_instance_shader_parameters();

    
    _FORCE_INLINE_ bool has_valid_oct_point_id();

    void _notification(int p_what); // some part use update_mutex
    void _lod_mesh_changed(); // use update_mutex
	bool _set(const StringName &p_name, const Variant &p_value);
	bool _get(const StringName &p_name, Variant &r_ret) const;
    void _get_property_list(List<PropertyInfo> *p_list) const;

};
#endif