#ifndef MGRASS
#define MGRASS

#define BUFFER_STRID_FLOAT 12
#define BUFFER_STRID_BYTE 48

#include <mutex>

#include "scene/main/timer.h"
#include "core/templates/vector.h"
#include "core/templates/vset.h"
#include "core/templates/hash_map.h"
#include "scene/3d/node_3d.h"
#include "servers/physics_server_3d.h"
#include "scene/resources/3d/shape_3d.h"
#include "scene/resources/physics_material.h"
#include "mgrass_data.h"
#include "mgrass_lod_setting.h"
#include "../mpixel_region.h"
#include "../mbound.h"
#include "core/variant/variant.h"
#include "../octmesh/mmesh_lod.h"

#include "mgrass_chunk.h"



class MGrid;

class MGrass : public Node3D {
    GDCLASS(MGrass,Node3D);
    private:
    static float time_rollover_secs;
    double current_shader_time=0;
    Ref<PhysicsMaterial> physics_material;
    int collision_layer=1;
    int collision_mask=1;
    int64_t update_id;
    std::mutex update_mutex;
    uint64_t final_count=0;
    int grass_count_limit=9000000;
    int cell_creation_time_data_limit = 10000;
    float nav_obstacle_radius=1.0;
    int shape_type=-1;
    bool visible = true;
    Variant shape_data;

    protected:
    static void _bind_methods();

    public:
    bool active = true;
    bool is_grass_init = false;
    RID scenario;
    RID space;
    Ref<MGrassData> grass_data;
    MGrid* grid = nullptr;
    //int grass_in_cell=1;
    uint32_t base_grid_size_in_pixel;
    uint32_t grass_region_pixel_width; // Width or Height both are equal
    uint32_t grass_region_pixel_size; // Total pixel size for each region
    uint32_t region_grid_width;
    uint32_t width;
    uint32_t height;
    MPixelRegion grass_pixel_region;
    MBound grass_bound_limit;
    int lod_count;
    int min_grass_cutoff=1;
    Array lod_settings;
    Array materials;
    Ref<MMeshLod> meshes;
    Vector<Ref<MGrassLodSetting>> settings;
    Ref<MGrassLodSetting> default_lod_setting;
    Vector<RID> material_rids;
    //Vector<RID> meshe_rids;
    Vector<PackedFloat32Array> rand_buffer_pull;
    HashMap<int64_t,MGrassChunk*> grid_to_grass;
    Vector<MGrassChunk*> to_be_visible;
    VSet<int>* dirty_points_id;
    HashMap<uint32_t,float> cell_creation;
    Vector<uint32_t> cell_creation_order; // use for set a limit for cell_creation, just remove the first element after a limit
    Vector3 shape_offset;
    Ref<Shape3D> shape;
    MBound physics_search_bound;
    MBound last_physics_search_bound;
    RID main_physics_body;
    Vector<uint64_t> shapes_ids;
    HashMap<int,RID> shapes_rids; //Key is random Index (Each Random index will result the same shape) and value is the RID shape
    float collision_radius=64;
    bool active_shape_resize=false;

    MGrass();
    ~MGrass();
    void init_grass(MGrid* _grid);
    void clear_grass();
    void update_grass();
    void update_dirty_chunks_gd();
    void update_dirty_chunks(bool update_lock=true);
    void apply_update_grass();
    void cull_out_of_bound();
    void create_grass_chunk(int grid_index,MGrassChunk* grass_chunk=nullptr); //If grid_index=-1 and grass_chunk is not null it will update grass chunk
    void recalculate_grass_config(int max_lod);
    void make_grass_dirty_by_pixel(uint32_t px, uint32_t py);
    void set_grass_by_pixel(uint32_t px, uint32_t py, bool p_value);
    void set_grass_sublayer_by_pixel(uint32_t px, uint32_t py, bool p_value);
    bool is_init();
    bool has_sublayer();
    void merge_sublayer();
    void create_sublayer();
    void clear_grass_sublayer_by_pixel(uint32_t px, uint32_t py);
    bool get_grass_by_pixel(uint32_t px, uint32_t py);
    Vector2i get_closest_pixel(Vector3 pos);
    Vector3 get_pixel_world_pos(uint32_t px, uint32_t py);
    Vector2i grass_px_to_grid_px(uint32_t px, uint32_t py);
    void draw_grass(Vector3 brush_pos,real_t radius,bool add);
    void clear_grass_sublayer_aabb(AABB aabb);
    _FORCE_INLINE_ void set_lod_setting_image_index(Ref<MGrassLodSetting> lod_setting); // Should be called after generate_random_number
    void set_active(bool input);
    bool get_active();
    void set_grass_data(Ref<MGrassData> d);
    Ref<MGrassData> get_grass_data();
    void set_cell_creation_time_data_limit(int input);
    int get_cell_creation_time_data_limit();
    void set_grass_count_limit(int input);
    int get_grass_count_limit();
    //void set_grass_in_cell(int input);
    //int get_grass_in_cell();
    void set_min_grass_cutoff(int input);
    int get_min_grass_cutoff();
    void set_lod_settings(Array input);
    Array get_lod_settings();
    void set_meshes(Variant input);
    Ref<MMeshLod> get_meshes();
    void set_materials(Array input);
    Array get_materials();
    uint32_t get_width();
    uint32_t get_height();

    int64_t get_count();

    void set_collision_radius(float input);
    float get_collision_radius();
    void set_shape_offset(Vector3 input);
    Vector3 get_shape_offset();
    void set_shape(Ref<Shape3D> input);
    Ref<Shape3D> get_shape();
    int get_collision_layer();
    void set_collision_layer(int input);
    int get_collision_mask();
    void set_collision_mask(int input);
    Ref<PhysicsMaterial> get_physics_material();
    void set_physics_material(Ref<PhysicsMaterial> input);
    void set_active_shape_resize(bool input);
    bool get_active_shape_resize();
    void set_nav_obstacle_radius(float input);
    float get_nav_obstacle_radius();
    void update_physics(Vector3 cam_pos);
    void remove_all_physics();
    RID get_resized_shape(Vector3 scale);
    PackedVector3Array get_physic_positions(Vector3 cam_pos,float radius);


    void _get_property_list(List<PropertyInfo> *p_list) const;
    bool _get(const StringName &p_name, Variant &r_ret) const;
    bool _set(const StringName &p_name, const Variant &p_value);


    Error save_grass_data();
    void _recreate_all_grass(); // used in gdscript
    void recreate_all_grass(int lod=-1); // LOD -1 means all grass
    void update_random_buffer_pull(int lod);
    void _lod_setting_changed();
    void check_undo(); // register a grass data stege for undo
    void undo();
    bool is_depend_on_image(int image_index);

    static float get_shader_time();
    _FORCE_INLINE_ void update_shader_time();
    _FORCE_INLINE_ float get_shader_time(uint32_t grass_cell_index);


    void _notification(int32_t what);

    bool is_visible();
    void _update_visibilty();
    void set_visibility(bool input);
};
#endif
