#ifndef MGRID
#define MGRID

//#define NO_MERGE

#include <thread>
#include <future>
#include <chrono>

#include "core/object/object.h"
#include "core/templates/rid.h"
#include "core/templates/vector.h"
#include "core/math/vector3.h"
#include "core/math/transform_3d.h"
#include "core/math/basis.h"
#include "servers/rendering_server.h"
#include "core/variant/variant.h"
#include "core/templates/vector.h"
#include "scene/resources/material.h"
#include "core/io/image.h"
#include "core/variant/dictionary.h"
#include "scene/resources/physics_material.h"
#include "core/io/config_file.h"



#include "mregion.h"
#include "mchunks.h"
#include "mconfig.h"
#include "mbound.h"
#include "mpixel_region.h"
#include "mcollision.h"
#include "mterrain_material.h"

class MBrushManager;
class MHeightBrush;
class MColorBrush;









// size -1 means it has been merged
// lod -1 means it is out of range
// lod -2 means it should be droped and never been drawn
struct MPoint
{
    RID instance = RID();
    RID mesh = RID();
    int8_t lod = -1;
    int8_t size=0;
    bool has_instance=false;

   
    void create_instance(const Vector3& pos,const RID scenario,const RID material,bool visible){
        has_instance = true;
        Transform3D xform(Basis(), pos);
        RenderingServer* rs = RenderingServer::get_singleton();
        instance = rs->instance_create();
        rs->instance_set_scenario(instance, scenario);
        rs->instance_set_transform(instance, xform);
        rs->instance_set_visible(instance,visible);
        if(material != RID())
            rs->instance_geometry_set_material_override(instance, material);
    }

    ~MPoint(){
        RenderingServer::get_singleton()->free(instance);
    }
};

struct MGridUpdateInfo
{
    RID terrain_instance;
    int region_id;
    Vector3 region_world_pos;
    Vector2 region_offset_ratio;
    int lod;
    int chunk_size;
    int distance;
};

struct InstanceDistance
{
    int64_t id;
    int distance;
    friend bool operator<(const InstanceDistance& c1, const InstanceDistance& c2){
        return c1.distance<c2.distance;
    }
    friend bool operator>(const InstanceDistance& c1, const InstanceDistance& c2){
        return c1.distance>c2.distance;
    }
};

struct MSaveConfig
{
    float accuracy=DEFAULT_ACCURACY;
    bool heightmap_compress_qtq=true;
    MResource::FileCompress heightmap_file_compress=MResource::FileCompress::FILE_COMPRESSION_NONE;
    HashMap<StringName,MResource::Compress> data_compress;
    HashMap<StringName,MResource::FileCompress> data_file_compress;

    MResource::Compress get_data_compress(const StringName& dname){
        return data_compress.has(dname) ? data_compress[dname] : MResource::Compress::COMPRESS_NONE;
    }

    MResource::FileCompress get_data_file_compress(const StringName& dname){
        return data_file_compress.has(dname) ? data_file_compress[dname] : MResource::FileCompress::FILE_COMPRESSION_NONE;
    }

    void clear(){
        data_compress.clear();
        data_file_compress.clear();
        heightmap_file_compress=MResource::FileCompress::FILE_COMPRESSION_NONE;
        heightmap_compress_qtq=true;
        accuracy=DEFAULT_ACCURACY;
    }
};

class MGrid {
    friend class MRegion;
    friend class MTerrain;
    friend class MImage;
    private:
    bool _is_opengl=false;
    uint8_t _update_id=0; // Only for mesh update not for physics
    MBrushManager* _brush_manager = nullptr;
    MPoint** points;
    MPoint* points_row;
    bool current_update = true;
    bool is_dirty = false;
    bool is_visible = true;
    MChunks* _chunks;
    MGridPos _size;
    MGridPos _size_meter;
    MGridPos _vertex_size;
    MBound _grid_bound;
    MBound _region_grid_bound;
    MBound _last_region_grid_bound;
    MBound _last_physics_grid_bound;
    MBound _search_bound;
    MBound _last_search_bound;
    MGridPos _cam_pos;
    Vector3 _cam_pos_real;
    MGridPos _lowest_distance;
    RID _scenario;
    int32_t num_chunks = 0;
    int32_t chunk_counter = 0;
    MGridPos _region_grid_size;
    int32_t _regions_count=0;
    Vector<MImage*> _all_image_list;
    Vector<MImage*> _all_heightmap_image_list;
    PackedVector3Array nvec8;
    Vector<MRegion*> load_region_list;
    Vector<MRegion*> unload_region_list;
    MBound current_region_bound;
    
    

    Ref<MTerrainMaterial> _terrain_material;
    int total_points_count=0;
    uint64_t update_count=0;
    uint64_t total_remove=0;
    uint64_t total_add=0;
    uint64_t total_chunks=0;

    _FORCE_INLINE_ bool _has_pixel(const uint32_t x,const uint32_t y);

    std::future<void> update_regions_future;
    bool is_update_regions_future_valid = false;


    public:
    std::mutex update_chunks_mutex;
    MSaveConfig save_config;
    Ref<PhysicsMaterial> physics_material;
    int collision_layer=1;
    int collision_mask=1;
    MRegion* regions;
    // This can be removed in future but right now I keep it
    Vector<RID> update_mesh_list;
    Vector<RID> remove_instance_list;
    Vector<MGridUpdateInfo> grid_update_info;
    Vector<InstanceDistance> instances_distance; // ordered by distance
    int active_heightmap_layer=0;
    // MImage does not check for visibility of layers
    // Here we should check that in the case someone want to draw on them it should give an error
    Vector<bool> heightmap_layers_visibility;
    PackedStringArray heightmap_layers;
    bool has_normals = false;
    Dictionary uniforms_id;
    int32_t regions_processing_physics = 1;
    int32_t region_limit = 2;
    RID space;
    String dataDir;
    String layersDataDir;
    PackedInt32Array lod_distance;
    int32_t region_size = 128;
    int32_t region_size_meter;
    uint32_t region_pixel_size; //Region width or height they are equal
    uint32_t rp;
    //MBound grid_pixel_bound;
    uint32_t pixel_width;
    uint32_t pixel_height;
    MPixelRegion grid_pixel_region;
    Vector3 offset;
    int32_t max_range = 128;
    /*
    Brush Stuff
    */
    float mask_cutoff=0.5; // Agreed initiale value here and in paint panel
    bool brush_mask_active=false;
    Ref<Image> brush_mask;
    Vector2i brush_mask_px_pos;
    uint32_t brush_px_pos_x;
    uint32_t brush_px_pos_y;
    uint32_t brush_px_radius;
    MPixelRegion draw_pixel_region;
    real_t brush_radius;
    Vector3 brush_world_pos;
    Vector3 brush_world_pos_start;
    Vector3 brush_radius_start;
    int32_t current_paint_index=-1;
    // Undo Redo stuff
    int current_undo_id=0;
    int lowest_undo_id=0;
    Vector<MImage*> last_images_undo_affected_list;
    // End
    MGrid();
    ~MGrid();
    uint64_t get_update_id();
    void clear();
    bool is_created();
    MGridPos get_size();
    void set_scenario(RID scenario);
    RID get_scenario();
    void create(const int32_t width,const int32_t height, MChunks* chunks);
    void update_all_image_list();
    Vector3 get_world_pos(const int32_t x,const int32_t y,const int32_t z);
    Vector3 get_world_pos(const MGridPos& pos);
    int get_point_id_by_non_offs_ws(const Vector2& input); // Get point id non offset world posiotion usefull for grass for now
    int64_t get_point_instance_id_by_point_id(int pid);
    MGridPos get_grid_pos(const Vector3& pos);
    int32_t get_regions_count();
    MGridPos get_region_grid_size();
    int32_t get_region_id_by_point(const int32_t x, const int32_t z);
    MRegion* get_region_by_point(const int32_t x, const int32_t z);
    MBound region_bound_to_point_bound(const MBound& rbound);
    MRegion* get_region(const int32_t x, const int32_t z);
    MGridPos get_region_pos_by_world_pos(Vector3 world_pos);
    int get_region_id_by_world_pos(Vector3 world_pos);
    Vector2 get_point_region_offset_ratio(int32_t x,int32_t z);
    Vector3 get_region_world_pos_by_point(int32_t x,int32_t z);
    int8_t get_lod_by_distance(const int32_t dis);
    void set_cam_pos(const Vector3& cam_world_pos);
    void update_search_bound();
    void cull_out_of_bound();
    void update_lods();
    void merge_chunks();
    _FORCE_INLINE_ bool check_bigger_size(const int8_t lod,const int8_t size,const int32_t region_id, const MBound& bound);
    _FORCE_INLINE_ int8_t get_edge_num(const bool left,const bool right,const bool top,const bool bottom);
    void create_ordered_instances_distance();

    void set_terrain_material(Ref<MTerrainMaterial> input);
    Ref<MTerrainMaterial> get_terrain_material();

    MGridPos get_3d_grid_pos_by_middle_point(MGridPos input);
    real_t get_closest_height(const Vector3& pos);
    real_t get_height(Vector3 pos);
    Ref<MCollision> get_ray_collision_point(Vector3 ray_origin,Vector3 ray_vector,real_t step,int max_step);

    // This will create the initiale region lod state, This is needed so two nested thread do a good job
    // In update chunk update_point thread should finish but update region will check in the next
    // update and if it is not finished we countiue with only update points and recheck region update again
    void update_chunks(const Vector3& cam_pos);
    void update_regions(); // This one need camera pos as this thread can last more than one terrain update!
    void update_regions_at_load();
    void apply_update_chunks();
    bool update_regions_bounds(const Vector3& cam_pos,bool _make_neighbors_normals_dirty);//Should be called in safe thread
    void clear_region_bounds();
    void update_physics(const Vector3& cam_pos);

    MImage* get_image_by_pixel(uint32_t x,uint32_t y, const int32_t index);
    Color get_pixel(uint32_t x,uint32_t y, const int32_t index);
    const uint8_t* get_pixel_by_pointer(uint32_t x,uint32_t y, const int32_t index);
    void set_pixel(uint32_t x,uint32_t y,const Color& col,const int32_t index);
    void set_pixel_by_pointer(uint32_t x,uint32_t y,uint8_t* ptr, const int32_t index);
    real_t get_height_by_pixel(uint32_t x,uint32_t y);
    void set_height_by_pixel(uint32_t x,uint32_t y,const real_t value);
    real_t get_height_by_pixel_in_layer(uint32_t x,uint32_t y);
    bool has_pixel(const uint32_t x,const uint32_t y);
    void generate_normals_thread(MPixelRegion pxr);
    void generate_normals(MPixelRegion pxr);
    void update_normals(uint32_t left, uint32_t right, uint32_t top, uint32_t bottom);
    Vector3 get_normal_by_pixel(uint32_t x,uint32_t y);
    Vector3 get_normal_accurate_by_pixel(uint32_t x,uint32_t y);
    Vector3 get_normal(Vector3 world_pos);
    Vector3 get_normal_accurate(Vector3 world_pos);
    void save_image(int index,bool force_save);
    bool has_unsave_image();
    void save_all_dirty_images();

    Vector2i get_closest_pixel(Vector3 world_pos);
    Vector3 get_pixel_world_pos(uint32_t x,uint32_t y);

    void set_brush_manager(MBrushManager* input);
    MBrushManager* get_brush_manager();
    void set_brush_start_point(Vector3 brush_pos,real_t radius);
    void draw_height(Vector3 brush_pos,real_t radius,int brush_id);
    void draw_height_region(MImage* img, MPixelRegion draw_pixel_region, MPixelRegion local_pixel_region, MHeightBrush* brush);

    void draw_color(Vector3 brush_pos,real_t radius,MColorBrush* brush, int32_t index);
    void draw_color_region(MImage* img, MPixelRegion draw_pixel_region, MPixelRegion local_pixel_region, MColorBrush* brush);

    void update_all_dirty_image_texture(bool update_physics=false);

    private:
    bool set_active_layer(String input);
    String get_active_layer();
    void add_heightmap_layer(String lname);
    void rename_heightmap_layer(int layer_index,String lname);
    void merge_heightmap_layer();
    void remove_heightmap_layer();
    void toggle_heightmap_layer_visible();
    bool is_layer_visible(int index);
    bool is_layer_visible(const String& lname);

    public:
    float get_h_scale();

    float get_brush_mask_value(uint32_t x,uint32_t y);
    bool get_brush_mask_value_bool(uint32_t x,uint32_t y);

    void images_add_undo_stage(); // This will called before drawing or change happen
    void images_undo();

    void refresh_all_regions_uniforms();

    void update_renderer_info();
    bool is_opengl();

    bool get_visibility();
    void set_visibility(bool input);
};

#endif