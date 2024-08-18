#ifndef MREGION
#define MREGION

#include "core/object/object.h"
#include "core/variant/variant.h"
#include "scene/resources/material.h"
#include "scene/resources/shader.h"
#include "core/templates/rid.h"
#include "core/io/image.h"
#include "core/io/image.h"
#include "core/templates/vector.h"
#include "scene/resources/image_texture.h"
#include "scene/resources/shader.h"
#include "core/templates/vset.h"
#include "core/variant/variant.h"
#include "core/io/resource_loader.h"
#include "servers/physics_server_3d.h"
#include "core/variant/variant.h"
#include "core/variant/dictionary.h"
#include "core/math/transform_3d.h"
#include "core/math/vector3.h"
#include "core/math/vector2i.h"
#include "mbound.h"
#include "mimage.h"
#include "mpixel_region.h"
#include "mresource.h"

#include <mutex>
#include <atomic>

class MGrid;





class MRegion : public Object{
    GDCLASS(MRegion, Object);

    private:
    Vector<bool> _images_init_status; // Defently should protected by mutex
    bool _is_online=false;
    RID _material_rid = RID();
    MImage* heightmap = nullptr;
    MImage* normals = nullptr;
    VSet<int8_t>* lods;
    int8_t last_lod = -2;
    RID physic_body;
    RID heightmap_shape;
    bool has_physic=false;
    double current_image_size = 5;
    int32_t current_scale=1;
    bool is_min_max_height_calculated = false;
    _FORCE_INLINE_ void _calculate_min_max_height();
    std::mutex physics_mutex;
    std::atomic<bool> is_data_loaded;


    protected:
    static void _bind_methods(){}

    public:
    bool is_data_loaded_reg_thread = false; // Must use only in region update thread
    bool is_edge_corrected = false; // Same as above only in region update thread
    //Bellow will be written in update region thread and only read in physics update thread to create warning
    bool is_min_max_bottom_considered = false; // region update thread
    bool is_min_max_right_considered = false; // region update thread
    bool to_be_remove = false;
    int32_t id=-1;
    Vector<MImage*> images;
    float min_height= 100000;	
    float max_height=-100000;
    MGrid* grid;
    MGridPos pos;
    Vector3 world_pos;
    MPixelRegion normals_pixel_region; // use for recalculating normals
    //int32_t region_size_meter;
    MRegion* left = nullptr;
    MRegion* right = nullptr;
    MRegion* top = nullptr;
    MRegion* bottom = nullptr;
    static Vector<Vector3> nvecs;
    
    
    MRegion();
    ~MRegion();
    void set_material(RID input);
    RID get_material_rid();
    void add_image(MImage* input);
    void configure();
    void load();
    void unload();
    String get_res_path();
    void update_region();
    void insert_lod(const int8_t input);
    void apply_update();
    void create_physics();
    void update_physics();
    void remove_physics();
    Color get_pixel(const uint32_t x, const uint32_t y, const int32_t& index) const;
    void set_pixel(const uint32_t x, const uint32_t y,const Color& color,const int32_t& index);
    Color get_normal_by_pixel(const uint32_t x, const uint32_t y) const;
    void set_normal_by_pixel(const uint32_t x, const uint32_t y,const Color& value);
    real_t get_height_by_pixel(const uint32_t x, const uint32_t y) const;
    void set_height_by_pixel(const uint32_t x, const uint32_t y,const real_t& value);
    real_t get_closest_height(Vector3 pos);
    real_t get_height_by_pixel_in_layer(const uint32_t x, const uint32_t y) const;

    void update_all_dirty_image_texture();
    void save_image(Ref<MResource> mres,int index,bool force_save);

    void recalculate_normals(bool use_thread=true,bool use_extra_margin=false);
    void refresh_all_uniforms();
    void make_normals_dirty();
    void make_neighbors_normals_dirty();

    void set_data_load_status(bool input);
    bool get_data_load_status();
    bool get_data_load_status_relax();
    void correct_edges();
    private:
    //Each region right and bottom edge pixel is copied from other region
    //All correct edges methods should be called in the same thread, After load
    //Top-Left and Bottom pixel will not correctet by bellow 
    //Those pixel will be corrected in corner pixel correcting
    void correct_left_edge();
    void correct_right_edge();
    void correct_top_edge();
    void correct_bottom_edge();
    //Cornert correcting
    void correct_bottom_right_corner();
    void correct_top_left_corner();
};
#endif